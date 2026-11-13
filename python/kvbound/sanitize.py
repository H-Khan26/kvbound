"""Runs the engine under compute-sanitizer on the committed fixture.

memcheck finds out-of-bounds and misaligned device accesses, racecheck finds shared-memory races
between the threads of a block, initcheck finds reads of device memory that was never written.
The run covers chunked prefill and decode, with and without CUDA graphs, one request at a time,
as a static batch, and under continuous batching over 4-token blocks in a pool small enough to
preempt a sequence and to evict CUDA graphs.

Usage: python -m kvbound.sanitize
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from kvbound.fixture import FIXTURE_DIR
from kvbound.paths import binary

TOOLS = ("memcheck", "racecheck", "initcheck")


def find_sanitizer() -> str:
    for name in ("compute-sanitizer", "compute-sanitizer.bat"):
        if found := shutil.which(name):
            return found
    candidate = Path(os.environ.get("CUDA_PATH", "")) / "bin" / "compute-sanitizer.bat"
    if candidate.exists():
        return str(candidate)
    raise SystemExit("compute-sanitizer not found; it ships with the CUDA Toolkit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="cuda")
    args = parser.parse_args()

    sanitizer = find_sanitizer()
    trace = {
        "requests": [
            {"id": 0, "prompt": list(range(3, 40)), "max_new_tokens": 20},
            {"id": 1, "prompt": [7], "max_new_tokens": 5},
            {"id": 2, "prompt": list(range(50, 62)), "max_new_tokens": 30, "arrival_s": 0.01},
        ]
    }
    # Eager and graph replay one request at a time, both as one static batch, then continuous
    # batching in a 16-block pool with room for one graph.
    static = ["--policy", "static"]
    paged = ["--policy", "continuous", "--kv-block-size", "4", "--kv-blocks", "16"]
    cases = [
        ("eager, 1 slot", ["--graph", "0", "--slots", "1", *static]),
        ("graph, 1 slot", ["--graph", "1", "--slots", "1", *static]),
        ("graph, 3 slots", ["--graph", "1", "--slots", "3", *static]),
        ("graph, paged", ["--graph", "1", "--slots", "3", "--max-graphs", "1", *paged]),
        ("eager, paged", ["--graph", "0", "--slots", "3", *paged]),
    ]
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        for tool in TOOLS:
            for name, options in cases:
                command = [
                    sanitizer, "--tool", tool, "--error-exitcode", "1",
                    str(binary("bench", args.preset)), "--model", str(FIXTURE_DIR),
                    "--trace", str(trace_path), "--max-ctx", "64", "--max-step-tokens", "16",
                    "--warmup", "0", "--iters", "1", *options,
                ]  # fmt: skip
                done = subprocess.run(command, capture_output=True, text=True, check=False)
                summary = [line for line in done.stdout.splitlines() if "SUMMARY" in line]
                ok = done.returncode == 0
                failed = failed or not ok
                status = "ok  " if ok else "FAIL"
                print(f"{tool:<10} {name:<15} {status}  {' '.join(summary)}")
                if not ok:
                    print(done.stdout[-4000:], done.stderr[-2000:])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
