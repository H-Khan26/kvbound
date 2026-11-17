"""Runs the engine under compute-sanitizer on the committed fixture.

memcheck finds out-of-bounds and misaligned device accesses, racecheck finds shared-memory races
between the threads of a block, initcheck finds reads of device memory that was never written.
The run covers chunked prefill and decode, with and without CUDA graphs, one request at a time,
as a static batch, and under continuous batching over 4-token blocks in a pool small enough to
preempt a sequence and to evict CUDA graphs. The paged run repeats with an INT8 KV cache and
with INT8 and INT4 copies of the fixture's weights, multiplied in place, on tensor cores and
expanded to FP16.

Usage: python -m kvbound.sanitize
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from kvbound import quantize
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
    fp16 = str(FIXTURE_DIR)
    cases: list[tuple[str, str, list[str]]] = [
        ("eager, 1 slot", fp16, ["--graph", "0", "--slots", "1", *static]),
        ("graph, 1 slot", fp16, ["--graph", "1", "--slots", "1", *static]),
        ("graph, 3 slots", fp16, ["--graph", "1", "--slots", "3", *static]),
        ("graph, paged", fp16, ["--graph", "1", "--slots", "3", "--max-graphs", "1", *paged]),
        ("eager, paged", fp16, ["--graph", "0", "--slots", "3", *paged]),
        ("unfused, cuBLAS", fp16, ["--slots", "3", "--fuse", "0", "--f16-gemv-rows", "0", *paged]),
        ("int8 kv, paged", fp16, ["--slots", "3", "--kv-dtype", "int8", *paged]),
    ]
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        for scheme in ("int8_pc", "int4_g"):
            quantized = Path(tmp) / scheme
            quantize.convert(FIXTURE_DIR, quantized, scheme, group_size=32, device="cpu")
            in_place = ["--slots", "3", "--kv-dtype", "int8", *paged]
            cases.append((f"{scheme}, in place", str(quantized), in_place))
            wide = [*in_place, "--quantized-gemm-rows", "0"]
            cases.append((f"{scheme}, tensor cores", str(quantized), wide))
            cases.append(
                (f"{scheme}, expanded", str(quantized), [*wide, "--expand-quantized", "1"])
            )
        for tool in TOOLS:
            for name, model_dir, options in cases:
                command = [
                    sanitizer, "--tool", tool, "--error-exitcode", "1",
                    str(binary("bench", args.preset)), "--model", model_dir,
                    "--trace", str(trace_path), "--max-ctx", "64", "--max-step-tokens", "16",
                    "--warmup", "0", "--iters", "1", *options,
                ]  # fmt: skip
                done = subprocess.run(command, capture_output=True, text=True, check=False)
                summary = [line for line in done.stdout.splitlines() if "SUMMARY" in line]
                ok = done.returncode == 0
                failed = failed or not ok
                status = "ok  " if ok else "FAIL"
                print(f"{tool:<10} {name:<18} {status}  {' '.join(summary)}")
                if not ok:
                    print(done.stdout[-4000:], done.stderr[-2000:])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
