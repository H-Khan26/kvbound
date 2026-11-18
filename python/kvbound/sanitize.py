"""Runs the engines under compute-sanitizer on the committed fixtures.

memcheck finds out-of-bounds and misaligned device accesses, racecheck finds shared-memory races
between the threads of a block, initcheck finds reads of device memory that was never written.

The Llama run covers chunked prefill and decode, with and without CUDA graphs, one request at a
time, as a static batch, and under continuous batching over 4-token blocks in a pool small enough
to preempt a sequence and to evict CUDA graphs. The paged run repeats with an INT8 KV cache and
with INT8 and INT4 copies of the fixture's weights, multiplied in place, on tensor cores and
expanded to FP16.

The Whisper run encodes the golden clips of fixtures/tiny-whisper, one and three to an encoder
step, and teacher-forces their decoder ids as one step and token by token: the fused and the
unfused programs, kvbound's kernels and cuBLAS, INT8 caches, and INT8 and INT4 copies of the
weights on each of their three paths. It then transcribes the clips greedily through the
scheduler: a static batch eager and with CUDA graphs, and continuous batching in a Self pool small
enough to preempt and re-encode a sequence and to evict a graph. Last, it transcribes them by beam
search over 4-token blocks: forks sharing blocks with copy-on-write under CUDA graphs, and forks
copying every block of an INT8 Self cache, scales included, eagerly.

Usage: python -m kvbound.sanitize [--family llama|whisper]
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from kvbound import fixture, golden, quantize
from kvbound.fixture import FIXTURE_DIR, WHISPER_FIXTURE_DIR
from kvbound.paths import binary

TOOLS = ("memcheck", "racecheck", "initcheck")
FAMILIES = ("llama", "whisper")

# A case is a name and the command the sanitizer runs.
Case = tuple[str, list[str]]


def find_sanitizer() -> str:
    for name in ("compute-sanitizer", "compute-sanitizer.bat"):
        if found := shutil.which(name):
            return found
    candidate = Path(os.environ.get("CUDA_PATH", "")) / "bin" / "compute-sanitizer.bat"
    if candidate.exists():
        return str(candidate)
    raise SystemExit("compute-sanitizer not found; it ships with the CUDA Toolkit")


def llama_cases(preset: str, tmp: Path) -> list[Case]:
    trace = {
        "requests": [
            {"id": 0, "prompt": list(range(3, 40)), "max_new_tokens": 20},
            {"id": 1, "prompt": [7], "max_new_tokens": 5},
            {"id": 2, "prompt": list(range(50, 62)), "max_new_tokens": 30, "arrival_s": 0.01},
        ]
    }
    trace_path = tmp / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    # Eager and graph replay one request at a time, both as one static batch, then continuous
    # batching in a 16-block pool with room for one graph.
    static = ["--policy", "static"]
    paged = ["--policy", "continuous", "--kv-block-size", "4", "--kv-blocks", "16"]
    fp16 = str(FIXTURE_DIR)
    runs: list[tuple[str, str, list[str]]] = [
        ("eager, 1 slot", fp16, ["--graph", "0", "--slots", "1", *static]),
        ("graph, 1 slot", fp16, ["--graph", "1", "--slots", "1", *static]),
        ("graph, 3 slots", fp16, ["--graph", "1", "--slots", "3", *static]),
        ("graph, paged", fp16, ["--graph", "1", "--slots", "3", "--max-graphs", "1", *paged]),
        ("eager, paged", fp16, ["--graph", "0", "--slots", "3", *paged]),
        ("unfused, cuBLAS", fp16, ["--slots", "3", "--fuse", "0", "--f16-gemv-rows", "0", *paged]),
        ("int8 kv, paged", fp16, ["--slots", "3", "--kv-dtype", "int8", *paged]),
    ]
    # head_dim 64 takes the tensor-core prefill attention, which the fixture's 16 does not.
    head_dim_64 = tmp / "head_dim_64"
    fixture.write_random_llama(head_dim_64, fixture.HEAD_DIM_64)
    for kv in ("fp16", "int8"):
        options = ["--slots", "3", "--kv-dtype", kv, *paged]
        runs.append((f"prefill tiles, {kv} kv", str(head_dim_64), options))
    for scheme in ("int8_pc", "int4_g"):
        quantized = tmp / scheme
        quantize.convert(FIXTURE_DIR, quantized, scheme, group_size=32, device="cpu")
        in_place = ["--slots", "3", "--kv-dtype", "int8", *paged]
        runs.append((f"{scheme}, in place", str(quantized), in_place))
        wide = [*in_place, "--quantized-gemm-rows", "0"]
        runs.append((f"{scheme}, tensor cores", str(quantized), wide))
        runs.append((f"{scheme}, expanded", str(quantized), [*wide, "--expand-quantized", "1"]))
    bench = [str(binary("bench", preset)), "--trace", str(trace_path), "--max-ctx", "64"]
    bench += ["--max-step-tokens", "16", "--warmup", "0", "--iters", "1"]
    return [(name, [*bench, "--model", model_dir, *options]) for name, model_dir, options in runs]


def whisper_cases(preset: str, tmp: Path) -> list[Case]:
    fp16 = str(WHISPER_FIXTURE_DIR)
    small_blocks = ["--kv-block-size", "4"]
    # Three clips over three rows; 8 blocks of 4 hold one full sequence, so the continuous run
    # preempts, re-encodes and evicts its one graph.
    transcribe = ["--mode", "transcribe", "--slots", "3", "--max-graphs", "1", *small_blocks]
    # Five beams over five rows.
    beam = ["--mode", "beam", "--slots", "5", "--beams", "5", *small_blocks]
    runs: list[tuple[str, str, list[str]]] = [
        ("whisper, fused", fp16, []),
        ("whisper, unfused, cuBLAS", fp16, ["--fuse", "0", "--f16-gemv-rows", "0", *small_blocks]),
        (
            "whisper, 3 clips a step",
            fp16,
            ["--slots", "3", "--encode-batch", "3", "--prefill-attention", "0"],
        ),
        (
            "whisper, int8 caches",
            fp16,
            ["--kv-dtype", "int8", "--cross-kv-dtype", "int8", *small_blocks],
        ),
        ("whisper transcribe, eager", fp16, [*transcribe, "--graph", "0"]),
        ("whisper transcribe, graphs", fp16, [*transcribe, "--encode-batch", "3"]),
        (
            "whisper transcribe, preempting",
            fp16,
            [*transcribe, "--policy", "continuous", "--encode-batch", "2", "--kv-blocks", "8"],
        ),
        ("whisper beam, shared", fp16, [*beam, "--graph", "1"]),
        (
            "whisper beam, private, int8",
            fp16,
            [*beam, "--graph", "0", "--beam-share-blocks", "0", "--kv-dtype", "int8"],
        ),
    ]
    for scheme in ("int8_pc", "int4_g"):
        quantized = tmp / f"whisper.{scheme}"
        quantize.convert(WHISPER_FIXTURE_DIR, quantized, scheme, group_size=32, device="cpu")
        runs.append((f"whisper {scheme}, in place", str(quantized), []))
        wide = ["--quantized-gemm-rows", "0", *small_blocks]
        runs.append((f"whisper {scheme}, tensor cores", str(quantized), wide))
        runs.append(
            (f"whisper {scheme}, expanded", str(quantized), [*wide, "--expand-quantized", "1"])
        )
    asr = str(binary("asr", preset))
    clips = str(WHISPER_FIXTURE_DIR / golden.GOLDEN_FILE)
    return [
        (name, [asr, "--model", model_dir, "--clips", clips, *options])
        for name, model_dir, options in runs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="cuda")
    parser.add_argument("--family", choices=FAMILIES, help="default: both")
    args = parser.parse_args()

    sanitizer = find_sanitizer()
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        cases: list[Case] = []
        if args.family != "whisper":
            cases += llama_cases(args.preset, Path(tmp))
        if args.family != "llama":
            cases += whisper_cases(args.preset, Path(tmp))
        for tool in TOOLS:
            for name, command in cases:
                done = subprocess.run(
                    [sanitizer, "--tool", tool, "--error-exitcode", "1", *command],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                summary = [line for line in done.stdout.splitlines() if "SUMMARY" in line]
                ok = done.returncode == 0
                failed = failed or not ok
                status = "ok  " if ok else "FAIL"
                print(f"{tool:<10} {name:<30} {status}  {' '.join(summary)}", flush=True)
                if not ok:
                    print(done.stdout[-4000:], done.stderr[-2000:])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
