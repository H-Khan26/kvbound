"""Profiles a decode run with Nsight Systems and keeps the summaries.

Runs kvb-bench under `nsys profile`, exports the per-kernel and per-API-call summaries as CSV
into results/profiles/<run>/, and prints where a decode step's time goes. The .nsys-rep report
itself is left in the same directory but is not committed (see .gitignore).

Usage: python -m kvbound.profile --model tinyllama-1.1b [--no-graph]
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kvbound import bench
from kvbound.models import load_spec
from kvbound.paths import REPO_ROOT, binary
from kvbound.perfmodel.model import Arch

PROFILES = REPO_ROOT / "results" / "profiles"
# Kernels that run once per weight tensor while the model loads, not in a decode step.
LOAD_KERNELS = {"bf16_to_f16", "f32_to_f16", "count_non_finite"}


def find_nsys() -> str:
    if found := shutil.which("nsys"):
        return found
    installs = sorted(Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Systems */"))
    for install in reversed(installs):
        exe = install / "target-windows-x64" / "nsys.exe"
        if exe.exists():
            return str(exe)
    raise SystemExit("nsys not found; install Nsight Systems or add it to PATH")


def kernel_family(name: str) -> str:
    """Groups demangled kernel names: the engine's own by name, cuBLAS by kind."""
    if own := re.search(r"kvb::kernels::.*?(\w+)_kernel\(", name):
        return own.group(1)
    if "gemv" in name:
        return "cublas gemv"
    if "gemm" in name or "cutlass" in name:
        return "cublas gemm"
    return "other"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(out: Path, stem: str, steps: int) -> dict[str, Any]:
    """Per-step GPU time by kernel family and per-step CPU time by CUDA API call.

    Kernels replayed from a CUDA graph do not appear in the kernel summary, so a graph run
    reports API calls only.
    """
    kernels: dict[str, dict[str, float]] = {}
    for row in read_csv(out / f"{stem}_cuda_gpu_kern_sum.csv"):
        family = kernel_family(row["Name"])
        if family in LOAD_KERNELS:
            continue
        entry = kernels.setdefault(family, {"ms": 0.0, "launches": 0.0})
        entry["ms"] += float(row["Total Time (ns)"]) / 1e6 / steps
        entry["launches"] += int(row["Instances"]) / steps
    api = {
        row["Name"]: {
            "ms": float(row["Total Time (ns)"]) / 1e6 / steps,
            "calls": int(row["Num Calls"]) / steps,
            "median_us": float(row["Med (ns)"]) / 1e3,
        }
        for row in read_csv(out / f"{stem}_cuda_api_sum.csv")
        if any(key in row["Name"] for key in ("LaunchKernel", "GraphLaunch", "StreamSynchronize"))
    }
    return {
        "steps": steps,
        "gpu_ms_per_step": dict(sorted(kernels.items(), key=lambda kv: -kv[1]["ms"])),
        "gpu_ms_per_step_total": sum(k["ms"] for k in kernels.values()),
        "api_per_step": api,
    }


def run(command: list[str]) -> None:
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"{Path(command[0]).name} failed:\n{done.stdout}\n{done.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preset", default="cuda")
    args = parser.parse_args()

    spec = load_spec(args.model)
    nsys = find_nsys()
    stem = "graph" if args.graph else "eager"
    # A one-token prompt makes every step decode-shaped, so per-step averages mean something.
    trace = {
        "requests": [
            {
                "id": 0,
                "prompt": bench.random_prompt(Arch.from_table(spec.arch).vocab_size, 1, 0),
                "max_new_tokens": args.new_tokens,
            }
        ]
    }

    # Everything is produced in a scratch directory; results/profiles/ gains a directory only
    # once the profile succeeded.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "profile"
        work.mkdir()
        trace_path = Path(tmp) / "trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        report = work / f"{stem}.nsys-rep"
        run([
            nsys, "profile", "--trace", "cuda", "--output", str(report),
            "--force-overwrite", "true",
            str(binary("bench", args.preset)), "--model", str(spec.checkpoint_dir),
            "--trace", str(trace_path), "--max-ctx", str(args.new_tokens + 1),
            "--warmup", "0", "--iters", "1", "--graph", str(int(args.graph)),
        ])  # fmt: skip
        run([
            nsys, "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum", "--format", "csv",
            "--output", str(work / stem), "--force-export", "true", str(report),
        ])  # fmt: skip
        summary = summarize(work, stem, args.new_tokens)
        with (work / "summary.json").open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(summary, indent=2) + "\n")
        out = PROFILES / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-decode-{spec.name}-{stem}"
        shutil.copytree(work, out)

    print(f"{spec.name}, {args.new_tokens} decode steps, graph={args.graph}")
    print(f"GPU time in kernels: {summary['gpu_ms_per_step_total']:.2f} ms per step")
    for family, k in summary["gpu_ms_per_step"].items():
        print(f"  {family:<14} {k['ms']:7.3f} ms  {k['launches']:6.1f} launches")
    for name, call in summary["api_per_step"].items():
        print(
            f"  {name:<28} {call['calls']:7.1f} calls  {call['ms']:6.3f} ms"
            f"  median {call['median_us']:.1f} us"
        )
    print(out)


if __name__ == "__main__":
    main()
