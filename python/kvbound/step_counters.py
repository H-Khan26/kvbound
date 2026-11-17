"""Reads the hardware counters of every kernel in one step with Nsight Compute.

Runs kvb-bench eagerly on one request of two new tokens, profiles every kernel launch, and keeps
those of one step: the prefill of the prompt (--phase prefill) or the decode step after it
(--phase decode). Each launch gets
its time, the bytes it read from DRAM and their rate against the measured ceiling, how busy the
tensor cores were, and how many warps were resident. The counters go to
results/profiles/<run>/counters.csv, with one summary row per kernel name in summary.json.

Nsight Compute replays each launch once per group of counters with the caches flushed, so a
launch reads its weights from DRAM as it does in a real step, where they never fit in L2.

Usage: python -m kvbound.step_counters --model tinyllama-1.1b --weights int4_g --phase decode
"""

import argparse
import csv
import io
import json
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kvbound import bench, fingerprint
from kvbound.counters import find_ncu, run
from kvbound.models import load_spec
from kvbound.paths import binary
from kvbound.perfmodel.model import Arch
from kvbound.profile import PROFILES
from kvbound.weight_format import VARIANTS

METRICS = {
    "time_ns": "gpu__time_duration.sum",
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_write": "dram__bytes_write.sum",
    "dram_pct_of_peak": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_pipe_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "warps_active_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers_per_thread": "launch__registers_per_thread",
}


def kernel_name(demangled: str) -> str:
    """The engine's kernels by name, cuBLAS kernels by their short name."""
    if own := re.search(r"kvb::kernels::.*?(\w+_kernel)", demangled):
        return own.group(1)
    return re.split(r"[(<]", demangled, maxsplit=1)[0].rsplit("::", maxsplit=1)[-1][:60]


def split_steps(points: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Launches grouped by step: every step starts with the embedding lookup."""
    steps: list[list[dict[str, Any]]] = []
    for point in points:
        if point["kernel"].startswith("embedding") or not steps:
            steps.append([])
        steps[-1].append(point)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=VARIANTS, default="fp16")
    parser.add_argument("--kv", choices=bench.KV_DTYPES, default="fp16")
    parser.add_argument("--phase", choices=["prefill", "decode"], default="decode")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--preset", default="cuda")
    parser.add_argument("bench_options", nargs="*", help="extra kvb-bench options, as name=value")
    args = parser.parse_args()

    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    ncu = find_ncu()
    trace = {
        "requests": [
            {
                "id": 0,
                "prompt": bench.random_prompt(arch.vocab_size, args.prompt_len, 0),
                "max_new_tokens": 2,
            }
        ]
    }
    options = {"max-ctx": args.prompt_len + 2, "graph": 0, "kv-dtype": args.kv, "warmup": 0}
    options |= dict(option.split("=", 1) for option in args.bench_options)
    model_dir = bench.checkpoint_dir(args.model, args.weights)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "counters"
        work.mkdir()
        trace_path = Path(tmp) / "trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        report = Path(tmp) / "step.ncu-rep"
        command = [
            ncu, "--kernel-name", r"regex:^(?!.*(bf16_to_f16|f32_to_f16|count_non_finite)).*",
            "--metrics", ",".join(METRICS.values()), "--export", str(report), "--force-overwrite",
            str(binary("bench", args.preset)), "--model", str(model_dir),
            "--trace", str(trace_path), "--iters", "1",
        ]  # fmt: skip
        for name, value in options.items():
            command += [f"--{name}", str(value)]
        run(command)
        table = run(
            [ncu, "--import", str(report), "--csv", "--page", "raw", "--print-units", "base"]
        )
        rows = list(csv.reader(io.StringIO(table[table.index('"ID"') :])))
        header, launches = rows[0], rows[2:]
        points = []
        for values in launches:
            row = dict(zip(header, values, strict=True))
            point: dict[str, Any] = {
                "kernel": kernel_name(row["Kernel Name"]),
                "grid": row["Grid Size"],
                "block": row["Block Size"],
            }
            for name, metric in METRICS.items():
                point[name] = float(row[metric].replace(",", ""))
            points.append(point)
        # Two new tokens: the prefill step, then one decode step.
        steps = split_steps(points)
        if len(steps) != 2:
            raise SystemExit(f"expected a prefill and a decode step, found {len(steps)} steps")
        points = steps[0] if args.phase == "prefill" else steps[1]
        with (work / "counters.csv").open("w", encoding="utf-8", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=list(points[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(points)

        dram_gbs = None
        if ceilings := bench.measured_ceilings():
            dram_gbs = ceilings.dram_gbs
        by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for point in points:
            by_kernel[point["kernel"]].append(point)
        kernels = []
        for name, group in sorted(
            by_kernel.items(), key=lambda kv: -sum(p["time_ns"] for p in kv[1])
        ):
            time_ns = sum(p["time_ns"] for p in group)
            read = sum(p["dram_bytes_read"] for p in group)
            kernels.append({
                "kernel": name,
                "launches": len(group),
                "time_us": time_ns / 1e3,
                "dram_mb_read": read / 1e6,
                "dram_gb_per_s": read / time_ns,
                "of_measured_ceiling": read / time_ns / dram_gbs if dram_gbs else None,
                "tensor_pipe_pct": max(p["tensor_pipe_pct"] for p in group),
                "warps_active_pct": sum(p["warps_active_pct"] for p in group) / len(group),
                "registers_per_thread": max(p["registers_per_thread"] for p in group),
            })  # fmt: skip
        summary: dict[str, Any] = {
            "tool": run([ncu, "--version"]).strip().splitlines()[-1],
            "gpu": fingerprint.gpu_info()["name"],
            "git": fingerprint.git_state(),
            "model": args.model,
            "weights": args.weights,
            "kv": args.kv,
            "phase": args.phase,
            "prompt_len": args.prompt_len,
            "options": options,
            "measured_dram_gbs": dram_gbs,
            "step_time_us": sum(p["time_ns"] for p in points) / 1e3,
            "metrics": METRICS,
            "kernels": kernels,
        }
        with (work / "summary.json").open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(summary, indent=2) + "\n")
        stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        out = PROFILES / f"{stamp}-counters-{args.phase}-{args.model}-{args.weights}"
        shutil.copytree(work, out)

    print(f"{'kernel':<34}{'launches':>9}{'us':>10}{'DRAM GB/s':>11}{'tensor':>8}{'warps':>7}")
    for k in kernels:
        print(
            f"{k['kernel']:<34}{k['launches']:>9}{k['time_us']:>10.1f}{k['dram_gb_per_s']:>11.0f}"
            f"{k['tensor_pipe_pct']:>7.0f}%{k['warps_active_pct']:>6.0f}%"
        )
    print(f"step: {summary['step_time_us']:.0f} us of kernel time\n{out}")


if __name__ == "__main__":
    main()
