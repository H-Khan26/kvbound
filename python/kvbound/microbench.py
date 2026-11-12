"""Runs the hardware microbenchmarks and records what they measured.

The C++/CUDA binaries under build/<preset>/bin print raw JSON; this wrapper samples NVML while
they run, reduces the raw points to the handful of ceilings the perf model needs, and writes the
result record.

Usage: python -m kvbound.microbench [bandwidth_sweep pointer_chase gemm_peak]
"""

import argparse
import json
import platform
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kvbound import record
from kvbound.nvml_sampler import NvmlSampler
from kvbound.paths import BUILD

MIB = 1 << 20
# Working sets at least this large are far beyond any cache on either card.
DRAM_MIN_BYTES = 256 * MIB


def summarize_bandwidth(raw: dict[str, Any]) -> dict[str, float]:
    """DRAM plateau (median over the large working sets) and the best in-cache rate.

    dram_<mode>_gbs is the burst rate, what the hardware delivers undisturbed and so the ceiling
    the perf model uses. dram_<mode>_sustained_gbs is what a long run got on this machine,
    including time lost to other GPU clients.
    """
    metrics = {}
    for mode in ("read", "write", "copy"):
        points = [p for p in raw["points"] if p["mode"] == mode]
        plateau = [p for p in points if p["working_set_bytes"] >= DRAM_MIN_BYTES]
        metrics[f"dram_{mode}_gbs"] = statistics.median(p["burst_gbs"] for p in plateau)
        metrics[f"dram_{mode}_sustained_gbs"] = statistics.median(p["best_gbs"] for p in plateau)
        metrics[f"cache_peak_{mode}_gbs"] = max(p["burst_gbs"] for p in points)
    return metrics


def summarize_pointer_chase(raw: dict[str, Any]) -> dict[str, float]:
    """Latency of one dependent load served by L1, by L2 and by DRAM."""
    points = raw["points"]
    l2_bytes = raw["device"]["l2_cache_bytes"]
    # A quarter of the L2 size is well inside L2 and well beyond L1.
    in_l2 = [p for p in points if p["working_set_bytes"] <= l2_bytes // 4]
    in_dram = [p["best_ns_per_hop"] for p in points if p["working_set_bytes"] >= DRAM_MIN_BYTES]
    return {
        "l1_ns": min(p["best_ns_per_hop"] for p in points),
        "l2_ns": in_l2[-1]["best_ns_per_hop"],
        "dram_ns": statistics.median(in_dram),
    }


def summarize_gemm(raw: dict[str, Any]) -> dict[str, Any]:
    """Best throughput per precision (TFLOPS, or TOPS for integers) and the decode read rate."""
    peak: dict[str, float] = {}
    unsupported: dict[str, str] = {}
    for p in raw["peak"]:
        if p["status"] == "success":
            peak[p["precision"]] = max(peak.get(p["precision"], 0.0), p["best_tops"])
        else:
            unsupported[p["precision"]] = p["status"]
    one_token = next(p for p in raw["shapes"]["points"] if p["tokens"] == 1)
    return {
        "peak_tops": peak,
        "unsupported": unsupported,
        # Weight bytes per second cuBLAS streams in a one-token (decode-shaped) GEMM.
        "decode_gemm_weight_gbs": one_token["weight_gbs"],
    }


SUMMARIZERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "bandwidth_sweep": summarize_bandwidth,
    "pointer_chase": summarize_pointer_chase,
    "gemm_peak": summarize_gemm,
}


def binary(name: str, preset: str) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    path = BUILD / preset / "bin" / f"kvb-{name}{suffix}"
    if not path.exists():
        raise SystemExit(f"{path} not found; run `make build PRESET={preset}` first")
    return path


def run(name: str, preset: str) -> Path:
    """Runs one microbenchmark and writes its record; returns the record path."""
    exe = binary(name, preset)
    with NvmlSampler() as sampler:
        done = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"{exe.name} failed:\n{done.stderr}")
    raw = json.loads(done.stdout)
    return record.write_record(
        kind=f"microbench.{name}",
        engine={"name": exe.name, "preset": preset},
        params=raw["params"],
        metrics=SUMMARIZERS[name](raw),
        samples={key: raw[key] for key in ("device", "points", "peak", "shapes") if key in raw},
        nvml=sampler.summary(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help=f"any of {list(SUMMARIZERS)}; default all")
    parser.add_argument("--preset", default="cuda")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    names = args.names or list(SUMMARIZERS)
    if unknown := set(names) - set(SUMMARIZERS):
        parser.error(f"unknown microbenchmarks: {sorted(unknown)}")

    record.require_clean_tree(args.allow_dirty)
    for name in names:
        path = run(name, args.preset)
        metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
        print(f"{name}: {json.dumps(metrics)}\n  {path}")


if __name__ == "__main__":
    main()
