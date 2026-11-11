"""Hardware ceilings the perf model divides by: memory bandwidth and peak arithmetic rates.

Measured ceilings come from the microbenchmark records in results/raw/; a device with no records
yet falls back to its spec sheet in configs/devices/.
"""

import tomllib
from dataclasses import dataclass
from typing import Any

from kvbound import record
from kvbound.paths import CONFIGS


@dataclass(frozen=True)
class Ceilings:
    device: str  # configs/devices/<device>.toml
    source: str  # "measured" or "spec"
    dram_gbs: float  # bytes/s the memory bus streams, in GB/s (1e9)
    peak_tops: dict[str, float]  # precision -> 1e12 operations per second
    l2_cache_bytes: int

    def peak_ops(self, precision: str) -> float:
        return self.peak_tops[precision] * 1e12


def _spec(device: str) -> Ceilings:
    table = tomllib.loads((CONFIGS / "devices" / f"{device}.toml").read_text(encoding="utf-8"))
    spec = table["spec_peak_dense"]
    peaks = {
        "fp32": spec["fp32_tflops"],
        "fp16_acc16": spec["fp16_tensor_fp16acc_tflops"],
        "fp16_acc32": spec["fp16_tensor_fp32acc_tflops"],
        "int8": spec["int8_tensor_tops"],
    }
    if "fp8_tensor_fp32acc_tflops" in spec:
        peaks["fp8_e4m3"] = spec["fp8_tensor_fp32acc_tflops"]
    return Ceilings(
        device=device,
        source="spec",
        dram_gbs=table["memory_bandwidth_gbs"],
        peak_tops=peaks,
        l2_cache_bytes=table["l2_cache_mb"] << 20,
    )


def latest_records(device: str) -> dict[str, dict[str, Any]]:
    """Newest record of each microbenchmark kind for a device config."""
    latest = {}
    for rec in record.load_records():
        if rec["fingerprint"]["device_config"] == device and rec["kind"].startswith("microbench."):
            latest[rec["kind"].removeprefix("microbench.")] = rec
    return latest


def load(device: str, source: str = "measured") -> Ceilings:
    """Ceilings for a device. source="measured" needs bandwidth_sweep and gemm_peak records."""
    spec = _spec(device)
    if source == "spec":
        return spec
    records = latest_records(device)
    missing = {"bandwidth_sweep", "gemm_peak"} - set(records)
    if missing:
        raise SystemExit(
            f"no {sorted(missing)} records for {device}; run `make microbench` on that machine "
            "or pass --ceilings spec"
        )
    return Ceilings(
        device=device,
        source="measured",
        dram_gbs=records["bandwidth_sweep"]["metrics"]["dram_read_gbs"],
        peak_tops=records["gemm_peak"]["metrics"]["peak_tops"],
        l2_cache_bytes=records["bandwidth_sweep"]["samples"]["device"]["l2_cache_bytes"],
    )
