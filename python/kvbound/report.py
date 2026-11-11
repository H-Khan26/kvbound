"""Turns microbenchmark records into the Phase 1 figures: roofline, bandwidth sweep, latency.

Usage: python -m kvbound.report --device rtx3060ti   (writes reports/<device>/*.png)
"""

import argparse
import tomllib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kvbound.models import load_spec
from kvbound.paths import CONFIGS, REPORTS
from kvbound.perfmodel import hardware
from kvbound.perfmodel.model import Arch

# Chart tokens: surface, inks, and the categorical series colors in their fixed order.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

PRECISION_LABELS = {
    "fp32": "FP32",
    "fp16_acc32": "FP16, FP32 accumulate",
    "fp16_acc16": "FP16, FP16 accumulate",
    "int8": "INT8 (column-major)",
    "fp8_e4m3": "FP8 E4M3",
}


def _figure(title: str, subtitle: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.07, y=0.97, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_2, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, which="both")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _bytes_label(n: float) -> str:
    for unit, size in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= size:
            return f"{n / size:g} {unit}"
    return f"{n:g} B"


def _size_axis(ax: Axes, lo: float, hi: float) -> None:
    ax.set_xscale("log", base=2)
    ticks = [float(1 << e) for e in range(10, 34, 2) if lo <= (1 << e) <= hi]
    ax.set_xticks(ticks, [_bytes_label(t) for t in ticks])
    ax.set_xlabel("working set", color=INK_2, fontsize=9)


def roofline(
    device: str, records: dict[str, dict[str, Any]], arch: Arch, model_name: str
) -> Figure:
    hw = hardware.load(device)
    shapes = records["gemm_peak"]["samples"]["shapes"]
    fig, ax = _figure(
        f"Roofline: {records['gemm_peak']['fingerprint']['gpu']['name']}",
        f"Memory caps throughput on the slope ({hw.dram_gbs:.0f} GB/s measured); arithmetic\n"
        "caps it on the flats (measured cuBLASLt GEMM peaks).",
    )
    x_lo, x_hi = 0.03, 4096.0  # room left of the slope for its labels
    top = max(hw.peak_tops.values()) * 1e3  # GFLOP/s

    # Ceilings: one flat per rate, joined to the shared memory slope. Precisions that peak
    # within 5% of each other share a flat and a label.
    flats: list[tuple[float, list[str]]] = []
    for precision, tops in sorted(hw.peak_tops.items(), key=lambda kv: kv[1]):
        if flats and tops <= flats[-1][0] * 1.05:
            flats[-1][1].append(precision)
        else:
            flats.append((tops, [precision]))
    for tops, precisions in flats:
        flat = tops * 1e3
        primary = "fp16_acc16" in precisions
        ax.plot(
            [flat / hw.dram_gbs, x_hi], [flat, flat], color=INK if primary else MUTED,
            lw=2 if primary else 1,
        )  # fmt: skip
        names = " / ".join(PRECISION_LABELS[name] for name in precisions)
        ax.text(x_hi * 1.06, flat, f"{names}: {tops:.1f} e12 ops/s", va="center", fontsize=8,
                color=INK if primary else INK_2)  # fmt: skip
    ax.plot([x_lo, top / hw.dram_gbs], [x_lo * hw.dram_gbs, top], color=INK, lw=2)

    def note(text: str, xy: tuple[float, float], offset: tuple[int, int], ha: str) -> None:
        ax.annotate(text, xy, offset, textcoords="offset points", fontsize=8, color=INK_2,
                    ha=ha, va="center")  # fmt: skip

    # Measured: one FP16 linear layer applied to 1..1024 token vectors.
    rows, cols = shapes["rows"], shapes["cols"]
    xs, ys = [], []
    for point in shapes["points"]:
        moved = rows * cols * 2 + point["tokens"] * (rows + cols) * 2
        xs.append(2 * rows * cols * point["tokens"] / moved)
        ys.append(point["best_gflops"])
    ax.plot(xs, ys, "o", ms=6, color=SERIES[0], mec=SURFACE, mew=1, zorder=3,
            label="measured cuBLASLt FP16 GEMM, 1 to 1024 tokens")  # fmt: skip

    # Where the model's operations sit on the bound.
    marks = [
        (1.0, "linear layers, decode batch 1"),
        (float(arch.n_heads // arch.n_kv_heads), "attention over the KV cache"),
        (8.0, "linear layers, decode batch 8"),
        (512.0, "linear layers, prefill 512 tokens"),
    ]
    roof_top = hw.peak_tops["fp16_acc16"] * 1e3
    mx = [mark[0] for mark in marks]
    my = [min(mark[0] * hw.dram_gbs, roof_top) for mark in marks]
    ax.plot(mx, my, "D", ms=7, color=SERIES[1], mec=SURFACE, mew=1, zorder=4,
            label=f"bound for {model_name} operations")  # fmt: skip
    for (x, text), y in zip(marks, my, strict=True):
        on_flat = x * hw.dram_gbs > roof_top
        note(text, (x, y), (0, 14) if on_flat else (-12, 10), "center" if on_flat else "right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(x_lo * hw.dram_gbs * 0.5, top * 2.5)
    ax.set_xlabel("arithmetic intensity (operations per byte moved)", color=INK_2, fontsize=9)
    ax.set_ylabel("throughput (GFLOP/s)", color=INK_2, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    return fig


def bandwidth(device: str, records: dict[str, dict[str, Any]]) -> Figure:
    rec = records["bandwidth_sweep"]
    spec = tomllib.loads((CONFIGS / "devices" / f"{device}.toml").read_text(encoding="utf-8"))
    metrics, points = rec["metrics"], rec["samples"]["points"]
    l2 = rec["samples"]["device"]["l2_cache_bytes"]
    fig, ax = _figure(
        f"Memory bandwidth vs working set: {rec['fingerprint']['gpu']['name']}",
        f"Past the caches every mode settles near {metrics['dram_read_gbs']:.0f} GB/s, "
        f"{metrics['dram_read_gbs'] / spec['memory_bandwidth_gbs']:.0%} of the "
        f"{spec['memory_bandwidth_gbs']} GB/s spec. That plateau is the decode ceiling.",
    )
    sizes = [p["working_set_bytes"] for p in points]
    for mode, color in zip(("read", "write", "copy"), SERIES, strict=True):
        series = [p for p in points if p["mode"] == mode]
        ax.plot([p["working_set_bytes"] for p in series], [p["best_gbs"] for p in series],
                color=color, lw=2, label=mode)  # fmt: skip
        # Labels sit inside the cache region, where the three lines are apart.
        in_cache = [p for p in series if p["working_set_bytes"] <= l2 // 4][-1]
        ax.annotate(mode, (in_cache["working_set_bytes"], in_cache["best_gbs"]), (0, 9),
                    textcoords="offset points", fontsize=8.5, color=INK_2, ha="center")  # fmt: skip
    ax.axhline(spec["memory_bandwidth_gbs"], color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(min(sizes), spec["memory_bandwidth_gbs"] * 1.08,
            f"spec {spec['memory_bandwidth_gbs']} GB/s", fontsize=8, color=INK_2)  # fmt: skip
    ax.axvline(l2, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(l2 * 1.1, 215, f"L2 {_bytes_label(l2)}", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ticks = [250, 500, 1000, 2000, 4000]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    _size_axis(ax, min(sizes), max(sizes))
    ax.set_ylabel("bytes moved per second (GB/s)", color=INK_2, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    return fig


def latency(records: dict[str, dict[str, Any]]) -> Figure:
    rec = records["pointer_chase"]
    m, points = rec["metrics"], rec["samples"]["points"]
    l2 = rec["samples"]["device"]["l2_cache_bytes"]
    fig, ax = _figure(
        f"Dependent-load latency vs working set: {rec['fingerprint']['gpu']['name']}",
        f"One thread chasing pointers: {m['l1_ns']:.0f} ns from L1, {m['l2_ns']:.0f} ns from L2, "
        f"{m['dram_ns']:.0f} ns from DRAM.",
    )
    sizes = [p["working_set_bytes"] for p in points]
    ax.plot(sizes, [p["best_ns_per_hop"] for p in points], color=SERIES[0], lw=2, marker="o",
            ms=5, mec=SURFACE, mew=1)  # fmt: skip
    for label, value, x in (("L1", m["l1_ns"], sizes[1]), ("L2", m["l2_ns"], l2 / 8),
                            ("DRAM", m["dram_ns"], sizes[-4])):  # fmt: skip
        ax.text(x, value + 14, f"{label}  {value:.0f} ns", fontsize=8.5, color=INK_2)
    ax.axvline(l2, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(l2 * 1.15, 6, f"L2 {_bytes_label(l2)}", fontsize=8, color=INK_2)
    _size_axis(ax, min(sizes), max(sizes))
    ax.set_ylim(0, max(p["best_ns_per_hop"] for p in points) * 1.15)
    ax.set_ylabel("time per pointer hop (ns)", color=INK_2, fontsize=9)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="name from configs/devices/")
    parser.add_argument("--model", default="llama-3.2-1b", help="model placed on the roofline")
    args = parser.parse_args()

    records = hardware.latest_records(args.device)
    missing = {"bandwidth_sweep", "pointer_chase", "gemm_peak"} - set(records)
    if missing:
        raise SystemExit(f"no {sorted(missing)} records for {args.device}; run `make microbench`")
    arch = Arch.from_table(load_spec(args.model).arch)
    out = REPORTS / args.device
    print(_save(roofline(args.device, records, arch, args.model), out / "roofline.png"))
    print(_save(bandwidth(args.device, records), out / "bandwidth.png"))
    print(_save(latency(records), out / "latency.png"))


if __name__ == "__main__":
    main()
