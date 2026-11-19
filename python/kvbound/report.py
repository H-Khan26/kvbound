"""Turns result records into figures under reports/<device>/.

Hardware: roofline, bandwidth sweep, latency. Engine: KV layout benchmark, KV memory under static
batching, the activation memory plan, the three serving setups in one KV pool, latency under
arriving load, and the precision comparison. Production comparison (both models): tokens per
second of each stack, and perplexity against bits per weight of every stack's formats. Figures
whose records do not exist yet are skipped.

Usage: python -m kvbound.report --device rtx3060ti --model tinyllama-1.1b
       python -m kvbound.report --device rtx3060ti --model tinyllama-1.1b --comparison-only
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
from matplotlib.patches import Rectangle

from kvbound import record, workload
from kvbound.models import load_spec
from kvbound.paths import CONFIGS, REPORTS
from kvbound.perfmodel import hardware
from kvbound.perfmodel.model import Arch, Precision, step_costs
from kvbound.weight_format import WeightFormat

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
    "int8": "INT8",
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


def _save(fig: Figure, path: Path, top: float = 0.95) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, top))
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
            label="measured cuBLASLt FP16 GEMM (FP32 accumulate), 1 to 1024 tokens")  # fmt: skip

    # Where the model's operations sit on the bound, taken from the perf model's own costs.
    fp16 = Precision.named("fp16", "fp16")
    marks = []
    for label, batch, new_tokens, context, op in (
        ("linear layers, decode batch 1", 1, 1, 1024, "gate_proj"),
        ("attention over the KV cache", 1, 1, 1024, "attention"),
        ("linear layers, decode batch 8", 8, 1, 1024, "gate_proj"),
        ("linear layers, prefill 512", 1, 512, 0, "gate_proj"),
    ):
        costs = step_costs(arch, hw, fp16, batch, new_tokens, context)
        cost = next(c for c in costs if c.name == op)
        marks.append((cost.intensity, cost.ops / cost.seconds / 1e9, cost.bound_by, label))
    ax.plot([m[0] for m in marks], [m[1] for m in marks], "D", ms=7, color=SERIES[1],
            mec=SURFACE, mew=1, zorder=4, label=f"bound for {model_name} operations")  # fmt: skip
    for x, y, bound_by, text in marks:
        # Labels go left of the slope, except attention and compute-bound marks, which go right.
        if bound_by == "compute":
            note(text, (x, y), (-22, 12), "left")
        elif text.startswith("attention"):
            note(text, (x, y), (12, -13), "left")
        else:
            note(text, (x, y), (-12, 10), "right")

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
        f"Past the caches reads settle near {metrics['dram_read_gbs']:.0f} GB/s, "
        f"{metrics['dram_read_gbs'] / spec['memory_bandwidth_gbs']:.0%} of the "
        f"{spec['memory_bandwidth_gbs']} GB/s spec: the decode ceiling.\n"
        f"Launches of 0.1 s average {metrics['dram_read_sustained_gbs']:.0f} GB/s (thin line) "
        "because other programs take turns on the GPU.",
    )
    sizes = [p["working_set_bytes"] for p in points]
    for mode, color in zip(("read", "write", "copy"), SERIES, strict=True):
        series = [p for p in points if p["mode"] == mode]
        ax.plot([p["working_set_bytes"] for p in series], [p["burst_gbs"] for p in series],
                color=color, lw=2, label=f"{mode}, best 2 ms burst")  # fmt: skip
        # Labels sit inside the cache region, where the three lines are apart.
        in_cache = [p for p in series if p["working_set_bytes"] <= l2 // 4][-1]
        ax.annotate(mode, (in_cache["working_set_bytes"], in_cache["burst_gbs"]), (0, 9),
                    textcoords="offset points", fontsize=8.5, color=INK_2, ha="center")  # fmt: skip
    reads = [p for p in points if p["mode"] == "read"]
    ax.plot([p["working_set_bytes"] for p in reads], [p["best_gbs"] for p in reads],
            color=SERIES[0], lw=1, label="read, sustained 0.1 s")  # fmt: skip
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


LAYOUT_LABELS = {
    "head_major": "head-major",
    "token_major": "token-major",
    "dim_major": "dimension-major",
}


def kv_layout(records: dict[str, dict[str, Any]]) -> Figure:
    rec = records["kv_layout"]
    # The FP16 points with one block per sequence. Points without the two fields are of
    # that kind.
    points = [
        p
        for p in rec["samples"]["points"]
        if p["shape"] == "decode"
        and p.get("placement", "single") == "single"
        and p.get("kv_dtype", "fp16") == "fp16"
    ]
    params = rec["params"]
    best = rec["metrics"]["attention_ms"]
    speedup = best["head_major/scalar"]["decode_batch1"] / best["head_major/wide"]["decode_batch1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(
        "Attention time per decode step, by KV layout and load width",
        x=0.07, y=0.97, ha="left", fontsize=13, fontweight="bold", color=INK,
    )  # fmt: skip
    fig.text(
        0.07, 0.9,
        f"Reading a key as 16-byte loads is {speedup:.1f}x faster than 2-byte loads. Which "
        "dimension is outermost matters far less.\n"
        f"{params['n_layers']} layers, {params['n_heads']} query heads over "
        f"{params['n_kv_heads']} KV heads of {params['head_dim']} dimensions, FP16.",
        fontsize=9.5, color=INK_2, va="top",
    )  # fmt: skip
    for ax, batch in zip(axes, (1, params["slots"]), strict=True):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5, which="both")
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for layout, color in zip(LAYOUT_LABELS, SERIES, strict=True):
            for loads, style in (("wide", "-"), ("scalar", (0, (4, 2)))):
                series = [
                    p for p in points
                    if (p["layout"], p["key_loads"], p["batch"]) == (layout, loads, batch)
                ]  # fmt: skip
                if not series:
                    continue
                label = (
                    f"{LAYOUT_LABELS[layout]}, {'16-byte' if loads == 'wide' else '2-byte'} loads"
                )
                ax.plot([p["ctx"] for p in series], [p["attention_ms"] for p in series],
                        color=color, lw=2, ls=style, marker="o", ms=3.5, label=label)  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        contexts = sorted({p["ctx"] for p in points})
        ax.set_xticks(contexts, [str(c) for c in contexts])
        ax.minorticks_off()
        ax.set_xlabel("tokens in the cache", color=INK_2, fontsize=9)
        ax.set_title(f"batch {batch}", loc="left", fontsize=9.5, color=INK_2)
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    axes[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
    axes[0].set_ylabel("milliseconds per step, all layers", color=INK_2, fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)
    return fig


def serve_setup(rec: dict[str, Any]) -> str:
    """What a serve record ran: "static" batching, "continuous" batching over whole-sequence
    slots, or "paged" (continuous over blocks smaller than a sequence)."""
    params = rec["params"]
    if params.get("policy", "static") == "static":
        return "static"
    return "paged" if params["kv_block_size"] < params["max_ctx"] else "continuous"


def latest_serve_records(
    device: str, model: str, weights: str = "fp16", kv: str = "fp16", pool_seqs: int = 8
) -> dict[tuple[str, str, int, float], dict[str, Any]]:
    """Newest serve record per (workload, setup, slots, arrival rate) for a device config, model
    and precision. Only runs of each workload's own request count are kept, and only runs in the
    shared KV pool of pool_seqs full sequences or with one slot (one request at a time)."""
    latest = {}
    for rec in record.load_records("serve"):
        params = rec["params"]
        if rec["fingerprint"]["device_config"] != device or rec["model"]["name"] != model:
            continue
        if record.model_family(rec) != record.DEFAULT_FAMILY or rec["engine"]["name"] != "kvbound":
            continue
        if (params.get("weights", "fp16"), params.get("kv_dtype", "fp16")) != (weights, kv):
            continue
        if params["n_requests"] != workload.load(params["workload"]).n_requests:
            continue
        pool_tokens = params["kv_blocks"] * params["kv_block_size"]
        if params["slots"] != 1 and pool_tokens != pool_seqs * params["max_ctx"]:
            continue
        key = (params["workload"], serve_setup(rec), params["slots"], params.get("arrival_rate", 0))
        latest[key] = rec
    return latest


def static_memory(serve: dict[tuple[str, str, int, float], dict[str, Any]], slots: int) -> Figure:
    runs = {
        name: rec
        for (name, setup, n, rate), rec in serve.items()
        if (setup, n, rate) == ("static", slots, 0)
    }
    first = next(iter(runs.values()))
    per_token = first["metrics"]["memory"]["kv_bytes_per_token"]
    pool = first["params"]["kv_blocks"] * first["params"]["kv_block_size"]
    to_mib = per_token / 2**20
    fig, grid = plt.subplots(1, len(runs), figsize=(9, 5.2), dpi=160, sharey=True, squeeze=False)
    axes = grid[0]
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(
        f"KV cache memory under static batching, {slots} slots",
        x=0.07, y=0.97, ha="left", fontsize=13, fontweight="bold", color=INK,
    )  # fmt: skip
    fig.text(
        0.07, 0.9,
        f"The whole pool ({pool * to_mib:.0f} MiB) is allocated up front and every admitted "
        f"sequence holds a full {first['params']['max_ctx']}-token slot.\n"
        "The filled area is the part that holds tokens; the line is the part reserved for "
        "admitted sequences.",
        fontsize=9.5, color=INK_2, va="top",
    )  # fmt: skip
    for ax, (name, rec) in zip(axes, runs.items(), strict=True):
        steps, m = rec["samples"]["steps"], rec["metrics"]
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        t = steps["end_s"]
        ax.fill_between(t, [u * to_mib for u in steps["used_tokens"]], step="pre",
                        color=SERIES[0], alpha=0.85, lw=0, label="holds tokens")  # fmt: skip
        ax.plot(t, [r * to_mib for r in steps["reserved_tokens"]], drawstyle="steps-pre",
                color=INK, lw=1.2, label="reserved for sequences")  # fmt: skip
        ax.set_title(
            f"{name}\n{m['throughput_tok_s']['p50']:.0f} tokens/s, "
            f"{m['kv_used_of_reserved']['p50']:.0%} of reserved in use,\n"
            f"slots busy {m['slot_occupancy']['p50']:.0%} of the time",
            loc="left", fontsize=9, color=INK_2,
        )  # fmt: skip
        ax.set_xlabel("seconds", color=INK_2, fontsize=9)
        ax.set_xlim(0, max(t))
    axes[0].set_ylabel("KV cache (MiB)", color=INK_2, fontsize=9)
    axes[-1].legend(loc="center right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    return fig


ServeRecords = dict[tuple[str, str, int, float], dict[str, Any]]

# The three ways a workload is served in one KV pool, in the order their colors are assigned.
SETUP_LABELS = {
    "static": "static batches",
    "continuous": "continuous, whole slots",
    "paged": "continuous, 16-token blocks",
}
WORKLOADS = ("fixed", "uniform", "longtail")


def _panels(rows: int, cols: int, title: str, subtitle: str, **kwargs: Any) -> tuple[Figure, Any]:
    """A grid of axes in the house style, with the title and subtitle above it."""
    fig, grid = plt.subplots(rows, cols, figsize=(9, 5.6), dpi=160, squeeze=False, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.07, y=0.97, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.07, 0.9, subtitle, fontsize=9.5, color=INK_2, va="top")
    for ax in grid.flat:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5, which="both")
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return fig, grid


def _bars(ax: Axes, groups: list[str], series: dict[str, list[float]], fmt: str) -> None:
    """Grouped bars with a value above each; series colors follow SERIES order."""
    width = 0.8 / len(series)
    for i, (label, values) in enumerate(series.items()):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        ax.bar(xs, values, width * 0.92, color=SERIES[i], label=label)
        for x, value in zip(xs, values, strict=True):
            ax.annotate(format(value, fmt), (x, value), (0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK_2)  # fmt: skip
    ax.set_xticks(range(len(groups)), groups)
    ax.margins(y=0.15)


def _bursts(serve: ServeRecords, setup: str) -> list[dict[str, Any]]:
    """One setup's record for each workload with every request queued at time zero, or an empty
    list when any is missing. Of several slot counts the largest is taken (static runs also exist
    one slot at a time)."""
    records = []
    for name in WORKLOADS:
        runs = [
            (n, rec) for (w, s, n, rate), rec in serve.items() if (w, s, rate) == (name, setup, 0)
        ]
        if not runs:
            return []
        records.append(max(runs, key=lambda run: run[0])[1])
    return records


def serving(serve: ServeRecords) -> Figure | None:
    """Static batching against continuous batching and paging on the three workloads."""
    runs = {setup: _bursts(serve, setup) for setup in SETUP_LABELS}
    if not all(runs.values()):
        return None
    first = runs["static"][0]
    pool_mib = first["metrics"]["memory"]["kv_pool_bytes"] / 2**20
    fig, grid = _panels(
        2, 2,
        f"One {pool_mib:.0f} MiB KV pool served three ways",
        f"48 requests queued at once per workload. Static batches hold {first['params']['slots']}"
        " whole-context slots;\nthe paged engine shares the same bytes block by block.",
    )  # fmt: skip
    panels = (
        ("throughput_tok_s", 1.0, "tokens per second", ".0f"),
        ("ttft_p50_s", 1.0, "time to first token, median (s)", ".2f"),
        ("tpot_p50_s", 1e3, "time per output token, median (ms)", ".1f"),
        ("running_peak", 1.0, "most sequences in one step", ".0f"),
    )
    for ax, (metric, scale, label, fmt) in zip(grid.flat, panels, strict=True):
        series = {
            SETUP_LABELS[setup]: [rec["metrics"][metric]["p50"] * scale for rec in recs]
            for setup, recs in runs.items()
        }
        _bars(ax, list(WORKLOADS), series, fmt)
        ax.set_title(label, loc="left", fontsize=9.5, color=INK_2)
    handles, labels = grid[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99), frameon=False,
               fontsize=8.5, labelcolor=INK_2)  # fmt: skip
    return fig


def paged_memory(serve: ServeRecords) -> Figure | None:
    """KV memory over time, static slots above and 16-token blocks below, per workload."""
    rows = {setup: _bursts(serve, setup) for setup in ("static", "paged")}
    if not all(rows.values()):
        return None
    first = rows["static"][0]
    to_mib = first["metrics"]["memory"]["kv_bytes_per_token"] / 2**20
    pool_mib = first["metrics"]["memory"]["kv_pool_bytes"] / 2**20
    fig, grid = _panels(
        2, 3,
        "KV cache memory: whole slots against 16-token blocks",
        f"The same {pool_mib:.0f} MiB pool and the same requests. The filled area holds tokens;"
        " the line is memory reserved\nfor admitted sequences. Top row: static batches. Bottom"
        " row: continuous batching over blocks.",
        sharey=True,
    )  # fmt: skip
    for row, setup in enumerate(rows):
        for col, rec in enumerate(rows[setup]):
            ax, steps, m = grid[row][col], rec["samples"]["steps"], rec["metrics"]
            t = steps["end_s"]
            ax.fill_between(t, [u * to_mib for u in steps["used_tokens"]], step="pre",
                            color=SERIES[0], alpha=0.85, lw=0, label="holds tokens")  # fmt: skip
            ax.plot(t, [r * to_mib for r in steps["reserved_tokens"]], drawstyle="steps-pre",
                    color=INK, lw=1.1, label="reserved")  # fmt: skip
            ax.set_title(
                f"{WORKLOADS[col]}: {m['throughput_tok_s']['p50']:.0f} tokens/s,\n"
                f"{m['kv_used_of_reserved']['p50']:.0%} of reserved in use",
                loc="left", fontsize=8.5, color=INK_2,
            )  # fmt: skip
            # Both rows of a column share the static run's time span.
            ax.set_xlim(0, max(rows["static"][col]["samples"]["steps"]["end_s"]))
        grid[row][0].set_ylabel("KV cache (MiB)", color=INK_2, fontsize=9)
    for ax in grid[1]:
        ax.set_xlabel("seconds", color=INK_2, fontsize=9)
    grid[0][2].legend(loc="center right", frameon=False, fontsize=8, labelcolor=INK_2)
    return fig


def serving_load(serve: ServeRecords, name: str = "longtail") -> Figure | None:
    """Latency as requests arrive faster: static batches against the paged engine."""
    series: dict[str, list[tuple[float, dict[str, Any]]]] = {"static": [], "paged": []}
    for (w, setup, _, rate), rec in sorted(serve.items(), key=lambda item: item[0][3]):
        if w == name and rate > 0 and setup in series:
            series[setup].append((rate, rec["metrics"]))
    if not series["static"] or not series["paged"]:
        return None
    fig, grid = _panels(
        1, 2,
        f"Latency as {name} requests arrive faster",
        "Poisson arrivals. Solid lines are medians over the requests,\ndashed lines the 90th"
        " percentile.",
        sharey=True,
    )  # fmt: skip
    panels = (("ttft", "time to first token (s)"), ("latency", "time to last token (s)"))
    for ax, (metric, label) in zip(grid[0], panels, strict=True):
        for setup, color in (("static", SERIES[0]), ("paged", SERIES[2])):
            rates = [rate for rate, _ in series[setup]]
            for q, style in (("p50", "-"), ("p90", (0, (4, 2)))):
                values = [m[f"{metric}_{q}_s"]["p50"] for _, m in series[setup]]
                quantile = "median" if q == "p50" else "90th percentile"
                ax.plot(rates, values, color=color, lw=2, ls=style, marker="o", ms=4,
                        label=f"{SETUP_LABELS[setup]}, {quantile}")  # fmt: skip
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        rates = sorted({rate for runs in series.values() for rate, _ in runs})
        ax.set_xticks(rates, [f"{r:g}" for r in rates])
        seconds = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
        ax.set_yticks(seconds, [f"{t:g}" for t in seconds])
        ax.minorticks_off()
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_xlabel("requests per second", color=INK_2, fontsize=9)
        ax.set_title(label, loc="left", fontsize=9.5, color=INK_2)
    handles, labels = grid[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99), frameon=False,
               fontsize=7.5, labelcolor=INK_2)  # fmt: skip
    return fig


PRECISION_ORDER = (
    ("fp16", "fp16"),
    ("fp16", "int8"),
    ("int8_pc", "fp16"),
    ("int8_pc", "int8"),
    ("int4_g", "fp16"),
    ("int4_g", "int8"),
)
WEIGHT_LABELS = {"fp16": "FP16", "int8_pc": "INT8", "int4_g": "INT4"}


def precision_records(device: str, model: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (weights, kv): the newest perplexity (whole test split), batch-1 decode benchmark
    (128-token prompt, with graphs) and paged serve record."""
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in record.load_records():
        if rec["fingerprint"]["device_config"] != device:
            continue
        if not rec["model"] or rec["model"]["name"] != model:
            continue
        if record.model_family(rec) != record.DEFAULT_FAMILY:
            continue
        params = rec["params"]
        if rec["engine"]["name"] != "kvbound":
            continue
        if rec["kind"] == "eval.perplexity" and params["max_windows"] == 0:
            table.setdefault((params["weights"], params["kv"]), {})["perplexity"] = rec
        elif rec["kind"] == "generate" and (params["prompt_len"], params["graph"]) == (128, True):
            key = (params.get("weights", "fp16"), params.get("kv_dtype", "fp16"))
            table.setdefault(key, {})["decode"] = rec
        elif rec["kind"] == "serve" and serve_setup(rec) == "paged" and params["n_requests"] == 96:
            key = (params.get("weights", "fp16"), params.get("kv_dtype", "fp16"))
            table.setdefault(key, {})["serve"] = rec
    return table


def precision(table: dict[tuple[str, str], dict[str, Any]], model: str) -> Figure | None:
    """What each weight and KV precision costs and buys."""
    rows = [(key, table.get(key, {})) for key in PRECISION_ORDER]
    if any({"perplexity", "decode", "serve"} - set(recs) for _, recs in rows):
        return None
    labels = [f"{WEIGHT_LABELS[w]}\n{'INT8' if kv == 'int8' else 'FP16'} KV" for (w, kv), _ in rows]
    fig, grid = _panels(
        2, 2,
        f"Precision: what each configuration costs and buys ({model})",
        "Labels give the weight format, then the KV cache format. The top row is one request at"
        " a time;\nthe bottom row is 96 requests served together in a KV pool of the same bytes.",
    )  # fmt: skip
    x = range(len(rows))

    def panel(ax: Axes, values: list[float], title: str, fmt: str, inside: bool = False) -> None:
        """Bars with their values above them, or inside the bar top where a tick sits above."""
        ax.bar(x, values, 0.62, color=SERIES[0])
        for i, value in enumerate(values):
            ax.annotate(format(value, fmt), (i, value), (0, -3 if inside else 2),
                        textcoords="offset points", ha="center",
                        va="top" if inside else "bottom", fontsize=8,
                        color=SURFACE if inside else INK_2)  # fmt: skip
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK_2)
        ax.margins(y=0.18)

    decode = [recs["decode"]["metrics"] for _, recs in rows]
    panel(grid[0][0], [m["decode_tok_s"]["p50"] for m in decode],
          "decode tokens per second (tick: memory-bandwidth ceiling)", ".0f", True)  # fmt: skip
    for i, m in enumerate(decode):
        if "ceiling" in m:
            ceiling = m["ceiling"]["decode_tok_s"]
            grid[0][0].plot([i - 0.36, i + 0.36], [ceiling, ceiling], color=INK, lw=1.4)
    panel(grid[0][1], [recs["perplexity"]["metrics"]["perplexity"] for _, recs in rows],
          "WikiText-2 perplexity (lower is better)", ".2f")  # fmt: skip
    serve = [recs["serve"]["metrics"] for _, recs in rows]
    panel(grid[1][0], [m["running_peak"]["p50"] for m in serve],
          "most sequences in one step", ".0f")  # fmt: skip
    panel(grid[1][1], [m["throughput_tok_s"]["p50"] for m in serve],
          "tokens per second under that load", ".0f")  # fmt: skip
    return fig


def arena(rec: dict[str, Any]) -> Figure:
    """Each activation as a rectangle: the ops it lives through by the arena bytes it occupies."""
    plan, memory = rec["samples"]["plan"], rec["metrics"]["memory"]
    mib = 2**20
    fig, ax = _figure(
        f"Activation memory plan: {rec['model']['name']}, "
        f"{rec['params']['max_step_tokens']}-token steps",
        f"{len(plan)} intermediate tensors would need {memory['arena_naive_bytes'] / mib:,.0f} "
        f"MiB side by side. Placed by lifetime they share "
        f"{memory['arena_bytes'] / mib:,.0f} MiB;\nno plan can use less than "
        f"{memory['arena_peak_live_bytes'] / mib:,.0f} MiB, the most that is alive at once.",
    )
    for value in plan:
        ax.add_patch(
            Rectangle(
                (value["first_op"], value["offset"] / mib),
                value["last_op"] - value["first_op"] + 1,
                value["bytes"] / mib,
                facecolor=SERIES[0], edgecolor=SURFACE, linewidth=0.4, alpha=0.9,
            )
        )  # fmt: skip
    residual = plan[0]
    ax.text(
        (residual["first_op"] + residual["last_op"]) / 2,
        (residual["offset"] + residual["bytes"] / 2) / mib,
        "residual stream, alive through every layer",
        fontsize=8.5, color=SURFACE, ha="center", va="center",
    )  # fmt: skip
    ax.set_xlim(0, max(v["last_op"] for v in plan) + 1)
    ax.set_ylim(0, memory["arena_bytes"] / mib * 1.02)
    ax.set_xlabel("op in the forward pass", color=INK_2, fontsize=9)
    ax.set_ylabel("offset in the arena (MiB)", color=INK_2, fontsize=9)
    ax.grid(False)
    return fig


# The stacks of the production comparison (docs/production-comparison.md) by engine name, in the
# order their colors and markers are assigned. The colors extend SERIES with its next two slots.
STACKS = {
    "kvbound": "kvbound",
    "llama.cpp": "llama.cpp",
    "vllm": "vLLM",
    "tensorrt-llm": "TensorRT-LLM",
    "exllamav3": "ExLlamaV3",
}
STACK_COLORS = dict(zip(STACKS, (*SERIES, "#eda100", "#e87ba4"), strict=True))
STACK_MARKERS = dict(zip(STACKS, ("o", "s", "^", "D", "P"), strict=True))
COMPARISON_MODELS = {"tinyllama-1.1b": "TinyLlama 1.1B", "llama-3.2-1b": "Llama 3.2 1B"}
# 4-bit formats as the comparison names them; a format not listed keeps its record name.
FORMAT_LABELS = {
    "int4_g": "INT4, round to nearest",
    "int4_g.awq": "INT4 AWQ",
    "int4_g.gptq": "INT4 GPTQ",
    "int4_g.awq-gptq": "INT4 AWQ+GPTQ",
    "int4_g.rot": "INT4 rotated",
    "int4_g.rot-gptq": "INT4 rotated, GPTQ",
    "Q4_0_PURE": "Q4_0 pure",
    "W4A16_AWQ": "W4A16 AWQ",
    "W4A16_GPTQ": "W4A16 GPTQ",
    "EXL3_4.0": "EXL3 4.0",
    "EXL3_4.0_H4_E4": "EXL3 4.0, 4-bit head",
}
# kvbound's variants in the accuracy comparison: its INT8 and int4_g checkpoints, without the
# clip ablations (int4_g.mse, int4_g.awq-noclip).
KVBOUND_QUANTIZED = (
    "int8_pc", "int4_g", "int4_g.awq", "int4_g.gptq", "int4_g.awq-gptq", "int4_g.rot",
    "int4_g.rot-gptq",
)  # fmt: skip
FOUR_BIT_PREFIXES = ("int4_g", "Q4_", "W4A16", "EXL3_4")

ComparisonRecords = dict[str, dict[tuple[Any, ...], dict[str, Any]]]


def _is_fp16(weights: str) -> bool:
    return weights.lower() in {"fp16", "f16"}


def _kv(params: dict[str, Any]) -> str:
    return str(params.get("kv_dtype", params.get("kv", "fp16")))


def stack_records(device: str) -> ComparisonRecords:
    """The newest records of each stack in the production comparison, by measurement:
    "decode": (engine, model, weights, prompt_len) -> one request, FP16 KV, 128 new tokens;
    "serve": (engine, model) -> 96 `fixed` requests at once, FP16 weights and KV, a pool of
    8 full sequences; "perplexity": (engine, model, weights) -> the whole test split, FP16 KV.
    kvbound's decode runs replay CUDA graphs; llama.cpp's evaluate the prompt in one batch."""
    out: ComparisonRecords = {"decode": {}, "serve": {}, "perplexity": {}}
    for rec in record.load_records():
        engine, params = rec["engine"]["name"], rec["params"]
        if rec["fingerprint"]["device_config"] != device or engine not in STACKS:
            continue
        if not rec["model"] or rec["model"]["name"] not in COMPARISON_MODELS:
            continue
        if record.model_family(rec) != record.DEFAULT_FAMILY:
            continue
        model, weights = rec["model"]["name"], params.get("weights", "fp16")
        if rec["kind"] == "generate":
            prompt_len, server = params["prompt_len"], params.get("server")
            # llama-server's micro-batch; the other stacks take the prompt whole.
            ubatch = server.get("--ubatch-size", prompt_len) if isinstance(server, dict) else 0
            if (params.get("batch"), params.get("new_tokens")) != (1, 128):
                continue
            if not _is_fp16(_kv(params)) or int(ubatch or prompt_len) < prompt_len:
                continue
            if engine == "kvbound" and not params.get("graph"):
                continue
            out["decode"][engine, model, weights, prompt_len] = rec
        elif rec["kind"] == "serve":
            run = (params.get("workload"), params.get("n_requests"), params.get("arrival_rate", 0))
            if run != ("fixed", 96, 0):
                continue
            if not (_is_fp16(weights) and _is_fp16(_kv(params))):
                continue
            if engine == "kvbound":
                pool_seqs = params["kv_blocks"] * params["kv_block_size"] / params["max_ctx"]
                if serve_setup(rec) != "paged":
                    continue
            else:
                pool_seqs = params["pool_seqs"]
            if pool_seqs == 8:
                out["serve"][engine, model] = rec
        elif (
            rec["kind"] == "eval.perplexity"
            and params["max_windows"] == 0
            and _kv(params) in {"fp16", "f16", "none"}  # ExLlamaV3's records name no cache type
        ):
            out["perplexity"][engine, model, weights] = rec
    return out


def bits_per_weight(rec: dict[str, Any]) -> float:
    """Stored bits of every weight tensor over the parameter count, scales included. An outside
    stack's record carries the figure; for a kvbound checkpoint it follows from the format in
    the record's header and the model's shapes: every matrix in that format, including the
    embedding, and the norm gains in FP16."""
    params = rec["params"]
    if "bits_per_weight" in params:
        return float(params["bits_per_weight"])
    fmt = WeightFormat.from_metadata(params.get("weights_header"))
    table = load_spec(rec["model"]["name"]).arch
    arch = Arch.from_table(table)
    # A tied vocabulary matrix is stored once, as the embedding, unless the file is rotated.
    tied = table.get("tie_word_embeddings", False) and not fmt.rotation
    matrices = [
        (rows, cols, count)
        for name, rows, cols, count in arch.linears()
        if not (name == "lm_head" and tied)
    ]
    matrices.append((arch.vocab_size, arch.hidden_size, 1))
    norms = (2 * arch.n_layers + 1) * arch.hidden_size
    n_weights = sum(rows * cols * count for rows, cols, count in matrices) + norms
    bits = sum(rows * cols * count * fmt.bits_per_weight(rows, cols)
               for rows, cols, count in matrices)  # fmt: skip
    return (bits + 16 * norms) / n_weights


def _stack_axis(ax: Axes, notes: dict[str, str] | None = None) -> None:
    """Stack names under the slots, kvbound's in bold; notes go in a smaller line below."""
    ax.set_xticks(range(len(STACKS)), list(STACKS.values()), fontsize=7.5)
    for i, (tick, engine) in enumerate(zip(ax.get_xticklabels(), STACKS, strict=True)):
        if engine == "kvbound":
            tick.set_color(INK)
            tick.set_fontweight("bold")
        if notes and engine in notes:
            ax.annotate(notes[engine], (i, 0), (0, -17), xycoords=("data", "axes fraction"),
                        textcoords="offset points", ha="center", va="top", fontsize=7,
                        color=MUTED)  # fmt: skip
    ax.set_xlim(-0.6, len(STACKS) - 0.4)
    ax.margins(y=0.16)


def _stack_bars(ax: Axes, values: dict[str, float]) -> None:
    """One bar per stack in its color with its value above; a stack without a value gets
    "not measured" in its slot."""
    top = max(values.values())
    for i, engine in enumerate(STACKS):
        if engine not in values:
            ax.text(i, top * 0.04, "not\nmeasured", ha="center", va="bottom", fontsize=7.5,
                    color=MUTED, style="italic")  # fmt: skip
            continue
        value = values[engine]
        ax.bar(i, value, 0.66, color=STACK_COLORS[engine])
        ours = engine == "kvbound"
        ax.annotate(f"{value:,.0f}", (i, value), (0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK if ours else INK_2,
                    fontweight="bold" if ours else "normal")  # fmt: skip


def stacks(comparison: ComparisonRecords) -> Figure | None:
    """Tokens per second of each stack: one request at FP16 and at its fastest 4-bit format on
    TinyLlama, and 96 requests at once with FP16 weights on both models."""
    decode, serve = comparison["decode"], comparison["serve"]
    tiny = "tinyllama-1.1b"

    def rate(rec: dict[str, Any], metric: str) -> float:
        return float(rec["metrics"][metric]["p50"])

    fp16: dict[str, float] = {}
    fastest: dict[str, tuple[float, str]] = {}
    for (engine, model, weights, prompt_len), rec in decode.items():
        if (model, prompt_len) != (tiny, 128):
            continue
        tok_s = rate(rec, "decode_tok_s")
        if _is_fp16(weights):
            fp16[engine] = tok_s
        elif weights.startswith(FOUR_BIT_PREFIXES) and tok_s > fastest.get(engine, (0.0, ""))[0]:
            fastest[engine] = (tok_s, weights)
    served = {
        model: {e: rate(rec, "throughput_tok_s") for (e, m), rec in serve.items() if m == model}
        for model in COMPARISON_MODELS
    }
    if not all("kvbound" in v for v in (fp16, fastest, served[tiny], served["llama-3.2-1b"])):
        return None
    fig, grid = _panels(
        2, 2,
        "kvbound against production stacks on an RTX 3060 Ti",
        "Tokens per second, higher is better. Same token ids, requests and KV memory in every"
        " stack. Top: one request, decode\nafter a 128-token prompt, median of 10 runs. Bottom:"
        " 96 requests of 256 prompt and 128 output tokens queued at once.",
    )  # fmt: skip
    panels = (
        (grid[0][0], fp16, None, f"{COMPARISON_MODELS[tiny]}, FP16 weights, one request"),
        (grid[0][1], {e: v for e, (v, _) in fastest.items()},
         {e: FORMAT_LABELS.get(w, w) for e, (_, w) in fastest.items()},
         f"{COMPARISON_MODELS[tiny]}, fastest 4-bit file, one request"),
        *(
            (grid[1][i], served[model], None, f"{name}, 96 requests at once, FP16 weights and KV")
            for i, (model, name) in enumerate(COMPARISON_MODELS.items())
        ),
    )  # fmt: skip
    for ax, values, notes, title in panels:
        _stack_bars(ax, values)
        _stack_axis(ax, notes)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK_2)
    return fig


# Direct labels of the notable points: text, offset in points and alignment; the moves adjust a
# label for one model where it would collide.
ACCURACY_LABELS = {
    ("kvbound", "int4_g.awq-gptq"): ("kvbound\nAWQ+GPTQ", (-8, -4), "right"),
    ("kvbound", "int4_g"): ("round to\nnearest", (-8, 0), "right"),
    ("kvbound", "int4_g.rot"): ("rotated", (-8, 0), "right"),
    ("llama.cpp", "Q4_0"): ("Q4_0", (7, 0), "left"),
    ("llama.cpp", "Q4_K_M"): ("Q4_K_M", (7, 0), "left"),
    ("exllamav3", "EXL3_4.0"): ("EXL3 4.0", (7, 0), "left"),
    ("exllamav3", "EXL3_4.0_H4_E4"): ("EXL3 4.0,\n4-bit head", (-8, 0), "right"),
}
ACCURACY_LABEL_MOVES = {
    ("tinyllama-1.1b", "llama.cpp", "Q4_K_M"): ((0, -11), "center"),
}


def accuracy_per_bit(comparison: ComparisonRecords) -> Figure | None:
    """WikiText-2 perplexity against bits per weight for every quantized format, per model."""
    points: dict[str, list[tuple[str, str, float, float]]] = {m: [] for m in COMPARISON_MODELS}
    fp16_ppl: dict[str, float] = {}
    for (engine, model, weights), rec in comparison["perplexity"].items():
        ppl = float(rec["metrics"]["perplexity"])
        if engine == "kvbound" and weights == "fp16":
            fp16_ppl[model] = ppl
        elif not _is_fp16(weights) and (engine != "kvbound" or weights in KVBOUND_QUANTIZED):
            points[model].append((engine, weights, bits_per_weight(rec), ppl))
    if set(fp16_ppl) != set(COMPARISON_MODELS):
        return None
    fig, grid = _panels(
        1, 2,
        "Accuracy per bit: every quantized format of every stack",
        "WikiText-2 test perplexity, lower is better, against the stored bits of all weight"
        " tensors per weight, scales included.\nkvbound's 4-bit points are one format with"
        " different scales and rounding. Its calibration text, and vLLM's and\nTensorRT-LLM's,"
        " is WikiText-2's train split; ExLlamaV3 calibrates on its own text, llama.cpp's files"
        " on none.",
    )  # fmt: skip
    for ax, (model, name) in zip(grid[0], COMPARISON_MODELS.items(), strict=True):
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.axhline(fp16_ppl[model], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        for engine, stack in STACKS.items():
            mine = [p for p in points[model] if p[0] == engine]
            if not mine:
                continue
            ax.plot([p[2] for p in mine], [p[3] for p in mine], ls="none",
                    marker=STACK_MARKERS[engine], ms=8 if engine == "vllm" else 7,
                    color=STACK_COLORS[engine], mec=SURFACE, mew=1.2, label=stack,
                    zorder=3)  # fmt: skip
        for engine, weights, bits, ppl in points[model]:
            if (engine, weights) not in ACCURACY_LABELS:
                continue
            text, offset, ha = ACCURACY_LABELS[engine, weights]
            offset, ha = ACCURACY_LABEL_MOVES.get((model, engine, weights), (offset, ha))
            ours = engine == "kvbound"
            ax.annotate(text, (bits, ppl), offset, textcoords="offset points", ha=ha,
                        va="center", fontsize=7.5, color=INK if ours else INK_2,
                        fontweight="bold" if weights == "int4_g.awq-gptq" else "normal",
                        linespacing=1.1)  # fmt: skip
        ax.set_title(f"{name}  (FP16: {fp16_ppl[model]:.2f}, dashed line)", loc="left",
                     fontsize=9.5, color=INK_2)  # fmt: skip
        # Left of 4 bits is room for the labels of kvbound's 4.13-bit column.
        ax.set_xlim(2.9, 9.1)
        ax.set_xticks(range(4, 10))
        ax.set_xlabel("bits per weight", color=INK_2, fontsize=9)
        ax.margins(y=0.06)
    grid[0][0].set_ylabel("perplexity", color=INK_2, fontsize=9)
    grid[0][1].legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="name from configs/devices/")
    parser.add_argument("--model", required=True, help="model placed on the roofline")
    parser.add_argument("--comparison-only", action="store_true",
                        help="render only the production comparison's figures")  # fmt: skip
    args = parser.parse_args()

    out = REPORTS / args.device
    comparison = stack_records(args.device)
    for name, figure in (
        ("stacks", stacks(comparison)),
        ("accuracy_per_bit", accuracy_per_bit(comparison)),
    ):
        if figure is not None:
            print(_save(figure, out / f"{name}.png", top=0.85))
    if args.comparison_only:
        return

    records = hardware.latest_records(args.device)
    missing = {"bandwidth_sweep", "pointer_chase", "gemm_peak"} - set(records)
    if missing:
        raise SystemExit(f"no {sorted(missing)} records for {args.device}; run `make microbench`")
    spec = load_spec(args.model)
    if spec.family != record.DEFAULT_FAMILY:
        raise SystemExit(f"{args.model} is a {spec.family} model; these figures are for Llamas")
    arch = Arch.from_table(spec.arch)
    print(_save(roofline(args.device, records, arch, args.model), out / "roofline.png"))
    print(_save(bandwidth(args.device, records), out / "bandwidth.png"))
    print(_save(latency(records), out / "latency.png"))
    if "kv_layout" in records:
        print(_save(kv_layout(records), out / "kv_layout.png", top=0.85))
    serve = latest_serve_records(args.device, args.model)
    static = [n for (_, setup, n, rate) in serve if setup == "static" and rate == 0]
    if static:
        slots = max(static)
        print(_save(static_memory(serve, slots), out / "static_memory.png", top=0.85))
        first = next(
            rec
            for (_, setup, n, rate), rec in serve.items()
            if (setup, n, rate) == ("static", slots, 0)
        )
        print(_save(arena(first), out / "arena.png"))
    for name, figure in (
        ("serving", serving(serve)),
        ("paged_memory", paged_memory(serve)),
        ("serving_load", serving_load(serve)),
        ("precision", precision(precision_records(args.device, args.model), args.model)),
    ):
        if figure is not None:
            print(_save(figure, out / f"{name}.png", top=0.85))


if __name__ == "__main__":
    main()
