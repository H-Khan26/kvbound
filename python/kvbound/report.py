"""Turns result records into figures under reports/<device>/.

Hardware: roofline, bandwidth sweep, latency. Engine: KV layout benchmark, KV memory under static
batching, and the activation memory plan. Figures whose records do not exist yet are skipped.

Usage: python -m kvbound.report --device rtx3060ti --model tinyllama-1.1b
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
    device: str, model: str, weights: str = "fp16", kv: str = "fp16"
) -> dict[tuple[str, str, int, float], dict[str, Any]]:
    """Newest serve record per (workload, setup, slots, arrival rate) for a device config, model
    and precision. Only runs of each workload's own request count are kept."""
    latest = {}
    for rec in record.load_records("serve"):
        params = rec["params"]
        if rec["fingerprint"]["device_config"] != device or rec["model"]["name"] != model:
            continue
        if (params.get("weights", "fp16"), params.get("kv_dtype", "fp16")) != (weights, kv):
            continue
        if params["n_requests"] != workload.load(params["workload"]).n_requests:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="name from configs/devices/")
    parser.add_argument("--model", required=True, help="model placed on the roofline")
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


if __name__ == "__main__":
    main()
