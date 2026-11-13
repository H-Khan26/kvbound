"""Reads the attention kernel's hardware counters with Nsight Compute.

Profiles one attention launch of kvb-kv_layout for each KV layout, key-load width and shape,
and writes the counters to results/profiles/<run>/counters.csv with derived ratios in
summary.json: sectors per load request, cache hit rates and bytes read from DRAM.

Nsight Compute needs access to GPU performance counters. On Windows an administrator enables it
in NVIDIA Control Panel: Desktop > Enable Developer Settings, then Developer > Manage GPU
Performance Counters.

Usage: python -m kvbound.counters
"""

import argparse
import csv
import io
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kvbound import fingerprint
from kvbound.paths import binary
from kvbound.profile import PROFILES

# (layout, key loads): every variant kvb-kv_layout times.
VARIANTS = [
    ("head_major", "scalar"),
    ("head_major", "wide"),
    ("token_major", "scalar"),
    ("token_major", "wide"),
    ("dim_major", "scalar"),
]
# (shape, batch, tokens in the cache)
SHAPES = [("decode", 1, 4096), ("decode", 8, 4096), ("prefill", 1, 512)]

LOADS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld"
METRICS = {
    "time_ns": "gpu__time_duration.sum",
    "load_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "load_sectors": f"{LOADS}.sum",
    "l1_hit_sectors": f"{LOADS}_lookup_hit.sum",
    "l1_miss_sectors": f"{LOADS}_lookup_miss.sum",
    "l2_read_sectors": "lts__t_sectors_srcunit_tex_op_read.sum",
    "l2_miss_sectors": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram_bytes_read": "dram__bytes_read.sum",
    "stalled_on_memory_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "warps_active_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
}
SECTOR_BYTES = 32


def find_ncu() -> str:
    if found := shutil.which("ncu"):
        return found
    installs = sorted(Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute */"))
    for install in reversed(installs):
        script = install / "ncu.bat"
        if script.exists():
            return str(script)
    raise SystemExit("ncu not found; install Nsight Compute or add it to PATH")


def run(command: list[str]) -> str:
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0 or "ERR_NVGPUCTRPERM" in done.stdout + done.stderr:
        raise SystemExit(f"{Path(command[0]).name} failed:\n{done.stdout}\n{done.stderr}")
    return done.stdout


def profile_point(
    ncu: str, app: Path, work: Path, layout: str, loads: str, shape: str, batch: int, ctx: int
) -> dict[str, Any]:
    """Counters of the first attention launch of one point, in base units."""
    report = work / "point.ncu-rep"
    # Nsight Compute replays the launch once per group of counters, with the GPU caches flushed
    # before each pass, so cache counters describe a cold start.
    run([
        ncu, "--kernel-name", "regex:attention_kernel", "--launch-count", "1",
        "--metrics", ",".join(METRICS.values()), "--export", str(report), "--force-overwrite",
        str(app), "--layout", layout, "--key-loads", loads, "--shape", shape,
        "--batch", str(batch), "--ctx", str(ctx), "--reps", "1",
    ])  # fmt: skip
    table = run([
        ncu, "--import", str(report), "--csv", "--page", "raw", "--print-units", "base",
    ])  # fmt: skip
    rows = list(csv.reader(io.StringIO(table[table.index('"ID"') :])))
    if len(rows) != 3:
        raise SystemExit(f"expected one profiled launch, got {len(rows) - 2}")
    values = dict(zip(rows[0], rows[2], strict=True))
    point: dict[str, Any] = {
        "layout": layout,
        "key_loads": loads,
        "shape": shape,
        "batch": batch,
        "ctx": ctx,
        "grid": values["Grid Size"],
        "block": values["Block Size"],
    }
    for name, metric in METRICS.items():
        point[name] = float(values[metric].replace(",", ""))
    return point


def derive(point: dict[str, Any]) -> dict[str, Any]:
    """Ratios that describe how the launch used the memory hierarchy."""
    l1_lookups = point["l1_hit_sectors"] + point["l1_miss_sectors"]
    return {
        **{key: point[key] for key in ("layout", "key_loads", "shape", "batch", "ctx")},
        "time_us": point["time_ns"] / 1e3,
        "load_requests": point["load_requests"],
        "sectors_per_request": point["load_sectors"] / point["load_requests"],
        # Sectors asked of L1 per sector that had to come from DRAM: 1 means nothing was asked
        # for twice.
        "sectors_requested_per_dram_sector": point["load_sectors"]
        / (point["dram_bytes_read"] / SECTOR_BYTES),
        "l1_hit_pct": 100 * point["l1_hit_sectors"] / l1_lookups,
        "l2_hit_pct": 100 * (1 - point["l2_miss_sectors"] / point["l2_read_sectors"]),
        "dram_mb_read": point["dram_bytes_read"] / 1e6,
        "dram_gb_per_s": point["dram_bytes_read"] / point["time_ns"],
        "stalled_on_memory_pct": point["stalled_on_memory_pct"],
        "warps_active_pct": point["warps_active_pct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="cuda")
    args = parser.parse_args()

    ncu = find_ncu()
    app = binary("kv_layout", args.preset)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "counters"
        work.mkdir()
        points = []
        for layout, loads in VARIANTS:
            for shape, batch, ctx in SHAPES:
                print(f"profiling {layout} {loads} {shape} batch {batch}", flush=True)
                points.append(profile_point(ncu, app, Path(tmp), layout, loads, shape, batch, ctx))
        with (work / "counters.csv").open("w", encoding="utf-8", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=list(points[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(points)
        summary: dict[str, Any] = {
            "tool": run([ncu, "--version"]).strip().splitlines()[-1],
            "gpu": fingerprint.gpu_info()["name"],
            "git": fingerprint.git_state(),
            "metrics": METRICS,
            "points": [derive(point) for point in points],
        }
        with (work / "summary.json").open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(summary, indent=2) + "\n")
        out = PROFILES / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-attention-counters"
        shutil.copytree(work, out)

    print(
        f"\n{'layout':<12}{'loads':<8}{'shape':<9}{'batch':>5}{'requests':>11}"
        f"{'sect/req':>10}{'L1 hit':>8}{'L2 hit':>8}{'DRAM MB':>9}{'stalled':>9}"
    )
    for p in summary["points"]:
        print(
            f"{p['layout']:<12}{p['key_loads']:<8}{p['shape']:<9}{p['batch']:>5}"
            f"{p['load_requests']:>11,.0f}{p['sectors_per_request']:>10.1f}"
            f"{p['l1_hit_pct']:>7.0f}%{p['l2_hit_pct']:>7.0f}%{p['dram_mb_read']:>9.1f}"
            f"{p['stalled_on_memory_pct']:>8.0f}%"
        )
    print(out)


if __name__ == "__main__":
    main()
