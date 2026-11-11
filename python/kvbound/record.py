"""Result records: the one place that defines the metrics and writes results/raw/.

Every benchmark, whatever produced the timings (Hugging Face, a kvbound binary, an external
stack), goes through write_record(). One JSON file per run, named by device and run id, so
machines can push results without merge conflicts and any two files can be compared.

Metric definitions (all times are wall-clock seconds on the host, GPU work synchronized):

  TTFT  time to first token: from submitting a prompt already on the device to the first
        generated token being available. Covers prefill plus one sampling step.
  TPOT  time per output token: (time of last token - time of first token) / (n_new_tokens - 1).
        Excludes prefill; this is the steady decode step time for the whole batch.
  decode_tok_s   batch_size / TPOT. Tokens per second across the batch during decode.
  e2e_tok_s      batch_size * n_new_tokens / total request time, prefill included.
  peak memory    largest allocator high-water mark during the run, as reported by the engine.

Benchmarks run with a fixed output length (EOS ignored), warmup iterations that are discarded,
and N measured iterations summarized as p50 / p90 / mean / stdev.
"""

import json
import secrets
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kvbound import fingerprint as fp
from kvbound.paths import RESULTS_RAW

SCHEMA_VERSION = 1


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending sequence, q in [0, 100]."""
    rank = (len(sorted_values) - 1) * q / 100
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


def summarize(samples: Sequence[float]) -> dict[str, float]:
    """n, mean, stdev, min, p50, p90, max of repeated measurements of one quantity."""
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "max": ordered[-1],
    }


def require_clean_tree(allow_dirty: bool) -> None:
    """Refuses to benchmark uncommitted code: a result must be reproducible from its git sha."""
    if fp.git_state()["dirty"] and not allow_dirty:
        raise SystemExit(
            "refusing to benchmark a dirty tree: commit first, or pass --allow-dirty "
            "(the record is then tagged dirty)"
        )


def _tags(fingerprint: dict[str, Any]) -> list[str]:
    """Flags that make a record unfit for published numbers."""
    tags = []
    if fingerprint["git"]["dirty"]:
        tags.append("dirty")
    os_class = fingerprint["software"]["os_class"]
    if os_class != "linux":
        tags.append(f"os={os_class}")  # clocks cannot be locked outside native Linux
    return tags


def write_record(
    kind: str,
    engine: dict[str, Any],
    params: dict[str, Any],
    metrics: dict[str, Any],
    *,
    model: dict[str, Any] | None = None,
    samples: dict[str, Any] | None = None,
    nvml: dict[str, Any] | None = None,
    gpu_index: int = 0,
) -> Path:
    """Writes one run to results/raw/<device_id>/<run_id>.json and returns the path.

    kind     what was measured, e.g. "generate" or "microbench.bandwidth".
    engine   {"name": ..., "version": ...} of whatever produced the timings.
    params   the knobs of this run (batch size, lengths, dtype, seeds).
    metrics  summarized results; samples holds the raw per-iteration values behind them.
    """
    fingerprint = fp.fingerprint(gpu_index)
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{kind.replace('.', '-')}-{secrets.token_hex(2)}"
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": kind,
        "created_utc": now.isoformat(timespec="seconds"),
        "tags": _tags(fingerprint),
        "engine": engine,
        "model": model,
        "params": params,
        "metrics": metrics,
        "samples": samples or {},
        "nvml": nvml,
        "fingerprint": fingerprint,
    }
    path = RESULTS_RAW / fingerprint["device_id"] / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, indent=2) + "\n")
    return path


def load_records(kind: str | None = None, device_id: str | None = None) -> list[dict[str, Any]]:
    """All records under results/raw/, oldest first, optionally filtered."""
    records = []
    for path in sorted(RESULTS_RAW.glob("*/*.json"), key=lambda p: p.name):
        record = json.loads(path.read_text(encoding="utf-8"))
        if kind is not None and record["kind"] != kind:
            continue
        if device_id is not None and record["fingerprint"]["device_id"] != device_id:
            continue
        records.append(record)
    return records
