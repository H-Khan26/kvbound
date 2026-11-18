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

Speech recognition (kind "asr", model family "whisper"). A run transcribes a set of clips, each
one audio window, and stops each transcription at its end-of-sequence token:

  RTF   real-time factor: seconds of processing per second of audio, as the sum over the
        clips of the processing time divided by the sum of their durations. Below 1 is faster
        than real time.
        rtf_model  from log-mel features on the host to token ids on the host: the encoder,
                   every decoder step, and the copies in and out.
        rtf_e2e    from the 16 kHz waveform to text: rtf_model plus the log-mel front end and
                   detokenization.
  encoder_ms   encoder latency: one encoder pass over one clip's features, from the features
               on the host to the cross-attention keys and values of every decoder layer on
               the device. Covers the conv stem, the encoder layers and those projections.
  first_token_ms       from a clip's features on the host to its first generated token: the
                       encoder pass plus the decoder step over the prompt.
  decode_ms_per_token  (time of last token - time of first token) / (generated tokens - 1).
  WER   word error rate: the word-level edit distance (substitutions + deletions + insertions)
        between each transcription and its reference, summed over the clips, divided by the
        number of reference words summed over the clips. It is a rate over the whole set, not
        a mean of per-clip rates. `wer` compares normalized text, with the normalization
        named in params; `wer_raw` compares the text as written.
  CER   the same over characters: `cer`.

Latencies are per clip and summarized like the benchmark iterations; the rates are single
numbers per run.
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
# Model families a record's "model" entry may name. Records written before the field existed
# have none and are Llama runs.
FAMILIES = ("llama", "whisper")
DEFAULT_FAMILY = "llama"


def model_family(record: dict[str, Any]) -> str:
    """The model family of a record: "llama" or "whisper"."""
    family: str = (record.get("model") or {}).get("family", DEFAULT_FAMILY)
    return family


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


# A run whose slowest SM clock sample under load is below this fraction of its fastest was
# throttled or down-clocked by the driver partway through.
CLOCK_STEADY_RATIO = 0.95


def _tags(fingerprint: dict[str, Any], nvml: dict[str, Any] | None) -> list[str]:
    """Flags that make a record unfit for published numbers."""
    tags = []
    if fingerprint["git"]["dirty"]:
        tags.append("dirty")
    os_class = fingerprint["software"]["os_class"]
    if os_class != "linux":
        tags.append(f"os={os_class}")  # clocks cannot be locked outside native Linux
    clock = (nvml or {}).get("sm_clock_mhz_busy")
    if clock and clock["min"] < CLOCK_STEADY_RATIO * clock["max"]:
        tags.append("clock_varied")
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

    kind     what was measured, e.g. "generate", "asr" or "microbench.bandwidth".
    engine   {"name": ..., "version": ...} of whatever produced the timings.
    params   the knobs of this run (batch size, lengths, dtype, seeds).
    metrics  summarized results; samples holds the raw per-iteration values behind them.
    model    {"name": ..., "hf_repo": ..., "revision": ...} plus "family", one of FAMILIES;
             a model entry without a family is a Llama.
    """
    if model is not None and model.get("family", DEFAULT_FAMILY) not in FAMILIES:
        raise ValueError(f"unknown model family {model['family']!r}; expected one of {FAMILIES}")
    fingerprint = fp.fingerprint(gpu_index)
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{kind.replace('.', '-')}-{secrets.token_hex(2)}"
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": kind,
        "created_utc": now.isoformat(timespec="seconds"),
        "tags": _tags(fingerprint, nvml),
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
