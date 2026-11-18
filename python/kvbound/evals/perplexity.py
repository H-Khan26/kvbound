"""WikiText-2 perplexity of the engine or the reference model at one weight and KV precision.

The text is the raw test split joined the standard way ("\\n\\n".join over its rows, as in the
Hugging Face perplexity guide) and tokenized once with the model's tokenizer, which puts one BOS
at the start. The token stream is cut into non-overlapping windows of `window` tokens (a last
window shorter than 2 tokens is dropped). Within a window, position i predicts token i + 1, so
the first token of each window is context only. Perplexity is exp(total NLL / predicted tokens),
with the NLL summed in float64.

The dataset file is pinned to a commit of its Hugging Face repository and checked against its
sha256 before use.

Usage: python -m kvbound.evals.perplexity --model tinyllama-1.1b --weights int4_g --kv fp16
"""

import argparse
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor
from transformers import AutoTokenizer

from kvbound import native, record
from kvbound.models import load_spec, sha256_file
from kvbound.paths import DATA
from kvbound.ref import load_llama
from kvbound.weight_format import VARIANTS, WeightFormat

DATASET = "wikitext-2-raw-v1"
DATASET_REPO = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_FILE = f"{DATASET}/test-00000-of-00001.parquet"
DATASET_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
SPLIT = "test"
# The train split of the same revision: calibration text for kvbound.awq, never scored.
TRAIN_FILE = f"{DATASET}/train-00000-of-00001.parquet"
TRAIN_SHA256 = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
DATASETS_DIR = DATA / "datasets"

KV_DTYPES = ("fp16", "int8")
BACKENDS = ("kvbound", "ref")

# Logits [n, vocab] of a window of token ids fed from position 0.
LogitsFn = Callable[[list[int]], Tensor]


def fetch_dataset(file: str = DATASET_FILE, sha256: str = DATASET_SHA256) -> Path:
    """data/datasets/<file> (by default the test parquet), downloaded once and hash-checked."""
    path = DATASETS_DIR / file
    if not path.exists() or sha256_file(path) != sha256:
        hf_hub_download(
            DATASET_REPO,
            file,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=DATASETS_DIR,
            force_download=path.exists(),
        )
    digest = sha256_file(path)
    if digest != sha256:
        raise ValueError(f"{path} has sha256 {digest}, expected {sha256}")
    return path


def load_text(path: Path) -> str:
    rows = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return "\n\n".join(rows)


def tokenize(text: str, tokenizer_dir: Path) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    # verbose=False silences the warning about exceeding the model's context length.
    return list(tokenizer(text, verbose=False).input_ids)


def split_windows(n_tokens: int, window: int, max_windows: int = 0) -> list[tuple[int, int]]:
    """[start, end) of each non-overlapping window; max_windows > 0 keeps only the first ones."""
    if window < 2:
        raise ValueError("a window needs at least 2 tokens to predict one")
    spans = [(start, min(start + window, n_tokens)) for start in range(0, n_tokens, window)]
    spans = [(start, end) for start, end in spans if end - start >= 2]
    return spans[:max_windows] if max_windows > 0 else spans


def window_nll(logits: Tensor, ids: Tensor) -> float:
    """Summed NLL in nats of ids[1:] under logits[:-1], logits [n, vocab] and ids [n]."""
    logits = logits[:-1].float()
    targets = ids[1:].to(logits.device, torch.int64)
    nll = torch.logsumexp(logits, dim=-1) - logits.gather(-1, targets[:, None])[:, 0]
    return nll.double().sum().item()


def score(
    ids: list[int], spans: list[tuple[int, int]], logits_fn: LogitsFn
) -> tuple[list[float], list[int], float]:
    """Per-window NLL sums and predicted-token counts, and the seconds the loop took."""
    sums, counts = [], []
    started = time.perf_counter()
    for start, end in spans:
        window_ids = ids[start:end]
        # The logits are not bound to a name, so one window's are freed before the next forward.
        sums.append(window_nll(logits_fn(window_ids), torch.tensor(window_ids)))
        counts.append(end - start - 1)
    return sums, counts, time.perf_counter() - started


def summarize(sums: list[float], counts: list[int]) -> dict[str, float]:
    """Perplexity, mean NLL per token, and the standard error of the per-window mean NLL."""
    total, n = math.fsum(sums), sum(counts)
    means = [s / c for s, c in zip(sums, counts, strict=True)]
    stderr = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0
    return {"perplexity": math.exp(total / n), "nll_mean": total / n, "nll_stderr": stderr}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ref_logits_fn(model_dir: Path) -> LogitsFn:
    """The FP32 reference model; quantized checkpoints are dequantized on load."""
    model = load_llama(model_dir).to(device())

    def logits_fn(window_ids: list[int]) -> Tensor:
        with torch.no_grad():
            return model(torch.tensor([window_ids], device=device()))[0]

    return logits_fn


def engine_logits_fn(
    model_dir: Path, window: int, preset: str, engine_options: dict[str, Any]
) -> LogitsFn:
    """The kvbound engine; each window is one step from position 0 of sequence 0."""
    kvb = native.load(preset)
    engine = kvb.Engine(str(model_dir), max_ctx=window, max_step_tokens=window, **engine_options)

    def logits_fn(window_ids: list[int]) -> Tensor:
        logits = engine.feed(window_ids, start=0, seq=0)["logits"]
        return torch.from_numpy(logits).to(device())

    return logits_fn


def weights_header(model_dir: Path) -> tuple[WeightFormat, dict[str, str]]:
    """The weight format of a checkpoint and its kvbound.* header keys (source hash included)."""
    path = model_dir / "model.safetensors"
    if not path.exists():
        raise SystemExit(
            f"{path} not found; run `make fetch`, then `python -m kvbound.quantize` for a variant"
        )
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
    header = {key: value for key, value in metadata.items() if key.startswith("kvbound.")}
    return WeightFormat.from_metadata(metadata), header


def evaluate(
    model: str,
    weights: str = "fp16",
    kv: str = "fp16",
    backend: str = "kvbound",
    window: int = 1024,
    max_windows: int = 0,
    preset: str = "cuda",
) -> dict[str, Any]:
    """Perplexity of one model at one precision; returns the arguments of record.write_record.

    weights names a variant directory (see ModelSpec.variant_dir); the tokenizer always comes from
    the fetched checkpoint. tokens_per_s counts every token fed, over the evaluation loop only.
    """
    if weights not in VARIANTS:
        raise ValueError(f"unknown weights {weights!r}; known: {VARIANTS}")
    if kv not in KV_DTYPES:
        raise ValueError(f"unknown KV dtype {kv!r}; known: {KV_DTYPES}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; known: {BACKENDS}")
    if backend == "ref" and kv != "fp16":
        raise ValueError("the reference model has no quantized KV cache; --kv is for the engine")
    spec = load_spec(model)
    model_dir = spec.variant_dir(weights)
    fmt, header = weights_header(model_dir)
    if fmt.variant != weights:
        raise ValueError(f"{model_dir} holds {fmt.variant} weights, not {weights}")

    ids = tokenize(load_text(fetch_dataset()), spec.checkpoint_dir)
    spans = split_windows(len(ids), window, max_windows)
    if backend == "ref":
        engine: dict[str, Any] = {"name": "ref", "version": torch.__version__}
        logits_fn = ref_logits_fn(model_dir)
    else:
        engine = {"name": "kvbound", "preset": preset}
        options = {"kv_dtype": kv} if kv != "fp16" else {}
        logits_fn = engine_logits_fn(model_dir, window, preset, options)
    sums, counts, seconds = score(ids, spans, logits_fn)

    return {
        "kind": "eval.perplexity",
        "engine": engine,
        "model": {"name": spec.name, "hf_repo": spec.hf_repo, "revision": spec.revision},
        "params": {
            "weights": weights,
            "weights_header": header,
            "kv": kv,
            "dataset": DATASET,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256": DATASET_SHA256,
            "split": SPLIT,
            "window": window,
            "max_windows": max_windows,
            "n_tokens": len(ids),
            "n_windows": len(spans),
            "n_predicted": sum(counts),
        },
        "metrics": {
            **summarize(sums, counts),
            "seconds": seconds,
            "tokens_per_s": sum(end - start for start, end in spans) / seconds,
        },
        "samples": {"window_nll_sum": sums, "window_predicted": counts},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=VARIANTS, default="fp16")
    parser.add_argument("--kv", choices=KV_DTYPES, default="fp16")
    parser.add_argument("--backend", choices=BACKENDS, default="kvbound")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=0, help="0: every window")
    parser.add_argument("--preset", default="cuda")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="print the result only")
    args = parser.parse_args()
    if args.backend == "ref" and args.kv != "fp16":
        parser.error("--kv applies to the kvbound backend only")

    if not args.no_record:
        record.require_clean_tree(args.allow_dirty)
    result = evaluate(
        args.model,
        args.weights,
        args.kv,
        args.backend,
        args.window,
        args.max_windows,
        args.preset,
    )
    m, p = result["metrics"], result["params"]
    print(
        f"{args.model} weights={args.weights} kv={args.kv} backend={args.backend}: "
        f"perplexity {m['perplexity']:.4f} (NLL {m['nll_mean']:.4f} +- {m['nll_stderr']:.4f}) "
        f"over {p['n_windows']} windows of {args.window}, {p['n_predicted']} predicted tokens, "
        f"{m['seconds']:.1f} s"
    )
    if not args.no_record:
        print(record.write_record(**result))


if __name__ == "__main__":
    main()
