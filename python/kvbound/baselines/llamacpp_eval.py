"""WikiText-2 perplexity of a llama.cpp GGUF file under kvbound's protocol.

The token stream, the windows and the arithmetic are kvbound.evals.perplexity's: the joined
test split tokenized once by the model's Hugging Face tokenizer, non-overlapping windows of
`window` tokens, every position after the first of a window scored, the NLL summed in float64.
Only the logits come from llama.cpp: each window goes through the pinned release's llama.dll
(kvbound.baselines.llamacpp_lib) as one batch from position 0 with the cache cleared, all
layers on the GPU, every position's logits returned.

llama-perplexity scores differently: it puts BOS in place of the first token of every window and
scores only the second half of each, so its numbers are not comparable with kvbound's and are not
used here. The record also holds llama.cpp's own tokenization of the same text, compared with
the stream that was scored: the GGUF must map the same text to the same ids for the fed ids to
mean what they mean to the Hugging Face tokenizer.

Usage: python -m kvbound.baselines.llamacpp_eval --model tinyllama-1.1b --weights Q4_0
"""

import argparse
from typing import Any

import torch

from kvbound import record
from kvbound.baselines import llamacpp
from kvbound.baselines.llamacpp_lib import FLASH_ATTN, GGML_TYPES, Llama
from kvbound.evals import perplexity as ppl
from kvbound.models import load_spec, model_entry


def compare_tokens(reference: list[int], other: list[int]) -> dict[str, Any]:
    """How a second tokenization of the same text differs from the reference one."""
    first = next(
        (i for i, (a, b) in enumerate(zip(reference, other, strict=False)) if a != b),
        None if len(reference) == len(other) else min(len(reference), len(other)),
    )
    return {
        "n_tokens": len(other),
        "same_count": len(other) == len(reference),
        "identical": reference == other,
        "first_difference": first,
    }


def evaluate(
    model: str,
    weights: str = "F16",
    window: int = 1024,
    max_windows: int = 0,
    flash_attn: str = "on",
    kv: str = "f16",
) -> dict[str, Any]:
    """Perplexity of one GGUF file; returns the arguments of record.write_record."""
    spec = load_spec(model)
    gguf = llamacpp.require_gguf(model, weights)
    text = ppl.load_text(ppl.fetch_dataset())
    ids = ppl.tokenize(text, spec.checkpoint_dir)
    spans = ppl.split_windows(len(ids), window, max_windows)
    llama = Llama(gguf, window, flash_attn=flash_attn, kv=kv)
    try:
        tokens = compare_tokens(ids, llama.tokenize(text))
        sums, counts, seconds = ppl.score(
            ids, spans, lambda window_ids: torch.from_numpy(llama.logits(window_ids))
        )
        settings = llama.settings
    finally:
        llama.close()
    return {
        "kind": "eval.perplexity",
        "engine": llamacpp.engine_entry("llama.dll"),
        "model": model_entry(spec),
        "params": {
            **llamacpp.gguf_params(model, weights),
            "kv": kv,
            "protocol": "kvbound",
            "llama_cpp": settings,
            "dataset": ppl.DATASET,
            "dataset_repo": ppl.DATASET_REPO,
            "dataset_revision": ppl.DATASET_REVISION,
            "dataset_sha256": ppl.DATASET_SHA256,
            "split": ppl.SPLIT,
            "window": window,
            "max_windows": max_windows,
            "n_tokens": len(ids),
            "n_windows": len(spans),
            "n_predicted": sum(counts),
            "llama_cpp_tokenization": tokens,
        },
        "metrics": {
            **ppl.summarize(sums, counts),
            "seconds": seconds,
            "tokens_per_s": sum(end - start for start, end in spans) / seconds,
        },
        "samples": {"window_nll_sum": sums, "window_predicted": counts},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=llamacpp.FORMATS, default="F16")
    parser.add_argument("--kv", choices=sorted(GGML_TYPES), default="f16", help="KV cache type")
    parser.add_argument("--flash-attn", choices=sorted(FLASH_ATTN), default="on")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=0, help="0: every window")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="print the result only")
    args = parser.parse_args()

    if not args.no_record:
        record.require_clean_tree(args.allow_dirty)
    result = evaluate(
        args.model, args.weights, args.window, args.max_windows, args.flash_attn, args.kv
    )
    m, p = result["metrics"], result["params"]
    tok = p["llama_cpp_tokenization"]
    print(
        f"{args.model} {args.weights} ({p['bits_per_weight']:.3f} bits per weight): "
        f"perplexity {m['perplexity']:.4f} (NLL {m['nll_mean']:.4f} +- {m['nll_stderr']:.4f}) "
        f"over {p['n_windows']} windows of {args.window}, {p['n_predicted']} predicted tokens, "
        f"{m['seconds']:.1f} s | llama.cpp tokenizes the text to {tok['n_tokens']} tokens "
        f"against {p['n_tokens']}, identical: {tok['identical']}"
    )
    if not args.no_record:
        print(record.write_record(**result))


if __name__ == "__main__":
    main()
