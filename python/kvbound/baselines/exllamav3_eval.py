"""WikiText-2 perplexity of an ExLlamaV3 model under kvbound's protocol.

The token stream, the windows and the arithmetic are kvbound.evals.perplexity's: the joined test
split tokenized once by the model's Hugging Face tokenizer, non-overlapping windows of `window`
tokens, every position after the first of a window scored, the NLL summed in float64. Only the
logits come from ExLlamaV3: each window goes through Model.forward from position 0 with no
cache, all layers on the GPU, in a worker process in the pinned environment
(kvbound.baselines.exllamav3_lib), which applies kvbound's rule to the logits and returns the
per-window sums; kvbound.evals.perplexity.summarize turns them into the perplexity.

ExLlamaV3's own eval/ppl.py scores differently (rows of 2048 tokens by default, its own
tokenization), so its numbers are not used here. The record holds ExLlamaV3's tokenization of
the same text, compared with the stream that was scored.

Usage: python -m kvbound.baselines.exllamav3_eval --model tinyllama-1.1b --weights EXL3_4.0
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from kvbound import record
from kvbound.baselines import exllamav3
from kvbound.baselines.llamacpp_eval import compare_tokens
from kvbound.evals import perplexity as ppl
from kvbound.models import load_spec, model_entry


def evaluate(
    model: str, weights: str = "EXL3_4.0", window: int = 1024, max_windows: int = 0
) -> dict[str, Any]:
    """Perplexity of one format; returns the arguments of record.write_record."""
    spec = load_spec(model)
    folder = exllamav3.require_model(model, weights)
    text = ppl.load_text(ppl.fetch_dataset())
    ids = ppl.tokenize(text, spec.checkpoint_dir)
    spans = ppl.split_windows(len(ids), window, max_windows)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        np.save(work / "ids.npy", np.array(ids, dtype=np.int32))
        np.save(work / "spans.npy", np.array(spans, dtype=np.int64))
        (work / "text.txt").write_text(text, encoding="utf-8")
        subprocess.run(
            exllamav3.worker_command(
                "ppl", str(work), "--model-dir", str(folder), "--window", str(window)
            ),
            check=True, cwd=exllamav3.worker_cwd(),
        )  # fmt: skip
        out = json.loads((work / "result.json").read_text(encoding="utf-8"))
        tokens = compare_tokens(ids, np.load(work / "tokens.npy").tolist())
    sums, counts, seconds = out["sums"], out["counts"], out["seconds"]
    return {
        "kind": "eval.perplexity",
        "engine": exllamav3.engine_entry("Model.forward"),
        "model": model_entry(spec),
        "params": {
            **exllamav3.weights_params(model, weights),
            "kv": "none",  # each window is one forward pass with no cache
            "protocol": "kvbound",
            "exllamav3": out["settings"],
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
            "exllamav3_tokenization": tokens,
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
    parser.add_argument("--weights", choices=exllamav3.FORMATS, default="EXL3_4.0")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=0, help="0: every window")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="print the result only")
    args = parser.parse_args()

    if not args.no_record:
        record.require_clean_tree(args.allow_dirty)
    result = evaluate(args.model, args.weights, args.window, args.max_windows)
    m, p = result["metrics"], result["params"]
    tok = p["exllamav3_tokenization"]
    print(
        f"{args.model} {args.weights} ({p['bits_per_weight']:.3f} bits per weight): "
        f"perplexity {m['perplexity']:.4f} (NLL {m['nll_mean']:.4f} +- {m['nll_stderr']:.4f}) "
        f"over {p['n_windows']} windows of {args.window}, {p['n_predicted']} predicted tokens, "
        f"{m['seconds']:.1f} s | ExLlamaV3 tokenizes the text to {tok['n_tokens']} tokens "
        f"against {p['n_tokens']}, identical: {tok['identical']}"
    )
    if not args.no_record:
        print(record.write_record(**result))


if __name__ == "__main__":
    main()
