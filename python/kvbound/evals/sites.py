"""Error of the engine at one precision at every capture site, against the FP32 golden tensors.

Each golden prompt is fed to the engine in one step and every capture site is read back and
compared with the golden tensor, which the reference model computed in FP32 from the unquantized
checkpoint: relative RMS error, cosine similarity and largest absolute error. The worst value per
layer depth shows how the error of quantized weights or of a quantized KV cache grows through the
model. The logits are also compared as next-token distributions.

The record ends with a greedy continuation of each golden prompt as text, for spot checks.

Usage: python -m kvbound.evals.sites --model tinyllama-1.1b --weights int4_g --kv int8
"""

import argparse
from typing import Any

import torch
from transformers import AutoTokenizer

from kvbound import bench, golden, native, record, tolerance
from kvbound.models import load_spec
from kvbound.weight_format import VARIANTS


def evaluate(
    model: str, weights: str = "fp16", kv: str = "fp16", new_tokens: int = 48, preset: str = "cuda"
) -> dict[str, Any]:
    """Per-site errors and sample text of one model at one precision, as write_record arguments."""
    spec = load_spec(model)
    model_dir = bench.checkpoint_dir(model, weights)
    manifest, tensors = golden.load_for_model(model)
    n_layers = manifest["model"]["config"]["n_layers"]
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint_dir)
    options = {"kv_dtype": kv} if kv != "fp16" else {}
    engine = native.load(preset).Engine(str(model_dir), max_ctx=256, **options)

    # Worst value of each metric over the prompts, per site and per depth.
    sites: dict[str, dict[str, float]] = {}
    depths = [{"rel_rms": 0.0, "cosine": 1.0, "max_abs": 0.0} for _ in range(n_layers + 1)]
    mismatches, positions, kl = 0, 0, 0.0
    texts = []
    for prompt in manifest["prompts"]:
        ids = tensors[f"{prompt['id']}/input_ids"].tolist()
        for site, value in engine.feed(ids, capture=True).items():
            actual, expected = torch.from_numpy(value), tensors[f"{prompt['id']}/{site}"]
            measured = {
                "rel_rms": tolerance.rel_rms(actual, expected),
                "cosine": tolerance.cosine(actual, expected),
                "max_abs": (actual.double() - expected.double()).abs().max().item(),
            }
            depth = golden.site_depth(site, n_layers)
            for worst in (sites.setdefault(site, dict(measured)), depths[depth]):
                worst["rel_rms"] = max(worst["rel_rms"], measured["rel_rms"])
                worst["cosine"] = min(worst["cosine"], measured["cosine"])
                worst["max_abs"] = max(worst["max_abs"], measured["max_abs"])
            if site == "logits":
                mismatches += int((actual.argmax(-1) != expected.argmax(-1)).sum())
                positions += len(ids)
                kl = max(kl, tolerance.mean_kl(actual, expected))
        generated = engine.generate(ids, new_tokens)
        texts.append(
            {"id": prompt["id"], "prompt": prompt.get("text"), "tokens": generated,
             "text": tokenizer.decode(generated)}
        )  # fmt: skip

    return {
        "kind": "eval.sites",
        "engine": {"name": "kvbound", "preset": preset},
        "model": {"name": spec.name, "hf_repo": spec.hf_repo, "revision": spec.revision},
        "params": {
            "weights": weights,
            "kv": kv,
            "golden_sha256": manifest["golden_sha256"],
            "n_prompts": len(manifest["prompts"]),
            "new_tokens": new_tokens,
        },
        "metrics": {
            "logits_rel_rms": sites["logits"]["rel_rms"],
            "logits_cosine": sites["logits"]["cosine"],
            "top1_mismatches": mismatches,
            "positions": positions,
            "mean_kl_max": kl,  # of the prompt whose mean KL is largest
            "by_depth": depths,
        },
        "samples": {"sites": sites, "generated": texts},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=VARIANTS, default="fp16")
    parser.add_argument("--kv", choices=bench.KV_DTYPES, default="fp16")
    parser.add_argument("--new-tokens", type=int, default=48)
    parser.add_argument("--preset", default="cuda")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="print the result only")
    args = parser.parse_args()

    if not args.no_record:
        record.require_clean_tree(args.allow_dirty)
    result = evaluate(args.model, args.weights, args.kv, args.new_tokens, args.preset)
    m = result["metrics"]
    print(
        f"{args.model} weights={args.weights} kv={args.kv}:"
        f" logits rel-RMS {m['logits_rel_rms']:.4f}, cosine {m['logits_cosine']:.5f},"
        f" {m['top1_mismatches']} of {m['positions']} top-1 choices differ,"
        f" largest per-prompt mean KL {m['mean_kl_max']:.5f}"
    )
    for sample in result["samples"]["generated"]:
        print(f"  {sample['id']}: {sample['text']!r}")
    if not args.no_record:
        print(record.write_record(**result))


if __name__ == "__main__":
    main()
