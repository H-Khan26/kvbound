"""Hugging Face Transformers baseline: the number every later phase is compared against.

Runs greedy generation of a fixed number of tokens from seeded random prompt ids (token ids in,
token ids out; no tokenizer in the timed path) and records TTFT / TPOT / throughput / peak memory
as defined in kvbound.record.

Usage: python -m kvbound.baselines.hf --model tinyllama-1.1b --batch 1 --prompt-len 128
"""

import argparse
import time
from typing import Any

import torch
import transformers
from transformers import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

from kvbound import record
from kvbound.models import load_spec
from kvbound.nvml_sampler import NvmlSampler

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class TokenTimer(BaseStreamer):
    """Timestamps each generated token. generate() calls put() with the prompt first."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self._seen_prompt = False

    def put(self, value: torch.Tensor) -> None:
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())

    def end(self) -> None:
        pass


def run_once(model: Any, input_ids: torch.Tensor, new_tokens: int) -> dict[str, float]:
    """One timed request; returns the per-request metrics."""
    batch = input_ids.shape[0]
    timer = TokenTimer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,  # suppresses EOS so every run decodes the same length
        streamer=timer,
        pad_token_id=0,
    )
    torch.cuda.synchronize()
    if len(timer.times) != new_tokens:
        raise RuntimeError(f"expected {new_tokens} tokens, timed {len(timer.times)}")

    first, last = timer.times[0], timer.times[-1]
    tpot = (last - first) / (new_tokens - 1)
    return {
        "ttft_s": first - start,
        "tpot_s": tpot,
        "decode_tok_s": batch / tpot,
        "e2e_tok_s": batch * new_tokens / (last - start),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.model)
    # Any: the transformers type stubs reject valid generate() and to() calls.
    model: Any = LlamaForCausalLM.from_pretrained(spec.checkpoint_dir, dtype=DTYPES[args.dtype])
    model = model.to("cuda").eval()
    weights_mib = torch.cuda.memory_allocated() / 2**20

    gen = torch.Generator().manual_seed(args.seed)
    input_ids = torch.randint(
        model.config.vocab_size, (args.batch, args.prompt_len), generator=gen
    ).to("cuda")

    for _ in range(args.warmup):
        run_once(model, input_ids, args.new_tokens)
    with NvmlSampler() as sampler:
        runs = [run_once(model, input_ids, args.new_tokens) for _ in range(args.iters)]

    samples = {name: [run[name] for run in runs] for name in runs[0]}
    metrics: dict[str, Any] = {name: record.summarize(values) for name, values in samples.items()}
    metrics["weights_mib"] = weights_mib
    return {
        "kind": "generate",
        "engine": {"name": "hf-transformers", "version": transformers.__version__},
        "model": {"name": spec.name, "hf_repo": spec.hf_repo, "revision": spec.revision},
        "params": {
            "dtype": args.dtype,
            "batch": args.batch,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "sampling": "greedy",
        },
        "metrics": metrics,
        "samples": samples,
        "nvml": sampler.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error("--new-tokens must be at least 2 to measure TPOT")

    record.require_clean_tree(args.allow_dirty)
    result = benchmark(args)
    path = record.write_record(**result)
    m = result["metrics"]
    print(
        f"TTFT p50 {m['ttft_s']['p50'] * 1e3:.1f} ms | TPOT p50 {m['tpot_s']['p50'] * 1e3:.2f} ms"
        f" | decode {m['decode_tok_s']['p50']:.1f} tok/s"
        f" | peak {m['peak_allocated_mib']['max']:.0f} MiB\n{path}"
    )


if __name__ == "__main__":
    main()
