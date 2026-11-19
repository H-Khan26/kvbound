"""Decode speed and time to first token of ExLlamaV3, measured the way kvbound.bench measures.

Route: ExLlamaV3's Generator in a worker process in the pinned environment
(kvbound.baselines.exllamav3_lib), one job at a time, all layers on one GPU, an FP16 paged
cache of prompt + new tokens + 1 positions rounded up to whole pages (see cache_tokens). Each job
is the prompt as token ids, greedy (ArgmaxSampler), no stop condition, so exactly new_tokens
tokens; the page table is reset before each job, so no prompt page is reused. The prompt is
kvbound.bench.random_prompt, the same seeded ids kvbound and the other baselines process.

Metrics follow kvbound.record and are computed by kvbound.bench.request_metrics from the times
iterate() hands each token to the caller, counted from just before the job is enqueued: TTFT,
TPOT, decode tokens/s, end-to-end tokens/s and the median step. They include the generator's
Python work per step, as a server's client times include its transport. The generator's own
timings of each job (time_prefill, time_generate) are kept beside them. Tokens that complete a
UTF-8 character are held back and arrive together; multi_token_events counts how often.

One worker per run: it loads the model, runs the warmup and measured jobs, and exits. GPU memory
is NVML's device memory in use, sampled once a second from before the worker starts to after it
ends: gpu_memory_mib is the largest sample less the smallest, as for the llama.cpp baseline.

Usage: python -m kvbound.baselines.exllamav3_bench --model tinyllama-1.1b --weights EXL3_4.0 \
           --prompt-len 128
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kvbound import record
from kvbound.baselines import exllamav3
from kvbound.bench import random_prompt, request_metrics
from kvbound.models import load_spec, model_entry
from kvbound.nvml_sampler import NvmlSampler
from kvbound.perfmodel.model import Arch

# Positions per page of ExLlamaV3's paged cache; a cache holds whole pages.
PAGE_SIZE = 256


def cache_tokens(prompt_len: int, new_tokens: int) -> int:
    """Positions of a cache that holds one job, in whole pages: the generator caps a job's new
    tokens at the cache's positions less the prompt less one."""
    return -(-(prompt_len + new_tokens + 1) // PAGE_SIZE) * PAGE_SIZE


def check_runs(runs: list[dict[str, Any]], prompt_len: int, new_tokens: int) -> None:
    """Raises unless every job decoded new_tokens tokens from the whole prompt, none cached."""
    for run in runs:
        final = run["final"]
        if len(run["times"]) != new_tokens or final["new_tokens"] != new_tokens:
            raise RuntimeError(f"expected {new_tokens} tokens, received {len(run['times'])}")
        if final["prompt_tokens"] != prompt_len:
            raise RuntimeError(f"the job held {final['prompt_tokens']} prompt tokens")
        if final["cached_pages"] or final["cached_tokens"]:
            raise RuntimeError("the job reused cached prompt pages")


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    folder = exllamav3.require_model(args.model, args.weights)
    weights = exllamav3.weights_params(args.model, args.weights)
    prompt = random_prompt(arch.vocab_size, args.prompt_len, args.seed)
    request = {
        "prompt": prompt,
        "new_tokens": args.new_tokens,
        "warmup": args.warmup,
        "iters": args.iters,
        "cache_tokens": cache_tokens(args.prompt_len, args.new_tokens),
        "max_chunk_size": args.chunk_size,
    }
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "request.json").write_text(json.dumps(request), encoding="utf-8")
        command = exllamav3.worker_command("generate", str(work), "--model-dir", str(folder))
        with NvmlSampler() as sampler:
            subprocess.run(command, check=True, cwd=exllamav3.worker_cwd())
        out = json.loads((work / "result.json").read_text(encoding="utf-8"))
    runs = out["runs"]
    check_runs(runs, args.prompt_len, args.new_tokens)

    per_run = [request_metrics(run["times"]) for run in runs]
    samples: dict[str, Any] = {name: [m[name] for m in per_run] for name in per_run[0]}
    metrics: dict[str, Any] = {name: record.summarize(values) for name, values in samples.items()}
    own = {
        "time_prefill_s": [run["final"]["time_prefill"] for run in runs],
        "time_generate_s": [run["final"]["time_generate"] for run in runs],
    }
    metrics["exllamav3"] = {name: record.summarize(v) for name, v in own.items()}
    metrics["multi_token_events"] = sum(run["multi_token_events"] for run in runs)
    metrics["held_tokens"] = sum(run["held_tokens"] for run in runs)
    used = sampler.samples["mem_used_mib"]
    metrics["gpu_memory_mib"] = max(used) - min(used)
    metrics["torch_memory_mib"] = out["settings"]["torch_memory"]
    return {
        "kind": "generate",
        "engine": exllamav3.engine_entry("Generator"),
        "model": model_entry(spec),
        "params": {
            "batch": 1,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "max_ctx": request["cache_tokens"],
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "sampling": "greedy",
            "prompt": "kvbound.bench.random_prompt",
            **weights,
            "kv": "fp16",
            "exllamav3": out["settings"],
        },
        "metrics": metrics,
        "samples": {
            **samples,
            "exllamav3": own,
            "tokens": runs[0]["tokens"],
            "same_tokens_every_iteration": all(r["tokens"] == runs[0]["tokens"] for r in runs),
        },
        "nvml": sampler.summary(),
    }


def summary_line(result: dict[str, Any]) -> str:
    m = result["metrics"]
    own = m["exllamav3"]
    return (
        f"{result['model']['name']} {result['params']['weights']}"
        f" prompt {result['params']['prompt_len']}:"
        f" TTFT p50 {m['ttft_s']['p50'] * 1e3:.1f} ms"
        f" (generator prefill {own['time_prefill_s']['p50'] * 1e3:.1f} ms)"
        f" | TPOT p50 {m['tpot_s']['p50'] * 1e3:.2f} ms"
        f" | decode {m['decode_tok_s']['p50']:.1f} tok/s,"
        f" median step {1 / m['decode_step_p50_s']['p50']:.1f} tok/s"
        f" | GPU memory +{m['gpu_memory_mib']:.0f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=exllamav3.FORMATS, default="EXL3_4.0")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=2048, help="prefill chunk, tokens")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="print the result only")
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error("--new-tokens must be at least 2 to measure TPOT")

    if not args.no_record:
        record.require_clean_tree(args.allow_dirty)
    result = benchmark(args)
    print(summary_line(result))
    if args.no_record:
        print(json.dumps({"metrics": result["metrics"]}, indent=1))
    else:
        print(record.write_record(**result))


if __name__ == "__main__":
    main()
