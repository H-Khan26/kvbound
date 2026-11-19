"""Serving with vLLM and TensorRT-LLM: kvbound's workloads replayed against the stack's server,
measured as kvbound.serve measures.

The requests are kvbound.workload.trace(), the trace kvbound.serve feeds kvb-bench: the same
prompt token ids, output lengths (EOS ignored) and arrival times, all at zero or a Poisson
process with --rate. The open-loop client of kvbound.baselines.openloop, shared with the
llama.cpp baseline, sends each request at its arrival time on its own thread as one streamed
POST to /v1/completions (the request of linux_stacks_bench: token ids in and out, greedy).
Throughput, TTFT, TPOT and latency come from kvbound.serve.latency_metrics over the times the
client receives each token; send lag, failed requests and held tokens as in llamacpp_serve.

The server batches continuously with a paged KV cache of the bytes of kvbound.serve's pool,
--pool-seqs sequences of --max-ctx FP16 tokens, cut into tokens of the chosen KV type (fp8
holds twice the tokens): vLLM gets the bytes (--kv-cache-memory-bytes, blocks of 16 tokens),
TensorRT-LLM the tokens they hold, rounded down to its blocks of 32 (kv_cache_config.max_tokens).
Up to --max-seqs sequences run at once and a step processes up to --max-step-tokens tokens,
prefill chunks included: kvbound's --slots and --max-step-tokens. Prefix caching is off.
During each pass the server's counters are read 10 times a second: vLLM's Prometheus gauges
(requests running and waiting, KV use) and counters (preemptions, engine steps and the tokens
they processed), TensorRT-LLM's per-iteration stats at /metrics (active and queued requests,
requests per iteration, KV blocks in use). GPU memory is sampled by NVML.

Not comparable with kvbound.serve:
  admission    vLLM admits by free blocks and preempts the newest request when the cache runs
               out, recomputing it later, as kvbound does. TensorRT-LLM's default capacity
               policy (GUARANTEED_NO_EVICT) admits a request only when blocks for its whole
               max_tokens are free and never preempts.
  KV layout    blocks of 16 (vLLM) or 32 (TensorRT-LLM) tokens against kvbound's 16; the pool
               holds the same bytes, rounded down to whole blocks.
  transport    every token travels as a server-sent event over HTTP through the stack's Python
               front end, and the client parses them on the same machine.
  memory       besides the pool, each stack holds CUDA graphs, compile workspaces and its
               sampler; NVML's figure covers them all.

Usage: python -m kvbound.baselines.linux_stacks_serve --stack vllm --model tinyllama-1.1b \
           --workload longtail --rate 8
"""

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from typing import Any

from kvbound import record, workload
from kvbound.baselines import linux_stacks as ls
from kvbound.baselines import openloop
from kvbound.baselines.linux_stacks_bench import (
    get,
    parse_prometheus,
    request_body,
    stream_completion,
)
from kvbound.models import load_spec, model_entry
from kvbound.nvml_sampler import NvmlSampler
from kvbound.perfmodel.model import Arch, Precision, kv_bytes_per_token

METRICS_INTERVAL_S = 0.1
CUDA_GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)


def pool(stack: str, arch: Arch, pool_seqs: int, max_ctx: int, kv: str) -> tuple[int, int, float]:
    """Tokens and bytes of the stack's KV cache in kvbound.serve's pool bytes, rounded down to
    whole blocks, and the pool's bytes."""
    fp16 = Precision.named("fp16", "fp16").kv_bytes
    pool_bytes = pool_seqs * max_ctx * kv_bytes_per_token(arch, fp16)
    per_token = kv_bytes_per_token(arch, ls.KV_VALUE_BYTES[kv])
    block = ls.KV_BLOCK_TOKENS[stack]
    tokens = int(pool_bytes // per_token) // block * block
    return tokens, int(tokens * per_token), pool_bytes


def serve_config(args: argparse.Namespace, kv_tokens: int, kv_bytes: int) -> ls.ServerConfig:
    return ls.ServerConfig(
        max_ctx=args.max_ctx,
        max_seqs=args.max_seqs,
        max_step_tokens=args.max_step_tokens,
        kv_tokens=kv_tokens,
        kv_bytes=kv_bytes,
        kv=args.kv,
        chunked_prefill=True,
        cuda_graph_batch_sizes=tuple(b for b in CUDA_GRAPH_BATCH_SIZES if b <= args.max_seqs),
    )


def read_counters(stack: str, port: int) -> Any:
    """One reading of the server's counters: vLLM's Prometheus values, or the TensorRT-LLM
    iteration stats gathered since the last reading."""
    if stack == "vllm":
        return parse_prometheus(get(port, "/metrics").decode(), "vllm:")
    return json.loads(get(port, "/metrics"))


def vllm_counts(before: dict, after: dict, samples: list[dict]) -> dict[str, float]:
    """Engine steps, tokens per step and preemptions in a pass, from the counters read before
    and after it, and the peaks of the gauges sampled during it."""
    steps = after.get("vllm:iteration_tokens_total_count", 0.0) - before.get(
        "vllm:iteration_tokens_total_count", 0.0
    )
    tokens = after.get("vllm:iteration_tokens_total_sum", 0.0) - before.get(
        "vllm:iteration_tokens_total_sum", 0.0
    )
    preempted = after.get("vllm:num_preemptions_total", 0.0) - before.get(
        "vllm:num_preemptions_total", 0.0
    )

    def peak(name: str) -> float:
        return max((s.get(name, 0.0) for s in samples), default=0.0)

    return {
        "steps": steps,
        "tokens_per_step": tokens / steps if steps else 0.0,
        "preemptions": preempted,
        "running_peak": peak("vllm:num_requests_running"),
        "waiting_peak": peak("vllm:num_requests_waiting"),
        "kv_used_peak": peak("vllm:kv_cache_usage_perc"),
    }


def trtllm_counts(iterations: list[dict[str, Any]]) -> dict[str, float]:
    """Iterations that scheduled work in a pass, requests per such iteration, and the peaks of
    active and queued requests and of the KV blocks in use, from TensorRT-LLM's iteration
    stats."""
    busy = [
        it for it in iterations if it.get("inflightBatchingStats", {}).get("numScheduledRequests")
    ]
    scheduled = [it["inflightBatchingStats"]["numScheduledRequests"] for it in busy]
    kv = [it.get("kvCacheStats", {}) for it in iterations]
    used_share = [k["usedNumBlocks"] / k["maxNumBlocks"] for k in kv if k.get("maxNumBlocks")]
    return {
        "steps": float(len(busy)),
        "requests_per_step": sum(scheduled) / len(scheduled) if scheduled else 0.0,
        "running_peak": float(
            max((it.get("numActiveRequests", 0) for it in iterations), default=0)
        ),
        "waiting_peak": float(
            max((it.get("numQueuedRequests", 0) for it in iterations), default=0)
        ),
        "kv_used_peak": max(used_share, default=0.0),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    ls.require_env(ls.STACKS[args.stack])
    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    load = workload.load(args.workload)
    if args.requests:
        load = dataclasses.replace(load, n_requests=args.requests)
    trace = workload.trace(load, arch.vocab_size, args.max_ctx, args.rate)
    kv_tokens, kv_bytes, pool_bytes = pool(args.stack, arch, args.pool_seqs, args.max_ctx, args.kv)
    cfg = serve_config(args, kv_tokens, kv_bytes)
    settings = ls.server_settings(args.stack, args.model, args.weights, cfg)
    weights = ls.weights_params(args.stack, args.model, args.weights)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = (
        ls.STACKS[args.stack].root / "logs" / f"{stamp}-{args.model}-{args.weights}-serve.log"
    )

    passes, counts = [], []
    with NvmlSampler() as sampler:
        server = ls.Server(args.stack, settings, log_path)
        with server:
            port = server.port

            def send(req: dict[str, Any], start: float) -> dict[str, Any]:
                body = request_body(args.stack, args.model, req["prompt"], req["max_new_tokens"])
                return stream_completion(port, body, origin=start)

            for i in range(args.warmup + args.iters):
                poller = openloop.Poller(
                    lambda: read_counters(args.stack, port), METRICS_INTERVAL_S
                )
                before = poller.read()
                poller.start()
                results = openloop.run_pass(trace, send)
                poller.stop()
                after = poller.read()
                if i < args.warmup:
                    continue
                passes.append(results)
                if args.stack == "vllm":
                    counts.append(vllm_counts(before, after, poller.samples))
                else:
                    iterations = [it for s in [*poller.samples, after] for it in s]
                    counts.append(trtllm_counts(iterations))
            log = server.log()

    runs = [
        {**openloop.pass_metrics(trace, results), **c}
        for results, c in zip(passes, counts, strict=True)
    ]
    samples: dict[str, Any] = {name: [run[name] for run in runs] for name in runs[0]}
    metrics: dict[str, Any] = {name: record.summarize(values) for name, values in samples.items()}
    used = sampler.samples["mem_used_mib"]
    metrics["gpu_memory_mib"] = max(used) - min(used)
    samples["requests"] = [
        {
            "id": req["id"],
            "n_prompt": len(req["prompt"]),
            "n_new": len(out["times"]),
            "arrival_s": req["arrival_s"],
            "sent_s": out["sent_s"],
            "first_token_s": out["times"][0] if out["times"] else None,
            "last_token_s": out["times"][-1] if out["times"] else None,
            "error": out["error"],
        }
        for req, out in zip(trace["requests"], passes[0], strict=True)
    ]
    return {
        "kind": "serve",
        "engine": ls.engine_entry(args.stack, "openai-completions"),
        "model": model_entry(spec),
        "params": {
            "workload": load.name,
            "workload_seed": load.seed,
            "n_requests": load.n_requests,
            "arrival_rate": trace["arrival_rate"],
            "sampling": "greedy",
            "max_ctx": args.max_ctx,
            "pool_seqs": args.pool_seqs,
            "pool_bytes": pool_bytes,
            "kv": args.kv,
            "kv_tokens": kv_tokens,
            "kv_bytes": kv_bytes,
            "max_seqs": args.max_seqs,
            "max_step_tokens": args.max_step_tokens,
            "warmup": args.warmup,
            "iters": args.iters,
            **weights,
            "server": settings,
            "server_log": ls.log_facts(args.stack, log),
        },
        "metrics": metrics,
        "samples": samples,
        "nvml": sampler.summary(),
    }


def summary_line(result: dict[str, Any]) -> str:
    m, p = result["metrics"], result["params"]
    ms = {name: m[name]["p50"] * 1e3 for name in ("ttft_p50_s", "ttft_p90_s")}
    ms |= {name: m[name]["p50"] * 1e3 for name in ("tpot_p50_s", "tpot_p90_s")}
    return (
        f"{result['engine']['name']} {result['model']['name']} {p['weights']} kv={p['kv']}"
        f" {p['workload']} x{p['n_requests']} rate={p['arrival_rate'] or 0:g},"
        f" {p['kv_tokens']} KV tokens: {m['throughput_tok_s']['p50']:.0f} tok/s"
        f" | TTFT p50 {ms['ttft_p50_s']:.0f} ms p90 {ms['ttft_p90_s']:.0f} ms"
        f" | TPOT p50 {ms['tpot_p50_s']:.1f} ms p90 {ms['tpot_p90_s']:.1f} ms"
        f" | latency p50 {m['latency_p50_s']['p50']:.2f} s p90 {m['latency_p90_s']['p50']:.2f} s"
        f" | up to {m['running_peak']['max']:.0f} running, {m['waiting_peak']['max']:.0f} waiting"
        f" | {m['failed']['max']:.0f} failed, {m['held_tokens']['max']:.0f} held"
        f" | send lag max {m['send_lag_max_s']['max'] * 1e3:.1f} ms"
        f" | GPU memory +{m['gpu_memory_mib']:.0f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=sorted(ls.STACKS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=ls.FORMATS, default="F16")
    parser.add_argument("--kv", choices=sorted(ls.KV_VALUE_BYTES), default="f16")
    parser.add_argument("--workload", required=True, help="name from configs/workloads/")
    parser.add_argument("--rate", type=float, default=None, help="arrivals per second")
    parser.add_argument("--requests", type=int, default=0, help="default: the workload's count")
    parser.add_argument("--max-seqs", type=int, default=64, help="kvbound.serve's --slots")
    parser.add_argument(
        "--pool-seqs", type=int, default=8,
        help="KV pool size, as FP16 sequences of max-ctx tokens, as in kvbound.serve",
    )  # fmt: skip
    parser.add_argument("--max-ctx", type=int, default=1024)
    parser.add_argument("--max-step-tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="print the result only")
    args = parser.parse_args()

    if not args.no_record:
        record.require_clean_tree(args.allow_dirty)
    result = benchmark(args)
    print(summary_line(result))
    if args.no_record:
        print(json.dumps({"metrics": result["metrics"]}, indent=1))
        return
    if failed := result["metrics"]["failed"]["max"]:
        raise SystemExit(f"not recorded: {failed:.0f} requests failed in a measured pass")
    print(record.write_record(**result))


if __name__ == "__main__":
    main()
