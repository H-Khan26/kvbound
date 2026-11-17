"""Runs a workload through the engine under one scheduling policy and records throughput,
latency and memory use.

The KV pool is the same number of bytes in every run, so runs compare what a policy or a
precision does with the same memory: --pool-seqs gives it as the bytes of that many FP16
sequences of --max-ctx tokens. With --kv int8 a cached token takes about half the bytes, so the
same pool holds about twice the tokens.

  static      --slots sequences form a batch that runs until its longest member finishes; each
              holds one block of max_ctx tokens.
  continuous  requests join and leave between steps. With --kv-block-size 16 this is the paged
              design: up to --slots sequences share the pool block by block.

Requests are queued at time zero, or arrive as a Poisson process with --rate requests per second.

Usage: python -m kvbound.serve --model tinyllama-1.1b --workload longtail --policy continuous \
           --slots 64 --kv-block-size 16
"""

import argparse
import dataclasses
from typing import Any

from kvbound import bench, record, workload
from kvbound.models import load_spec
from kvbound.perfmodel.model import Arch, Precision, kv_bytes_per_token
from kvbound.weight_format import VARIANTS


def weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(v * w for v, w in zip(values, weights, strict=True)) / sum(weights)


def pass_metrics(run: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    """Metrics of one pass over the workload.

    Latency is per request, from its arrival: TTFT to its first token, TPOT between its tokens
    (requests with one token have none), latency to its last token.

    KV occupancy is sampled after every step and weighted by the step's duration, so time the
    engine spends idle between arrivals does not count.
      kv_used_of_reserved  share of the cache positions held by admitted sequences that contain a
                           token; one minus this is the internal fragmentation.
      kv_used_of_pool      the same against the whole preallocated pool.
      running_peak         most sequences processed in one step: the concurrency reached.
      slot_occupancy       share of slot-time during which the slot's sequence was still running.
      recomputed_tokens    rows processed beyond one per cached token: the cost of preemption.
    """
    steps = run["steps"]
    durations = [end - start for start, end in zip(steps["start_s"], steps["end_s"], strict=True)]
    total = run["total_s"]
    pool_tokens = config["kv_blocks"] * config["kv_block_size"]
    used = weighted_mean(steps["used_tokens"], durations)
    reserved = weighted_mean(steps["reserved_tokens"], durations)
    requests = run["requests"]
    first = sorted(r["token_times_s"][0] - r["arrival_s"] for r in requests)
    last = sorted(r["token_times_s"][-1] - r["arrival_s"] for r in requests)
    tpot = sorted(
        (r["token_times_s"][-1] - r["token_times_s"][0]) / (len(r["token_times_s"]) - 1)
        for r in requests
        if len(r["token_times_s"]) > 1
    )
    generated = sum(len(r["token_times_s"]) for r in requests)
    needed_rows = sum(r["n_prompt"] + len(r["token_times_s"]) - 1 for r in requests)
    graphs = run["graphs"]
    return {
        "total_s": total,
        "busy_s": sum(durations),
        "throughput_tok_s": generated / total,
        "ttft_p50_s": record.percentile(first, 50),
        "ttft_p90_s": record.percentile(first, 90),
        "tpot_p50_s": record.percentile(tpot, 50),
        "tpot_p90_s": record.percentile(tpot, 90),
        "latency_p50_s": record.percentile(last, 50),
        "latency_p90_s": record.percentile(last, 90),
        "kv_used_mean_tokens": used,
        "kv_reserved_mean_tokens": reserved,
        "kv_used_peak_tokens": max(steps["used_tokens"]),
        "kv_reserved_peak_tokens": max(steps["reserved_tokens"]),
        "kv_used_of_reserved": used / reserved,
        "kv_used_of_pool": used / pool_tokens,
        "kv_reserved_of_pool": reserved / pool_tokens,
        "running_mean": weighted_mean(steps["seqs"], durations),
        "running_peak": max(steps["seqs"]),
        "slot_occupancy": weighted_mean(steps["running"], durations) / config["slots"],
        "preemptions": sum(steps["preempted"]),
        "recomputed_tokens": sum(steps["rows"]) - needed_rows,
        "steps": len(durations),
        "graph_replayed_share": graphs["replayed_steps"] / len(durations),
        "graphs_captured": graphs["captured"],
        "graph_capture_s": graphs["capture_s"],
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    load = workload.load(args.workload)
    if args.requests:
        load = dataclasses.replace(load, n_requests=args.requests)
    trace = workload.trace(load, arch.vocab_size, args.max_ctx, args.rate)
    block_size = args.kv_block_size or args.max_ctx
    # The pool is sized in bytes, as FP16 sequences, then cut into blocks of the actual dtype.
    fp16_token = kv_bytes_per_token(arch, Precision.named("fp16", "fp16").kv_bytes)
    token = kv_bytes_per_token(arch, Precision.named("fp16", args.kv, arch.head_dim).kv_bytes)
    pool_bytes = (args.pool_seqs or args.slots) * args.max_ctx * fp16_token
    kv_blocks = int(pool_bytes // (block_size * token))
    options = {
        "policy": args.policy,
        "slots": args.slots,
        "max-ctx": args.max_ctx,
        "max-step-tokens": args.max_step_tokens,
        "kv-block-size": block_size,
        "kv-blocks": kv_blocks,
        "kv-layout": args.kv_layout,
        "kv-dtype": args.kv,
        "quantized-gemm-rows": args.quantized_gemm_rows,
        "fuse": args.fuse,
        "f16-gemv-rows": args.f16_gemv_rows,
        "expand-quantized": args.expand_quantized,
        "max-graphs": args.max_graphs,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    model_dir = bench.checkpoint_dir(args.model, args.weights)
    raw, nvml = bench.run_kvb_bench(args.preset, model_dir, trace, options)

    runs = [pass_metrics(run, raw["config"]) for run in raw["iterations"]]
    samples: dict[str, Any] = {name: [run[name] for run in runs] for name in runs[0]}
    metrics: dict[str, Any] = {name: record.summarize(values) for name, values in samples.items()}
    metrics["memory"] = raw["memory"]
    # The first pass in full: the step-by-step occupancy and each request's timeline.
    first = raw["iterations"][0]
    samples["steps"] = first["steps"]
    samples["requests"] = [
        {
            "id": r["id"],
            "n_prompt": r["n_prompt"],
            "n_new": len(r["token_times_s"]),
            "arrival_s": r["arrival_s"],
            "first_token_s": r["token_times_s"][0],
            "last_token_s": r["token_times_s"][-1],
        }
        for r in first["requests"]
    ]
    samples["plan"] = raw["plan"]
    return {
        "kind": "serve",
        "engine": {"name": "kvbound", "preset": args.preset},
        "model": {"name": spec.name, "hf_repo": spec.hf_repo, "revision": spec.revision},
        "params": {
            "workload": load.name,
            "workload_seed": load.seed,
            "n_requests": load.n_requests,
            "arrival_rate": trace["arrival_rate"],
            "sampling": "greedy",
            **raw["config"],
            "weights": args.weights,  # the variant name; the binary knows only the scheme
        },
        "metrics": metrics,
        "samples": samples,
        "nvml": nvml,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, help="name from configs/workloads/")
    parser.add_argument("--policy", choices=["static", "continuous"], default="static")
    parser.add_argument("--slots", type=int, default=8, help="sequences cached at the same time")
    parser.add_argument(
        "--pool-seqs", type=int, default=0,
        help="KV pool size, as FP16 sequences of max-ctx tokens; default: --slots",
    )  # fmt: skip
    parser.add_argument("--kv-block-size", type=int, default=0, help="default: max-ctx")
    parser.add_argument("--weights", choices=VARIANTS, default="fp16")
    parser.add_argument("--kv", choices=bench.KV_DTYPES, default="fp16")
    parser.add_argument("--rate", type=float, default=None, help="arrivals per second")
    parser.add_argument("--requests", type=int, default=0, help="default: the workload's count")
    parser.add_argument("--max-ctx", type=int, default=1024)
    parser.add_argument("--max-step-tokens", type=int, default=1024)
    parser.add_argument("--kv-layout", default="head_major")
    parser.add_argument("--max-graphs", type=int, default=16)
    parser.add_argument(
        "--quantized-gemm-rows", type=int, default=16,
        help="widest step that multiplies quantized weights without expanding them to FP16",
    )  # fmt: skip
    parser.add_argument(
        "--fuse", action=argparse.BooleanOptionalAction, default=True, help="the fused program"
    )
    parser.add_argument(
        "--f16-gemv-rows", type=int, default=4, help="rows up to which kvbound's FP16 kernel runs"
    )
    parser.add_argument(
        "--expand-quantized", action="store_true",
        help="wide steps expand quantized matrices to FP16 for cuBLAS",
    )  # fmt: skip
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--preset", default="cuda")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    record.require_clean_tree(args.allow_dirty)
    result = benchmark(args)
    path = record.write_record(**result)
    m = result["metrics"]
    print(
        f"{args.workload}, {args.policy}, {args.slots} slots:"
        f" {m['throughput_tok_s']['p50']:.0f} tok/s"
        f" | up to {m['running_peak']['p50']:.0f} sequences at once"
        f" | TTFT p50 {m['ttft_p50_s']['p50']:.2f} s"
        f" | TPOT p50 {m['tpot_p50_s']['p50'] * 1e3:.1f} ms"
        f" | KV in use {m['kv_used_of_reserved']['p50']:.0%} of reserved,"
        f" {m['kv_used_of_pool']['p50']:.0%} of the pool"
        f" | {m['preemptions']['p50']:.0f} preemptions\n{path}"
    )


if __name__ == "__main__":
    main()
