"""Runs a workload through the engine with static batching and records throughput and memory use.

Every request of the workload is queued at time zero. The engine serves them in batches of up
to --slots sequences; a batch runs until its longest sequence finishes. The record carries
throughput, per-request latency, and how much of the preallocated KV cache held tokens.

Usage: python -m kvbound.serve --model tinyllama-1.1b --workload longtail --slots 8
"""

import argparse
import itertools
from typing import Any

from kvbound import bench, record, workload
from kvbound.models import load_spec
from kvbound.perfmodel.model import Arch


def weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(v * w for v, w in zip(values, weights, strict=True)) / sum(weights)


def pass_metrics(run: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    """Metrics of one pass over the workload.

    KV occupancy is sampled after every step and weighted by the step's duration.
      kv_used_of_reserved  share of the cache positions held by admitted sequences that contain a
                           token; one minus this is the internal fragmentation.
      kv_used_of_pool      the same against the whole preallocated pool.
      slot_occupancy       share of slot-time during which the slot's sequence was still running.
    """
    steps = run["steps"]
    durations = [b - a for a, b in itertools.pairwise([0.0, *steps["end_s"]])]
    total = run["total_s"]
    pool_tokens = config["kv_blocks"] * config["kv_block_size"]
    used = weighted_mean(steps["used_tokens"], durations)
    reserved = weighted_mean(steps["reserved_tokens"], durations)
    first = sorted(r["token_times_s"][0] for r in run["requests"])
    last = sorted(r["token_times_s"][-1] for r in run["requests"])
    generated = sum(len(r["token_times_s"]) for r in run["requests"])
    return {
        "total_s": total,
        "throughput_tok_s": generated / total,
        "ttft_p50_s": record.percentile(first, 50),
        "ttft_p90_s": record.percentile(first, 90),
        "latency_p50_s": record.percentile(last, 50),
        "latency_p90_s": record.percentile(last, 90),
        "kv_used_mean_tokens": used,
        "kv_reserved_mean_tokens": reserved,
        "kv_used_peak_tokens": max(steps["used_tokens"]),
        "kv_used_of_reserved": used / reserved,
        "kv_used_of_pool": used * sum(durations) / (pool_tokens * total),
        "slot_occupancy": weighted_mean(steps["running"], durations) / config["slots"],
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    load = workload.load(args.workload)
    trace = workload.trace(load, arch.vocab_size, args.max_ctx)
    options = {
        "slots": args.slots,
        "max-ctx": args.max_ctx,
        "max-step-tokens": args.max_step_tokens,
        "kv-layout": args.kv_layout,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    raw, nvml = bench.run_kvb_bench(args.preset, spec.checkpoint_dir, trace, options)

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
            "dtype": "fp16",
            "workload": load.name,
            "workload_seed": load.seed,
            "n_requests": load.n_requests,
            "policy": "static",
            "sampling": "greedy",
            **raw["config"],
        },
        "metrics": metrics,
        "samples": samples,
        "nvml": nvml,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, help="name from configs/workloads/")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--max-ctx", type=int, default=1024)
    parser.add_argument("--max-step-tokens", type=int, default=1024)
    parser.add_argument("--kv-layout", default="head_major")
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
        f"{args.workload}, {args.slots} slots: {m['throughput_tok_s']['p50']:.0f} tok/s"
        f" | TTFT p50 {m['ttft_p50_s']['p50']:.2f} s"
        f" | KV in use {m['kv_used_of_reserved']['p50']:.0%} of reserved"
        f" | slots busy {m['slot_occupancy']['p50']:.0%}\n{path}"
    )


if __name__ == "__main__":
    main()
