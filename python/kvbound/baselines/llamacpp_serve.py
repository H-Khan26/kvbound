"""Serving with llama.cpp: kvbound's workloads replayed against llama-server, measured as
kvbound.serve measures.

The requests are kvbound.workload.trace(), the trace kvbound.serve feeds kvb-bench: the same
prompt token ids, output lengths (EOS ignored) and arrival times, all at zero or a Poisson
process with --rate. An open-loop client sends each request at its arrival time on its own
thread, never waiting for earlier ones, as one streamed POST to /completion: the prompt as token
ids, greedy (top_k 1), nothing reused from the prompt cache. Throughput, TTFT, TPOT and latency
come from kvbound.serve.latency_metrics, the code kvbound's own records use, over the times the
client receives each token, counted from the start of the pass and from each request's scheduled
arrival. send_lag is how late the client sent each request after its arrival; a request the
server answers with an error, or with fewer tokens than asked, counts as failed and leaves the
latency figures. Tokens the server held back inside a UTF-8 character arrive with the closing
event and take its time (held_tokens, see llamacpp_bench.read_stream).

The server is llama-server with --parallel slots, a unified KV cache shared by all of them
(--kv-unified) and continuous batching. The cache holds the bytes of kvbound.serve's pool,
--pool-seqs sequences of --max-ctx FP16 tokens, cut into cells of the chosen type: f16 cells are
kvbound's FP16 token bytes, so the pool holds the same tokens; q8_0 (34 bytes per 32 values,
the analog of kvbound's INT8 cache) holds about 1.9 times as many, rounded down to a multiple of
256. --batch-size and --ubatch-size are kvbound's --max-step-tokens. The server's /metrics are
read 10 times a second: requests processing and deferred, and per pass the decode calls and the
busy slots per decode call. GPU memory is sampled by NVML, as in llamacpp_bench.

Not comparable with kvbound.serve:
  admission    llama.cpp admits a request whenever a slot is free and never preempts. When the
               shared cache fills it shrinks the batch and then fails every request in flight
               ("Context size has been exceeded"). kvbound admits by free blocks and preempts
               the newest sequence, recomputing it later. --parallel 0 picks the most slots that
               cannot overflow the cache (cells over the longest request); longtail's 48
               requests fit at once, so it runs at 64 slots as kvbound's paged runs do.
  KV layout    one unified cache of single-token cells against 16-token blocks: neither the
               internal fragmentation nor the preemption counts of kvbound's records exist here.
  transport    every token travels as a server-sent event over its own HTTP connection; with
               tens of streams the HTTP threads and the client's parsing are part of the timings.
  CPU          client, HTTP threads and the server's sampling share this machine's cores, where
               kvb-bench samples on the GPU inside one process.
  steps        decode calls and busy slots per call come from llama.cpp's counters; kvbound's
               running_mean weights each step by its duration.

Usage: python -m kvbound.baselines.llamacpp_serve --model tinyllama-1.1b --weights Q4_0 \
           --workload longtail --rate 8 --parallel 64
"""

import argparse
import dataclasses
import http.client
import json
import os
from datetime import UTC, datetime
from typing import Any

from kvbound import record, workload
from kvbound.baselines import llamacpp, openloop
from kvbound.baselines.llamacpp_bench import (
    HOST,
    Server,
    buffers_mib,
    command_line,
    request_body,
    stream_completion,
)
from kvbound.models import load_spec, model_entry
from kvbound.nvml_sampler import NvmlSampler
from kvbound.perfmodel.model import Arch, Precision, kv_bytes_per_token

# Bytes of one cached value in each llama.cpp KV type: q8_0 stores 32 values in 34 bytes.
KV_VALUE_BYTES = {"f16": 2.0, "q8_0": 34 / 32}
# llama.cpp pads its context to this many cells; the pool is rounded down to it.
CELL_ROUND = 256
MAX_SLOTS = 64
METRICS_INTERVAL_S = 0.1
# llama-server's error when a request no longer fits in the cache, in lower case.
CONTEXT_EXCEEDED = "context size has been exceeded"


def kv_cells(arch: Arch, pool_seqs: int, max_ctx: int, kv: str) -> tuple[int, float]:
    """Cells of llama.cpp's cache in kvbound.serve's pool bytes, and those bytes."""
    fp16 = Precision.named("fp16", "fp16").kv_bytes
    pool_bytes = pool_seqs * max_ctx * kv_bytes_per_token(arch, fp16)
    cell_bytes = kv_bytes_per_token(arch, KV_VALUE_BYTES[kv])
    return int(pool_bytes // cell_bytes) // CELL_ROUND * CELL_ROUND, pool_bytes


def safe_slots(cells: int, trace: dict[str, Any]) -> int:
    """The most slots, up to MAX_SLOTS, whose requests cannot overflow the cache together."""
    longest = max(len(r["prompt"]) + r["max_new_tokens"] for r in trace["requests"])
    return max(1, min(MAX_SLOTS, cells // longest))


def parse_metrics(text: str) -> dict[str, float]:
    """The llamacpp:* values of a Prometheus /metrics page."""
    values = {}
    for line in text.splitlines():
        if line.startswith("llamacpp:"):
            name, _, value = line.partition(" ")
            values[name.removeprefix("llamacpp:")] = float(value)
    return values


def read_metrics(port: int) -> dict[str, float]:
    """llama-server's /metrics, parsed (parse_metrics)."""
    conn = http.client.HTTPConnection(HOST, port, timeout=10)
    try:
        conn.request("GET", "/metrics")
        return parse_metrics(conn.getresponse().read().decode())
    finally:
        conn.close()


def step_counts(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Decode calls in a pass and their mean busy slots, from the cumulative counters."""
    decodes = after["n_decode_total"] - before["n_decode_total"]
    busy = (
        after["n_busy_slots_per_decode"] * after["n_decode_total"]
        - before["n_busy_slots_per_decode"] * before["n_decode_total"]
    )
    return {"steps": decodes, "busy_slots_per_step": busy / decodes if decodes else 0.0}


def run_pass(port: int, trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Sends every request of the trace at its arrival time (kvbound.baselines.openloop);
    returns what each received, with times in seconds from the start of the pass."""

    def send(req: dict[str, Any], start: float) -> dict[str, Any]:
        body = request_body(req["prompt"], req["max_new_tokens"])
        return stream_completion(port, body, origin=start)

    return openloop.run_pass(trace, send)


def failure_cause(results: list[dict[str, Any]]) -> str:
    """The first error of a pass's requests, with what to change when the cache overflowed."""
    error = openloop.first_error(results)
    if CONTEXT_EXCEEDED in error.lower():
        error += (
            "; with more slots than the cache holds requests llama.cpp fails every request in "
            "flight: lower --parallel, or pass 0 for the most slots that cannot overflow"
        )
    return error


def pass_metrics(trace: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, float]:
    """kvbound.baselines.openloop.pass_metrics, with llama.cpp's failure advice."""
    return openloop.pass_metrics(trace, results, failure_cause)


def server_settings(args: argparse.Namespace, cells: int, slots: int, n_requests: int) -> dict:
    """Every llama-server setting that changes the numbers, by its flag."""
    return {
        "--ctx-size": cells,
        "--parallel": slots,
        "--kv-unified": True,
        "--cont-batching": True,
        "--gpu-layers": "all",
        "--fit": "off",
        "--flash-attn": "on",
        "--batch-size": args.max_step_tokens,
        "--ubatch-size": args.max_step_tokens,
        "--cache-type-k": args.kv,
        "--cache-type-v": args.kv,
        "--threads": args.threads,
        "--threads-batch": args.threads,
        # One HTTP thread per request in flight, so that connections never queue in front of
        # the slots.
        "--threads-http": max(slots, n_requests) + 2,
        "--cache-ram": 0,
        "--no-cache-prompt": True,
        "--no-webui": True,
        "--metrics": True,
        "--log-verbosity": 4,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    load = workload.load(args.workload)
    if args.requests:
        load = dataclasses.replace(load, n_requests=args.requests)
    trace = workload.trace(load, arch.vocab_size, args.max_ctx, args.rate)
    cells, pool_bytes = kv_cells(arch, args.pool_seqs, args.max_ctx, args.kv)
    slots = args.parallel or safe_slots(cells, trace)
    gguf = llamacpp.require_gguf(args.model, args.weights)
    settings = server_settings(args, cells, slots, load.n_requests)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = llamacpp.ROOT / "logs" / f"{stamp}-{args.model}-{args.weights}-serve.log"

    passes, steps, processing, deferred = [], [], [], []
    with NvmlSampler() as sampler:
        server = Server(["--model", str(gguf), *command_line(settings)], log_path)
        with server:
            for i in range(args.warmup + args.iters):
                poller = openloop.Poller(lambda: read_metrics(server.port), METRICS_INTERVAL_S)
                before = poller.read()
                poller.start()
                results = run_pass(server.port, trace)
                poller.stop()
                if i < args.warmup:
                    continue
                passes.append(results)
                steps.append(step_counts(before, poller.read()))
                processing.append(
                    max((s["requests_processing"] for s in poller.samples), default=0.0)
                )
                deferred.append(max((s["requests_deferred"] for s in poller.samples), default=0.0))
            log = server.log()

    runs = [
        {**pass_metrics(trace, results), **counts, "processing_peak": proc, "deferred_peak": dfr}
        for results, counts, proc, dfr in zip(passes, steps, processing, deferred, strict=True)
    ]
    samples: dict[str, Any] = {name: [run[name] for run in runs] for name in runs[0]}
    metrics: dict[str, Any] = {name: record.summarize(values) for name, values in samples.items()}
    used = sampler.samples["mem_used_mib"]
    metrics["gpu_memory_mib"] = max(used) - min(used)
    metrics["buffers_mib"] = buffers_mib(log)
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
        "engine": llamacpp.engine_entry("llama-server"),
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
            "kv_cells": cells,
            "slots": slots,
            "warmup": args.warmup,
            "iters": args.iters,
            **llamacpp.gguf_params(args.model, args.weights),
            "server": settings,
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
        f"{result['model']['name']} {p['weights']} kv={p['kv']} {p['workload']}"
        f" x{p['n_requests']} rate={p['arrival_rate'] or 0:g}, {p['slots']} slots,"
        f" {p['kv_cells']} cells: {m['throughput_tok_s']['p50']:.0f} tok/s"
        f" | TTFT p50 {ms['ttft_p50_s']:.0f} ms p90 {ms['ttft_p90_s']:.0f} ms"
        f" | TPOT p50 {ms['tpot_p50_s']:.1f} ms p90 {ms['tpot_p90_s']:.1f} ms"
        f" | latency p50 {m['latency_p50_s']['p50']:.2f} s p90 {m['latency_p90_s']['p50']:.2f} s"
        f" | up to {m['processing_peak']['max']:.0f} processing,"
        f" {m['busy_slots_per_step']['p50']:.1f} per step"
        f" | {m['failed']['max']:.0f} failed, {m['held_tokens']['max']:.0f} held"
        f" | send lag max {m['send_lag_max_s']['max'] * 1e3:.1f} ms"
        f" | GPU memory +{m['gpu_memory_mib']:.0f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=llamacpp.FORMATS, default="F16")
    parser.add_argument("--kv", choices=sorted(KV_VALUE_BYTES), default="f16")
    parser.add_argument("--workload", required=True, help="name from configs/workloads/")
    parser.add_argument("--rate", type=float, default=None, help="arrivals per second")
    parser.add_argument("--requests", type=int, default=0, help="default: the workload's count")
    parser.add_argument(
        "--parallel", type=int, default=0,
        help="llama-server slots; 0: the most that cannot overflow the cache",
    )  # fmt: skip
    parser.add_argument(
        "--pool-seqs", type=int, default=8,
        help="KV pool size, as FP16 sequences of max-ctx tokens, as in kvbound.serve",
    )  # fmt: skip
    parser.add_argument("--max-ctx", type=int, default=1024)
    parser.add_argument("--max-step-tokens", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
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
    # Throughput and latency of a pass cover its completed requests only, so a pass that lost
    # requests is not the workload kvbound's records measure.
    if failed := result["metrics"]["failed"]["max"]:
        raise SystemExit(f"not recorded: {failed:.0f} requests failed in a measured pass")
    print(record.write_record(**result))


if __name__ == "__main__":
    main()
