"""Decode speed and time to first token of vLLM and TensorRT-LLM, measured as kvbound.bench
measures.

Route: the stack's OpenAI-compatible server (`vllm serve`, `trtllm-serve`) from its pinned
environment (kvbound.baselines.linux_stacks), one sequence at a time, on 127.0.0.1. Each
request is a POST to /v1/completions with the prompt as token ids, so the server tokenizes
nothing; greedy (temperature 0, top_k 1), EOS ignored, prefix caching off, and the tokens
streamed back as server-sent events carrying their ids (vLLM: return_token_ids; TensorRT-LLM:
detokenize false) and the running usage counts. The prompt is kvbound.bench.random_prompt, the
same seeded ids kvbound, the Hugging Face baseline and llama.cpp process.

Metrics follow kvbound.record and are computed by kvbound.bench.request_metrics from the times
the client receives each token, counted from just before the request is sent: TTFT, TPOT, decode
tokens/s, end-to-end tokens/s and the median step. The client times include the HTTP round
trip, the server's Python front end and the event stream. The server's own timings of each
request are kept beside them: vLLM's from its Prometheus histograms at /metrics (time to first
token and mean time per output token, read before and after each request), TensorRT-LLM's from
/perf_metrics (arrival, first and last token times on the executor's clock). Tokens a server
sends without their ids in an event are counted from the usage it reports; held_tokens counts
tokens that only the closing usage revealed, which take its time.

One server process per run, sized like kvbound's bench: the longest sequence is prompt + new
tokens, the KV cache holds that many tokens (rounded up to the stack's block) at the chosen KV
type, one sequence at a time, the whole prompt in one step (no chunked prefill), CUDA graphs
for batch 1. GPU memory is NVML's device memory in use, sampled once a second from before the
server starts to after it ends: gpu_memory_mib is the largest sample less the smallest, as
kvbound.bench prints for kvb-bench. It includes what the stack reserves beyond the model and
the cache (CUDA graphs, torch.compile workspaces, the sampler); the lines of the server log that
report memory are kept in server_log.

Usage: python -m kvbound.baselines.linux_stacks_bench --stack vllm --model tinyllama-1.1b \
           --weights F16 --prompt-len 128
"""

import argparse
import http.client
import json
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from kvbound import record
from kvbound.baselines import linux_stacks as ls
from kvbound.baselines.http_server import HOST
from kvbound.bench import random_prompt, request_metrics
from kvbound.models import load_spec, model_entry
from kvbound.nvml_sampler import NvmlSampler
from kvbound.perfmodel.model import Arch, kv_bytes_per_token

# The histograms of vLLM's /metrics that time each request on the server.
VLLM_TIMINGS = {
    "ttft_s": "vllm:time_to_first_token_seconds",
    "tpot_s": "vllm:request_time_per_output_token_seconds",
    "prefill_s": "vllm:request_prefill_time_seconds",
    "decode_s": "vllm:request_decode_time_seconds",
}
TIMINGS_WAIT_S = 2.0


def request_body(stack: str, model: str, prompt: list[int], new_tokens: int) -> dict[str, Any]:
    """Greedy decoding of exactly new_tokens tokens from token ids, streamed with their ids
    and a usage count in every event."""
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": new_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    if stack == "vllm":
        body["return_token_ids"] = True
    else:
        body["detokenize"] = False  # trtllm-serve then sends token_ids
    return body


def read_stream(
    lines: Iterable[bytes], origin: float, clock: Callable[[], float] = time.perf_counter
) -> dict[str, Any]:
    """The server-sent events of one streamed completion: the ids received, the seconds from
    origin to the event that carried each, the last usage reported, and the error the server
    sent, if any. Ids that arrive together take their event's time (multi_token_events counts
    such events). An event without ids counts the tokens its usage adds; tokens that only the
    closing usage counts take its time (held_tokens)."""
    times: list[float] = []
    tokens: list[int] = []
    multi = held = 0
    usage: dict[str, Any] = {}
    finish = None
    error = None
    for line in lines:
        if not line.startswith(b"data: "):
            continue
        now = clock() - origin
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        if "error" in event or event.get("object") == "error":
            error = event.get("error", event)
            break
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            ids = choice.get("token_ids")
            if ids is None:
                n = max(0, int(usage.get("completion_tokens", len(times))) - len(times))
            else:
                tokens += ids
                n = len(ids)
            times += [now] * n
            multi += n > 1
            finish = choice.get("finish_reason") or finish
        if not event.get("choices") and usage:
            held = max(0, int(usage.get("completion_tokens", len(times))) - len(times))
            times += [now] * held
    return {
        "times": times,
        "tokens": tokens,
        "usage": usage,
        "finish_reason": finish,
        "multi_token_events": multi,
        "held_tokens": held,
        "error": error,
    }


def stream_completion(
    port: int, body: dict[str, Any], origin: float | None = None
) -> dict[str, Any]:
    """POSTs one streamed /v1/completions and reads it (read_stream). Times count from origin,
    a time.perf_counter() value, or from just before the request is sent; sent_s is when it was
    sent. An HTTP error status counts as the server's error."""
    payload = json.dumps(body).encode()
    conn = http.client.HTTPConnection(HOST, port, timeout=600)
    sent = time.perf_counter()
    origin = sent if origin is None else origin
    try:
        conn.request("POST", "/v1/completions", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            out = read_stream([], origin)
            out["error"] = f"HTTP {response.status}: {response.read()[:500]!r}"
        else:
            out = read_stream(response, origin)
    finally:
        conn.close()
    out["sent_s"] = sent - origin
    return out


def get(port: int, path: str) -> bytes:
    conn = http.client.HTTPConnection(HOST, port, timeout=30)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"GET {path}: HTTP {response.status}: {body[:300]!r}")
        return body
    finally:
        conn.close()


def parse_prometheus(text: str, prefix: str) -> dict[str, float]:
    """The values of a Prometheus page whose names start with prefix, summed over labels."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        name_labels, _, value = line.rpartition(" ")
        name = name_labels.split("{", 1)[0]
        values[name] = values.get(name, 0.0) + float(value)
    return values


def histogram_mean(before: dict[str, float], after: dict[str, float], name: str) -> float | None:
    """The mean of a histogram's observations between two readings, or None without any."""
    count = after.get(f"{name}_count", 0.0) - before.get(f"{name}_count", 0.0)
    if count <= 0:
        return None
    return (after[f"{name}_sum"] - before.get(f"{name}_sum", 0.0)) / count


class ServerTimings:
    """The server's own timing of each request: read() before a request, timings() after."""

    def __init__(self, stack: str, port: int) -> None:
        self.stack = stack
        self.port = port
        self.before: dict[str, float] = {}

    def read(self) -> None:
        if self.stack == "vllm":
            self.before = parse_prometheus(get(self.port, "/metrics").decode(), "vllm:")
        else:
            get(self.port, "/perf_metrics")  # drains what earlier requests left

    def timings(self, n_new: int) -> dict[str, float | None]:
        """The request's server timings. A server may record a request just after its last
        event, so this polls for up to TIMINGS_WAIT_S."""
        deadline = time.monotonic() + TIMINGS_WAIT_S
        if self.stack == "vllm":
            name = VLLM_TIMINGS["ttft_s"] + "_count"
            while True:
                after = parse_prometheus(get(self.port, "/metrics").decode(), "vllm:")
                if after.get(name, 0.0) > self.before.get(name, 0.0) or time.monotonic() > deadline:
                    break
                time.sleep(0.05)
            return {k: histogram_mean(self.before, after, v) for k, v in VLLM_TIMINGS.items()}
        entries: list[dict[str, Any]] = []
        while not entries and time.monotonic() <= deadline:
            entries = json.loads(get(self.port, "/perf_metrics"))
            if not entries:
                time.sleep(0.05)
        if len(entries) != 1:
            raise RuntimeError(f"/perf_metrics holds {len(entries)} requests, expected 1")
        t = entries[0]["perf_metrics"]["timing_metrics"]
        return {
            "ttft_s": t["first_token_time"] - t["arrival_time"],
            "tpot_s": (t["last_token_time"] - t["first_token_time"]) / (n_new - 1),
            "queue_s": t["first_scheduled_time"] - t["arrival_time"],
        }


def bench_config(stack: str, arch: Arch, max_ctx: int, kv: str) -> ls.ServerConfig:
    """One sequence of max_ctx tokens, its whole prompt in one step, a KV cache that holds it."""
    tokens = ls.round_up(max_ctx, ls.KV_BLOCK_TOKENS[stack])
    per_token = kv_bytes_per_token(arch, ls.KV_VALUE_BYTES[kv])
    return ls.ServerConfig(
        max_ctx=max_ctx,
        max_seqs=1,
        max_step_tokens=max_ctx,
        kv_tokens=tokens,
        kv_bytes=int(tokens * per_token),
        kv=kv,
        chunked_prefill=False,
        cuda_graph_batch_sizes=(1,),
    )


def summarize_server(samples: list[dict[str, float | None]]) -> dict[str, Any]:
    """record.summarize of each server timing that every request reported."""
    out = {}
    for name in samples[0]:
        values = [s[name] for s in samples]
        if all(v is not None for v in values):
            out[name] = record.summarize([float(v) for v in values if v is not None])
    return out


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    ls.require_env(ls.STACKS[args.stack])
    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    ctx = args.ctx or args.prompt_len + args.new_tokens
    cfg = bench_config(args.stack, arch, ctx, args.kv)
    settings = ls.server_settings(args.stack, args.model, args.weights, cfg)
    weights = ls.weights_params(args.stack, args.model, args.weights)
    prompt = random_prompt(arch.vocab_size, args.prompt_len, args.seed)
    body = request_body(args.stack, args.model, prompt, args.new_tokens)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = (
        ls.STACKS[args.stack].root / "logs" / f"{stamp}-{args.model}-{args.weights}-bench.log"
    )

    outs, server_samples = [], []
    with NvmlSampler() as sampler:
        server = ls.Server(args.stack, settings, log_path)
        with server:
            timings = ServerTimings(args.stack, server.port)
            for i in range(args.warmup + args.iters):
                timings.read()
                out = stream_completion(server.port, body)
                if out["error"] is not None:
                    raise RuntimeError(f"/v1/completions failed: {out['error']}")
                if i >= args.warmup:
                    outs.append(out)
                    server_samples.append(timings.timings(args.new_tokens))
            log = server.log()
    for out in outs:
        if len(out["times"]) != args.new_tokens:
            raise RuntimeError(f"expected {args.new_tokens} tokens, received {len(out['times'])}")
        prompt_n = out["usage"].get("prompt_tokens")
        if prompt_n != args.prompt_len:
            raise RuntimeError(
                f"the server counted {prompt_n} prompt tokens, not {args.prompt_len}"
            )

    runs = [request_metrics(out["times"]) for out in outs]
    samples: dict[str, Any] = {name: [run[name] for run in runs] for name in runs[0]}
    metrics: dict[str, Any] = {name: record.summarize(values) for name, values in samples.items()}
    metrics["server"] = summarize_server(server_samples)
    metrics["multi_token_events"] = sum(out["multi_token_events"] for out in outs)
    metrics["held_tokens"] = sum(out["held_tokens"] for out in outs)
    used = sampler.samples["mem_used_mib"]
    metrics["gpu_memory_mib"] = max(used) - min(used)
    return {
        "kind": "generate",
        "engine": ls.engine_entry(args.stack, "openai-completions"),
        "model": model_entry(spec),
        "params": {
            "batch": 1,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "max_ctx": ctx,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "sampling": "greedy",
            "prompt": "kvbound.bench.random_prompt",
            **weights,
            "kv": args.kv,
            "kv_tokens": cfg.kv_tokens,
            "kv_bytes": cfg.kv_bytes,
            "server": settings,
            "request": {k: v for k, v in body.items() if k != "prompt"},
            "server_log": ls.log_facts(args.stack, log),
        },
        "metrics": metrics,
        "samples": {
            **samples,
            "server": server_samples,
            "tokens": outs[0]["tokens"],
            "same_tokens_every_iteration": all(o["tokens"] == outs[0]["tokens"] for o in outs),
        },
        "nvml": sampler.summary(),
    }


def summary_line(result: dict[str, Any]) -> str:
    m, p = result["metrics"], result["params"]
    server = m["server"]
    ttft = server.get("ttft_s", {}).get("p50")
    tpot = server.get("tpot_s", {}).get("p50")
    return (
        f"{result['engine']['name']} {result['model']['name']} {p['weights']} kv={p['kv']}"
        f" prompt {p['prompt_len']}:"
        f" TTFT p50 {m['ttft_s']['p50'] * 1e3:.1f} ms"
        + (f" (server {ttft * 1e3:.1f} ms)" if ttft is not None else "")
        + f" | TPOT p50 {m['tpot_s']['p50'] * 1e3:.2f} ms"
        + (f" (server {tpot * 1e3:.2f} ms)" if tpot is not None else "")
        + f" | decode {m['decode_tok_s']['p50']:.1f} tok/s,"
        f" median step {1 / m['decode_step_p50_s']['p50']:.1f} tok/s"
        f" | GPU memory +{m['gpu_memory_mib']:.0f} MiB"
        f" | {m['multi_token_events']} multi-token events, {m['held_tokens']} held"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=sorted(ls.STACKS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=ls.FORMATS, default="F16")
    parser.add_argument("--kv", choices=sorted(ls.KV_VALUE_BYTES), default="f16")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=0, help="default: prompt + new tokens")
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
