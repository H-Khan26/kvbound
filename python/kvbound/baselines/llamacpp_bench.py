"""Decode speed and time to first token of llama.cpp, measured the way kvbound.bench measures.

Route: the pinned release's llama-server, one slot, all layers on the GPU, on 127.0.0.1. Each
request is a POST to /completion with the prompt as token ids (the server then adds no BOS),
greedy (top_k 1, temperature 0, no other sampler), EOS ignored, the prompt cache off, and the
tokens streamed back as server-sent events with their ids. The prompt is
kvbound.bench.random_prompt, the same seeded ids kvbound and the Hugging Face baseline process.

Metrics follow kvbound.record and are computed by kvbound.bench.request_metrics from the times
the client receives each token, counted from just before the request is sent: TTFT, TPOT, decode
tokens/s, end-to-end tokens/s and the median step. The client times therefore include the HTTP
round trip, JSON and the event stream; the server's own timings of each request (prompt_ms,
predicted_per_token_ms) are kept beside them, so the transport's share can be read off. When the
server holds a token back to complete a UTF-8 character, it arrives in the next event and takes
that event's time; multi_token_events counts how often that happened.

One server process per run: its context is prompt + new tokens, as kvbound's max_ctx, and it
exits before the run returns. GPU memory is NVML's device memory in use, sampled once a second
from before the server starts to after it ends: gpu_memory_mib is the largest sample less the
smallest, the reading kvbound.bench prints for kvb-bench. The
buffers llama.cpp reports for the model, KV cache and compute are parsed from its log.

Usage: python -m kvbound.baselines.llamacpp_bench --model tinyllama-1.1b --weights Q4_K_M \
           --prompt-len 128
"""

import argparse
import http.client
import json
import os
import re
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from kvbound import record
from kvbound.baselines import llamacpp
from kvbound.bench import random_prompt, request_metrics
from kvbound.models import load_spec, model_entry
from kvbound.nvml_sampler import NvmlSampler
from kvbound.perfmodel.model import Arch

KV_TYPES = ("f16", "q8_0", "q4_0")
HOST = "127.0.0.1"
LOAD_TIMEOUT_S = 300
# The buffer lines llama.cpp logs when it loads a model and creates a context.
BUFFER_LINE = re.compile(
    r"\b(CUDA\d+|CPU|CUDA_Host)\w* (model|KV|compute) buffer size =\s*([\d.]+) MiB"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        port: int = s.getsockname()[1]
        return port


class Server:
    """llama-server as a context manager: started and loaded on enter, ended on exit."""

    def __init__(self, args: list[str], log_path: Path) -> None:
        self.port = free_port()
        self.command = [str(llamacpp.exe("llama-server")), *args, "--host", HOST]
        self.command += ["--port", str(self.port)]
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("wb") as log:
            self.process = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT,
                cwd=self.log_path.parent, env=llamacpp.runtime_env(),
            )  # fmt: skip
        try:
            self._wait_loaded()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _wait_loaded(self) -> None:
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        assert self.process is not None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.log_path}:\n{self.log()}")
            try:
                conn = http.client.HTTPConnection(HOST, self.port, timeout=5)
                conn.request("GET", "/health")
                if conn.getresponse().status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"llama-server did not load in {LOAD_TIMEOUT_S} s; see {self.log_path}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def log(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        """One streamed completion: the ids received, the seconds from just before the request
        was sent to the event that carried each, and the server's closing event."""
        payload = json.dumps({**body, "stream": True, "return_tokens": True}).encode()
        conn = http.client.HTTPConnection(HOST, self.port, timeout=600)
        start = time.perf_counter()
        conn.request("POST", "/completion", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"/completion returned {response.status}: {response.read()!r}")
        times: list[float] = []
        tokens: list[int] = []
        events = multi = 0
        final: dict[str, Any] = {}
        for line in response:
            if not line.startswith(b"data: "):
                continue
            now = time.perf_counter() - start
            event = json.loads(line[6:])
            ids = event.get("tokens") or []
            times += [now] * len(ids)
            tokens += ids
            events += 1
            multi += len(ids) > 1
            if event.get("stop"):
                final = event
                break
        conn.close()
        return {"times": times, "tokens": tokens, "final": final, "multi_token_events": multi}


def buffers_mib(log: str) -> dict[str, float]:
    """MiB of each buffer llama.cpp reports, summed by "<device> <kind>"."""
    totals: dict[str, float] = {}
    for device, kind, mib in BUFFER_LINE.findall(log):
        key = f"{device} {kind}"
        totals[key] = totals.get(key, 0.0) + float(mib)
    return totals


def server_settings(args: argparse.Namespace, ctx: int) -> dict[str, Any]:
    """Every llama-server setting that changes the numbers, by its flag."""
    return {
        "--ctx-size": ctx,
        "--parallel": 1,
        "--gpu-layers": "all",
        "--fit": "off",
        "--flash-attn": args.flash_attn,
        "--batch-size": args.batch_size,
        "--ubatch-size": args.ubatch_size,
        "--cache-type-k": args.kv,
        "--cache-type-v": args.kv,
        "--threads": args.threads,
        "--threads-batch": args.threads,
        "--cache-ram": 0,
        "--no-cache-prompt": True,
        "--no-webui": True,
    }


def command_line(settings: dict[str, Any]) -> list[str]:
    out = []
    for flag, value in settings.items():
        out += [flag] if value is True else [flag, str(value)]
    return out


def request_body(prompt: list[int], new_tokens: int) -> dict[str, Any]:
    """Greedy decoding of exactly new_tokens tokens from token ids, nothing reused."""
    return {
        "prompt": prompt,
        "n_predict": new_tokens,
        "ignore_eos": True,
        "cache_prompt": False,
        "samplers": ["top_k"],
        "top_k": 1,
        "temperature": 0.0,
        "seed": 0,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    ctx = args.ctx or args.prompt_len + args.new_tokens
    gguf = llamacpp.require_gguf(args.model, args.weights)
    weights = llamacpp.gguf_params(args.model, args.weights)
    settings = server_settings(args, ctx)
    prompt = random_prompt(arch.vocab_size, args.prompt_len, args.seed)
    body = request_body(prompt, args.new_tokens)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = llamacpp.ROOT / "logs" / f"{stamp}-{args.model}-{args.weights}-server.log"

    with NvmlSampler() as sampler:
        server = Server(["--model", str(gguf), *command_line(settings)], log_path)
        with server:
            for _ in range(args.warmup):
                server.complete(body)
            outs = [server.complete(body) for _ in range(args.iters)]
            log = server.log()
    for out in outs:
        if len(out["tokens"]) != args.new_tokens:
            raise RuntimeError(f"expected {args.new_tokens} tokens, received {len(out['tokens'])}")
        prompt_n = out["final"]["timings"]["prompt_n"]
        if prompt_n != args.prompt_len:
            raise RuntimeError(
                f"the server evaluated {prompt_n} prompt tokens, not {args.prompt_len}"
            )

    runs = [request_metrics(out["times"]) for out in outs]
    samples: dict[str, Any] = {name: [run[name] for run in runs] for name in runs[0]}
    metrics: dict[str, Any] = {name: record.summarize(values) for name, values in samples.items()}
    timings = [out["final"]["timings"] for out in outs]
    server_samples = {
        "prompt_ms": [t["prompt_ms"] for t in timings],
        "predicted_per_token_ms": [t["predicted_per_token_ms"] for t in timings],
        "predicted_per_second": [t["predicted_per_second"] for t in timings],
    }
    metrics["server"] = {name: record.summarize(v) for name, v in server_samples.items()}
    metrics["multi_token_events"] = sum(out["multi_token_events"] for out in outs)
    used = sampler.samples["mem_used_mib"]
    metrics["gpu_memory_mib"] = max(used) - min(used)
    metrics["buffers_mib"] = buffers_mib(log)
    return {
        "kind": "generate",
        "engine": llamacpp.engine_entry("llama-server"),
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
            "server": settings,
            "request": {k: v for k, v in body.items() if k != "prompt"},
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
    m = result["metrics"]
    server = m["server"]
    return (
        f"{result['model']['name']} {result['params']['weights']}"
        f" prompt {result['params']['prompt_len']}:"
        f" TTFT p50 {m['ttft_s']['p50'] * 1e3:.1f} ms"
        f" (server prompt {server['prompt_ms']['p50']:.1f} ms)"
        f" | TPOT p50 {m['tpot_s']['p50'] * 1e3:.2f} ms"
        f" | decode {m['decode_tok_s']['p50']:.1f} tok/s,"
        f" median step {1 / m['decode_step_p50_s']['p50']:.1f} tok/s"
        f" (server {server['predicted_per_second']['p50']:.1f})"
        f" | GPU memory +{m['gpu_memory_mib']:.0f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=llamacpp.FORMATS, default="F16")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=0, help="default: prompt + new tokens")
    parser.add_argument("--kv", choices=KV_TYPES, default="f16", help="KV cache type")
    parser.add_argument("--flash-attn", choices=["on", "off", "auto"], default="on")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
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
