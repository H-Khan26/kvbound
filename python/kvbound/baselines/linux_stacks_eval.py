"""WikiText-2 perplexity of vLLM and TensorRT-LLM under kvbound's protocol.

The token stream, the windows and the arithmetic are kvbound.evals.perplexity's: the joined
test split tokenized once by the model's Hugging Face tokenizer, non-overlapping windows of
`window` tokens, every position after the first of a window scored, the NLL summed in float64.
Only the model's numbers come from the stack, fed each window as token ids from position 0
with nothing cached:

  vLLM          its server, the one linux_stacks_bench times: POST /v1/completions with the
                window as the prompt, max_tokens 1 and prompt_logprobs 0, which returns the
                log-probability of every prompt token after the first under the model's raw
                logits (computed in float32 by vLLM's sampler). The NLL is their negated sum.
  TensorRT-LLM  trtllm-serve's completions refuse logprobs, and its one other route, the full
                context logits as JSON, would carry 33 million (TinyLlama) to 131 million (Llama
                3.2) numbers per window. Its LLM API returns them as a tensor instead:
                linux_stacks_worker feeds each window through it in TensorRT-LLM's environment
                and applies kvbound's rule to the logits (float32 logsumexp, float64 sum).

Neither stack tokenizes here: ids go in, so the Hugging Face tokenizer of the checkpoint is the
only one involved, as in kvbound's own evaluation.

Usage: python -m kvbound.baselines.linux_stacks_eval --stack vllm --model tinyllama-1.1b \
           --weights F16
"""

import argparse
import http.client
import json
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from kvbound import record
from kvbound.baselines import linux_stacks as ls
from kvbound.baselines.http_server import HOST
from kvbound.evals import perplexity as ppl
from kvbound.models import load_spec, model_entry
from kvbound.paths import REPO_ROOT
from kvbound.perfmodel.model import Arch, kv_bytes_per_token


def logprob_request(model: str, window: list[int]) -> dict[str, Any]:
    """One window scored by vLLM: its prompt log-probabilities, one token generated."""
    return {
        "model": model,
        "prompt": window,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
        "return_token_ids": True,
    }


def nll_from_prompt_logprobs(window: list[int], entries: list[Any]) -> float:
    """Summed NLL of window[1:] from vLLM's prompt_logprobs: one entry per prompt token, None
    for the first, then a map from token id (a string in JSON) to its log-probability."""
    if len(entries) != len(window):
        raise RuntimeError(f"{len(entries)} prompt logprobs for {len(window)} tokens")
    total = 0.0
    for token, entry in zip(window[1:], entries[1:], strict=True):
        total -= float(entry[str(token)]["logprob"])
    return total


def vllm_scores(
    model: str, weights: str, kv: str, window: int, ids: list[int], spans: list[tuple[int, int]]
) -> tuple[list[float], list[int], float, dict[str, Any]]:
    """Per-window NLL sums and counts, the seconds the loop took and the server's settings."""
    arch = Arch.from_table(load_spec(model).arch)
    tokens = ls.sequence_kv_tokens("vllm", window + 1)
    cfg = ls.ServerConfig(
        max_ctx=window + 1,
        max_seqs=1,
        max_step_tokens=window + 1,
        kv_tokens=tokens,
        kv_bytes=int(tokens * kv_bytes_per_token(arch, ls.KV_VALUE_BYTES[kv])),
        kv=kv,
    )
    settings = ls.server_settings("vllm", model, weights, cfg)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = ls.VLLM.root / "logs" / f"{stamp}-{model}-{weights}-eval.log"
    sums, counts = [], []
    with ls.Server("vllm", settings, log_path) as server:
        started = time.perf_counter()
        for start, end in spans:
            piece = ids[start:end]
            conn = http.client.HTTPConnection(HOST, server.port, timeout=600)
            try:
                body = json.dumps(logprob_request(model, piece)).encode()
                conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
                response = conn.getresponse()
                data = response.read()
            finally:
                conn.close()
            if response.status != 200:
                raise RuntimeError(f"/v1/completions: HTTP {response.status}: {data[:300]!r}")
            entries = json.loads(data)["choices"][0]["prompt_logprobs"]
            sums.append(nll_from_prompt_logprobs(piece, entries))
            counts.append(end - start - 1)
        seconds = time.perf_counter() - started
    return sums, counts, seconds, {"route": "openai-completions prompt_logprobs", **settings}


def trtllm_scores(
    model: str, weights: str, kv: str, window: int, ids: list[int], spans: list[tuple[int, int]]
) -> tuple[list[float], list[int], float, dict[str, Any]]:
    """The same through linux_stacks_worker's nll in TensorRT-LLM's environment."""
    python = ls.require_env(ls.TRTLLM)
    model_dir = ls.require_checkpoint("tensorrt-llm", model, weights)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = ls.TRTLLM.root / "logs" / f"{stamp}-{model}-{weights}-eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        np.save(folder / "ids.npy", np.array(ids, dtype=np.int32))
        np.save(folder / "spans.npy", np.array(spans, dtype=np.int64))
        with log_path.open("wb") as log:
            subprocess.run(
                [str(python), "-E", "-s", "-m", "kvbound.baselines.linux_stacks_worker", "nll",
                 str(folder), "--model-dir", str(model_dir),
                 "--tokenizer", str(load_spec(model).checkpoint_dir), "--window", str(window),
                 "--kv-tokens", str(ls.sequence_kv_tokens("tensorrt-llm", window + 1)),
                 "--kv", ls.KV_TYPES["tensorrt-llm"][kv]],
                check=True, cwd=REPO_ROOT / "python", env=ls.runtime_env(), stdout=log,
                stderr=subprocess.STDOUT,
            )  # fmt: skip
        out = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    settings = {"route": "LLM API context logits", "log": str(log_path), **out["settings"]}
    return out["sums"], out["counts"], out["seconds"], settings


def evaluate(
    stack: str,
    model: str,
    weights: str = "F16",
    kv: str = "f16",
    window: int = 1024,
    max_windows: int = 0,
) -> dict[str, Any]:
    """Perplexity of one stack's checkpoint; returns the arguments of record.write_record."""
    ls.require_env(ls.STACKS[stack])
    spec = load_spec(model)
    ids = ppl.tokenize(ppl.load_text(ppl.fetch_dataset()), spec.checkpoint_dir)
    spans = ppl.split_windows(len(ids), window, max_windows)
    scores = vllm_scores if stack == "vllm" else trtllm_scores
    sums, counts, seconds, settings = scores(model, weights, kv, window, ids, spans)
    return {
        "kind": "eval.perplexity",
        "engine": ls.engine_entry(stack, settings["route"]),
        "model": model_entry(spec),
        "params": {
            **ls.weights_params(stack, model, weights),
            "kv": kv,
            "protocol": "kvbound",
            "stack": settings,
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
    parser.add_argument("--stack", choices=sorted(ls.STACKS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=ls.FORMATS, default="F16")
    parser.add_argument("--kv", choices=sorted(ls.KV_VALUE_BYTES), default="f16")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=0, help="0: every window")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="print the result only")
    args = parser.parse_args()

    if not args.no_record:
        record.require_clean_tree(args.allow_dirty)
    result = evaluate(args.stack, args.model, args.weights, args.kv, args.window, args.max_windows)
    m, p = result["metrics"], result["params"]
    print(
        f"{args.stack} {args.model} {args.weights} kv={args.kv}"
        f" ({p['bits_per_weight']:.3f} bits per weight): perplexity {m['perplexity']:.4f}"
        f" (NLL {m['nll_mean']:.4f} +- {m['nll_stderr']:.4f}) over {p['n_windows']} windows of"
        f" {args.window}, {p['n_predicted']} predicted tokens, {m['seconds']:.1f} s"
    )
    if not args.no_record:
        print(record.write_record(**result))


if __name__ == "__main__":
    main()
