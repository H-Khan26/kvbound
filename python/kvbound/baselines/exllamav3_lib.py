"""ExLlamaV3 worker: perplexity logits and timed greedy generation in the pinned environment.

Runs in kvbound.baselines.exllamav3's environment, not the repository's: it imports torch,
exllamav3 and, from the repository, only kvbound.baselines.llamacpp_lib's scoring rule, which
needs numpy alone. The repository's side (exllamav3_eval, exllamav3_bench) writes a request into
a folder, starts this module with that folder, and reads result.json back.

  ppl       every window of ids.npy (spans.npy) goes through Model.forward from position 0 with
            no cache (attn_mode flash_attn_nc), every position's logits returned; the logits
            are scored by llamacpp_lib.window_nll, kvbound's rule in float64. tokens.npy holds
            ExLlamaV3's own tokenization of text.txt, BOS added.
  generate  the prompt of request.json goes through the Generator as one job, greedy
            (ArgmaxSampler), no stop condition, the page table reset before each job so that
            no prompt page is reused; each token is timed when iterate() hands it over, from
            just before the job is enqueued.

Usage (by exllamav3_eval and exllamav3_bench):
    python -m kvbound.baselines.exllamav3_lib ppl <folder> --model-dir <dir> --window 1024
    python -m kvbound.baselines.exllamav3_lib generate <folder> --model-dir <dir>
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from exllamav3 import ArgmaxSampler, Cache, Config, Generator, Job, Model, Tokenizer

from kvbound.baselines.llamacpp_lib import window_nll

DEVICE = "cuda:0"


def load(model_dir: Path, chunk: int, cache_tokens: int = 0) -> tuple[Model, Config, Cache | None]:
    """The model on one GPU, with a cache of cache_tokens positions when that is not 0."""
    config = Config.from_directory(str(model_dir))
    model = Model.from_config(config)
    # A cache must exist before the model loads, which allocates its tensors.
    cache = Cache(model, max_num_tokens=cache_tokens) if cache_tokens else None
    model.load(device=DEVICE, max_chunk_size=chunk, max_output_size=chunk)
    return model, config, cache


def storage(model: Model) -> dict[str, float]:
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()
    return {"bpw_layers": bpw_layer, "bpw_head": bpw_head, "vram_bytes": vram_bits / 8}


def memory() -> dict[str, float]:
    return {
        "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def perplexity(folder: Path, model_dir: Path, window: int) -> dict[str, Any]:
    ids = np.load(folder / "ids.npy")
    spans = np.load(folder / "spans.npy")
    text = (folder / "text.txt").read_text(encoding="utf-8")
    model, config, _ = load(model_dir, window)
    vocab = config.vocab_size
    tokenizer = Tokenizer.from_config(config)
    tokens = tokenizer.encode(text, add_bos=True)[0].numpy().astype(np.int32)
    np.save(folder / "tokens.npy", tokens)
    sums, counts = [], []
    logits_dtype = ""
    started = time.perf_counter()
    with torch.inference_mode():
        for start, end in spans.tolist():
            window_ids = ids[start:end]
            x = torch.from_numpy(window_ids.astype(np.int64))[None]
            logits = model.forward(x, {"attn_mode": "flash_attn_nc"})[0]
            if logits.shape[-1] < vocab:
                raise RuntimeError(f"logits have {logits.shape[-1]} columns, vocabulary {vocab}")
            logits_dtype = str(logits.dtype)
            rows = logits[:, :vocab].float().cpu().numpy()
            sums.append(window_nll(rows, window_ids))
            counts.append(end - start - 1)
    seconds = time.perf_counter() - started
    settings = {
        "device": DEVICE,
        "attn_mode": "flash_attn_nc",
        "max_chunk_size": window,
        "logits_dtype": logits_dtype,
        "storage": storage(model),
        "torch_memory": memory(),
    }
    return {"sums": sums, "counts": counts, "seconds": seconds, "settings": settings}


def run_job(generator: Generator, prompt: torch.Tensor, new_tokens: int) -> dict[str, Any]:
    """One greedy job from an emptied page table; token times count from just before enqueue."""
    generator.pagetable.reset_page_table()
    job = Job(input_ids=prompt, max_new_tokens=new_tokens, sampler=ArgmaxSampler())
    times: list[float] = []
    tokens: list[int] = []
    multi = held = 0
    final: dict[str, Any] = {}
    torch.cuda.synchronize()
    origin = time.perf_counter()
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] != "streaming":
                continue
            now = time.perf_counter() - origin
            ids = r["token_ids"][0].tolist() if "token_ids" in r else []
            times += [now] * len(ids)
            tokens += ids
            multi += len(ids) > 1
            if r["eos"]:
                # Tokens still held back at the end arrive with the closing result.
                rest = r.get("held", {}).get("token_ids")
                if rest is not None:
                    times += [now] * rest.shape[-1]
                    tokens += rest[0].tolist()
                    held += rest.shape[-1]
                final = {
                    key: r[key]
                    for key in ("eos_reason", "new_tokens", "prompt_tokens", "time_enqueued",
                                "time_prefill", "time_generate", "cached_pages", "cached_tokens")
                }  # fmt: skip
    return {
        "times": times,
        "tokens": tokens,
        "final": final,
        "multi_token_events": multi,
        "held_tokens": held,
    }


def generate(folder: Path, model_dir: Path) -> dict[str, Any]:
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    chunk, cache_tokens = request["max_chunk_size"], request["cache_tokens"]
    model, config, cache = load(model_dir, chunk, cache_tokens)
    assert cache is not None
    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=Tokenizer.from_config(config),
        max_batch_size=1,
        max_chunk_size=chunk,
    )
    prompt = torch.tensor([request["prompt"]], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(request["warmup"]):
            run_job(generator, prompt, request["new_tokens"])
        torch.cuda.reset_peak_memory_stats()
        runs = [run_job(generator, prompt, request["new_tokens"]) for _ in range(request["iters"])]
    settings = {
        "device": DEVICE,
        "cache_tokens": cache_tokens,
        "cache_layer": "CacheLayer_fp16",
        "max_batch_size": 1,
        "max_chunk_size": chunk,
        "sampler": "ArgmaxSampler",
        "prefix_cache": "off (page table reset before each job)",
        "storage": storage(model),
        "torch_memory": memory(),
    }
    return {"runs": runs, "settings": settings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ppl", "generate"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1024)
    args = parser.parse_args()
    if args.command == "ppl":
        result = perplexity(args.folder, args.model_dir, args.window)
    else:
        result = generate(args.folder, args.model_dir)
    (args.folder / "result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
