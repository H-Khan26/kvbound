"""Work that needs a stack's Python API, run in that stack's own environment.

The harnesses (linux_stacks_eval, linux_stacks_quantize) run in kvbound's environment and start
this module with the stack's interpreter, the repository's python/ folder as the working folder
so that `-m kvbound.baselines.linux_stacks_worker` resolves. It imports nothing else of kvbound,
since the stack's environment holds neither kvbound's dependencies nor its torch, and nothing at
module level but numpy. Inputs and outputs are files in a folder the harness names.

  nll        TensorRT-LLM: each window of ids.npy (spans.npy) fed from position 0 as one prompt
             through the LLM API with its context logits returned, scored by window_nll;
             writes result.json (per-window NLL sums and counts, seconds, settings).
  quantize   llm-compressor (vLLM's checkpoints) or Model Optimizer (TensorRT-LLM's): 4-bit
             weights calibrated on calib.npy, written to --out; writes worker.json (the tool's
             version and the recipe).

Usage (by the harnesses): python -m kvbound.baselines.linux_stacks_worker nll <folder> ...
"""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# Calibration windows per forward pass of Model Optimizer's calibration loop.
CALIB_BATCH = 8
# Rows of a window's logits scored together, which bounds the float32 copies.
SCORE_ROWS = 256


def window_nll(logits: Any, ids: list[int]) -> float:
    """Summed NLL in nats of ids[1:] under logits[:-1] (a torch tensor [n, vocab]):
    kvbound.evals.perplexity.window_nll's arithmetic, logsumexp in float32 and the sum in
    float64, on the logits' device."""
    import torch  # noqa: PLC0415

    targets = torch.tensor(ids[1:], dtype=torch.int64, device=logits.device)
    total = torch.zeros((), dtype=torch.float64, device=logits.device)
    for start in range(0, len(ids) - 1, SCORE_ROWS):
        rows = logits[start : min(start + SCORE_ROWS, len(ids) - 1)].float()
        picked = targets[start : start + len(rows)]
        nll = torch.logsumexp(rows, dim=-1) - rows.gather(-1, picked[:, None])[:, 0]
        total += nll.double().sum()
    return float(total.item())


def nll(args: argparse.Namespace) -> None:
    # The executor then runs in this process, so a window's context logits reach the scoring
    # as a tensor instead of being serialized from a worker process (hundreds of MB a window).
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    from tensorrt_llm import LLM, SamplingParams  # noqa: PLC0415
    from tensorrt_llm.llmapi import KvCacheConfig  # noqa: PLC0415

    ids = np.load(args.folder / "ids.npy")
    spans = np.load(args.folder / "spans.npy")
    kv_tokens = args.kv_tokens
    settings: dict[str, Any] = {
        "dtype": "float16",
        "max_batch_size": 1,
        "max_seq_len": args.window + 1,
        "max_num_tokens": args.window + 1,
        "kv_cache_config": {"max_tokens": kv_tokens, "dtype": args.kv, "enable_block_reuse": False},
        "cuda_graph_config": None,
    }
    llm = LLM(
        model=str(args.model_dir),
        tokenizer=str(args.tokenizer),
        dtype="float16",
        max_batch_size=1,
        max_seq_len=args.window + 1,
        max_num_tokens=args.window + 1,
        kv_cache_config=KvCacheConfig(
            max_tokens=kv_tokens, dtype=args.kv, enable_block_reuse=False
        ),
        cuda_graph_config=None,
    )
    params = SamplingParams(max_tokens=1, temperature=0.0, top_k=1, return_context_logits=True)
    sums, counts = [], []
    started = time.perf_counter()
    try:
        for start, end in spans.tolist():
            window = ids[start:end]
            out = llm.generate([window.tolist()], params, use_tqdm=False)[0]
            logits = out.context_logits
            if logits is None or logits.shape[0] != len(window):
                shape = None if logits is None else tuple(logits.shape)
                raise RuntimeError(f"context logits of shape {shape} for {len(window)} tokens")
            sums.append(window_nll(logits.to("cuda"), window.tolist()))
            counts.append(end - start - 1)
    finally:
        llm.shutdown()
    seconds = time.perf_counter() - started
    result = {"sums": sums, "counts": counts, "seconds": seconds, "settings": settings}
    (args.folder / "result.json").write_text(json.dumps(result), encoding="utf-8")


def quantize_llmcompressor(model: Any, calib: np.ndarray, fmt: str, out: Path) -> dict[str, Any]:
    """vLLM's checkpoint: llm-compressor's oneshot with its W4A16 preset (symmetric INT4,
    groups of 128, the model's dtype for scales), lm_head left in FP16 as its examples do."""
    from datasets import Dataset  # noqa: PLC0415
    from llmcompressor import oneshot  # noqa: PLC0415
    from llmcompressor.modifiers.gptq import GPTQModifier  # noqa: PLC0415
    from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: PLC0415
    from llmcompressor.modifiers.transform.awq import AWQModifier  # noqa: PLC0415

    scheme = {"targets": ["Linear"], "scheme": "W4A16", "ignore": ["lm_head"]}
    if fmt == "W4A16_AWQ":
        recipe: list[Any] = [AWQModifier(), QuantizationModifier(**scheme)]
    else:
        recipe = [GPTQModifier(**scheme)]
    dataset = Dataset.from_dict(
        {"input_ids": calib.tolist(), "attention_mask": np.ones_like(calib).tolist()}
    )
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=calib.shape[1],
        num_calibration_samples=calib.shape[0],
        shuffle_calibration_samples=False,
    )
    model.save_pretrained(str(out), save_compressed=True)
    return {
        "tool": "llmcompressor",
        "version": importlib.metadata.version("llmcompressor"),
        "compressed_tensors": importlib.metadata.version("compressed-tensors"),
        "recipe": [type(m).__name__ for m in recipe],
        "scheme": scheme,
    }


def quantize_modelopt(model: Any, calib: np.ndarray, out: Path, device: str) -> dict[str, Any]:
    """TensorRT-LLM's checkpoint: Model Optimizer's INT4_AWQ_CFG (INT4 groups of 128, its
    awq_lite search), exported as a Hugging Face checkpoint with hf_quant_config.json."""
    import modelopt.torch.quantization as mtq  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from modelopt.torch.export import export_hf_checkpoint  # noqa: PLC0415

    batches = torch.from_numpy(calib.astype(np.int64))

    def forward_loop(m: Any) -> None:
        with torch.no_grad():
            for i in range(0, len(batches), CALIB_BATCH):
                m(batches[i : i + CALIB_BATCH].to(device))

    mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop)
    export_hf_checkpoint(model, export_dir=str(out))
    return {
        "tool": "nvidia-modelopt",
        "version": importlib.metadata.version("nvidia-modelopt"),
        "config": "INT4_AWQ_CFG",
        "algorithm": json.loads(json.dumps(mtq.INT4_AWQ_CFG["algorithm"])),
    }


def quantize(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    torch.manual_seed(0)
    calib = np.load(args.folder / "calib.npy")
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), torch_dtype=torch.float16)
    started = time.perf_counter()
    if args.tool == "llmcompressor":
        info = quantize_llmcompressor(model, calib, args.format, args.out)
    else:
        info = quantize_modelopt(model.to(args.device), calib, args.out, args.device)
    info |= {
        "seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "device": args.device,
    }
    (args.folder / "worker.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("nll")
    p.add_argument("folder", type=Path)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--kv-tokens", type=int, required=True, help="KV cache size in tokens")
    p.add_argument("--kv", default="auto")
    p = sub.add_parser("quantize")
    p.add_argument("folder", type=Path)
    p.add_argument("--tool", choices=["llmcompressor", "modelopt"], required=True)
    p.add_argument("--format", choices=["W4A16_AWQ", "W4A16_GPTQ"], required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "nll":
        nll(args)
    else:
        quantize(args)


if __name__ == "__main__":
    main()
