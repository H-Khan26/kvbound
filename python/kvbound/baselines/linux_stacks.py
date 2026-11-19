"""TensorRT-LLM and vLLM baselines: pinned environments, their servers, checkpoints and bits.

Both stacks are Linux-only Python packages whose torch pins conflict with kvbound's, so each
lives in an environment of its own, made by uv from configs/stacks/<env>/requirements.txt
(every package pinned with its hashes, resolved for Python 3.12 on x86_64 Linux). The
harnesses run in kvbound's environment, start the stack's OpenAI-compatible server
(`vllm serve`, `trtllm-serve`) as a child process and talk to it over HTTP on 127.0.0.1, as the
llama.cpp baseline does with llama-server. What needs a stack's Python API (quantization,
TensorRT-LLM's logits) runs as a worker process in that stack's interpreter
(kvbound.baselines.linux_stacks_worker). llm-compressor, the tool vLLM's documentation
quantizes with, pins a different transformers than vLLM and gets an environment of its own;
NVIDIA Model Optimizer comes with TensorRT-LLM.

Everything lives under $KVBOUND_EXT/<env>/<version>/ (kvbound.paths.EXT):

  venv/                           the environment
  logs/                           server and worker logs
  checkpoints/<model>/<format>/   4-bit checkpoints made by the stack's own tooling

Formats: F16 is the fetched Hugging Face checkpoint (data/models/<name>, BF16 on disk) served
in float16, as kvbound runs it. W4A16_AWQ and W4A16_GPTQ are 4-bit weights with FP16
activations, made by kvbound.baselines.linux_stacks_quantize: llm-compressor's checkpoints for
vLLM, Model Optimizer's for TensorRT-LLM (which loads no GPTQ checkpoint; see QUANT_FORMATS).

Under WSL2 records carry the tag os=wsl (kvbound.record._tags) and the WSL kernel in their
fingerprint. Records are written by a clone of the repository inside the Linux file system, at
the commit being measured, since a checkout on /mnt/c seen from Linux differs in file modes;
the record files are then copied into the Windows checkout's results/raw/.

Usage: python -m kvbound.baselines.linux_stacks setup --env vllm
       python -m kvbound.baselines.linux_stacks info --stack vllm --model tinyllama-1.1b
"""

import argparse
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvbound import record
from kvbound.baselines.http_server import HOST, ServerProcess, free_port
from kvbound.models import load_spec
from kvbound.paths import CONFIGS, EXT, REPO_ROOT

PYTHON = "3.12"
# Written into an environment once every package is installed.
DONE_FILE = "kvbound-env.json"


@dataclass(frozen=True)
class Env:
    """A pinned environment: its folder under configs/stacks/ and EXT, the package it is for,
    that package's version, and the indexes beyond PyPI its requirements come from."""

    name: str
    package: str
    version: str
    indexes: tuple[str, ...] = ()

    @property
    def requirements(self) -> Path:
        return CONFIGS / "stacks" / self.name / "requirements.txt"

    @property
    def root(self) -> Path:
        return EXT / self.name / self.version

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    def bin(self, name: str) -> Path:
        return self.venv / "bin" / name


VLLM = Env("vllm", "vllm", "0.34.0")
TRTLLM = Env(
    "tensorrt-llm",
    "tensorrt-llm",
    "1.2.1",
    ("https://download.pytorch.org/whl/cu130", "https://pypi.nvidia.com"),
)
LLMCOMPRESSOR = Env("llmcompressor", "llmcompressor", "0.15.0")
STACKS = {env.name: env for env in (VLLM, TRTLLM)}
ENVS = {env.name: env for env in (VLLM, TRTLLM, LLMCOMPRESSOR)}

FORMATS = ("F16", "W4A16_AWQ", "W4A16_GPTQ")
# The 4-bit formats each stack has a quantization path and kernels for. TensorRT-LLM 1.2's
# PyTorch backend runs Model Optimizer's W4A16 AWQ checkpoints only: its linear layers have no
# method for W4A16_GPTQ (that path was the removed TensorRT engine builder's) and Model
# Optimizer 0.37 has no GPTQ algorithm.
QUANT_FORMATS = {"vllm": ("W4A16_AWQ", "W4A16_GPTQ"), "tensorrt-llm": ("W4A16_AWQ",)}
# The tool that makes each stack's 4-bit checkpoints, by the environment it runs in.
QUANT_TOOL = {"vllm": LLMCOMPRESSOR, "tensorrt-llm": TRTLLM}
# KV cache element types: the stack's name for FP16 (the model's dtype) and for FP8.
KV_TYPES = {"vllm": {"f16": "auto", "fp8": "fp8"}, "tensorrt-llm": {"f16": "auto", "fp8": "fp8"}}
# Bytes of one cached value of each KV type.
KV_VALUE_BYTES = {"f16": 2.0, "fp8": 1.0}
# The share of the card vLLM requires free at start-up (see vllm_settings).
GPU_MEMORY_UTILIZATION = 0.6
# Tokens per KV block: vLLM's default for these models and TensorRT-LLM's tokens_per_block.
KV_BLOCK_TOKENS = {"vllm": 16, "tensorrt-llm": 32}

# Lines of each server's log worth keeping in a record: memory and the KV cache it built.
LOG_FACTS = {
    "vllm": re.compile(
        r"Model loading took .*|Available KV cache memory: .*|GPU KV cache size: .*"
        r"|Maximum concurrency for .*|Graph capturing finished .*|Using .* backend.*"
        r"|Memory profiling takes .*"
    ),
    "tensorrt-llm": re.compile(
        r"\[MemUsageChange\].*|.*[Pp]aged KV cache.*|.*max_tokens.*|.*[Aa]ttention backend.*"
        r"|.*cuda graph.*batch sizes.*"
    ),
}


def require_linux() -> None:
    if platform.system() != "Linux":
        raise SystemExit("TensorRT-LLM and vLLM run on Linux only; under Windows, use WSL2")


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(env: Env) -> Path:
    """The environment's interpreter, made with uv from the hash-pinned requirements when it is
    missing or was made from other requirements."""
    require_linux()
    python = env.bin("python")
    digest = sha256_text(env.requirements)
    done = env.venv / DONE_FILE
    if done.exists() and json.loads(done.read_text(encoding="utf-8"))["sha256"] == digest:
        return python
    # --clear replaces an environment whose install did not finish.
    subprocess.run(["uv", "venv", "--clear", "--python", PYTHON, str(env.venv)], check=True)
    indexes = [arg for url in env.indexes for arg in ("--extra-index-url", url)]
    if indexes:
        indexes += ["--index-strategy", "unsafe-best-match"]
    subprocess.run(
        ["uv", "pip", "sync", "--python", str(python), "--require-hashes", *indexes,
         str(env.requirements)],
        check=True, cwd=REPO_ROOT,
    )  # fmt: skip
    stamp = {"requirements": str(env.requirements.relative_to(REPO_ROOT)), "sha256": digest}
    done.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return python


def require_env(env: Env) -> Path:
    """The environment's interpreter; exits with a hint when it was not set up from the
    current requirements."""
    require_linux()
    done = env.venv / DONE_FILE
    if not done.exists():
        raise SystemExit(f"{env.venv} not set up; run `make setup-stacks`")
    if json.loads(done.read_text(encoding="utf-8"))["sha256"] != sha256_text(env.requirements):
        raise SystemExit(f"{env.requirements} changed since {env.venv} was made; run setup again")
    return env.bin("python")


def engine_entry(stack: str, route: str) -> dict[str, Any]:
    """The record's engine entry: the stack's version, the route the numbers took, and the
    requirements file (with its sha256) that pins every package of its environment."""
    env = STACKS[stack]
    return record.external_engine(
        stack,
        env.version,
        route=route,
        requirements=str(env.requirements.relative_to(REPO_ROOT)).replace("\\", "/"),
        requirements_sha256=sha256_text(env.requirements),
    )


def checkpoint_dir(stack: str, model: str, fmt: str) -> Path:
    if fmt == "F16":
        return load_spec(model).checkpoint_dir
    return STACKS[stack].root / "checkpoints" / model / fmt


def require_checkpoint(stack: str, model: str, fmt: str) -> Path:
    if fmt != "F16" and fmt not in QUANT_FORMATS[stack]:
        raise SystemExit(
            f"{stack} has no {fmt} path here; its 4-bit formats: {QUANT_FORMATS[stack]}"
        )
    path = checkpoint_dir(stack, model, fmt)
    if not any(path.glob("*.safetensors")):
        hint = "make fetch" if fmt == "F16" else "make quantize-stacks"
        raise SystemExit(f"{path} holds no safetensors; run `{hint}` for {model}")
    return path


# Bytes per element of each safetensors dtype.
SAFETENSORS_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1,
}  # fmt: skip


def safetensors_tensors(folder: Path) -> dict[str, dict[str, Any]]:
    """Every tensor of a folder's safetensors files, by name: dtype, shape and stored bytes,
    read from the files' headers."""
    tensors = {}
    for path in sorted(folder.glob("*.safetensors")):
        with path.open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            tensors[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "bytes": end - start}
    return tensors


def bits_per_weight(tensors: dict[str, dict[str, Any]], n_weights: int) -> dict[str, Any]:
    """8 * the bytes of every stored tensor (packed weights, scales, zero points, norms and
    embeddings alike) / the model's number of weights, n_weights, counted from its FP16
    checkpoint. A quantized file's element counts do not give the weights: packed integers hold
    several each."""
    stored = sum(t["bytes"] for t in tensors.values())
    by_dtype: dict[str, int] = {}
    for t in tensors.values():
        by_dtype[t["dtype"]] = by_dtype.get(t["dtype"], 0) + t["bytes"]
    return {
        "bits_per_weight": 8 * stored / n_weights,
        "weights": n_weights,
        "tensor_bytes": stored,
        "bytes_by_dtype": dict(sorted(by_dtype.items())),
    }


def n_weights(model: str) -> int:
    """Weights of a model: the elements of every tensor of its fetched checkpoint."""
    tensors = safetensors_tensors(load_spec(model).checkpoint_dir)
    total = 0
    for t in tensors.values():
        n = 1
        for d in t["shape"]:
            n *= d
        total += n
    return total


def weights_params(stack: str, model: str, fmt: str) -> dict[str, Any]:
    """The weight entries of a record's params for one checkpoint. F16 counts 16 bits: the
    BF16 file is served in float16."""
    folder = require_checkpoint(stack, model, fmt)
    count = n_weights(model)
    if fmt == "F16":
        info: dict[str, Any] = {"bits_per_weight": 16.0, "weights": count}
    else:
        info = bits_per_weight(safetensors_tensors(folder), count)
    out = {"weights": fmt, "bits_per_weight": info["bits_per_weight"], "checkpoint": info}
    out["checkpoint"]["folder"] = str(folder)
    quant_file = folder / "quantization.json"
    if quant_file.exists():  # written by linux_stacks_quantize
        out["checkpoint"]["quantization"] = json.loads(quant_file.read_text(encoding="utf-8"))
    return out


def round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


@dataclass(frozen=True)
class ServerConfig:
    """What sizes a server, in kvbound's terms."""

    max_ctx: int  # longest sequence, prompt plus output
    max_seqs: int  # sequences processed together
    max_step_tokens: int  # tokens one step may process (prefill chunks included)
    kv_tokens: int  # tokens the KV cache holds
    kv_bytes: int  # bytes of those tokens at kv
    kv: str = "f16"  # KV cache element type, a key of KV_VALUE_BYTES
    chunked_prefill: bool = True
    # Batch sizes TensorRT-LLM captures CUDA graphs for; vLLM picks its own from max_seqs.
    cuda_graph_batch_sizes: tuple[int, ...] = (1,)


def vllm_settings(model_dir: Path, tokenizer: Path, model: str, cfg: ServerConfig) -> dict:
    """Every `vllm serve` flag that changes the numbers, by its flag."""
    return {
        "model": str(model_dir),
        "--tokenizer": str(tokenizer),
        "--served-model-name": model,
        "--dtype": "float16",
        "--max-model-len": cfg.max_ctx,
        "--max-num-seqs": cfg.max_seqs,
        "--max-num-batched-tokens": cfg.max_step_tokens,
        "--kv-cache-memory-bytes": cfg.kv_bytes,
        # With the KV bytes given, the utilization only gates start-up: vLLM refuses to start
        # unless this share of the card is free, and the desktop holds part of it.
        "--gpu-memory-utilization": GPU_MEMORY_UTILIZATION,
        "--kv-cache-dtype": KV_TYPES["vllm"][cfg.kv],
        "--block-size": KV_BLOCK_TOKENS["vllm"],
        "--disable-prefix-caching": True,
        "--enable-chunked-prefill" if cfg.chunked_prefill else "--disable-chunked-prefill": True,
        # Sampling defaults come from requests only, not from a generation_config.json.
        "--generation-config": "vllm",
        "--seed": 0,
        "--disable-uvicorn-access-log": True,
    }


def trtllm_settings(model_dir: Path, tokenizer: Path, model: str, cfg: ServerConfig) -> dict:
    """Every trtllm-serve option that changes the numbers: its flags, and under
    "extra_llm_api_options" the LLM API arguments it reads from a YAML file."""
    return {
        "model": str(model_dir),
        "--tokenizer": str(tokenizer),
        "--backend": "pytorch",
        "--max_batch_size": cfg.max_seqs,
        "--max_num_tokens": cfg.max_step_tokens,
        "--max_seq_len": cfg.max_ctx,
        "extra_llm_api_options": {
            "dtype": "float16",
            "kv_cache_config": {
                "max_tokens": cfg.kv_tokens,
                "dtype": KV_TYPES["tensorrt-llm"][cfg.kv],
                "enable_block_reuse": False,
                "tokens_per_block": KV_BLOCK_TOKENS["tensorrt-llm"],
            },
            "enable_chunked_prefill": cfg.chunked_prefill,
            "cuda_graph_config": {"batch_sizes": list(cfg.cuda_graph_batch_sizes)},
            # Per-request server timings at /perf_metrics, per-iteration stats at /metrics.
            "return_perf_metrics": True,
            "perf_metrics_max_requests": 10000,
            "enable_iter_perf_stats": True,
        },
    }


def server_settings(stack: str, model: str, fmt: str, cfg: ServerConfig) -> dict[str, Any]:
    model_dir = require_checkpoint(stack, model, fmt)
    tokenizer = load_spec(model).checkpoint_dir
    if stack == "vllm":
        return vllm_settings(model_dir, tokenizer, model, cfg)
    return trtllm_settings(model_dir, tokenizer, model, cfg)


def command_line(stack: str, settings: dict[str, Any], options_file: Path, port: int) -> list[str]:
    """The server's command line from its settings; TensorRT-LLM's LLM API arguments go into
    options_file (JSON, which is YAML)."""
    env = STACKS[stack]
    if stack == "vllm":
        command = [str(env.bin("vllm")), "serve", settings["model"]]
    else:
        command = [str(env.bin("trtllm-serve")), settings["model"]]
    for flag, value in settings.items():
        if flag == "model":
            continue
        if flag == "extra_llm_api_options":
            options_file.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
            command += ["--extra_llm_api_options", str(options_file)]
        elif value is True:
            command.append(flag)
        else:
            command += [flag, str(value)]
    return [*command, "--host", HOST, "--port", str(port)]


def runtime_env() -> dict[str, str]:
    """Environment of a stack's processes: offline (the checkpoints are local), no usage
    reporting."""
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "TRTLLM_NO_USAGE_STATS": "1",
    }


class Server(ServerProcess):
    """A stack's OpenAI-compatible server, started and loaded on enter, ended on exit."""

    def __init__(self, stack: str, settings: dict[str, Any], log_path: Path) -> None:
        port = free_port()
        options = log_path.with_suffix(".yaml")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = command_line(stack, settings, options, port)
        # Loading includes torch.compile and CUDA graph capture on the first start.
        super().__init__(command, log_path, port, name=stack, env=runtime_env(), load_timeout_s=900)
        self.stack = stack


def log_facts(stack: str, log: str) -> list[str]:
    """The lines of a server log that LOG_FACTS keeps, in order, without repeats."""
    seen: dict[str, None] = {}
    for match in LOG_FACTS[stack].finditer(log):
        seen.setdefault(match.group(0).strip(), None)
    return list(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup", help="make a pinned environment")
    p.add_argument("--env", choices=sorted(ENVS), nargs="+", default=sorted(ENVS))
    p = sub.add_parser("info", help="print the bits per weight of a stack's checkpoints")
    p.add_argument("--stack", choices=sorted(STACKS), required=True)
    p.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "setup":
        for name in args.env:
            print(f"{name}: {setup(ENVS[name])}")
    else:
        for fmt in ("F16", *QUANT_FORMATS[args.stack]):
            if any(checkpoint_dir(args.stack, args.model, fmt).glob("*.safetensors")):
                info = weights_params(args.stack, args.model, fmt)
                print(json.dumps({"model": args.model, **info}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
