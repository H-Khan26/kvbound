"""Benchmarks the kvbound engine and records the result.

kvb-bench (C++) runs a trace of requests and prints raw token timestamps. This wrapper builds the
trace, samples NVML while the binary runs, turns the timestamps into the metrics defined in
kvbound.record, compares decode speed with the perf model's ceiling, and writes the record.

The prompt is the same seeded random token ids the Hugging Face baseline uses, so both engines
process identical tokens.

--weights picks a quantized variant of the checkpoint (see kvbound.quantize) and --kv the
element type of the KV cache; the ceiling is the perf model's for that precision.

Usage: python -m kvbound.bench --model tinyllama-1.1b --prompt-len 128 --new-tokens 128
"""

import argparse
import itertools
import json
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from kvbound import fingerprint, record
from kvbound.models import load_spec
from kvbound.nvml_sampler import NvmlSampler
from kvbound.paths import binary
from kvbound.perfmodel import hardware
from kvbound.perfmodel.model import Arch, Precision, predict
from kvbound.weight_format import DEFAULT_GROUP_SIZE, VARIANTS

KV_DTYPES = ("fp16", "int8")


def checkpoint_dir(model: str, weights: str) -> Path:
    """The directory of a model's weight variant; exits with a hint when it was not made."""
    path = load_spec(model).variant_dir(weights)
    if not (path / "model.safetensors").exists():
        raise SystemExit(f"{path} not found; run `make fetch` and `make quantize` for {model}")
    return path


def random_prompt(vocab_size: int, prompt_len: int, seed: int) -> list[int]:
    """The first row of the Hugging Face baseline's seeded random batch."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(vocab_size, (1, prompt_len), generator=gen)[0].tolist()


def run_kvb_bench(
    preset: str, model_dir: Path, trace: dict[str, Any], options: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs kvb-bench on a trace; returns its raw JSON and the NVML summary of the run."""
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        command = [
            str(binary("bench", preset)),
            "--model",
            str(model_dir),
            "--trace",
            str(trace_path),
        ]
        for name, value in options.items():
            command += [f"--{name}", str(int(value) if isinstance(value, bool) else value)]
        with NvmlSampler() as sampler:
            done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"kvb-bench failed:\n{done.stderr}")
    return json.loads(done.stdout), sampler.summary()


def request_metrics(token_times: list[float]) -> dict[str, float]:
    """Metrics of one request from the times its tokens became available (see kvbound.record)."""
    first, last = token_times[0], token_times[-1]
    steps = [b - a for a, b in itertools.pairwise(token_times)]
    tpot = (last - first) / (len(token_times) - 1)
    return {
        "ttft_s": first,
        "tpot_s": tpot,
        "decode_tok_s": 1 / tpot,
        "e2e_tok_s": len(token_times) / last,
        # The median step is not moved by the occasional step that another GPU client delays.
        "decode_step_p50_s": statistics.median(steps),
    }


def op_profile_ms(raw: dict[str, Any]) -> dict[str, Any]:
    """Milliseconds per step spent in each op kind, from kvb-bench's synchronized timing pass."""
    profile = {}
    for phase, data in raw["op_profile"].items():
        steps = data["steps"]
        if steps == 0:
            continue  # a trace of one-token outputs has no decode step
        profile[phase] = {
            "steps": steps,
            "ms_per_step": {
                op: {"ms": 1e3 * t["seconds"] / steps, "calls": t["calls"] / steps}
                for op, t in data["ops"].items()
            },
        }
    return profile


def measured_ceilings() -> hardware.Ceilings | None:
    """Ceilings of this machine's GPU, or None when it has no microbenchmark records yet."""
    config = fingerprint.device_config(fingerprint.gpu_info()["name"])
    if config is None:
        return None
    if {"bandwidth_sweep", "gemm_peak"} - set(hardware.latest_records(config[0])):
        return None
    return hardware.load(config[0])


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.model)
    arch = Arch.from_table(spec.arch)
    max_ctx = args.max_ctx or args.prompt_len + args.new_tokens
    trace = {
        "requests": [
            {
                "id": 0,
                "prompt": random_prompt(arch.vocab_size, args.prompt_len, args.seed),
                "max_new_tokens": args.new_tokens,
            }
        ]
    }
    options = {
        "max-ctx": max_ctx,
        "graph": args.graph,
        "kv-layout": args.kv_layout,
        "kv-dtype": args.kv,
        "gemm": args.gemm,
        "fuse": args.fuse,
        "f16-gemv-rows": args.f16_gemv_rows,
        "expand-quantized": args.expand_quantized,
        "warmup": args.warmup,
        "iters": args.iters,
        "profile-ops": args.profile_ops,
    }
    hw = measured_ceilings() if args.gemm == "fp32" else None
    model_dir = checkpoint_dir(args.model, args.weights)
    raw, nvml = run_kvb_bench(args.preset, model_dir, trace, options)

    runs = [request_metrics(it["requests"][0]["token_times_s"]) for it in raw["iterations"]]
    samples = {name: [run[name] for run in runs] for name in runs[0]}
    metrics: dict[str, Any] = {name: record.summarize(values) for name, values in samples.items()}
    metrics["memory"] = raw["memory"]
    metrics["load_s"] = raw["load_s"]
    metrics["launches_per_step"] = raw["program"]["ops"]
    if "op_profile" in raw:
        metrics["op_profile"] = op_profile_ms(raw)

    # The ceiling for this decode: the perf model's bound at the middle of the generated span.
    if hw is not None:
        context = args.prompt_len + args.new_tokens // 2
        group_size = raw["config"]["weight_group_size"] or DEFAULT_GROUP_SIZE
        precision = Precision.named(args.weights, args.kv, arch.head_dim, group_size)
        bound = predict(arch, hw, precision, 1, args.prompt_len, context)
        metrics["ceiling"] = {
            "dram_gbs": hw.dram_gbs,
            "decode_tok_s": bound.decode_tok_s,
            "prefill_s": bound.prefill_s,
            "decode_fraction": metrics["decode_tok_s"]["p50"] / bound.decode_tok_s,
        }

    return {
        "kind": "generate",
        "engine": {"name": "kvbound", "preset": args.preset},
        "model": {"name": spec.name, "hf_repo": spec.hf_repo, "revision": spec.revision},
        "params": {
            "batch": 1,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "seed": args.seed,
            "sampling": "greedy",
            **raw["config"],
            "weights": args.weights,  # the variant name; the binary knows only the scheme
        },
        "metrics": metrics,
        "samples": {**samples, "tokens": raw["iterations"][0]["requests"][0]["tokens"]},
        "nvml": nvml,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--max-ctx", type=int, default=0, help="default: prompt + new tokens")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kv-layout", default="head_major")
    parser.add_argument("--weights", choices=VARIANTS, default="fp16")
    parser.add_argument("--kv", choices=KV_DTYPES, default="fp16")
    parser.add_argument("--gemm", choices=["fp32", "fp16"], default="fp32")
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
    parser.add_argument("--profile-ops", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="cuda")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error("--new-tokens must be at least 2 to measure TPOT")

    record.require_clean_tree(args.allow_dirty)
    result = benchmark(args)
    path = record.write_record(**result)
    m = result["metrics"]
    ceiling = m.get("ceiling")
    print(
        f"TTFT p50 {m['ttft_s']['p50'] * 1e3:.1f} ms | TPOT p50 {m['tpot_s']['p50'] * 1e3:.2f} ms"
        f" | decode {m['decode_tok_s']['p50']:.1f} tok/s"
        + (f" ({ceiling['decode_fraction']:.0%} of the ceiling)" if ceiling else "")
        + f"\n{path}"
    )


if __name__ == "__main__":
    main()
