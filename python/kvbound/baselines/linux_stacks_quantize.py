"""4-bit weight checkpoints for vLLM and TensorRT-LLM, made with each stack's own tooling.

  vLLM          llm-compressor (its own environment): W4A16_AWQ is AWQModifier's scales then
                QuantizationModifier's W4A16 preset, W4A16_GPTQ is GPTQModifier's W4A16
                preset; symmetric INT4 in groups of 128 with the model's dtype for scales,
                lm_head kept in FP16, saved as a compressed-tensors checkpoint.
  TensorRT-LLM  NVIDIA Model Optimizer (in TensorRT-LLM's environment): INT4_AWQ_CFG, INT4 in
                groups of 128 with its awq_lite search, exported as a Hugging Face checkpoint
                with hf_quant_config.json. No GPTQ: see linux_stacks.QUANT_FORMATS.

Every tool calibrates on the tokens kvbound's own AWQ and GPTQ use
(kvbound.awq.calibration_windows: 128 windows of 512 tokens of the WikiText-2 train split,
seeded), so the comparison is between methods, not calibration sets. Embeddings stay FP16 in
every stack. The checkpoint goes to $KVBOUND_EXT/<stack>/<version>/checkpoints/<model>/<format>/
with quantization.json beside it: the tool and its version, the recipe, the calibration's
source, and the seconds it took.

Usage: python -m kvbound.baselines.linux_stacks_quantize --stack vllm --model tinyllama-1.1b \
           --format W4A16_GPTQ
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from kvbound.awq import calibration_meta, calibration_windows
from kvbound.baselines import linux_stacks as ls
from kvbound.models import CHECKPOINT_FILES, load_spec
from kvbound.paths import REPO_ROOT

TOOLS = {"vllm": "llmcompressor", "tensorrt-llm": "modelopt"}
# The tokenizer files a checkpoint folder needs to be served on its own.
TOKENIZER_FILES = [name for name in CHECKPOINT_FILES if name.startswith("tokenizer")]


def quantize(stack: str, model: str, fmt: str, device: str = "cuda") -> Path:
    if fmt not in ls.QUANT_FORMATS[stack]:
        raise SystemExit(f"{stack} has no {fmt} path; its 4-bit formats: {ls.QUANT_FORMATS[stack]}")
    tool_env = ls.QUANT_TOOL[stack]
    python = ls.require_env(tool_env)
    spec = load_spec(model)
    out = ls.checkpoint_dir(stack, model, fmt)
    if any(out.glob("*.safetensors")):
        raise SystemExit(f"{out} exists; remove it to quantize again")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # a folder this function made and did not finish
    partial.mkdir(parents=True)
    log_path = ls.STACKS[stack].root / "logs" / f"quantize-{model}-{fmt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = ls.runtime_env()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        np.save(folder / "calib.npy", calibration_windows(spec).numpy())
        with log_path.open("wb") as log:
            subprocess.run(
                [str(python), "-E", "-s", "-m", "kvbound.baselines.linux_stacks_worker",
                 "quantize", str(folder), "--tool", TOOLS[stack], "--format", fmt,
                 "--model-dir", str(spec.checkpoint_dir), "--out", str(partial),
                 "--device", device],
                check=True, cwd=REPO_ROOT / "python", env=env, stdout=log,
                stderr=subprocess.STDOUT,
            )  # fmt: skip
        worker = json.loads((folder / "worker.json").read_text(encoding="utf-8"))
    for name in TOKENIZER_FILES:
        if not (partial / name).exists():
            shutil.copyfile(spec.checkpoint_dir / name, partial / name)
    info = {
        "format": fmt,
        "source": {"hf_repo": spec.hf_repo, "revision": spec.revision},
        **worker,
        "calibration": calibration_meta(),
        "log": str(log_path),
    }
    (partial / "quantization.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    partial.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=sorted(ls.STACKS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", choices=["W4A16_AWQ", "W4A16_GPTQ"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    out = quantize(args.stack, args.model, args.format, args.device)
    params = ls.weights_params(args.stack, args.model, args.format)
    print(f"{out}: {params['bits_per_weight']:.3f} bits per weight")


if __name__ == "__main__":
    main()
