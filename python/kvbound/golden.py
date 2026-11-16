"""Golden tensors: FP32 per-site activations and logits for a fixed prompt set.

A golden directory holds goldens.safetensors (tensors named "<prompt>/<site>", no batch dim) and
a manifest, which lists the prompts, carries the hash of the tensor file and the pass/fail
thresholds. C++ and Python tests both read their thresholds from the manifest, so the two sides
cannot drift apart.

The committed fixture's manifest is manifest.json. Real-model goldens are computed on each machine
and FP32 CPU math differs in the last bits between machines, so data/golden/<model> holds one
manifest per GPU, named <device id>.json (see fingerprint.device_id).

Usage: python -m kvbound.golden <model-name>
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import AutoTokenizer

from kvbound import fingerprint, tolerance
from kvbound.models import load_spec, sha256_file
from kvbound.ref import LAYER_SITES, Llama, load_llama

SCHEMA_VERSION = 1
GOLDEN_FILE = "goldens.safetensors"
MANIFEST_FILE = "manifest.json"

# FP32 implementations of the same math must agree with the goldens this closely.
FP32_THRESHOLDS = {"rel_rms_max": 1e-5, "cosine_min": 0.99999}
# FP16 thresholds are this multiple of the error torch's own FP16 forward shows.
FP16_MARGIN = 4.0

REAL_MODEL_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. The crew consisted of "
    "Neil Armstrong, Buzz Aldrin and Michael Collins. Armstrong became the first person to",
    "Memory bandwidth, not arithmetic, limits how fast a language model can generate text one "
    "token at a time. Every decode step reads each weight exactly once, so the time per token "
    "cannot be shorter than the size of the weights divided by the rate at which the GPU can "
    "read memory. This is why smaller number formats make generation",
]


@dataclass(frozen=True)
class Prompt:
    id: str
    input_ids: list[int]
    text: str | None = None


def capture_sites(model: Llama, input_ids: list[int]) -> dict[str, Tensor]:
    """Runs one prompt and returns every capture site as float32 on the CPU, without a batch dim."""
    device = model.embed.weight.device
    capture: dict[str, Tensor] = {}
    model(torch.tensor([input_ids], device=device), capture)
    return {site: value[0].float().cpu() for site, value in capture.items()}


def site_depth(site: str, n_layers: int) -> int:
    """Number of layers of accumulated error behind a site."""
    if site == "embed":
        return 0
    if site.startswith("layers."):
        return int(site.split(".")[1])
    return n_layers


def fp16_thresholds(
    model_fp16: Llama, prompts: list[Prompt], golden: dict[str, Tensor]
) -> dict[str, Any]:
    """Measures torch's FP16 forward against the FP32 goldens and sets limits above it.

    The per-site rel-RMS limit is a line through depth (layers of accumulated error) that sits
    FP16_MARGIN above every observed error. Limits are stored per site, so readers need no
    formula.
    """
    n_layers = model_fp16.cfg.n_layers
    worst = [0.0] * (n_layers + 1)  # indexed by site depth
    mismatches, positions, kl = 0, 0, 0.0
    site_names: list[str] = []
    for prompt in prompts:
        sites = capture_sites(model_fp16, prompt.input_ids)
        site_names = list(sites)
        for site, value in sites.items():
            depth = site_depth(site, n_layers)
            err = tolerance.rel_rms(value, golden[f"{prompt.id}/{site}"])
            worst[depth] = max(worst[depth], err)
        expected = golden[f"{prompt.id}/logits"]
        mismatches += int((sites["logits"].argmax(-1) != expected.argmax(-1)).sum())
        positions += len(prompt.input_ids)
        kl = max(kl, tolerance.mean_kl(sites["logits"], expected))

    base = FP16_MARGIN * worst[0]
    slope = max((FP16_MARGIN * worst[d] - base) / d for d in range(1, n_layers + 1))
    slope = max(slope, 0.0)
    return {
        "rel_rms_max": {site: base + slope * site_depth(site, n_layers) for site in site_names},
        # Teacher-forced argmax mismatches allowed, pooled over all positions of all prompts.
        # Near-ties flip under any rounding change, so at least one is always allowed.
        "top1_mismatches_max": max(1, math.ceil(FP16_MARGIN * mismatches)),
        "positions": positions,
        # Largest per-prompt mean KL(golden || actual), in nats.
        "mean_kl_max": FP16_MARGIN * kl,
        "observed_torch_fp16": {
            "rel_rms_by_depth": worst,
            "top1_mismatches": mismatches,
            "mean_kl": kl,
        },
    }


def dump(
    model: Llama,
    prompts: list[Prompt],
    out_dir: Path,
    model_meta: dict[str, Any],
    fp16_device: str = "cpu",
    manifest_file: str = MANIFEST_FILE,
) -> None:
    """Writes goldens.safetensors and the manifest named manifest_file.

    Goldens come from `model` as given (FP32, on the CPU so they do not depend on a GPU). The
    model is then converted to FP16 in place and moved to `fp16_device` to calibrate thresholds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    golden: dict[str, Tensor] = {}
    for prompt in prompts:
        golden[f"{prompt.id}/input_ids"] = torch.tensor(prompt.input_ids, dtype=torch.int32)
        for site, value in capture_sites(model, prompt.input_ids).items():
            golden[f"{prompt.id}/{site}"] = value
    save_file(golden, str(out_dir / GOLDEN_FILE))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": {**model_meta, "config": model.cfg.to_dict()},
        "golden_file": GOLDEN_FILE,
        "golden_sha256": sha256_file(out_dir / GOLDEN_FILE),
        "golden_dtype": "F32",
        "tensor_name": "<prompt id>/<site>",
        "sites": ["embed", *(f"layers.<i>.{s}" for s in LAYER_SITES), "final_norm", "logits"],
        "prompts": [
            {"id": p.id, "n_tokens": len(p.input_ids), **({"text": p.text} if p.text else {})}
            for p in prompts
        ],
        "thresholds": {
            "fp32": FP32_THRESHOLDS,
            "fp16": fp16_thresholds(model.half().to(fp16_device), prompts, golden),
        },
        "torch_version": torch.__version__,
    }
    with (out_dir / manifest_file).open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")


def load(
    golden_dir: Path, manifest_file: str = MANIFEST_FILE
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Returns (manifest, tensors) after checking the tensor file against the manifest hash."""
    manifest = json.loads((golden_dir / manifest_file).read_text(encoding="utf-8"))
    path = golden_dir / manifest["golden_file"]
    if sha256_file(path) != manifest["golden_sha256"]:
        raise ValueError(f"{path} does not match the hash in its manifest; regenerate the goldens")
    return manifest, load_file(str(path))


def device_manifest(device: str | None = None) -> str:
    """Manifest file name of a GPU's real-model goldens; by default this machine's first GPU."""
    return f"{device or fingerprint.device_id(fingerprint.gpu_info())}.json"


def load_for_model(
    name: str, device: str | None = None
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Loads data/golden/<name> through the manifest of `device` (default: this machine's GPU)."""
    golden_dir = load_spec(name).golden_dir
    manifest_file = device_manifest(device)
    if not (golden_dir / manifest_file).exists():
        raise FileNotFoundError(
            f"{golden_dir / manifest_file} not found; run `make golden MODEL={name}` on this device"
        )
    return load(golden_dir, manifest_file)


def dump_for_model(name: str) -> Path:
    """Dumps goldens for a fetched real model into data/golden/<name>; returns the manifest path."""
    spec = load_spec(name)
    manifest_file = device_manifest()
    checkpoint = spec.checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prompts = [
        Prompt(f"p{i}", tokenizer(text).input_ids, text)
        for i, text in enumerate(REAL_MODEL_PROMPTS)
    ]
    meta = {
        "name": spec.name,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "weights_precision": "fp16",
    }
    model = load_llama(checkpoint, weights="fp16")
    fp16_device = "cuda" if torch.cuda.is_available() else "cpu"
    dump(model, prompts, spec.golden_dir, meta, fp16_device, manifest_file)
    return spec.golden_dir / manifest_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    print(dump_for_model(parser.parse_args().model))


if __name__ == "__main__":
    main()
