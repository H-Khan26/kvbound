"""Golden tensors: FP32 per-site activations and logits for a fixed prompt set.

A golden directory holds goldens.safetensors (tensors named "<prompt>/<site>", no batch dim) and
a manifest, which lists the prompts, carries the hash of the tensor file and the pass/fail
thresholds. C++ and Python tests both read their thresholds from the manifest, so the two sides
cannot drift apart.

A Llama prompt is a list of token ids, stored as "<prompt>/input_ids". A Whisper prompt is one
window of log-mel features with the decoder ids fed alongside it (teacher forcing), stored as
"<prompt>/input_features" and "<prompt>/input_ids"; "<prompt>/greedy_ids" is the sequence greedy
decoding gives for those features.

The committed fixture's manifest is manifest.json. Real-model goldens are computed on each machine
and FP32 CPU math differs in the last bits between machines, so data/golden/<model> holds one
manifest per GPU, named <device id>.json (see fingerprint.device_id).

Usage: python -m kvbound.golden <model-name>
"""

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import AutoTokenizer

from kvbound import audio, fingerprint, tolerance
from kvbound.evals import asr
from kvbound.models import GENERATION_FILE, load_spec, sha256_file
from kvbound.ref import (
    LAYER_SITES,
    Llama,
    Whisper,
    WhisperConfig,
    WhisperRules,
    load_llama,
    load_whisper,
)
from kvbound.ref import whisper as ref_whisper

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
# Recitation clips a real Whisper model's goldens cover, each from another reciter.
REAL_MODEL_CLIPS = 3


@dataclass(frozen=True)
class Prompt:
    id: str
    input_ids: list[int]
    text: str | None = None

    def inputs(self) -> dict[str, Tensor]:
        """The tensors stored in the golden file beside the sites."""
        return {"input_ids": torch.tensor(self.input_ids, dtype=torch.int32)}

    def entry(self) -> dict[str, Any]:
        """The manifest's description of this prompt."""
        text = {"text": self.text} if self.text else {}
        return {"id": self.id, "n_tokens": len(self.input_ids), **text}


@dataclass(frozen=True)
class AudioPrompt:
    """A Whisper prompt: one window of audio and the decoder ids fed with it."""

    id: str
    # Log-mel features [n_mels, frames], float32.
    features: Tensor
    # Decoder input, teacher-forced from position 0.
    input_ids: list[int]
    # What greedy decoding gives for the features, prompt included.
    greedy_ids: list[int]
    text: str | None = None

    def inputs(self) -> dict[str, Tensor]:
        return {
            "input_features": self.features.to(torch.float32),
            "input_ids": torch.tensor(self.input_ids, dtype=torch.int32),
            "greedy_ids": torch.tensor(self.greedy_ids, dtype=torch.int32),
        }

    def entry(self) -> dict[str, Any]:
        text = {"text": self.text} if self.text else {}
        n_frames = self.features.shape[1]
        return {"id": self.id, "n_tokens": len(self.input_ids), "n_frames": n_frames, **text}


# Runs one prompt through a model; returns every site as float32 on the CPU without a batch dim.
CaptureFn = Callable[[Any, Any], dict[str, Tensor]]
# Number of layers of accumulated error behind a site.
DepthFn = Callable[[str], int]


def capture_sites(model: Llama, input_ids: list[int]) -> dict[str, Tensor]:
    """Runs one prompt and returns every capture site as float32 on the CPU, without a batch dim."""
    device = model.embed.weight.device
    capture: dict[str, Tensor] = {}
    model(torch.tensor([input_ids], device=device), capture)
    return {site: value[0].float().cpu() for site, value in capture.items()}


def capture_whisper_sites(
    model: Whisper, features: Tensor, input_ids: list[int]
) -> dict[str, Tensor]:
    """capture_sites for Whisper: features [n_mels, frames] and teacher-forced decoder ids."""
    device = model.decoder.embed.weight.device
    capture: dict[str, Tensor] = {}
    model(features[None].to(device), torch.tensor([input_ids], device=device), capture)
    # The frame-major conv sites are transposed views; a tensor file stores contiguous data.
    return {site: value[0].float().cpu().contiguous() for site, value in capture.items()}


def site_depth(site: str, n_layers: int) -> int:
    """Number of layers of accumulated error behind a site."""
    if site == "embed":
        return 0
    if site.startswith("layers."):
        return int(site.split(".")[1])
    return n_layers


def whisper_site_depth(site: str, cfg: WhisperConfig) -> int:
    """site_depth for Whisper. Every decoder layer reads the whole encoder, so decoder layers
    count on from the encoder's depth; the decoder's embedding reads nothing before it."""
    parts = site.split(".")
    layer = int(parts[2]) if parts[1:2] == ["layers"] else None
    if parts[0] == "encoder":
        if layer is not None:
            return layer
        return cfg.encoder_layers if parts[1] in ("final_norm", "cross_kv") else 0
    if site == "decoder.embed":
        return 0
    return cfg.encoder_layers + (cfg.decoder_layers if layer is None else layer)


def fp16_thresholds(
    model_fp16: Any,
    prompts: Sequence[Any],
    golden: dict[str, Tensor],
    capture: CaptureFn,
    depth_of: DepthFn,
) -> dict[str, Any]:
    """Measures torch's FP16 forward against the FP32 goldens and sets limits above it.

    The per-site rel-RMS limit is a line through depth (layers of accumulated error) that sits
    FP16_MARGIN above every observed error. Limits are stored per site, so readers need no
    formula.
    """
    worst_at: dict[int, float] = {}
    mismatches, positions, kl = 0, 0, 0.0
    site_names: list[str] = []
    for prompt in prompts:
        sites = capture(model_fp16, prompt)
        site_names = list(sites)
        for site, value in sites.items():
            depth = depth_of(site)
            err = tolerance.rel_rms(value, golden[f"{prompt.id}/{site}"])
            worst_at[depth] = max(worst_at.get(depth, 0.0), err)
        expected = golden[f"{prompt.id}/logits"]
        mismatches += int((sites["logits"].argmax(-1) != expected.argmax(-1)).sum())
        positions += len(prompt.input_ids)
        kl = max(kl, tolerance.mean_kl(sites["logits"], expected))

    worst = [worst_at.get(depth, 0.0) for depth in range(max(worst_at) + 1)]  # indexed by depth
    base = FP16_MARGIN * worst[0]
    slope = max((FP16_MARGIN * worst[d] - base) / d for d in range(1, len(worst)))
    slope = max(slope, 0.0)
    return {
        "rel_rms_max": {site: base + slope * depth_of(site) for site in site_names},
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
    n_layers = model.cfg.n_layers
    dump_sites(
        model,
        prompts,
        out_dir,
        model_meta,
        capture=lambda m, prompt: capture_sites(m, prompt.input_ids),
        depth_of=lambda site: site_depth(site, n_layers),
        sites=["embed", *(f"layers.<i>.{s}" for s in LAYER_SITES), "final_norm", "logits"],
        fp16_device=fp16_device,
        manifest_file=manifest_file,
    )


def dump_whisper(
    model: Whisper,
    prompts: list[AudioPrompt],
    out_dir: Path,
    model_meta: dict[str, Any],
    fp16_device: str = "cpu",
    manifest_file: str = MANIFEST_FILE,
) -> None:
    """dump for a Whisper model: the same files, with encoder and decoder sites."""
    cfg = model.cfg
    dump_sites(
        model,
        prompts,
        out_dir,
        model_meta,
        capture=lambda m, prompt: capture_whisper_sites(m, prompt.features, prompt.input_ids),
        depth_of=lambda site: whisper_site_depth(site, cfg),
        sites=[s.replace("{i}", "<i>").replace("{j}", "<j>") for s in ref_whisper.SITES],
        fp16_device=fp16_device,
        manifest_file=manifest_file,
    )


def dump_sites(
    model: Any,
    prompts: Sequence[Any],
    out_dir: Path,
    model_meta: dict[str, Any],
    capture: CaptureFn,
    depth_of: DepthFn,
    sites: list[str],
    fp16_device: str,
    manifest_file: str,
) -> None:
    """Writes the golden file and manifest of either model family (see dump).

    capture runs one prompt, depth_of orders the sites for the FP16 thresholds and `sites` is
    the manifest's list of site patterns.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    golden: dict[str, Tensor] = {}
    for prompt in prompts:
        for name, value in prompt.inputs().items():
            golden[f"{prompt.id}/{name}"] = value
        for site, value in capture(model, prompt).items():
            golden[f"{prompt.id}/{site}"] = value
    save_file(golden, str(out_dir / GOLDEN_FILE))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": {**model_meta, "config": model.cfg.to_dict()},
        "golden_file": GOLDEN_FILE,
        "golden_sha256": sha256_file(out_dir / GOLDEN_FILE),
        "golden_dtype": "F32",
        "tensor_name": "<prompt id>/<site>",
        "sites": sites,
        "prompts": [prompt.entry() for prompt in prompts],
        "thresholds": {
            "fp32": FP32_THRESHOLDS,
            "fp16": fp16_thresholds(
                model.half().to(fp16_device), prompts, golden, capture, depth_of
            ),
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


def real_audio_prompts(model: Whisper, rules: WhisperRules) -> list[AudioPrompt]:
    """The first fetched clip (kvbound.evals.asr) of each of REAL_MODEL_CLIPS reciters.

    The decoder ids fed with a clip are the ones greedy decoding gives for it, so the goldens
    cover the path the engine takes when it transcribes the clip. A prompt carries the clip's id
    and not its transcript: the manifest is committed and the dataset's text is not.
    """
    clips: dict[str, asr.Clip] = {}
    for clip in asr.load_clips():
        clips.setdefault(clip.reciter, clip)
    prompts = []
    for clip in list(clips.values())[:REAL_MODEL_CLIPS]:
        features = audio.features(clip.audio(), model.cfg.n_mels)
        ids = ref_whisper.greedy(model, features, rules)
        prompts.append(AudioPrompt(clip.id, features, ids, ids))
    return prompts


def dump_for_model(name: str) -> Path:
    """Dumps goldens for a fetched real model into data/golden/<name>; returns the manifest path."""
    spec = load_spec(name)
    manifest_file = device_manifest()
    checkpoint = spec.checkpoint_dir
    meta = {
        "name": spec.name,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "weights_precision": "fp16",
    }
    fp16_device = "cuda" if torch.cuda.is_available() else "cpu"
    if spec.family == "whisper":
        whisper = load_whisper(checkpoint, weights="fp16")
        rules = WhisperRules.from_json(checkpoint / GENERATION_FILE)
        clips = real_audio_prompts(whisper, rules)
        dump_whisper(whisper, clips, spec.golden_dir, meta, fp16_device, manifest_file)
        return spec.golden_dir / manifest_file
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prompts = [
        Prompt(f"p{i}", tokenizer(text).input_ids, text)
        for i, text in enumerate(REAL_MODEL_PROMPTS)
    ]
    model = load_llama(checkpoint, weights="fp16")
    dump(model, prompts, spec.golden_dir, meta, fp16_device, manifest_file)
    return spec.golden_dir / manifest_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    print(dump_for_model(parser.parse_args().model))


if __name__ == "__main__":
    main()
