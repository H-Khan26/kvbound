"""Model registry: configs/models/*.toml, checkpoint locations and fetching.

A Llama checkpoint is used as downloaded. A Whisper checkpoint is converted after the download
(see convert_whisper), because the engine reads one layout: model.safetensors with a bias for
every projection and no duplicate of the tied output projection, plus generation.json and a
fast tokenizer.json.

Usage: python -m kvbound.models fetch <name>
"""

import argparse
import hashlib
import json
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import list_repo_files, snapshot_download

from kvbound.paths import CONFIGS, DATA

CHECKPOINT_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]

# Files of a Whisper repository other than its weights; a repository holds a subset of them.
WHISPER_SOURCE_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
    "vocab.json",
    "merges.txt",
]
# Weight files of a Whisper repository, most preferred first. The second is a pickle.
WHISPER_WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]
GENERATION_FILE = "generation.json"
# Whisper's timestamp tokens, "<|0.00|>" to "<|30.00|>": the only added tokens that are text.
TIMESTAMP_TOKEN = re.compile(r"<\|\d+\.\d\d\|>")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_repo: str
    revision: str
    gated: bool
    arch: dict[str, Any]
    # "llama" or "whisper".
    family: str = "llama"
    # Decoding rules of a Whisper model (kvbound.ref.whisper.WhisperRules).
    generation: dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_dir(self) -> Path:
        return DATA / "models" / self.name

    @property
    def source_dir(self) -> Path:
        """The repository files a Whisper checkpoint is converted from, as downloaded."""
        return self.checkpoint_dir / "hf"

    @property
    def golden_dir(self) -> Path:
        return DATA / "golden" / self.name

    def variant_dir(self, scheme: str) -> Path:
        """data/models/<name>.<scheme> for a quantized variant; the fetched checkpoint for fp16.

        scheme is a variant name of kvbound.weight_format.VARIANTS, such as "int8_pc", "int4_g",
        "int4_g.mse" (INT4 with mse clipping) or "int4_g.awq" (INT4 with activation-aware scales).
        """
        if scheme == "fp16":
            return self.checkpoint_dir
        return DATA / "models" / f"{self.name}.{scheme}"


def load_spec(name: str) -> ModelSpec:
    path = CONFIGS / "models" / f"{name}.toml"
    if not path.exists():
        known = sorted(p.stem for p in (CONFIGS / "models").glob("*.toml"))
        raise FileNotFoundError(f"unknown model {name!r}; known: {known}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return ModelSpec(
        name=raw["name"],
        hf_repo=raw["hf_repo"],
        revision=raw["revision"],
        gated=raw["gated"],
        arch=raw["arch"],
        family=raw.get("family", "llama"),
        generation=raw.get("generation", {}),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=2) + "\n")


def whisper_weights_file(present: list[str], where: str) -> str:
    """The preferred weight file among the file names `present` in a Whisper repository."""
    for name in WHISPER_WEIGHT_FILES:
        if name in present:
            return name
    raise FileNotFoundError(f"{where} holds none of {WHISPER_WEIGHT_FILES}")


def convert_whisper(
    source_dir: Path, out_dir: Path, generation: dict[str, Any], weights: str | None = None
) -> None:
    """Writes the checkpoint the engine and the reference model load, from a Hugging Face one.

    `weights` names the weight file in source_dir; by default it is the preferred one there.
    out_dir receives:
      config.json        copied.
      model.safetensors  the source tensors in their stored precision under their Hugging Face
                         names, plus a zero "<attention>.k_proj.bias" for every attention (Whisper
                         has no key bias), minus "proj_out.weight" when stored (it must equal
                         "model.decoder.embed_tokens.weight", which the engine uses for both).
      generation.json    `generation`, the decoding rules, checked against the config.
      tokenizer.json     the fast tokenizer, with every added token except the timestamps marked
                         special so that decoding can skip them. Written when the source has a
                         tokenizer.
    """
    # Imported here so the registry loads without torch.
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    from kvbound.ref.config import WhisperConfig  # noqa: PLC0415
    from kvbound.ref.whisper import WhisperRules, hf_weight_names  # noqa: PLC0415

    source_dir, out_dir = Path(source_dir), Path(out_dir)
    cfg = WhisperConfig.from_json(source_dir / "config.json")
    rules = WhisperRules.from_dict(generation)
    rules.check(cfg)

    if weights is None:
        present = [path.name for path in source_dir.iterdir()]
        weights = whisper_weights_file(present, str(source_dir))
    weights_file = source_dir / weights
    if weights_file.suffix == ".safetensors":
        state = load_file(str(weights_file))
    else:
        # weights_only refuses to unpickle anything but tensors and plain containers.
        state = torch.load(weights_file, map_location="cpu", weights_only=True)

    embed = state["model.decoder.embed_tokens.weight"]
    if "proj_out.weight" in state and not torch.equal(state.pop("proj_out.weight"), embed):
        raise ValueError(f"{weights_file}: proj_out.weight is not tied to the token embedding")
    expected = set(hf_weight_names(cfg).values())
    for name in sorted(expected - set(state)):
        if not name.endswith(".k_proj.bias"):
            raise ValueError(f"{weights_file}: {name} is missing")
        weight = state[name.removesuffix("bias") + "weight"]
        state[name] = torch.zeros(weight.shape[0], dtype=weight.dtype)
    if unexpected := set(state) - expected:
        raise ValueError(f"{weights_file}: unexpected tensors {sorted(unexpected)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / "config.json", out_dir / "config.json")
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    write_json(out_dir / GENERATION_FILE, rules.to_dict())
    if any((source_dir / name).exists() for name in ("tokenizer.json", "vocab.json")):
        write_whisper_tokenizer(source_dir, out_dir / "tokenizer.json")


def write_whisper_tokenizer(source_dir: Path, out_file: Path) -> None:
    """Saves the fast tokenizer of a Whisper checkpoint as one tokenizer.json.

    Checkpoints saved from the slow tokenizer list the language and task tokens as plain added
    tokens, which decoding would keep in the text; here they are special, as in openai/whisper-*.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    data = json.loads(tokenizer.backend_tokenizer.to_str())
    for token in data["added_tokens"]:
        if not TIMESTAMP_TOKEN.fullmatch(token["content"]):
            token["special"], token["normalized"] = True, False
    with out_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def fetch_whisper(spec: ModelSpec) -> dict[str, Any]:
    """Downloads a pinned Whisper repository into <checkpoint>/hf and converts it.

    Returns the manifest fields that describe the source weights.
    """
    files = list_repo_files(spec.hf_repo, revision=spec.revision)
    weights = whisper_weights_file(files, f"{spec.hf_repo}@{spec.revision}")
    snapshot_download(
        spec.hf_repo,
        revision=spec.revision,
        allow_patterns=[*WHISPER_SOURCE_FILES, weights],
        local_dir=spec.source_dir,
    )
    # Imported here so the registry loads without torch.
    from kvbound.ref.config import WhisperConfig  # noqa: PLC0415

    arch = WhisperConfig.from_json(spec.source_dir / "config.json").to_dict()
    if any(arch[key] != value for key, value in spec.arch.items()):
        raise ValueError(f"{spec.name}: [arch] {spec.arch} differs from config.json {arch}")
    convert_whisper(spec.source_dir, spec.checkpoint_dir, spec.generation, weights)
    return {
        "source_weights": weights,
        "source_weights_sha256": sha256_file(spec.source_dir / weights),
    }


def fetch(name: str) -> Path:
    """Downloads the pinned checkpoint into data/models/<name> and writes its manifest."""
    spec = load_spec(name)
    out = spec.checkpoint_dir
    source: dict[str, Any] = {}
    if spec.family == "whisper":
        source = fetch_whisper(spec)
    else:
        snapshot_download(
            spec.hf_repo, revision=spec.revision, allow_patterns=CHECKPOINT_FILES, local_dir=out
        )
    manifest = {
        "name": spec.name,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "weights_sha256": sha256_file(out / "model.safetensors"),
        **source,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").add_argument("name")
    args = parser.parse_args()
    print(fetch(args.name))


if __name__ == "__main__":
    main()
