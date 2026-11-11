"""Model registry: configs/models/*.toml, checkpoint locations and fetching.

Usage: python -m kvbound.models fetch <name>
"""

import argparse
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from kvbound.paths import CONFIGS, DATA

CHECKPOINT_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_repo: str
    revision: str
    gated: bool
    arch: dict[str, Any]

    @property
    def checkpoint_dir(self) -> Path:
        return DATA / "models" / self.name

    @property
    def golden_dir(self) -> Path:
        return DATA / "golden" / self.name


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
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str) -> Path:
    """Downloads the pinned checkpoint into data/models/<name> and writes its manifest."""
    spec = load_spec(name)
    out = spec.checkpoint_dir
    snapshot_download(
        spec.hf_repo, revision=spec.revision, allow_patterns=CHECKPOINT_FILES, local_dir=out
    )
    manifest = {
        "name": spec.name,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "weights_sha256": sha256_file(out / "model.safetensors"),
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
