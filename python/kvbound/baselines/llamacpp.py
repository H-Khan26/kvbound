"""llama.cpp baseline: the pinned release, its GGUF files, and how many bits each weight takes.

Everything lives under $KVBOUND_EXT/llama.cpp/<tag>/ (kvbound.paths.EXT):

  downloads/<artifact>/   each pinned zip, alone in its own folder, checked against its sha256
  bin/                    the official Windows CUDA 12 release: llama-server, llama-quantize,
                          llama.dll and the ggml backends
  cudart/                 the CUDA runtime zip of the same release (cudart, cuBLAS, cuBLASLt)
  src/                    the source tree at the tag, for convert_hf_to_gguf.py and llama.h
  convert-venv/           a Python environment holding the converter's requirements at the tag
  hf/<model>/             the converter's input while it runs (see converter_input)
  gguf/<model>/<FORMAT>.gguf

The F16 file is converted from the fetched Hugging Face checkpoint (data/models/<name>), plus the
repository's tokenizer.model where it has one, by the tag's own converter; Q8_0, Q4_K_M and
Q4_0 are quantized from it with the tag's llama-quantize, which keeps its default mix of tensor
types per format (see gguf_info).

Downloads are treated as untrusted: each zip goes into a new folder, is extracted into another
new folder with every member's path checked, and no interpreter runs from inside them. The
converter is the one exception the baseline needs: it runs from the source tree with -E -s, its
working folder outside the tree.

Usage: python -m kvbound.baselines.llamacpp fetch
       python -m kvbound.baselines.llamacpp convert --model tinyllama-1.1b
       python -m kvbound.baselines.llamacpp info --model tinyllama-1.1b
"""

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from huggingface_hub import hf_hub_download, list_repo_files

from kvbound import record
from kvbound.models import CHECKPOINT_FILES, ModelSpec, load_spec, sha256_file
from kvbound.paths import EXT, REPO_ROOT

TAG = "b12503"
COMMIT = "4577932fab595eaa8b27ad71734d5de1f3c07d8c"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}"


@dataclass(frozen=True)
class Artifact:
    name: str  # also the folder under downloads/ and the extraction folder under the tag
    file: str
    url: str
    sha256: str


ARTIFACTS = (
    Artifact(
        "bin",
        f"llama-{TAG}-bin-win-cuda-12.4-x64.zip",
        f"{RELEASE_URL}/llama-{TAG}-bin-win-cuda-12.4-x64.zip",
        "9df9b6316051ce258debb234b3475762eecd8721e6bb41474f0743d54aad08aa",
    ),
    Artifact(
        "cudart",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        f"{RELEASE_URL}/cudart-llama-bin-win-cuda-12.4-x64.zip",
        "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    ),
    Artifact(
        "src",
        f"llama.cpp-{TAG}.zip",
        f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{TAG}.zip",
        "adef5ebb7c8715328d7de6524c9a3a483f1b9a9b9e8d351559ee0221d32f0233",
    ),
)
# Written into an extraction folder once every member is in place.
DONE_FILE = "kvbound-fetched.json"

# Formats llama-quantize makes from the F16 file. Q4_0 (blocks of 32 weights with one F16
# scale, round to nearest) is the nearest relative of kvbound's int4_g; Q4_K_M is the format
# most often downloaded.
FORMATS = ("F16", "Q8_0", "Q4_K_M", "Q4_0")

ROOT = EXT / "llama.cpp" / TAG
BIN = ROOT / "bin"
CUDART = ROOT / "cudart"
SRC = ROOT / "src" / f"llama.cpp-{TAG}"
CONVERT_VENV = ROOT / "convert-venv"
CONVERTER_REQUIREMENTS = "requirements/requirements-convert_hf_to_gguf.txt"
# Files the converter reads beyond the fetched checkpoint, taken from the model's repository when
# it has them. A SentencePiece model carries the token scores llama.cpp's tokenizer merges by:
# from tokenizer.json alone the converter writes a placeholder score for every token, and
# llama.cpp then splits words unlike the Hugging Face tokenizer (TinyLlama: 344,076 tokens
# against 341,469 on the WikiText-2 test split).
CONVERTER_EXTRA_FILES = ("tokenizer.model",)


def engine_entry(route: str) -> dict[str, Any]:
    """The record's engine entry: the release tag, its commit, the route the numbers took, and
    the sha256 of each zip they came from."""
    return record.external_engine(
        "llama.cpp",
        TAG,
        commit=COMMIT,
        route=route,
        artifacts={a.file: a.sha256 for a in ARTIFACTS},
    )


def exe(name: str) -> Path:
    """bin/<name>.exe of the pinned release; exits with a hint when it was not fetched."""
    path = BIN / f"{name}.exe"
    if not path.exists():
        raise SystemExit(f"{path} not found; run `make fetch-llamacpp`")
    return path


def runtime_env(*, gpu: bool = True) -> dict[str, str]:
    """Environment for the release's programs: the CUDA runtime folder on PATH. gpu=False hides
    the GPU, for programs such as llama-quantize that do not need it."""
    env = {**os.environ, "PATH": f"{CUDART}{os.pathsep}{os.environ.get('PATH', '')}"}
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def gguf_path(model: str, fmt: str) -> Path:
    return ROOT / "gguf" / model / f"{fmt}.gguf"


def require_gguf(model: str, fmt: str) -> Path:
    path = gguf_path(model, fmt)
    if not path.exists():
        raise SystemExit(f"{path} not found; run `make fetch-llamacpp MODEL={model}`")
    return path


def _download(artifact: Artifact) -> Path:
    """downloads/<name>/<file>, downloaded when it is missing or its sha256 differs."""
    folder = ROOT / "downloads" / artifact.name
    path = folder / artifact.file
    if path.exists() and sha256_file(path) == artifact.sha256:
        return path
    if folder.exists() and any(p != path for p in folder.iterdir()):
        raise SystemExit(f"{folder} holds other files; empty it before fetching again")
    folder.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    print(f"downloading {artifact.url}")
    with urllib.request.urlopen(artifact.url) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f, 1 << 22)
    digest = sha256_file(partial)
    if digest != artifact.sha256:
        partial.unlink()
        raise SystemExit(f"{artifact.url} has sha256 {digest}, expected {artifact.sha256}")
    partial.replace(path)
    return path


def _extract(archive: Path, dest: Path) -> None:
    """Extracts a zip into a new folder, refusing members that would land outside it."""
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # a folder this function made and did not finish
    partial.mkdir(parents=True)
    root = partial.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (partial / member).resolve()
            if not target.is_relative_to(root):
                raise SystemExit(f"{archive}: member {member!r} escapes the extraction folder")
        zf.extractall(partial)
    partial.replace(dest)


def fetch() -> None:
    """Downloads, checks and extracts every pinned artifact that is not already in place."""
    if platform.system() != "Windows":
        raise SystemExit(
            f"the pinned artifacts are the Windows CUDA release; elsewhere, build llama.cpp "
            f"{TAG} with CUDA and put its programs in {BIN}"
        )
    for artifact in ARTIFACTS:
        dest = ROOT / artifact.name
        done = dest / DONE_FILE
        if done.exists() and json.loads(done.read_text(encoding="utf-8"))["sha256"] == (
            artifact.sha256
        ):
            continue
        if dest.exists():
            raise SystemExit(f"{dest} exists but was not completed; remove it and fetch again")
        _extract(_download(artifact), dest)
        stamp = {"file": artifact.file, "url": artifact.url, "sha256": artifact.sha256}
        done.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        print(f"{artifact.file} -> {dest}")


def _converter_python() -> Path:
    """The converter environment's interpreter, made with uv from the tag's requirements."""
    python = CONVERT_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python.exists():
        return python
    requirements = SRC / CONVERTER_REQUIREMENTS
    if not requirements.exists():
        raise SystemExit(f"{requirements} not found; run `fetch` first")
    subprocess.run(["uv", "venv", "--python", "3.12", str(CONVERT_VENV)], check=True)
    # Wheels only, so that installing runs no build script from a source distribution.
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--only-binary", ":all:",
         "--index-strategy", "unsafe-best-match", "-r", str(requirements)],
        check=True, cwd=REPO_ROOT,
    )  # fmt: skip
    return python


def converter_input(spec: ModelSpec) -> Path:
    """hf/<model>: the fetched checkpoint's files plus CONVERTER_EXTRA_FILES from its repository
    at the pinned revision, where the repository has them. Files are hard links when the
    external root shares the checkpoint's drive and copies otherwise."""
    out = ROOT / "hf" / spec.name
    out.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_FILES:
        if not (out / name).exists():
            try:
                os.link(spec.checkpoint_dir / name, out / name)
            except OSError:
                shutil.copyfile(spec.checkpoint_dir / name, out / name)
    present = set(list_repo_files(spec.hf_repo, revision=spec.revision))
    for name in CONVERTER_EXTRA_FILES:
        if name in present and not (out / name).exists():
            hf_hub_download(spec.hf_repo, name, revision=spec.revision, local_dir=out)
    return out


def convert(model: str, formats: tuple[str, ...] = FORMATS) -> None:
    """The F16 GGUF of a fetched checkpoint, then each other format quantized from it."""
    spec = load_spec(model)
    if spec.family != record.DEFAULT_FAMILY:
        raise SystemExit(f"{model} is a {spec.family} model; the llama.cpp baseline is for Llamas")
    if not (spec.checkpoint_dir / "model.safetensors").exists():
        raise SystemExit(f"{spec.checkpoint_dir} not found; run `make fetch MODEL={model}`")
    f16 = gguf_path(model, "F16")
    f16.parent.mkdir(parents=True, exist_ok=True)
    if not f16.exists():
        source = converter_input(spec)
        partial = f16.with_name("F16.gguf.partial")
        # The converter imports its own package from the script's folder and adds the tag's
        # gguf-py itself; -E -s keep PYTHON* variables and the user's site-packages out, and the
        # working folder is the output folder, not the source tree.
        subprocess.run(
            [str(_converter_python()), "-E", "-s", str(SRC / "convert_hf_to_gguf.py"),
             str(source), "--outtype", "f16", "--outfile", str(partial)],
            check=True, cwd=f16.parent,
        )  # fmt: skip
        partial.replace(f16)
        shutil.rmtree(source)  # links or copies, made above
    for fmt in formats:
        out = gguf_path(model, fmt)
        if out.exists():
            continue
        partial = out.with_name(out.name + ".partial")
        subprocess.run(
            [str(exe("llama-quantize")), str(f16), str(partial), fmt],
            check=True, cwd=out.parent, env=runtime_env(gpu=False),
        )  # fmt: skip
        partial.replace(out)
    for fmt in ("F16", *formats):
        info = gguf_info(gguf_path(model, fmt))
        print(f"{model} {fmt}: {info['bits_per_weight']:.3f} bits per weight, {info['types']}")


# ggml tensor types: (weights per block, bytes per block). Types not listed are not written by
# the formats above.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}
# GGUF metadata value types: struct format of the scalar ones.
GGUF_SCALARS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}  # fmt: skip
GGUF_STRING, GGUF_ARRAY = 8, 9


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(size))[0]


def _read_string(f: BinaryIO) -> str:
    return f.read(_read(f, "Q")).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, kind: int) -> Any:
    if kind in GGUF_SCALARS:
        return _read(f, GGUF_SCALARS[kind])
    if kind == GGUF_STRING:
        return _read_string(f)
    if kind == GGUF_ARRAY:
        item, count = _read(f, "I"), _read(f, "Q")
        return [_read_value(f, item) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {kind}")


def gguf_info(path: Path) -> dict[str, Any]:
    """Weights, tensor bytes and bits per weight of a GGUF file, from its header.

    bits_per_weight is 8 * the bytes of every tensor (block scales included) / the number of
    weights, so norms and other F32 tensors count. llama.cpp's own "BPW" divides the whole file,
    header and vocabulary included, and reads slightly higher.
    """
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version, n_tensors, n_kv = _read(f, "I"), _read(f, "Q"), _read(f, "Q")
        metadata = {}
        for _ in range(n_kv):
            key = _read_string(f)
            value = _read_value(f, _read(f, "I"))
            if not isinstance(value, list):  # vocabularies are long and not needed here
                metadata[key] = value
        weights = stored = 0
        types: dict[str, int] = {}
        for _ in range(n_tensors):
            _read_string(f)
            dims = [_read(f, "Q") for _ in range(_read(f, "I"))]
            kind = _read(f, "I")
            _read(f, "Q")  # data offset
            if kind not in GGML_TYPES:
                raise ValueError(f"{path}: tensor type {kind} has no block size here")
            name, block, block_bytes = GGML_TYPES[kind]
            n = 1
            for d in dims:
                n *= d
            weights += n
            stored += n // block * block_bytes
            types[name] = types.get(name, 0) + n
    return {
        "gguf_version": version,
        "file_type": metadata.get("general.file_type"),
        "weights": weights,
        "tensor_bytes": stored,
        "file_bytes": path.stat().st_size,
        "bits_per_weight": 8 * stored / weights,
        # Share of the weights stored in each tensor type.
        "types": {name: round(n / weights, 4) for name, n in sorted(types.items())},
    }


def gguf_params(model: str, fmt: str) -> dict[str, Any]:
    """The weight entries of a record's params for one GGUF file."""
    path = require_gguf(model, fmt)
    info = gguf_info(path)
    return {
        "weights": fmt,
        "bits_per_weight": info["bits_per_weight"],
        "gguf": {"file": str(path.relative_to(EXT)), **info},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download and extract the pinned release")
    for name, text in (("convert", "make the GGUF files"), ("info", "print bits per weight")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--model", required=True)
        p.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
    elif args.command == "convert":
        convert(args.model, tuple(args.formats))
    else:
        for fmt in args.formats:
            info = gguf_info(require_gguf(args.model, fmt))
            print(json.dumps({"model": args.model, "format": fmt, **info}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
