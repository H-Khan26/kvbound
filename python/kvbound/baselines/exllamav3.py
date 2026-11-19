"""ExLlamaV3 baseline: the pinned release, its EXL3 conversions, and the bits each weight takes.

Everything lives under $KVBOUND_EXT/exllamav3/<version>/ (kvbound.paths.EXT):

  downloads/wheel/<file>   the release's Windows wheel for CUDA 12.8, torch 2.11 and CPython 3.12
  venv/                    a Python environment holding that wheel, torch and triton-windows
  work/<model>/<FORMAT>/   the converter's working folder while it runs
  exl3/<model>/<FORMAT>/   each converted model, with kvbound-converted.json: the converter's
                           arguments as it resolved them, its seconds and the packages it ran with
  logs/

ExLlamaV3 is not part of the repository's environment: it runs in venv/ as worker processes
(kvbound.baselines.exllamav3_lib), and the repository's environment starts them and writes the
records. The EXL3 formats are converted from the fetched checkpoint (data/models/<name>) by the
release's own converter with its defaults (calibration on its bundled text mix, 250 rows of 2048
tokens) except for the options in CONVERT_ARGS. FP16 is no conversion: ExLlamaV3 loads the
fetched checkpoint unquantized, which checks the protocol against kvbound's FP16 numbers.

Downloads are treated as untrusted: the wheel goes into its own folder and is installed only
once its sha256 matches. Its dependencies come from PyPI and the PyTorch index as wheels only,
so installing runs no build script.

Usage: python -m kvbound.baselines.exllamav3 fetch
       python -m kvbound.baselines.exllamav3 convert --model tinyllama-1.1b
       python -m kvbound.baselines.exllamav3 info --model tinyllama-1.1b
"""

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kvbound import record
from kvbound.baselines.llamacpp import Artifact, download
from kvbound.models import load_spec
from kvbound.paths import EXT, REPO_ROOT

VERSION = "1.6.7"
COMMIT = "v1.6.7"  # the release tag; GitHub builds the wheels from it
RELEASE_URL = f"https://github.com/turboderp-org/exllamav3/releases/download/v{VERSION}"
WHEEL = Artifact(
    "wheel",
    f"exllamav3-{VERSION}+cu128.torch2.11.0-cp312-cp312-win_amd64.whl",
    f"{RELEASE_URL}/exllamav3-{VERSION}%2Bcu128.torch2.11.0-cp312-cp312-win_amd64.whl",
    "dac589bed35f19e7558e124f6d067547949a69ba97bb9b0ca42663a026c2c7c5",
)
# The packages the wheel was built against, pinned; the rest of its requirements resolve to the
# newest wheels and are listed in the environment's DONE_FILE.
REQUIREMENTS = ("torch==2.11.0", "triton-windows==3.6.0.post26")
TORCH_BACKEND = "cu128"
DONE_FILE = "kvbound-fetched.json"
CONVERTED_FILE = "kvbound-converted.json"

# Formats: converter arguments of each EXL3 format. The converter's --bits is the average over
# the decoder's linear layers, each stored at a whole number of bits from 4 bits up; it keeps the
# token embedding in FP16 and stores the output layer at 6 bits by default, as a separate tensor
# even when the source ties it to the embedding. EXL3_4.0_H4_E4 stores the embedding and the
# output layer at 4 bits too: every matrix near 4 bits, the scope of kvbound's int4_g.
CONVERT_ARGS = {
    "EXL3_4.0": ["--bits", "4.0"],
    "EXL3_4.5": ["--bits", "4.5"],
    "EXL3_5.0": ["--bits", "5.0"],
    "EXL3_4.0_H4_E4": ["--bits", "4.0", "--head_bits", "4", "--embed_bits", "4"],
}
FORMATS = ("FP16", *CONVERT_ARGS)

ROOT = EXT / "exllamav3" / VERSION
VENV = ROOT / "venv"
# Files of the source checkpoint the converter copies but the converted model must not carry.
NOT_COPIED = ("manifest.json",)
# Keys of the converter's args.json that hold absolute folders.
JOB_FOLDERS = ("work_dir", "in_dir", "out_dir")


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed() -> dict[str, Any]:
    """The environment's DONE_FILE; exits with a hint when it was not fetched."""
    done = VENV / DONE_FILE
    if not done.exists():
        raise SystemExit(f"{done} not found; run `make fetch-exllamav3`")
    stamp: dict[str, Any] = json.loads(done.read_text(encoding="utf-8"))
    return stamp


def engine_entry(route: str) -> dict[str, Any]:
    """The record's engine entry: the release, the route the numbers took, the wheel's sha256
    and the versions of the packages it ran on."""
    packages = installed()["packages"]
    return record.external_engine(
        "exllamav3",
        VERSION,
        commit=COMMIT,
        route=route,
        artifacts={WHEEL.file: WHEEL.sha256},
        packages={name: packages[name] for name in ("exllamav3", "torch", "triton-windows")},
    )


def worker_command(*args: str) -> list[str]:
    """The command that runs kvbound.baselines.exllamav3_lib in the pinned environment. -E -s
    keep PYTHON* variables and the user's site-packages out; the worker imports kvbound from the
    repository's python/ folder, its working folder (see worker_cwd)."""
    installed()
    return [str(venv_python()), "-E", "-s", "-m", "kvbound.baselines.exllamav3_lib", *args]


def worker_cwd() -> Path:
    return REPO_ROOT / "python"


def parse_freeze(text: str) -> dict[str, str]:
    """{name: version} of `uv pip freeze` output, names lowercased; the wheel's line reads
    `exllamav3 @ file:///...` and takes the pinned version."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
        elif " @ " in line:
            name, version = line.split(" @ ", 1)[0], VERSION
        else:
            raise ValueError(f"unexpected freeze line {line!r}")
        out[name.strip().lower()] = version.strip()
    return out


def fetch() -> None:
    """Downloads and checks the wheel, then makes the environment unless it is in place."""
    if platform.system() != "Windows":
        raise SystemExit(
            f"the pinned wheel is the Windows one; elsewhere, install exllamav3 {VERSION} for "
            f"torch 2.11 into {VENV}"
        )
    wheel = download(WHEEL, ROOT)
    done = VENV / DONE_FILE
    if done.exists() and json.loads(done.read_text(encoding="utf-8"))["sha256"] == WHEEL.sha256:
        return
    python = venv_python()
    # --clear replaces an environment whose install did not finish.
    subprocess.run(["uv", "venv", "--clear", "--python", "3.12", str(VENV)], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--only-binary", ":all:",
         "--torch-backend", TORCH_BACKEND, *REQUIREMENTS, str(wheel)],
        check=True, cwd=REPO_ROOT,
    )  # fmt: skip
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(python)],
        check=True, capture_output=True, text=True, cwd=REPO_ROOT,
    )  # fmt: skip
    stamp = {"file": WHEEL.file, "sha256": WHEEL.sha256, "packages": parse_freeze(freeze.stdout)}
    done.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"{WHEEL.file} -> {VENV}")


def model_dir(model: str, fmt: str) -> Path:
    """The folder ExLlamaV3 loads for a format: the fetched checkpoint for FP16."""
    if fmt == "FP16":
        return load_spec(model).checkpoint_dir
    return ROOT / "exl3" / model / fmt


def require_model(model: str, fmt: str) -> Path:
    path = model_dir(model, fmt)
    done = path / ("model.safetensors" if fmt == "FP16" else CONVERTED_FILE)
    if not done.exists():
        target = "fetch" if fmt == "FP16" else "fetch-exllamav3"
        raise SystemExit(f"{done} not found; run `make {target} MODEL={model}`")
    return path


def convert(model: str, formats: tuple[str, ...] = FORMATS) -> None:
    """Each EXL3 format of a fetched checkpoint not converted yet, one converter run each."""
    spec = load_spec(model)
    if spec.family != record.DEFAULT_FAMILY:
        raise SystemExit(f"{model} is a {spec.family} model; the ExLlamaV3 baseline is for Llamas")
    if not (spec.checkpoint_dir / "model.safetensors").exists():
        raise SystemExit(f"{spec.checkpoint_dir} not found; run `make fetch MODEL={model}`")
    python = venv_python()
    packages = installed()["packages"]
    for fmt in formats:
        out = model_dir(model, fmt)
        if fmt == "FP16" or (out / CONVERTED_FILE).exists():
            continue
        work = ROOT / "work" / model / fmt
        partial = out.with_name(out.name + ".partial")
        for folder in (work, partial):
            if folder.exists():
                shutil.rmtree(folder)  # a run this function started and did not finish
        work.mkdir(parents=True)
        started = time.perf_counter()
        subprocess.run(
            [str(python), "-E", "-s", "-m", "exllamav3.conversion.convert_model",
             "-i", str(spec.checkpoint_dir), "-o", str(partial), "-w", str(work),
             *CONVERT_ARGS[fmt]],
            check=True, cwd=work,
        )  # fmt: skip
        seconds = time.perf_counter() - started
        for name in NOT_COPIED:
            (partial / name).unlink(missing_ok=True)
        job = json.loads((work / "args.json").read_text(encoding="utf-8"))
        stamp = {
            "arguments": CONVERT_ARGS[fmt],
            # Every setting of the job as the converter resolved it, defaults included; the
            # folders are this machine's and are left out.
            "job": {k: v for k, v in job.items() if k not in JOB_FOLDERS},
            "seconds": seconds,
            "source": str(spec.checkpoint_dir.relative_to(REPO_ROOT)),
            "packages": {name: packages[name] for name in ("exllamav3", "torch", "triton-windows")},
        }
        (partial / CONVERTED_FILE).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        partial.replace(out)
        shutil.rmtree(work)
        info = exl3_info(out, spec.checkpoint_dir)
        print(f"{model} {fmt}: {info['bits_per_weight']:.3f} bits per weight in {seconds:.0f} s")


# Bytes per element of the safetensors dtypes.
SAFETENSORS_DTYPES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "U64": 8, "I32": 4, "U32": 4,
    "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}  # fmt: skip
# Module groups of exl3_info, by the start of a module's key.
EMBEDDING, HEAD, LAYERS, OTHER = "embedding", "head", "layers", "other"
# A converted model's output layer, when the source ties it to the embedding.
TIED = {"lm_head": "model.embed_tokens"}


def safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """The tensor entries of a safetensors file's header: dtype, shape and data_offsets."""
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header: dict[str, dict[str, Any]] = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def module_group(module: str) -> str:
    if module.startswith("model.embed_tokens"):
        return EMBEDDING
    if module.startswith("lm_head"):
        return HEAD
    if module.startswith("model.layers.") and module.endswith("_proj"):
        return LAYERS
    return OTHER


def stored_modules(folder: Path) -> tuple[dict[str, int], dict[str, float], int]:
    """Bytes stored per module (a tensor's key less its last part) over every safetensors file
    of a folder, the trellis bits of each EXL3 module, and the files' total size."""
    stored: dict[str, int] = {}
    trellis_bits: dict[str, float] = {}
    file_bytes = 0
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise ValueError(f"{folder} holds no safetensors file")
    for path in files:
        file_bytes += path.stat().st_size
        for name, entry in safetensors_header(path).items():
            module, part = name.rsplit(".", 1)
            start, end = entry["data_offsets"]
            stored[module] = stored.get(module, 0) + end - start
            if part == "trellis":
                # A tile of 16 x 16 weights takes 16 * K uint16 words at K bits per weight.
                trellis_bits[module] = entry["shape"][-1] / 16
    return stored, trellis_bits, file_bytes


def source_weights(folder: Path) -> dict[str, int]:
    """Weights per module of an unquantized checkpoint."""
    weights: dict[str, int] = {}
    for path in sorted(folder.glob("*.safetensors")):
        for name, entry in safetensors_header(path).items():
            if entry["dtype"] not in SAFETENSORS_DTYPES:
                raise ValueError(f"{path}: {name} has dtype {entry['dtype']}")
            n = 1
            for d in entry["shape"]:
                n *= d
            module = name.rsplit(".", 1)[0]
            weights[module] = weights.get(module, 0) + n
    return weights


def exl3_info(folder: Path, source: Path) -> dict[str, Any]:
    """Weights, stored bytes and bits per weight of a model folder ExLlamaV3 loads, against the
    unquantized checkpoint it came from.

    bits_per_weight is 8 * the bytes of every stored tensor (trellis, sign and scale vectors,
    FP16 embedding, norms) / the number of weights in the source, each counted once. A source
    that ties the output layer to the embedding stores it once; the converted model stores the
    embedding and a quantized output layer, and both count. groups gives the same ratio per
    module group, where the output layer counts its vocabulary x hidden weights even when tied;
    the layers group is the average that the converter's --bits targets. trellis_bits is the
    share of the layers' weights stored at each number of bits.
    """
    weights = source_weights(source)
    stored, trellis_bits, file_bytes = stored_modules(folder)
    groups = {g: {"weights": 0, "bytes": 0} for g in (EMBEDDING, HEAD, LAYERS, OTHER)}
    for module, n_bytes in stored.items():
        n = weights.get(module) or weights.get(TIED.get(module, ""))
        if not n:
            raise ValueError(f"{folder}: {module} has no tensor in {source}")
        group = groups[module_group(module)]
        group["weights"] += n
        group["bytes"] += n_bytes
    if missing := sorted(set(weights) - set(stored)):
        raise ValueError(f"{folder} stores nothing for {missing[:5]}")
    shares: dict[str, int] = {}
    for module, bits in trellis_bits.items():
        if module_group(module) == LAYERS:
            key = f"{bits:g}"
            shares[key] = shares.get(key, 0) + weights[module]
    n_weights, n_bytes = sum(weights.values()), sum(stored.values())
    return {
        "weights": n_weights,
        "tensor_bytes": n_bytes,
        "file_bytes": file_bytes,
        "bits_per_weight": 8 * n_bytes / n_weights,
        "groups": {
            name: {**g, "bits_per_weight": 8 * g["bytes"] / g["weights"]}
            for name, g in groups.items()
            if g["weights"]
        },
        "trellis_bits": {
            k: round(n / groups[LAYERS]["weights"], 4) for k, n in sorted(shares.items())
        },
    }


def weights_params(model: str, fmt: str) -> dict[str, Any]:
    """The weight entries of a record's params for one format."""
    folder = require_model(model, fmt)
    info = exl3_info(folder, load_spec(model).checkpoint_dir)
    entry: dict[str, Any] = {"dir": str(folder.relative_to(EXT if fmt != "FP16" else REPO_ROOT))}
    if fmt != "FP16":
        entry["converted"] = json.loads((folder / CONVERTED_FILE).read_text(encoding="utf-8"))
    return {"weights": fmt, "bits_per_weight": info["bits_per_weight"], "exl3": {**entry, **info}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the pinned wheel and make its environment")
    for name, text in (("convert", "make the EXL3 models"), ("info", "print bits per weight")):
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
            info = exl3_info(require_model(args.model, fmt), load_spec(args.model).checkpoint_dir)
            print(json.dumps({"model": args.model, "format": fmt, **info}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
