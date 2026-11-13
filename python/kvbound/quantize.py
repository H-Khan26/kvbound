"""Weight quantization: INT8 per output channel and INT4 in groups, written in the kvbound format.

Both schemes are symmetric round-to-nearest with F16 scales. Each scale is rounded to F16 before
the integers are computed from it, so a stored (q, scale) pair reproduces exactly the values the
engine multiplies by, and load_dequantized() returns those values for the reference model. The
format itself is described in kvbound.weight_format.

Usage: python -m kvbound.quantize tinyllama-1.1b --scheme int4_g --group-size 128
"""

import argparse
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from kvbound.models import load_spec, sha256_file
from kvbound.weight_format import (
    CLIPS,
    DEFAULT_GROUP_SIZE,
    INT4_G,
    INT8_PC,
    KEY_SOURCE,
    KEY_SOURCE_SHA256,
    QMAX,
    SCALE_SUFFIX,
    WeightFormat,
)

WEIGHTS_FILE = "model.safetensors"
MANIFEST_FILE = "quantize_manifest.json"
COPIED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
)

# Clip ratios the mse search tries per group, largest first; 1.0 is plain min-max.
CLIP_RATIOS = tuple(1.0 - 0.5 * i / 19 for i in range(20))
# Rows of a tensor quantized together are capped at this many elements to bound device memory.
CHUNK_ELEMENTS = 1 << 24


def f16_scales(amax: Tensor, qmax: int) -> Tensor:
    """amax / qmax rounded to F16, with 1.0 where that is 0 (all-zero or vanishing groups)."""
    scale = (amax / qmax).to(torch.float16)
    if not torch.isfinite(scale).all():
        raise ValueError("a weight scale overflows F16")
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def round_to_grid(w: Tensor, scale: Tensor, qmax: int) -> Tensor:
    """clamp(rint(w / scale), -qmax, qmax) as int8; scale has one value per row of w's last dim."""
    q = torch.round(w / scale.float().unsqueeze(-1))
    return torch.clamp(q, -qmax, qmax).to(torch.int8)


def quantize_int8_pc(w: Tensor) -> tuple[Tensor, Tensor]:
    """Per-output-channel INT8 of w [out, in]: q int8 [out, in] and scale F16 [out]."""
    w = w.float()
    scale = f16_scales(w.abs().amax(dim=1), QMAX[INT8_PC])
    return round_to_grid(w, scale, QMAX[INT8_PC]), scale


def dequantize_int8_pc(q: Tensor, scale: Tensor) -> Tensor:
    return q.float() * scale.float().unsqueeze(-1)


def check_int4_shape(shape: tuple[int, ...] | list[int], group_size: int, name: str = "") -> None:
    """Raises unless the input dimension splits into whole groups and whole bytes."""
    if len(shape) != 2 or shape[1] % group_size or shape[1] % 2:
        raise ValueError(
            f"{name or 'tensor'}: shape {tuple(shape)} does not split into int4 groups of "
            f"{group_size} along the input dimension"
        )


def pack_int4(q: Tensor) -> Tensor:
    """Packs int8 [out, in] with values in [-8, 7] into uint8 [out, in / 2], low nibble first."""
    nibbles = (q & 0xF).to(torch.uint8)
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)


def unpack_int4(packed: Tensor) -> Tensor:
    """Inverse of pack_int4: uint8 [out, in / 2] to int8 [out, in], sign-extending each nibble."""
    lo = (packed & 0xF).to(torch.int8)
    hi = (packed >> 4).to(torch.int8)
    q = torch.stack((lo, hi), dim=-1).flatten(-2)
    return torch.where(q >= 8, q - 16, q)


def mse_clip_scales(groups: Tensor, amax: Tensor) -> Tensor:
    """Per group, the F16 scale among CLIP_RATIOS * amax / 7 with the smallest squared error.

    groups is [..., group_size] float32 and amax its absolute maximum per group. A ratio replaces
    the current best only when it is strictly better, so ratio 1.0 (min-max) wins ties.
    """
    qmax = QMAX[INT4_G]
    best_scale = f16_scales(amax, qmax)
    best_error = squared_error(groups, best_scale, qmax)
    for ratio in CLIP_RATIOS[1:]:
        scale = f16_scales(ratio * amax, qmax)
        error = squared_error(groups, scale, qmax)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
    return best_scale


def squared_error(groups: Tensor, scale: Tensor, qmax: int) -> Tensor:
    """Sum over each group of (w - scale * q)^2."""
    q = round_to_grid(groups, scale, qmax)
    return (groups - q.float() * scale.float().unsqueeze(-1)).pow(2).sum(-1)


def quantize_int4_g(w: Tensor, group_size: int, clip: str = "minmax") -> tuple[Tensor, Tensor]:
    """Group-wise INT4 of w [out, in]: packed uint8 [out, in / 2], scale F16 [out, in / group]."""
    check_int4_shape(tuple(w.shape), group_size)
    if clip not in CLIPS:
        raise ValueError(f"unknown clip {clip!r}; known: {CLIPS}")
    out_features, in_features = w.shape
    groups = w.float().reshape(out_features, in_features // group_size, group_size)
    amax = groups.abs().amax(dim=-1)
    scale = mse_clip_scales(groups, amax) if clip == "mse" else f16_scales(amax, QMAX[INT4_G])
    q = round_to_grid(groups, scale, QMAX[INT4_G]).reshape(out_features, in_features)
    return pack_int4(q), scale


def dequantize_int4_g(packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    q = unpack_int4(packed).float()
    out_features, in_features = q.shape
    groups = q.reshape(out_features, in_features // group_size, group_size)
    return (groups * scale.float().unsqueeze(-1)).reshape(out_features, in_features)


def quantize(w: Tensor, fmt: WeightFormat) -> tuple[Tensor, Tensor]:
    if fmt.scheme == INT8_PC:
        return quantize_int8_pc(w)
    return quantize_int4_g(w, fmt.group_size, fmt.clip)


def dequantize(q: Tensor, scale: Tensor, fmt: WeightFormat) -> Tensor:
    if fmt.scheme == INT8_PC:
        return dequantize_int8_pc(q, scale)
    return dequantize_int4_g(q, scale, fmt.group_size)


def quantize_tensor(
    w: Tensor, fmt: WeightFormat, device: str
) -> tuple[Tensor, Tensor, float | None]:
    """Quantizes w [out, in] a block of rows at a time on `device`.

    Returns q and scale on the CPU and the SQNR in dB (None when there is no error to measure).
    """
    rows = max(1, CHUNK_ELEMENTS // w.shape[1])
    qs, scales = [], []
    signal = noise = 0.0
    for start in range(0, w.shape[0], rows):
        chunk = w[start : start + rows].to(device).float()
        q, scale = quantize(chunk, fmt)
        error = chunk - dequantize(q, scale, fmt)
        signal += chunk.double().pow(2).sum().item()
        noise += error.double().pow(2).sum().item()
        qs.append(q.cpu())
        scales.append(scale.cpu())
    sqnr = 10 * math.log10(signal / noise) if noise > 0 else None
    return torch.cat(qs), torch.cat(scales), sqnr


def to_f16(tensor: Tensor, name: str) -> Tensor:
    half = tensor.to(torch.float16)
    if not torch.isfinite(half).all():
        raise ValueError(f"{name}: values overflow FP16")
    return half


def convert(
    src_dir: Path,
    out_dir: Path,
    scheme: str,
    group_size: int = DEFAULT_GROUP_SIZE,
    clip: str = "minmax",
    source: str = "",
    device: str | None = None,
) -> dict[str, Any]:
    """Writes a quantized copy of the plain checkpoint in src_dir to out_dir; returns its manifest.

    Every 2-D tensor is quantized under its own name with a "<name>.scale" companion; 1-D tensors
    are stored as F16. Tensors are converted one at a time, a block of rows at a time on `device`
    (the GPU when available). out_dir also gets config.json, the tokenizer files and
    quantize_manifest.json (format, per-tensor SQNR and bits per weight, sizes, source hash).
    """
    fmt = WeightFormat.make(scheme, group_size, clip)
    if not fmt.quantized:
        raise ValueError(f"{scheme!r} is not a quantized scheme")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    src = src_dir / WEIGHTS_FILE
    started = time.perf_counter()

    tensors: dict[str, Tensor] = {}
    report: dict[str, dict[str, Any]] = {}
    n_params = n_bytes = 0
    with safe_open(str(src), framework="pt") as f:
        if WeightFormat.from_metadata(f.metadata()).quantized:
            raise ValueError(f"{src} is already quantized")
        names = list(f.keys())
        shapes = {name: f.get_slice(name).get_shape() for name in names}
        for name, shape in shapes.items():
            if len(shape) not in (1, 2):
                raise ValueError(f"{name}: cannot quantize a {len(shape)}-D tensor")
            if len(shape) == 2 and fmt.scheme == INT4_G:
                check_int4_shape(shape, fmt.group_size, name)

        for name in names:
            w = f.get_tensor(name)
            if w.ndim == 1:
                tensors[name] = to_f16(w, name)
                continue
            q, scale, sqnr = quantize_tensor(w, fmt, device)
            tensors[name], tensors[name + SCALE_SUFFIX] = q, scale
            n_params += w.numel()
            n_bytes += q.nbytes + scale.nbytes
            report[name] = {
                "shape": list(w.shape),
                "sqnr_db": sqnr,
                "bits_per_weight": fmt.bits_per_weight(*w.shape),
            }

    source_sha256 = sha256_file(src)
    metadata = fmt.to_metadata()
    if source:
        metadata[KEY_SOURCE] = source
    metadata[KEY_SOURCE_SHA256] = source_sha256
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / WEIGHTS_FILE), metadata=metadata)
    for file in COPIED_FILES:
        if (src_dir / file).exists():
            shutil.copyfile(src_dir / file, out_dir / file)

    sqnrs = [t["sqnr_db"] for t in report.values() if t["sqnr_db"] is not None]
    manifest = {
        "format": metadata,
        "variant": fmt.variant,
        "source_sha256": source_sha256,
        "quantized_params": n_params,
        "quantized_bytes": n_bytes,
        "bits_per_weight": 8 * n_bytes / n_params if n_params else 0.0,
        "file_bytes": (out_dir / WEIGHTS_FILE).stat().st_size,
        "mean_sqnr_db": statistics.fmean(sqnrs) if sqnrs else None,
        "min_sqnr_db": min(sqnrs) if sqnrs else None,
        "seconds": time.perf_counter() - started,
        "device": device,
        "torch_version": torch.__version__,
        "tensors": report,
    }
    with (out_dir / MANIFEST_FILE).open("w", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def stored_names(keys: list[str], fmt: WeightFormat) -> list[str]:
    """The checkpoint's tensor names without the scale companions of quantized tensors."""
    if not fmt.quantized:
        return list(keys)
    present = set(keys)
    return [
        k for k in keys if not (k.endswith(SCALE_SUFFIX) and k[: -len(SCALE_SUFFIX)] in present)
    ]


def read_tensor(f: Any, name: str, fmt: WeightFormat) -> Tensor:
    """Tensor `name` of an open safetensors file as the float32 values the engine computes with.

    Quantized tensors are dequantized from their stored integers and F16 scales; float tensors are
    rounded to F16, the engine's weight precision.
    """
    tensor = f.get_tensor(name)
    if not tensor.dtype.is_floating_point:
        expected = {torch.int8: INT8_PC, torch.uint8: INT4_G}.get(tensor.dtype)
        if expected != fmt.scheme:
            raise ValueError(f"{name}: dtype {tensor.dtype} does not belong to scheme {fmt.scheme}")
        scale = f.get_tensor(name + SCALE_SUFFIX)
        rows, cols = tensor.shape
        if fmt.scheme == INT8_PC:
            scale_shape: tuple[int, ...] = (rows,)
        else:
            check_int4_shape((rows, 2 * cols), fmt.group_size, name)
            scale_shape = (rows, 2 * cols // fmt.group_size)
        if scale.dtype != torch.float16 or tuple(scale.shape) != scale_shape:
            raise ValueError(
                f"{name}{SCALE_SUFFIX}: expected F16 {scale_shape}, got {scale.dtype} "
                f"{tuple(scale.shape)}"
            )
        return dequantize(tensor, scale, fmt)
    return to_f16(tensor, name).float()


def load_dequantized(model_dir: Path) -> tuple[WeightFormat, dict[str, Tensor]]:
    """Every tensor of <model_dir>/model.safetensors, plain or quantized, as float32.

    Names and shapes are the original ones (no scale entries); the values are exactly what the
    engine's kernels multiply by.
    """
    with safe_open(str(Path(model_dir) / WEIGHTS_FILE), framework="pt") as f:
        fmt = WeightFormat.from_metadata(f.metadata())
        names = stored_names(list(f.keys()), fmt)
        return fmt, {name: read_tensor(f, name, fmt) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--scheme", choices=[INT8_PC, INT4_G], required=True)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE, help="int4_g only")
    parser.add_argument("--clip", choices=CLIPS, default="minmax", help="mse: int4_g only")
    parser.add_argument("--device", default=None, help="default: cuda when available")
    args = parser.parse_args()
    if args.scheme == INT8_PC and args.clip != "minmax":
        parser.error("--clip mse applies to int4_g only")

    spec = load_spec(args.model)
    fmt = WeightFormat.make(args.scheme, args.group_size, args.clip)
    out_dir = spec.variant_dir(fmt.variant)
    m = convert(
        spec.checkpoint_dir,
        out_dir,
        args.scheme,
        args.group_size,
        args.clip,
        source=f"{spec.hf_repo}@{spec.revision}",
        device=args.device,
    )
    print(
        f"{spec.name} {fmt.variant}: SQNR mean {m['mean_sqnr_db']:.2f} dB, "
        f"min {m['min_sqnr_db']:.2f} dB | {m['bits_per_weight']:.3f} bits per weight | "
        f"{m['quantized_bytes'] / 1e9:.3f} GB quantized, file {m['file_bytes'] / 1e9:.3f} GB | "
        f"{m['seconds']:.1f} s\n{out_dir}"
    )


if __name__ == "__main__":
    main()
