"""Weight quantization: INT8 per output channel and INT4 in groups, written in the kvbound format.

Both schemes are symmetric round-to-nearest with F16 scales. Each scale is rounded to F16 before
the integers are computed from it, so a stored (q, scale) pair reproduces exactly the values the
engine multiplies by, and load_dequantized() returns those values for the reference model. The
format itself is described in kvbound.weight_format.

With --awq, int4_g weights are rescaled by activation statistics of calibration text before they
are rounded, and clipped by their effect on the output (kvbound.awq). That search runs text
through a Llama, so it does not apply to a Whisper checkpoint.

With --gptq, int4_g integers are chosen by GPTQ instead of round-to-nearest, on the same grid
(kvbound.gptq), alone or after the AWQ scales and clip ranges of --awq. It too calibrates on text
through a Llama.

Usage: python -m kvbound.quantize tinyllama-1.1b --scheme int4_g --group-size 128
       python -m kvbound.quantize tinyllama-1.1b --scheme int4_g --awq --gptq
       python -m kvbound.quantize whisper-base-ar-quran --scheme int8_pc
"""

import argparse
import dataclasses
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from kvbound.models import GENERATION_FILE, load_spec, sha256_file
from kvbound.ref import load_llama
from kvbound.weight_format import (
    CALIB_KEYS,
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

if TYPE_CHECKING:
    from kvbound.gptq import GptqConfig

WEIGHTS_FILE = "model.safetensors"
MANIFEST_FILE = "quantize_manifest.json"
COPIED_FILES = (
    "config.json",
    GENERATION_FILE,
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
)

# Tensors of the transformer layers, the ones activation scales are folded into.
LAYER_PREFIX = "model.layers."
# Tensors named like this are position tables: 2-D, but added to activations, never multiplied.
POSITION_TABLE = "embed_positions"

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
    if clip not in CLIPS:
        raise ValueError(f"unknown clip {clip!r}; known: {CLIPS}")
    WeightFormat.make(INT4_G, group_size, clip).check_shape(tuple(w.shape))
    out_features, in_features = w.shape
    groups = w.float().reshape(out_features, in_features // group_size, group_size)
    amax = groups.abs().amax(dim=-1)
    scale = mse_clip_scales(groups, amax) if clip == "mse" else f16_scales(amax, QMAX[INT4_G])
    return quantize_int4_g_with_scales(w, scale), scale


def quantize_int4_g_with_scales(w: Tensor, scale: Tensor) -> Tensor:
    """w [out, in] packed as uint8 [out, in / 2] against given F16 scales [out, in / group].

    Weights beyond 7 scales saturate, which is how a scale below a group's min-max one clips.
    """
    out_features, in_features = w.shape
    if scale.dtype != torch.float16 or scale.ndim != 2 or scale.shape[0] != out_features:
        raise ValueError(
            f"expected F16 scales [{out_features}, groups], got {scale.dtype} {tuple(scale.shape)}"
        )
    groups = w.float().reshape(out_features, scale.shape[1], -1)
    q = round_to_grid(groups, scale, QMAX[INT4_G]).reshape(out_features, in_features)
    return pack_int4(q)


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
    w: Tensor, fmt: WeightFormat, device: str, given_scale: Tensor | None = None
) -> tuple[Tensor, Tensor, float | None]:
    """Quantizes w [out, in] a block of rows at a time on `device`.

    The scales are those of fmt's clip, or given_scale (int4_g, F16 [out, in / group]) when set.
    Returns q and scale on the CPU and the SQNR in dB (None when there is no error to measure).
    """
    rows = max(1, CHUNK_ELEMENTS // w.shape[1])
    qs, scales = [], []
    signal = noise = 0.0
    for start in range(0, w.shape[0], rows):
        chunk = w[start : start + rows].to(device).float()
        if given_scale is None:
            q, scale = quantize(chunk, fmt)
        else:
            scale = given_scale[start : start + rows].to(device)
            q = quantize_int4_g_with_scales(chunk, scale)
        error = chunk - dequantize(q, scale, fmt)
        signal += chunk.double().pow(2).sum().item()
        noise += error.double().pow(2).sum().item()
        qs.append(q.cpu())
        scales.append(scale.cpu())
    sqnr = 10 * math.log10(signal / noise) if noise > 0 else None
    return torch.cat(qs), torch.cat(scales), sqnr


def sqnr_db(w: Tensor, w_hat: Tensor) -> float | None:
    """10 log10(|w|^2 / |w - w_hat|^2), None when w_hat is exact."""
    noise = (w.double() - w_hat.double()).pow(2).sum().item()
    return 10 * math.log10(w.double().pow(2).sum().item() / noise) if noise > 0 else None


def is_matrix(name: str, shape: tuple[int, ...] | list[int]) -> bool:
    """Whether a checkpoint tensor is a matrix activations are multiplied by: the tensors that
    are quantized. Biases, norm parameters, convolution weights [out, in, kernel] and position
    tables are not."""
    return len(shape) == 2 and POSITION_TABLE not in name


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
    awq: Tensor | None = None,
    calib_meta: dict[str, str] | None = None,
    gptq: Tensor | None = None,
    gptq_config: "GptqConfig | None" = None,
) -> dict[str, Any]:
    """Writes a quantized copy of the plain checkpoint in src_dir to out_dir; returns its manifest.

    Every matrix (is_matrix) is quantized under its own name with a "<name>.scale" companion;
    every other tensor is stored as F16. Tensors are converted one at a time, a block of rows at
    a time on `device` (the GPU when available). out_dir also gets config.json, the decoding
    rules and tokenizer files and quantize_manifest.json (format, per-tensor SQNR and bits per
    weight, sizes, source hash).

    awq, calibration token ids [n, len], makes an activation-aware int4_g checkpoint (kvbound.awq):
    scales searched on those tokens on `device` are folded into the layers' tensors before they
    are quantized, one layer in memory at a time, and clip "awq" also takes the clip scales the
    search finds. SQNR is then relative to the scaled weights. out_dir also gets
    awq_scales.safetensors and the manifest an "awq" section; calib_meta holds the header keys
    that say where the tokens came from. A Whisper checkpoint takes no awq.

    gptq, calibration token ids [n, len] (the same as awq's when both are set), chooses the
    int4_g integers of every matrix but the embedding and a tied lm_head by GPTQ
    (kvbound.gptq.quantize_model, run with gptq_config, a kvbound.gptq.GptqConfig), on the AWQ
    scales and clip scales when awq is set too. The manifest gets a "gptq" section.
    """
    fmt = WeightFormat.make(
        scheme,
        group_size,
        clip,
        "awq" if awq is not None else "",
        "gptq" if gptq is not None else "",
    )
    if not fmt.quantized:
        raise ValueError(f"{scheme!r} is not a quantized scheme")
    calibrated = awq is not None or gptq is not None
    if set(calib_meta or {}) - set(CALIB_KEYS) or (calib_meta and not calibrated):
        raise ValueError(f"calib_meta takes the keys {CALIB_KEYS}, with awq or gptq only")
    if awq is not None and gptq is not None and not torch.equal(awq, gptq):
        raise ValueError("awq and gptq calibrate on the same tokens")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    if calibrated and config.get("model_type") == "whisper":
        raise ValueError(
            "awq and gptq calibrate on text through a Llama; a Whisper checkpoint takes none"
        )
    src = src_dir / WEIGHTS_FILE
    started = time.perf_counter()

    tensors: dict[str, Tensor] = {}
    report: dict[str, dict[str, Any]] = {}
    n_params = n_bytes = 0
    # With awq: the tensors of the layer being written with its scales folded in, and what the
    # search found.
    scaled: dict[str, Tensor] = {}
    awq_result = awq_report = None
    # With gptq: the integers and scales GPTQ chose, and its report.
    gptq_result = gptq_report = None
    with safe_open(str(src), framework="pt") as f:
        if WeightFormat.from_metadata(f.metadata()).quantized:
            raise ValueError(f"{src} is already quantized")
        names = list(f.keys())
        shapes = {name: f.get_slice(name).get_shape() for name in names}
        for name, shape in shapes.items():
            if is_matrix(name, shape):
                fmt.check_shape(shape, name)

        on_cuda = torch.device(device).type == "cuda"
        model = load_llama(src_dir, "fp16") if calibrated else None
        if awq is not None:
            # Imported here because kvbound.awq builds on this module.
            from kvbound.awq import (  # noqa: PLC0415
                SCALES_FILE,
                collect_and_search,
                save_scales,
                scale_layer,
            )

            assert model is not None
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            awq_result = collect_and_search(model, awq, group_size, device, clip == "awq")
            awq_report = {
                "calib_windows": awq.shape[0],
                "calib_window": awq.shape[1],
                "search_seconds": time.perf_counter() - started,
                "layers": awq_result.report,
            }
            if on_cuda:
                awq_report["cuda_max_memory_allocated"] = torch.cuda.max_memory_allocated(device)
        if gptq is not None:
            # Imported here because kvbound.gptq builds on this module.
            from kvbound.gptq import GptqConfig, fold_awq, quantize_model  # noqa: PLC0415

            assert model is not None
            gptq_config = gptq_config or GptqConfig()
            gptq_started = time.perf_counter()
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            if awq_result:
                fold_awq(model, awq_result.s_attn, awq_result.s_mlp, awq_result.s_down)
            gptq_result = quantize_model(
                model,
                gptq,
                group_size,
                device,
                gptq_config,
                awq_result.clip_scales if awq_result else None,
            )
            gptq_report = {
                "config": dataclasses.asdict(gptq_config),
                "calib_windows": gptq.shape[0],
                "calib_window": gptq.shape[1],
                "seconds": time.perf_counter() - gptq_started,
                "layers": gptq_result.report,
            }
            if on_cuda:
                gptq_report["cuda_max_memory_allocated"] = torch.cuda.max_memory_allocated(device)
                gptq_report["cuda_max_memory_reserved"] = torch.cuda.max_memory_reserved(device)
        del model
        # The scales of a tensor the clip search did not visit are min-max ones.
        clip_scales = awq_result.clip_scales if awq_result else {}
        unclipped = WeightFormat.make(scheme, group_size) if awq_result else fmt

        for name in names:
            if awq_result and name.startswith(LAYER_PREFIX) and name not in scaled:
                # The first tensor of a layer: its scales are folded into the whole layer.
                i = int(name.split(".")[2])
                prefix = f"{LAYER_PREFIX}{i}."
                scaled = {n: f.get_tensor(n).float() for n in names if n.startswith(prefix)}
                scaled |= scale_layer(
                    scaled, i, awq_result.s_attn[i], awq_result.s_mlp[i], awq_result.s_down[i]
                )
            w = scaled.pop(name) if name in scaled else f.get_tensor(name)
            if not is_matrix(name, shapes[name]):
                tensors[name] = to_f16(w, name)
                continue
            if gptq_result and name in gptq_result.quantized:
                q, scale = gptq_result.quantized.pop(name)
                sqnr = sqnr_db(w.float(), dequantize_int4_g(q, scale, group_size))
            else:
                q, scale, sqnr = quantize_tensor(w, unclipped, device, clip_scales.get(name))
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
    metadata.update(calib_meta or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / WEIGHTS_FILE), metadata=metadata)
    for file in COPIED_FILES:
        if (src_dir / file).exists():
            shutil.copyfile(src_dir / file, out_dir / file)
    if awq is not None and awq_result is not None:
        save_scales(out_dir / SCALES_FILE, awq_result, awq)

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
    if awq_report is not None:
        manifest["awq"] = awq_report
    if gptq_report is not None:
        manifest["gptq"] = gptq_report
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
            fmt.check_shape((rows, cols), name)
            scale_shape: tuple[int, ...] = (rows,)
        else:
            fmt.check_shape((rows, 2 * cols), name)
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
    parser.add_argument(
        "--clip", choices=CLIPS, default=None, help="int4_g only; default: awq with --awq"
    )
    parser.add_argument(
        "--awq", action="store_true", help="int4_g only: activation-aware scales (kvbound.awq)"
    )
    parser.add_argument(
        "--gptq", action="store_true", help="int4_g only: GPTQ rounding (kvbound.gptq)"
    )
    parser.add_argument(
        "--gptq-damp", type=float, default=None, help="--gptq: damping, of mean diag H"
    )
    parser.add_argument(
        "--gptq-input-order",
        action="store_true",
        help="--gptq: round columns in input order with group scales set as reached",
    )
    parser.add_argument("--calib-windows", type=int, default=128, help="--awq, --gptq: windows")
    parser.add_argument("--calib-len", type=int, default=512, help="--awq, --gptq: window length")
    parser.add_argument("--calib-seed", type=int, default=0, help="--awq, --gptq: which windows")
    parser.add_argument("--device", default=None, help="default: cuda when available")
    args = parser.parse_args()
    clip = args.clip or ("awq" if args.awq else "minmax")
    if args.scheme == INT8_PC and (args.awq or args.gptq or clip != "minmax"):
        parser.error("--clip, --awq and --gptq apply to int4_g only")
    if (args.awq or args.gptq) and clip == "mse":
        parser.error("--awq and --gptq take --clip awq or --clip minmax")
    if (args.gptq_damp is not None or args.gptq_input_order) and not args.gptq:
        parser.error("--gptq-damp and --gptq-input-order need --gptq")
    if clip == "awq" and not args.awq:
        parser.error("--clip awq needs --awq")

    spec = load_spec(args.model)
    if (args.awq or args.gptq) and spec.family != "llama":
        parser.error(
            f"--awq and --gptq calibrate on text through a Llama; {spec.name} is a {spec.family}"
        )
    fmt = WeightFormat.make(
        args.scheme,
        args.group_size,
        clip,
        "awq" if args.awq else "",
        "gptq" if args.gptq else "",
    )
    out_dir = spec.variant_dir(fmt.variant)
    windows = calib_meta = gptq_config = None
    if args.gptq:
        from kvbound.gptq import DEFAULT_DAMP, GptqConfig  # noqa: PLC0415

        damp = DEFAULT_DAMP if args.gptq_damp is None else args.gptq_damp
        gptq_config = GptqConfig(damp=damp, act_order=not args.gptq_input_order)
    if args.awq or args.gptq:
        # Imported here because kvbound.awq builds on this module.
        from kvbound.awq import calibration_meta, calibration_windows  # noqa: PLC0415

        calib = (args.calib_windows, args.calib_len, args.calib_seed)
        windows, calib_meta = calibration_windows(spec, *calib), calibration_meta(*calib)
    m = convert(
        spec.checkpoint_dir,
        out_dir,
        args.scheme,
        args.group_size,
        clip,
        source=f"{spec.hf_repo}@{spec.revision}",
        device=args.device,
        awq=windows if args.awq else None,
        calib_meta=calib_meta,
        gptq=windows if args.gptq else None,
        gptq_config=gptq_config,
    )
    print(
        f"{spec.name} {fmt.variant}: SQNR mean {m['mean_sqnr_db']:.2f} dB, "
        f"min {m['min_sqnr_db']:.2f} dB | {m['bits_per_weight']:.3f} bits per weight | "
        f"{m['quantized_bytes'] / 1e9:.3f} GB quantized, file {m['file_bytes'] / 1e9:.3f} GB | "
        f"{m['seconds']:.1f} s\n{out_dir}"
    )
    if args.awq:
        layers = m["awq"]["layers"]
        for group in ("attn", "mlp", "down"):
            found = [layer[group] for layer in layers]
            loss, rtn = (math.fsum(g[key] for g in found) for key in ("loss", "loss_rtn"))
            print(
                f"awq {group}: mean alpha {statistics.fmean(g['alpha'] for g in found):.2f}, "
                f"calibration loss {loss / rtn:.3f} of plain rounding's, "
                f"largest scaled activation {max(g['max_abs_input'] for g in found):.1f}"
            )
        peak = m["awq"].get("cuda_max_memory_allocated", 0)
        print(f"awq search: {m['awq']['search_seconds']:.1f} s, peak {peak / 1e9:.2f} GB on GPU")
    if args.gptq:
        from kvbound.gptq import stage_summary  # noqa: PLC0415

        for stage, (vs_rtn, vs_signal) in stage_summary(m["gptq"]["layers"]).items():
            print(
                f"gptq {stage}: output error {vs_rtn:.3f} of rounding to nearest's, "
                f"{vs_signal:.2e} of the output"
            )
        peak = m["gptq"].get("cuda_max_memory_allocated", 0)
        print(f"gptq: {m['gptq']['seconds']:.1f} s, peak {peak / 1e9:.2f} GB on GPU")


if __name__ == "__main__":
    main()
