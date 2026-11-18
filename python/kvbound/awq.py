"""Activation-aware scales and clip ranges for int4_g weights (AWQ, Lin et al., 2023).

Rounding a matrix to INT4 treats every weight alike, but a weight's error reaches the output
multiplied by the activation it meets. AWQ rescales the input channels of a matrix before it is
quantized, W -> W diag(s), and divides the activations that reach it by s, so the channels with
large activations get a finer step. The division is folded into whatever produces those
activations, so before quantization the model computes the same function:

  q/k/v    s over the hidden size, folded into input_layernorm's weight
  gate/up  s over the hidden size, folded into post_attention_layernorm's weight
  down     s over the intermediate size, folded into up_proj's rows, since
           silu(g) * (u / s) = (silu(g) * u) / s

o_proj (its input is v's output repeated over query heads), the embedding, the final norm and
lm_head keep their values. Norm weights are stored F16: the stored weight is gamma' =
f16(gamma / s) and the matrices are scaled by s_eff = gamma / gamma', the scale gamma' realizes,
so F16 rounding of the norm costs nothing.

Each group's s is x_mean^alpha, x_mean the mean |activation| per channel over the calibration
tokens, with the alpha among N_GRID that least changes the group's output. With clipping, each
matrix but q and k then gets per group the clip ratio that least changes its output on a sample
of the scaled activations. Both follow mit-han-lab/llm-awq (auto_scale.py, auto_clip.py).

The result is an ordinary int4_g checkpoint; see convert() in kvbound.quantize.

Usage: python -m kvbound.quantize tinyllama-1.1b --scheme int4_g --awq
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from kvbound.models import ModelSpec
from kvbound.quantize import WEIGHTS_FILE, f16_scales, round_to_grid, to_f16
from kvbound.ref.model import Block, Llama, rope_tables
from kvbound.weight_format import (
    INT4_G,
    KEY_CALIB_DATASET,
    KEY_CALIB_REVISION,
    KEY_CALIB_SEED,
    KEY_CALIB_SHA256,
    KEY_CALIB_WINDOW,
    KEY_CALIB_WINDOWS,
    QMAX,
    WeightFormat,
)

# The scales and calibration tokens of an AWQ checkpoint, beside its model.safetensors. The
# engine and the reference model do not read it, since the scales are already folded into the
# weights; kvbound.evals.sites uses the scales to compare norm outputs with unscaled goldens.
SCALES_FILE = "awq_scales.safetensors"
CALIB_IDS = "calib_ids"

# Alphas the scale search tries: i / N_GRID for i in 0..N_GRID - 1. Alpha 0 is s = 1.
N_GRID = 20
# Clip ratios the clip search tries per group, largest first; 1.0 is plain min-max.
CLIP_RATIOS = tuple(1.0 - i / 20 for i in range(10))
# Calibration tokens the clip search weighs errors with, taken at a regular stride.
CLIP_TOKENS = 4096
# Calibration windows per forward pass.
CHUNK_WINDOWS = 8
# Smallest normal F16 magnitude; below it a rescaled norm weight loses precision.
F16_MIN_NORMAL = 2.0**-14


def pick_windows(ids: list[int], n: int, length: int, seed: int) -> Tensor:
    """n of the non-overlapping `length`-token windows of ids, drawn by a seeded permutation:
    int32 [n, length]. A tail shorter than a window is dropped."""
    n_windows = len(ids) // length
    if n_windows < n:
        raise ValueError(f"{len(ids)} tokens hold {n_windows} windows of {length}, fewer than {n}")
    stream = torch.tensor(ids[: n_windows * length], dtype=torch.int32).reshape(n_windows, length)
    picks = torch.randperm(n_windows, generator=torch.Generator().manual_seed(seed))[:n]
    return stream[picks]


def calibration_windows(spec: ModelSpec, n: int = 128, length: int = 512, seed: int = 0) -> Tensor:
    """Calibration tokens for `spec`: n windows of the WikiText-2 train split, int32 [n, length].

    The text is joined and tokenized exactly as the perplexity evaluation does with the test
    split, which the calibration never sees.
    """
    # Imported here so the search does not load the tokenizer and the dataset reader.
    from kvbound.evals import perplexity  # noqa: PLC0415

    path = perplexity.fetch_dataset(perplexity.TRAIN_FILE, perplexity.TRAIN_SHA256)
    ids = perplexity.tokenize(perplexity.load_text(path), spec.checkpoint_dir)
    return pick_windows(ids, n, length, seed)


def calibration_meta(n: int = 128, length: int = 512, seed: int = 0) -> dict[str, str]:
    """The header keys that say where calibration_windows(spec, n, length, seed) come from."""
    from kvbound.evals import perplexity  # noqa: PLC0415

    return {
        KEY_CALIB_DATASET: f"{perplexity.DATASET_REPO}/{perplexity.TRAIN_FILE}",
        KEY_CALIB_REVISION: perplexity.DATASET_REVISION,
        KEY_CALIB_SHA256: perplexity.TRAIN_SHA256,
        KEY_CALIB_WINDOWS: str(n),
        KEY_CALIB_WINDOW: str(length),
        KEY_CALIB_SEED: str(seed),
    }


@dataclass
class AwqResult:
    """What collect_and_search finds, one list entry per layer.

    s_attn and s_mlp [hidden] scale the inputs of q/k/v and of gate/up, as the rescaled norm
    weights realize them (s_eff of rescaled_norm); s_down [intermediate] scales the input of
    down_proj. clip_scales maps the Hugging Face name of each clipped matrix to its F16 scales
    [out, in / group_size]. report holds per layer each group's alpha, losses and ranges, and
    each clipped matrix's loss before and after.
    """

    group_size: int
    s_attn: list[Tensor] = field(default_factory=list)
    s_mlp: list[Tensor] = field(default_factory=list)
    s_down: list[Tensor] = field(default_factory=list)
    clip_scales: dict[str, Tensor] = field(default_factory=dict)
    report: list[dict[str, Any]] = field(default_factory=list)


def alpha_scales(x_mean: Tensor, alpha: float) -> Tensor:
    """s = x_mean^alpha, floored at 1e-4 and normalized so that max(s) * min(s) = 1; float32."""
    s = x_mean.pow(alpha).clamp(min=1e-4)
    return (s / (s.max() * s.min()).sqrt()).float()


def rescaled_norm(gamma: Tensor, s: Tensor) -> tuple[Tensor, Tensor] | None:
    """The norm weight that divides the norm's output by s, and the scale it realizes.

    Returns gamma' = f16(gamma / s) as float32 and s_eff = gamma / gamma' (s where gamma is 0):
    gamma' * s_eff is gamma however coarsely F16 rounds gamma / s. None when a weight that s
    changes overflows F16 or falls below its normal range. Applied to an F16 gamma and its own
    s_eff, it returns the same pair again.
    """
    gamma_q = (gamma / s).to(torch.float16).float()
    lost = (s != 1) & (gamma != 0) & (gamma_q.abs() < F16_MIN_NORMAL)
    if lost.any() or not torch.isfinite(gamma_q).all():
        return None
    return gamma_q, torch.where(gamma_q != 0, gamma / gamma_q, s)


def scale_layer(
    tensors: dict[str, Tensor], i: int, s_attn: Tensor, s_mlp: Tensor, s_down: Tensor
) -> dict[str, Tensor]:
    """The tensors of layer i that its scales change, with the scales folded in.

    `tensors` holds float32 values under Hugging Face names. Norm weights are taken at F16, the
    precision checkpoints store them at, and come out F16-representable.
    """
    layer = f"model.layers.{i}"
    out: dict[str, Tensor] = {}
    for norm, s, projs in (
        ("input_layernorm", s_attn, ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        ("post_attention_layernorm", s_mlp, ("mlp.gate_proj", "mlp.up_proj")),
    ):
        name = f"{layer}.{norm}.weight"
        scaled = rescaled_norm(to_f16(tensors[name], name).float(), s)
        if scaled is None:
            raise ValueError(f"{name}: divided by its scales it leaves the normal F16 range")
        out[name], s_eff = scaled
        for proj in projs:
            out[f"{layer}.{proj}.weight"] = tensors[f"{layer}.{proj}.weight"] * s_eff
    out[f"{layer}.mlp.up_proj.weight"] = out[f"{layer}.mlp.up_proj.weight"] / s_down[:, None]
    out[f"{layer}.mlp.down_proj.weight"] = tensors[f"{layer}.mlp.down_proj.weight"] * s_down
    return out


def apply_scales(tensors: dict[str, Tensor], result: AwqResult) -> dict[str, Tensor]:
    """`tensors` (Hugging Face names) as float32 with the scales of `result` folded in.

    The scaled model computes the same function as the original with F16 norm weights, up to
    float32 rounding. Every layer of `result` must be in `tensors`; tensors that no scale touches
    pass through.
    """
    out = {name: tensor.float() for name, tensor in tensors.items()}
    scales = zip(result.s_attn, result.s_mlp, result.s_down, strict=True)
    for i, (s_attn, s_mlp, s_down) in enumerate(scales):
        out.update(scale_layer(out, i, s_attn, s_mlp, s_down))
    return out


def fake_quantize(w: Tensor, group_size: int) -> Tensor:
    """w [out, in] as int4_g with min-max scales stores it, dequantized."""
    groups = w.reshape(w.shape[0], -1, group_size)
    scale = f16_scales(groups.abs().amax(dim=-1), QMAX[INT4_G])
    q = round_to_grid(groups, scale, QMAX[INT4_G])
    return (q.float() * scale.float().unsqueeze(-1)).reshape(w.shape)


def act_clip_scales(groups: Tensor, x_groups: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per group, the F16 scale among CLIP_RATIOS * amax / 7 that least changes the output.

    groups is [out, n_groups, group_size] float32 and x_groups [tokens, n_groups, group_size] the
    activations each group multiplies. A group's error is the sum over tokens of
    (x . (w - w_hat))^2, its own share of the output error, computed as d G d with d = w - w_hat
    and G the Gram matrix of the group's activations. A ratio replaces the current best only when
    it is strictly better, so ratio 1.0 (min-max) wins ties.

    Returns the scales [out, n_groups] and, in float64, the errors under min-max scales and
    under the returned ones.
    """
    qmax = QMAX[INT4_G]
    x = x_groups.double().transpose(0, 1)
    gram = x.transpose(1, 2) @ x
    amax = groups.abs().amax(dim=-1)

    def output_error(scale: Tensor) -> Tensor:
        deq = round_to_grid(groups, scale, qmax).float() * scale.float().unsqueeze(-1)
        d = (groups - deq).double()
        return (torch.einsum("ogi,gij->ogj", d, gram) * d).sum(-1)

    best_scale = f16_scales(amax, qmax)
    minmax_error = best_error = output_error(best_scale)
    for ratio in CLIP_RATIOS[1:]:
        scale = f16_scales(ratio * amax, qmax)
        error = output_error(scale)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
    return best_scale, minmax_error, best_error


def channel_stats(features: Tensor) -> tuple[Tensor, Tensor]:
    """Per channel of features [n, len, channels]: the mean |x| in float64 and the largest |x|."""
    total = torch.zeros(features.shape[-1], dtype=torch.float64, device=features.device)
    largest = torch.zeros(features.shape[-1], device=features.device)
    for chunk in features.split(CHUNK_WINDOWS):
        magnitude = chunk.float().abs()
        total += magnitude.double().sum(dim=(0, 1))
        largest = torch.maximum(largest, magnitude.amax(dim=(0, 1)))
    if not torch.isfinite(largest).all():
        raise ValueError("calibration activations overflow FP16")
    return total / (features.shape[0] * features.shape[1]), largest


def output_mse(fn: Callable[[Tensor], Tensor], inputs: Tensor, target: Tensor) -> float:
    """Mean squared error of fn(inputs) against target, CHUNK_WINDOWS windows at a time, the
    squares summed in float64."""
    total = 0.0
    for start in range(0, inputs.shape[0], CHUNK_WINDOWS):
        rows = slice(start, start + CHUNK_WINDOWS)
        total += (fn(inputs[rows].float()) - target[rows]).double().pow(2).sum().item()
    return total / target.numel()


def search_scales(
    x_mean: Tensor,
    gamma: Tensor | None,
    linears: Sequence[nn.Linear],
    group_size: int,
    loss_fn: Callable[[], float],
) -> tuple[Tensor, list[float]]:
    """The scales among N_GRID alphas with the smallest loss_fn(), and the loss of every alpha.

    A trial sets the weights of `linears` to Q(W s) / s, what they act as when their inputs are
    divided by s and they are quantized as W s. gamma is the weight of the norm that feeds them,
    or None: with it s is the scale the rescaled norm realizes, and an alpha whose norm cannot
    be stored scores inf. A later alpha wins only when strictly better and alpha 0 (s = 1, plain
    rounding) is tried first, so the chosen loss is never above plain rounding's.
    """
    originals = [linear.weight.clone() for linear in linears]
    best_s, best_loss = torch.ones_like(x_mean, dtype=torch.float32), math.inf
    losses = []
    for i in range(N_GRID):
        s = alpha_scales(x_mean, i / N_GRID)
        if gamma is not None:
            scaled = rescaled_norm(gamma, s)
            if scaled is None:
                losses.append(math.inf)
                continue
            s = scaled[1]
        for linear, w in zip(linears, originals, strict=True):
            linear.weight.copy_(fake_quantize(w * s, group_size) / s)
        loss = loss_fn()
        losses.append(loss if math.isfinite(loss) else math.inf)
        if losses[-1] < best_loss:
            best_s, best_loss = s, losses[-1]
    for linear, w in zip(linears, originals, strict=True):
        linear.weight.copy_(w)
    return best_s, losses


def group_report(
    losses: list[float], s: Tensor, x_max: Tensor, gamma: Tensor | None = None
) -> dict[str, Any]:
    """One group's entry of AwqResult.report: the chosen alpha, the loss of plain rounding and
    of the choice, the range of s, the largest activation once divided by s and, with the norm
    weight gamma that s is folded into, the smallest stored norm weight."""
    best = losses.index(min(losses))
    report: dict[str, Any] = {
        "alpha": best / N_GRID,
        "loss_rtn": losses[0],
        "loss": losses[best],
        "losses": [loss if math.isfinite(loss) else None for loss in losses],
        "s_min": s.min().item(),
        "s_max": s.max().item(),
        "max_abs_input": (x_max / s).max().item(),
    }
    if gamma is not None:
        stored = (gamma / s).to(torch.float16).float().abs()
        report["min_abs_norm"] = stored[stored != 0].min().item() if stored.any() else 0.0
    return report


def block_tensors(block: Block, i: int) -> dict[str, Tensor]:
    """The weights of block i under their Hugging Face names."""
    layer = f"model.layers.{i}"
    return {
        f"{layer}.input_layernorm.weight": block.attn_norm.weight,
        f"{layer}.self_attn.q_proj.weight": block.attn.q_proj.weight,
        f"{layer}.self_attn.k_proj.weight": block.attn.k_proj.weight,
        f"{layer}.self_attn.v_proj.weight": block.attn.v_proj.weight,
        f"{layer}.self_attn.o_proj.weight": block.attn.o_proj.weight,
        f"{layer}.post_attention_layernorm.weight": block.mlp_norm.weight,
        f"{layer}.mlp.gate_proj.weight": block.mlp.gate_proj.weight,
        f"{layer}.mlp.up_proj.weight": block.mlp.up_proj.weight,
        f"{layer}.mlp.down_proj.weight": block.mlp.down_proj.weight,
    }


def sample_tokens(features: Tensor, stride: int) -> Tensor:
    """Every stride-th token of features [n, len, channels], float32 [tokens, channels]."""
    return features.reshape(-1, features.shape[-1])[::stride].float()


def window_chunks(n: int) -> list[slice]:
    return [slice(start, start + CHUNK_WINDOWS) for start in range(0, n, CHUNK_WINDOWS)]


def search_attention(
    block: Block,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    group_size: int,
    stride: int,
    report: dict[str, Any],
    samples: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """The scales of q/k/v on the stream x entering the block, and the attention's output.

    Adds the group's entry to `report`, and to `samples` every stride-th input of q/k/v ("attn")
    and of o_proj ("o").
    """
    attn = block.attn
    h = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    o_in = torch.empty_like(h)
    target = torch.empty_like(x)
    seen: list[Tensor] = []
    hook = attn.o_proj.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    for rows in window_chunks(x.shape[0]):
        h[rows] = block.attn_norm(x[rows])
        target[rows] = attn(h[rows].float(), cos, sin)
        o_in[rows] = seen.pop()
    hook.remove()
    x_mean, x_max = channel_stats(h)
    s_attn, losses = search_scales(
        x_mean,
        block.attn_norm.weight,
        (attn.q_proj, attn.k_proj, attn.v_proj),
        group_size,
        lambda: output_mse(lambda a: attn(a, cos, sin), h, target),
    )
    report["attn"] = group_report(losses, s_attn, x_max, block.attn_norm.weight)
    samples |= {"attn": sample_tokens(h, stride), "o": sample_tokens(o_in, stride)}
    return s_attn, target


def search_mlp(
    block: Block,
    x: Tensor,
    group_size: int,
    stride: int,
    report: dict[str, Any],
    samples: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """The scales of gate/up and of down_proj on the stream x after the block's attention, and
    the MLP's output.

    Adds the two groups' entries to `report`, and to `samples` every stride-th input of gate/up
    ("mlp") and of down_proj ("down").
    """
    mlp = block.mlp
    n, length, _ = x.shape
    h = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    act = torch.empty((n, length, mlp.down_proj.in_features), dtype=h.dtype, device=x.device)
    target = torch.empty_like(x)
    for rows in window_chunks(n):
        h[rows] = block.mlp_norm(x[rows])
        act[rows] = F.silu(mlp.gate_proj(h[rows].float())) * mlp.up_proj(h[rows].float())
        target[rows] = mlp.down_proj(act[rows].float())
    x_mean, x_max = channel_stats(h)
    s_mlp, losses = search_scales(
        x_mean,
        block.mlp_norm.weight,
        (mlp.gate_proj, mlp.up_proj),
        group_size,
        lambda: output_mse(mlp, h, target),
    )
    report["mlp"] = group_report(losses, s_mlp, x_max, block.mlp_norm.weight)
    x_mean, x_max = channel_stats(act)
    s_down, losses = search_scales(
        x_mean, None, (mlp.down_proj,), group_size, lambda: output_mse(mlp.down_proj, act, target)
    )
    report["down"] = group_report(losses, s_down, x_max)
    samples |= {"mlp": sample_tokens(h, stride), "down": sample_tokens(act, stride)}
    return s_mlp, s_down, target


def search_layer(
    block: Block,
    i: int,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    stride: int,
    clip: bool,
    result: AwqResult,
) -> None:
    """Searches layer i's scales, and its clip scales when `clip`, on the residual stream x
    [n, len, hidden] entering it; appends them to `result` and advances x, in place, to the
    stream leaving the layer.

    The inputs of the layer's matrices are cached in FP16, the precision the engine holds
    activations in. Every loss is taken on those cached inputs with the layer's other weights at
    full precision, and so is the stream the next layer is searched on.
    """
    group_size = result.group_size
    report: dict[str, Any] = {}
    samples: dict[str, Tensor] = {}
    s_attn, attn_out = search_attention(block, x, cos, sin, group_size, stride, report, samples)
    x += attn_out
    del attn_out
    s_mlp, s_down, mlp_out = search_mlp(block, x, group_size, stride, report, samples)
    x += mlp_out
    del mlp_out

    if clip:
        # Each clipped matrix with its scales folded in, against its inputs divided by them.
        tensors = block_tensors(block, i)
        tensors |= scale_layer(tensors, i, s_attn, s_mlp, s_down)
        inputs = {
            "self_attn.v_proj": samples["attn"] / s_attn,
            "self_attn.o_proj": samples["o"],
            "mlp.gate_proj": samples["mlp"] / s_mlp,
            "mlp.up_proj": samples["mlp"] / s_mlp,
            "mlp.down_proj": samples["down"] / s_down,
        }
        report["clip"] = {}
        for proj, tokens in inputs.items():
            name = f"model.layers.{i}.{proj}.weight"
            w = tensors[name]
            scale, minmax_error, error = act_clip_scales(
                w.reshape(w.shape[0], -1, group_size),
                tokens.reshape(tokens.shape[0], -1, group_size),
            )
            result.clip_scales[name] = scale.cpu()
            # Summed over the matrix's groups, per sampled token.
            report["clip"][proj] = {
                "loss_minmax": minmax_error.sum().item() / tokens.shape[0],
                "loss": error.sum().item() / tokens.shape[0],
                "clipped_groups": (error < minmax_error).double().mean().item(),
            }

    result.s_attn.append(s_attn.cpu())
    result.s_mlp.append(s_mlp.cpu())
    result.s_down.append(s_down.cpu())
    result.report.append(report)


@torch.no_grad()
def collect_and_search(
    model: Llama, windows: Tensor, group_size: int, device: str = "cpu", clip: bool = True
) -> AwqResult:
    """Searches the scales of every layer of `model` on calibration tokens `windows` [n, len].

    `model` is the reference model on the CPU with the checkpoint's weights at full precision;
    its blocks visit `device` one at a time and return unchanged. Each layer is searched on the
    activations the full-precision layers before it produce. Without `clip` the clip search is
    skipped and clip_scales stays empty.
    """
    cfg = model.cfg
    n, length = windows.shape
    positions = torch.arange(length, device=device)
    cos, sin = rope_tables(cfg.head_dim, cfg.rope_theta, positions, cfg.rope_scaling)
    stride = max(1, n * length // CLIP_TOKENS)
    result = AwqResult(group_size)
    x = model.embed(windows.to(model.embed.weight.device, torch.int64)).to(device)
    for i, block in enumerate(model.layers):
        assert isinstance(block, Block)
        block.to(device)
        search_layer(block, i, x, cos, sin, stride, clip, result)
        block.to("cpu")
    return result


def save_scales(path: Path, result: AwqResult, windows: Tensor) -> None:
    """Writes the scales of `result` (float32, "layers.<i>.s_attn" and so on) and the
    calibration tokens (int32 [n, len]) they were searched on."""
    tensors = {CALIB_IDS: windows.to(torch.int32)}
    for i, scales in enumerate(zip(result.s_attn, result.s_mlp, result.s_down, strict=True)):
        for group, s in zip(("s_attn", "s_mlp", "s_down"), scales, strict=True):
            tensors[f"layers.{i}.{group}"] = s.float()
    save_file(tensors, str(path))


def norm_site_scales(model_dir: Path) -> dict[str, Tensor]:
    """Per norm capture site of an AWQ checkpoint, the scales its output was divided by; empty
    for a checkpoint whose header names no activation scales."""
    with safe_open(str(Path(model_dir) / WEIGHTS_FILE), framework="pt") as f:
        if not WeightFormat.from_metadata(f.metadata()).act_scales:
            return {}
    path = Path(model_dir) / SCALES_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found beside a checkpoint with activation scales")
    sites = {"s_attn": "attn_norm", "s_mlp": "mlp_norm"}
    scales = {}
    for name, s in load_file(str(path)).items():
        layer, _, group = name.rpartition(".")
        if group in sites:
            scales[f"{layer}.{sites[group]}"] = s
    return scales
