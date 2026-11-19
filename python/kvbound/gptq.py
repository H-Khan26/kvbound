"""GPTQ rounding for int4_g weights (Frantar et al., 2022; reference code IST-DASLab/gptq).

Round-to-nearest picks each integer on its own, but what the model feels is a matrix's output.
For a matrix W [out, in] and the inputs x it multiplies, the output error of a rounded W_q is

  mean over tokens of |(W - W_q) x|^2  =  sum over rows r of  d_r H d_r^T,
  d_r = W[r] - W_q[r],  H = mean over tokens of x x^T  [in, in]

H is the Hessian of that squared error with respect to one row (up to a factor 2). It depends
only on the inputs, so it is the same for every row, and matrices fed the same activations
(q/k/v, gate/up) share it. Its off-diagonal entries say which input channels move together: the
error of one rounded weight can be partly cancelled by moving the weights of the same row that
are not rounded yet.

GPTQ rounds the columns of W one at a time, in order. After rounding column j it moves the
columns still free by the optimal-brain-surgeon update, the change that least raises the error:

  W[:, f] -= (W[:, j] - W_q[:, j]) * Hinv[j, f] / Hinv[j, j]

with f the free columns and Hinv the inverse of H restricted to column j and those. Removing
column j from that inverse at every step would cost a Gaussian elimination each, but for a
fixed order the rows the reduced inverses provide are the rows of U, the upper Cholesky factor
of H^-1 (H^-1 = U^T U): reduced row j is U[j, j] * U[j, j:]. One Cholesky up front replaces
every elimination, and the step becomes

  e = (W[:, j] - W_q[:, j]) / U[j, j]      W[:, j:] -= e U[j, j:]

e^2 summed over rows is the step's predicted rise in the output error above.

Lazy block updates: column j's update touches every later column, which would read and write
all of W per column. Columns go instead in blocks of BLOCK_COLUMNS: within a block each update
applies at once, the updates to columns past the block are kept as the block's errors E, and
they land once per block as one matrix product W[:, after] -= E U[block, after]. The arithmetic
is the same; the memory traffic is one pass over W per block instead of per column.

Damping: H is often near singular (channels that rarely fire, channels that move together), so
damp * mean(diag H) is added to its diagonal, and the Cholesky factorizations run in float64.
An input channel no calibration token reaches has diag H = 0; its diagonal is set to 1, which
rounds its weights to nearest without moving any other.

Column order: by default the columns go by decreasing diag H, the inputs with the most energy
first ("act order"). The first columns are rounded before any update reaches them, and the
rest absorb their errors. This matters on Llama 3.2 1B: one input channel of layer 1's
down_proj carries 99.9% of trace(H), all of it from the first token of each window (the
"massive activation" an attention sink is built on). In input order, the errors of earlier
columns flow into that channel's weights, and the solver trades the other 65,000 tokens for
those 128: their error falls ninefold while everyone else's grows 2.5 times. Put into the FP16
model alone, that matrix scores perplexity 178 where rounding it to nearest scores 11.8. Rounded
first, the channel keeps its rounded weights, and on held-out text both the sink tokens' error
and the other tokens' fall below rounding to nearest's.

The grid is int4_g's: integers in [-7, 7], one F16 scale per group of group_size inputs. With
act order a group's columns are spread over the order, so its scale is fixed up front: the
min-max scale of its weights, or a given one (AWQ's clip scales). In input order it can instead
be the min-max scale of the weights when the solver reaches the group, after earlier columns'
updates have moved them. Either way it is rounded to F16 before any integer is chosen against
it, so the stored pair is what the engine multiplies by.

Calibration is sequential: layer by layer, the inputs come from the earlier layers already
quantized, and within a layer each stage (q/k/v, o_proj, gate/up, down_proj) sees the stages
before it quantized, so every matrix also compensates the error upstream of it. The embedding is
a lookup with no input to take a Hessian of, and rounds to nearest; an untied lm_head gets GPTQ
with the Hessian of the final norm's output; a tied one is the embedding and rounds to nearest.

The result is an ordinary int4_g checkpoint; see convert() in kvbound.quantize.

Usage: python -m kvbound.quantize tinyllama-1.1b --scheme int4_g --gptq [--awq]
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from kvbound.awq import CHUNK_WINDOWS, block_tensors, scale_layer, window_chunks
from kvbound.quantize import (
    dequantize_int4_g,
    f16_scales,
    pack_int4,
    quantize_tensor,
    round_to_grid,
)
from kvbound.ref.model import Block, Llama, rope_tables
from kvbound.weight_format import INT4_G, QMAX, WeightFormat

# Damping as a fraction of the mean diagonal of H.
DEFAULT_DAMP = 0.01
# Columns per lazy block update (rounded up to whole groups).
BLOCK_COLUMNS = 128
LM_HEAD = "lm_head.weight"
# Rows of an untied lm_head on the device at once. Rows share H and the column order and are
# solved independently, so the chunks give the integers the whole matrix would.
HEAD_CHUNK_ROWS = 32768


@dataclass(frozen=True)
class GptqConfig:
    """How the solver runs.

    damp is added to H's diagonal as a fraction of its mean. act_order rounds the columns by
    decreasing diag H and fixes the group scales up front. Without it the columns go in input
    order, and static_groups fixes each group's scale from the weights before any update
    instead of when the solver reaches the group.
    """

    damp: float = DEFAULT_DAMP
    act_order: bool = True
    static_groups: bool = False


@dataclass
class GptqResult:
    """What quantize_model chooses: per Hugging Face name, the packed int4 weights [out, in / 2]
    and their F16 scales [out, in / group_size] on the CPU, and per layer (then the lm_head)
    each stage's report."""

    group_size: int
    quantized: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    report: list[dict[str, Any]] = field(default_factory=list)


class HessianSum:
    """Accumulates H = mean over tokens of x x^T for inputs of width n, in float64."""

    def __init__(self, n: int, device: torch.device | str) -> None:
        self.total = torch.zeros(n, n, dtype=torch.float64, device=device)
        self.count = 0

    def add(self, x: Tensor) -> None:
        # Each chunk's product runs in float32; the running sum is float64.
        x = x.reshape(-1, self.total.shape[0]).float()
        self.total += (x.T @ x).double()
        self.count += x.shape[0]

    def mean(self) -> Tensor:
        return self.total / self.count


def collect_hessians(
    run: Callable[[slice], object], n_windows: int, inputs: dict[str, nn.Linear]
) -> dict[str, Tensor]:
    """Per key, H of the inputs that reach inputs[key] while run(rows) runs over the windows.

    run takes a slice of windows, CHUNK_WINDOWS at a time, and computes whatever reaches the
    matrices; its return value is dropped.
    """
    sums: dict[str, HessianSum] = {}
    hooks = []
    for key, linear in inputs.items():
        sums[key] = HessianSum(linear.in_features, linear.weight.device)
        hooks.append(linear.register_forward_pre_hook(lambda _m, args, s=sums[key]: s.add(args[0])))
    try:
        for rows in window_chunks(n_windows):
            run(rows)
    finally:
        for hook in hooks:
            hook.remove()
    return {key: s.mean() for key, s in sums.items()}


def minmax_scales(w: Tensor, group_size: int) -> Tensor:
    """F16 min-max scales [out, in / group_size] of w [out, in]."""
    groups = w.float().reshape(w.shape[0], -1, group_size)
    return f16_scales(groups.abs().amax(dim=-1), QMAX[INT4_G])


def inverse_cholesky(h: Tensor, damp: float, overwrite: bool = False) -> Tensor:
    """U, the upper Cholesky factor of (H + damp * mean(diag H) * I)^-1, in float64.

    Channels with diag H = 0 get a diagonal of 1 first. H itself is not changed unless
    `overwrite`, which saves a copy of a float64 H.
    """
    h = h.double() if overwrite else h.double().clone()
    diag = torch.diagonal(h)
    diag[diag == 0] = 1.0
    diag += damp * diag.mean()
    try:
        lower = torch.linalg.cholesky(h)
        del h
        hinv = torch.cholesky_inverse(lower)
        del lower
        return torch.linalg.cholesky(hinv, upper=True)
    except torch.linalg.LinAlgError as e:
        raise ValueError(f"H is not positive definite with damping {damp}; raise it") from e


def solve(
    w: Tensor,
    h: Tensor,
    group_size: int,
    scale: Tensor | None = None,
    config: GptqConfig = GptqConfig(),  # noqa: B008
) -> tuple[Tensor, Tensor, float]:
    """GPTQ integers for w [out, in] on the int4_g grid, given the Hessian h [in, in] of its
    inputs.

    scale, F16 [out, in / group_size], fixes the group scales; without it they are min-max
    scales, of the weights as they stand when the solver reaches each group, or before any
    update with config.static_groups. Returns q int8 [out, in] in [-7, 7], the F16 scales and the
    predicted output error (the sum of e^2), all on w's device.
    """
    qmax = QMAX[INT4_G]
    out_features, in_features = w.shape
    WeightFormat.make(INT4_G, group_size).check_shape(tuple(w.shape))
    w = w.float().clone()
    if scale is None and (config.static_groups or config.act_order):
        scale = minmax_scales(w, group_size)
    if scale is not None and (scale.dtype != torch.float16 or scale.shape[0] != out_features):
        raise ValueError(f"expected F16 scales [{out_features}, groups], got {scale.dtype}")

    perm = None
    if config.act_order:
        # The columns, H and the scales are permuted alike; the permuted H is a copy.
        perm = torch.argsort(torch.diagonal(h), descending=True)
        w = w[:, perm]
        u = inverse_cholesky(h[perm][:, perm], config.damp, overwrite=True).float()
    else:
        u = inverse_cholesky(h, config.damp).float()
    # Per column, its scale as float32 when the scales are fixed.
    col_scale = None
    if scale is not None:
        col_scale = scale.float().repeat_interleave(group_size, dim=1)
        if perm is not None:
            col_scale = col_scale[:, perm]
    scales = (
        scale
        if scale is not None
        else torch.empty(
            out_features, in_features // group_size, dtype=torch.float16, device=w.device
        )
    )

    q = torch.empty(out_features, in_features, dtype=torch.int8, device=w.device)
    loss = torch.zeros((), dtype=torch.float64, device=w.device)
    block = max(1, BLOCK_COLUMNS // group_size) * group_size
    group_scale = torch.empty(0)
    for start in range(0, in_features, block):
        end = min(start + block, in_features)
        w1 = w[:, start:end]  # a view: the updates below land in w
        u1 = u[start:end, start:end]
        err = torch.empty_like(w1)
        for j in range(end - start):
            col = start + j
            if col_scale is None:
                if col % group_size == 0:
                    # The group's weights as earlier columns' updates have left them.
                    g = col // group_size
                    scales[:, g] = minmax_scales(w1[:, j : j + group_size], group_size)[:, 0]
                    group_scale = scales[:, g].float()
                s = group_scale
            else:
                s = col_scale[:, col]
            x = w1[:, j]
            qj = torch.clamp(torch.round(x / s), -qmax, qmax)
            q[:, col] = qj.to(torch.int8)
            e = (x - qj * s) / u1[j, j]
            loss += e.double().pow(2).sum()
            w1[:, j:] -= e[:, None] * u1[j, j:][None, :]
            err[:, j] = e
        w[:, end:] -= err @ u[start:end, end:]
    if perm is not None:
        q = q[:, torch.argsort(perm)]
    return q, scales, loss.item()


def output_error(d: Tensor, h: Tensor) -> float:
    """Mean over tokens of |d x|^2 for a weight error d [out, in]: sum of (d H d^T)'s diagonal."""
    d = d.float()
    return ((d @ h.float()) * d).double().sum().item()


def quantize_stage(
    linears: dict[str, nn.Linear],
    h: Tensor,
    group_size: int,
    config: GptqConfig,
    given: dict[str, Tensor],
    result: GptqResult,
) -> dict[str, Any]:
    """Runs the solver once on the matrices `linears` (Hugging Face name -> module) that share
    the Hessian h, stacked row-wise, and sets their weights to the dequantized result.

    Matrices named in `given` keep those F16 scales; when any is, the others in the stack take
    min-max scales of their weights before any update, so the whole stack has fixed scales.
    Returns the stage's report: whether the group scales were fixed before any update, and per
    matrix the output error after GPTQ and after rounding to nearest on the scales GPTQ starts
    from (fixed ones, or min-max of the original weights).
    """
    names = list(linears)
    w = torch.cat([linears[name].weight.float() for name in names])
    bounds, row = [], 0
    for name in names:
        bounds.append(slice(row, row + linears[name].out_features))
        row += linears[name].out_features
    fixed = None
    if any(name in given for name in names):
        fixed = torch.cat(
            [
                given[name].to(w.device) if name in given else minmax_scales(w[rows], group_size)
                for name, rows in zip(names, bounds, strict=True)
            ]
        )
    q, scale, loss = solve(w, h, group_size, fixed, config)
    deq = q.float() * scale.float().repeat_interleave(group_size, dim=1)
    rtn_scale = fixed if fixed is not None else minmax_scales(w, group_size)
    rtn = round_to_grid(w.reshape(w.shape[0], -1, group_size), rtn_scale, QMAX[INT4_G])
    rtn_deq = (rtn.float() * rtn_scale.float().unsqueeze(-1)).reshape(w.shape)

    report: dict[str, Any] = {
        "dead_inputs": int((torch.diagonal(h) == 0).sum().item()),
        "fixed_scales": fixed is not None or config.act_order or config.static_groups,
        "loss_predicted": loss,
        "matrices": {},
    }
    for name, rows in zip(names, bounds, strict=True):
        report["matrices"][name] = {
            "error": output_error(w[rows] - deq[rows], h),
            "error_rtn": output_error(w[rows] - rtn_deq[rows], h),
            "signal": output_error(w[rows], h),
        }
        linears[name].weight.copy_(deq[rows])
        result.quantized[name] = (pack_int4(q[rows]).cpu(), scale[rows].cpu())
    return report


def layer_linears(block: Block, i: int) -> dict[str, nn.Linear]:
    """The matrices of block i under their Hugging Face names."""
    layer = f"model.layers.{i}"
    return {
        f"{layer}.self_attn.q_proj.weight": block.attn.q_proj,
        f"{layer}.self_attn.k_proj.weight": block.attn.k_proj,
        f"{layer}.self_attn.v_proj.weight": block.attn.v_proj,
        f"{layer}.self_attn.o_proj.weight": block.attn.o_proj,
        f"{layer}.mlp.gate_proj.weight": block.mlp.gate_proj,
        f"{layer}.mlp.up_proj.weight": block.mlp.up_proj,
        f"{layer}.mlp.down_proj.weight": block.mlp.down_proj,
    }


def quantize_layer(
    block: Block,
    i: int,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    config: GptqConfig,
    given: dict[str, Tensor],
    result: GptqResult,
) -> None:
    """Quantizes block i stage by stage on the residual stream x [n, len, hidden] entering it,
    and advances x, in place, to the stream leaving the quantized block.

    Each stage's Hessian is taken with the stages before it already quantized: q/k/v on the
    attention norm's output, o_proj on the attention of the quantized q/k/v, gate/up on the norm
    of the stream after the quantized attention, down_proj on the quantized gate/up's product.
    """
    linears = layer_linears(block, i)
    names = list(linears)
    n = x.shape[0]
    group_size = result.group_size
    report: dict[str, Any] = {}

    def attention(rows: slice) -> Tensor:
        return block.attn(block.attn_norm(x[rows]), cos, sin)

    def mlp(rows: slice) -> Tensor:
        return block.mlp(block.mlp_norm(x[rows]))

    stages = (
        ("attn", attention, names[0:3]),
        ("o", attention, names[3:4]),
        ("mlp", mlp, names[4:6]),
        ("down", mlp, names[6:7]),
    )
    for stage, run, stage_names in stages:
        first = linears[stage_names[0]]
        h = collect_hessians(run, n, {stage: first})[stage]
        stage_linears = {name: linears[name] for name in stage_names}
        report[stage] = quantize_stage(stage_linears, h, group_size, config, given, result)
        del h
        if x.is_cuda:
            # Hessians differ in size from stage to stage; cached blocks would only pile up.
            torch.cuda.empty_cache()
        if stage in ("o", "down"):
            # The sublayer is quantized: the stream moves past it.
            for rows in window_chunks(n):
                x[rows] += run(rows)
    result.report.append(report)


def fold_awq(model: Llama, s_attn: list[Tensor], s_mlp: list[Tensor], s_down: list[Tensor]) -> None:
    """Folds AWQ scales (kvbound.awq) into the model's weights in place, as convert stores them."""
    for i, block in enumerate(model.layers):
        assert isinstance(block, Block)
        params = block_tensors(block, i)
        scaled = scale_layer(
            {name: p.detach() for name, p in params.items()}, i, s_attn[i], s_mlp[i], s_down[i]
        )
        with torch.no_grad():
            for name, tensor in scaled.items():
                params[name].copy_(tensor)


@torch.no_grad()
def quantize_model(
    model: Llama,
    windows: Tensor,
    group_size: int,
    device: str = "cpu",
    config: GptqConfig = GptqConfig(),  # noqa: B008
    given: dict[str, Tensor] | None = None,
) -> GptqResult:
    """GPTQ integers for every matrix of `model` that has an input to take a Hessian of.

    `model` is the reference model on the CPU with the weights as they are to be quantized (any
    AWQ scales folded in, see fold_awq); its blocks visit `device` one at a time and come back
    with their matrices set to the dequantized result. windows [n, len] are the calibration
    tokens; `given` maps Hugging Face names to fixed F16 scales (AWQ's clip scales). The
    embedding is set to its round-to-nearest values, the ones convert stores, and is not in the
    result; nor is a tied lm_head.
    """
    given = given or {}
    cfg = model.cfg
    result = GptqResult(group_size)
    fmt = WeightFormat.make(INT4_G, group_size)
    q, scale, _ = quantize_tensor(model.embed.weight, fmt, device)
    model.embed.weight.copy_(dequantize_int4_g(q, scale, group_size))

    n, length = windows.shape
    positions = torch.arange(length, device=device)
    cos, sin = rope_tables(cfg.head_dim, cfg.rope_theta, positions, cfg.rope_scaling)
    x = model.embed(windows.to(torch.int64)).to(device)
    for i, block in enumerate(model.layers):
        assert isinstance(block, Block)
        block.to(device)
        quantize_layer(block, i, x, cos, sin, config, given, result)
        block.to("cpu")

    if not cfg.tie_word_embeddings:
        model.final_norm.to(device)
        head = HessianSum(cfg.hidden_size, device)
        for start in range(0, n, CHUNK_WINDOWS):
            head.add(model.final_norm(x[start : start + CHUNK_WINDOWS]))
        del x
        h = head.mean()
        del head
        model.final_norm.to("cpu")
        result.report.append(
            {"lm_head": quantize_head(model.lm_head, h, config, given.get(LM_HEAD), result)}
        )
    return result


def quantize_head(
    lm_head: nn.Linear, h: Tensor, config: GptqConfig, given: Tensor | None, result: GptqResult
) -> dict[str, Any]:
    """quantize_stage for an untied lm_head on the CPU, HEAD_CHUNK_ROWS rows at a time on h's
    device; the chunks' reports are summed into one."""
    rows = lm_head.out_features
    report: dict[str, Any] = {}
    packed, scales = [], []
    for start in range(0, rows, HEAD_CHUNK_ROWS):
        end = min(start + HEAD_CHUNK_ROWS, rows)
        part = nn.Linear(lm_head.in_features, end - start, bias=False, device=h.device)
        part.weight.copy_(lm_head.weight[start:end])
        part_given = {LM_HEAD: given[start:end]} if given is not None else {}
        part_result = GptqResult(result.group_size)
        entry = quantize_stage(
            {LM_HEAD: part}, h, result.group_size, config, part_given, part_result
        )
        lm_head.weight[start:end] = part.weight.cpu()
        q, scale = part_result.quantized[LM_HEAD]
        packed.append(q)
        scales.append(scale)
        del part, part_result
        if not report:
            report = entry
            continue
        report["loss_predicted"] += entry["loss_predicted"]
        for key, value in entry["matrices"][LM_HEAD].items():
            report["matrices"][LM_HEAD][key] += value
    result.quantized[LM_HEAD] = (torch.cat(packed), torch.cat(scales))
    return report


def stage_summary(report: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Per stage, the summed output error of GPTQ over that of rounding to nearest, and the
    summed GPTQ error over the summed signal, across the layers of a report."""
    totals: dict[str, list[float]] = {}
    for layer in report:
        for stage, entry in layer.items():
            t = totals.setdefault(stage, [0.0, 0.0, 0.0])
            for m in entry["matrices"].values():
                t[0] += m["error"]
                t[1] += m["error_rtn"]
                t[2] += m["signal"]
    return {
        stage: (e / r if r else 0.0, e / s if s else 0.0) for stage, (e, r, s) in totals.items()
    }
