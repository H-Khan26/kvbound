"""Analytical performance model: a lower bound on prefill and decode latency.

Every operation needs some bytes moved through memory and some arithmetic done. Neither can go
faster than the hardware ceiling, so

    time(op) >= max(bytes / memory_bandwidth, ops / peak_arithmetic_rate)

and a forward pass is the sum over its operations. Whichever term is larger says whether the
operation is memory-bound or compute-bound; the ratio ops / bytes is its arithmetic intensity,
the x-axis of a roofline plot.

The model counts only what cannot be avoided: each weight is read once per step, each cached
key and value is read once per decode step, and the matrix arithmetic is done once. Kernel launch
overhead, elementwise ops (norms, RoPE, SiLU, residuals) and sampling are left out, so a real
engine is always slower. The gap to this bound is what Phase 2 onward reports.
"""

from dataclasses import dataclass
from typing import Any

from kvbound.perfmodel.hardware import Ceilings

ACTIVATION_BYTES = 2  # activations are FP16


@dataclass(frozen=True)
class Arch:
    """The dimensions that determine cost; see configs/models/*.toml [arch]."""

    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    intermediate_size: int

    @classmethod
    def from_table(cls, table: dict[str, Any]) -> "Arch":
        return cls(**{name: table[name] for name in cls.__dataclass_fields__})

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    def linears(self) -> list[tuple[str, int, int, int]]:
        """(name, out_features, in_features, count) of every weight matrix read per token."""
        h, kv, mlp = self.hidden_size, self.n_kv_heads * self.head_dim, self.intermediate_size
        n = self.n_layers
        return [
            ("q_proj", h, h, n),
            ("k_proj", kv, h, n),
            ("v_proj", kv, h, n),
            ("o_proj", h, h, n),
            ("gate_proj", mlp, h, n),
            ("up_proj", mlp, h, n),
            ("down_proj", h, mlp, n),
            ("lm_head", self.vocab_size, h, 1),
        ]

    def linear_params(self) -> int:
        return sum(rows * cols * count for _, rows, cols, count in self.linears())


@dataclass(frozen=True)
class Precision:
    """How many bytes each stored number takes and which arithmetic ceiling applies."""

    weight_bytes: float  # per weight element
    kv_bytes: float  # per cached key/value element
    compute: str  # key into Ceilings.peak_tops

    @classmethod
    def named(cls, weights: str, kv: str, head_dim: int = 64) -> "Precision":
        """weights is a variant name (fp16, int8_pc, int4_g, int4_g.mse); kv is fp16, int8 or fp8.

        Bytes include the FP16 scales: one per 128 weights for int4_g (one per row for int8_pc,
        which is negligible), one per cached vector of head_dim elements for an int8 cache.
        """
        # Quantization here is weight-only: INT8 and INT4 weights are widened inside the kernel
        # and multiplied with FP16 activations, or expanded to FP16 for a tensor-core product,
        # so the FP16 ceiling applies to all. The ceiling is FP16 accumulate, the faster of the
        # two FP16 modes, so the result stays a lower bound for kernels that accumulate in FP32.
        weight_bytes = {"fp16": 2.0, "int8_pc": 1.0, "int4_g": 0.5 + 2 / 128}
        kv_bytes = {"fp16": 2.0, "int8": 1.0 + 2 / head_dim, "fp8": 1.0}
        return cls(weight_bytes[weights.removesuffix(".mse")], kv_bytes[kv], "fp16_acc16")


def kv_bytes_per_token(arch: Arch, kv_bytes: float) -> float:
    """Bytes one token adds to the KV cache: a key and a value per layer per KV head."""
    return 2 * arch.n_layers * arch.n_kv_heads * arch.head_dim * kv_bytes


@dataclass(frozen=True)
class OpCost:
    name: str
    bytes: float
    ops: float
    seconds: float
    bound_by: str  # "memory" or "compute"

    @property
    def intensity(self) -> float:
        """Arithmetic intensity: operations per byte moved."""
        return self.ops / self.bytes


def _cost(name: str, n_bytes: float, n_ops: float, hw: Ceilings, compute: str) -> OpCost:
    t_memory = n_bytes / (hw.dram_gbs * 1e9)
    t_compute = n_ops / hw.peak_ops(compute)
    bound = "memory" if t_memory >= t_compute else "compute"
    return OpCost(name, n_bytes, n_ops, max(t_memory, t_compute), bound)


def step_costs(
    arch: Arch, hw: Ceilings, precision: Precision, batch: int, new_tokens: int, context: int
) -> list[OpCost]:
    """Cost of one forward pass that processes `new_tokens` positions for each of `batch`
    sequences which already have `context` tokens in the KV cache.

    Decode is new_tokens=1 with context = tokens so far; prefill is new_tokens = prompt length
    with context = 0.
    """
    positions = batch * new_tokens
    costs = []
    for name, rows, cols, count in arch.linears():
        # Only the last position of each sequence goes through the output projection.
        positions_here = batch if name == "lm_head" else positions
        weight = rows * cols * precision.weight_bytes
        activations = positions_here * (rows + cols) * ACTIVATION_BYTES
        ops = 2 * rows * cols * positions_here
        costs.append(
            _cost(name, count * (weight + activations), count * ops, hw, precision.compute)
        )

    # Attention: every new position attends to the cached tokens and to the new ones before it.
    attended = new_tokens * context + new_tokens * (new_tokens + 1) / 2  # pairs per sequence
    # Each cached or new key and value must be read at least once per pass; a tiled kernel can
    # reuse it from cache across the queries of one prefill.
    kv_read = (context + new_tokens) * 2 * arch.n_kv_heads * arch.head_dim * precision.kv_bytes
    # One multiply-add per element for Q.K^T and again for probs.V, for every query head.
    attention_ops = attended * 2 * 2 * arch.n_heads * arch.head_dim
    costs.append(
        _cost(
            "attention",
            arch.n_layers * batch * kv_read,
            arch.n_layers * batch * attention_ops,
            hw,
            "fp16_acc16",
        )
    )
    return costs


@dataclass(frozen=True)
class Prediction:
    prefill_s: float  # lower bound on time to first token
    decode_step_s: float  # lower bound on time per output token (whole batch)
    decode_tok_s: float  # upper bound on decode tokens/s across the batch
    weights_gb: float
    kv_gb: float  # KV cache for the whole batch at the final context
    prefill_costs: list[OpCost]
    decode_costs: list[OpCost]


def predict(
    arch: Arch, hw: Ceilings, precision: Precision, batch: int, prompt_len: int, context: int
) -> Prediction:
    """Bounds for prefilling `prompt_len` tokens and for one decode step at `context` tokens."""
    prefill = step_costs(arch, hw, precision, batch, prompt_len, 0)
    decode = step_costs(arch, hw, precision, batch, 1, context)
    decode_s = sum(c.seconds for c in decode)
    return Prediction(
        prefill_s=sum(c.seconds for c in prefill),
        decode_step_s=decode_s,
        decode_tok_s=batch / decode_s,
        weights_gb=arch.linear_params() * precision.weight_bytes / 1e9,
        kv_gb=batch * context * kv_bytes_per_token(arch, precision.kv_bytes) / 1e9,
        prefill_costs=prefill,
        decode_costs=decode,
    )
