"""Each CUDA kernel against the same operation in torch, over shape sweeps.

Inputs are FP16. The torch side computes in FP32 from those FP16 values, so the only expected
difference is the kernel's final rounding to FP16 (about 3e-4 relative RMS). Kernels that read
quantized data are compared with torch computing on the same data dequantized, so the
quantization error itself is not part of the difference.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from kvbound import quantize, tolerance
from kvbound.ref.model import apply_rope, rope_tables

pytestmark = pytest.mark.gpu

F16_ROUNDING = 1e-3


def rand_f16(gen: torch.Generator, *shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape, generator=gen) * scale).to(torch.float16)


def bits(x: torch.Tensor) -> np.ndarray:
    """FP16 tensor to the uint16 array the bindings take."""
    return np.ascontiguousarray(x.numpy()).view(np.uint16)


def from_bits(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.view(np.float16).astype(np.float32))


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


@pytest.mark.parametrize("dim", [16, 64, 100, 2048, 5000])
@pytest.mark.parametrize("rows", [1, 3])
def test_rms_norm(kvb, gen, rows, dim):
    x, w = rand_f16(gen, rows, dim, scale=3.0), rand_f16(gen, dim)
    eps = 1e-5
    xf = x.float()
    expected = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    actual = from_bits(kvb.rms_norm(bits(x), bits(w), eps))
    assert tolerance.rel_rms(actual, expected) < F16_ROUNDING


def test_embedding_copies_rows(kvb, gen):
    table = rand_f16(gen, 50, 24)
    ids = torch.tensor([49, 0, 7, 7, 31], dtype=torch.int32)
    actual = kvb.embedding(bits(table), ids.numpy())
    assert np.array_equal(actual, bits(table[ids.long()]))


@pytest.mark.parametrize("head_dim", [16, 64])
@pytest.mark.parametrize("n_heads", [1, 4])
def test_rope(kvb, gen, n_heads, head_dim):
    n_tokens, max_pos = 7, 300
    x = rand_f16(gen, n_tokens, n_heads, head_dim)
    positions = torch.randint(max_pos, (n_tokens,), generator=gen, dtype=torch.int32)
    cos, sin = rope_tables(head_dim, 10000.0, torch.arange(max_pos))
    # The reference takes [B, heads, T, head_dim] with tables indexed by T.
    expected = apply_rope(x.float().transpose(0, 1)[None], cos[positions], sin[positions])
    expected = expected[0].transpose(0, 1)
    table = torch.stack((cos, sin), dim=-1).contiguous().numpy()
    actual = from_bits(kvb.rope(bits(x), positions.numpy(), table))
    assert tolerance.rel_rms(actual, expected) < F16_ROUNDING


@pytest.mark.parametrize("n", [1, 255, 4097])
def test_silu_mul_and_add(kvb, gen, n):
    a, b = rand_f16(gen, n, scale=4.0), rand_f16(gen, n)
    silu = from_bits(kvb.silu_mul(bits(a), bits(b)))
    assert tolerance.rel_rms(silu, F.silu(a.float()) * b.float()) < F16_ROUNDING
    total = from_bits(kvb.add(bits(a), bits(b)))
    assert tolerance.rel_rms(total, a.float() + b.float()) < F16_ROUNDING


def test_gather_rows(kvb, gen):
    x = rand_f16(gen, 9, 33)
    rows = torch.tensor([8, 0, 3], dtype=torch.int32)
    assert np.array_equal(kvb.gather_rows(bits(x), rows.numpy()), bits(x[rows.long()]))


@pytest.mark.parametrize("n_cols", [5, 256, 1000, 32000])
def test_argmax(kvb, gen, n_cols):
    logits = torch.randn(4, n_cols, generator=gen)
    logits[1] = float("-inf")  # every column ties: the lowest index wins
    logits[2, [n_cols - 1, n_cols // 2]] = 100.0  # a two-way tie for the maximum
    expected = [int(logits[0].argmax()), 0, n_cols // 2, int(logits[3].argmax())]
    assert kvb.argmax(logits.numpy()).tolist() == expected


@pytest.mark.parametrize(("n_out", "n_in"), [(8, 8), (256, 2048), (5632, 2048), (100, 37)])
@pytest.mark.parametrize("rows", [1, 5])
def test_linear(kvb, gen, rows, n_out, n_in):
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    x = rand_f16(gen, rows, n_in)
    expected = x.float() @ w.float().T

    f16_out = from_bits(kvb.linear(bits(w), bits(x)))
    assert tolerance.rel_rms(f16_out, expected) < F16_ROUNDING
    f32_out = torch.from_numpy(kvb.linear(bits(w), bits(x), f32_out=True))
    assert tolerance.rel_rms(f32_out, expected) < 1e-5
    # FP16 running sums lose a few bits more than the final rounding does.
    f16_sums = from_bits(kvb.linear(bits(w), bits(x), accumulate="fp16"))
    assert tolerance.rel_rms(f16_sums, expected) < 2e-2


def quantized(w: torch.Tensor, scheme: str, group: int):
    """w quantized by kvbound.quantize: (stored bytes as uint8, scale bits, dequantized)."""
    if scheme == "int8_pc":
        q, scale = quantize.quantize_int8_pc(w)
        deq = quantize.dequantize_int8_pc(q, scale)
    else:
        q, scale = quantize.quantize_int4_g(w, group)
        deq = quantize.dequantize_int4_g(q, scale, group)
    return np.ascontiguousarray(q.numpy()).view(np.uint8), bits(scale), deq


@pytest.mark.parametrize(
    ("n_out", "n_in", "group"), [(8, 64, 32), (256, 2048, 128), (2048, 5632, 128), (100, 96, 32)]
)
@pytest.mark.parametrize("rows", [1, 2, 5, 16, 21])
@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_linear_quantized(kvb, gen, scheme, rows, n_out, n_in, group):
    """One to 21 rows: every row-count variant of the kernel, and a second pass past 16."""
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    x = rand_f16(gen, rows, n_in)
    data, scale, deq = quantized(w, scheme, group)
    expected = x.float() @ deq.T

    f32_out = kvb.linear_quantized(scheme, data, scale, bits(x), group_size=group, f32_out=True)
    assert tolerance.rel_rms(torch.from_numpy(f32_out), expected) < 1e-5
    f16_out = kvb.linear_quantized(scheme, data, scale, bits(x), group_size=group)
    assert tolerance.rel_rms(from_bits(f16_out), expected) < F16_ROUNDING
    # The expanded matrix is what the tensor-core path multiplies by.
    expanded = from_bits(kvb.dequantize(scheme, data, scale, group_size=group))
    assert tolerance.rel_rms(expanded, deq) < F16_ROUNDING
    # A slice of output channels equals the same rows of the whole.
    part = kvb.dequantize(scheme, data, scale, group_size=group, first_channel=3, n_channels=4)
    assert np.array_equal(part, bits(expanded.to(torch.float16))[3:7])


@pytest.mark.parametrize(
    ("n_out", "n_in", "group"),
    [
        (8, 64, 32),
        (200, 96, 32),
        (300, 2048, 128),
        (2056, 512, 64),
        (130, 5632, 128),
        (5000, 128, 32),
    ],
)
@pytest.mark.parametrize("rows", [1, 17, 40, 64, 80, 300])
@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_gemm_quantized(kvb, gen, scheme, rows, n_out, n_in, group):
    """Every tile shape the kernel picks, with partial tiles of rows, channels and inputs."""
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    x = rand_f16(gen, rows, n_in)
    data, scale, deq = quantized(w, scheme, group)
    # The kernel multiplies by the FP16 weights the expanded matrix holds.
    expected = x.float() @ deq.to(torch.float16).float().T

    f32_out = kvb.gemm_quantized(scheme, data, scale, bits(x), group_size=group, f32_out=True)
    assert tolerance.rel_rms(torch.from_numpy(f32_out), expected) < 1e-5
    f16_out = kvb.gemm_quantized(scheme, data, scale, bits(x), group_size=group)
    assert tolerance.rel_rms(from_bits(f16_out), expected) < F16_ROUNDING


@pytest.mark.parametrize("rows", [5, 300])
@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_gemm_quantized_weights_equal_dequantized(kvb, gen, scheme, rows):
    """Rows of the identity pick out single weights: each is the FP16 value dequantize writes."""
    n_out, n_in, group = 136, 512, 32
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    data, scale, _ = quantized(w, scheme, group)
    picks = torch.randperm(n_in, generator=gen)[:rows]
    x = torch.eye(n_in, dtype=torch.float16)[picks]
    expanded = kvb.dequantize(scheme, data, scale, group_size=group)
    actual = kvb.gemm_quantized(scheme, data, scale, bits(x), group_size=group)
    assert np.array_equal(actual, np.ascontiguousarray(expanded[:, picks.numpy()].T))


@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_embedding_quantized_reads_dequantized_rows(kvb, gen, scheme):
    table = rand_f16(gen, 50, 64)
    ids = torch.tensor([49, 0, 7, 7, 31], dtype=torch.int32)
    data, scale, deq = quantized(table, scheme, 32)
    actual = kvb.embedding_quantized(scheme, data, scale, ids.numpy(), group_size=32)
    assert tolerance.rel_rms(from_bits(actual), deq[ids.long()]) < F16_ROUNDING


def test_quantized_kernels_reject_rows_they_cannot_read(kvb, gen):
    x = rand_f16(gen, 1, 40)
    data = np.zeros((4, 40), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="divisible by 16"):
        kvb.linear_quantized("int8_pc", data, bits(rand_f16(gen, 4)), bits(x))
    with pytest.raises(RuntimeError, match="power of two"):
        kvb.linear_quantized(
            "int4_g", np.zeros((4, 48), dtype=np.uint8), bits(rand_f16(gen, 4, 2)),
            bits(rand_f16(gen, 1, 96)), group_size=48,
        )  # fmt: skip


def int8_readback(x: torch.Tensor) -> torch.Tensor:
    """What an INT8 cache returns for vectors x [..., dim]: integers times one FP16 scale each."""
    xf = x.float()
    scale = (xf.abs().amax(-1, keepdim=True) / 127).to(torch.float16).float()
    q = torch.where(scale > 0, torch.round(xf / scale), torch.zeros_like(xf)).clamp(-127, 127)
    return q * scale


def reference_attention(q, k, v, positions, seqs, n_rep, query_from=0):
    """Causal grouped-query attention per token, over its sequence's earlier tokens, in FP32.

    Rows before query_from are left zero.
    """
    q, k, v = q.float(), k.float(), v.float()
    out = torch.zeros_like(q)
    for t in range(query_from, q.shape[0]):
        visible = [
            u for u in range(q.shape[0]) if seqs[u] == seqs[t] and positions[u] <= positions[t]
        ]
        visible.sort(key=lambda u: int(positions[u]))
        for h in range(q.shape[1]):
            keys, values = k[visible, h // n_rep], v[visible, h // n_rep]
            scores = keys @ q[t, h] / math.sqrt(q.shape[2])
            out[t, h] = torch.softmax(scores, dim=0) @ values
    return out


@pytest.mark.parametrize("kv_dtype", ["fp16", "int8"])
@pytest.mark.parametrize("wide", [True, False])
@pytest.mark.parametrize("layout", ["head_major", "token_major", "dim_major"])
@pytest.mark.parametrize("block_size", [4, 64])
@pytest.mark.parametrize("head_dim", [16, 64])
def test_attention_over_interleaved_sequences(
    kvb, gen, head_dim, block_size, layout, wide, kv_dtype
):
    """Three sequences of different lengths, tokens shuffled, blocks assigned out of order."""
    n_heads, n_kv_heads, lengths = 4, 2, [1, 23, 50]
    blocks_per_seq = math.ceil(max(lengths) / block_size)
    n_blocks = len(lengths) * blocks_per_seq
    physical = torch.randperm(n_blocks, generator=gen).view(len(lengths), blocks_per_seq)

    tokens = [(s, p) for s, length in enumerate(lengths) for p in range(length)]
    order = torch.randperm(len(tokens), generator=gen).tolist()
    seqs = torch.tensor([tokens[i][0] for i in order], dtype=torch.int32)
    positions = torch.tensor([tokens[i][1] for i in order], dtype=torch.int32)
    slots = (
        physical[seqs.long(), (positions // block_size).long()] * block_size
        + positions % block_size
    ).to(torch.int32)

    n = len(tokens)
    q, k, v = (rand_f16(gen, n, h, head_dim) for h in (n_heads, n_kv_heads, n_kv_heads))
    actual = from_bits(
        kvb.kv_attention(
            bits(q), bits(k), bits(v), positions.numpy(), seqs.numpy(), slots.numpy(),
            physical.to(torch.int32).numpy(), block_size, n_blocks, layout,
            wide_key_loads=wide, kv_dtype=kv_dtype,
        )
    )  # fmt: skip
    if kv_dtype == "int8":
        k, v = int8_readback(k), int8_readback(v)
    expected = reference_attention(q, k, v, positions, seqs, n_heads // n_kv_heads)
    assert tolerance.rel_rms(actual, expected) < 2 * F16_ROUNDING


@pytest.mark.parametrize("kv_dtype", ["fp16", "int8"])
@pytest.mark.parametrize("wide", [True, False])
@pytest.mark.parametrize("layout", ["head_major", "token_major", "dim_major"])
def test_attention_long_context_single_query(kvb, gen, layout, wide, kv_dtype):
    """One query over 1,500 cached positions: the decode shape, with several keys per thread."""
    n_heads, n_kv_heads, head_dim, ctx = 4, 2, 64, 1500
    q, k, v = (rand_f16(gen, ctx, h, head_dim) for h in (n_heads, n_kv_heads, n_kv_heads))
    positions = torch.arange(ctx, dtype=torch.int32)
    seqs = torch.zeros(ctx, dtype=torch.int32)
    table = np.zeros((1, 1), dtype=np.int32)
    actual = from_bits(
        kvb.kv_attention(
            bits(q), bits(k), bits(v), positions.numpy(), seqs.numpy(), positions.numpy(),
            table, ctx, 1, layout, query_from=ctx - 1, wide_key_loads=wide, kv_dtype=kv_dtype,
        )
    )  # fmt: skip
    if kv_dtype == "int8":
        k, v = int8_readback(k), int8_readback(v)
    expected = reference_attention(q, k, v, positions, seqs, n_heads // n_kv_heads, ctx - 1)[-1:]
    assert tolerance.rel_rms(actual, expected) < 2 * F16_ROUNDING
