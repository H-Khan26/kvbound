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


# 100 and 9000 take the scalar kernel: not a multiple of 8, and longer than the wide one holds.
@pytest.mark.parametrize(("rows", "dim"), [(1, 64), (3, 2048), (2, 5000), (2, 100), (1, 9000)])
def test_add_rms_norm(kvb, gen, rows, dim):
    x, y, w = rand_f16(gen, rows, dim, scale=3.0), rand_f16(gen, rows, dim), rand_f16(gen, dim)
    eps = 1e-5
    total, normed = kvb.add_rms_norm(bits(x), bits(y), bits(w), eps)
    # The sum is rounded to FP16 before the norm reads it, as a separate add would leave it.
    expected_total = (x.float() + y.float()).to(torch.float16)
    assert np.array_equal(total, bits(expected_total))
    t = expected_total.float()
    expected = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    assert tolerance.rel_rms(from_bits(normed), expected) < F16_ROUNDING


@pytest.mark.parametrize(("rows", "dim"), [(1, 5), (4, 5632)])
def test_silu_mul_reads_the_stacked_halves(kvb, gen, rows, dim):
    gate_up = rand_f16(gen, rows, 2 * dim, scale=4.0)
    gate, up = gate_up[:, :dim].float(), gate_up[:, dim:].float()
    actual = from_bits(kvb.silu_mul_stacked(bits(gate_up)))
    assert tolerance.rel_rms(actual, F.silu(gate) * up) < F16_ROUNDING


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


# Shorter than one chunk of the kernel, exactly one, one and a bit, and Whisper's vocabulary.
@pytest.mark.parametrize("vocab", [7, 2048, 2049, 51865])
@pytest.mark.parametrize("k", [1, 6, 9])
def test_beam_candidates_equal_the_hosts(kvb, gen, vocab, k):
    """The device ranking against append_beam_candidates() on the host, bit for bit: ties
    (logits on a grid of 0.25), suppressed tokens, the begin list in the row at begin_pos, and a
    row with fewer finite logits than k, which is padded."""
    rows = 5
    logits = torch.round(torch.randn(rows, vocab, generator=gen) * 12.0) / 4.0
    logits[3, torch.randperm(vocab, generator=gen)[: vocab - 4]] = float("-inf")
    positions = np.array([2, 1, 1, 3, 1], dtype=np.int32)
    picks = torch.randperm(vocab, generator=gen).tolist()
    always, begin = picks[: vocab // 10 + 1], picks[vocab // 10 + 1 : vocab // 5 + 2]
    args = (logits.numpy(), positions, always, begin, 1, k)
    tokens, log_probs = kvb.beam_candidates(*args)
    host_tokens, host_log_probs = kvb.beam_candidates(*args, on_host=True)
    assert np.array_equal(tokens, host_tokens)
    assert np.array_equal(log_probs, host_log_probs)
    assert not np.isin(tokens, always).any() and not np.isin(tokens[[1, 2, 4]], begin).any()
    if k > 4:
        assert (tokens[3, 4:] == -1).all() and np.isneginf(log_probs[3, 4:]).all()


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


@pytest.mark.parametrize(("n_out", "n_in"), [(8, 8), (256, 2048), (2560, 2048), (100, 5632)])
@pytest.mark.parametrize("rows", [1, 2, 5, 16, 21])
def test_linear_f16(kvb, gen, rows, n_out, n_in):
    """kvbound's decode product; more than 16 rows take a second pass over the matrix."""
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    x = rand_f16(gen, rows, n_in)
    expected = x.float() @ w.float().T
    f16_out = from_bits(kvb.linear_f16(bits(w), bits(x)))
    assert tolerance.rel_rms(f16_out, expected) < F16_ROUNDING
    f32_out = torch.from_numpy(kvb.linear_f16(bits(w), bits(x), f32_out=True))
    assert tolerance.rel_rms(f32_out, expected) < 1e-5


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


def masked_reference(q, k, v, positions, seqs, n_rep):
    """reference_attention over every row at once: one mask over (query, key) pairs."""
    q = q.float()
    k, v = (x.float().repeat_interleave(n_rep, 1) for x in (k, v))
    visible = (seqs[:, None] == seqs[None, :]) & (positions[None, :] <= positions[:, None])
    scores = torch.einsum("thd,uhd->htu", q, k) / math.sqrt(q.shape[2])
    weights = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
    return torch.einsum("htu,uhd->thd", weights, v)


def check_prefill(kvb, gen, steps, *, n_heads, n_kv_heads, head_dim, block_size, kv_dtype,
                  layout="head_major", query_from=0):  # fmt: skip
    """Runs both attention kernels over the rows (seq, position) of `steps`, from query_from on,
    and compares each with the reference. Every sequence gets blocks in a random physical order.
    """
    seqs = torch.tensor([s for s, _ in steps], dtype=torch.int32)
    positions = torch.tensor([p for _, p in steps], dtype=torch.int32)
    n_seqs = int(seqs.max()) + 1
    blocks_per_seq = math.ceil((int(positions.max()) + 1) / block_size)
    n_blocks = n_seqs * blocks_per_seq
    physical = torch.randperm(n_blocks, generator=gen).view(n_seqs, blocks_per_seq)
    slots = (
        physical[seqs.long(), (positions // block_size).long()] * block_size
        + positions % block_size
    ).to(torch.int32)
    n = len(steps)
    q, k, v = (rand_f16(gen, n, h, head_dim) for h in (n_heads, n_kv_heads, n_kv_heads))

    def run(kernel):
        return from_bits(
            kvb.kv_attention(
                bits(q), bits(k), bits(v), positions.numpy(), seqs.numpy(), slots.numpy(),
                physical.to(torch.int32).numpy(), block_size, n_blocks, layout,
                query_from=query_from, kv_dtype=kv_dtype, kernel=kernel,
            )
        )  # fmt: skip

    prefill, decode = run("prefill"), run("decode")
    if kv_dtype == "int8":
        k, v = int8_readback(k), int8_readback(v)
    expected = masked_reference(q, k, v, positions, seqs, n_heads // n_kv_heads)[query_from:]
    assert tolerance.rel_rms(prefill, expected) < 2 * F16_ROUNDING
    assert tolerance.rel_rms(prefill, decode) < 2 * F16_ROUNDING


@pytest.mark.parametrize("kv_dtype", ["fp16", "int8"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(("n_heads", "n_kv_heads"), [(8, 2), (16, 2), (3, 3)])
@pytest.mark.parametrize("length", [1, 17, 64, 65, 300])
def test_prefill_attention_one_sequence(kvb, gen, length, n_heads, n_kv_heads, head_dim, kv_dtype):
    """A prompt in one cache block, under GQA ratios 4, 8 and 1 (an odd one)."""
    check_prefill(
        kvb, gen, [(0, p) for p in range(length)], n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, block_size=length, kv_dtype=kv_dtype,
    )  # fmt: skip


@pytest.mark.parametrize("kv_dtype", ["fp16", "int8"])
@pytest.mark.parametrize("layout", ["head_major", "token_major"])
def test_prefill_attention_paged_sequences(kvb, gen, layout, kv_dtype):
    """Four prompts of different lengths in one step, over scattered 16-position blocks."""
    steps = [(s, p) for s, length in enumerate([5, 64, 130, 300]) for p in range(length)]
    check_prefill(
        kvb, gen, steps, n_heads=8, n_kv_heads=2, head_dim=64, block_size=16, kv_dtype=kv_dtype,
        layout=layout,
    )  # fmt: skip


@pytest.mark.parametrize("kv_dtype", ["fp16", "int8"])
@pytest.mark.parametrize("block_size", [16, 256])
@pytest.mark.parametrize("cached", [37, 128])
def test_prefill_attention_chunked(kvb, gen, cached, block_size, kv_dtype):
    """A chunk of 150 rows whose first position is `cached`, over keys already in the cache."""
    check_prefill(
        kvb, gen, [(0, p) for p in range(cached + 150)], n_heads=8, n_kv_heads=2, head_dim=64,
        block_size=block_size, kv_dtype=kv_dtype, query_from=cached,
    )  # fmt: skip


@pytest.mark.parametrize("kv_dtype", ["fp16", "int8"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_prefill_attention_mixed_step(kvb, gen, head_dim, kv_dtype):
    """A prompt split by decode rows of two sequences with cached history: one-row tiles."""
    history = [(1, p) for p in range(50)] + [(2, p) for p in range(81)]
    step = [(0, p) for p in range(60)] + [(1, 50)] + [(0, p) for p in range(60, 100)] + [(2, 81)]
    check_prefill(
        kvb, gen, history + step, n_heads=32, n_kv_heads=4, head_dim=head_dim, block_size=16,
        kv_dtype=kv_dtype, query_from=len(history),
    )  # fmt: skip


def test_prefill_attention_needs_adjacent_key_dimensions(kvb, gen):
    q, k = rand_f16(gen, 4, 2, 64), rand_f16(gen, 4, 1, 64)
    positions = np.arange(4, dtype=np.int32)
    with pytest.raises(RuntimeError, match="adjacent key dimensions"):
        kvb.kv_attention(
            bits(q), bits(k), bits(k), positions, np.zeros(4, dtype=np.int32), positions,
            np.zeros((1, 1), dtype=np.int32), 4, 1, "dim_major", kernel="prefill",
        )  # fmt: skip


# 100 and 9000 take the scalar kernel, as in RMSNorm; 384 and 512 are Whisper's widths.
@pytest.mark.parametrize("dim", [16, 64, 100, 384, 512, 2048, 9000])
@pytest.mark.parametrize("rows", [1, 3])
def test_layer_norm(kvb, gen, rows, dim):
    # A mean far from zero: the variance has to be taken about it.
    x = (torch.randn(rows, dim, generator=gen) * 3.0 + 20.0).to(torch.float16)
    w, b = rand_f16(gen, dim), rand_f16(gen, dim)
    eps = 1e-5
    expected = F.layer_norm(x.float(), (dim,), w.float(), b.float(), eps)
    actual = from_bits(kvb.layer_norm(bits(x), bits(w), bits(b), eps))
    assert tolerance.rel_rms(actual, expected) < F16_ROUNDING


@pytest.mark.parametrize(("rows", "dim"), [(1, 64), (3, 512), (2, 5000), (2, 100), (1, 9000)])
def test_add_layer_norm(kvb, gen, rows, dim):
    x, y = rand_f16(gen, rows, dim, scale=3.0), rand_f16(gen, rows, dim)
    w, b = rand_f16(gen, dim), rand_f16(gen, dim)
    eps = 1e-5
    total, normed = kvb.add_layer_norm(bits(x), bits(y), bits(w), bits(b), eps)
    # The sum is rounded to FP16 before the norm reads it, as a separate add would leave it.
    expected_total = (x.float() + y.float()).to(torch.float16)
    assert np.array_equal(total, bits(expected_total))
    expected = F.layer_norm(expected_total.float(), (dim,), w.float(), b.float(), eps)
    assert tolerance.rel_rms(from_bits(normed), expected) < F16_ROUNDING


@pytest.mark.parametrize("n", [1, 255, 4097])
def test_gelu(kvb, gen, n):
    x = rand_f16(gen, n, scale=3.0)
    actual = from_bits(kvb.gelu(bits(x)))
    assert tolerance.rel_rms(actual, F.gelu(x.double())) < F16_ROUNDING


def test_gelu_is_the_erf_form_in_both_tails(kvb):
    """Inputs from -10 to 10: within one FP16 step of erf GELU computed in FP64, which holds
    the tanh approximation out and covers outputs down to the smallest FP16 magnitudes."""
    x = torch.linspace(-10, 10, 8001).to(torch.float16)
    expected = bits(F.gelu(x.double()).to(torch.float16)).astype(np.int32)
    actual = kvb.gelu(bits(x)).astype(np.int32)
    assert np.abs(actual - expected).max() <= 1


@pytest.mark.parametrize(("rows", "dim"), [(1, 8), (5, 384), (1500, 100)])
def test_add_positions(kvb, gen, rows, dim):
    x, table = rand_f16(gen, rows, dim), rand_f16(gen, 1500, dim)
    positions = torch.randint(1500, (rows,), generator=gen, dtype=torch.int32)
    positions[0] = 1499
    actual = kvb.add_positions(bits(x), bits(table), positions.numpy())
    expected = (x.float() + table[positions.long()].float()).to(torch.float16)
    assert np.array_equal(actual, bits(expected))


def run_linear(kvb, scheme: str, w: torch.Tensor, x: torch.Tensor, group: int = 0, **options):
    """x through the engine's Linear dispatch with w stored under `scheme`.

    Returns the product and the weights it multiplies by in FP32.
    """
    if scheme == "fp16":
        data, scale, deq = bits(w).view(np.uint8), np.zeros(0, dtype=np.uint16), w.float()
    else:
        data, scale, deq = quantized(w, scheme, group)
    return kvb.run_linear(scheme, data, scale, bits(x), group_size=group, **options), deq


# The last two are Whisper's stem on one 30 s window: 3,000 frames of 80 mels, then of 384.
@pytest.mark.parametrize(
    ("n_seqs", "frames", "channels", "d"),
    [(1, 128, 16, 128), (2, 37, 8, 24), (3, 1, 8, 16), (2, 2, 8, 16), (1, 3000, 80, 384)],
)
@pytest.mark.parametrize("stride", [1, 2])
def test_unfold3_then_linear_is_conv1d(kvb, gen, stride, n_seqs, frames, channels, d):
    x = rand_f16(gen, n_seqs, frames, channels)
    w = rand_f16(gen, d, channels, 3, scale=1 / math.sqrt(3 * channels))
    b = rand_f16(gen, d)
    by_channel = x.float().transpose(1, 2)  # [n_seqs, channels, frames], as Conv1d takes it

    unfolded = kvb.unfold3(bits(x.reshape(-1, channels)), n_seqs=n_seqs, stride=stride)
    # [n_seqs, channels, out_frames, 3]: the three frames under each position of the filter.
    windows = F.pad(by_channel, (1, 1)).unfold(2, 3, stride)
    out_frames = windows.shape[2]
    assert out_frames == (frames - 1) // stride + 1
    expected_windows = windows.permute(0, 2, 1, 3).reshape(n_seqs * out_frames, 3 * channels)
    assert np.array_equal(unfolded, bits(expected_windows.to(torch.float16)))

    # The Hugging Face weight [d, channels, 3], flattened as the checkpoint stores it.
    actual, _ = run_linear(kvb, "fp16", w.reshape(d, -1), from_bits(unfolded).half(), bias=bits(b))
    expected = F.conv1d(by_channel, w.float(), b.float(), stride=stride, padding=1)
    expected = expected.transpose(1, 2).reshape(-1, d)
    assert tolerance.rel_rms(from_bits(actual), expected) < F16_ROUNDING


@pytest.mark.parametrize(("rows", "n_out"), [(1, 5), (3, 4097)])
def test_add_bias(kvb, gen, rows, n_out):
    y, b = rand_f16(gen, rows, n_out), rand_f16(gen, n_out)
    expected = (y.float() + b.float()).to(torch.float16)
    assert np.array_equal(kvb.add_bias(bits(y), bits(b)), bits(expected))
    y32 = torch.randn(rows, n_out, generator=gen)
    assert np.array_equal(kvb.add_bias(y32.numpy(), bits(b)), (y32 + b.float()).numpy())


@pytest.mark.parametrize(("n_out", "n_in"), [(8, 8), (300, 512), (51865, 384)])
@pytest.mark.parametrize("rows", [1, 3, 4, 17, 64])
def test_linear_f16_adds_bias(kvb, gen, rows, n_out, n_in):
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    x, b = rand_f16(gen, rows, n_in), rand_f16(gen, n_out)
    expected = x.float() @ w.float().T + b.float()
    f16_out = from_bits(kvb.linear_f16(bits(w), bits(x), bias=bits(b)))
    assert tolerance.rel_rms(f16_out, expected) < F16_ROUNDING
    f32_out = torch.from_numpy(kvb.linear_f16(bits(w), bits(x), f32_out=True, bias=bits(b)))
    assert tolerance.rel_rms(f32_out, expected) < 1e-5


@pytest.mark.parametrize(("n_out", "n_in", "group"), [(8, 64, 32), (300, 512, 128), (200, 96, 32)])
@pytest.mark.parametrize("rows", [1, 3, 4, 17, 64])
@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_quantized_products_add_bias(kvb, gen, scheme, rows, n_out, n_in, group):
    """The bias of both quantized kernels, over every row-count variant and tile shape."""
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    x, b = rand_f16(gen, rows, n_in), rand_f16(gen, n_out)
    data, scale, deq = quantized(w, scheme, group)
    for kernel, weights in (
        (kvb.linear_quantized, deq),
        (kvb.gemm_quantized, deq.to(torch.float16).float()),
    ):
        expected = x.float() @ weights.T + b.float()
        f32_out = kernel(scheme, data, scale, bits(x), group_size=group, f32_out=True, bias=bits(b))
        assert tolerance.rel_rms(torch.from_numpy(f32_out), expected) < 1e-5
        f16_out = kernel(scheme, data, scale, bits(x), group_size=group, bias=bits(b))
        assert tolerance.rel_rms(from_bits(f16_out), expected) < F16_ROUNDING


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("rows", [1, 3, 4, 17, 64])
@pytest.mark.parametrize("scheme", ["fp16", "int8_pc", "int4_g"])
def test_run_linear_with_and_without_bias(kvb, gen, scheme, rows, expand):
    """The engine's dispatch over its five products, each with a bias: the two FP16 kernels
    (cuBLAS past 4 rows), the two quantized ones (tensor cores past 3 rows), and with `expand`
    cuBLAS on the expanded matrix, four slices of output channels at a time."""
    n_out, n_in, group = 200, 512, 128
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    x, b = rand_f16(gen, rows, n_in), rand_f16(gen, n_out)
    options = {"expand_quantized": expand, "dequant_slice_bytes": 64 * n_in * 2}
    for bias in (None, bits(b)):
        f16_out, deq = run_linear(kvb, scheme, w, x, group, bias=bias, **options)
        # Past 3 rows a quantized matrix is multiplied as the FP16 weights it expands to.
        weights = deq if rows <= 3 else deq.to(torch.float16).float()
        expected = x.float() @ weights.T + (0 if bias is None else b.float())
        assert tolerance.rel_rms(from_bits(f16_out), expected) < F16_ROUNDING
        f32_out, _ = run_linear(kvb, scheme, w, x, group, bias=bias, f32_out=True, **options)
        assert tolerance.rel_rms(torch.from_numpy(f32_out), expected) < 1e-5


def test_run_linear_picks_each_kernel_by_scheme_and_rows(kvb, gen):
    """Without a bias the dispatch returns, bit for bit, what the kernel it picks returns."""
    n_out, n_in, group = 200, 512, 128
    w = rand_f16(gen, n_out, n_in, scale=1 / math.sqrt(n_in))
    xs = {rows: rand_f16(gen, rows, n_in) for rows in (1, 3, 4, 5)}

    def dispatched(scheme, rows, **options):
        return run_linear(kvb, scheme, w, xs[rows], group, **options)[0]

    for rows in (1, 4):
        assert np.array_equal(dispatched("fp16", rows), kvb.linear_f16(bits(w), bits(xs[rows])))
    for rows, options in ((5, {}), (1, {"f16_gemv_rows": 0})):
        blas = kvb.linear(bits(w), bits(xs[rows]))
        assert np.array_equal(dispatched("fp16", rows, **options), blas)
    for scheme in ("int8_pc", "int4_g"):
        data, scale, _ = quantized(w, scheme, group)
        for rows, kernel in ((3, kvb.linear_quantized), (4, kvb.gemm_quantized)):
            direct = kernel(scheme, data, scale, bits(xs[rows]), group_size=group)
            assert np.array_equal(dispatched(scheme, rows), direct)
        # A wider in-place limit keeps 4 rows off the tensor cores.
        direct = kvb.linear_quantized(scheme, data, scale, bits(xs[4]), group_size=group)
        assert np.array_equal(dispatched(scheme, 4, quantized_gemm_rows=4), direct)
        expanded = kvb.dequantize(scheme, data, scale, group_size=group)
        blas = kvb.linear(expanded, bits(xs[4]))
        assert np.array_equal(dispatched(scheme, 4, expand_quantized=True), blas)


@pytest.mark.parametrize(("n_always", "n_begin"), [(0, 0), (3, 0), (0, 2), (88, 2), (700, 40)])
def test_suppress_logits(kvb, gen, n_always, n_begin):
    """Six step rows, five of them outputs; the two at position 3 are at the begin position."""
    vocab, begin_pos = 1000, 3
    positions = torch.tensor([0, 3, 2, 3, 4, 7], dtype=torch.int32)
    out_rows = torch.tensor([5, 1, 2, 3, 0], dtype=torch.int32)
    logits = torch.randn(len(out_rows), vocab, generator=gen)
    always = torch.randint(vocab, (n_always,), generator=gen, dtype=torch.int32)
    begin = torch.randint(vocab, (n_begin,), generator=gen, dtype=torch.int32)
    if n_always:
        always[:2] = torch.tensor([0, vocab - 1])

    expected = logits.clone()
    expected[:, always.long()] = float("-inf")
    for o, row in enumerate(out_rows.tolist()):
        if positions[row] == begin_pos:
            expected[o, begin.long()] = float("-inf")
    actual = kvb.suppress_logits(
        logits.numpy(), positions.numpy(), out_rows.numpy(), always.numpy(), begin.numpy(),
        begin_pos,
    )  # fmt: skip
    assert np.array_equal(actual, expected.numpy())
    # What greedy decoding then picks.
    assert kvb.argmax(actual).tolist() == expected.argmax(-1).tolist()


# Whisper's encoder shapes: 1,500 frames with 6 heads (tiny) and 8 (base), and the fixture's 64
# frames with 2. Then a single key, tiles that end mid-way, and head_dim 128.
@pytest.mark.parametrize(
    ("n_seqs", "n_keys", "n_heads", "head_dim"),
    [(1, 1500, 6, 64), (2, 1500, 8, 64), (1, 64, 2, 64), (1, 1, 2, 64), (3, 65, 4, 64),
     (2, 100, 2, 128)],
)  # fmt: skip
def test_dense_attention_over_the_stacked_product(kvb, gen, n_seqs, n_keys, n_heads, head_dim):
    """Every frame attends every frame of its audio; q, k and v are thirds of one value."""
    dim = n_heads * head_dim
    qkv = rand_f16(gen, n_seqs * n_keys, 3 * dim)
    q, k, v = (
        part.float().view(n_seqs, n_keys, n_heads, head_dim).transpose(1, 2)
        for part in qkv.split(dim, dim=1)
    )
    expected = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(-1, dim)
    actual = from_bits(kvb.attention_dense_stacked(bits(qkv), n_seqs, n_heads))
    assert tolerance.rel_rms(actual, expected) < 2 * F16_ROUNDING


# 1,500 keys in one block per sequence is Whisper's cross cache; 100 keys in blocks of 16 pages
# it; 64 fills one key tile exactly.
@pytest.mark.parametrize(("n_keys", "block_size"), [(1500, 1505), (100, 16), (64, 128)])
@pytest.mark.parametrize(("n_heads", "n_kv_heads"), [(8, 8), (4, 2)])
@pytest.mark.parametrize("layout", ["head_major", "token_major", "dim_major"])
@pytest.mark.parametrize("kv_dtype", ["fp16", "int8"])
def test_attention_over_a_fixed_context(
    kvb, gen, kv_dtype, layout, n_heads, n_kv_heads, n_keys, block_size
):
    """Cross attention: queries at any position attend the first n_keys cached positions of their
    sequence, by attention(fixed_ctx) and by attention_dense. The cache is written from one value
    that holds keys and values side by side, as an encoder's stacked product does. Each sequence
    caches five positions more, which no query may see."""
    head_dim, n_seqs, cached = 64, 2, n_keys + 5
    blocks_per_seq = math.ceil(cached / block_size)
    n_blocks = n_seqs * blocks_per_seq
    physical = torch.randperm(n_blocks, generator=gen).view(n_seqs, blocks_per_seq)
    # The cached rows of both sequences, then the queries: 70 consecutive positions of sequence
    # 1 (two tiles), single rows of both, and a run of sequence 0 that starts past the keys.
    keys = [(s, p) for s in range(n_seqs) for p in range(cached)]
    queries = [(1, p) for p in range(70)] + [(0, 3), (1, 0), (0, 447)]
    queries += [(0, p) for p in range(n_keys + 2, n_keys + 9)]
    seqs = torch.tensor([s for s, _ in keys + queries], dtype=torch.int32)
    positions = torch.tensor([p for _, p in keys + queries], dtype=torch.int32)
    key_seqs, key_pos = seqs[: len(keys)].long(), positions[: len(keys)]
    slots = torch.zeros(len(seqs), dtype=torch.int32)
    in_block = key_pos % block_size
    slots[: len(keys)] = physical[key_seqs, (key_pos // block_size).long()] * block_size + in_block
    q = rand_f16(gen, len(seqs), n_heads, head_dim)
    k, v = (rand_f16(gen, len(keys), n_kv_heads, head_dim) for _ in range(2))

    def run(kernel, wide=True):
        return from_bits(
            kvb.kv_attention(
                bits(q), bits(k), bits(v), positions.numpy(), seqs.numpy(), slots.numpy(),
                physical.to(torch.int32).numpy(), block_size, n_blocks, layout,
                query_from=len(keys), wide_key_loads=wide, kv_dtype=kv_dtype, kernel=kernel,
                fixed_ctx=n_keys, stacked_kv=True,
            )
        )  # fmt: skip

    # The tensor-core kernel needs adjacent key dimensions.
    actual = [run("decode", wide) for wide in (True, False)]
    if layout != "dim_major":
        actual.append(run("dense"))

    cached_k, cached_v = (int8_readback(x) if kv_dtype == "int8" else x.float() for x in (k, v))
    expected = torch.zeros(len(queries), n_heads, head_dim)
    for s in range(n_seqs):
        mine = seqs[len(keys) :] == s
        # [heads, positions, head_dim], each KV head repeated for the query heads that read it.
        ks, vs = (
            x[s * cached : s * cached + n_keys].transpose(0, 1)
            .repeat_interleave(n_heads // n_kv_heads, 0)
            for x in (cached_k, cached_v)
        )  # fmt: skip
        qs = q[len(keys) :][mine].float().transpose(0, 1)
        expected[mine] = F.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1)
    for out in actual:
        assert tolerance.rel_rms(out, expected) < 2 * F16_ROUNDING


def test_dense_attention_needs_adjacent_key_dimensions(kvb, gen):
    q, k = rand_f16(gen, 4, 2, 64), rand_f16(gen, 4, 1, 64)
    positions = np.arange(4, dtype=np.int32)
    with pytest.raises(RuntimeError, match="adjacent key dimensions"):
        kvb.kv_attention(
            bits(q), bits(k), bits(k), positions, np.zeros(4, dtype=np.int32), positions,
            np.zeros((1, 1), dtype=np.int32), 4, 1, "dim_major", kernel="dense", fixed_ctx=4,
        )  # fmt: skip


@pytest.mark.parametrize("kernel", ["decode", "dense"])
def test_fixed_context_must_fit_the_cache(kvb, gen, kernel):
    """Five keys asked of a cache whose sequences hold one block of four positions."""
    q, k = rand_f16(gen, 4, 2, 64), rand_f16(gen, 4, 1, 64)
    positions = np.arange(4, dtype=np.int32)
    with pytest.raises(RuntimeError, match="sequence's cache"):
        kvb.kv_attention(
            bits(q), bits(k), bits(k), positions, np.zeros(4, dtype=np.int32), positions,
            np.zeros((1, 1), dtype=np.int32), 4, 1, kernel=kernel, fixed_ctx=5,
        )  # fmt: skip
