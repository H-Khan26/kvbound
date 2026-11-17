# Optimization log (Phase 6)

Each entry gives the change, the hypothesis behind it, before and after measurements, and the
profiler evidence. Everything was measured on the RTX 4070 (`rtx4070-d6f35005`, Windows,
unlocked clocks, records tagged `os=windows`), where runs vary by a few percent. To keep that
noise out of the comparisons, every before/after pair ran back to back at one commit
(`bdcdb3d`). The "before" side uses the engine options that keep the Phase 5 paths:
`--no-fuse --f16-gemv-rows 0 --no-prefill-attention --expand-quantized`, plus
`--quantized-gemm-rows 16` when serving. Two changes have no switch, so the "before" runs already
include them (entries 3 and 4). Measured against the Phase 5 records from `3c3d2b3` instead,
every gain below is larger.

Reproduce with the `make bench`, `make serve`, `make precision` and `make step-counters`
targets. The counter tables come from `kvbound.step_counters` (Nsight Compute, one step, caches
flushed before each replay) and are kept in `results/profiles/*-counters-*`.

## Summary

Batch-1 decode with a 128-token prompt and 128 new tokens; TTFT for a 1,024-token prompt;
serving is 96 equal-length requests (256-token prompts, 128 new tokens) in the 176 MiB paged
pool, 16-token blocks, up to 64 sequences.

| TinyLlama 1.1B | Before | After | Change |
| --- | --- | --- | --- |
| FP16 decode | 202 tok/s (88% of the ceiling) | 212 tok/s (92%) | +4.5% |
| INT8 decode | 370 tok/s (81%) | 391 tok/s (86%) | +5.5% |
| INT4 decode | 610 tok/s (69%) | 673 tok/s (76%) | +10% |
| TTFT, 128-token prompt (FP16 / INT4) | 9.1 / 13.6 ms | 6.7 / 6.5 ms | 1.4x / 2.1x |
| TTFT, 1,024-token prompt (FP16) | 115.8 ms | 38.6 ms | 3.0x |
| Serving FP16 / INT8 / INT4 | 2,320 / 1,979 / 1,797 tok/s | 2,824 / 3,623 / 3,943 tok/s | 1.2x / 1.8x / 2.2x |

| Llama 3.2 1B | Before | After | Change |
| --- | --- | --- | --- |
| FP16 decode | 178 tok/s (93%) | 183 tok/s (96%) | +3.1% |
| INT8 decode | 336 tok/s (88%) | 348 tok/s (91%) | +3.6% |
| INT4 decode | 585 tok/s (79%) | 624 tok/s (85%) | +6.7% |
| TTFT, 1,024-token prompt (FP16) | 94.7 ms | 39.6 ms | 2.4x |
| Serving FP16 / INT8 / INT4 | 2,298 / 1,592 / 1,657 tok/s | 2,608 / 3,562 / 4,017 tok/s | 1.1x / 2.2x / 2.4x |

Under load, quantized weights were 15% to 31% slower than FP16 before Phase 6. They are now 28%
to 54% faster. Accuracy is unchanged: WikiText-2 perplexity is 8.8723 (TinyLlama FP16), 9.9563
(TinyLlama INT4), 10.9867 (Llama 3.2 FP16) and 14.7617 (Llama 3.2 INT4), the Phase 5 values.
Both models still match the reference token for token (`make test-real`).

## 1. Fuse the decode step

**Change.** `build_llama_program(cfg, fuse=true)` emits 9 ops per layer instead of 16:
- The q, k and v projections are stacked into one matrix at load time, as are gate and up.
  Every weight format stores rows contiguously, so stacking is concatenation, also for INT8 and
  INT4 with their scales.
- Each residual add folds into the RMSNorm that follows it (`AddRmsNorm`).
- RoPE folds into the KV-cache write (`RopeKvWrite`, which also quantizes an INT8 cache).
- SiLU-and-multiply reads the two halves of the stacked gate|up output.

A TinyLlama step drops from 357 launches to 204. The capture sites are unchanged, so the
goldens still apply. Option: `fuse`.

**Hypothesis.** About a millisecond of every decode step was small kernels and the gaps between
them. That is fixed overhead at any precision, so it matters most for INT4, where it was 46% of
the step. Small projections were also inefficient on their own: k and v are 1 MiB each and
streamed at about 270 GB/s under cuBLAS.

**Measured.** With fusion alone, before the other changes: TinyLlama FP16 decode steps fell from
4.97 to 4.83 ms (graphs on). INT4 fell from 1.79 to 1.66 ms and INT8 from 2.81 to 2.70 ms.
Prefill gains more, since every unfused op at 128 rows is a full pass over activations:
TTFT 9.1 to 6.7 ms in FP16.

**Evidence.** In Nsight Systems, the fused step's three small kernels (`rope_kv_write`,
`silu_mul`, the norms) take 1.3, 1.1 and 2.9 µs each. They replace 5 kernels of 1.1 to 2.8 µs
per layer.

## 2. kvbound's own FP16 GEMV instead of cuBLAS for decode

**Change.** `linear_f16` (`engine/kernels/gemv.cu`) uses one warp per output channel and
16-byte loads with a streaming cache hint, accumulating in FP32. Steps of up to `f16_gemv_rows`
rows (4) use it, and wider steps keep cuBLAS.

**Hypothesis.** cuBLAS's GEMV reached 412 GB/s on a decode step, 87% of the 475 GB/s measured
ceiling. A kernel built only to stream the matrix should get within a few percent of it.

**Measured.**
- Per matrix in a TinyLlama decode step, from Nsight Systems: o_proj 436 GB/s, q|k|v 443,
  down 463, gate|up 472, and the vocabulary projection 477, which is the measured ceiling.
- FP16 decode step at batch 1: 4.83 to 4.63 ms with fusion on.
- Against cuBLAS by rows per step (TinyLlama, both fused):

  | Rows | 1 | 2 | 4 | 8 | 16 |
  | --- | --- | --- | --- | --- | --- |
  | cuBLAS (ms) | 4.75 | 5.07 | 5.09 | 5.19 | 5.51 |
  | kvbound (ms) | 4.70 | 4.85 | 4.93 | 5.37 | 7.52 |

  From 8 rows the per-row FP32 arithmetic costs more than the reads it shares, hence the
  default of 4.

**What did not work.** Four or eight loads in flight per lane (predicated, unrolled) measured
0.8% and 1.6% slower than one. The extra registers cut resident warps, and at this occupancy
warps hide memory latency better than per-lane parallelism does. The same change made the INT4
kernel 15% slower, and it was reverted. Without the streaming hint, the FP16 kernel was 1% slower.

## 3. Streaming loads in the quantized GEMV (no switch)

**Change.** `linear_quantized` reads weights with `__ldcs`, which keeps them out of L1 so L1
holds the rows of x that every warp of a block rereads.

**Measured.** Alternating two builds on TinyLlama decode: INT4 1.531 to 1.461 ms (−4.6%), INT8
2.555 to 2.515 ms. Both builds include entries 1 and 4. The "before" records include this
change.

## 4. RMSNorm in registers with 16-byte accesses (no switch)

**Change.** A row of up to 8,192 elements is loaded once as 16-byte chunks, kept in registers
between the sum of squares and the scaling, and written as 16-byte chunks. Before, the kernel
made two strided passes of 2-byte accesses.

**Hypothesis.** A decode step normalizes one row per block, so the kernel's time is its chain
of dependent memory accesses, and shortening that chain cuts its latency.

**Measured.** INT4 decode with fusion off, so 45 norms per step: 1.79 to 1.69 ms between the
Phase 5 build and one with this change (plus entry 3's build state). The "before" records
include this change.

## 5. Quantized weights on tensor cores for steps wider than 3 rows

**Change.** `gemm_quantized` (`engine/kernels/quantized.cu`) is a W8A16/W4A16 product on
`mma.sync.m16n8k16`:
- Packed weights and x arrive by `cp.async` in 2 to 4 stages.
- Each weight is expanded to FP16 in the registers the MMA reads, so the weights cross DRAM and
  shared memory at their packed size.
- The tile shape is picked by row count.
- Each dequantized weight equals what `dequantize` writes (`__float2half(q * scale)`), and a
  test checks this bit for bit.

It replaces expanding every matrix to FP16 in 32 MiB slices for cuBLAS, which `expand_quantized`
keeps. `quantized_gemm_rows`, the widest step the in-place kernel takes, moved from 16 to 3.

**Hypothesis.** Expanding cost about 8 ms a step under load, and the in-place kernel turns
arithmetic-bound beyond a few rows: every row adds FP32 work per weight on CUDA cores. A
tensor-core product that reads the packed weights once removes both.

**Measured.**
- Kernel alone, TinyLlama gate|up (2048 × 11264), against expand + cuBLAS:
  - INT8: 53 against 193 µs at 17 rows, 82 against 245 µs at 64 rows, 863 against 949 µs at
    1,024 rows (55 TFLOPS).
  - INT4: 37 against 196 µs at 17 rows.
  - At 512 to 1,024 rows it reaches 53 to 57 TFLOPS, against 59 for FP16 cuBLAS with FP32
    accumulation.
- Decode steps by rows, TinyLlama:

  | Rows | 1 | 2 | 3 | 4 | 8 | 16 |
  | --- | --- | --- | --- | --- | --- | --- |
  | INT4 in place (ms) | 1.48 | 1.82 | 2.39 | 3.04 | 5.52 | 10.78 |
  | INT4 tensor cores (ms) | 2.27 | 2.42 | 2.39 | 2.44 | 2.57 | 2.88 |
  | INT8 in place (ms) | 2.55 | 2.78 | 2.81 | 2.96 | 4.05 | 7.05 |
  | INT8 tensor cores (ms) | 2.78 | 2.94 | 2.92 | 2.96 | 3.12 | 3.40 |

  The two tie at 3 rows (INT4) and 4 rows (INT8), hence the default of 3.
- Serving 96 requests: TinyLlama INT4 1,797 to 3,943 tokens/s, Llama 3.2 INT4 1,657 to 4,017.
  Time per output token falls from 12.6 to 5.7 ms and from 13.7 to 5.6 ms.
- TTFT with INT4 weights and a 128-token prompt: 13.6 to 6.5 ms.

**Evidence.** Nsight Compute shows the weights read once per launch at 32 rows: 67.6 MB of DRAM
reads for 65.5 MB of INT8 weights. At small row counts INT4 is bound by instruction issue, not
memory (170 to 400 GB/s), which is why the in-place kernel still wins at 1 and 2 rows.

## 6. Prefill attention on tensor cores

**Change.** `attention_prefill` (`engine/kernels/attention.cu`) is flash attention:
- One block per (tile of up to 64 query rows, query head), with 4 warps of 16 rows each.
- 64-position K/V tiles are double-buffered in shared memory with an XOR swizzle, and read
  through the block table, so paged and static caches both work, and INT8 is expanded on load.
- Online softmax in FP32, `mma.sync` for both products, causal masking per row.

`Engine::forward` plans the tiles on the host from the step's rows (`plan_attention_tiles`). A
step that contains a run of several rows of one sequence takes this kernel and runs eagerly.
Decode-shaped steps keep the one-block-per-query kernel and their CUDA graphs. Option:
`prefill_attention`.

**Hypothesis.** Prefill reused the decode kernel. Every query of a 1,024-token prompt read every
earlier key of its sequence from L2 on CUDA cores, so attention was 3.8 ms per layer and the
prompt took 116 ms, of which 35 ms is matrix products.

**Measured.**
- Per layer at 1,024 tokens: 3,777 to 114 µs (33x). Paged 16-token blocks: 4,061 to 118 µs.
  INT8 cache: 3,812 to 124 µs.
- TTFT for 1,024 tokens: TinyLlama 115.8 to 38.6 ms, Llama 3.2 94.7 to 39.6 ms.
- Matrix products now take 35 of TinyLlama's 38.6 ms, which is 1.98 TFLOP at 56 TFLOPS, 95% of
  the 59 TFLOPS FP16-with-FP32-accumulate peak. The perf model's 23 ms bound uses the FP16
  accumulate peak (91 TFLOPS), which the engine gives up for correctness.

**Evidence.** In Nsight Compute the old kernel uses no tensor pipe. The new one runs the tensor
pipe at 41% of its FP16-accumulate peak at 1,024 tokens (about 39 TFLOPS of FP32-accumulate
work, roughly 70% of what that mode allows). DRAM is at 9%, so the kernel is bound by
arithmetic, as prefill attention should be.

**What did not work.** Serving 2 or 4 query heads of a KV head from one staged tile was slower
(124 to 141 µs against 117 µs). Larger blocks fit fewer warps per SM, and the K/V tiles came
from L2 anyway.

## Per-kernel counters, one TinyLlama step

Microseconds of kernel time in one TinyLlama step under Nsight Compute, with the caches flushed
before each replay; launches in parentheses. These are Nsight Compute replays, not the recorded steps,
so the small kernels look slower than they do in a CUDA graph. Profiles:
`results/profiles/20261117T1318*` and `20261117T1319*`.

| Decode step (128-token prompt) | FP16 before | FP16 after | INT4 before | INT4 after |
| --- | --- | --- | --- | --- |
| Matrix products | 4,769 (155), cuBLAS GEMV at 435 GB/s | 4,503 (89), own GEMV at 460 GB/s | 1,484 (155) at 361 GB/s | 1,331 (89) at 401 GB/s |
| Attention | 170 (22) | 172 (22) | 171 (22) | 172 (22) |
| Norms | 102 (45) | 106 (45) | 102 (45) | 106 (45) |
| RoPE and cache write | 136 (66) | 53 (22) | 137 (66) | 52 (22) |
| Residual adds | 72 (44) | 2 (1) | 73 (44) | 2 (1) |
| SwiGLU | 47 (22) | 46 (22) | 47 (22) | 46 (22) |
| Step | 5,305 (357) | 4,891 (204) | 2,024 (357) | 1,718 (204) |

| Prefill, 1,024 tokens | FP16 before | FP16 after | INT4 before | INT4 after |
| --- | --- | --- | --- | --- |
| Attention | 82,704 (22), no tensor pipe | 2,561 (22), tensor pipe 41% | 83,037 (22) | 2,539 (22), 42% |
| Matrix products | 38,707 (154), cuBLAS, tensor pipe 45 to 49% | 38,231 (88), cuBLAS | 44,747 (308): expand 5,743 + cuBLAS | 40,695 (88), own, tensor pipe 45% |
| Step | 125,386 (357) | 44,223 (204) | 131,560 (511) | 46,442 (204) |

The tensor-pipe figure is against the FP16-accumulate rate, so with FP32 accumulation about 50% is
the practical ceiling. In decode every product kernel shows 0% tensor use and 84 to 88% of warps
resident: it is a memory stream, and the remaining gap to 475 GB/s is the smaller matrices (o_proj at
436 GB/s against gate|up at 472 in Nsight Systems).

## Not done in Phase 6

- **The 3060 Ti comparison** the roadmap asks for. That card re-records the final state with the
  same targets, and the explanation goes to the results page.
- **Decode attention at long context.** At batch 1 a 1,100-token context costs about 0.3 ms
  more per step than a 190-token one. The kernel runs one block per head, 32 blocks on 46 SMs.
  Splitting each head's context across blocks (flash decoding) would use the idle SMs.
- **FP16 products for 5 to 64 rows.** cuBLAS takes them at 5.2 to 5.5 ms per TinyLlama step,
  about 15% above the weights' memory time. An FP16 variant of `gemm_quantized` would cover the
  range.
- **CUDA graphs for mixed prefill and decode steps.** These steps run eagerly. Launch cost has
  halved with fusion, but graphs would remove the rest.
- **FP8 KV cache and AWQ scales**, still open from Phase 5.
