# Optimization log (Phase 6)

Each entry gives the change, the hypothesis behind it, before and after measurements, and the
profiler evidence. The six entries were measured on the RTX 4070 (`rtx4070-d6f35005`, Windows,
unlocked clocks, records tagged `os=windows`), where runs vary by a few percent. To keep that
noise out of the comparisons, every before/after pair ran back to back at one commit
(`bdcdb3d`). The "before" side uses the engine options that keep the Phase 5 paths:
`--no-fuse --f16-gemv-rows 0 --no-prefill-attention --expand-quantized`, plus
`--quantized-gemm-rows 16` when serving. Two changes have no switch, so the "before" runs already
include them (entries 3 and 4). Measured against the Phase 5 records from `3c3d2b3` instead,
every gain below is larger.

The RTX 3060 Ti (`rtx3060ti-fd7b40d5`, same setup) then ran the same pairs and counters at
`c442759`. Its tables follow the 4070 material, and
[The two cards compared](#the-two-cards-compared) explains the differences through memory
bandwidth, tensor-core throughput, L2 size and FP8.

Reproduce with the `make bench`, `make serve`, `make precision` and `make step-counters`
targets. The counter tables come from `kvbound.step_counters` (Nsight Compute, one step, caches
flushed before each replay) and are kept in `results/profiles/*-counters-*`.

## Summary (RTX 4070)

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

## Per-kernel counters, one TinyLlama step (RTX 4070)

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

## Summary (RTX 3060 Ti)

The same pairs with the same options, workloads and pool sizes, back to back at `c442759`. Each
share is of this card's own ceiling, set by the 415 GB/s measured here (475 GB/s on the 4070).

| TinyLlama 1.1B | Before | After | Change |
| --- | --- | --- | --- |
| FP16 decode | 162 tok/s (81% of the ceiling) | 178 tok/s (89%) | +9.6% |
| INT8 decode | 293 tok/s (73%) | 315 tok/s (79%) | +7.2% |
| INT4 decode | 450 tok/s (58%) | 477 tok/s (62%) | +5.8% |
| TTFT, 128-token prompt (FP16 / INT4) | 14.3 / 20.3 ms | 10.5 / 11.4 ms | 1.4x / 1.8x |
| TTFT, 1,024-token prompt (FP16) | 205.4 ms | 76.4 ms | 2.7x |
| Serving FP16 / INT8 / INT4 | 1,463 / 979 / 1,013 tok/s | 1,740 / 2,163 / 2,219 tok/s | 1.2x / 2.2x / 2.2x |

| Llama 3.2 1B | Before | After | Change |
| --- | --- | --- | --- |
| FP16 decode | 138 tok/s (82%) | 146 tok/s (87%) | +5.5% |
| INT8 decode | 253 tok/s (76%) | 266 tok/s (80%) | +5.3% |
| INT4 decode | 414 tok/s (64%) | 441 tok/s (69%) | +6.8% |
| TTFT, 1,024-token prompt (FP16) | 167.0 ms | 72.4 ms | 2.3x |
| Serving FP16 / INT8 / INT4 | 1,554 / 937 / 1,002 tok/s | 1,827 / 2,217 / 2,342 tok/s | 1.2x / 2.4x / 2.3x |

Under load, quantized weights were 31% to 40% slower than FP16 through the Phase 5 paths on this
card. They are now 21% to 28% faster. The median time per output token while serving INT4 falls
from 22.4 to 10.0 ms on TinyLlama and from 22.7 to 9.5 ms on Llama 3.2. Perplexity matches the
4070's to within 0.0001: 8.8723, 9.9563, 10.9866 and 14.7617.

This card sits in a shared desktop. Three things in these records come from that, not from the
engine:

- **The TinyLlama INT4 "after" run was disturbed.** Its ten iterations read 414 to 509 tok/s, a
  standard deviation of 34.8 tok/s against 8.5 before. By the median step time, which every record
  keeps because an occasional delayed step does not move it, the pair reads 449 and 494 tok/s
  (+10.0%), and the `make precision` run later the same evening reads 504 tok/s. The table's
  +5.8% is the low end of the gain, not a result that differs from the 4070's +10%.
- **The Llama 3.2 pair ran about 5% slow on both sides.** In each of its decode runs the mean
  step is longer than the median step (FP16 after: 146 tok/s by the mean, 152 by the median). The
  later `make bench` and `make precision` runs read 154 to 155 / 284 / 472 tok/s where the pair's
  "after" reads 146 / 266 / 441. Both sides of each pair were slowed alike: by the median step
  the changes are +7.1% / +4.5% / +7.3%, against +5.5% / +5.3% / +6.8% in the table.
- **Run-to-run spread is about 6%.** TinyLlama FP16 decode in the final state reads 178 tok/s in
  the pair, 168 in `make bench` and 170 in `make precision`. A difference of that size between
  two records is not a gain or a loss. Only the back-to-back pairs support a before/after claim.

Of the records in the tables, the serving "after" runs with quantized weights are tagged
`clock_varied` (TinyLlama INT8 and INT4, Llama 3.2 INT8): the SM clock moved between 1,815 and
1,950 MHz during them.

## Per-kernel counters, one TinyLlama step (RTX 3060 Ti)

The same method as the 4070 tables: microseconds of kernel time under Nsight Compute, launches in
parentheses. Profiles: `results/profiles/20261117T2123*` to `20261117T2125*`.

| Decode step (128-token prompt) | FP16 before | FP16 after | INT4 before | INT4 after |
| --- | --- | --- | --- | --- |
| Matrix products | 5,912 (177), cuBLAS at 351 GB/s | 5,281 (89), own GEMV at 394 GB/s | 2,237 (155) at 241 GB/s | 1,971 (89) at 273 GB/s |
| Attention | 274 (22) | 272 (22) | 275 (22) | 275 (22) |
| Norms | 153 (45) | 159 (45) | 151 (45) | 158 (45) |
| RoPE and cache write | 199 (66) | 80 (22) | 202 (66) | 80 (22) |
| Residual adds | 105 (44) | 2 (1) | 104 (44) | 2 (1) |
| SwiGLU | 68 (22) | 66 (22) | 66 (22) | 66 (22) |
| Step | 6,725 (379) | 5,874 (204) | 3,049 (357) | 2,566 (204) |

| Prefill, 1,024 tokens | FP16 before | FP16 after | INT4 before | INT4 after |
| --- | --- | --- | --- | --- |
| Attention | 173,538 (22), no tensor pipe | 4,976 (22), tensor pipe 42% | 173,554 (22) | 4,889 (22), 43% |
| Matrix products | 90,104 (199), cuBLAS, tensor pipe 41 to 48% | 88,499 (89), cuBLAS, 47 to 48% | 97,386 (353): expand 7,533 + cuBLAS | 83,428 (89), own, tensor pipe 46% |
| Step | 270,234 (401) | 99,424 (204) | 277,521 (555) | 94,267 (204) |

What the counters show on this card:

- **cuBLAS uses two kernels for the decode products.** Of the 155 products of an unfused step,
  133 go through its GEMV at 360 GB/s. The 22 `down_proj` matrices (2,048 × 5,632, 23.1 MB each)
  go through `cublas_Kernel2`, each followed by a `splitKreduce` launch: 1,562 µs for those 44
  launches, at 326 GB/s, with the tensor pipe 12% active. That is why the unfused FP16 step has
  379 launches for 357 ops. Together cuBLAS streams at 351 GB/s, 85% of the measured 415. On the
  4070 all 155 products went through the GEMV, at 92% of that card's ceiling.
- **kvbound's FP16 GEMV reads at 394 GB/s, 95% of the ceiling.** By matrix: q|k|v 380 GB/s,
  o_proj and down_proj 373, gate|up 411, and the vocabulary projection 417, which is the measured
  ceiling. Product time falls 10.7% (5.6% on the 4070), because cuBLAS started further from the
  ceiling here.
- **The INT4 GEMV does not reach the memory ceiling on any matrix.** Fusion raises it from 241 to
  273 GB/s (warps resident from 63% to 81%), which is 66% of the ceiling. By matrix it reads at
  211 GB/s (q|k|v) to 348 GB/s (the vocabulary projection, 84% of the ceiling). On the 4070 the
  same kernel reads at 401 GB/s, 85% of its ceiling, and at 98% on the vocabulary projection.
- **Tiled prefill attention: 7,888 to 226 µs per layer (35x), tensor pipe 42%.** The old kernel
  was 64% of the profiled step.
- **Prefill products run the tensor pipe at 46% to 48%**, against the FP16-accumulate rate, as on
  the 4070. cuBLAS picks other kernels here (`sm80_xmma_gemm` and `ampere_fp16_s16816gemm`
  variants). Under the profiler `gemm_quantized` is the faster of the two on this card: 83,330 µs
  for the 88 INT4 products against 88,169 µs for the same 88 in FP16 through cuBLAS. On the 4070
  cuBLAS was the faster one.
- **The profiler overstates prefill time on this card.** The profiled FP16 decode step is 5.87 ms
  against 5.62 ms recorded (+4%; +3% on the 4070). The profiled 1,024-token prefill is 99.4 ms
  against a recorded TTFT of 76.4 ms (+30%); on the 4070 it is 44.2 against 38.6 ms (+15%). The
  shares and counters inside one profile stand. Ratios of prefill kernel time between the two
  cards do not: the profiles say 2.25x, the records 1.98x.

## The two cards compared

What Phase 1 measured on each card (`microbench` records under `results/raw/`):

| Hardware | RTX 3060 Ti | RTX 4070 | 4070 / 3060 Ti |
| --- | --- | --- | --- |
| Memory bandwidth beyond cache, best 2 ms burst | 415 GB/s | 475 GB/s | 1.14 |
| GEMM peak, FP16 with FP32 accumulate | 32.7 TFLOPS | 59.1 TFLOPS | 1.81 |
| GEMM peak, FP16 with FP16 accumulate / INT8 | 57.7 TFLOPS / 114.6 TOPS | 91.5 TFLOPS / 215.2 TOPS | 1.58 / 1.88 |
| GEMM peak, FP8 E4M3 | not supported | 118.3 TFLOPS | |
| L2 cache, as the driver reports it | 3 MiB | 36 MiB | 12 |
| Dependent-load latency at a 4 MiB working set | 308 ns | 141 ns | |
| SMs; SM clock under load | 38; about 1,980 MHz | 46; about 2,790 MHz | 1.21; 1.41 |

The 3 MiB is what the device query in the bandwidth sweep returns; `configs/devices/rtx3060ti.toml`
lists the spec sheet's 4 MB.

What Phase 6 measured, final state, from each card's back-to-back pair:

| Phase 6 paths | RTX 3060 Ti | RTX 4070 | 4070 / 3060 Ti |
| --- | --- | --- | --- |
| TinyLlama decode, FP16 / INT8 / INT4 | 178 / 315 / 477 tok/s | 212 / 391 / 673 tok/s | 1.19 / 1.24 / 1.41 |
| Llama 3.2 decode, FP16 / INT8 / INT4 | 146 / 266 / 441 tok/s | 183 / 348 / 624 tok/s | 1.26 / 1.31 / 1.41 |
| TinyLlama share of the ceiling | 89% / 79% / 62% | 92% / 86% / 76% | |
| Llama 3.2 share of the ceiling | 87% / 80% / 69% | 96% / 91% / 85% | |
| TTFT, 1,024-token prompt, TinyLlama / Llama 3.2 | 76.4 / 72.4 ms | 38.6 / 39.6 ms | 1.98 / 1.83 |
| TinyLlama serving, FP16 / INT8 / INT4 | 1,740 / 2,163 / 2,219 tok/s | 2,824 / 3,623 / 3,943 tok/s | 1.62 / 1.67 / 1.78 |
| Llama 3.2 serving, FP16 / INT8 / INT4 | 1,827 / 2,217 / 2,342 tok/s | 2,608 / 3,562 / 4,017 tok/s | 1.43 / 1.61 / 1.72 |

The 3060 Ti's later `make precision` records differ from its pair by up to 7% (see the notes under
its summary). Taking either for that card, the decode ratios span 1.18 to 1.26 in FP16, 1.23 to
1.31 in INT8 and 1.32 to 1.41 in INT4, and the serving ratios 1.36 to 1.78.

### Memory bandwidth: FP16 decode, and less of the step as the weights shrink

- The FP16 products are a memory stream on both cards. `linear_f16` reads at 394 GB/s here and
  460 GB/s on the 4070, 95% and 97% of each ceiling, so its time differs by 1.17x where the
  bandwidths differ by 1.14x.
- The rest of the step does not follow memory. The 115 launches that are not products take
  593 µs here and 388 µs on the 4070 (1.53x), and each small kernel is 1.42 to 1.58 times slower.
  That is close to the ratio of the SM clocks (1.41), not to that of the bandwidths: at one row
  these kernels read 0.03 to 0.19 MB per launch, far too little for bandwidth to matter.
- So TinyLlama's FP16 step differs by 1.20x in the profiles and 1.19x in the pair records, and
  the ratio grows as the weights shrink and the fixed part becomes a larger share: 1.23x to 1.31x
  in INT8 and 1.32x to 1.41x in INT4 over both models.
- In INT4 the product kernel itself is off the memory bound on this card: 273 of 415 GB/s (66%),
  against 401 of 475 (85%) on the 4070, 1.48x in time. Resident warps are about the same (81% and
  84%), so occupancy is not what separates the cards. The counters say that memory is not the
  limit here. They do not say which instruction resource is: 1.48x lies between the clock ratio
  (1.41) and the ratio of SMs times clock (1.71).
- This is why the 3060 Ti is further from its own ceiling at every precision, and furthest in
  INT4: TinyLlama's recorded step is 1.99 to 2.10 ms against a floor of 1.30 ms.

### Tensor-core throughput: prefill and most of serving

- The 1,024-token prompt is 1.98x (TinyLlama) and 1.83x (Llama 3.2) slower here. The FP16 GEMM
  peak with FP32 accumulation, which the engine's products use, differs by 1.81x.
- TinyLlama's prompt is 1.98 TFLOP of matrix products. At each card's Phase 1 peak that is
  60.7 ms of the 76.4 ms here and 33.6 ms of the 38.6 ms on the 4070. For Llama 3.2 it is 61.0 of
  72.4 ms and 33.7 of 39.6 ms. For TinyLlama that leaves 15.6 ms here and 5.1 ms on the 4070 for
  everything else, including any shortfall of the products from their peak.
- The tensor pipe is as busy on one card as on the other, each against its own peak: 47 to 48%
  here and 48 to 49% on the 4070 in cuBLAS, 46% and 45% in `gemm_quantized`, 42% and 41% in
  prefill attention.
- Serving steps carry up to 30 rows and sit between the two bounds: 1.43x to 1.78x. The ratio is
  highest with INT4 weights (1.72x and 1.78x), where the least memory traffic is left in a step.

### L2 size: DRAM traffic in wide steps

- **Decode does not depend on it.** The weights never fit either L2. One sequence's keys and
  values for a layer are 132 KB at 129 tokens and 1 MiB at 1,024 tokens, which fit both. Decode
  attention takes 272 µs here and 172 µs on the 4070 for 22 layers (1.58x), the same ratio as the
  other small kernels, and reads 4.1 against 3.6 MB from DRAM.
- **The Phase 3 KV layout benchmark agrees.** For decode attention at batch 1 the ratio between
  the cards falls from 1.59 at 128 cached tokens to 1.32 at 4,096 tokens, where a layer's keys
  and values are 4 MiB. At batch 8 it stays between 1.34 and 1.42 from 2 MiB to 32 MiB per layer.
  Nothing changes where the working set passes 3 MiB, because the kernel reads each key once.
- **A 1,024-row step does depend on it.** Its activations are 4.2 MB per tensor at the hidden
  width and up to 23.1 MB (the gate|up output): more than 3 MiB, less than 36 MiB.
  - The fused add-and-norm reads 8.2 MB and writes 6.8 MB per launch. Here both cross DRAM while
    the kernel runs: 397 GB/s of reads plus writes, 95% of the ceiling, 1,708 µs for the 45
    launches. On the 4070 the counters show no DRAM writes during the kernel and reads at
    411 GB/s: 891 µs. That is 1.92x where the bandwidths differ by 1.14x. Norms, SwiGLU, RoPE
    with the cache write, the add and the embedding are 5,938 µs of the profiled step here and
    3,147 µs on the 4070, 6.0% and 7.1% of each.
  - The tensor-core products move far more bytes here. For the 88 INT4 products the 4070 reads
    1,032 MB from DRAM and writes 262 MB. The 3060 Ti reads 7,114 MB and writes 4,128 MB, 8.7
    times the traffic. Per launch, gate|up reads 175.7 MB here against 16.1 MB, and q|k|v 37.6
    against 6.9 MB. The FP16 products through cuBLAS read 5,970 MB against 2,608 MB. We read the
    excess as each column of blocks fetching the step's input rows again once they have left L2.
  - No record shows what that traffic costs. In `gemm_quantized` it adds up to 135 GB/s, 32% of
    the ceiling. The tensor pipe is as busy as on the 4070 (45.7% against 45.5%). And in the
    profiles the three matrix shapes are 1.95 to 2.12 times slower than on the 4070, about the
    same for each, although they read 4.6 to 10.9 times the 4070's bytes.
- **Phase 1 isolates L2 on its own.** A chain of dependent loads over a 4 MiB working set costs
  308 ns per load here and 141 ns on the 4070, whose L2 still holds it.
- **What would isolate it in the engine:** the same product at 256 rows, whose input is 1 MiB,
  and at 1,024 rows on this card, timed outside the profiler; or a variant that walks the rows in
  tiles small enough to stay in L2. Phase 6 recorded neither.

### FP8: explains none of it

Only the 4070 has FP8 tensor cores. Phase 1 measured 118.3 TFLOPS for an FP8 E4M3 GEMM there,
and the 3060 Ti (compute capability 8.6) cannot run it. The engine has no FP8 path, since the
FP8 KV cache is not built, so no number in this log depends on it.

### What Phase 6 bought on each card

- **The FP16 GEMV bought twice as much here**: +9.6% against +4.5% on TinyLlama decode, +5.5%
  against +3.1% on Llama 3.2. cuBLAS was at 85% of the ceiling on this card and 92% on the 4070.
- **Tensor-core products for quantized weights bought about the same**: serving is 2.2x to 2.4x
  faster here and 1.8x to 2.4x on the 4070. Before, quantized weights were 31% to 40% slower than
  FP16 under load here and 15% to 31% there.
- **Tiled prefill attention bought slightly less in TTFT**: 2.7x and 2.3x here, 3.0x and 2.4x on
  the 4070. The old kernel was the same share of the profiled step on both (64% and 66%). What
  remains is products, for which this card's peak is 1.81 times lower.

## Not done in Phase 6

- **Wide products on a small L2.** At 1,024 rows the 3060 Ti moves 8.7 times the DRAM bytes of
  the 4070 through `gemm_quantized`. Whether tiling the rows would shorten the step has not been
  measured.
- **INT4 decode on the 3060 Ti.** The product kernel reads at 66% of the memory ceiling there,
  and the step is at 62% to 65% of its ceiling. The entries above were developed and their
  crossovers (`f16_gemv_rows`, `quantized_gemm_rows`) chosen on the 4070.
- **Decode attention at long context.** At batch 1 a 1,100-token context costs about 0.3 ms
  more per step than a 190-token one. The kernel runs one block per head, 32 blocks on 46 SMs.
  Splitting each head's context across blocks (flash decoding) would use the idle SMs.
- **FP16 products for 5 to 64 rows.** cuBLAS takes them at 5.2 to 5.5 ms per TinyLlama step,
  about 15% above the weights' memory time. An FP16 variant of `gemm_quantized` would cover the
  range.
- **CUDA graphs for mixed prefill and decode steps.** These steps run eagerly. Launch cost has
  halved with fusion, but graphs would remove the rest.
- **FP8 KV cache**, still open from Phase 5. AWQ scales for INT4, the other item open from
  Phase 5, were built after this log's measurements: WikiText-2 perplexity goes from 9.96 to 9.40
  on TinyLlama and from 14.76 to 12.91 on Llama 3.2 (README, Quantization). The decode speed of
  those checkpoints has not been recorded.
