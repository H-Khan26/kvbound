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

Decode on this card is read by the median step of a run (`decode_step_p50_s`), not by its mean
step (`decode_tok_s`, which the 4070 summary above prints). Other programs on this desktop take
the GPU during some of a run's steps. That lengthens the mean step and leaves the median where it
was, which is why every record keeps both. On the 4070 the two readings agree within 0.5% in every
pair record. Here they differ by more than 2% in 14 of the 34 Phase 6 decode records, by 3.1% to
6.3%; those runs are called disturbed below. TTFT and serving throughput have no second reading
and are as recorded.

| TinyLlama 1.1B | Before | After | Change |
| --- | --- | --- | --- |
| FP16 decode | 162 tok/s (81% of the ceiling) | 178 tok/s (89%) | +9.5% |
| INT8 decode | 293 tok/s (73%) | 314 tok/s (79%) | +7.1% |
| INT4 decode | 449 tok/s (58%) | 494 tok/s (64%) | +10.0% |
| TTFT, 128-token prompt (FP16 / INT4) | 14.3 / 20.3 ms | 10.5 / 11.4 ms | 1.4x / 1.8x |
| TTFT, 1,024-token prompt (FP16) | 205.4 ms | 76.4 ms | 2.7x |
| Serving FP16 / INT8 / INT4 | 1,463 / 979 / 1,013 tok/s | 1,740 / 2,163 / 2,219 tok/s | 1.2x / 2.2x / 2.2x |

| Llama 3.2 1B | Before | After | Change |
| --- | --- | --- | --- |
| FP16 decode | 142 tok/s (85%) | 152 tok/s (91%) | +7.1% |
| INT8 decode | 269 tok/s (81%) | 281 tok/s (84%) | +4.5% |
| INT4 decode | 437 tok/s (68%) | 469 tok/s (73%) | +7.3% |
| TTFT, 1,024-token prompt (FP16) | 167.0 ms | 72.4 ms | 2.3x |
| Serving FP16 / INT8 / INT4 | 1,554 / 937 / 1,002 tok/s | 1,827 / 2,217 / 2,342 tok/s | 1.2x / 2.4x / 2.3x |

Under load, quantized weights were 31% to 40% slower than FP16 through the Phase 5 paths on this
card. They are now 21% to 28% faster. The median time per output token while serving INT4 falls
from 22.4 to 10.0 ms on TinyLlama and from 22.7 to 9.5 ms on Llama 3.2. Perplexity matches the
4070's to within 0.0001: 8.8723, 9.9563, 10.9866 and 14.7617.

Which runs were disturbed, and what each reading gives:

- **TinyLlama INT4 "after".** Its ten iterations read 414 to 509 tok/s by the mean step, with a
  standard deviation of 34.8 tok/s against 8.5 before. As recorded the run is 477 tok/s and the
  pair +5.8%. By the median step it is 494 tok/s and the pair +10.0%.
- **All six Llama 3.2 decode runs of the pair.** Their mean steps are 3.1% to 6.3% longer than
  their medians. As recorded they read 138 to 146, 253 to 266 and 414 to 441 tok/s (+5.5%, +5.3%,
  +6.8%). The table has the median reading.
- **The 1,024-token runs: TinyLlama "after" and both Llama 3.2 runs** (3.2% to 4.5% in their
  decode phase). Their TTFT is as recorded and may be a few percent long. The later Llama 3.2
  `make bench` run is clean and reads 70.3 ms where the pair reads 72.4 ms. TinyLlama has no clean
  1,024-token run; its `make bench` run reads 76.3 ms and is disturbed as well.
- **After the pairs: the TinyLlama `make bench` runs with and without CUDA graphs and the FP16
  `make precision` decode run.** As recorded they read 168, 158 and 170 tok/s. By the median step
  they read 175, 167 and 176.

By the median step, the final-state runs of one configuration (pair, `make bench`,
`make precision`) agree within 1.4% in FP16 and INT8 on both models, and within 2.6% in INT4. By
the mean step the same runs are up to 6.8% apart, which is the disturbance, not the engine.

The three passes of every serving run in the tables are within 1.6% of each other. Of those runs,
the "after" runs with quantized weights are tagged `clock_varied` (TinyLlama INT8 and INT4,
Llama 3.2 INT8): the SM clock moved between 1,815 and 1,950 MHz during them.

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
| Matrix products | 89,789 (198), cuBLAS, tensor pipe 41 to 48% | 88,169 (88), cuBLAS, 47 to 48% | 97,288 (352): expand 7,533 + cuBLAS | 83,330 (88), own, tensor pipe 46% |
| Step | 270,234 (401) | 99,424 (204) | 277,521 (555) | 94,267 (204) |

As in the 4070 tables, the decode row of matrix products includes the vocabulary projection, and
the prefill row leaves out the one GEMV launch that projects the last row onto the vocabulary
(97 to 330 µs here). The step totals include everything.

What the counters show on this card:

- **cuBLAS uses two kernels for the decode products.** Of the 155 products of an unfused step,
  133 go through its GEMV at 360 GB/s. The 22 `down_proj` matrices (2,048 × 5,632, 23.1 MB each)
  go through `cublas_Kernel2`, each followed by a `splitKreduce` launch: 1,562 µs for those 44
  launches, at 326 GB/s, with the tensor pipe 12% active. That is why the unfused FP16 step has
  379 launches for 357 ops. Together cuBLAS streams at 351 GB/s, 85% of the measured 415. On the
  4070 all 155 products went through the GEMV, at 92% of that card's ceiling.
- **kvbound's FP16 GEMV reads at 394 GB/s, 95% of the ceiling.** By matrix: q|k|v 380 GB/s,
  o_proj and down_proj 373, gate|up 411, and the vocabulary projection 417, which is the measured
  ceiling. Product time falls 10.7% within this card's profiles and 5.6% within the 4070's. The
  drop comes from the GEMV and from stacking q|k|v and gate|up, which turns 155 products into 89;
  the counters do not separate the two. cuBLAS started further from the ceiling here.
- **In the profile, the INT4 GEMV does not reach the memory ceiling on any matrix.** Fusion
  raises it from 241 to 273 GB/s (warps resident from 63% to 81%), which is 66% of the ceiling.
  By matrix it reads at 211 GB/s (q|k|v) to 348 GB/s (the vocabulary projection, 84% of the
  ceiling). In the 4070's profile the same kernel reads at 401 GB/s, 85% of its ceiling, and at
  98% on the vocabulary projection. These are profile rates: the profiled INT4 step is 27% longer
  than the recorded one here (see the next items), so the kernel is faster in a recorded step,
  by an amount the records cannot give.
- **Tiled prefill attention: 7,888 to 226 µs per layer (35x), tensor pipe 42%.** The old kernel
  was 64% of the profiled step.
- **Prefill products run the tensor pipe at 46% to 48%**, against the FP16-accumulate rate, as on
  the 4070. cuBLAS picks other kernels here (`sm80_xmma_gemm` and `ampere_fp16_s16816gemm`
  variants). Under the profiler `gemm_quantized` is the faster of the two on this card: 83,330 µs
  for the 88 INT4 products against 88,169 µs for the same 88 in FP16 through cuBLAS. On the 4070
  cuBLAS was the faster one.
- **Profiled time is not recorded time, and the gap differs between the cards.** The profiled
  FP16 decode step is 5.87 ms against 5.63 ms recorded (+4%, and +4% on the 4070). The profiled
  INT4 decode step is 2.57 ms against 2.03 ms (+27%; +16% on the 4070). The profiled 1,024-token
  prefill is 99.4 ms against a recorded TTFT of 76.4 ms (+30%; +15% on the 4070, 44.2 against
  38.6 ms). So a time in a profile is compared below only with other times in the same profile.
  The one exception is the FP16 decode step, where both profiles are within 4% of their records.
- **The clock during profiling is not recorded.** `step_counters` does not set Nsight Compute's
  `--clock-control` option, so the tool's default applies, and its summaries hold no SM clock.
  Whether a profile ran at the clock of the recorded runs is unknown. The counters it collects
  are DRAM bytes, tensor-pipe activity, resident warps and registers per thread: no L2 or
  local-memory counters.

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
| SMs | 38 | 46 | 1.21 |
| SM clock under load, NVML samples of the decode pair runs | about 1,980 MHz | about 2,790 MHz | 1.41 |

The 3 MiB is what the device query in the bandwidth sweep returns; `configs/devices/rtx3060ti.toml`
lists the spec sheet's 4 MB.

What Phase 6 measured, final state, from each card's back-to-back pair. Decode is by the median
step on both cards, so a 4070 figure here can differ by a unit from its summary above, which
prints the mean step. TTFT and serving are as recorded.

| Phase 6 paths | RTX 3060 Ti | RTX 4070 | 4070 / 3060 Ti |
| --- | --- | --- | --- |
| TinyLlama decode, FP16 / INT8 / INT4 | 178 / 314 / 494 tok/s | 212 / 392 / 674 tok/s | 1.19 / 1.25 / 1.37 |
| Llama 3.2 decode, FP16 / INT8 / INT4 | 152 / 281 / 469 tok/s | 183 / 349 / 625 tok/s | 1.20 / 1.24 / 1.33 |
| TinyLlama share of the ceiling | 89% / 79% / 64% | 93% / 86% / 77% | |
| Llama 3.2 share of the ceiling | 91% / 84% / 73% | 96% / 91% / 85% | |
| TinyLlama decode step above its memory floor | 0.63 / 0.68 / 0.73 ms | 0.34 / 0.36 / 0.35 ms | |
| Llama 3.2 decode step above its memory floor | 0.59 / 0.56 / 0.58 ms | 0.24 / 0.25 / 0.24 ms | |
| TTFT, 1,024-token prompt, TinyLlama / Llama 3.2 | 76.4 / 72.4 ms | 38.6 / 39.6 ms | 1.98 / 1.83 |
| TinyLlama serving, FP16 / INT8 / INT4 | 1,740 / 2,163 / 2,219 tok/s | 2,824 / 3,623 / 3,943 tok/s | 1.62 / 1.67 / 1.78 |
| Llama 3.2 serving, FP16 / INT8 / INT4 | 1,827 / 2,217 / 2,342 tok/s | 2,608 / 3,562 / 4,017 tok/s | 1.43 / 1.61 / 1.72 |

Over all of the 3060 Ti's final-state records (the pair, `make bench`, `make precision`) the
decode ratios span 1.19 to 1.21 in FP16, 1.23 to 1.25 in INT8 and 1.32 to 1.37 in INT4. The
Llama 3.2 TTFT ratio is 1.77 with that card's clean `make bench` run. The serving ratios span
1.36 to 1.78 with its `make precision` runs included.

### Memory bandwidth: the floor of a decode step, not the part above it

- **The FP16 products are a memory stream on both cards.** `linear_f16` reads at 394 GB/s here
  and 460 GB/s on the 4070, 95% and 97% of each card's ceiling in its own profile.
- **The records split a step into a floor and a remainder.** The floor is the weights read once
  at the measured bandwidth: 5.00 ms here and 4.37 ms on the 4070 for TinyLlama in FP16, the
  ratio of the bandwidths. The remainder is 0.56 to 0.73 ms here and 0.24 to 0.36 ms on the 4070
  over both models and all three precisions. It is about twice as long on this card, and it does
  not shrink with the weights.
- **So the cards move apart as the weights shrink.** The 4070 is 1.19x to 1.21x faster in FP16,
  where the bandwidths differ by 1.14x, then 1.23x to 1.25x in INT8 and 1.32x to 1.37x in INT4.
  For the same reason the 3060 Ti is further from its own ceiling at every precision, and
  furthest in INT4, where TinyLlama's step is 1.97 to 2.03 ms against a floor of 1.30 ms.
- **What the remainder consists of rests on the profiles, and is less certain.**
  - The 115 launches of a step that are not products take 593 µs in this card's profile and
    388 µs in the 4070's, and each small kernel is 1.42 to 1.58 times as long here. They read
    0.03 to 0.19 MB per launch, far too little for bandwidth to matter. The SM clock is a
    candidate: NVML shows about 1,980 MHz here and 2,790 MHz on the 4070 during the recorded
    decode runs, a ratio of 1.41. It is not established, because the clock during profiling is
    not recorded and the profiler lengthens small kernels on both cards.
  - In its profile the INT4 product kernel reads at 273 of 415 GB/s here (66%), and at 401 of
    475 GB/s (85%) in the 4070's, with about the same resident warps (81% and 84%). That points
    at a kernel further from the memory bound on this card. The profiled INT4 steps are 27% and
    16% longer than the recorded ones, so the records neither confirm the size of that gap nor
    say what limits the kernel.

### Tensor-core throughput: prefill and most of serving

- The 1,024-token prompt takes 1.98x (TinyLlama) and 1.83x (Llama 3.2) as long here, and 1.77x
  for Llama 3.2 with this card's clean run. The FP16 GEMM peak with FP32 accumulation, which the
  engine's products use, differs by 1.81x. TinyLlama's 76.4 ms comes from a disturbed run, so its
  ratio may be a few percent high.
- TinyLlama's prompt is 1.98 TFLOP of matrix products. At each card's Phase 1 peak that is
  60.7 ms of the 76.4 ms here and 33.6 ms of the 38.6 ms on the 4070. For Llama 3.2 it is 61.0 of
  72.4 ms and 33.7 of 39.6 ms. For TinyLlama that leaves 15.6 ms here and 5.1 ms on the 4070 for
  everything else, including any shortfall of the products from their peak.
- Within each card's own profile the tensor pipe is about as busy, against that card's peak: 47
  to 48% here and 48 to 49% on the 4070 in cuBLAS, 46% and 45% in `gemm_quantized`, 42% and 41%
  in prefill attention.
- Serving steps carry up to 30 rows and sit between the two bounds: 1.43x to 1.78x in the pairs.
  The ratio is highest with INT4 weights (1.72x and 1.78x), where the least memory traffic is
  left in a step.

### L2 size: what the counters show, and what they do not

- **Decode does not depend on it.** The weights never fit either L2. One sequence's keys and
  values for a layer are 132 KB at 129 tokens and 1 MiB at 1,024 tokens, which fit both. In each
  card's decode profile attention reads about the same bytes from DRAM over the 22 layers (4.1 MB
  here, 3.6 MB on the 4070) and is about the same share of the non-product kernel time (46% and
  44%).
- **The Phase 3 KV layout benchmark agrees.** For decode attention at batch 1 the ratio between
  the cards falls from 1.59 at 128 cached tokens to 1.32 at 4,096 tokens, where a layer's keys
  and values are 4 MiB. At batch 8 it stays between 1.34 and 1.42 from 2 MiB to 32 MiB per layer.
  Nothing changes where the working set passes 3 MiB, because the kernel reads each key once.
- **Phase 1 isolates L2 on its own.** A chain of dependent loads over a 4 MiB working set costs
  308 ns per load here and 141 ns on the 4070, whose L2 still holds it.
- **In a 1,024-row step the two cards' counters differ in DRAM traffic.** The step's activations
  are 4.2 MB per tensor at the hidden width and up to 23.1 MB (the gate|up output): more than
  3 MiB, less than 36 MiB. Each observation below is within one card's profile.
  - **Norms.** A launch reads 8.2 MB and, on this card, writes 6.8 MB to DRAM while it runs.
    Reads alone are at 216 GB/s, 52% of the ceiling; reads plus writes are at 397 GB/s, 95% of
    it. In the 4070's profile the same kernel writes nothing to DRAM while it runs and reads at
    411 GB/s, 86% of that card's ceiling. Over the whole FP16 step the 3060 Ti writes 1,752 MB to
    DRAM and the 4070 285 MB. Outputs that leave a 3 MiB L2 during the kernel and stay in a
    36 MiB one would look like this; no counter here measures L2 itself.
  - **Reads of the tensor-core products.** The 88 INT4 products through `gemm_quantized` read
    7,114 MB here and 1,032 MB on the 4070 (6.9x). The 88 FP16 products through cuBLAS read
    5,832 MB here and 2,476 MB there (2.4x).
  - **Writes of the tensor-core products.** `gemm_quantized` writes 4,128 MB here and 262 MB on
    the 4070. The outputs of the 88 products are 807 MB. On this card the kernel writes 5.1 times
    its output for each of its three matrix shapes (26.88, 21.53 and 117.73 MB per launch for
    outputs of 5.24, 4.19 and 23.07 MB), where cuBLAS on the same card writes 978 MB, 1.2 times
    the outputs.
  - **The mechanism is not verified.** Input rows that leave L2 and are fetched again would add
    reads, not writes, and they would not single out one kernel. Writes of five times the output
    point at `gemm_quantized` itself: a store pattern, or register spills to local memory, that
    stay inside the 36 MiB L2 on the 4070 and reach DRAM here. `step_counters` collects no L2 or
    local-memory counters, so this log does not claim an L2-size effect for the products.
  - **The cost is not known either.** In this card's profile the traffic of `gemm_quantized`
    adds up to 135 GB/s, 32% of the ceiling, and its tensor pipe is 45.7% active (45.5% in the
    4070's).

### FP8: explains none of it

Only the 4070 has FP8 tensor cores. Phase 1 measured 118.3 TFLOPS for an FP8 E4M3 GEMM there,
and the 3060 Ti (compute capability 8.6) cannot run it. The engine has no FP8 path, since the
FP8 KV cache is not built, so no number in this log depends on it.

### What Phase 6 changed on each card

Each pair switches all six changes at once, so a pair gives the gain of Phase 6 as a whole. Only
the counters can be tied to one change.

- **Decode.** By the median step, Phase 6 as a whole gains +9.5% / +7.1% / +10.0% here and
  +4.7% / +5.5% / +10.1% on the 4070 for TinyLlama in FP16 / INT8 / INT4, and +7.1% / +4.5% /
  +7.3% against +3.1% / +3.7% / +6.8% for Llama 3.2. Within each card's profiles the FP16 product
  time falls 10.7% here and 5.6% on the 4070, from the GEMV and the stacking together; cuBLAS was
  at 85% of the ceiling in this card's profile and 92% in the 4070's.
- **Serving with quantized weights.** Phase 6 as a whole is 2.2x to 2.4x here and 1.8x to 2.4x on
  the 4070. Before, quantized weights were 31% to 40% slower than FP16 under load here and 15% to
  31% there.
- **The 1,024-token prompt.** Phase 6 as a whole is 2.7x and 2.3x here, 3.0x and 2.4x on the
  4070. Within each card's profiles tiled attention takes a layer from 7,888 to 226 µs here (35x)
  and from 3,759 to 116 µs on the 4070 (32x), and the old kernel was 64% and 66% of the step.
  What remains is products, for which this card's peak is 1.81 times lower.

## Not done in Phase 6

- **Profile `gemm_quantized` on the 3060 Ti with L2 and local-memory counters.** At 1,024 rows it
  reads 6.9 times and writes 15.7 times the DRAM bytes it does on the 4070, and writes 5.1 times
  its output. The L2 hit rate and the local-memory traffic would say whether that is the kernel's
  own stores or spills. Timing the same product at 256 rows, whose input is 1 MiB, and at 1,024
  rows outside the profiler would say whether it costs time.
- **Record the SM clock in the step-counter summaries**, or pin it, so that kernel times can be
  compared between cards and with the recorded runs.
- **INT4 decode on the 3060 Ti.** The step is at 64% to 66% of its ceiling there. The entries
  above were developed and their crossovers (`f16_gemv_rows`, `quantized_gemm_rows`) chosen on
  the 4070.
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
