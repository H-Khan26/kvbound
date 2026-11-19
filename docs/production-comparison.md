# kvbound against production stacks (Phase 7)

kvbound was written to see where an inference engine's time and memory go. This document puts it next to stacks that
people deploy, on the same card, with the same token ids, the same requests and the same KV memory, and says for each
measurement whether kvbound is ahead, level or behind, and what in the two engines explains it.

| Stack | Where | What was compared |
| --- | --- | --- |
| llama.cpp `b12503` | RTX 3060 Ti, Windows | Accuracy per bit, one request, serving, kernels under Nsight Systems |
| Qualcomm AI Hub export of Llama 3.2 1B | Snapdragon NPU (Galaxy S25, S26) | Memory design and per-graph latency |

<!-- STACKS -->

The llama.cpp records are one session of 2026-11-19: driver 621.24, kvbound at commit `d360ec1`, the official Windows
CUDA 12 build of the llama.cpp release, 96 records per model under `results/raw/rtx3060ti-fd7b40d5/` (`make
precision serve-load bench`, then `make eval-llamacpp bench-llamacpp serve-llamacpp`). kvbound was measured again in
that session, so its figures here differ a little from the README's tables, which come from earlier sessions on
driver 576.02: FP16 decode on TinyLlama reads 180 tokens/s here and 177 there.

## Summary against llama.cpp

**Ahead.**

- *Decode of one request.* 13% faster with FP16 weights, 21 to 24% with 8-bit weights, and 31 to 33% at 4 bits
  against llama.cpp's closest format (39 to 48% against Q4_K_M, the format most people download). The profiles
  below show a step that is about 0.7 ms longer in llama.cpp at every precision: more and smaller products, more
  bytes per weight at 4 bits, a pass that quantizes the activations, and more waiting between kernels.
- *More requests than the KV memory holds.* With 96 requests at once kvbound serves 1.3 to 1.5 times the tokens per
  second on TinyLlama and 1.9 to 2.1 times on Llama 3.2. It admits by free blocks and runs 30 sequences where
  llama.cpp's slots must be sized for 21, and its step for 30 rows is shorter than llama.cpp's for 21.
- *First token under load.* 9 to 16 ms at the median while requests arrive at 2 to 16 per second, against 25 to 35
  ms.

**Level.**

- *Accuracy at 16 and 8 bits.* From the same FP16 weights the two engines score the same perplexity within 0.05%,
  and each one's 8-bit format costs at most 0.3%.
- *Serving below saturation.* At 2 to 8 requests per second both engines deliver what arrives, and throughput is the
  offered load.
- *Accuracy at 4 bits per weight, with a caveat.* kvbound's best 4.13-bit checkpoint scores between llama.cpp's
  4.63-bit and 4.85-bit files on TinyLlama and better than its 4.50 and 4.94-bit files on Llama 3.2. kvbound's is
  calibrated on text from the same corpus as the test; llama.cpp's files here use no calibration at all.

**Behind.**

- *A long prompt.* llama.cpp evaluates a 1,024-token prompt in 59 to 68 ms where kvbound's first token takes 70 to
  73 ms. kvbound accumulates its products in FP32 because its accuracy contract failed with FP16 accumulation;
  llama.cpp's FP16 products use FP16 accumulation.
- *Accuracy between 4 and 8 bits.* Q4_K_M spends 5.18 bits per weight on Llama 3.2 and scores 11.44 where
  kvbound's best 4-bit checkpoint scores 12.13. kvbound has no format between 4.13 and 8 bits, so it cannot make
  that trade.
- *Everything that is not the forward pass.* kvbound runs two Llama checkpoints and Whisper from token ids. Chat
  templates, samplers beyond greedy, grammars, other architectures, other GPUs and CPUs were never in scope.

## Method

- **Same inputs.** Prompts are the seeded token ids of `kvbound.bench.random_prompt` and `kvbound.workload.trace`,
  sent to llama.cpp as ids. Decoding is greedy with EOS ignored, so every request produces the number of tokens the
  trace asks for.
- **Route.** llama.cpp runs as `llama-server` with all layers on the GPU and flash attention on; requests are
  streamed `POST /completion` calls on 127.0.0.1. kvbound runs `kvb-bench`, one process, on the same traces. Times
  for llama.cpp are taken by the client when each token arrives. The server's own timings are kept in each record:
  for one request its decode rate equals the client's to 0.1 tokens/s, so HTTP is not part of the decode figures;
  for the first token the client's median is 1 to 12 ms above the server's prompt time, and both are shown.
- **Same KV memory.** Serving runs give llama.cpp a unified cache of exactly the bytes of kvbound's pool (8
  sequences of 1,024 FP16 tokens): 8,192 cells as `f16`, or 15,360 cells as `q8_0`, the counterpart of kvbound's
  INT8 cache.
- **Perplexity.** WikiText-2 test, non-overlapping 1,024-token windows, every position after the first scored,
  summed in float64: kvbound's protocol, applied to logits read from `llama.dll`. `llama-perplexity` scores
  differently and is not used.
- **Bits per weight.** The bytes of all weight tensors over the parameter count, scales included. kvbound's `int4_g`
  is 4 bits and one FP16 scale per 128 weights, 4.125 bits, for every matrix including the embedding; with the FP16
  norm gains the file comes to 4.13.
- **Formats.** `Q4_0` is 4 bits with one FP16 scale per 32 weights, rounded to nearest; `llama-quantize` keeps the
  vocabulary matrix at 6.56 bits. `Q4_0_PURE` (`--pure`) quantizes every matrix that way and is the closest
  counterpart of `int4_g`. `Q4_K_M` mixes 4-bit and 6-bit blocks with 6-bit block scales.

What the comparison cannot equalize is listed in
[`llamacpp_serve.py`](../python/kvbound/baselines/llamacpp_serve.py): llama.cpp admits by free slot and fails
requests when its cache overflows, where kvbound admits by free blocks and preempts; llama.cpp samples on the CPU
and streams each token over HTTP, where `kvb-bench` takes the argmax on the GPU inside one process. Those are
differences between a server and a benchmark binary, and the serving section says where they matter.

## Accuracy per bit

WikiText-2 perplexity, FP16 KV cache, RTX 3060 Ti. kvbound's rows are the engine's logits; `int4_g` and
`int4_g.awq` are records of 2026-11-17, the rest of this session.

| Weights | Bits per weight | TinyLlama 1.1B | Bits per weight | Llama 3.2 1B |
| --- | --- | --- | --- | --- |
| kvbound FP16 | 16 | 8.872 | 16 | 10.987 |
| llama.cpp F16 | 16 | 8.873 | 16 | 10.992 |
| kvbound INT8 (`int8_pc`) | 8.01 | 8.873 | 8.01 | 11.015 |
| llama.cpp Q8_0 | 8.50 | 8.875 | 8.50 | 10.997 |
| llama.cpp Q4_K_M | 4.85 | 9.163 | 5.18 | 11.443 |
| llama.cpp Q4_0 | 4.63 | 9.236 | 4.94 | 12.410 |
| llama.cpp Q4_0, every matrix (`--pure`) | 4.50 | 9.285 | 4.50 | 12.570 |
| kvbound INT4, AWQ and GPTQ (`int4_g.awq-gptq`) | 4.13 | 9.197 | 4.13 | 12.130 |
| kvbound INT4, GPTQ (`int4_g.gptq`) | 4.13 | 9.309 | 4.13 | 12.369 |
| kvbound INT4, AWQ (`int4_g.awq`) | 4.13 | 9.396 | 4.13 | 12.912 |
| kvbound INT4, round to nearest (`int4_g`) | 4.13 | 9.956 | 4.13 | 14.762 |

- **The engines agree where the weights are the same.** FP16 differs by 0.003% on TinyLlama and 0.04% on Llama 3.2,
  so differences further down are the formats', not the forward passes'.
- **Rounding to nearest, llama.cpp's format is much better than kvbound's**: 9.29 against 9.96, and 12.57 against
  14.76. Both round each weight to the nearest of 15 or 16 levels; the difference is one scale per 32 weights
  instead of per 128, which costs 0.375 bits per weight. A group's step is set by its largest weight, and a smaller
  group has a smaller largest weight.
- **Calibration closes that gap and more, at fewer bits.** With AWQ's scales and GPTQ's rounding, kvbound's 4.13-bit
  checkpoint scores better than both Q4_0 files on both models. Against Q4_K_M it is 0.03 behind on TinyLlama with
  0.72 fewer bits per weight, and 0.69 behind on Llama 3.2 with 1.05 fewer.
- **The caveat is the calibration text.** kvbound calibrates on 128 windows of the WikiText-2 train split and is
  scored on the test split of the same corpus. The llama.cpp files are quantized without data. Their converter can
  use an importance matrix from calibration text (`llama-imatrix`); that was not run, and kvbound's checkpoints were
  not scored on text from another source. Both would be needed before calling the 4-bit rows a win.

## One request

Decode is the median of 10 runs of 128 generated tokens after a 128-token prompt, one request at a time.

| Decode, tokens/s | TinyLlama: kvbound | llama.cpp | Ratio | Llama 3.2: kvbound | llama.cpp | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 against F16 | 179.9 | 159.9 | 1.13 | 156.3 | 138.2 | 1.13 |
| INT8 against Q8_0 | 315.9 | 261.3 | 1.21 | 283.8 | 229.0 | 1.24 |
| INT4 against Q4_0 `--pure` | 510.0 | 390.7 | 1.31 | 472.4 | 355.7 | 1.33 |
| INT4 against Q4_0 | 510.0 | 381.3 | 1.34 | 472.4 | 337.0 | 1.40 |
| INT4 against Q4_K_M | 510.0 | 367.2 | 1.39 | 472.4 | 319.9 | 1.48 |

kvbound's calibrated checkpoints decode like plain `int4_g`: 512.0 tokens/s with AWQ and GPTQ on TinyLlama and 474.1
on Llama 3.2. Its FP16 decode is 90% and 93% of the card's bandwidth ceiling for the two models; llama.cpp's is 80%
and 83% of the same ceilings.

### Where a decode step goes in each engine

Both engines replay a decode step from a CUDA graph, which hides its kernels from Nsight Systems, so each was traced
with graphs off: 255 or 256 decode steps of TinyLlama after a prompt of one or two tokens (`make profile`, `make
profile-llamacpp`; summaries under `results/profiles/`, the runs of 2026-11-19 from 10:50 UTC on). The first four rows
are GPU time per step by kind of kernel.

| ms per decode step, TinyLlama | kvbound FP16 | llama.cpp F16 | kvbound INT4 | llama.cpp Q4_0 |
| --- | --- | --- | --- | --- |
| Matrix products | 5.06 | 5.32 | 1.50 | 1.78 |
| Quantizing the activations to 8 bits | | | | 0.19 |
| Attention | 0.19 | 0.14 | 0.19 | 0.14 |
| Norms, RoPE, cache write and the rest | 0.16 | 0.22 | 0.16 | 0.21 |
| All kernels of a step | 5.41 | 5.67 | 1.85 | 2.32 |
| Kernel launches per step | 204 | 292 | 204 | 425 |
| Measured step, graphs on, not traced (1 / decode rate) | 5.56 | 6.25 | 1.96 | 2.62 |
| Measured step less its kernels | 0.15 | 0.58 | 0.11 | 0.30 |

The last row subtracts numbers from two different runs, so it is good to about 0.05 ms. The first llama.cpp trace of
the session read its products 20% slower than two repeats that agree to 0.1%; the repeats are used. kvbound's own
repeats moved by about 1%.

llama.cpp's step is 0.69 ms longer with FP16 weights and 0.66 ms longer at 4 bits. It is nearly the same amount at
both precisions, which is why the ratio grows from 1.13 to 1.34 as the step shrinks. The amount is made of different
things in the two cases:

- **FP16: 0.26 ms in the products, 0.43 ms between kernels.** Both engines read the same 2.07 GB per step. kvbound's
  products stream them at 409 GB/s, where the card's measured bandwidth is 415; llama.cpp's at 389 GB/s. kvbound
  stacks q, k and v into one product and gate and up into another at load time, 89 products per step. llama.cpp
  also fuses, running gate, up and the activation between them as one kernel, but multiplies q, k and v separately,
  133 products per step. A likely reason for its lower rate is that k and v are 1 MB each in this model, too small
  for a kernel to reach the full rate. Everything else on the GPU costs the two engines the same 0.35 ms.
- **4 bits: 0.28 ms in the products, 0.19 ms to quantize activations, 0.19 ms between kernels.** Q4_0 reads 599 MB
  per step where `int4_g` reads 533 MB: a scale per 32 weights instead of per 128, and a 6.56-bit vocabulary matrix.
  At kvbound's rate those 65 MB are 0.18 ms. llama.cpp's quantized products are integer dot products, so before
  each one a second kernel quantizes the activation vector to 8 bits: 133 more launches per step. kvbound's kernel
  multiplies the 4-bit weights by the FP16 activations directly.
- **Between kernels.** In a step llama.cpp waits on the stream about 20 times and samples on the CPU from logits it
  copies back. kvbound waits once and takes the argmax on the GPU. Which of these the 0.2 to 0.4 ms are was not
  measured.

What kvbound does better at batch 1 is small and specific: fewer and larger products, no second pass over the
activations, a denser 4-bit format, and one synchronization per step.

### The first token

| Time to first token, ms | TinyLlama: FP16 | 8-bit | 4-bit | Llama 3.2: FP16 | 8-bit | 4-bit |
| --- | --- | --- | --- | --- | --- | --- |
| 128-token prompt, kvbound | 10.4 | 12.1 | 11.3 | 10.8 | 11.9 | 11.1 |
| 128-token prompt, llama.cpp, server's prompt time | 13.4 | 13.6 | 15.3 | 12.9 | 12.7 | 12.2 |
| 128-token prompt, llama.cpp, at the client | 17.9 | 17.7 | 24.9 | 13.7 | 22.0 | 17.5 |
| 1,024-token prompt, kvbound | 72.6 | 71.5 | 71.1 | 69.5 | 70.2 | 69.7 |
| 1,024-token prompt, llama.cpp, server's prompt time | 67.9 | 62.5 | 67.7 | 63.6 | 58.6 | 60.7 |
| 1,024-token prompt, llama.cpp, at the client | 69.9 | 64.1 | 78.8 | 65.7 | 69.8 | 72.3 |

The 4-bit column is `int4_g` against Q4_0, and all values are medians of 10 runs. kvbound's figure is the whole wait
for the first token. llama.cpp's server figure is the prompt evaluation alone, and its client figure the whole wait
including HTTP. The client's medians are noisy on this machine; its fastest run is 0.6 to 1.2 ms above the server's
fastest prompt time in every row. The 1,024-token rows give llama.cpp one batch of 1,024 (`--ubatch-size 1024`), as
kvbound runs it; in batches of 512, the server's default, its prompt times are 68 to 71 ms on TinyLlama and 61 to 66
ms on Llama 3.2.

For a short prompt kvbound is 1 to 4 ms ahead. For a long one llama.cpp is ahead, by 4 to 6 ms with FP16 weights and
by up to 12 ms with 8-bit ones. Reading its source at the tag gives a likely reason, which this project did not
profile: llama.cpp's wide FP16 products go to cuBLAS with FP16 accumulation
(`batched_mul_mat_traits<GGML_TYPE_F16>` in `ggml-cuda.cu`), and its quantized ones to its own integer kernels on
activations quantized to 8 bits (`mmq.cu`). kvbound accumulates every wide product in FP32. On this card the tensor
cores deliver 58 TFLOPS with FP16 accumulation and 33 with FP32 (`make microbench`), and kvbound chose FP32 because
FP16 accumulation failed its per-site error limits against the reference model.

## Serving

Both engines get the same pool bytes and the same trace. `longtail` is 48 requests with long-tailed prompt and output
lengths arriving as a Poisson process; `fixed` is 96 requests of 256 prompt tokens and 128 output tokens, all at
time zero. No request failed in any run.

### Requests arriving over time

| `longtail`, FP16 KV | 2 per s | 4 per s | 8 per s | 16 per s |
| --- | --- | --- | --- | --- |
| **TinyLlama, tokens/s** | | | | |
| kvbound FP16 | 182 | 348 | 613 | 942 |
| llama.cpp F16 | 181 | 343 | 599 | 877 |
| kvbound INT4 | 186 | 365 | 674 | 1,143 |
| llama.cpp Q4_K_M | 185 | 362 | 672 | 1,136 |
| llama.cpp Q4_0 | 185 | 362 | 678 | 1,109 |
| **Llama 3.2, tokens/s** | | | | |
| kvbound FP16 | 181 | 343 | 598 | 896 |
| llama.cpp F16 | 180 | 334 | 567 | 743 |
| kvbound INT4 | 186 | 363 | 670 | 1,140 |
| llama.cpp Q4_K_M | 185 | 359 | 652 | 1,010 |
| llama.cpp Q4_0 | 185 | 360 | 655 | 979 |
| **TinyLlama, median time to first token, ms** | | | | |
| kvbound FP16 / INT4 | 13.5 / 10.6 | 12.2 / 9.2 | 13.0 / 9.9 | 14.7 / 11.8 |
| llama.cpp F16 / Q4_0 | 31.3 / 28.5 | 28.6 / 27.2 | 29.3 / 28.2 | 33.1 / 27.0 |
| **Llama 3.2, median time to first token, ms** | | | | |
| kvbound FP16 / INT4 | 13.8 / 9.8 | 13.1 / 10.2 | 14.2 / 10.5 | 15.7 / 11.6 |
| llama.cpp F16 / Q4_0 | 33.2 / 26.0 | 33.3 / 25.8 | 33.6 / 28.6 | 33.9 / 27.6 |

Up to 8 requests per second the trace sets the throughput: both engines finish requests about as fast as they
arrive, and the difference shows in latency instead. kvbound's median time per output token with FP16 weights is 5.9
to 6.6 ms on TinyLlama against 7.0 to 7.7 ms, the batch-1 decode gap carried into small batches. At 16 per second
llama.cpp falls behind on throughput: by 7% (FP16) and 1 to 3% (4-bit) on TinyLlama, and by 17% and 11 to 14% on
Llama 3.2.

The first token takes llama.cpp 25 to 34 ms at every rate, two to three times as long as kvbound, although the
median prompt here is 64 tokens and the same server evaluates a 128-token prompt in 12 to 15 ms. A request that
arrives while a step is running waits for it in both engines. Where the rest goes was not measured; the HTTP path
is part of it, and `kvb-bench` has none.

### More requests than the memory holds

| `fixed`, 96 requests at once | TinyLlama: tokens/s | Sequences running | Median ms per token | Llama 3.2: tokens/s | Sequences running | Median ms per token |
| --- | --- | --- | --- | --- | --- | --- |
| kvbound FP16, FP16 KV | 1,819 | 30 | 12.4 | 1,917 | 30 | 11.8 |
| llama.cpp F16, `f16` KV | 1,340 | 21 | 14.6 | 1,026 | 21 | 19.3 |
| kvbound FP16, INT8 KV | 2,182 | 58 | 19.0 | 2,230 | 58 | 18.3 |
| llama.cpp F16, `q8_0` KV | 1,439 | 40 | 25.0 | 1,078 | 40 | 33.7 |
| kvbound INT4, FP16 KV | 2,334 | 30 | 9.5 | 2,461 | 30 | 9.0 |
| llama.cpp Q4_K_M, `f16` KV | 1,817 | 21 | 10.8 | 1,284 | 21 | 15.6 |
| llama.cpp Q4_0, `f16` KV | 1,660 | 21 | 11.8 | 1,225 | 21 | 16.3 |
| kvbound INT4 (AWQ, GPTQ), INT8 KV | 2,440 | 58 | 17.9 | 2,244 | 58 | 19.3 |
| llama.cpp Q4_0, `q8_0` KV | 1,654 | 40 | 22.4 | 1,190 | 40 | 30.9 |

Two things separate the rows, and the table shows both.

**Admission.** A request here ends at 384 tokens. llama.cpp never preempts and fails every request in flight if its
cache overflows, so its slots have to be few enough that all of them can finish: 8,192 cells over 384 is 21. kvbound
admits a request when its prompt's blocks are free, keeps one spare block per growing sequence, and when the pool
runs out preempts the newest sequence and computes it again later. It starts 30 sequences in the same bytes. With an
8-bit cache the same arithmetic gives llama.cpp 40 slots and kvbound 58.

**The step itself.** kvbound's step for 30 rows is shorter than llama.cpp's for 21 at every precision, and the gap
is wider on Llama 3.2 than on TinyLlama: 11.8 against 19.3 ms with FP16 weights, where TinyLlama's are 12.4 against
14.6. The 8-bit cache rows show what a row costs each engine. Going from 21 to 40 slots lengthens llama.cpp's FP16
step by 0.55 ms per added row on TinyLlama and 0.76 ms on Llama 3.2; going from 30 to 58 sequences lengthens
kvbound's by 0.24 and 0.23 ms.

Most of llama.cpp's cost per row is outside its engine. A diagnostic run, not a record, timed the engine alone on
21 parallel sequences of this shape with the release's `llama-batched-bench`, which calls the same decode without a
server, a sampler or HTTP (`-npp 256 -ntg 128 -npl 1,21 -c 8192 -b 1024 -ub 1024 -ngl all -fa on`):

| Step for 21 rows, ms | TinyLlama F16 | TinyLlama Q4_0 | Llama 3.2 F16 | Llama 3.2 Q4_0 |
| --- | --- | --- | --- | --- |
| llama.cpp engine alone (`llama-batched-bench`) | 10.2 | 7.3 | 11.3 | 8.2 |
| llama-server, median time per token (table above) | 14.6 | 11.8 | 19.3 | 16.3 |
| kvbound, 30 rows, median time per token | 12.4 | 9.5 | 11.8 | 9.0 |

The server adds 4.4 to 4.5 ms to a 21-row step on TinyLlama and 8.0 to 8.1 ms on Llama 3.2, the same at both weight
precisions. Llama 3.2's vocabulary is four times TinyLlama's (128,256 against 32,000), and the server samples every
row's token on the CPU from its full logits, so work that grows with rows times vocabulary fits the pattern; which
part of the server it is was not profiled. kvbound takes the argmax on the GPU and copies one token id per row.
Engine against engine the two are close: llama.cpp steps 21 rows in 10.2 ms where kvbound steps 30 in 12.4. The
serving gap in the table is therefore admission plus the server's work per row, not faster kernels.

That cost per row decides what an 8-bit cache is worth. It nearly doubles the sequences in a step for both engines.
With FP16 weights that raises kvbound's throughput by 20% and 16% on the two models and llama.cpp's by 7% and 5%.
With 4-bit weights, whose steps are shorter to begin with, it changes either engine's throughput by less than 10% in
either direction.

## A static design on a phone NPU

The second comparison is a different kind of system: Qualcomm's stock export of Llama 3.2 1B Instruct with 4-bit
weights, profiled on the Hexagon NPU of a Galaxy S25 and a Galaxy S26 through Qualcomm AI Hub
([`qai_hub.py`](../python/kvbound/baselines/qai_hub.py), summaries under `results/profiles/`). It is the production
form of the static memory design that kvbound's Phase 3 built, and the design document compares the two in detail
([A production static design](static-vs-paged-memory.md#a-production-static-design-llama-32-1b-on-a-snapdragon-npu)).
In short:

- From per-graph latencies, decode works out to 58.2 tokens/s on the S25 and 59.9 on the S26, and a 128-token prompt
  pass to 31 and 27 ms. These are graph latencies, not end-to-end runs: the host runtime's work between graphs is
  not in them.
- Every shape is compiled in: a context of 4,096 positions, one graph for 128 prompt tokens and one for a single
  token, one sequence at a time. Each step attends over all 4,096 positions under a mask, so attention costs the
  same at position 10 as at position 4,000, and `softmax` and attention's `matmul` are the costliest operators in the
  profiles.
- kvbound's decode step is the opposite: 90% of it is weights streaming through the products, and attention grows
  with the sequence's real length.

The same checkpoint decodes at 472 tokens/s on the RTX 3060 Ti with kvbound's INT4 weights. The two numbers are not
a comparison of engines: one is a phone NPU under a power budget of a few watts running one sequence, the other a
200 W desktop card. What carries over is the design lesson, that a compiler which needs every shape fixed pays for
the full context on every step.

## What was not done

- **llama.cpp with an importance matrix**, and kvbound's calibrated checkpoints scored on text from another corpus.
  Until both exist, the 4-bit accuracy rows compare a calibrated format on in-domain text with uncalibrated ones.
- **A profile of llama.cpp's prompt processing.** The explanation of the long-prompt gap is from its source.
- **A breakdown of llama.cpp's first-token time under load**, and of its server's per-step work beyond the
  diagnostic above.
- **Locked clocks.** Every record is from Windows, where the GPU clock cannot be locked; records whose clock moved
  are tagged `clock_varied`.
- **The same comparison on the RTX 4070**, and llama.cpp built from source with other options than the release's.
<!-- NOTDONE -->
