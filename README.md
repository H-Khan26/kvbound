# kvbound

A from-scratch LLM inference engine in C++ and CUDA. It implements two KV-cache designs for the same engine, a static
preallocated one (how phone NPU runtimes work) and a paged one with continuous batching (how GPU serving systems
work), then quantizes and profiles both against the roofline of an RTX 3060 Ti (Ampere) and an RTX 4070 (Ada).

**Status:** Phases 0 to 6 are done on the RTX 3060 Ti and the RTX 4070, measured on TinyLlama 1.1B and Llama 3.2 1B:
reference model, golden tensors, hardware ceilings, the analytical performance model, an FP16 engine that generates the
same tokens as the reference model, the static memory design, the paged KV cache with continuous batching,
quantization (INT8 and INT4 weights, INT4 with activation-aware scales, an INT8 KV cache), and profiling and
optimization. Phase 6 is summarized next for both cards; its log, which also compares the two cards, is
[`docs/optimization-log.md`](docs/optimization-log.md). The sections after it describe the engine on the RTX 3060 Ti
as Phase 6 left it.

Track W runs a speech model, Whisper, on the same engine: an encoder and a decoder program over one weight table, a
second cache for the encoder's output, greedy decoding and beam search. It is measured on Quran recitation against
Hugging Face Transformers and faster-whisper ([Track W](#track-w-whisper-on-the-same-engine)).

Open items: beam search ranks each step's candidates on the host, the likely main reason it is slow, and ranking them
on the GPU is in progress; timestamps, word alignment and decoding of audio longer than one 30-second window are not
built, and the Sawt Sakinah app, which transcribes with faster-whisper today, needs all three before it could switch;
on the RTX 3060 Ti
the DRAM writes of `gemm_quantized` are not explained; the step-counter profiles do not record the SM clock.

## Phase 6: profile and optimize

Six changes, each an engine option so that its before and after ran back to back at one commit (the Phase 5 paths are
`--no-fuse --f16-gemv-rows 0 --no-prefill-attention --expand-quantized`):

- **A fused decode step** (`fuse`): q|k|v and gate|up stacked into one product each at load time, residual adds folded
  into the norm after them, RoPE folded into the KV-cache write. 9 ops per layer instead of 16.
- **kvbound's own FP16 GEMV** (`f16_gemv_rows`, 4) instead of cuBLAS for decode-shaped steps. On the RTX 4070 it
  streams matrices at 436 to 477 GB/s of the measured 475; on the RTX 3060 Ti a decode step's matrices at 394 GB/s of
  the measured 415.
- **Streaming loads** in the quantized GEMV and a **register-resident RMSNorm** (no switch).
- **Quantized weights on tensor cores** (`quantized_gemm_rows`, 3; `expand_quantized` for the old path): a W8A16/W4A16
  `mma.sync` product that expands weights in registers and reads each matrix once at its packed size.
- **Prefill attention on tensor cores** (`prefill_attention`): flash attention over tiles of query rows, through the
  paged block table, FP16 or INT8 cache. 3.8 ms to 0.11 ms per layer at 1,024 tokens on the RTX 4070, 7.9 ms to
  0.23 ms on the RTX 3060 Ti.

| RTX 4070 | Phase 5 paths | Phase 6 |
| --- | --- | --- |
| TinyLlama decode, batch 1, FP16 / INT8 / INT4 | 202 / 370 / 610 tok/s | **212 / 391 / 673 tok/s** (92% / 86% / 76% of the ceiling) |
| Llama 3.2 decode, batch 1, FP16 / INT8 / INT4 | 178 / 336 / 585 tok/s | **183 / 348 / 624 tok/s** (96% / 91% / 85%) |
| TinyLlama / Llama 3.2 time to first token, 1,024-token prompt | 115.8 / 94.7 ms | **38.6 / 39.6 ms** |
| TinyLlama serving 96 requests, FP16 / INT8 / INT4 | 2,320 / 1,979 / 1,797 tok/s | **2,824 / 3,623 / 3,943 tok/s** |
| Llama 3.2 serving 96 requests, FP16 / INT8 / INT4 | 2,298 / 1,592 / 1,657 tok/s | **2,608 / 3,562 / 4,017 tok/s** |

| RTX 3060 Ti | Phase 5 paths | Phase 6 |
| --- | --- | --- |
| TinyLlama decode, batch 1, FP16 / INT8 / INT4 | 162 / 291 / 450 tok/s | **177 / 312 / 507 tok/s** (89% / 78% / 66% of the ceiling) |
| Llama 3.2 decode, batch 1, FP16 / INT8 / INT4 | 145 / 270 / 437 tok/s | **154 / 281 / 475 tok/s** (92% / 84% / 74%) |
| TinyLlama / Llama 3.2 time to first token, 1,024-token prompt | 201.6 / 163.4 ms | **73.3 / 70.6 ms** |
| TinyLlama serving 96 requests, FP16 / INT8 / INT4 | 1,517 / 1,013 / 1,060 tok/s | **1,804 / 2,214 / 2,290 tok/s** |
| Llama 3.2 serving 96 requests, FP16 / INT8 / INT4 | 1,623 / 980 / 1,039 tok/s | **1,901 / 2,279 / 2,403 tok/s** |

Quantized weights went from 15% to 31% slower than FP16 under load to 28% to 54% faster on the RTX 4070, and from 30%
to 40% slower to 20% to 27% faster on the RTX 3060 Ti. WikiText-2 perplexity is unchanged (8.8723 and 9.9563 for
TinyLlama FP16 and INT4, 10.9867 and 14.7617 for Llama 3.2; the RTX 3060 Ti reads the same to within 0.0001), and both
models still match the reference token for token. On the RTX 4070 the 1,024-token prompt is now 95% matrix products,
at the 59 TFLOPS FP16 peak with FP32 accumulation. On the RTX 3060 Ti that peak is 32.7 TFLOPS, which accounts for
60.7 of TinyLlama's 73.3 ms. Per-kernel Nsight Compute counters of one step (`make step-counters`) are in
`results/profiles/*-counters-*`.

Each table is one set of back-to-back runs at one commit. The RTX 3060 Ti shares its desktop. In its first Phase 6
session (2026-11-17, `c442759`) other programs took the GPU during some steps of a run, and in 14 of the 34 decode
runs the mean step was 3.1% to 6.3% longer than the median step. Its pairs were recorded again on 2026-11-18 at
`4089b55` with the desktop kept quiet, and the table above is that session: in its 22 decode runs the two readings
differ by at most 0.87%, the three passes of each serving run are within 0.8% of each other, and no decode run is
tagged `clock_varied`. Decode on this card is read by the median step, in this table and in the sections below,
which also use the first session's `make precision`, `make serve` and `make serve-load` records. In this table the
reading moves no cell by more than one unit except the INT4 "after" runs, which read 503 (TinyLlama) and 471 tok/s
(Llama 3.2) as recorded. On the RTX 4070 the two readings agree within 0.5%, and its table is as recorded. Time to
first token and serving have no second reading.

**The two cards.** The RTX 4070 has 1.14 times the memory bandwidth of the RTX 3060 Ti (475 against 415 GB/s), 1.81
times its tensor-core peak for FP16 products with FP32 accumulation (59.1 against 32.7 TFLOPS), and a 36 MiB L2
against 3 MiB.

- **Memory bandwidth** sets the floor of a decode step, the weights read once: 5.00 ms on the 3060 Ti and 4.37 ms on
  the 4070 for TinyLlama in FP16, where the FP16 GEMV reads at 95% and 97% of the two ceilings. The rest of the step
  is 0.51 to 0.70 ms on the 3060 Ti and 0.24 to 0.36 ms on the 4070 at every weight precision, so the 4070 is 1.19
  to 1.20 times faster in FP16 and 1.32 to 1.33 times in INT4. Why the rest is twice as long is not established: the
  SM clock (1.42 times higher on the 4070) is a candidate.
- **Tensor-core throughput** sets the 1,024-token prompt (1.78 to 1.90 times faster on the 4070) and most of serving
  (1.37 to 1.72 times).
- **L2 size** does not show in decode. In a 1,024-row step the two cards' counters differ in DRAM traffic: the INT4
  tensor-core products read 6.9 times and write 15.7 times the bytes on the 3060 Ti that they do on the 4070. The
  writes are 5.1 times the products' output, which points at kvbound's own kernel and not only at cache size. The
  mechanism has not been profiled, and no record says what the traffic costs.
- **FP8** tensor cores exist only on the 4070. No FP8 path is built, so they explain none of this.

The evidence for each line is in the [log](docs/optimization-log.md#the-two-cards-compared).

## Where things stand (RTX 3060 Ti, TinyLlama 1.1B)

| Quantity | Value | Source |
| --- | --- | --- |
| Memory bandwidth beyond cache, best 2 ms burst | about 415 GB/s (93% of the 448 GB/s spec) | `kvb-bandwidth_sweep` |
| Memory bandwidth beyond cache, sustained 0.1 s | about 367 GB/s; other desktop programs take turns on the GPU | `kvb-bandwidth_sweep` |
| Dependent-load latency, L1 / L2 / DRAM | about 21 / 135 / 312 ns | `kvb-pointer_chase` |
| GEMM peak, FP16 with FP16 / FP32 accumulate; INT8 | about 58 / 33 TFLOPS; 115 TOPS | `kvb-gemm_peak` (cuBLASLt) |
| Decode ceiling, batch 1, FP16 | about 200 tokens/s (weights 2.07 GB read once per token) | `python -m kvbound.perfmodel` |
| **kvbound decode, batch 1, FP16** | **177 tokens/s, 88% of the ceiling** (5.66 ms per token; 170 tokens/s without CUDA graphs) | `python -m kvbound.bench` (`make bench`) |
| kvbound time to first token | 11 ms for a 128-token prompt, 74 ms for 1,024 tokens | `python -m kvbound.bench` (`make bench`) |
| **kvbound decode, batch 1, INT8 / INT4 weights** | **314 / 507 tokens/s**, 79% / 66% of their ceilings | `python -m kvbound.bench --weights ...` (`make precision`) |
| kvbound static batching, 8 slots | 955 tokens/s on equal-length requests, 451 on long-tailed lengths (173 to 175 one at a time) | `python -m kvbound.serve` |
| **kvbound paged KV, continuous batching, same pool** | **1776 tokens/s on equal-length requests, 1469 on long-tailed lengths** | `python -m kvbound.serve --policy continuous` |
| Hugging Face Transformers decode, batch 1 | 17 to 33 tokens/s, depending on where the unlocked GPU clock settles | `python -m kvbound.baselines.hf` |

Greedy output matches the reference model token for token on every golden prompt, and every intermediate activation
stays within the FP16 limits in the golden manifest (`make test-real`).

Where an FP16 decode step's kernel time goes (Nsight Compute counters of one step, `results/profiles/*-counters-*`):
90% in kvbound's GEMV streaming the weights, 5% in attention, 5% in the small kernels (norms, RoPE with the cache
write, SwiGLU). The step is 204 kernel launches. Replayed from a CUDA graph it takes 5.66 ms; launched one by one,
5.88 ms.

These runs come from a shared Windows desktop with unlocked GPU clocks. The engine's decode, time-to-first-token and
serving numbers in this section and the ones below are from the Phase 6 records of this card: `make bench` from the
second session (2026-11-18, `4089b55`), `make precision`, `make serve` and `make serve-load` from the first
(2026-11-17, `c442759`). The kernel microbenchmarks keep the records of the phase that made them. Decode is read by
the median step of each run, as explained under the Phase 6 tables. By that reading FP16 decode is 177 tokens/s in
`make bench`, 176 in `make precision` and 177 in the second session's pair, 0.7% apart. Through the Phase 5 paths
the same request reads 162 tokens/s.

### What the static memory design costs

The static design preallocates the whole KV cache (8 slots of 1,024 tokens, 176 MiB) and gives every admitted sequence
a full slot until its batch ends.

| Workload (48 requests) | Tokens/s | Reserved KV memory holding tokens | Slots busy |
| --- | --- | --- | --- |
| `fixed`: 256-token prompts, 128-token outputs | 955 | 30% | 99% |
| `uniform`: prompts 32 to 512, outputs 16 to 256 | 723 | 33% | 71% |
| `longtail`: lognormal, median prompt 64 and output 48 | 451 | 15% | 38% |

![KV memory under static batching](reports/rtx3060ti/static_memory.png)

Two more results from this phase:

- **KV layout.** Three orderings of the cache's dimensions were benchmarked with the engine's own kernels. Between
  the two layouts that keep a key's dimensions adjacent, which dimension is outermost changes attention time by 2% at
  batch 1 and 11% at batch 8. What matters more is how many memory requests a warp issues: reading each 128-byte key
  as eight 16-byte loads instead of 64 two-byte loads is 2.9 times faster (3.4 at batch 8), and only those two layouts
  allow it. Figure: `reports/rtx3060ti/kv_layout.png`. Hardware counters (Nsight Compute, `make counters`) match the
  counts derived from the code: for one layer at batch 1 with 4,096 cached tokens, the 2-byte kernel asks the L1 cache
  for 8.9 million sectors and the 16-byte kernel for 1.6 million, while both read the same 4.2 MB from DRAM, the
  layer's keys and values exactly once. Attention moves roughly 17 to 50 GB/s, far below the memory bandwidth, so the wide
  loads help by cutting the traffic to the L1 cache, not the traffic to DRAM.
- **Activation memory.** The fused forward pass has 203 intermediate tensors. Side by side they would need 1,369 MiB
  for 1,024-token steps; placed by lifetime in one arena they share 37 MiB, which is also the lower bound. Figure:
  `reports/rtx3060ti/arena.png`.

### Paged KV cache and continuous batching

The paged design cuts the same pool into 16-token blocks. A request is admitted when the pool holds the blocks its
known tokens need plus one spare block for every running sequence that can still grow; a sequence gets another block
when it grows into one; and when the pool runs dry the most recently admitted sequence is preempted and recomputed
from its first token once readmitted. Requests join and leave between any two steps.

Same 176 MiB pool, same 48 requests queued at once. The middle setup runs the continuous scheduler over the 8 whole
slots, which separates what the scheduler buys from what paging buys.

| Workload | Setup | Tokens/s | Most sequences in a step | TTFT median | TPOT median | Reserved KV holding tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `fixed` | static batches, 8 slots | 955 | 8 | 2.79 s | 7.5 ms | 30% |
| `fixed` | continuous, 8 whole slots | 966 | 8 | 2.75 s | 7.9 ms | 31% |
| `fixed` | continuous, 16-token blocks | 1776 | 30 | 0.51 s | 13.0 ms | 89% |
| `uniform` | static batches, 8 slots | 723 | 8 | 4.70 s | 7.3 ms | 33% |
| `uniform` | continuous, 8 whole slots | 874 | 8 | 3.75 s | 8.2 ms | 33% |
| `uniform` | continuous, 16-token blocks | 1593 | 28 | 0.51 s | 11.7 ms | 92% |
| `longtail` | static batches, 8 slots | 451 | 8 | 4.15 s | 6.5 ms | 15% |
| `longtail` | continuous, 8 whole slots | 947 | 8 | 1.25 s | 7.1 ms | 19% |
| `longtail` | continuous, 16-token blocks | 1469 | 48 | 0.18 s | 12.0 ms | 89% |

![One KV pool served three ways](reports/rtx3060ti/serving.png)

![KV memory, whole slots against blocks](reports/rtx3060ti/paged_memory.png)

- **What it costs.** Steps carry more rows, so each token takes longer to arrive: the median time per output token on
  `fixed` rises from 7.5 to 13.0 ms. Admission does not know how long a response will get, so a full pool
  preempts: 9 preemptions and 2,769 tokens processed twice per pass on `fixed`, 5 and 1,684 on `uniform`, none on
  `longtail`, whose 48 requests all fit at once.
- **The block table.** Attention finds each cached position's block once per element it reads. Doing that with a
  division made attention 1.37 to 1.84 times slower than one block per sequence; a shift and a mask bring it to
  1.07 to 1.12 times, so a paged cache needs a power-of-two block size. Whether a sequence's blocks are in
  order or scattered over the pool makes no difference (`python -m kvbound.microbench kv_layout`).
- **Arrivals.** With `longtail` requests arriving as a Poisson process at 8 per second, static batching has a median
  time to first token of 1.9 s and the paged engine 13 ms; at 16 per second, 3.3 s and 15 ms
  (`make serve-load`, figure `reports/rtx3060ti/serving_load.png`).
- **CUDA graphs.** A graph replays one row count and continuous batching keeps changing it, so the engine keeps the 16
  most recently used. On `longtail` that is 41 captures and 55 ms per pass with 99% of steps replayed. In Phase 4,
  room for 64 graphs removed the captures and gave 1386 tokens/s against 1339; that comparison has not been run on
  the Phase 6 engine.

### Quantization

Weights are quantized by round-to-nearest with FP16 scales: INT8 per output channel, or INT4 in groups of 128 along
the input dimension, two values to a byte. The format is written into the checkpoint's safetensors metadata
(`python -m kvbound.quantize`) and read by the C++ loader and by the reference model, which dequantizes the same
file. The KV cache can hold INT8 with one FP16 scale per cached vector, quantized when written and scaled inside the
attention kernel. Perplexity is WikiText-2, the whole test split in 1,024-token windows, from the engine's logits.
The last two columns serve 96 `fixed` requests at once in a KV pool of the same bytes.

| TinyLlama 1.1B | Perplexity | Weights on device | KV bytes per token | Decode tokens/s | Of its ceiling | Sequences at once | Tokens/s under load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 weights, FP16 KV | 8.87 | 2,098 MiB | 22,528 | 176 | 88% | 30 | 1746 |
| FP16 weights, INT8 KV | 8.87 | 2,098 MiB | 11,616 | 177 | 88% | 58 | 2168 |
| INT8 weights, FP16 KV | 8.87 | 1,050 MiB | 22,528 | 314 | 79% | 30 | 2270 |
| INT8 weights, INT8 KV | 8.87 | 1,050 MiB | 11,616 | 313 | 78% | 58 | 2370 |
| INT4 weights, FP16 KV | 9.96 | 541 MiB | 22,528 | 507 | 66% | 30 | 2356 |
| INT4 weights, INT8 KV | 9.96 | 541 MiB | 11,616 | 499 | 65% | 58 | 2440 |

![Precision](reports/rtx3060ti/precision.png)

The table is the first session's `make precision` (2026-11-17), with decode by the median step. The figure
(`make report`) plots the newest record of each configuration, by the mean step. For the FP16-cache bars of decode
and serving those are the second session's runs (2026-11-18): 177, 313 and 503 tokens/s, and 1804, 2214 and 2290
tokens/s under load. The INT8-cache bars and the perplexities are the table's records, whose decode reads 177, 314
and 496 tokens/s by the mean step. The figure therefore sets bars from two sessions side by side; the table keeps to
one.

Read against the roofline:

- **Decode speed follows the bytes.** A decode step is the weights crossing the memory bus once plus 0.7 ms of
  everything else, at every weight precision. The floor set by the weights shrinks from 5.0 to 2.5 to 1.3 ms and the
  measured step from 5.7 to 3.2 to 2.0 ms. The share of the ceiling falls because the rest does not shrink.
- **INT8 weights are free in accuracy. INT4 is not, and activation-aware scales recover half of what it loses.**
  Round-to-nearest INT4 raises perplexity by 12% on TinyLlama and 34% on Llama 3.2. With AWQ scales it is 6% and
  18% above FP16; the numbers and the method follow this list.
- **An INT8 cache buys concurrency, not decode speed.** The same pool holds 992 blocks instead of 512, so 58
  sequences run at once instead of 30. Under load that is 24% more tokens/s with FP16 weights, and each token takes
  longer to arrive: the median time per output token rises from 12.9 to 19.2 ms. With INT8 or INT4 weights it makes
  no measurable difference in these records: the INT8-cache runs read 4.4% and 3.6% higher, and all four of those
  runs are tagged `clock_varied`. Batch-1 decode does not change: 176 and 177 tokens/s with FP16 weights. Attention
  over an INT8 cache is 10% faster at batch 8, but attention is a few percent of a step and was never bound by
  bandwidth.
- **Under load, quantized weights are now faster than FP16**: 2270 and 2356 tokens/s with INT8 and INT4 weights
  against 1746, 30% and 35% more. Steps wider than 3 rows multiply the packed weights on tensor cores and read each
  matrix once at its packed size. Phase 5 found the opposite, because wide steps expanded each matrix to FP16 for
  cuBLAS: through those paths, in the pair of the same session, the same workload runs at 979 and 1,013 tokens/s
  against 1,463 in FP16.
- The engine and the dequantizing reference agree on every capture site within the FP16 limits (`make test-gpu`); the
  reference model scores 8.87 on the FP16 checkpoint where the engine scores 8.87.

**INT4 with activation-aware scales (AWQ).** Rounding treats every weight alike, but a weight's error reaches the
output multiplied by the activation it meets. AWQ (Lin et al., 2023) rescales the input channels of a matrix before
rounding it, so that channels with large activations get a finer step, and divides the activations by the same
scales (`python -m kvbound.quantize <model> --scheme int4_g --awq`, `python/kvbound/awq.py`):

- **Scales.** Each layer has three: one for the inputs of q, k and v, folded into the weight of the RMSNorm before
  them; one for the inputs of gate and up, folded into the RMSNorm before those; one for the input of down_proj,
  folded into the rows of up_proj. A scale is the mean activation magnitude of its channel raised to a power, and
  the power is the one of 20 that least changes the output of the attention or MLP it feeds. o_proj, the embedding
  and the vocabulary matrix are not scaled.
- **Clipping.** Each group of 128 weights in v, o_proj, gate, up and down_proj then gets the clip range, of 10
  tried, that changes that matrix's output least on a sample of the calibration activations.
- **Calibration.** 128 windows of 512 tokens from the WikiText-2 train split, at the same pinned dataset revision as
  the test split that perplexity is measured on. Calibration is therefore in-domain for this metric, and text from
  elsewhere may gain less.
- **Format.** The scales are folded into the weights before quantization, so the result is an ordinary `int4_g`
  checkpoint, and neither the engine nor its kernels change.

| WikiText-2 perplexity, FP16 KV (RTX 3060 Ti) | TinyLlama 1.1B | Llama 3.2 1B |
| --- | --- | --- |
| FP16 weights | 8.87 | 10.99 |
| INT8 weights | 8.87 | 11.01 |
| INT4, round to nearest (`int4_g`) | 9.96 | 14.76 |
| INT4, clip chosen by weight error (`int4_g.mse`) | 10.84 | 14.63 |
| INT4, AWQ scales without the clip search (`int4_g.awq-noclip`) | 9.60 | 13.47 |
| INT4, AWQ scales and clip (`int4_g.awq`) | 9.40 | 12.91 |

AWQ recovers 52% of the gap between INT4 and FP16 on TinyLlama and 49% on Llama 3.2; the scales alone recover 33%
and 34%, and the clip search the rest. Clipping by weight error alone, the earlier attempt, makes TinyLlama worse
and barely moves Llama 3.2: the clip has to be chosen by its effect on the output. INT4 with AWQ still costs 6% in
perplexity over FP16 on TinyLlama and 18% on Llama 3.2. Its decode and serving speed have not been recorded: the
speed columns of the precision table are plain INT4.

Known gaps after Phase 6: INT4 decode on this card stops at 66% of its ceiling; INT4 with AWQ scales still
costs the perplexity above and has no speed record; FP16 steps of 5 to 64 rows still go through cuBLAS; decode
attention still runs one block per head at long context; steps that mix prefill and decode run without CUDA graphs;
prefix sharing is not built, and the FP8 KV cache is not built. The largest Phase 5 gap is closed: a 1,024-token
prompt takes 73 ms where the Phase 5 paths take 202 ms.

### The same engine on Llama 3.2 1B

Llama 3.2 1B (16 layers, 8 KV heads, tied embeddings, Llama 3 RoPE scaling) runs on the same code with no changes.
Greedy output matches the reference token for token on every golden prompt, and the worst intermediate activation is at
18% of its FP16 limit, as on TinyLlama.

| Quantity (RTX 3060 Ti) | Llama 3.2 1B | TinyLlama 1.1B |
| --- | --- | --- |
| Weights read per token, FP16 | 2.47 GB | 2.07 GB |
| Decode ceiling, batch 1, FP16 | 168 tokens/s | 200 tokens/s |
| kvbound decode, batch 1, FP16 (`make bench`) | 154 tokens/s, 92% of the ceiling | 177 tokens/s, 88% |
| kvbound decode without CUDA graphs | 150 tokens/s | 170 tokens/s |
| Time to first token, 128 / 1,024 prompt tokens | 11 ms / 70 ms | 11 ms / 74 ms |
| Static batching, 8 slots: `fixed` / `uniform` / `longtail` | 883 / 658 / 397 tokens/s | 955 / 723 / 451 tokens/s |
| Paged, continuous batching, same pool (256 / 176 MiB) | 1877 / 1628 / 1397 tokens/s | 1776 / 1593 / 1469 tokens/s |
| Most sequences in a step, paged | 30 / 28 / 48 | 30 / 28 / 48 |
| Perplexity, FP16 / INT8 / INT4 weights | 10.99 / 11.01 / 14.76 | 8.87 / 8.87 / 9.96 |
| Perplexity, INT4 weights with AWQ scales | 12.91 | 9.40 |
| Decode, batch 1, FP16 / INT8 / INT4 weights (`make precision`) | 154 / 283 / 474 tokens/s | 176 / 314 / 507 tokens/s |
| Sequences at once, FP16 / INT8 KV, 96 `fixed` requests | 30 / 58 | 30 / 58 |
| Hugging Face Transformers decode, batch 1 | 47 tokens/s | 17 to 33 tokens/s |

Llama 3.2 reads 19% more weight bytes per token than TinyLlama but has 16 layers instead of 22, so a decode step is
150 kernel launches instead of 204 and less of the step goes to work that is not reading weights: the step ends 0.5
to 0.6 ms above its floor at every weight precision, where TinyLlama's ends 0.7 ms above. Decode rows are by the
median step, like every decode figure for this card.

These runs come from Windows (WDDM), where GPU clocks cannot be locked, so every record is tagged `os=windows`, and
`clock_varied` when the clock moved during the run. Numbers meant for publication come from native Ubuntu with locked
clocks. Exact values are in `results/raw/` and the figures in `reports/`.

![Roofline](reports/rtx3060ti/roofline.png)

## Track W: Whisper on the same engine

Whisper is a speech model in two parts. An encoder turns a 30-second window of audio, as log-mel features, into
1,500 positions; a decoder then writes the transcript one token at a time, attending to its own tokens and to the
encoder's output (cross attention). Track W runs `whisper-base-ar-quran`, Tarteel's fine-tune of Whisper base for
Quran recitation (6 encoder and 6 decoder layers of width 512), on kvbound's engine.

What the engine gained for it:

- **Two programs over one weight table.** The encoder program runs the convolution stem as `Unfold` (each frame
  with its two neighbours) and a matrix product, then GELU, the position table, pre-norm layers with LayerNorm, and
  for each decoder layer the projection of the cross-attention keys and values into the second cache. The decoder
  program runs the embedding and positions, self attention over a paged cache, cross attention over the encoder's
  keys and values, the MLP, the tied vocabulary projection, the suppression of tokens the rules forbid, and argmax.
  Both index one table of weights, loaded once. New ops: `LayerNorm`, `AddLayerNorm`, `Gelu`, `Unfold`,
  `AddPositions`, `DenseAttention` (the encoder's attention, on the tiled tensor-core prefill kernel without the
  causal mask) and `SuppressTokens`. Biases are an epilogue of kvbound's own product kernels, which Llama calls with
  none, and a separate kernel after cuBLAS. Fusion folds each residual add into the LayerNorm after it.
- **`AsrEngine`**, a sibling of `Engine` with two caches. The Self cache, of the decoder's own keys and values, is
  the paged cache of Phase 4 in 16-token blocks. The Cross cache holds one block of 1,500 positions per sequence,
  filled by the encoder step. The scheduler gives an admitted request an encoder step before its first decoder rows;
  a preempted sequence loses both caches and is encoded again when it is readmitted. Decode-shaped steps replay
  CUDA graphs from the graph cache that `Engine` also uses.
- **Decoding rules from the checkpoint.** Fetching writes the prompt, the suppressed tokens, the tokens not allowed
  first, the end token and the 448-token limit into `generation.json`, as Hugging Face `generate` applies them to
  this checkpoint. Its prompt is the start token alone; the model writes the language and task tokens itself.
- **Beam search** is Hugging Face's algorithm (`kvb::BeamSearch`). The live beams are the rows of one decoder step.
  At a fork each child takes its parent's list of Self blocks, with reference counts, and a block that another beam
  still holds is copied before a beam writes into it (copy on write). Every beam points at the clip's one Cross
  block. Each step copies the logits to the host and ranks the candidates there.
- **Quantization** covers the matrices only (INT8 per channel, INT4 in groups of 128); convolution weights, position
  tables, biases and norm parameters stay FP16. Either cache can hold INT8.

**Tests.** `fixtures/tiny-whisper/` is a 2+2-layer random Whisper with goldens of every encoder and decoder site.
The engine matches each site within the FP16 limits, fused and unfused, at FP16, INT8 and INT4 against the
dequantizing reference. Greedy output equals the reference's token for token under both scheduling policies, with
and without graphs, and through preemptions. Beam search reproduces the reference search's committed traces, with
blocks shared between beams or every block private. On 20 local recitation clips the slow tests require greedy
output to equal Hugging Face FP16 `generate` token for token, or to part from it first where the FP32 reference
scores the two tokens within FP16 noise of each other (`NEAR_TIE` and `FP16_GAP_NOISE` in
`tests/py/test_asr_greedy.py`); beam 5 must equal Hugging Face's beam 5 or end on hypotheses the FP32 reference
scores within `RESCORE_TOLERANCE` of each other.

### Measured against Hugging Face and faster-whisper (RTX 3060 Ti)

The clips are `everyayah-test-1000`: 1,000 clips of the test split of `tarteel-ai/everyayah`, 3.79 hours from 36
reciters, drawn by reciter (`configs/evals/everyayah-test-1000.json`). Each clip is decoded as one window, without
timestamps, one clip at a time. All 16 records come from one session at `4089b55`; faster-whisper's `int8` runs on
this machine's CPU with 4 threads, the setting the Sawt Sakinah app uses.

- **WER**, the word error rate, is the word-level edit distance from the reference (substitutions, insertions and
  deletions) summed over the clips and divided by the reference words, 10,945 here. Both sides are first normalized
  as the Sawt Sakinah app matches words: diacritics removed and one spelling kept for each letter Arabic writes
  several ways (`kvbound.evals.text`).
- **RTF**, the real-time factor, is processing time divided by audio time, summed over the clips; 0.0047 means 4.7
  ms per second of audio. The *model* RTF runs from the log-mel features on the host to the token ids on the host:
  the encoder, every decoder step and the copies. The *end-to-end* RTF runs from the waveform to the text, so it
  adds the log-mel front end and detokenization.

| Stack | WER, greedy / beam 5 | RTF, model, greedy / beam 5 | RTF, end to end, greedy / beam 5 |
| --- | --- | --- | --- |
| kvbound, FP16 | 0.58% / 0.48% | 0.0047 / 0.0313 | 0.0051 / 0.0317 |
| kvbound, INT8 weights | 0.55% / 0.46% | 0.0040 / 0.0308 | 0.0045 / 0.0313 |
| kvbound, INT4 weights | 1.42% / 1.28% | 0.0039 / 0.0300 | 0.0043 / 0.0304 |
| kvbound, FP16 weights, INT8 caches | 0.58% / 0.48% | 0.0047 / 0.0312 | 0.0052 / 0.0316 |
| Hugging Face, FP16 | 0.58% / 0.48% | 0.0568 / 0.0684 | 0.0572 / 0.0689 |
| faster-whisper GPU, `float16` | 0.72% / 0.60% | 0.0164 / 0.0209 | 0.0189 / 0.0235 |
| faster-whisper GPU, `int8_float16` | 0.67% / 0.57% | 0.0263 / 0.0308 | 0.0287 / 0.0334 |
| faster-whisper CPU, `int8` | 0.66% / 0.65% | 0.0616 / 0.0839 | 0.0647 / 0.0870 |

- **kvbound and Hugging Face agree clip by clip.** In FP16, greedy and at beam 5, the two have the same number of
  word errors and the same number of decoder steps on each of the 1,000 clips. The records hold those counts, not
  the token ids; the slow tests above compare the ids.
- **Greedy, kvbound is the fastest stack here**: by model RTF 12.1 times faster than Hugging Face and 3.5 times
  faster than faster-whisper `float16`, at 0.70 ms per generated token.
- **INT8 weights cost no accuracy and INT4 weights do.** INT8 takes 14% less time than FP16 at 0.55% WER against
  0.58%; INT4 takes 17% less and raises WER to 1.42%. FP16 weights with INT8 caches give the same word errors and
  steps as FP16 on every clip, at the same speed.
- **faster-whisper differs in more than numerics.** With the language set to Arabic it forces the prompt
  `<|startoftranscript|><|ar|><|transcribe|><|notimestamps|>`, where Hugging Face and kvbound feed the start token
  alone and the model writes the other three. It also computes the log-mel features of the clip before padding it
  to 30 seconds, where the other two pad first. Its per-clip word errors equal Hugging Face's on 984 to 994 of the
  1,000 clips, depending on the configuration (994 for `float16`, 984 for the CPU's `int8` greedy), so its higher WER
  comes from a few clips, and the records cannot separate what the prompt and the front end contribute from what its
  arithmetic does.
- **Beam 5 on kvbound is slow for now.** A generated token takes 5.09 ms against 0.70 greedy, and faster-whisper
  `float16` at beam 5 is faster (model RTF 0.0209 against 0.0313). Every step copies the logits to the host and ranks
  the candidates there, the likely main cost, though no record times it apart from the rest of the step. Ranking on
  the GPU is in progress, and the beam-5 rows will be recorded again. The beam runs' encoder step also reads longer
  than the greedy runs' (10.0 against 5.9 ms for FP16); why is not established.
- **What a WER this low means.** The model was fine-tuned on recitations of this text: the clips are the dataset's
  test split, but every split reads the same Quran. A WER near zero shows that the stacks agree, not that the task is
  hard. The subset leans toward the start of each audio page of the dataset, which keeps the download small; its
  clips average 13.64 s against 12.61 s over the 20,914 test clips of at most 30 s. The other 2,559 of the 23,473
  test clips are longer than one window and left out. The recordings carry no documented license, so the clips are
  evaluated only, never committed or redistributed.

What each stack's timings include, the encoder and per-token times, and the clocks of each run are in the
[log](docs/optimization-log.md#track-w-whisper-measurements-rtx-3060-ti).

```sh
make fetch MODEL=whisper-base-ar-quran          # converts the checkpoint and writes generation.json
uv run python -m kvbound.quantize whisper-base-ar-quran --scheme int8_pc   # and --scheme int4_g
uv run python -m kvbound.evals.asr subset       # the 1,000 clips into data/datasets/, checked against the manifest
uv run python -m kvbound.evals.asr eval --model whisper-base-ar-quran --backend kvbound --compute-type fp16:int8 --beams 5
make bench-asr MODEL=whisper-base-ar-quran      # Hugging Face and faster-whisper, greedy and beam 5
```

## Layout

| Path | What lives there |
| --- | --- |
| `engine/` | the C++ side. `kvbound` (CPU only): safetensors and JSON loaders, the weight format, the Llama and Whisper op programs, the activation memory planner, the scheduler (static and continuous, with encoder steps), the sampler, the beam search, cache bookkeeping with reference-counted blocks. `kvbound_cuda`: kernels, device weights, KV pool, the engine and the speech engine (`AsrEngine`) |
| `engine/kernels/` | CUDA kernels: embedding, RMSNorm (also fused with the residual add), LayerNorm (likewise), RoPE (also fused with the cache write), SwiGLU gate, GELU, position add, the convolution unfold, residual add, decode attention, tiled tensor-core prefill attention and its non-causal encoder form over an FP16 or INT8 KV cache, token suppression, argmax, an FP16 GEMV, products and lookups on INT8 and INT4 weights in place and on tensor cores, with an optional bias; cuBLAS for wide FP16 products |
| `apps/` | `kvb-bench` runs a trace of requests through the scheduler and prints raw timings; `kvb-asr` transcribes log-mel features greedily or by beam search; `kvb-kv_layout` times attention on each KV layout, on 16-token blocks and on an INT8 cache |
| `bindings/` | `_kvb`, the engine and each kernel as a Python extension, for tests and evals |
| `microbench/` | standalone CUDA binaries that measure bandwidth, latency and GEMM peaks |
| `python/kvbound/ref/` | from-scratch PyTorch reference models: Llama (RMSNorm, RoPE, grouped-query attention, SwiGLU), which loads quantized checkpoints by dequantizing them, and Whisper (conv stem, encoder, decoder with cross attention, greedy decoding and Hugging Face's beam search) |
| `python/kvbound/` | golden tensors, tolerance metrics, result records, fingerprint, baselines (Hugging Face for both models, faster-whisper for Whisper), perf model, workloads, the quantizer and its AWQ scale search, the log-mel front end, speech recognition through the engine (`asr.py`), bench, serve and profile wrappers, plots |
| `python/kvbound/evals/` | WikiText-2 perplexity, per-site error against the goldens with sample text, and the speech evaluation: the recitation subset, Arabic normalization, word and character error rates, real-time factors |
| `fixtures/tiny-llama/`, `fixtures/tiny-whisper/` | committed 2-layer random models with goldens and thresholds (and reference beam-search traces for Whisper); what CI tests against |
| `tests/cpp/` | GoogleTest, CPU only: parser, loader, weight format, fixture contracts, op programs, memory plan, both scheduling policies, sampler, allocator and block reference counts, beam search against the reference traces |
| `tests/py/` | pytest: reference vs Hugging Face, the quantizer, the perplexity and error-rate arithmetic; under the `gpu` marker, each kernel vs torch and both engines vs the goldens, FP16 and quantized, and Whisper greedy and beam search vs the reference |
| `configs/` | model, device, workload and evaluation-subset descriptions |
| `results/raw/<device>/` | one JSON record per benchmark or evaluation run, written only by `kvbound.record` |
| `results/profiles/` | Nsight Systems summaries (CSV) of decode runs; Nsight Compute counters of the attention kernel and of every kernel in one step |
| `docs/` | `optimization-log.md`: each Phase 6 change with its hypothesis, before and after, and profiler evidence, the comparison of the two cards, and the Track W measurements in detail |
| `reports/<device>/` | figures generated from the records |

## Build and test

```sh
make setup                 # TORCH=cu128 on GPU machines, TORCH=cpu elsewhere
make test                  # cpu preset: C++ tests and the fixture-based Python tests (what CI runs)
make test-gpu              # builds the cuda preset; every kernel vs torch, the engine vs the fixture goldens
make sanitize              # the engine under compute-sanitizer: memcheck, racecheck, initcheck
make help                  # every target
```

Requires CMake 3.27+, Ninja, a C++20 compiler, CUDA 12.x for the `cuda` presets, and [uv](https://docs.astral.sh/uv/).
On Windows, `make` needs Git Bash on `PATH` and finds MSVC through `scripts/vsenv.cmd`.

## Reproduce the measurements on a GPU machine

```sh
make doctor                          # GPU, toolchain, profilers, clock state
make fetch MODEL=tinyllama-1.1b      # pinned checkpoint into data/models/
make golden MODEL=tinyllama-1.1b     # golden tensors into data/golden/ (the per-GPU manifest is committed)
make quantize MODEL=tinyllama-1.1b   # INT8, INT4 and INT4-with-AWQ copies of the checkpoint into data/models/
make test-real                       # reference vs Hugging Face, engine vs reference, on the real checkpoint
make bench-hf MODEL=tinyllama-1.1b   # Hugging Face baseline records
make microbench                      # hardware ceilings and the KV layout benchmark; builds the cuda preset first
make bench MODEL=tinyllama-1.1b      # kvbound engine records, one request at a time
make serve MODEL=tinyllama-1.1b      # the three workloads in one KV pool: static, continuous over slots, paged
make serve-load MODEL=tinyllama-1.1b # static against paged as requests arrive at 2 to 16 per second
make precision MODEL=tinyllama-1.1b  # each weight and KV precision: perplexity, per-layer error, decode, concurrency
make profile MODEL=tinyllama-1.1b    # Nsight Systems decode profiles
make counters                        # Nsight Compute counters of the attention kernel (needs GPU counter access)
make step-counters MODEL=tinyllama-1.1b WEIGHTS=int4_g   # Nsight Compute counters of one decode and one prefill step
make report MODEL=tinyllama-1.1b DEVICE=rtx3060ti   # figures into reports/
make predict MODEL=llama-3.2-1b DEVICE=rtx3060ti
```

Benchmarks refuse to run on a dirty tree unless given `--allow-dirty`, so every record maps to a commit.
`llama-3.2-1b` is gated: accept its license on Hugging Face and run `hf auth login` before `make fetch`.
