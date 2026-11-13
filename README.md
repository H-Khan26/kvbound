# kvbound

A from-scratch LLM inference engine in C++ and CUDA. It implements two KV-cache designs for the same engine, a static
preallocated one (how phone NPU runtimes work) and a paged one with continuous batching (how GPU serving systems
work), then quantizes and profiles both against the roofline of an RTX 3060 Ti (Ampere) and an RTX 4070 Ti (Ada).

**Status:** Phases 0 to 5 are done on the RTX 3060 Ti, measured on TinyLlama 1.1B and Llama 3.2 1B: reference model,
golden tensors, hardware ceilings, the analytical performance model, an FP16 engine that generates the same tokens as
the reference model, the static memory design, the paged KV cache with continuous batching, and quantization (INT8
and INT4 weights, an INT8 KV cache). Profiling and optimization (Phase 6) is next. The RTX 4070 Ti runs are pending.

## Where things stand (RTX 3060 Ti, TinyLlama 1.1B)

| Quantity | Value | Source |
| --- | --- | --- |
| Memory bandwidth beyond cache, best 2 ms burst | about 415 GB/s (93% of the 448 GB/s spec) | `kvb-bandwidth_sweep` |
| Memory bandwidth beyond cache, sustained 0.1 s | about 367 GB/s; other desktop programs take turns on the GPU | `kvb-bandwidth_sweep` |
| Dependent-load latency, L1 / L2 / DRAM | about 21 / 135 / 312 ns | `kvb-pointer_chase` |
| GEMM peak, FP16 with FP16 / FP32 accumulate; INT8 | about 58 / 33 TFLOPS; 115 TOPS | `kvb-gemm_peak` (cuBLASLt) |
| Decode ceiling, batch 1, FP16 | about 200 tokens/s (weights 2.07 GB read once per token) | `python -m kvbound.perfmodel` |
| **kvbound decode, batch 1, FP16** | **160 tokens/s, 80% of the ceiling** (6.3 ms per token; 144 tokens/s without CUDA graphs) | `python -m kvbound.bench` |
| kvbound time to first token | 14 ms for a 128-token prompt, 204 ms for 1,024 tokens | `python -m kvbound.bench` |
| **kvbound decode, batch 1, INT8 / INT4 weights** | **282 / 418 tokens/s**, 71% / 54% of their ceilings | `python -m kvbound.bench --weights ...` |
| kvbound static batching, 8 slots | 883 tokens/s on equal-length requests, 403 on long-tailed lengths (153 to 156 one at a time) | `python -m kvbound.serve` |
| **kvbound paged KV, continuous batching, same pool** | **1453 tokens/s on equal-length requests, 1339 on long-tailed lengths** | `python -m kvbound.serve --policy continuous` |
| Hugging Face Transformers decode, batch 1 | 17 to 33 tokens/s, depending on where the unlocked GPU clock settles | `python -m kvbound.baselines.hf` |

Greedy output matches the reference model token for token on every golden prompt, and every intermediate activation
stays within the FP16 limits in the golden manifest (`make test-real`).

Where an FP16 decode step's GPU time goes (Nsight Systems, `results/profiles/`): 90% in cuBLAS streaming the weights, 3%
in attention, 7% in the small kernels (norms, RoPE, residual adds). Launching the step's 357 kernels one by one costs
about 1.5 ms of CPU time; replaying them from a CUDA graph costs 0.13 ms.

These runs come from a shared Windows desktop with unlocked GPU clocks, and a run can differ from the next by about a
tenth. The decode figure above was 144 tokens/s when Phase 2 was first recorded; every number here is from the newest
record of its kind.

### What the static memory design costs

The static design preallocates the whole KV cache (8 slots of 1,024 tokens, 176 MiB) and gives every admitted sequence
a full slot until its batch ends.

| Workload (48 requests) | Tokens/s | Reserved KV memory holding tokens | Slots busy |
| --- | --- | --- | --- |
| `fixed`: 256-token prompts, 128-token outputs | 883 | 29% | 99% |
| `uniform`: prompts 32 to 512, outputs 16 to 256 | 658 | 32% | 71% |
| `longtail`: lognormal, median prompt 64 and output 48 | 403 | 16% | 38% |

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
- **Activation memory.** The forward pass has 225 intermediate tensors. Side by side they would need 1,039 MiB for
  1,024-token steps; placed by lifetime in one arena they share 30 MiB, which is also the lower bound. Figure:
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
| `fixed` | static batches, 8 slots | 883 | 8 | 3.08 s | 7.7 ms | 29% |
| `fixed` | continuous, 8 whole slots | 894 | 8 | 3.03 s | 8.3 ms | 30% |
| `fixed` | continuous, 16-token blocks | 1453 | 30 | 0.83 s | 15.7 ms | 86% |
| `uniform` | static batches, 8 slots | 658 | 8 | 5.24 s | 7.5 ms | 32% |
| `uniform` | continuous, 8 whole slots | 842 | 8 | 3.75 s | 8.3 ms | 33% |
| `uniform` | continuous, 16-token blocks | 1331 | 28 | 0.93 s | 13.5 ms | 89% |
| `longtail` | static batches, 8 slots | 403 | 8 | 4.63 s | 6.7 ms | 16% |
| `longtail` | continuous, 8 whole slots | 899 | 8 | 1.31 s | 7.2 ms | 19% |
| `longtail` | continuous, 16-token blocks | 1339 | 48 | 0.25 s | 14.1 ms | 87% |

![One KV pool served three ways](reports/rtx3060ti/serving.png)

![KV memory, whole slots against blocks](reports/rtx3060ti/paged_memory.png)

- **What it costs.** Steps carry more rows, so each token takes longer to arrive: the median time per output token on
  `fixed` rises from 7.7 to 15.7 ms. Admission does not know how long a response will get, so a full pool
  preempts: 9 preemptions and 2,769 tokens processed twice per pass on `fixed`, 5 and 1,684 on `uniform`, none on
  `longtail`, whose 48 requests all fit at once.
- **The block table.** Attention finds each cached position's block once per element it reads. Doing that with a
  division made attention 1.37 to 1.84 times slower than one block per sequence; a shift and a mask bring it to
  1.07 to 1.12 times, so a paged cache needs a power-of-two block size. Whether a sequence's blocks are in
  order or scattered over the pool makes no difference (`python -m kvbound.microbench kv_layout`).
- **Arrivals.** With `longtail` requests arriving as a Poisson process at 8 per second, static batching has a median
  time to first token of 2.4 s and the paged engine 16 ms; at 16 per second, 3.8 s and 18 ms
  (`make serve-load`, figure `reports/rtx3060ti/serving_load.png`).
- **CUDA graphs.** A graph replays one row count and continuous batching keeps changing it, so the engine keeps the 16
  most recently used. On `longtail` that is 41 captures and 95 ms per pass with 99% of steps replayed; with room
  for 64 graphs, 0 captures and 1386 tokens/s against 1339.

### Quantization

Weights are quantized by round-to-nearest with FP16 scales: INT8 per output channel, or INT4 in groups of 128 along
the input dimension, two values to a byte. The format is written into the checkpoint's safetensors metadata
(`python -m kvbound.quantize`) and read by the C++ loader and by the reference model, which dequantizes the same
file. The KV cache can hold INT8 with one FP16 scale per cached vector, quantized when written and scaled inside the
attention kernel. Perplexity is WikiText-2, the whole test split in 1,024-token windows, from the engine's logits.
The last two columns serve 96 `fixed` requests at once in a KV pool of the same bytes.

| TinyLlama 1.1B | Perplexity | Weights on device | KV bytes per token | Decode tokens/s | Of its ceiling | Sequences at once | Tokens/s under load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 weights, FP16 KV | 8.87 | 2,098 MiB | 22,528 | 160 | 80% | 30 | 1504 |
| FP16 weights, INT8 KV | 8.87 | 2,098 MiB | 11,616 | 159 | 79% | 58 | 1754 |
| INT8 weights, FP16 KV | 8.87 | 1,050 MiB | 22,528 | 282 | 71% | 30 | 1014 |
| INT8 weights, INT8 KV | 8.87 | 1,050 MiB | 11,616 | 279 | 70% | 58 | 1394 |
| INT4 weights, FP16 KV | 9.96 | 541 MiB | 22,528 | 418 | 54% | 30 | 1063 |
| INT4 weights, INT8 KV | 9.96 | 541 MiB | 11,616 | 413 | 53% | 58 | 1446 |

![Precision](reports/rtx3060ti/precision.png)

Read against the roofline:

- **Decode speed follows the bytes.** A decode step is the weights crossing the memory bus once plus about a
  millisecond of everything else. The floor set by the weights shrinks from 5.0 to 2.5 to 1.3 ms and the measured
  step from 6.3 to 3.6 to 2.4 ms. The share of the ceiling falls because the rest does not shrink.
- **INT8 weights are free in accuracy; plain INT4 is not.** Activation-aware scales (AWQ) are not built. Choosing each
  group's clipping to minimise weight error does not fix it: TinyLlama scores 10.84 with it against 9.96 without, Llama 3.2 scores 14.63 with it against 14.76 without.
- **An INT8 cache buys concurrency, not decode speed.** The same pool holds 992 blocks instead of 512. Attention
  over an INT8 cache is 10% faster at batch 8, but attention is a few percent of a step and was never bound by
  bandwidth.
- **Under load, quantized weights are slower than FP16.** The in-place kernel multiplies quantized weights for steps
  of up to 16 rows; wider steps expand each matrix to FP16 for the tensor-core product, which costs about 8 ms a step.
  A decode step at batch 2 takes 3.9 ms with INT8 weights in place against 6.6 ms in FP16; at batch 16 it takes 15.0 ms in place, 16.5 ms expanded and 8.9 ms in FP16. Past a few rows the step is bound by arithmetic, and smaller weights save memory, not time.
- The engine and the dequantizing reference agree on every capture site within the FP16 limits (`make test-gpu`); the
  reference model scores 8.87 on the FP16 checkpoint where the engine scores 8.87.

Known gaps: prefill attention is still the simple per-query kernel (204 ms for 1,024 tokens against a 36 ms bound),
cuBLAS is slower per token at batch 8 than at batch 1, INT4 needs better scales, prefix sharing is not built, and the
FP8 KV cache waits for the RTX 4070 Ti.

### The same engine on Llama 3.2 1B

Llama 3.2 1B (16 layers, 8 KV heads, tied embeddings, Llama 3 RoPE scaling) runs on the same code with no changes.
Greedy output matches the reference token for token on every golden prompt, and the worst intermediate activation is at
18% of its FP16 limit, as on TinyLlama.

| Quantity (RTX 3060 Ti) | Llama 3.2 1B | TinyLlama 1.1B |
| --- | --- | --- |
| Weights read per token, FP16 | 2.47 GB | 2.07 GB |
| Decode ceiling, batch 1, FP16 | 168 tokens/s | 200 tokens/s |
| kvbound decode, batch 1, FP16 | 144 tokens/s, 86% of the ceiling | 160 tokens/s, 80% |
| kvbound decode without CUDA graphs | 135 tokens/s | 144 tokens/s |
| Time to first token, 128 / 1,024 prompt tokens | 14 ms / 165 ms | 14 ms / 204 ms |
| Static batching, 8 slots: `fixed` / `uniform` / `longtail` | 828 / 613 / 370 tokens/s | 883 / 658 / 403 tokens/s |
| Paged, continuous batching, same pool (256 / 176 MiB) | 1577 / 1404 / 1299 tokens/s | 1453 / 1331 / 1339 tokens/s |
| Most sequences in a step, paged | 30 / 28 / 48 | 30 / 28 / 48 |
| Perplexity, FP16 / INT8 / INT4 weights | 10.99 / 11.01 / 14.76 | 8.87 / 8.87 / 9.96 |
| Decode, batch 1, FP16 / INT8 / INT4 weights | 144 / 264 / 413 tokens/s | 160 / 282 / 418 tokens/s |
| Sequences at once, FP16 / INT8 KV, 96 `fixed` requests | 30 / 58 | 30 / 58 |
| Hugging Face Transformers decode, batch 1 | 47 tokens/s | 17 to 33 tokens/s |

Llama 3.2 reads 19% more weight bytes per token than TinyLlama but has 16 layers instead of 22, so a decode step is
about 278 kernels instead of 357 and less of the step goes to work that is not reading weights.

These runs come from Windows (WDDM), where GPU clocks cannot be locked, so every record is tagged `os=windows`, and
`clock_varied` when the clock moved during the run. Numbers meant for publication come from native Ubuntu with locked
clocks. Exact values are in `results/raw/` and the figures in `reports/`.

![Roofline](reports/rtx3060ti/roofline.png)

## Layout

| Path | What lives there |
| --- | --- |
| `engine/` | the C++ side. `kvbound` (CPU only): safetensors and JSON loaders, the weight format, the op program, the activation memory planner, the scheduler (static and continuous), the sampler, cache bookkeeping. `kvbound_cuda`: kernels, device weights, KV pool, the engine |
| `engine/kernels/` | CUDA kernels: embedding, RMSNorm, RoPE, SwiGLU gate, residual add, attention over an FP16 or INT8 KV cache, argmax, products and lookups on INT8 and INT4 weights; cuBLAS for FP16 matrix products |
| `apps/` | `kvb-bench` runs a trace of requests through the scheduler and prints raw timings; `kvb-kv_layout` times attention on each KV layout, on 16-token blocks and on an INT8 cache |
| `bindings/` | `_kvb`, the engine and each kernel as a Python extension, for tests and evals |
| `microbench/` | standalone CUDA binaries that measure bandwidth, latency and GEMM peaks |
| `python/kvbound/ref/` | from-scratch PyTorch reference model (RMSNorm, RoPE, grouped-query attention, SwiGLU); loads quantized checkpoints by dequantizing them |
| `python/kvbound/` | golden tensors, tolerance metrics, result records, fingerprint, baselines, perf model, workloads, the quantizer, bench, serve and profile wrappers, plots |
| `python/kvbound/evals/` | WikiText-2 perplexity, and per-site error against the goldens with sample text |
| `fixtures/tiny-llama/` | committed 2-layer random model with goldens and thresholds; what CI tests against |
| `tests/cpp/` | GoogleTest, CPU only: parser, loader, weight format, fixture contract, op program, memory plan, both scheduling policies, sampler, allocator |
| `tests/py/` | pytest: reference vs Hugging Face, the quantizer, the perplexity arithmetic; under the `gpu` marker, each kernel vs torch and the engine vs the goldens, FP16 and quantized |
| `configs/` | model, device and workload descriptions |
| `results/raw/<device>/` | one JSON record per benchmark or evaluation run, written only by `kvbound.record` |
| `results/profiles/` | Nsight Systems summaries (CSV) of decode runs; Nsight Compute counters of the attention kernel |
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
make golden MODEL=tinyllama-1.1b     # golden tensors into data/golden/ (manifest is committed)
make quantize MODEL=tinyllama-1.1b   # INT8 and INT4 copies of the checkpoint into data/models/
make test-real                       # reference vs Hugging Face, engine vs reference, on the real checkpoint
make bench-hf MODEL=tinyllama-1.1b   # Hugging Face baseline records
make microbench                      # hardware ceilings and the KV layout benchmark; builds the cuda preset first
make bench MODEL=tinyllama-1.1b      # kvbound engine records, one request at a time
make serve MODEL=tinyllama-1.1b      # the three workloads in one KV pool: static, continuous over slots, paged
make serve-load MODEL=tinyllama-1.1b # static against paged as requests arrive at 2 to 16 per second
make precision MODEL=tinyllama-1.1b  # each weight and KV precision: perplexity, per-layer error, decode, concurrency
make profile MODEL=tinyllama-1.1b    # Nsight Systems decode profiles
make counters                        # Nsight Compute counters of the attention kernel (needs GPU counter access)
make report MODEL=tinyllama-1.1b DEVICE=rtx3060ti   # figures into reports/
make predict MODEL=llama-3.2-1b DEVICE=rtx3060ti
```

Benchmarks refuse to run on a dirty tree unless given `--allow-dirty`, so every record maps to a commit.
`llama-3.2-1b` is gated: accept its license on Hugging Face and run `hf auth login` before `make fetch`.
