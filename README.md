# kvbound

A from-scratch LLM inference engine in C++ and CUDA. It implements two KV-cache designs for the same engine, a static
preallocated one (how phone NPU runtimes work) and a paged one with continuous batching (how GPU serving systems
work), then quantizes and profiles both against the roofline of an RTX 3060 Ti (Ampere) and an RTX 4070 Ti (Ada).

**Status:** Phases 0 to 2 are done on the RTX 3060 Ti: reference model, golden tensors, hardware ceilings, the
analytical performance model, and a single-sequence FP16 engine that generates the same tokens as the reference model.
The RTX 4070 Ti runs are pending.

## Where things stand (RTX 3060 Ti, TinyLlama 1.1B, FP16, batch 1)

| Quantity | Value | Source |
| --- | --- | --- |
| Memory bandwidth beyond cache, best 2 ms burst | about 415 GB/s (93% of the 448 GB/s spec) | `kvb-bandwidth_sweep` |
| Memory bandwidth beyond cache, sustained 0.1 s | about 367 GB/s; other desktop programs take turns on the GPU | `kvb-bandwidth_sweep` |
| Dependent-load latency, L1 / L2 / DRAM | about 21 / 135 / 312 ns | `kvb-pointer_chase` |
| GEMM peak, FP16 with FP16 / FP32 accumulate; INT8 | about 58 / 33 TFLOPS; 115 TOPS | `kvb-gemm_peak` (cuBLASLt) |
| Decode ceiling | about 200 tokens/s (weights 2.07 GB read once per token) | `python -m kvbound.perfmodel` |
| **kvbound decode** | **136 tokens/s, 68% of the ceiling** (7.3 ms per token; 123 tokens/s without CUDA graphs) | `python -m kvbound.bench` |
| kvbound time to first token | 20 ms for a 128-token prompt, 511 ms for 1,024 tokens | `python -m kvbound.bench` |
| Hugging Face Transformers decode | 17 to 33 tokens/s, depending on where the unlocked GPU clock settles | `python -m kvbound.baselines.hf` |

Greedy output matches the reference model token for token on every golden prompt, and every intermediate activation
stays within the FP16 limits in the golden manifest (`make test-real`).

Where a decode step's GPU time goes (Nsight Systems, `results/profiles/`): 89% in cuBLAS streaming the weights, 4% in
attention, 7% in the small kernels (norms, RoPE, residual adds). Launching the step's 357 kernels one by one costs about
1.5 ms of CPU time; replaying them from a CUDA graph costs 0.11 ms.

Known gap: prefill attention is the naive per-query kernel, so long prompts are slow (511 ms against a 36 ms bound).

These runs come from Windows (WDDM), where GPU clocks cannot be locked, so every record is tagged `os=windows`, and
`clock_varied` when the clock moved during the run. Numbers meant for publication come from native Ubuntu with locked
clocks. Exact values are in `results/raw/` and the figures in `reports/`.

![Roofline](reports/rtx3060ti/roofline.png)

## Layout

| Path | What lives there |
| --- | --- |
| `engine/` | the C++ side. `kvbound` (CPU only): safetensors and JSON loaders, the op program, the activation memory planner, the sampler, cache bookkeeping. `kvbound_cuda`: kernels, device weights, KV pool, the engine |
| `engine/kernels/` | CUDA kernels: embedding, RMSNorm, RoPE, SwiGLU gate, residual add, attention over the KV cache, argmax; cuBLAS for matrix products |
| `apps/kvb-bench` | runs a trace of requests and prints raw token timestamps |
| `bindings/` | `_kvb`, the engine and each kernel as a Python extension, for tests |
| `microbench/` | standalone CUDA binaries that measure bandwidth, latency and GEMM peaks |
| `python/kvbound/ref/` | from-scratch PyTorch reference model (RMSNorm, RoPE, grouped-query attention, SwiGLU) |
| `python/kvbound/` | golden tensors, tolerance metrics, result records, fingerprint, baselines, perf model, bench and profile wrappers, plots |
| `fixtures/tiny-llama/` | committed 2-layer random model with goldens and thresholds; what CI tests against |
| `tests/cpp/` | GoogleTest, CPU only: parser, loader, fixture contract, op program, memory plan, sampler, allocator |
| `tests/py/` | pytest: reference vs Hugging Face; under the `gpu` marker, each kernel vs torch and the engine vs the goldens |
| `configs/` | model and device descriptions |
| `results/raw/<device>/` | one JSON record per benchmark run, written only by `kvbound.record` |
| `results/profiles/` | Nsight Systems summaries (CSV) of decode runs |
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
make test-real                       # reference vs Hugging Face, engine vs reference, on the real checkpoint
make bench-hf MODEL=tinyllama-1.1b   # Hugging Face baseline records
make microbench                      # hardware ceilings; builds the cuda preset first
make bench MODEL=tinyllama-1.1b      # kvbound engine records
make profile MODEL=tinyllama-1.1b    # Nsight Systems decode profiles
make report MODEL=tinyllama-1.1b DEVICE=rtx3060ti   # figures into reports/
make predict MODEL=llama-3.2-1b DEVICE=rtx3060ti
```

Benchmarks refuse to run on a dirty tree unless given `--allow-dirty`, so every record maps to a commit.
`llama-3.2-1b` is gated: accept its license on Hugging Face and run `hf auth login` before `make fetch`.
