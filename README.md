# kvbound

A from-scratch LLM inference engine in C++ and CUDA. It implements two KV-cache designs for the same engine, a static
preallocated one (how phone NPU runtimes work) and a paged one with continuous batching (how GPU serving systems
work), then quantizes and profiles both against the roofline of an RTX 3060 Ti (Ampere) and an RTX 4070 Ti (Ada).

**Status:** Phases 0 and 1 are done on the RTX 3060 Ti: reference model, golden tensors, benchmark harness, hardware
ceilings and the analytical performance model. The engine itself starts in Phase 2. The RTX 4070 Ti runs are pending.

## Where things stand (RTX 3060 Ti, TinyLlama 1.1B, FP16)

| Quantity | Value | Source |
| --- | --- | --- |
| Memory bandwidth, working set beyond cache | about 371 GB/s (83% of the 448 GB/s spec) | `kvb-bandwidth_sweep` |
| Dependent-load latency, L1 / L2 / DRAM | about 21 / 135 / 312 ns | `kvb-pointer_chase` |
| GEMM peak, FP16 with FP16 / FP32 accumulate; INT8 | about 58 / 33 TFLOPS; 115 TOPS | `kvb-gemm_peak` (cuBLASLt) |
| Decode ceiling, batch 1 | about 179 tokens/s (weights 2.07 GB read once per token) | `python -m kvbound.perfmodel` |
| Hugging Face Transformers decode, batch 1 | 17 to 33 tokens/s, depending on where the unlocked GPU clock settles | `python -m kvbound.baselines.hf` |

These runs come from Windows (WDDM), where GPU clocks cannot be locked, so every record is tagged `os=windows`, and `clock_varied` when the
clock moved during the run.
Numbers meant for publication come from native Ubuntu with locked clocks. Exact values are in `results/raw/` and the
figures in `reports/`.

![Roofline](reports/rtx3060ti/roofline.png)

## Layout

| Path | What lives there |
| --- | --- |
| `engine/` | `libkvbound`: the C++ side. Today: JSON parser, memory-mapped safetensors loader, config reader |
| `microbench/` | standalone CUDA binaries that measure bandwidth, latency and GEMM peaks |
| `python/kvbound/ref/` | from-scratch PyTorch reference model (RMSNorm, RoPE, grouped-query attention, SwiGLU) |
| `python/kvbound/` | golden tensors, tolerance metrics, result records, fingerprint, baselines, perf model, plots |
| `fixtures/tiny-llama/` | committed 2-layer random model with goldens and thresholds; what CI tests against |
| `tests/cpp/`, `tests/py/` | GoogleTest (parser, loader, fixture contract) and pytest (reference vs Hugging Face) |
| `configs/` | model and device descriptions |
| `results/raw/<device>/` | one JSON record per benchmark run, written only by `kvbound.record` |
| `reports/<device>/` | figures generated from the records |

## Build and test

```sh
make setup                 # TORCH=cu128 on GPU machines, TORCH=cpu elsewhere
make test                  # cpu preset: C++ tests and the fixture-based Python tests (what CI runs)
make build PRESET=cuda     # adds the CUDA microbenchmarks
make help                  # every target
```

Requires CMake 3.27+, Ninja, a C++20 compiler, CUDA 12.x for the `cuda` presets, and [uv](https://docs.astral.sh/uv/).
On Windows, `make` needs Git Bash on `PATH` and finds MSVC through `scripts/vsenv.cmd`.

## Reproduce Phases 0 and 1 on a GPU machine

```sh
make doctor                          # GPU, toolchain, profilers, clock state
make fetch MODEL=tinyllama-1.1b      # pinned checkpoint into data/models/
make test-real                       # reference model vs Hugging Face on the real checkpoint
make golden MODEL=tinyllama-1.1b     # golden tensors into data/golden/ (manifest is committed)
make bench-hf MODEL=tinyllama-1.1b   # Hugging Face baseline records
make microbench                      # builds the cuda preset first
make report MODEL=tinyllama-1.1b DEVICE=rtx3060ti   # figures into reports/
make predict MODEL=llama-3.2-1b DEVICE=rtx3060ti
```

Benchmarks refuse to run on a dirty tree unless given `--allow-dirty`, so every record maps to a commit.
`llama-3.2-1b` is gated: accept its license on Hugging Face and run `hf auth login` before `make fetch`.
