# kvbound

A from-scratch LLM inference engine in C++ and CUDA. It implements two KV-cache designs for the same engine, a static
preallocated one (how phone NPU runtimes work) and a paged one with continuous batching (how GPU serving systems
work), then quantizes and profiles both against the roofline of an RTX 3060 Ti (Ampere) and an RTX 4070 Ti (Ada).

**Status:** Stage 0, reference model and test harness. No performance numbers yet.

## Layout

| Path | What lives there |
| --- | --- |
| `engine/` | `libkvbound`: the C++/CUDA hot path (loader, kernels, KV cache, scheduler, runtime) |
| `python/kvbound/` | from-scratch PyTorch reference model, golden tensors, result records, analysis |
| `tests/cpp/`, `tests/py/` | GoogleTest (CPU logic) and pytest (reference model, GPU kernels via bindings) |
| `configs/` | model and device descriptions; later workloads and experiments |

## Build

```sh
make setup                 # TORCH=cu128 on GPU machines, TORCH=cpu elsewhere
make build PRESET=cuda     # PRESET=cpu builds without CUDA
make test PRESET=cuda
make help                  # every target
```

Requires CMake 3.27+, Ninja, a C++20 compiler, CUDA 12.x for the `cuda` presets, and [uv](https://docs.astral.sh/uv/).
Published benchmark numbers come from native Ubuntu with locked clocks.
