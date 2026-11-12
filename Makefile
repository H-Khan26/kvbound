.PHONY: help setup doctor build test test-gpu test-real sanitize fetch fixture golden bench-hf bench serve profile microbench report predict lint format typecheck clean

# torch wheel: cu128 on GPU machines, cpu in CI and on machines without a GPU
TORCH ?= cu128
# CMake preset: cpu | cuda | cuda-relwithdebinfo
PRESET ?= cpu
# Model name from configs/models/
MODEL ?= tinyllama-1.1b
# Device name from configs/devices/
DEVICE ?= rtx3060ti

# On Windows, recipes need a POSIX sh (Git Bash) and cmake/ctest run inside the MSVC environment.
ifeq ($(OS),Windows_NT)
SHELL := sh.exe
VSENV := cmd //c "scripts\vsenv.cmd"
endif

CXX_FILES = git ls-files '*.h' '*.hpp' '*.cpp' '*.cu' '*.cuh'

help:             ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:            ## install the locked Python env (TORCH=cu128|cpu) and pre-commit hooks
	uv sync --locked --extra $(TORCH)
	uv run pre-commit install

doctor:           ## check GPU, toolchain, profilers and clock state on this machine
	uv run python -m kvbound.doctor

build:            ## configure and build C++/CUDA targets (PRESET=cpu|cuda|cuda-relwithdebinfo)
	$(VSENV) cmake --preset $(PRESET)
	$(VSENV) cmake --build --preset $(PRESET)

test: build       ## C++ tests via ctest, then Python tests via pytest
	$(VSENV) ctest --preset $(PRESET)
	uv run pytest; s=$$?; [ $$s -eq 0 ] || [ $$s -eq 5 ]  # 5 = no tests collected

test-gpu:         ## build the cuda preset, then kernel and engine tests on the fixture (needs a GPU)
	$(MAKE) build PRESET=cuda
	uv run pytest -m "gpu and not slow"

test-real:        ## reference vs Hugging Face and engine vs reference on fetched checkpoints (minutes)
	uv run pytest -m slow

sanitize:         ## engine under compute-sanitizer (memcheck, racecheck, initcheck) on the fixture
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.sanitize

fetch:            ## download the pinned checkpoint into data/models/ (MODEL=...)
	uv run python -m kvbound.models fetch $(MODEL)

fixture:          ## regenerate the committed tiny-llama fixture and its goldens
	uv run python -m kvbound.fixture

golden:           ## dump real-model golden tensors into data/golden/ (MODEL=...)
	uv run python -m kvbound.golden $(MODEL)

bench-hf:         ## Hugging Face baseline: batch 1, batch 8, and a long prompt (MODEL=...)
	uv run python -m kvbound.baselines.hf --model $(MODEL) --batch 1 --prompt-len 128
	uv run python -m kvbound.baselines.hf --model $(MODEL) --batch 8 --prompt-len 128
	uv run python -m kvbound.baselines.hf --model $(MODEL) --batch 1 --prompt-len 1024

bench:            ## kvbound engine: batch 1 decode and a long prompt, like bench-hf (MODEL=...)
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.bench --model $(MODEL) --prompt-len 128 --profile-ops
	uv run python -m kvbound.bench --model $(MODEL) --prompt-len 128 --no-graph
	uv run python -m kvbound.bench --model $(MODEL) --prompt-len 1024

serve:            ## static batching on the three workloads, 1 and 8 slots (MODEL=...)
	$(MAKE) build PRESET=cuda
	for w in fixed uniform longtail; do for s in 1 8; do \
	  uv run python -m kvbound.serve --model $(MODEL) --workload $$w --slots $$s || exit 1; \
	done; done

profile:          ## Nsight Systems decode profile, with and without CUDA graphs (MODEL=...)
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.profile --model $(MODEL) --no-graph
	uv run python -m kvbound.profile --model $(MODEL)

microbench:       ## build the cuda preset, then run bandwidth, latency, GEMM and KV-layout microbenchmarks
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.microbench --preset cuda

report:           ## roofline, bandwidth and latency figures from the records (MODEL=... DEVICE=...)
	uv run python -m kvbound.report --device $(DEVICE) --model $(MODEL)

predict:          ## latency bounds from the perf model (MODEL=... DEVICE=...)
	uv run python -m kvbound.perfmodel --model $(MODEL) --device $(DEVICE)

lint:             ## ruff lint + format check, clang-format check
	uv run ruff check .
	uv run ruff format --check .
	$(CXX_FILES) | xargs -r uv run clang-format --dry-run --Werror

format:           ## auto-format Python and C++/CUDA
	uv run ruff format .
	uv run ruff check --fix .
	$(CXX_FILES) | xargs -r uv run clang-format -i

typecheck:        ## mypy on python/
	uv run mypy

clean:
	rm -rf build .pytest_cache .mypy_cache .ruff_cache
