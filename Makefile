.PHONY: help setup doctor build test test-gpu test-real sanitize fetch fixture golden bench-hf bench serve serve-load profile counters step-counters microbench quantize eval precision eval-asr bench-asr qaihub report predict lint format typecheck clean

# torch wheel: cu128 on GPU machines, cpu in CI and on machines without a GPU
TORCH ?= cu128
# CMake preset: cpu | cuda | cuda-relwithdebinfo
PRESET ?= cpu
# Model name from configs/models/
MODEL ?= tinyllama-1.1b
# Device name from configs/devices/
DEVICE ?= rtx3060ti
# Arrival rates of serve-load, in requests per second
RATES ?= 2 4 8 16
# Weight variant and KV dtype for `make eval`:
# fp16 | int8_pc | int4_g | int4_g.mse | int4_g.awq | int4_g.awq-noclip | int4_g.gptq |
# int4_g.awq-gptq | int4_g.awq-noclip-gptq, fp16 | int8
WEIGHTS ?= fp16
KV ?= fp16
# The weights:kv pairs `make precision` measures
PRECISIONS ?= fp16:fp16 fp16:int8 int8_pc:fp16 int8_pc:int8 int4_g:fp16 int4_g:int8 int4_g.awq:fp16 int4_g.awq:int8
# Speech stack and beam width of `make eval-asr`: hf | faster-whisper | kvbound, 1 for greedy
BACKEND ?= hf
BEAMS ?= 1

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

fixture:          ## regenerate the committed tiny-llama and tiny-whisper fixtures and their goldens
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

# The KV pool of every serve run: 8 sequences of 1,024 tokens.
SERVE = uv run python -m kvbound.serve --model $(MODEL) --pool-seqs 8
# The paged design: the same pool in 16-token blocks, shared by up to this many sequences.
PAGED = --policy continuous --kv-block-size 16 --slots $(PAGED_SLOTS)
PAGED_SLOTS ?= 64

serve:            ## the three workloads in one KV pool: static, continuous over slots, paged (MODEL=...)
	$(MAKE) build PRESET=cuda
	for w in fixed uniform longtail; do \
	  uv run python -m kvbound.serve --model $(MODEL) --workload $$w --slots 1 || exit 1; \
	  $(SERVE) --workload $$w --slots 8 || exit 1; \
	  $(SERVE) --workload $$w --slots 8 --policy continuous || exit 1; \
	  $(SERVE) --workload $$w $(PAGED) || exit 1; \
	done

serve-load:       ## static against paged as longtail requests arrive at a growing rate (MODEL=...)
	$(MAKE) build PRESET=cuda
	for r in $(RATES); do \
	  $(SERVE) --workload longtail --rate $$r --slots 8 || exit 1; \
	  $(SERVE) --workload longtail --rate $$r $(PAGED) || exit 1; \
	done

profile:          ## Nsight Systems decode profile, with and without CUDA graphs (MODEL=...)
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.profile --model $(MODEL) --no-graph
	uv run python -m kvbound.profile --model $(MODEL)

counters:         ## hardware counters of the attention kernel on each KV layout, via Nsight Compute
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.counters

step-counters:    ## hardware counters of every kernel of a prefill and a decode step (MODEL=... WEIGHTS=...)
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.step_counters --model $(MODEL) --weights $(WEIGHTS) --phase decode
	uv run python -m kvbound.step_counters --model $(MODEL) --weights $(WEIGHTS) --phase prefill --prompt-len 1024

microbench:       ## build the cuda preset, then run bandwidth, latency, GEMM and KV-layout microbenchmarks
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.microbench --preset cuda

quantize:         ## quantized checkpoints: INT8 per channel, INT4 in groups of 128, and INT4 with AWQ scales (MODEL=...)
	uv run python -m kvbound.quantize $(MODEL) --scheme int8_pc
	uv run python -m kvbound.quantize $(MODEL) --scheme int4_g --group-size 128
	uv run python -m kvbound.quantize $(MODEL) --scheme int4_g --group-size 128 --awq

eval:             ## WikiText-2 perplexity of the engine at one precision (MODEL=... WEIGHTS=fp16 KV=fp16)
	$(MAKE) build PRESET=cuda
	uv run python -m kvbound.evals.perplexity --model $(MODEL) --weights $(WEIGHTS) --kv $(KV)

precision:        ## every pair in PRECISIONS: perplexity, per-layer error, decode speed, concurrency (MODEL=...)
	$(MAKE) build PRESET=cuda
	for p in $(PRECISIONS); do w=$${p%%:*}; k=$${p##*:}; \
	  uv run python -m kvbound.evals.perplexity --model $(MODEL) --weights $$w --kv $$k || exit 1; \
	  uv run python -m kvbound.evals.sites --model $(MODEL) --weights $$w --kv $$k || exit 1; \
	  uv run python -m kvbound.bench --model $(MODEL) --weights $$w --kv $$k --prompt-len 128 --profile-ops || exit 1; \
	  $(SERVE) --weights $$w --kv $$k --workload fixed --requests 96 $(PAGED) || exit 1; \
	done

# One speech stack on the 1,000-clip subset; `uv run python -m kvbound.evals.asr subset` fetches it.
ASR = uv run python -m kvbound.evals.asr eval --model $(MODEL)

eval-asr:         ## WER and real-time factor of one speech stack on the recitation subset (MODEL=... BACKEND=hf BEAMS=1)
	$(ASR) --backend $(BACKEND) --beams $(BEAMS)

bench-asr:        ## speech baselines, greedy and beam 5: Hugging Face on the GPU, faster-whisper on the GPU and the CPU (MODEL=...; needs `uv sync --group faster-whisper`)
	uv run python -c "import faster_whisper"
	for b in 1 5; do \
	  $(ASR) --backend hf --beams $$b || exit 1; \
	  $(ASR) --backend faster-whisper --compute-type float16 --beams $$b || exit 1; \
	  $(ASR) --backend faster-whisper --compute-type int8_float16 --beams $$b || exit 1; \
	  $(ASR) --backend faster-whisper --device cpu --compute-type int8 --beams $$b || exit 1; \
	done

# The jobs to download: a finished `qai-hub-models export`'s log, or a summary under
# results/profiles/ to rebuild. QAIHUB_RAW is a folder outside the repo for the downloads.
QAIHUB_LOG ?=
QAIHUB_SUMMARY ?=
QAIHUB_RAW ?=

qaihub:           ## re-download AI Hub profile jobs and summarize them into results/profiles/ (QAIHUB_RAW=... and QAIHUB_LOG=... or QAIHUB_SUMMARY=...)
	@test -n "$(QAIHUB_RAW)" || { echo "set QAIHUB_RAW to a folder for the downloads"; exit 1; }
	@test -n "$(QAIHUB_LOG)$(QAIHUB_SUMMARY)" || { echo "set QAIHUB_LOG or QAIHUB_SUMMARY"; exit 1; }
	uv run --with "qai-hub>=0.58.0" python -m kvbound.baselines.qai_hub fetch --raw "$(QAIHUB_RAW)" \
	  $(if $(QAIHUB_LOG),--log "$(QAIHUB_LOG)") $(if $(QAIHUB_SUMMARY),--summary "$(QAIHUB_SUMMARY)")
	uv run python -m kvbound.baselines.qai_hub summarize --raw "$(QAIHUB_RAW)"

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
