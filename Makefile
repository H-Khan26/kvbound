.PHONY: help setup build test lint format typecheck clean

# torch wheel: cu128 on GPU machines, cpu in CI and on machines without a GPU
TORCH ?= cu128
# CMake preset: cpu | cuda | cuda-relwithdebinfo
PRESET ?= cpu

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

build:            ## configure and build C++/CUDA targets (PRESET=cpu|cuda|cuda-relwithdebinfo)
	$(VSENV) cmake --preset $(PRESET)
	$(VSENV) cmake --build --preset $(PRESET)

test: build       ## C++ tests via ctest, then Python tests via pytest
	$(VSENV) ctest --preset $(PRESET)
	uv run pytest; s=$$?; [ $$s -eq 0 ] || [ $$s -eq 5 ]  # 5 = no tests collected

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
