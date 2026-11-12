"""Synthetic workloads: a list of requests with prompt and output lengths drawn from a
distribution described in configs/workloads/<name>.toml.

A trace stores token ids, not text, so every engine that replays it processes the same tokens.
Prompt ids are random: with a fixed output length (EOS ignored) their content does not change
the amount of work.

Usage: python -m kvbound.workload <name> [--max-ctx 1024]   # prints a summary of the lengths
"""

import argparse
import statistics
import tomllib
from dataclasses import dataclass
from typing import Any

import numpy as np

from kvbound.paths import CONFIGS


@dataclass(frozen=True)
class Lengths:
    """A length distribution: fixed, uniform or lognormal, clipped to [min, max]."""

    dist: str
    min: int
    max: int
    median: float = 0.0  # lognormal: the median length
    sigma: float = 0.0  # lognormal: spread of log(length)

    def sample(self, rng: np.random.Generator, n: int) -> list[int]:
        if self.dist == "fixed":
            values = np.full(n, self.min, dtype=np.float64)
        elif self.dist == "uniform":
            values = rng.integers(self.min, self.max + 1, n).astype(np.float64)
        elif self.dist == "lognormal":
            values = rng.lognormal(np.log(self.median), self.sigma, n)
        else:
            raise ValueError(f"unknown length distribution {self.dist!r}")
        return [int(v) for v in np.clip(np.rint(values), self.min, self.max)]


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    n_requests: int
    seed: int
    prompt: Lengths
    output: Lengths


def load(name: str) -> Workload:
    path = CONFIGS / "workloads" / f"{name}.toml"
    if not path.exists():
        known = sorted(p.stem for p in (CONFIGS / "workloads").glob("*.toml"))
        raise FileNotFoundError(f"unknown workload {name!r}; known: {known}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Workload(
        name=raw["name"],
        description=raw["description"],
        n_requests=raw["n_requests"],
        seed=raw["seed"],
        prompt=Lengths(**raw["prompt"]),
        output=Lengths(**raw["output"]),
    )


def trace(workload: Workload, vocab_size: int, max_ctx: int) -> dict[str, Any]:
    """The requests of a workload, in arrival order, as kvb-bench reads them.

    A request longer than max_ctx keeps its prompt (cut to leave room for two tokens) and loses
    output length.
    """
    rng = np.random.default_rng(workload.seed)
    prompts = workload.prompt.sample(rng, workload.n_requests)
    outputs = workload.output.sample(rng, workload.n_requests)
    requests = []
    for i, (drawn_prompt, drawn_output) in enumerate(zip(prompts, outputs, strict=True)):
        n_prompt = min(drawn_prompt, max_ctx - 2)
        requests.append(
            {
                "id": i,
                "prompt": rng.integers(0, vocab_size, n_prompt).tolist(),
                "max_new_tokens": max(2, min(drawn_output, max_ctx - n_prompt)),
            }
        )
    return {"name": workload.name, "seed": workload.seed, "requests": requests}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--max-ctx", type=int, default=1024)
    args = parser.parse_args()
    workload = load(args.name)
    requests = trace(workload, 32000, args.max_ctx)["requests"]
    print(f"{workload.name}: {workload.description}")
    for label, values in (
        ("prompt", [len(r["prompt"]) for r in requests]),
        ("output", [r["max_new_tokens"] for r in requests]),
        ("total", [len(r["prompt"]) + r["max_new_tokens"] for r in requests]),
    ):
        print(
            f"  {label:<7} min {min(values):>4}  median {statistics.median(values):>6.0f}"
            f"  mean {statistics.fmean(values):>6.0f}  max {max(values):>4}"
        )


if __name__ == "__main__":
    main()
