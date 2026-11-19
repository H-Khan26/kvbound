"""Profiles llama.cpp decode steps with Nsight Systems and keeps the summaries.

Runs the pinned release's llama-completion under `nsys profile` on one GGUF format and writes
the per-kernel and per-API-call summaries into results/profiles/<run>/, as kvbound.profile does
for the engine. The prompt is one word, so every evaluation after it is a decode step, and the
per-step figures divide by the number of those evaluations as llama.cpp reports it. The prompt's
own kernels are few and keep their own names (mul_mat_f, and the instantiations with two
columns).

llama.cpp replays a decode step from a CUDA graph, which hides its kernels from the kernel
summary; --no-graph sets GGML_CUDA_DISABLE_GRAPHS to see them. Tracing slows the run, so the
wall time per step in the summary is not a benchmark figure.

Usage: python -m kvbound.baselines.llamacpp_profile --model tinyllama-1.1b --weights Q4_0
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from kvbound import profile
from kvbound.baselines import llamacpp
from kvbound.models import load_spec

# llama_perf_context_print's line for the evaluations after the prompt.
EVAL_LINE = re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) runs")


def kernel_name(signature: str) -> str:
    """A demangled kernel signature without its return type and parameter list: the kernel's
    name with its template arguments, which in ggml tell the variants of one kernel apart."""
    name = signature.removeprefix("void ")
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            return name[:i]
    return name


def decode_evals(output: str) -> tuple[int, float]:
    """The number of decode evaluations and their wall time in ms, from llama.cpp's report."""
    found = EVAL_LINE.search(output)
    if not found:
        raise SystemExit(f"no eval time line in llama-completion's output:\n{output[-2000:]}")
    return int(found.group(2)), float(found.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=llamacpp.FORMATS, default="F16")
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    spec = load_spec(args.model)
    gguf = llamacpp.require_gguf(spec.name, args.weights)
    nsys = profile.find_nsys()
    stem = "graph" if args.graph else "eager"
    env = llamacpp.runtime_env()
    if not args.graph:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "profile"
        work.mkdir()
        report = work / f"{stem}.nsys-rep"
        # nsys does not pass the program's output on, so llama.cpp's report is read from its log.
        log = Path(tmp) / "llama-completion.log"
        command = [
            nsys, "profile", "--trace", "cuda", "--output", str(report),
            "--force-overwrite", "true",
            str(llamacpp.exe("llama-completion")), "-m", str(gguf),
            "-p", "The", "-n", str(args.new_tokens), "--ignore-eos", "-no-cnv",
            "--temp", "0", "-c", str(args.new_tokens + 16), "-ngl", "all", "-fa", "on",
            "--no-warmup", "--no-display-prompt", "--log-file", str(log),
        ]  # fmt: skip
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
        )
        logged = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if done.returncode != 0:
            output = done.stdout + done.stderr
            raise SystemExit(
                f"llama-completion under nsys failed:\n{output[-4000:]}\n{logged[-4000:]}"
            )
        steps, wall_ms = decode_evals(logged)
        summary, out = profile.keep(
            nsys,
            work,
            stem,
            steps,
            f"llamacpp-decode-{spec.name}-{args.weights}-{stem}",
            {
                "engine": llamacpp.engine_entry("llama-completion under nsys"),
                "weights": args.weights,
                "traced_wall_ms_per_step": wall_ms / steps,
            },
            kernel_name,
        )

    print(f"llama.cpp {llamacpp.TAG} {spec.name} {args.weights}, {steps} decode steps, {stem}")
    print(f"GPU time in kernels: {summary['gpu_ms_per_step_total']:.2f} ms per step")
    for name, k in summary["gpu_ms_per_step"].items():
        print(f"  {k['ms']:7.3f} ms  {k['launches']:6.1f} launches  {name}")
    for name, call in summary["api_per_step"].items():
        print(f"  {name:<28} {call['calls']:7.1f} calls  {call['ms']:6.3f} ms")
    print(out)


if __name__ == "__main__":
    main()
