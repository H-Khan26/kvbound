"""Qualcomm AI Hub baseline: Llama 3.2 1B Instruct on the Hexagon NPU of a hosted Snapdragon device.

`qai-hub-models export llama_v3_2_1b_instruct` (run in its own environment, see below) splits
the model into three parts, compiles a 128-token prompt graph and a 1-token generation graph of
each part at one context length, links each part's two graphs into one context binary, and
profiles every graph on a hosted device. This module turns those profile jobs into one summary.

  fetch      downloads the profile of each job, with the job's name, device and options, into a
             raw folder (one <job id>.profile.json each, and jobs.json). Job ids come from the
             command line or from the export's log, which prints every job it schedules.
  summarize  reads a raw folder and writes results/profiles/<stamp>-qaihub-<model>-<device>/
             summary.json, the stamp being the creation time of the newest profile job, so
             summarizing the same jobs again rewrites the same file.

Per graph the summary keeps AI Hub's estimated inference time and the spread of the profiled
iterations, memory, first and warm load time, how many layers, and how much of the per-layer
cost, each compute unit (NPU, GPU, CPU) took, the operator types of any layer that left the
NPU, and the share of the cost of the ten costliest operators.

  memory   process_peak_mib is the peak memory of the profiling process on the device, the
           profiler's own included; inference_memory_peak_range_mib is how far inference
           raised it, as a range. Weights in the mapped context binary show in neither.
  cost     HTP profiles give each layer's cost in cycles, and a graph's cycles add up to
           several times its latency at any NPU clock: they count work on parallel units.
           The shares are relative costs, not shares of the graph's latency.

From the graphs it derives:

  decode_tok_s    1 / (sum of the three token-graph latencies). The latency is AI Hub's
                  estimate, which in these profiles is the fastest profiled iteration; the
                  same rates from the median iterations are under from_median_iteration.
  prefill_tok_s   prompt length / (sum of the three prompt-graph latencies)
  ttft_ms[n]      ceil(n / prompt length) prompt passes. Every pass has the same static shape
                  and attends over the whole context buffer, so its cost does not depend on
                  how far into the context it runs.

These are graph latencies only: they leave out the runtime's work between parts and between
steps (moving hidden states and KV outputs from one graph to the next, writing new KV entries
into the cache buffers, the attention mask and RoPE inputs), sampling, and detokenization.
Qualcomm's own figures, measured end to end with the Genie runtime, are kept beside them.

The export itself needs qai-hub-models, whose pins conflict with this repo's torch, so it runs
in a separate environment; this module only needs the AI Hub client and a configured token:
uv sync --locked --extra cu128 --group qai-hub

Usage: python -m kvbound.baselines.qai_hub fetch --log <export log> --raw <folder>
       python -m kvbound.baselines.qai_hub summarize --raw <folder>
"""

import argparse
import json
import math
import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kvbound.paths import REPO_ROOT

PROFILES = REPO_ROOT / "results" / "profiles"
MODEL = {
    "name": "llama-3.2-1b-instruct",
    "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
    "family": "llama",
    "qai_hub_model_id": "llama_v3_2_1b_instruct",
}
# Graph names of the export: prompt_ar128_cl4096_2_of_3 is part 2 of 3, 128 tokens per pass,
# 4,096 positions of context.
GRAPH_NAME = re.compile(
    r"(?P<kind>prompt|token)_ar(?P<seq>\d+)_cl(?P<ctx>\d+)_(?P<part>\d+)_of_(?P<parts>\d+)"
)
# The precision in a job name, as in llama_v3_2_1b_instruct_w4a16_...
PRECISION = re.compile(r"_(?P<p>w\d+(?:a\d+)?)_")
# What the AI Hub client prints for each job it schedules.
SCHEDULED = re.compile(r"Scheduled (?P<type>\w+) job \((?P<id>\w+)\) successfully")
# The QAIRT version qai-hub-models prints at the end of an export.
QAIRT = re.compile(r"^qairt: (?P<v>\S+)", re.MULTILINE)
TTFT_PROMPTS = (128, 1024)
# Operators listed by name in each graph's cost breakdown; the rest are summed as "other".
TOP_OPS = 10
# Qualcomm's figures for this model at context 4,096, W4A16, measured with the Genie and GenieX
# runtimes (qai-hub-models 0.67.0, models/llama_v3_2_1b_instruct/perf.yaml).
PUBLISHED = {
    "source": "qai-hub-models 0.67.0, models/llama_v3_2_1b_instruct/perf.yaml",
    "context_length": 4096,
    "precision": "w4a16",
    "devices": {
        "Samsung Galaxy S25": {
            "genie": {"decode_tok_s": 67.22, "prefill_tok_s": 1782.53},
            "geniex_qairt": {"decode_tok_s": 46.04, "prefill_tok_s": 2529.58},
        },
        "Samsung Galaxy S26": {
            "genie": {"decode_tok_s": 65.24, "prefill_tok_s": 1626.02},
            "geniex_qairt": {"decode_tok_s": 48.23, "prefill_tok_s": 2897.87},
        },
    },
}
MIB = 1 << 20


def parse_export_log(text: str) -> list[dict[str, str]]:
    """Every job an export log says it scheduled, in order: {"type", "id"}."""
    return [{"type": m["type"], "id": m["id"]} for m in SCHEDULED.finditer(text)]


def graph_of(job_name: str) -> dict[str, Any] | None:
    """The graph a profile job ran, from its name, or None for a job of no graph."""
    m = GRAPH_NAME.search(job_name)
    if m is None:
        return None
    return {
        "graph": m.group(0),
        "kind": m["kind"],
        "seq_len": int(m["seq"]),
        "context_length": int(m["ctx"]),
        "part": int(m["part"]),
        "parts": int(m["parts"]),
    }


def _ms(us: float | None) -> float | None:
    return None if us is None else us / 1000


def _mib(n_bytes: float | None) -> float | None:
    return None if n_bytes is None else n_bytes / MIB


def _spread(values_us: list[int]) -> dict[str, float] | None:
    if not values_us:
        return None
    ms = sorted(v / 1000 for v in values_us)
    return {"n": len(ms), "min": ms[0], "p50": statistics.median(ms), "max": ms[-1]}


def _range_mib(value: Any) -> list[float] | None:
    """A [lower, upper] byte range in MiB; AI Hub sends None for a stage it did not measure."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    return [value[0] / MIB, value[1] / MIB]


def op_stem(layer_name: str) -> str:
    """The operator a layer name stands for. HTP profiles type most layers as "node" and name
    them after the ONNX node: node_matmul_17 and node_matmul_18 are both "matmul"."""
    stem = re.sub(r"_converted_.*$", "", layer_name)
    stem = re.sub(r"^node_", "", stem)
    stem = re.sub(r"(_\d+)+$", "", stem)
    return stem.rstrip("_").lower()


def layer_units(layers: list[dict[str, Any]]) -> dict[str, Any]:
    """Layers per compute unit, the share of the per-layer cost each unit took, and the operator
    types of every layer that did not run on the NPU.

    The cost is execution_time where the profile has it; HTP profiles report each layer's
    execution_cycles and leave execution_time at 0, and then the cycles are the cost.
    """
    count = Counter(layer["compute_unit"] for layer in layers)
    measure = "execution_time"
    if not any(layer.get(measure) for layer in layers):
        measure = "execution_cycles"
    cost: Counter[str] = Counter()
    for layer in layers:
        cost[layer["compute_unit"]] += layer.get(measure, 0)
    total = sum(cost.values())
    off_npu: dict[str, Counter[str]] = {}
    for layer in layers:
        if layer["compute_unit"] != "NPU":
            off_npu.setdefault(layer["compute_unit"], Counter())[layer["type"]] += 1
    by_op: Counter[str] = Counter()
    for layer in layers:
        by_op[op_stem(layer["name"])] += layer.get(measure, 0)
    top = dict(by_op.most_common(TOP_OPS))
    if len(by_op) > TOP_OPS:
        top["other"] = total - sum(top.values())
    return {
        "layers": len(layers),
        "layers_by_unit": dict(count.most_common()),
        "cost_measure": measure if total else None,
        "cost_share_by_unit": {u: c / total for u, c in cost.most_common()} if total else {},
        "cost_share_by_op": {op: c / total for op, c in top.items()} if total else {},
        "off_npu_types": {u: dict(c.most_common()) for u, c in off_npu.items()},
    }


def graph_summary(profile: dict[str, Any]) -> dict[str, Any]:
    """What the summary keeps of one profile: times in ms, memory in MiB."""
    s = profile["execution_summary"]
    return {
        "latency_ms": _ms(s["estimated_inference_time"]),
        "iterations_ms": _spread(s.get("all_inference_times") or []),
        "process_peak_mib": _mib(s.get("estimated_inference_peak_memory")),
        "inference_memory_peak_range_mib": _range_mib(s.get("inference_memory_peak_range")),
        "first_load_ms": _ms(s.get("first_load_time")),
        "warm_load_ms": _ms(s.get("warm_load_time")),
        "first_load_memory_peak_range_mib": _range_mib(s.get("first_load_memory_peak_range")),
        **layer_units(profile["execution_detail"]),
    }


def derive(graphs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Token rates and time to first token from the per-graph latencies (see the module doc)."""
    passes: dict[str, list[dict[str, Any]]] = {"prompt": [], "token": []}
    for g in graphs.values():
        passes[g["kind"]].append(g)
    for kind, parts in passes.items():
        expected = [[i + 1, len(parts)] for i in range(len(parts))]
        if not parts or sorted([g["part"], g["parts"]] for g in parts) != expected:
            raise ValueError(f"the {kind} graphs do not cover every part once")
    prompt_len = passes["prompt"][0]["seq_len"]
    prompt_ms = sum(g["latency_ms"] for g in passes["prompt"])
    token_ms = sum(g["latency_ms"] for g in passes["token"])
    derived = {
        "prompt_len": prompt_len,
        "context_length": passes["token"][0]["context_length"],
        "prompt_pass_ms": prompt_ms,
        "token_pass_ms": token_ms,
        "decode_tok_s": 1000 / token_ms,
        "prefill_tok_s": prompt_len * 1000 / prompt_ms,
        "ttft_ms": {str(n): math.ceil(n / prompt_len) * prompt_ms for n in TTFT_PROMPTS},
        "largest_process_peak_mib": max(g["process_peak_mib"] or 0 for g in graphs.values()),
    }
    # In the recorded profiles AI Hub's estimate is the fastest iteration; these are the same
    # rates from each graph's median iteration, when every graph has its iterations.
    if all(g.get("iterations_ms") for g in graphs.values()):
        prompt_p50 = sum(g["iterations_ms"]["p50"] for g in passes["prompt"])
        token_p50 = sum(g["iterations_ms"]["p50"] for g in passes["token"])
        derived["from_median_iteration"] = {
            "prompt_pass_ms": prompt_p50,
            "token_pass_ms": token_p50,
            "decode_tok_s": 1000 / token_p50,
            "prefill_tok_s": prompt_len * 1000 / prompt_p50,
            "ttft_ms": {str(n): math.ceil(n / prompt_len) * prompt_p50 for n in TTFT_PROMPTS},
        }
    return derived


def compare(derived: dict[str, Any], device: str) -> dict[str, Any] | None:
    """The derived rates as fractions of Qualcomm's published ones for the same device."""
    published = PUBLISHED["devices"].get(device)  # type: ignore[attr-defined]
    if published is None:
        return None
    return {
        runtime: {
            key: {"published": rates[key], "derived_over_published": derived[key] / rates[key]}
            for key in ("decode_tok_s", "prefill_tok_s")
        }
        for runtime, rates in published.items()
    }


def summarize(raw: Path) -> dict[str, Any]:
    """The summary of a raw folder written by fetch."""
    jobs = json.loads((raw / "jobs.json").read_text(encoding="utf-8"))
    graphs: dict[str, dict[str, Any]] = {}
    devices = set()
    for job in jobs:
        graph = graph_of(job["name"]) if job["type"] in ("inference", "profile") else None
        if graph is None or job["status"] != "SUCCESS":
            continue
        profile = json.loads((raw / f"{job['id']}.profile.json").read_text(encoding="utf-8"))
        devices.add((job["device"], job["device_os"]))
        graphs[graph["graph"]] = {
            **graph,
            "name": job["name"],
            "job_id": job["id"],
            "url": job["url"],
            "created_utc": job["created_utc"],
            **graph_summary(profile),
        }
    if len(devices) != 1:
        raise ValueError(f"expected profile jobs of one device, found {sorted(devices)}")
    device, device_os = devices.pop()
    derived = derive(graphs)
    export_file = raw / "export.json"
    export = json.loads(export_file.read_text(encoding="utf-8")) if export_file.exists() else {}
    precisions = {m["p"] for g in graphs.values() if (m := PRECISION.search(g["name"]))}
    return {
        "kind": "qaihub.llm_profile",
        "model": MODEL,
        "device": {"name": device, "os": device_os},
        "params": {
            "precision": precisions.pop() if len(precisions) == 1 else None,
            "prompt_len": derived["prompt_len"],
            "context_length": derived["context_length"],
            "export_command": export.get("command"),
            "qairt": export.get("qairt"),
        },
        "derived": derived,
        "published": compare(derived, device),
        "published_source": PUBLISHED["source"],
        "graphs": dict(sorted(graphs.items(), key=lambda kv: (kv[1]["kind"], kv[1]["part"]))),
        "jobs": [{k: job[k] for k in ("type", "id", "name", "url", "status")} for job in jobs],
        "not_covered": [
            "runtime work between the three parts and between steps",
            "writing each step's new KV entries into the cache buffers",
            "attention mask and RoPE inputs prepared on the host",
            "sampling and detokenization",
        ],
    }


def summary_path(summary: dict[str, Any]) -> Path:
    """results/profiles/<newest profile job>-qaihub-<model>-<device>/summary.json"""
    newest = max(g["created_utc"] for g in summary["graphs"].values())
    stamp = datetime.fromisoformat(newest).strftime("%Y%m%dT%H%M%SZ")
    device = re.sub(r"[^a-z0-9]+", "-", summary["device"]["name"].lower()).strip("-")
    return PROFILES / f"{stamp}-qaihub-{summary['model']['name']}-{device}" / "summary.json"


def fetch(job_ids: list[str], raw: Path) -> list[dict[str, Any]]:
    """Downloads every job's metadata, and each successful inference job's profile, into raw."""
    import qai_hub as hub  # noqa: PLC0415  (the optional qai-hub group)

    raw.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job_id in job_ids:
        job = hub.get_job(job_id)
        device = getattr(job, "device", None)
        entry = {
            "type": type(job).__name__.removesuffix("Job").lower(),
            "id": job.job_id,
            "name": job.name,
            "url": job.url,
            "status": job.get_status().code,
            # The client gives the submission time as a naive local time.
            "created_utc": job.date.astimezone(UTC).isoformat(timespec="seconds"),
            "device": device.name if device else None,
            "device_os": device.os if device else None,
            "options": job.options,
        }
        if isinstance(job, hub.InferenceJob) and entry["status"] == "SUCCESS":
            job.download_profile(str(raw / f"{job.job_id}.profile.json"))
        jobs.append(entry)
        print(f"{entry['type']:9} {entry['id']} {entry['status']:8} {entry['name']}")
    (raw / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return jobs


def print_summary(summary: dict[str, Any]) -> None:
    print(f"{summary['device']['name']}, context {summary['params']['context_length']}")
    for name, g in summary["graphs"].items():
        units = ", ".join(f"{u} {n}" for u, n in g["layers_by_unit"].items())
        print(
            f"  {name:28} {g['latency_ms']:8.2f} ms  peak {g['process_peak_mib'] or 0:7.1f} MiB  "
            f"first load {g['first_load_ms'] or 0:8.1f} ms  layers: {units}"
        )
    d = summary["derived"]
    print(f"  decode  {d['decode_tok_s']:7.1f} tokens/s ({d['token_pass_ms']:.2f} ms per token)")
    print(f"  prefill {d['prefill_tok_s']:7.1f} tokens/s ({d['prompt_pass_ms']:.2f} ms per pass)")
    for n, ms in d["ttft_ms"].items():
        print(f"  TTFT {n:>5} tokens {ms:8.1f} ms")
    if median := d.get("from_median_iteration"):
        print(
            f"  from the median iterations: {median['decode_tok_s']:.1f} / "
            f"{median['prefill_tok_s']:.1f} tokens/s decode / prefill"
        )
    for runtime, rates in (summary["published"] or {}).items():
        print(
            f"  {runtime}: published {rates['decode_tok_s']['published']:.1f} / "
            f"{rates['prefill_tok_s']['published']:.1f} tokens/s decode / prefill; derived is "
            f"{rates['decode_tok_s']['derived_over_published']:.2f}x / "
            f"{rates['prefill_tok_s']['derived_over_published']:.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the profiles of an export's jobs")
    f.add_argument("--raw", type=Path, required=True, help="folder for the downloads")
    f.add_argument("--log", type=Path, help="export log; its scheduled jobs are fetched")
    f.add_argument("--jobs", nargs="*", default=[], help="job ids, in addition to the log's")
    f.add_argument("--export-command", help="kept in the summary as given (export.json)")
    f.add_argument("--qairt", help="QAIRT version of the jobs, if the log does not print it")
    s = sub.add_parser("summarize", help="write the summary of a raw folder")
    s.add_argument("--raw", type=Path, required=True)
    s.add_argument("--out", type=Path, help="summary file; default under results/profiles/")
    args = parser.parse_args()

    if args.command == "fetch":
        ids = list(args.jobs)
        log = args.log.read_text(encoding="utf-8", errors="replace") if args.log else ""
        ids += [j["id"] for j in parse_export_log(log) if j["id"] not in ids]
        if not ids:
            raise SystemExit("no job ids: pass --log or --jobs")
        fetch(ids, args.raw)
        export_file = args.raw / "export.json"
        export = json.loads(export_file.read_text("utf-8")) if export_file.exists() else {}
        if args.export_command:
            export["command"] = args.export_command
        if qairt := args.qairt or ((m := QAIRT.search(log)) and m["v"]):
            export["qairt"] = qairt
        export_file.write_text(json.dumps(export, indent=2) + "\n", "utf-8")
        return

    summary = summarize(args.raw)
    out = args.out or summary_path(summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
