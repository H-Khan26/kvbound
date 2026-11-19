"""Parsing of AI Hub profile jobs and the token rates derived from them, on a committed sample."""

import json
import math
import shutil
from pathlib import Path

import pytest

from kvbound.baselines import qai_hub

SAMPLE = Path(__file__).parent / "data" / "qai_hub"


def graph(kind, part, latency_ms, seq_len=None, parts=3, peak=100.0):
    seq_len = seq_len or (128 if kind == "prompt" else 1)
    return {
        "kind": kind,
        "part": part,
        "parts": parts,
        "seq_len": seq_len,
        "context_length": 4096,
        "latency_ms": latency_ms,
        "process_peak_mib": peak,
    }


def test_parse_export_log_finds_every_scheduled_job():
    log = (
        "Uploading model\n"
        "Scheduled compile job (jp4abc123) successfully. To see the status and results:\n"
        "    https://workbench.aihub.qualcomm.com/jobs/jp4abc123/\n"
        "Scheduled link job (jgxyz789) successfully. To see the status and results:\n"
        "Scheduled inference job (j5q0w1e2) successfully. To see the status and results:\n"
    )
    assert qai_hub.parse_export_log(log) == [
        {"type": "compile", "id": "jp4abc123"},
        {"type": "link", "id": "jgxyz789"},
        {"type": "inference", "id": "j5q0w1e2"},
    ]
    assert qai_hub.QAIRT.search(log + "qairt: 2.51.0.260929181845\n")["v"] == "2.51.0.260929181845"


def test_graph_of_reads_part_and_shape_from_the_job_name():
    name = "llama_v3_2_1b_instruct_w4a16_DEFAULT_W4A16_part2_of_3_prompt_ar128_cl4096_2_of_3"
    g = qai_hub.graph_of(name)
    assert g == {
        "graph": "prompt_ar128_cl4096_2_of_3",
        "kind": "prompt",
        "seq_len": 128,
        "context_length": 4096,
        "part": 2,
        "parts": 3,
    }
    assert qai_hub.graph_of("llama_v3_2_1b_instruct_w4a16_DEFAULT_W4A16_part1_of_3") is None


def test_graph_summary_converts_units_and_counts_compute_units():
    profile = {
        "execution_summary": {
            "estimated_inference_time": 4500,
            "estimated_inference_peak_memory": 3 * qai_hub.MIB,
            "first_load_time": 250_000,
            "warm_load_time": 90_000,
            "inference_memory_peak_range": [qai_hub.MIB, 2 * qai_hub.MIB],
            "all_inference_times": [4600, 4400, 4500, 5000],
        },
        "execution_detail": [
            {"name": "a", "type": "Conv2d", "compute_unit": "NPU", "execution_time": 60},
            {"name": "b", "type": "Conv2d", "compute_unit": "NPU", "execution_time": 20},
            {"name": "c", "type": "Gather", "compute_unit": "CPU", "execution_time": 20},
        ],
    }
    s = qai_hub.graph_summary(profile)
    assert s["latency_ms"] == 4.5
    assert s["iterations_ms"] == {"n": 4, "min": 4.4, "p50": 4.55, "max": 5.0}
    assert s["process_peak_mib"] == 3.0
    assert s["inference_memory_peak_range_mib"] == [1.0, 2.0]
    assert (s["first_load_ms"], s["warm_load_ms"]) == (250.0, 90.0)
    assert s["layers"] == 3
    assert s["layers_by_unit"] == {"NPU": 2, "CPU": 1}
    assert s["cost_measure"] == "execution_time"
    assert s["cost_share_by_unit"] == pytest.approx({"NPU": 0.8, "CPU": 0.2})
    assert s["off_npu_ops"] == {"CPU": {"Gather": 1}}


def htp_layer(name, cycles, unit="NPU", kind="node"):
    """A layer as HTP profiles report it: no time, only cycles."""
    return {
        "name": name,
        "type": kind,
        "compute_unit": unit,
        "execution_time": 0,
        "execution_cycles": cycles,
    }


def test_layer_units_fall_back_to_cycles_when_times_are_zero():
    layers = [
        htp_layer("Input", 0, kind="Input"),
        htp_layer("node_matmul_17", 200),
        htp_layer("node_matmul_18", 100),
        htp_layer("node_slice_3", 100, unit="GPU"),
    ]
    u = qai_hub.layer_units(layers)
    assert u["cost_measure"] == "execution_cycles"
    assert u["cost_share_by_unit"] == pytest.approx({"NPU": 0.75, "GPU": 0.25})
    assert u["cost_share_by_op"] == pytest.approx({"matmul": 0.75, "slice": 0.25, "input": 0.0})
    assert u["off_npu_ops"] == {"GPU": {"slice": 1}}


def test_op_stem_names_the_operator():
    assert qai_hub.op_stem("node_matmul_17") == "matmul"
    assert qai_hub.op_stem("node_Softmax_3") == "softmax"
    assert qai_hub.op_stem("rms_norm_node__") == "rms_norm_node"
    assert qai_hub.op_stem("permute_40_converted_QNN_DATATYPE_UFIXED_POINT_8") == "permute"
    assert qai_hub.op_stem("Output") == "output"


def test_graph_summary_tolerates_missing_measurements():
    profile = {
        "execution_summary": {
            "estimated_inference_time": 1000,
            "inference_memory_peak_range": None,
        },
        "execution_detail": [{"name": "a", "type": "MatMul", "compute_unit": "NPU"}],
    }
    s = qai_hub.graph_summary(profile)
    assert s["latency_ms"] == 1.0
    assert s["iterations_ms"] is None
    assert s["inference_memory_peak_range_mib"] is None
    assert s["cost_share_by_unit"] == {}
    assert s["cost_measure"] is None


def test_fetch_ids_merge_the_sources_once_each():
    log = "Scheduled compile job (c1) successfully\nScheduled inference job (p1) successfully\n"
    earlier = {"jobs": [{"id": "p1"}, {"id": "p2"}]}
    assert qai_hub.fetch_ids(["x9", "c1"], log, earlier) == ["x9", "c1", "p1", "p2"]
    assert qai_hub.fetch_ids([], "", None) == []


def test_export_meta_prefers_the_command_line_then_the_log_then_the_summary():
    earlier = {"params": {"export_command": "export a", "qairt": "2.50.0", "note": None}}
    assert qai_hub.export_meta(earlier, "", None, None, None) == {
        "command": "export a",
        "qairt": "2.50.0",
    }
    assert qai_hub.export_meta(earlier, "qairt: 2.51.0\n", None, None, "second device") == {
        "command": "export a",
        "qairt": "2.51.0",
        "note": "second device",
    }
    assert qai_hub.export_meta(None, "qairt: 2.51.0\n", "export b", "2.52.0", None) == {
        "command": "export b",
        "qairt": "2.52.0",
    }


def test_derive_sums_the_parts():
    graphs = {
        "p1": graph("prompt", 1, 20.0),
        "p2": graph("prompt", 2, 30.0),
        "p3": graph("prompt", 3, 30.0, peak=400.0),
        "t1": graph("token", 1, 4.0),
        "t2": graph("token", 2, 5.0),
        "t3": graph("token", 3, 6.0),
    }
    d = qai_hub.derive(graphs)
    assert d["prompt_pass_ms"] == 80.0
    assert d["token_pass_ms"] == 15.0
    assert d["decode_tok_s"] == pytest.approx(1000 / 15)
    assert d["prefill_tok_s"] == pytest.approx(128 * 1000 / 80)
    # A 1,024-token prompt is eight 128-token passes; 128 tokens is one.
    assert d["ttft_ms"] == {"128": 80.0, "1024": 640.0}
    assert d["largest_process_peak_mib"] == 400.0
    assert (d["prompt_len"], d["context_length"]) == (128, 4096)


def test_derive_from_the_median_iterations_only_when_every_graph_has_them():
    graphs = {f"p{i}": graph("prompt", i, 10.0) for i in (1, 2, 3)}
    graphs |= {f"t{i}": graph("token", i, 2.0) for i in (1, 2, 3)}
    assert "from_median_iteration" not in qai_hub.derive(graphs)
    for g in graphs.values():
        g["iterations_ms"] = {"n": 5, "min": g["latency_ms"], "p50": 2 * g["latency_ms"]}
    median = qai_hub.derive(graphs)["from_median_iteration"]
    assert (median["prompt_pass_ms"], median["token_pass_ms"]) == (60.0, 12.0)
    assert median["decode_tok_s"] == pytest.approx(1000 / 12)
    assert median["ttft_ms"]["1024"] == 8 * 60.0


def test_derive_rounds_a_partial_prompt_pass_up():
    graphs = {f"p{i}": graph("prompt", i, 10.0, seq_len=100) for i in (1, 2, 3)}
    graphs |= {f"t{i}": graph("token", i, 1.0) for i in (1, 2, 3)}
    assert qai_hub.derive(graphs)["ttft_ms"]["1024"] == math.ceil(1024 / 100) * 30.0


def test_derive_refuses_a_missing_part():
    graphs = {f"p{i}": graph("prompt", i, 10.0) for i in (1, 2, 3)}
    graphs |= {"t1": graph("token", 1, 1.0), "t3": graph("token", 3, 1.0)}
    with pytest.raises(ValueError, match="token graphs"):
        qai_hub.derive(graphs)


def test_summarize_the_committed_sample(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(SAMPLE, raw)
    summary = qai_hub.summarize(raw)
    assert summary["device"]["name"] == "Samsung Galaxy S25"
    assert summary["params"]["precision"] == "w4a16"
    assert list(summary["graphs"]) == [
        *(f"prompt_ar128_cl4096_{i}_of_3" for i in (1, 2, 3)),
        *(f"token_ar1_cl4096_{i}_of_3" for i in (1, 2, 3)),
    ]
    jobs = json.loads((raw / "jobs.json").read_text(encoding="utf-8"))
    profiles = {
        j["id"]: json.loads((raw / f"{j['id']}.profile.json").read_text(encoding="utf-8"))
        for j in jobs
        if (raw / f"{j['id']}.profile.json").exists()
    }
    token_us = sum(
        profiles[g["job_id"]]["execution_summary"]["estimated_inference_time"]
        for g in summary["graphs"].values()
        if g["kind"] == "token"
    )
    assert summary["derived"]["decode_tok_s"] == pytest.approx(1e6 / token_us)
    published = summary["published"]["genie"]["decode_tok_s"]
    assert published["derived_over_published"] == pytest.approx(
        summary["derived"]["decode_tok_s"] / published["published"]
    )
    path = qai_hub.summary_path(summary)
    assert path.parent.parent == qai_hub.PROFILES
    assert path.parent.name.endswith("-qaihub-llama-3.2-1b-instruct-samsung-galaxy-s25")
