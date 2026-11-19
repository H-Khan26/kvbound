"""The llama.cpp baseline: outside-stack records, fetching and conversion inputs, GGUF bit
counting, the server harnesses' parsing and arithmetic, and (slow) llama.cpp's perplexity
against kvbound's on the same windows."""

import contextlib
import hashlib
import io
import json
import struct
import zipfile

import pytest

from kvbound import models, record, serve, workload
from kvbound.baselines import llamacpp, llamacpp_serve
from kvbound.baselines.llamacpp_bench import buffers_mib, command_line, read_stream, request_body
from kvbound.baselines.llamacpp_eval import compare_tokens
from kvbound.baselines.llamacpp_profile import decode_evals, kernel_name
from kvbound.perfmodel.model import Arch


def test_external_engine_needs_a_version():
    engine = record.external_engine("llama.cpp", "b1", commit="abc")
    assert engine == {"name": "llama.cpp", "version": "b1", "external": True, "commit": "abc"}
    with pytest.raises(ValueError):
        record.external_engine("llama.cpp", "")


def test_outside_stack_records_need_weights_and_bits_per_weight():
    engine = record.external_engine("llama.cpp", "b1")
    record.check_external(engine, {"weights": "Q4_0", "bits_per_weight": 4.5})
    for params in ({}, {"weights": "Q4_0"}, {"weights": "Q4_0", "bits_per_weight": 0}):
        with pytest.raises(ValueError):
            record.check_external(engine, params)
    # write_record checks before it fingerprints the machine, so this needs no GPU.
    with pytest.raises(ValueError):
        record.write_record("generate", engine, {"weights": "Q4_0"}, {})


def test_kvbound_and_older_baseline_engines_are_not_checked():
    record.check_external({"name": "kvbound", "preset": "cuda"}, {})
    record.check_external({"name": "hf-transformers", "version": "5"}, {"dtype": "fp16"})


def _string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def _gguf(tmp_path, tensors):
    """A GGUF header with three metadata entries and the given (name, dims, type) tensors."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 3)
    out += _string("general.architecture") + struct.pack("<I", 8) + _string("llama")
    out += _string("general.file_type") + struct.pack("<II", 4, 2)
    out += _string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2)
    out += _string("a") + _string("b")
    for name, dims, kind in tensors:
        out += _string(name) + struct.pack("<I", len(dims))
        out += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", kind, 0)
    path = tmp_path / "toy.gguf"
    path.write_bytes(out)
    return path


def test_gguf_bits_per_weight_counts_block_scales_and_every_tensor(tmp_path):
    # Q4_0: 18 bytes per 32 weights; Q6_K: 210 bytes per 256; F32: 4 bytes per weight.
    path = _gguf(
        tmp_path,
        [("blk.0.ffn_up.weight", [256, 64], 2), ("output.weight", [256, 8], 14),
         ("blk.0.attn_norm.weight", [256], 0)],
    )  # fmt: skip
    info = llamacpp.gguf_info(path)
    weights = 256 * 64 + 256 * 8 + 256
    stored = 256 * 64 // 32 * 18 + 256 * 8 // 256 * 210 + 256 * 4
    assert info["weights"] == weights
    assert info["tensor_bytes"] == stored
    assert info["bits_per_weight"] == pytest.approx(8 * stored / weights)
    assert info["file_type"] == 2
    assert set(info["types"]) == {"Q4_0", "Q6_K", "F32"}


def test_gguf_rejects_other_files_and_unknown_tensor_types(tmp_path):
    other = tmp_path / "x.gguf"
    other.write_bytes(b"GGML" + bytes(20))
    with pytest.raises(ValueError):
        llamacpp.gguf_info(other)
    with pytest.raises(ValueError):
        llamacpp.gguf_info(_gguf(tmp_path, [("t", [32], 99)]))


def test_buffers_are_summed_by_device_and_kind():
    log = "\n".join(
        [
            "load_tensors:        CUDA0 model buffer size =  2098.00 MiB",
            "load_tensors:   CPU_Mapped model buffer size =   125.00 MiB",
            "llama_kv_cache:      CUDA0 KV buffer size =    22.00 MiB",
            "sched_reserve:      CUDA0 compute buffer size =    66.50 MiB",
            "sched_reserve:  CUDA_Host compute buffer size =     4.01 MiB",
            "llama_kv_cache:      CUDA0 KV buffer size =     2.00 MiB",
        ]
    )
    assert buffers_mib(log) == {
        "CUDA0 model": 2098.0,
        "CPU model": 125.0,
        "CUDA0 KV": 24.0,
        "CUDA0 compute": 66.5,
        "CUDA_Host compute": 4.01,
    }


def test_server_flags_and_greedy_request():
    assert command_line({"--ctx-size": 256, "--no-webui": True}) == [
        "--ctx-size",
        "256",
        "--no-webui",
    ]
    body = request_body([5, 6, 7], 128)
    assert body["prompt"] == [5, 6, 7]  # ids, so the server adds no BOS
    assert (body["samplers"], body["top_k"], body["ignore_eos"]) == (["top_k"], 1, True)
    assert body["cache_prompt"] is False


def test_worker_nll_is_kvbounds_rule():
    """The llama.cpp worker scores a window as kvbound.evals.perplexity.window_nll does, across
    more rows than one float64 chunk."""
    import numpy as np
    import torch

    from kvbound.baselines.llamacpp_lib import SCORE_ROWS, window_nll
    from kvbound.evals import perplexity

    rng = np.random.default_rng(0)
    n, vocab = SCORE_ROWS * 2 + 7, 300
    logits = (rng.standard_normal((n, vocab)) * 4).astype(np.float32)
    ids = rng.integers(0, vocab, n)
    expected = perplexity.window_nll(torch.from_numpy(logits), torch.from_numpy(ids))
    assert window_nll(logits, ids) == pytest.approx(expected, rel=1e-6)


def test_compare_tokens_reports_the_first_difference():
    assert compare_tokens([1, 2, 3], [1, 2, 3]) == {
        "n_tokens": 3, "same_count": True, "identical": True, "first_difference": None,
    }  # fmt: skip
    assert compare_tokens([1, 2, 3], [1, 9, 3])["first_difference"] == 1
    shorter = compare_tokens([1, 2, 3], [1, 2])
    assert (shorter["same_count"], shorter["first_difference"]) == (False, 2)


def test_extract_refuses_a_member_outside_the_folder(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("bin/ok.txt", "ok")
        zf.writestr("../escaped.txt", "no")
    dest = tmp_path / "out" / "bin"
    with pytest.raises(SystemExit):
        llamacpp._extract(archive, dest)
    assert not dest.exists()
    assert not (tmp_path / "out" / "escaped.txt").exists()


def test_download_keeps_nothing_whose_sha256_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(llamacpp, "ROOT", tmp_path)
    monkeypatch.setattr(
        llamacpp.urllib.request,
        "urlopen",
        lambda url, timeout: contextlib.closing(io.BytesIO(b"not the pinned bytes")),
    )
    artifact = llamacpp.Artifact("bin", "x.zip", "https://example.invalid/x.zip", "0" * 64)
    with pytest.raises(SystemExit):
        llamacpp.download(artifact)
    assert list((tmp_path / "downloads" / "bin").iterdir()) == []
    good = llamacpp.Artifact(
        "bin", "x.zip", artifact.url, hashlib.sha256(b"not the pinned bytes").hexdigest()
    )
    assert llamacpp.download(good).read_bytes() == b"not the pinned bytes"


@pytest.mark.parametrize("repo_has_spm", [True, False])
def test_converter_input_holds_the_checkpoint_and_the_repos_tokenizer_model(
    tmp_path, monkeypatch, repo_has_spm
):
    monkeypatch.setattr(models, "DATA", tmp_path / "data")
    monkeypatch.setattr(llamacpp, "ROOT", tmp_path / "ext")
    spec = models.ModelSpec("toy", "org/toy", "abc123", False, {})
    spec.checkpoint_dir.mkdir(parents=True)
    for name in models.CHECKPOINT_FILES:
        (spec.checkpoint_dir / name).write_text(name)
    asked = []
    repo = ["config.json", "tokenizer.model"] if repo_has_spm else ["config.json"]

    def list_repo_files(repo_id, revision):
        assert (repo_id, revision) == ("org/toy", "abc123")
        return repo

    def hf_hub_download(repo_id, filename, revision, local_dir):
        asked.append(filename)
        (local_dir / filename).write_text("spm")

    monkeypatch.setattr(llamacpp, "list_repo_files", list_repo_files)
    monkeypatch.setattr(llamacpp, "hf_hub_download", hf_hub_download)
    out = llamacpp.converter_input(spec)
    for name in models.CHECKPOINT_FILES:
        assert (out / name).read_text() == name
    assert asked == (["tokenizer.model"] if repo_has_spm else [])
    assert (out / "tokenizer.model").exists() == repo_has_spm


def test_every_quantized_format_has_quantize_arguments():
    assert set(llamacpp.QUANTIZE_ARGS) == set(llamacpp.FORMATS) - {"F16"}
    assert llamacpp.QUANTIZE_ARGS["Q4_0_PURE"] == ["--pure", "Q4_0"]


def _events(*events):
    return [b"data: " + json.dumps(e).encode() + b"\n" for e in events]


def test_tokens_sharing_an_event_take_its_time():
    clock = iter([1.0, 1.5, 2.25, 3.0]).__next__
    lines = [b": keep-alive\n", *_events({"tokens": [5]}, {"tokens": [6, 7]}, {"tokens": [8]})]
    lines += _events({"tokens": [], "stop": True, "timings": {"prompt_n": 3}})
    out = read_stream(lines, 0.5, clock)
    assert out["tokens"] == [5, 6, 7, 8]
    assert out["times"] == [0.5, 1.0, 1.0, 1.75]
    assert out["multi_token_events"] == 1
    assert out["final"]["timings"] == {"prompt_n": 3}
    assert out["error"] is None


def test_tokens_held_inside_a_character_arrive_with_the_closing_event():
    clock = iter([1.0, 2.0]).__next__
    lines = _events({"tokens": [5]}, {"tokens": [], "stop": True, "tokens_predicted": 3})
    out = read_stream(lines, 0.0, clock)
    assert out["times"] == [1.0, 2.0, 2.0]
    assert (out["tokens"], out["held_tokens"]) == ([5], 2)


def test_an_error_event_ends_the_stream():
    out = read_stream(_events({"tokens": [5]}, {"error": {"message": "full"}}), 0.0, lambda: 1.0)
    assert out["error"] == {"message": "full"}
    assert out["tokens"] == [5]


def test_kernel_names_keep_template_arguments_and_drop_parameters():
    signature = "void mul_mat_vec_q<(ggml_type)2, (int)1, (bool)1>(const void *, const void *, int)"
    assert kernel_name(signature) == "mul_mat_vec_q<(ggml_type)2, (int)1, (bool)1>"
    assert kernel_name("quantize_q8_1(const float *, void *, long long)") == "quantize_q8_1"


def test_decode_evals_skip_the_prompt_line():
    report = (
        "prompt eval time =      31.10 ms /     2 tokens (   15.55 ms per token)\n"
        "       eval time =    1530.00 ms /   255 runs   (    6.00 ms per token)\n"
    )
    assert decode_evals(report) == (255, 1530.0)
    with pytest.raises(SystemExit):
        decode_evals("prompt eval time = 31.10 ms / 2 tokens")


def test_latency_metrics_follow_arrivals():
    requests = [
        {"arrival_s": 0.0, "token_times_s": [0.1, 0.2, 0.3]},
        {"arrival_s": 1.0, "token_times_s": [1.5]},
    ]
    m = serve.latency_metrics(requests, 2.0)
    assert m["throughput_tok_s"] == pytest.approx(4 / 2.0)
    assert m["ttft_p50_s"] == pytest.approx(0.3)  # 0.1 and 0.5, interpolated
    assert m["tpot_p50_s"] == pytest.approx(0.1)  # the one-token request has no TPOT
    assert m["latency_p90_s"] == pytest.approx(0.3 + 0.9 * 0.2)


def test_the_pool_holds_kvbounds_bytes():
    arch = Arch.from_table(models.load_spec("tinyllama-1.1b").arch)
    cells, pool_bytes = llamacpp_serve.kv_cells(arch, 8, 1024, "f16")
    assert pool_bytes == 8 * 1024 * 22528
    assert cells == 8192
    cells, _ = llamacpp_serve.kv_cells(arch, 8, 1024, "q8_0")
    assert cells == 15360  # 8 * 1024 * 22528 / 11968, rounded down to 256


def test_safe_slots_cannot_overflow_the_cache():
    trace = workload.trace(workload.load("fixed"), 32000, 1024)
    assert llamacpp_serve.safe_slots(8192, trace) == 8192 // 384
    assert llamacpp_serve.safe_slots(10**6, trace) == llamacpp_serve.MAX_SLOTS


def test_failed_requests_leave_the_latency_figures():
    trace = {"requests": [{"arrival_s": 0.0, "max_new_tokens": 2}] * 2}
    ok = {"times": [0.2, 0.3], "error": None, "sent_s": 0.001}
    failed = {"times": [0.2], "error": "Context size has been exceeded.", "sent_s": 0.003}
    m = llamacpp_serve.pass_metrics(trace, [ok, failed])
    assert m["failed"] == 1
    assert m["total_s"] == pytest.approx(0.3)
    assert m["throughput_tok_s"] == pytest.approx(2 / 0.3)
    assert m["send_lag_max_s"] == pytest.approx(0.003)


def test_a_request_whose_connection_breaks_fails_alone(monkeypatch):
    def stream(port, body, origin):
        if body["prompt"] == [2]:
            raise ConnectionResetError("reset")
        return {"times": [0.1, 0.2], "error": None, "sent_s": 0.0}

    monkeypatch.setattr(llamacpp_serve, "stream_completion", stream)
    requests = [{"arrival_s": 0.0, "prompt": [i], "max_new_tokens": 2} for i in (1, 2)]
    ok, broken = llamacpp_serve.run_pass(0, {"requests": requests})
    assert ok["error"] is None
    assert broken["times"] == [] and broken["error"] == "ConnectionResetError: reset"
    assert llamacpp_serve.pass_metrics({"requests": requests}, [ok, broken])["failed"] == 1


def test_failure_cause_gives_slot_advice_only_for_a_cache_overflow():
    short = {"times": [0.1], "error": None}
    assert "no error event" in llamacpp_serve.failure_cause([short])
    http_error = {"times": [], "error": "HTTP 500: b'boom'"}
    assert llamacpp_serve.failure_cause([short, http_error]) == "HTTP 500: b'boom'"
    overflow = {"times": [], "error": "Context size has been exceeded."}
    assert "lower --parallel" in llamacpp_serve.failure_cause([overflow])


def test_server_counters_give_steps_and_busy_slots():
    page = "# HELP x\nllamacpp:n_decode_total 30\nllamacpp:n_busy_slots_per_decode 2.5\n"
    after = llamacpp_serve.parse_metrics(page)
    before = {"n_decode_total": 10.0, "n_busy_slots_per_decode": 1.0}
    counts = llamacpp_serve.step_counts(before, after)
    assert counts == {"steps": 20.0, "busy_slots_per_step": (75 - 10) / 20}


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model", ["tinyllama-1.1b", "llama-3.2-1b"])
def test_f16_gguf_scores_like_the_engine(kvb, model):
    """The F16 GGUF, fed kvbound's windows through llama.dll, gives kvbound's perplexity on the
    first windows of WikiText-2, and llama.cpp tokenizes the text to the same ids."""
    from kvbound.baselines.llamacpp_eval import evaluate
    from kvbound.evals.perplexity import evaluate as kvbound_evaluate

    if not llamacpp.gguf_path(model, "F16").exists():
        pytest.skip(f"no F16 GGUF of {model}; run `make fetch-llamacpp MODEL={model}`")
    ours = evaluate(model, "F16", max_windows=4)
    theirs = kvbound_evaluate(model, max_windows=4)
    assert ours["params"]["llama_cpp_tokenization"]["identical"]
    assert ours["params"]["n_predicted"] == theirs["params"]["n_predicted"]
    assert ours["metrics"]["perplexity"] == pytest.approx(theirs["metrics"]["perplexity"], rel=2e-3)
