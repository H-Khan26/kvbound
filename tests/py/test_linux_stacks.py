"""The vLLM and TensorRT-LLM harnesses without the stacks: stream parsing, server settings,
bits per weight, KV sizing, counters and the shared open-loop client. No GPU, no network."""

import json
import struct
import sys
import time

import pytest

from kvbound import models, record
from kvbound.baselines import linux_stacks as ls
from kvbound.baselines import linux_stacks_eval, linux_stacks_serve, openloop
from kvbound.baselines.http_server import ServerProcess, free_port
from kvbound.baselines.linux_stacks_bench import (
    bench_config,
    histogram_mean,
    parse_prometheus,
    read_stream,
    request_body,
)
from kvbound.perfmodel.model import Arch


def _events(*events):
    return [b"data: " + json.dumps(e).encode() + b"\n" for e in events]


def _chunk(ids, completion_tokens, prompt_tokens=3):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return {"choices": [{"index": 0, "text": "", "token_ids": ids}], "usage": usage}


def test_request_asks_each_stack_for_greedy_ids_and_running_usage():
    vllm = request_body("vllm", "m", [1, 2], 8)
    trt = request_body("tensorrt-llm", "m", [1, 2], 8)
    for body in (vllm, trt):
        assert (body["temperature"], body["top_k"], body["ignore_eos"]) == (0.0, 1, True)
        assert body["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
        assert body["prompt"] == [1, 2] and body["max_tokens"] == 8
    assert vllm["return_token_ids"] and "detokenize" not in vllm
    assert trt["detokenize"] is False and "return_token_ids" not in trt


def test_ids_sharing_an_event_take_its_time():
    clock = iter([1.0, 1.5, 2.25, 3.0]).__next__
    lines = [b": keep-alive\n", *_events(_chunk([5], 1), _chunk([6, 7], 3))]
    lines += [*_events({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 3}})]
    lines += [b"data: [DONE]\n"]
    out = read_stream(lines, 0.5, clock)
    assert out["tokens"] == [5, 6, 7]
    assert out["times"] == [0.5, 1.0, 1.0]
    assert (out["multi_token_events"], out["held_tokens"]) == (1, 0)
    assert out["usage"]["prompt_tokens"] == 3 and out["error"] is None


def test_an_event_without_ids_counts_the_tokens_its_usage_adds():
    clock = iter([1.0, 2.0, 3.0]).__next__
    no_ids = {"choices": [{"index": 0, "text": "ab"}], "usage": {"completion_tokens": 3}}
    out = read_stream([*_events(_chunk([5], 1), no_ids), b"data: [DONE]\n"], 0.0, clock)
    assert out["times"] == [1.0, 2.0, 2.0] and out["tokens"] == [5]


def test_tokens_only_the_closing_usage_counts_are_held():
    clock = iter([1.0, 2.0]).__next__
    closing = {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 3}}
    out = read_stream(_events(_chunk([5], 1), closing), 0.0, clock)
    assert out["times"] == [1.0, 2.0, 2.0] and out["held_tokens"] == 2


@pytest.mark.parametrize(
    "event", [{"error": {"message": "full"}}, {"object": "error", "message": "full"}]
)
def test_an_error_event_ends_the_stream(event):
    out = read_stream(_events(_chunk([5], 1), event, _chunk([6], 2)), 0.0, lambda: 1.0)
    assert out["error"] is not None and out["tokens"] == [5]


def test_prometheus_values_are_summed_over_labels_and_histograms_averaged():
    page = (
        "# HELP vllm:time_to_first_token_seconds x\n"
        'vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 0.5\n'
        'vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 4.0\n'
        'vllm:num_requests_running{engine="0"} 2.0\n'
        'vllm:num_requests_running{engine="1"} 1.0\n'
        "process_cpu_seconds_total 9\n"
    )
    after = parse_prometheus(page, "vllm:")
    assert after["vllm:num_requests_running"] == 3.0 and "process_cpu_seconds_total" not in after
    before = {
        "vllm:time_to_first_token_seconds_sum": 0.2,
        "vllm:time_to_first_token_seconds_count": 2.0,
    }
    assert histogram_mean(before, after, "vllm:time_to_first_token_seconds") == pytest.approx(0.15)
    assert histogram_mean(after, after, "vllm:time_to_first_token_seconds") is None


def _safetensors(path, tensors):
    header, offset = {}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def test_bits_per_weight_counts_every_stored_byte_over_the_models_weights(tmp_path):
    # A 256 x 256 matrix as packed INT4 in int32 (8 per element) with FP16 scales per 128.
    _safetensors(
        tmp_path / "model.safetensors",
        {
            "w.weight_packed": ("I32", [256, 32], 256 * 32 * 4),
            "w.weight_scale": ("F16", [256, 2], 256 * 2 * 2),
            "w.weight_shape": ("I64", [2], 16),
        },
    )
    tensors = ls.safetensors_tensors(tmp_path)
    assert tensors["w.weight_packed"] == {"dtype": "I32", "shape": [256, 32], "bytes": 32768}
    info = ls.bits_per_weight(tensors, 256 * 256)
    assert info["bits_per_weight"] == pytest.approx(8 * (32768 + 1024 + 16) / 65536)
    assert info["bytes_by_dtype"] == {"F16": 1024, "I32": 32768, "I64": 16}


def test_bench_cache_holds_one_sequence_in_whole_blocks():
    arch = Arch.from_table(models.load_spec("tinyllama-1.1b").arch)
    cfg = bench_config("tensorrt-llm", arch, 1152, "f16")
    assert (cfg.kv_tokens, cfg.kv_bytes) == (1216, 1216 * 22528)
    assert (cfg.max_seqs, cfg.max_step_tokens) == (1, 1152)
    assert bench_config("vllm", arch, 1000, "fp8").kv_tokens == 1024


def test_serving_pool_holds_kvbounds_bytes_in_whole_blocks():
    arch = Arch.from_table(models.load_spec("tinyllama-1.1b").arch)
    tokens, nbytes, pool_bytes = linux_stacks_serve.pool("vllm", arch, 8, 1024, "f16")
    assert pool_bytes == 8 * 1024 * 22528
    assert (tokens, nbytes) == (8192, pool_bytes)
    tokens, nbytes, _ = linux_stacks_serve.pool("tensorrt-llm", arch, 8, 1024, "fp8")
    assert tokens == 16384 and nbytes == pool_bytes


def _cfg(**kw):
    base = {"max_ctx": 1024, "max_seqs": 64, "max_step_tokens": 1024, "kv_tokens": 8192}
    return ls.ServerConfig(kv_bytes=8192 * 22528, **base, **kw)


def test_vllm_command_line(tmp_path):
    settings = ls.vllm_settings(tmp_path / "ckpt", tmp_path / "tok", "m", _cfg(kv="fp8"))
    command = ls.command_line("vllm", settings, tmp_path / "x.yaml", 1234)
    assert command[1:3] == ["serve", str(tmp_path / "ckpt")]
    joined = " ".join(command)
    assert "--kv-cache-memory-bytes 184549376" in joined and "--kv-cache-dtype fp8" in joined
    assert "--disable-prefix-caching" in command and "--enable-chunked-prefill" in command
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.6"
    assert command[-4:] == ["--host", "127.0.0.1", "--port", "1234"]
    assert not (tmp_path / "x.yaml").exists()


def test_trtllm_command_line_writes_the_llm_api_options(tmp_path):
    cfg = _cfg(cuda_graph_batch_sizes=(1, 2, 4))
    settings = ls.trtllm_settings(tmp_path / "ckpt", tmp_path / "tok", "m", cfg)
    command = ls.command_line("tensorrt-llm", settings, tmp_path / "x.yaml", 1234)
    assert command[1] == str(tmp_path / "ckpt")
    assert command[command.index("--max_batch_size") + 1] == "64"
    assert command[command.index("--extra_llm_api_options") + 1] == str(tmp_path / "x.yaml")
    options = json.loads((tmp_path / "x.yaml").read_text(encoding="utf-8"))
    assert options["dtype"] == "float16"
    assert options["kv_cache_config"]["max_tokens"] == 8192
    assert options["kv_cache_config"]["enable_block_reuse"] is False
    assert options["cuda_graph_config"] == {"batch_sizes": [1, 2, 4], "enable_padding": True}
    assert options["return_perf_metrics"] and options["enable_iter_perf_stats"]


def test_a_stack_without_a_4bit_path_is_refused():
    with pytest.raises(SystemExit, match="no W4A16_GPTQ path"):
        ls.require_checkpoint("tensorrt-llm", "tinyllama-1.1b", "W4A16_GPTQ")


def test_engine_entry_pins_the_requirements():
    engine = ls.engine_entry("vllm", "openai-completions")
    assert (engine["name"], engine["version"], engine["external"]) == ("vllm", "0.34.0", True)
    assert engine["requirements"] == "configs/stacks/vllm/requirements.txt"
    assert len(engine["requirements_sha256"]) == 64


@pytest.mark.parametrize("name", sorted(ls.ENVS))
def test_every_requirements_file_pins_its_package_with_hashes(name):
    env = ls.ENVS[name]
    text = env.requirements.read_text(encoding="utf-8")
    assert f"\n{env.package}=={env.version} \\\n    --hash=sha256:" in text


def test_records_made_under_wsl_are_tagged(monkeypatch):
    fingerprint = {"git": {"dirty": False}, "software": {"os_class": "wsl"}}
    assert record._tags(fingerprint, None) == ["os=wsl"]


def test_server_log_facts_keep_memory_lines_once():
    log = (
        "INFO 11-18 gpu_model_runner.py:1: Model loading took 2.05 GiB memory and 3.1 seconds\n"
        "INFO noise\nINFO GPU KV cache size: 8,192 tokens\n"
        "INFO GPU KV cache size: 8,192 tokens\n"
    )
    assert ls.log_facts("vllm", log) == [
        "Model loading took 2.05 GiB memory and 3.1 seconds",
        "GPU KV cache size: 8,192 tokens",
    ]


def test_prompt_logprobs_give_the_windows_nll():
    window = [1, 7, 9]
    entries = [None, {"7": {"logprob": -0.5, "rank": 1}}, {"9": {"logprob": -1.25, "rank": 3}}]
    assert linux_stacks_eval.nll_from_prompt_logprobs(window, entries) == pytest.approx(1.75)
    with pytest.raises(RuntimeError, match="2 prompt logprobs for 3 tokens"):
        linux_stacks_eval.nll_from_prompt_logprobs(window, entries[:2])


def test_worker_nll_is_kvbounds_rule():
    import torch

    from kvbound.baselines.linux_stacks_worker import window_nll
    from kvbound.evals.perplexity import window_nll as kvbound_window_nll

    logits = torch.randn(600, 50, generator=torch.Generator().manual_seed(0)).half()
    ids = torch.randint(50, (600,), generator=torch.Generator().manual_seed(1))
    assert window_nll(logits, ids.tolist()) == pytest.approx(kvbound_window_nll(logits, ids))


def test_vllm_pass_counts_come_from_counter_differences_and_gauge_peaks():
    before = {
        "vllm:iteration_tokens_total_count": 10.0,
        "vllm:iteration_tokens_total_sum": 100.0,
        "vllm:num_preemptions_total": 1.0,
    }
    after = {
        "vllm:iteration_tokens_total_count": 30.0,
        "vllm:iteration_tokens_total_sum": 500.0,
        "vllm:num_preemptions_total": 3.0,
    }
    samples = [{"vllm:num_requests_running": 5.0, "vllm:kv_cache_usage_perc": 0.5}, {}]
    counts = linux_stacks_serve.vllm_counts(before, after, samples)
    assert counts["steps"] == 20.0 and counts["tokens_per_step"] == 20.0
    assert counts["preemptions"] == 2.0
    assert (counts["running_peak"], counts["waiting_peak"], counts["kv_used_peak"]) == (5, 0, 0.5)


def test_trtllm_pass_counts_come_from_iteration_stats():
    def it(active, queued, scheduled, used):
        return {
            "numActiveRequests": active,
            "numQueuedRequests": queued,
            "inflightBatchingStats": {"numScheduledRequests": scheduled},
            "kvCacheStats": {"usedNumBlocks": used, "maxNumBlocks": 256},
        }

    counts = linux_stacks_serve.trtllm_counts([it(0, 0, 0, 0), it(4, 2, 4, 64), it(2, 0, 2, 32)])
    assert counts["steps"] == 2.0 and counts["requests_per_step"] == 3.0
    assert (counts["running_peak"], counts["waiting_peak"]) == (4.0, 2.0)
    assert counts["kv_used_peak"] == 0.25


def test_open_loop_requests_fail_alone_and_leave_the_figures():
    def send(req, start):
        if req["prompt"] == [2]:
            raise ConnectionResetError("reset")
        return {"times": [0.1, 0.2], "error": None, "sent_s": 0.0}

    requests = [{"arrival_s": 0.0, "prompt": [i], "max_new_tokens": 2} for i in (1, 2)]
    ok, broken = openloop.run_pass({"requests": requests}, send)
    assert broken["error"] == "ConnectionResetError: reset" and broken["times"] == []
    m = openloop.pass_metrics({"requests": requests}, [ok, broken])
    assert m["failed"] == 1 and m["throughput_tok_s"] == pytest.approx(2 / 0.2)
    with pytest.raises(RuntimeError, match="all 1 requests failed: ConnectionResetError"):
        openloop.pass_metrics({"requests": requests[1:]}, [broken])


def test_poller_keeps_readings_until_stopped():
    values = iter(range(1000))
    poller = openloop.Poller(lambda: next(values), interval_s=0.01)
    poller.start()
    time.sleep(0.1)
    poller.stop()
    n = len(poller.samples)
    time.sleep(0.05)
    assert n >= 2 and len(poller.samples) == n and poller.samples[:2] == [0, 1]


def test_poller_counts_failed_reads_and_keeps_sampling():
    calls = iter(range(1000))

    def read():
        n = next(calls)
        if n % 2:
            raise (RuntimeError if n % 4 == 1 else ValueError)(f"read {n} failed")
        return n

    poller = openloop.Poller(read, interval_s=0.01)
    poller.start()
    time.sleep(0.15)
    poller.stop()
    assert poller.failed_reads >= 2 and len(poller.samples) >= 2
    assert all(s % 2 == 0 for s in poller.samples)


def _serve_metrics(failed, failed_reads):
    return {"failed": {"max": failed}, "failed_reads": {"max": failed_reads}}


def test_a_trtllm_pass_that_missed_a_counter_read_is_not_recorded():
    refusal = linux_stacks_serve.refusal
    assert refusal("tensorrt-llm", _serve_metrics(0, 0)) is None
    assert "reads of /metrics failed" in refusal("tensorrt-llm", _serve_metrics(0, 1))
    assert refusal("vllm", _serve_metrics(0, 3)) is None  # cumulative counters
    assert "2 requests failed" in refusal("vllm", _serve_metrics(2, 0))


def test_records_state_the_settings_that_are_not_the_stacks_defaults():
    assert "cuda_graph_config.enable_padding" in ls.NOT_STACK_DEFAULTS["tensorrt-llm"]
    assert "--disable-prefix-caching" in ls.NOT_STACK_DEFAULTS["vllm"]


def test_server_process_waits_for_health_and_ends_the_server(tmp_path):
    port = free_port()
    command = [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"]
    server = ServerProcess(command, tmp_path / "server.log", port, name="http.server", health="/")
    with server:
        assert server.process is not None and server.process.poll() is None
    assert server.process.poll() is not None


def test_server_process_reports_an_early_exit(tmp_path):
    command = [sys.executable, "-c", "print('bad flag'); raise SystemExit(2)"]
    server = ServerProcess(command, tmp_path / "server.log", free_port(), name="fake")
    with pytest.raises(RuntimeError, match="fake exited early"), server:
        pass
