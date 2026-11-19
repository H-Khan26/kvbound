"""The llama.cpp baseline: outside-stack records, GGUF bit counting, the server harness's
parsing, and (slow) llama.cpp's perplexity against kvbound's on the same windows."""

import struct

import pytest

from kvbound import record
from kvbound.baselines import llamacpp
from kvbound.baselines.llamacpp_bench import buffers_mib, command_line, request_body
from kvbound.baselines.llamacpp_eval import compare_tokens


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
