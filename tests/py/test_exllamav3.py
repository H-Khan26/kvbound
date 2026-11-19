"""The ExLlamaV3 baseline: EXL3 bit counting against the source checkpoint, the environment's
package list, and the generator harness's cache size and run checks."""

import json
import struct

import pytest

from kvbound import record
from kvbound.baselines import exllamav3
from kvbound.baselines.exllamav3_bench import cache_tokens, check_runs

BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "U32": 4}


def _safetensors(path, tensors):
    """A safetensors file of zeros holding the given {name: (dtype, shape)} tensors."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = BYTES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    header["__metadata__"] = {"format": "pt"}
    data = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(data)) + data + bytes(offset))
    return path


def _source(folder, tied):
    tensors = {
        "model.embed_tokens.weight": ("BF16", [1024, 256]),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [256, 256]),
        "model.layers.0.mlp.down_proj.weight": ("BF16", [256, 512]),
        "model.layers.0.input_layernorm.weight": ("BF16", [256]),
        "model.norm.weight": ("BF16", [256]),
    }
    if not tied:
        tensors["lm_head.weight"] = ("BF16", [1024, 256])
    return _safetensors(folder / "model.safetensors", tensors).parent


def _exl3(folder, k_q, k_down, k_head):
    """An EXL3 layout: trellis [in/16, out/16, 16 * K] int16 plus FP16 sign vectors per linear
    layer, the embedding in FP16 and the norms as stored by the converter."""

    def linear(key, n_in, n_out, k):
        return {
            f"{key}.trellis": ("I16", [n_in // 16, n_out // 16, 16 * k]),
            f"{key}.suh": ("F16", [n_in]),
            f"{key}.svh": ("F16", [n_out]),
            f"{key}.mul1": ("U32", [1]),
        }

    tensors = {
        "model.embed_tokens.weight": ("F16", [1024, 256]),
        **linear("model.layers.0.self_attn.q_proj", 256, 256, k_q),
        **linear("model.layers.0.mlp.down_proj", 512, 256, k_down),
        **linear("lm_head", 256, 1024, k_head),
        "model.layers.0.input_layernorm.weight": ("F16", [256]),
        "model.norm.weight": ("F16", [256]),
    }
    return _safetensors(folder / "model.safetensors", tensors).parent


def _linear_bytes(n_in, n_out, k):
    return n_in * n_out * k // 8 + 2 * (n_in + n_out) + 4


@pytest.mark.parametrize("tied", [False, True])
def test_bits_per_weight_count_every_stored_tensor_over_the_source_weights(tmp_path, tied):
    source = _source(tmp_path / "src", tied)
    info = exllamav3.exl3_info(_exl3(tmp_path / "exl3", 4, 5, 6), source)
    embed, q, down, head = 1024 * 256, 256 * 256, 512 * 256, 1024 * 256
    norms = 2 * 256
    weights = embed + q + down + norms + (0 if tied else head)
    stored = (
        embed * 2
        + _linear_bytes(256, 256, 4)
        + _linear_bytes(512, 256, 5)
        + _linear_bytes(256, 1024, 6)
        + norms * 2
    )
    assert info["weights"] == weights
    assert info["tensor_bytes"] == stored
    assert info["bits_per_weight"] == pytest.approx(8 * stored / weights)
    groups = info["groups"]
    # A tied output layer counts its own weights in its group, not in the model's total.
    assert groups["head"]["weights"] == head
    assert groups["head"]["bits_per_weight"] == pytest.approx(
        8 * _linear_bytes(256, 1024, 6) / head
    )
    assert groups["embedding"]["bits_per_weight"] == 16
    layers = _linear_bytes(256, 256, 4) + _linear_bytes(512, 256, 5)
    assert groups["layers"]["bits_per_weight"] == pytest.approx(8 * layers / (q + down))
    assert info["trellis_bits"] == {"4": round(q / (q + down), 4), "5": round(down / (q + down), 4)}


def test_the_unquantized_checkpoint_counts_sixteen_bits(tmp_path):
    source = _source(tmp_path / "src", tied=True)
    info = exllamav3.exl3_info(source, source)
    assert info["bits_per_weight"] == 16
    assert info["trellis_bits"] == {}


def test_a_module_missing_on_either_side_is_refused(tmp_path):
    source = _source(tmp_path / "src", tied=False)
    extra = _safetensors(
        tmp_path / "a" / "model.safetensors",
        {"model.layers.9.mlp.up_proj.trellis": ("I16", [16, 16, 64])},
    ).parent
    with pytest.raises(ValueError, match="no tensor"):
        exllamav3.exl3_info(extra, source)
    partial = _safetensors(
        tmp_path / "b" / "model.safetensors", {"model.norm.weight": ("F16", [256])}
    ).parent
    with pytest.raises(ValueError, match="stores nothing"):
        exllamav3.exl3_info(partial, source)


def test_freeze_output_gives_versions_and_the_wheel_takes_the_pinned_one():
    text = "# comment\nTorch==2.11.0+cu128\nexllamav3 @ file:///D:/x.whl\ntriton-windows==3.6.0\n"
    assert exllamav3.parse_freeze(text) == {
        "torch": "2.11.0+cu128",
        "exllamav3": exllamav3.VERSION,
        "triton-windows": "3.6.0",
    }
    with pytest.raises(ValueError):
        exllamav3.parse_freeze("torch>=2\n")


def test_engine_entry_is_an_outside_stack_with_its_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(exllamav3, "VENV", tmp_path)
    with pytest.raises(SystemExit):
        exllamav3.engine_entry("Generator")
    packages = {"exllamav3": exllamav3.VERSION, "torch": "2.11.0+cu128", "triton-windows": "3.6"}
    (tmp_path / exllamav3.DONE_FILE).write_text(json.dumps({"packages": packages}))
    engine = exllamav3.engine_entry("Generator")
    assert (engine["name"], engine["version"], engine["external"]) == (
        "exllamav3",
        exllamav3.VERSION,
        True,
    )
    assert engine["packages"] == packages
    record.check_external(engine, {"weights": "EXL3_4.0", "bits_per_weight": 4.2})


def test_every_exl3_format_has_converter_arguments():
    assert set(exllamav3.CONVERT_ARGS) == set(exllamav3.FORMATS) - {"FP16"}
    assert all(args[:1] == ["--bits"] for args in exllamav3.CONVERT_ARGS.values())


def test_the_cache_holds_whole_pages():
    assert cache_tokens(128, 128) == 512  # the generator needs one position more
    assert cache_tokens(1024, 128) == 1280
    assert cache_tokens(1, 1) == 256
    assert cache_tokens(127, 128) == 256


def _run(n_times, new_tokens=4, prompt_tokens=8, cached_pages=0):
    final = {
        "new_tokens": new_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_pages": cached_pages,
        "cached_tokens": 0,
    }
    return {"times": [0.1] * n_times, "final": final}


def test_runs_must_decode_every_token_from_the_whole_uncached_prompt():
    check_runs([_run(4)], prompt_len=8, new_tokens=4)
    for bad in (_run(3), _run(4, prompt_tokens=7), _run(4, cached_pages=1)):
        with pytest.raises(RuntimeError):
            check_runs([bad], prompt_len=8, new_tokens=4)
