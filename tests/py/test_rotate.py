"""Residual-stream rotation: the matrix, the folding algebra, the rotated fixture, and the int4_g
checkpoints quantized from it."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kvbound import gptq, tolerance
from kvbound.fixture import FIXTURE_DIR, HF_CONFIG, write_random_llama
from kvbound.gptq import GptqConfig, quantize_model
from kvbound.quantize import MANIFEST_FILE, convert, load_dequantized
from kvbound.ref import load_llama
from kvbound.ref.config import LlamaConfig
from kvbound.rotate import (
    ROTATE_MANIFEST_FILE,
    hadamard,
    random_hadamard,
    rotate_checkpoint,
    rotate_tensors,
)
from kvbound.weight_format import (
    KEY_CALIB_DATASET,
    KEY_ROTATION,
    KEY_ROTATION_SEED,
    KEY_SCHEME,
    VARIANTS,
    WeightFormat,
)

GROUP = 32
HIDDEN = HF_CONFIG["hidden_size"]
# Measured on random prompts of the fixture: rel-RMS 4.6e-4 of the logits, the size of FP16
# rounding of the rotated weights. The limit leaves about 4x of margin.
FP16_LOGIT_REL_RMS_MAX = 2e-3
# The int4_g limit of test_quantize (group 32 on the fixture).
INT4_LOGIT_REL_RMS_MAX = 0.5


def rms_norm(x, gain, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gain


def prompts(n=3, length=24, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(HF_CONFIG["vocab_size"], (n, length), generator=gen)


def test_random_hadamard_is_orthogonal_and_flat():
    h = hadamard(8)
    assert torch.equal(h @ h.T, 8 * torch.eye(8, dtype=torch.float64))
    q = random_hadamard(HIDDEN, 0)
    assert q.dtype == torch.float64
    assert (q @ q.T - torch.eye(HIDDEN, dtype=torch.float64)).abs().max() < 1e-12
    # Every entry has magnitude 1 / sqrt(n): each rotated channel mixes all channels equally.
    assert torch.allclose(q.abs(), torch.full_like(q, HIDDEN**-0.5))
    assert torch.equal(q, random_hadamard(HIDDEN, 0))
    assert not torch.equal(q, random_hadamard(HIDDEN, 1))
    # One outsized channel spreads evenly over all of them.
    x = torch.zeros(HIDDEN, dtype=torch.float64)
    x[5] = 100.0
    assert torch.allclose((x @ q).abs(), torch.full_like(x, 100.0 * HIDDEN**-0.5))
    with pytest.raises(ValueError, match="power-of-two"):
        hadamard(96)


def test_norm_gains_fold_into_the_readers_and_writers_rotate_their_output():
    tensors = load_file(str(FIXTURE_DIR / "model.safetensors"))
    cfg = LlamaConfig.from_json(FIXTURE_DIR / "config.json")
    q = random_hadamard(HIDDEN, 0)
    rotated = rotate_tensors(tensors, cfg, q)
    layer = "model.layers.1."
    gain = tensors[layer + "input_layernorm.weight"].double()
    h = torch.randn(10, HIDDEN, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    # The rotated norm has unit gain and gives the reader the same product on the rotated stream.
    assert torch.equal(rotated[layer + "input_layernorm.weight"], torch.ones(HIDDEN).half())
    for proj in ("q_proj", "k_proj", "v_proj"):
        name = f"{layer}self_attn.{proj}.weight"
        expected = rms_norm(h, gain) @ tensors[name].double().T
        actual = rms_norm(h @ q, 1.0) @ rotated[name].double().T
        assert tolerance.rel_rms(actual, expected) < 1e-3, proj
    # A writer's output lands in the rotated stream.
    name = layer + "mlp.down_proj.weight"
    z = torch.randn(10, cfg.intermediate_size, dtype=torch.float64)
    expected = (z @ tensors[name].double().T) @ q
    assert tolerance.rel_rms(z @ rotated[name].double().T, expected) < 1e-3
    # Every gain is folded away; the tied head becomes its own tensor with the final gain in it.
    for name, w in rotated.items():
        if name.endswith("norm.weight"):
            assert torch.equal(w, torch.ones_like(w)), name
    head = tensors["model.embed_tokens.weight"].double() * tensors["model.norm.weight"].double()
    assert tolerance.rel_rms(rotated["lm_head.weight"].double(), head @ q) < 1e-3
    assert set(rotated) == set(tensors) | {"lm_head.weight"}


def assert_same_function(rotated_dir, source_dir):
    original, rotated = load_llama(source_dir), load_llama(rotated_dir)
    assert not rotated.cfg.tie_word_embeddings
    assert rotated.lm_head.weight is not rotated.embed.weight
    for ids in prompts():
        actual, expected = rotated(ids[None])[0], original(ids[None])[0]
        assert tolerance.rel_rms(actual, expected) <= FP16_LOGIT_REL_RMS_MAX
        assert tolerance.top1_agreement(actual, expected) == 1.0


def test_rotated_tied_fixture_computes_the_same_logits(tmp_path):
    out = tmp_path / "rot"
    manifest = rotate_checkpoint(FIXTURE_DIR, out, seed=3, source="fixture@0")
    assert_same_function(out, FIXTURE_DIR)

    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        metadata = f.metadata()
    fmt = WeightFormat.from_metadata(metadata)
    assert fmt == WeightFormat.make("fp16", rotation="hadamard") and fmt.variant == "fp16.rot"
    assert not fmt.quantized and KEY_SCHEME not in metadata
    assert metadata[KEY_ROTATION_SEED] == "3"
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config == HF_CONFIG | {"tie_word_embeddings": False}
    written = json.loads((out / ROTATE_MANIFEST_FILE).read_text(encoding="utf-8"))
    vocab = HF_CONFIG["vocab_size"]
    assert written["added_params"] == manifest["added_params"] == vocab * HIDDEN
    assert written["source_tied"] and written["seed"] == 3

    with pytest.raises(ValueError, match="not a plain checkpoint"):
        rotate_checkpoint(out, tmp_path / "twice")


def test_rotated_untied_checkpoint_computes_the_same_logits(tmp_path):
    src = tmp_path / "untied"
    write_random_llama(src, {"tie_word_embeddings": False})
    out = tmp_path / "rot"
    manifest = rotate_checkpoint(src, out)
    assert manifest["added_params"] == 0
    assert_same_function(out, src)


def test_rotation_needs_a_power_of_two_hidden_size(tmp_path):
    src = tmp_path / "wide"
    write_random_llama(src, {"hidden_size": 96, "num_attention_heads": 4})
    with pytest.raises(ValueError, match="power-of-two"):
        rotate_checkpoint(src, tmp_path / "rot")
    assert not (tmp_path / "rot").exists()


def test_rotation_metadata(tmp_path):
    path = tmp_path / "x.safetensors"
    for scheme, rounding, variant in (
        ("fp16", "", "fp16.rot"),
        ("int4_g", "", "int4_g.rot"),
        ("int4_g", "gptq", "int4_g.rot-gptq"),
    ):
        fmt = WeightFormat.make(scheme, GROUP, "minmax", "", rounding, "hadamard")
        assert fmt.to_metadata()[KEY_ROTATION] == "hadamard"
        save_file({"x": torch.zeros(2)}, str(path), metadata=fmt.to_metadata())
        with safe_open(str(path), framework="pt") as f:
            assert WeightFormat.from_metadata(f.metadata()) == fmt
        assert fmt.variant == variant and variant in VARIANTS
    # Files without a rotation keep their header.
    assert WeightFormat.make("fp16").to_metadata() == {}
    assert KEY_ROTATION not in WeightFormat.make("int4_g", GROUP).to_metadata()
    for scheme, clip, act_scales, rotation in (
        ("int8_pc", "minmax", "", "hadamard"),
        ("int4_g", "mse", "", "hadamard"),
        ("int4_g", "awq", "awq", "hadamard"),
        ("int4_g", "minmax", "", "other"),
    ):
        with pytest.raises(ValueError):
            WeightFormat.make(scheme, GROUP, clip, act_scales, "", rotation)


def calibration_ids(n=16, length=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(HF_CONFIG["vocab_size"], (n, length), generator=gen, dtype=torch.int32)


@pytest.mark.parametrize("use_gptq", [False, True], ids=["rot", "rot-gptq"])
def test_convert_a_rotated_checkpoint(tmp_path, use_gptq):
    rotated = tmp_path / "rot"
    rotate_checkpoint(FIXTURE_DIR, rotated, seed=3)
    out = tmp_path / "q"
    calib = {"calib": calibration_ids(), "calib_meta": {KEY_CALIB_DATASET: "random"}}
    convert(
        rotated,
        out,
        "int4_g",
        GROUP,
        device="cpu",
        gptq=GptqConfig() if use_gptq else None,
        **(calib if use_gptq else {}),
    )
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        metadata = f.metadata()
    fmt = WeightFormat.from_metadata(metadata)
    rounding = "gptq" if use_gptq else ""
    assert fmt == WeightFormat.make("int4_g", GROUP, rounding=rounding, rotation="hadamard")
    assert metadata[KEY_ROTATION_SEED] == "3"
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["variant"] == fmt.variant == ("int4_g.rot-gptq" if use_gptq else "int4_g.rot")
    assert manifest["rotation"]["seed"] == 3
    assert manifest["rotation"]["source_header"][KEY_ROTATION] == "hadamard"
    # The untied head is stored quantized, and with GPTQ it is solved like any reader.
    assert "lm_head.weight" in manifest["tensors"]
    if use_gptq:
        assert "lm_head" in manifest["gptq"]["layers"][-1]

    # The reference model reads it, and it stays close to the unrotated FP16 model.
    model = load_llama(out)
    _, tensors = load_dequantized(out)
    assert torch.equal(model.lm_head.weight, tensors["lm_head.weight"])
    original = load_llama(FIXTURE_DIR)
    for ids in prompts():
        error = tolerance.rel_rms(model(ids[None])[0], original(ids[None])[0])
        assert error <= INT4_LOGIT_REL_RMS_MAX


def test_awq_refuses_a_rotated_checkpoint(tmp_path):
    rotated = tmp_path / "rot"
    rotate_checkpoint(FIXTURE_DIR, rotated)
    with pytest.raises(ValueError, match="awq does not apply"):
        convert(
            rotated,
            tmp_path / "q",
            "int4_g",
            GROUP,
            device="cpu",
            calib=calibration_ids(),
            awq=True,
        )
    assert not (tmp_path / "q").exists()


def test_head_chunks_give_the_integers_of_the_whole_head(tmp_path, monkeypatch):
    src = tmp_path / "untied"
    write_random_llama(src, {"tie_word_embeddings": False})
    ids = calibration_ids()
    whole = quantize_model(load_llama(src, "fp16"), ids, GROUP, "cpu")
    # The fixture's vocabulary of 256 rows in chunks of 96: two whole chunks and a partial one.
    monkeypatch.setattr(gptq, "HEAD_CHUNK_ROWS", 96)
    chunked = quantize_model(load_llama(src, "fp16"), ids, GROUP, "cpu")
    for a, b in zip(
        whole.quantized["lm_head.weight"], chunked.quantized["lm_head.weight"], strict=True
    ):
        assert torch.equal(a, b)
    report, parts = whole.report[-1]["lm_head"], chunked.report[-1]["lm_head"]
    assert parts["loss_predicted"] == pytest.approx(report["loss_predicted"], rel=1e-5)
    for key, value in report["matrices"]["lm_head.weight"].items():
        assert parts["matrices"]["lm_head.weight"][key] == pytest.approx(value, rel=1e-5), key
