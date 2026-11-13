"""Weight quantization: the INT8 and INT4 schemes, the file format, and the quantized fixture."""

import json
import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kvbound import golden, tolerance
from kvbound.fixture import FIXTURE_DIR
from kvbound.models import sha256_file
from kvbound.quantize import (
    MANIFEST_FILE,
    convert,
    dequantize_int4_g,
    dequantize_int8_pc,
    load_dequantized,
    pack_int4,
    quantize_int4_g,
    quantize_int8_pc,
    unpack_int4,
)
from kvbound.ref import load_llama
from kvbound.weight_format import (
    KEY_SCHEME,
    KEY_SOURCE,
    KEY_SOURCE_SHA256,
    KEY_VERSION,
    WeightFormat,
)

# The fixture's hidden size is 64 and its intermediate size 128.
GROUP = 32


def weights_with_outliers(shape, seed=0):
    """Gaussian weights with a few large entries of both signs, as real checkpoints have."""
    gen = torch.Generator().manual_seed(seed)
    w = 0.02 * torch.randn(shape, generator=gen)
    rows = torch.randint(shape[0], (8,), generator=gen)
    cols = torch.randint(shape[1], (8,), generator=gen)
    w[rows, cols] = 0.5 * torch.randn(8, generator=gen)
    return w


def group_errors(w, deq, group_size):
    """Squared error summed per group, float64."""
    err = (w.double() - deq.double()).pow(2)
    return err.reshape(w.shape[0], -1, group_size).sum(-1)


def test_int8_round_trip_is_within_half_a_step():
    w = weights_with_outliers((48, 96))
    w[5] = 0.0
    q, scale = quantize_int8_pc(w)
    assert q.dtype == torch.int8 and q.shape == w.shape
    assert scale.dtype == torch.float16 and scale.shape == (48,)
    assert int(q.abs().max()) <= 127
    deq = dequantize_int8_pc(q, scale)
    assert deq.dtype == torch.float32
    half_step = scale.float()[:, None] / 2
    assert ((w - deq).abs() <= half_step * (1 + 1e-6)).all()
    # An all-zero row gets scale 1.0 and zeros; every other row reaches the end of the range.
    assert scale[5] == 1.0 and (q[5] == 0).all()
    assert (q.abs().amax(dim=1)[torch.arange(48) != 5] == 127).all()


def test_int8_scale_is_rounded_to_f16_before_rounding_the_weights():
    w = weights_with_outliers((16, 64), seed=1)
    q, scale = quantize_int8_pc(w)
    expected = torch.clamp(torch.round(w / scale.float()[:, None]), -127, 127)
    assert torch.equal(q.float(), expected)


def test_int4_packs_the_low_nibble_first():
    q = torch.tensor([[1, -1, 7, -7], [0, 3, -8, 2]], dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.dtype == torch.uint8
    assert packed.tolist() == [[0xF1, 0x97], [0x30, 0x28]]
    assert torch.equal(unpack_int4(packed), q)

    # Through the quantizer: one group of 32 with scale 0.1 in F16, so q starts [1, -1, 7, -7]
    # and the rest is zero.
    row = torch.zeros(1, GROUP)
    row[0, :4] = torch.tensor([0.1, -0.1, 0.7, -0.7])
    packed, scale = quantize_int4_g(row, group_size=GROUP)
    assert packed.tolist() == [[0xF1, 0x97] + [0] * (GROUP // 2 - 2)]
    assert scale.tolist() == [[torch.tensor(0.1, dtype=torch.float16).item()]]


def test_int4_round_trip_is_within_half_a_step():
    w = weights_with_outliers((40, 128))
    w[3, :GROUP] = 0.0
    packed, scale = quantize_int4_g(w, GROUP)
    assert packed.shape == (40, 64) and scale.shape == (40, 128 // GROUP)
    assert scale.dtype == torch.float16
    deq = dequantize_int4_g(packed, scale, GROUP)
    half_step = scale.float().repeat_interleave(GROUP, dim=1) / 2
    assert ((w - deq).abs() <= half_step * (1 + 1e-6)).all()
    assert scale[3, 0] == 1.0 and (unpack_int4(packed)[3, :GROUP] == 0).all()


@pytest.mark.parametrize("clip", ["minmax", "mse"])
def test_int4_never_writes_minus_eight(clip):
    w = weights_with_outliers((64, 256), seed=2)
    w[0, :GROUP] = -1.0  # a group at the negative end of the range
    packed, _ = quantize_int4_g(w, GROUP, clip)
    q = unpack_int4(packed)
    assert int(q.min()) >= -7 and int(q.max()) <= 7
    assert not ((packed & 0xF) == 8).any() and not ((packed >> 4) == 8).any()


def test_int4_mse_clip_is_never_worse_than_minmax():
    w = weights_with_outliers((64, 512), seed=3)
    errors = {}
    for clip in ("minmax", "mse"):
        packed, scale = quantize_int4_g(w, GROUP, clip)
        errors[clip] = group_errors(w, dequantize_int4_g(packed, scale, GROUP), GROUP)
    assert (errors["mse"] <= errors["minmax"] * (1 + 1e-5)).all()
    assert errors["mse"].sum() < errors["minmax"].sum()


def test_int4_rejects_shapes_that_do_not_split_into_groups(tmp_path):
    with pytest.raises(ValueError, match="groups"):
        quantize_int4_g(torch.zeros(4, 48), 32)
    # The fixture's rows are 64 and 128 wide: groups of 128 do not divide the narrow ones.
    with pytest.raises(ValueError, match="groups"):
        convert(FIXTURE_DIR, tmp_path / "q", "int4_g", group_size=128)
    # The kernels need a power of two of at least 32.
    for group_size in (48, 16):
        with pytest.raises(ValueError, match="power of two"):
            convert(FIXTURE_DIR, tmp_path / "q", "int4_g", group_size=group_size)
    assert not (tmp_path / "q").exists()


def test_metadata_round_trips_through_a_safetensors_file(tmp_path):
    path = tmp_path / "x.safetensors"
    for fmt in (WeightFormat.make("int8_pc"), WeightFormat.make("int4_g", 32, "mse")):
        save_file({"x": torch.zeros(2)}, str(path), metadata=fmt.to_metadata())
        with safe_open(str(path), framework="pt") as f:
            assert WeightFormat.from_metadata(f.metadata()) == fmt
    assert WeightFormat.make("int4_g", 32, "mse").variant == "int4_g.mse"
    assert WeightFormat.make("int8_pc").to_metadata()["kvbound.group_size"] == "0"

    # Plain checkpoints: no scheme key, or no metadata at all.
    assert WeightFormat.from_metadata({"format": "pt"}).scheme == "fp16"
    assert WeightFormat.from_metadata(None) == WeightFormat()
    assert WeightFormat().to_metadata() == {}

    with pytest.raises(ValueError, match="format_version"):
        WeightFormat.from_metadata({**WeightFormat.make("int8_pc").to_metadata(), KEY_VERSION: "2"})
    with pytest.raises(ValueError):
        WeightFormat.make("int8_pc", clip="mse")
    with pytest.raises(ValueError):
        WeightFormat.make("int4_g", group_size=0)


def test_bits_per_weight_counts_the_scales():
    assert WeightFormat.make("int8_pc").bits_per_weight(2048, 2048) == 8 + 16 / 2048
    assert WeightFormat.make("int4_g", 128).bits_per_weight(2048, 5632) == 4.125
    assert WeightFormat().bits_per_weight(2048, 2048) == 16


@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_convert_fixture_and_load_dequantized(tmp_path, scheme):
    out = tmp_path / scheme
    manifest = convert(FIXTURE_DIR, out, scheme, GROUP, source="fixture@0", device="cpu")
    source = load_file(str(FIXTURE_DIR / "model.safetensors"))
    stored = load_file(str(out / "model.safetensors"))
    fmt = WeightFormat.make(scheme, GROUP)

    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        metadata = f.metadata()
    assert WeightFormat.from_metadata(metadata) == fmt
    assert metadata[KEY_SCHEME] == scheme and metadata[KEY_SOURCE] == "fixture@0"
    assert metadata[KEY_SOURCE_SHA256] == sha256_file(FIXTURE_DIR / "model.safetensors")
    assert (out / "config.json").read_bytes() == (FIXTURE_DIR / "config.json").read_bytes()

    loaded_fmt, loaded = load_dequantized(out)
    assert loaded_fmt == fmt
    assert set(loaded) == set(source)
    for name, w in source.items():
        assert loaded[name].dtype == torch.float32 and loaded[name].shape == w.shape
        if w.ndim == 1:
            assert stored[name].dtype == torch.float16
            assert torch.equal(loaded[name], w.float())
            continue
        if scheme == "int8_pc":
            q, scale = quantize_int8_pc(w)
            expected = dequantize_int8_pc(q, scale)
            assert stored[name].dtype == torch.int8 and stored[name].shape == w.shape
        else:
            q, scale = quantize_int4_g(w, GROUP)
            expected = dequantize_int4_g(q, scale, GROUP)
            assert stored[name].dtype == torch.uint8
            assert stored[name].shape == (w.shape[0], w.shape[1] // 2)
        assert torch.equal(stored[name], q) and torch.equal(stored[f"{name}.scale"], scale)
        assert torch.equal(loaded[name], expected)

    written = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert written["variant"] == scheme
    quantized = [name for name, w in source.items() if w.ndim == 2]
    assert set(written["tensors"]) == set(quantized)
    n_bytes = sum(stored[name].nbytes + stored[f"{name}.scale"].nbytes for name in quantized)
    assert manifest["quantized_bytes"] == written["quantized_bytes"] == n_bytes


# Measured on the fixture's three prompts (rel-RMS of the logits against the FP16 reference):
# int8_pc 0.013, int4_g (group 32) 0.21 to 0.25. The limits leave about 3x and 2x of margin.
LOGIT_REL_RMS_MAX = {"int8_pc": 0.04, "int4_g": 0.5}


@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_reference_runs_quantized_fixture(tmp_path, scheme):
    out = tmp_path / scheme
    convert(FIXTURE_DIR, out, scheme, GROUP, device="cpu")
    model = load_llama(out)
    _, tensors = load_dequantized(out)
    # The reference computes with exactly the dequantized values.
    assert torch.equal(
        model.layers[0].attn.q_proj.weight, tensors["model.layers.0.self_attn.q_proj.weight"]
    )
    assert torch.equal(model.embed.weight, tensors["model.embed_tokens.weight"])

    reference = load_llama(FIXTURE_DIR)
    manifest, goldens = golden.load(FIXTURE_DIR)
    for prompt in manifest["prompts"]:
        ids = torch.tensor([goldens[f"{prompt['id']}/input_ids"].tolist()])
        actual, expected = model(ids)[0], reference(ids)[0]
        assert tolerance.rel_rms(actual, expected) <= LOGIT_REL_RMS_MAX[scheme], prompt["id"]
        assert tolerance.top1_agreement(actual, expected) == 1.0, prompt["id"]


def test_tied_embeddings(tmp_path):
    # The fixture ties embeddings and stores no lm_head, so neither does its quantized copy.
    out = tmp_path / "tied"
    convert(FIXTURE_DIR, out, "int8_pc", device="cpu")
    keys = set(load_file(str(out / "model.safetensors")))
    assert "lm_head.weight" not in keys and "lm_head.weight.scale" not in keys
    model = load_llama(out)
    assert model.lm_head.weight is model.embed.weight

    # A tied checkpoint that stores the duplicate lm_head keeps it, quantized.
    src = tmp_path / "with_head"
    src.mkdir()
    shutil.copyfile(FIXTURE_DIR / "config.json", src / "config.json")
    weights = load_file(str(FIXTURE_DIR / "model.safetensors"))
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
    save_file(weights, str(src / "model.safetensors"))
    out = tmp_path / "with_head_q"
    convert(src, out, "int4_g", GROUP, device="cpu")
    stored = load_file(str(out / "model.safetensors"))
    assert stored["lm_head.weight"].dtype == torch.uint8 and "lm_head.weight.scale" in stored
    assert torch.equal(load_llama(out).embed.weight, load_dequantized(out)[1]["lm_head.weight"])


def test_convert_refuses_a_quantized_source(tmp_path):
    convert(FIXTURE_DIR, tmp_path / "q", "int8_pc", device="cpu")
    with pytest.raises(ValueError, match="already quantized"):
        convert(tmp_path / "q", tmp_path / "qq", "int8_pc", device="cpu")
