"""Weight quantization: the INT8 and INT4 schemes, the file format, the quantized fixture, and
activation-aware (AWQ) scales and clipping."""

import json
import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kvbound import golden, tolerance
from kvbound.awq import (
    CALIB_IDS,
    CLIP_RATIOS,
    SCALES_FILE,
    AwqResult,
    act_clip_scales,
    apply_scales,
    collect_and_search,
    norm_site_scales,
    pick_windows,
    rescaled_norm,
)
from kvbound.fixture import FIXTURE_DIR, HF_CONFIG, WHISPER_FIXTURE_DIR
from kvbound.models import GENERATION_FILE, load_spec, sha256_file
from kvbound.perfmodel.model import Precision
from kvbound.quantize import (
    MANIFEST_FILE,
    convert,
    dequantize,
    dequantize_int4_g,
    dequantize_int8_pc,
    f16_scales,
    load_dequantized,
    pack_int4,
    quantize,
    quantize_int4_g,
    quantize_int4_g_with_scales,
    quantize_int8_pc,
    unpack_int4,
)
from kvbound.ref import load_llama, load_whisper
from kvbound.weight_format import (
    KEY_ACT_SCALES,
    KEY_CALIB_DATASET,
    KEY_CALIB_SEED,
    KEY_CALIB_WINDOW,
    KEY_CALIB_WINDOWS,
    KEY_SCHEME,
    KEY_SOURCE,
    KEY_SOURCE_SHA256,
    KEY_VERSION,
    VARIANTS,
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


def test_act_scales_metadata(tmp_path):
    path = tmp_path / "x.safetensors"
    awq = WeightFormat.make("int4_g", 32, "awq", "awq")
    noclip = WeightFormat.make("int4_g", 32, "minmax", "awq")
    for fmt in (awq, noclip):
        save_file({"x": torch.zeros(2)}, str(path), metadata=fmt.to_metadata())
        with safe_open(str(path), framework="pt") as f:
            assert WeightFormat.from_metadata(f.metadata()) == fmt
        assert fmt.to_metadata()[KEY_ACT_SCALES] == "awq"
    assert awq.variant == "int4_g.awq" and noclip.variant == "int4_g.awq-noclip"
    assert {awq.variant, noclip.variant} <= set(VARIANTS)

    # Formats without activation scales write the header they always did, and read back from it.
    plain = WeightFormat.make("int4_g", 32, "mse")
    assert plain.to_metadata() == {
        "kvbound.format_version": "1",
        "kvbound.scheme": "int4_g",
        "kvbound.group_size": "32",
        "kvbound.pack_order": "lo_nibble_first",
        "kvbound.scale_dtype": "F16",
        "kvbound.clip": "mse",
    }
    assert WeightFormat.from_metadata(plain.to_metadata()) == plain and plain.act_scales == ""
    assert KEY_ACT_SCALES not in WeightFormat.make("int8_pc").to_metadata()

    for scheme, clip, act_scales in (
        ("int8_pc", "minmax", "awq"),
        ("fp16", "minmax", "awq"),
        ("int4_g", "mse", "awq"),
        ("int4_g", "awq", ""),  # the awq clip search needs the scaled activations
        ("int4_g", "minmax", "other"),
    ):
        with pytest.raises(ValueError):
            WeightFormat.make(scheme, 32, clip, act_scales)


def test_perf_model_counts_a_variant_by_its_scheme():
    int4 = Precision.named("int4_g", "fp16")
    for variant in ("int4_g.awq", "int4_g.awq-noclip", "int4_g.mse"):
        assert Precision.named(variant, "fp16") == int4
    assert set(VARIANTS) >= {"fp16", "int8_pc", "int4_g", "int4_g.mse"}
    for variant in VARIANTS:
        Precision.named(variant, "int8")


# The capture site of each layer norm whose weight activation scales are folded into.
NORM_SITES = {"attn_norm": "input_layernorm", "mlp_norm": "post_attention_layernorm"}


def calibration_ids(n=8, length=32, seed=0):
    """Seeded random tokens of the fixture's vocabulary, int32 [n, length]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(HF_CONFIG["vocab_size"], (n, length), generator=gen, dtype=torch.int32)


def test_calibration_windows_are_whole_distinct_and_seeded():
    ids = list(range(1000))
    windows = pick_windows(ids, 5, 64, seed=0)
    assert windows.dtype == torch.int32 and windows.shape == (5, 64)
    starts = windows[:, 0].tolist()
    assert len(set(starts)) == 5 and all(start % 64 == 0 for start in starts)
    assert torch.equal(windows, torch.tensor(starts)[:, None] + torch.arange(64))
    assert torch.equal(windows, pick_windows(ids, 5, 64, seed=0))
    assert not torch.equal(windows, pick_windows(ids, 5, 64, seed=1))
    with pytest.raises(ValueError, match="fewer"):
        pick_windows(ids, 16, 64, seed=0)  # 1,000 tokens hold 15 whole windows


def test_rescaled_norm_absorbs_f16_rounding():
    gen = torch.Generator().manual_seed(5)
    gamma = (1.0 + 0.1 * torch.randn(64, generator=gen)).half().float()
    gamma[3] = 0.0
    s = 8 ** (2 * torch.rand(64, generator=gen) - 1)
    gamma_q, s_eff = rescaled_norm(gamma, s)
    assert torch.equal(gamma_q, gamma_q.half().float())
    assert torch.allclose(gamma_q * s_eff, gamma, rtol=1e-6, atol=0)
    assert torch.allclose(s_eff, s, rtol=1e-3) and not torch.equal(s_eff, s)
    assert gamma_q[3] == 0 and s_eff[3] == s[3]
    # The realized scale is a fixed point: it gives the same stored weight and itself.
    again = rescaled_norm(gamma, s_eff)
    assert torch.equal(again[0], gamma_q) and torch.equal(again[1], s_eff)

    # A weight pushed out of the normal F16 range is refused; an untouched one never is.
    assert rescaled_norm(torch.tensor([1e-3, 1.0]), torch.tensor([100.0, 1.0])) is None
    assert rescaled_norm(torch.tensor([1e3, 1.0]), torch.tensor([1e-3, 1.0])) is None
    tiny = torch.tensor([2.0**-20, 1.0])
    assert torch.equal(rescaled_norm(tiny, torch.ones(2))[1], torch.ones(2))


def test_awq_clip_minimizes_each_groups_output_error():
    gen = torch.Generator().manual_seed(6)
    w = weights_with_outliers((16, 128), seed=4)
    x = torch.randn(50, 128, generator=gen) * torch.rand(128, generator=gen).pow(4)
    x[:, GROUP : 2 * GROUP] = 0.0  # a group no activation reaches
    groups, x_groups = w.reshape(16, -1, GROUP), x.reshape(50, -1, GROUP)
    scale, minmax_error, error = act_clip_scales(groups, x_groups)
    assert scale.dtype == torch.float16 and scale.shape == (16, 4)

    def output_error(scale):
        """Sum over tokens of (x . (w - w_hat))^2 per group, as llm-awq computes it."""
        deq = dequantize_int4_g(quantize_int4_g_with_scales(w, scale), scale, GROUP)
        d = (w - deq).reshape(16, -1, GROUP).double()
        return torch.einsum("tgi,ogi->otg", x_groups.double(), d).pow(2).sum(1)

    amax = groups.abs().amax(-1)
    by_ratio = torch.stack([output_error(f16_scales(ratio * amax, 7)) for ratio in CLIP_RATIOS])
    assert CLIP_RATIOS[0] == 1.0 and len(CLIP_RATIOS) == 10 and min(CLIP_RATIOS) > 0.54
    assert torch.allclose(minmax_error, by_ratio[0], rtol=1e-9, atol=1e-18)
    assert torch.allclose(error, by_ratio.amin(0), rtol=1e-9, atol=1e-18)
    assert torch.allclose(error, output_error(scale), rtol=1e-9, atol=1e-18)
    assert error.sum() < minmax_error.sum()
    # Ties keep min-max: a group whose activations are all zero is not clipped.
    assert torch.equal(scale[:, 1], f16_scales(amax, 7)[:, 1])


def test_awq_scales_leave_the_function_unchanged(tmp_path):
    source = load_file(str(FIXTURE_DIR / "model.safetensors"))
    reference = load_llama(FIXTURE_DIR)
    cfg = reference.cfg
    gen = torch.Generator().manual_seed(1)
    sizes = (cfg.hidden_size, cfg.hidden_size, cfg.intermediate_size)
    # Log-uniform in [1/8, 8], far wider than a search picks.
    random = AwqResult(
        GROUP,
        *([8 ** (2 * torch.rand(size, generator=gen) - 1) for _ in range(cfg.n_layers)]
          for size in sizes),
    )  # fmt: skip
    searched = collect_and_search(load_llama(FIXTURE_DIR), calibration_ids(), GROUP, "cpu")
    assert len(searched.s_attn) == len(searched.s_mlp) == len(searched.s_down) == cfg.n_layers

    manifest, goldens = golden.load(FIXTURE_DIR)
    for label, result in (("random", random), ("searched", searched)):
        scaled = apply_scales(source, result)
        assert set(scaled) == set(source)
        assert all(tensor.dtype == torch.float32 for tensor in scaled.values())
        for name in ("model.embed_tokens.weight", "model.norm.weight"):
            assert torch.equal(scaled[name], source[name].float())
        # The scale each norm weight realizes; a searched scale is already that one.
        s_eff = {}
        for i in range(cfg.n_layers):
            for site, norm in NORM_SITES.items():
                name = f"model.layers.{i}.{norm}.weight"
                assert torch.equal(scaled[name], scaled[name].half().float())
                s_eff[f"layers.{i}.{site}"] = source[name].float() / scaled[name]
            if label == "searched":
                assert torch.equal(s_eff[f"layers.{i}.attn_norm"], result.s_attn[i])
                assert torch.equal(s_eff[f"layers.{i}.mlp_norm"], result.s_mlp[i])

        out = tmp_path / label
        out.mkdir()
        shutil.copyfile(FIXTURE_DIR / "config.json", out / "config.json")
        save_file(scaled, str(out / "model.safetensors"))
        model = load_llama(out, "native")
        for prompt in manifest["prompts"]:
            ids = goldens[f"{prompt['id']}/input_ids"].tolist()
            actual = golden.capture_sites(model, ids)
            expected = golden.capture_sites(reference, ids)
            assert tolerance.rel_rms(actual["logits"], expected["logits"]) <= 1e-5, prompt["id"]
            assert tolerance.rel_rms(actual["logits"], goldens[f"{prompt['id']}/logits"]) <= 1e-5
            for site, s in s_eff.items():
                assert tolerance.rel_rms(actual[site] * s, expected[site]) <= 1e-5, site
                assert tolerance.rel_rms(actual[site], expected[site]) > 1e-3, site


@pytest.mark.parametrize("clip", ["awq", "minmax"])
def test_convert_with_awq(tmp_path, clip):
    ids = calibration_ids()
    calib_meta = {
        KEY_CALIB_DATASET: "random",
        KEY_CALIB_WINDOWS: "8",
        KEY_CALIB_WINDOW: "32",
        KEY_CALIB_SEED: "0",
    }
    out, plain_dir = tmp_path / "awq", tmp_path / "plain"
    manifest = convert(
        FIXTURE_DIR, out, "int4_g", GROUP, clip, device="cpu", awq=ids, calib_meta=calib_meta
    )
    convert(FIXTURE_DIR, plain_dir, "int4_g", GROUP, device="cpu")
    source = load_file(str(FIXTURE_DIR / "model.safetensors"))
    stored = load_file(str(out / "model.safetensors"))
    plain = load_file(str(plain_dir / "model.safetensors"))
    n_layers = load_llama(FIXTURE_DIR).cfg.n_layers

    # The header names the scales and where the calibration tokens came from.
    fmt = WeightFormat.make("int4_g", GROUP, clip, "awq")
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        metadata = f.metadata()
    assert WeightFormat.from_metadata(metadata) == fmt
    assert metadata[KEY_ACT_SCALES] == "awq" and calib_meta.items() <= metadata.items()
    written = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert written["variant"] == fmt.variant
    assert fmt.variant == ("int4_g.awq" if clip == "awq" else "int4_g.awq-noclip")
    assert written["awq"]["calib_windows"] == 8 and written["awq"]["calib_window"] == 32

    # No group's calibration loss is above plain rounding's, and no clip raises a matrix's.
    assert manifest["awq"]["layers"] == written["awq"]["layers"]
    assert len(written["awq"]["layers"]) == n_layers
    for layer in written["awq"]["layers"]:
        for group in ("attn", "mlp", "down"):
            assert layer[group]["loss"] <= layer[group]["loss_rtn"]
            assert layer[group]["loss"] == min(layer[group]["losses"])
            assert layer[group]["losses"][0] == layer[group]["loss_rtn"]
            assert layer[group]["s_min"] <= 1 <= layer[group]["s_max"]
        if clip == "awq":
            assert set(layer["clip"]) == {
                "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj",
            }  # fmt: skip
            assert all(c["loss"] < c["loss_minmax"] for c in layer["clip"].values())
        else:
            assert "clip" not in layer
    assert any(layer["down"]["alpha"] > 0 for layer in written["awq"]["layers"])

    # The side file holds the tokens and the scales the stored norm weights realize.
    side = load_file(str(out / SCALES_FILE))
    assert torch.equal(side[CALIB_IDS], ids) and side[CALIB_IDS].dtype == torch.int32
    assert len(side) == 1 + 3 * n_layers
    result = AwqResult(GROUP, *([side[f"layers.{i}.{g}"] for i in range(n_layers)]
                                for g in ("s_attn", "s_mlp", "s_down")))  # fmt: skip
    site_scales = norm_site_scales(out)
    assert set(site_scales) == {f"layers.{i}.{s}" for i in range(n_layers)
                                for s in ("attn_norm", "mlp_norm")}  # fmt: skip
    assert norm_site_scales(plain_dir) == {} and norm_site_scales(FIXTURE_DIR) == {}
    for i in range(n_layers):
        for site, norm in NORM_SITES.items():
            name = f"model.layers.{i}.{norm}.weight"
            assert stored[name].dtype == torch.float16
            s_eff = plain[name].float() / stored[name].float()
            assert torch.equal(s_eff, site_scales[f"layers.{i}.{site}"])

    # Tensors no scale or clip touches are the bytes plain int4_g writes; the others are the
    # scaled weights rounded against min-max scales or, when clipped, smaller ones.
    assert set(stored) == set(plain)
    scaled = apply_scales(source, result)
    clipped = ("v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if clip == "awq" else ()
    for name, w in source.items():
        if w.ndim == 1:
            continue
        scale = stored[f"{name}.scale"]
        untouched = not name.startswith("model.layers.") or "o_proj" in name
        if any(proj in name for proj in clipped):
            minmax = quantize_int4_g(scaled[name], GROUP)[1]
            assert (scale <= minmax).all() and (scale < minmax).any(), name
            assert torch.equal(stored[name], quantize_int4_g_with_scales(scaled[name], scale))
        elif untouched:
            assert torch.equal(stored[name], plain[name]), name
            assert torch.equal(scale, plain[f"{name}.scale"]), name
        else:
            q, minmax = quantize_int4_g(scaled[name], GROUP)
            assert torch.equal(stored[name], q) and torch.equal(scale, minmax), name
            assert not torch.equal(stored[name], plain[name]), name
    assert torch.equal(stored["model.norm.weight"], plain["model.norm.weight"])

    # The reference runs it as any int4_g checkpoint, within the limits of plain int4_g.
    model, reference = load_llama(out), load_llama(FIXTURE_DIR)
    _, goldens = golden.load(FIXTURE_DIR)
    for prompt in golden.load(FIXTURE_DIR)[0]["prompts"]:
        prompt_ids = torch.tensor([goldens[f"{prompt['id']}/input_ids"].tolist()])
        actual, expected = model(prompt_ids)[0], reference(prompt_ids)[0]
        assert tolerance.rel_rms(actual, expected) <= LOGIT_REL_RMS_MAX["int4_g"], prompt["id"]
        assert tolerance.top1_agreement(actual, expected) == 1.0, prompt["id"]

    # The header says the norm sites are scaled, so the scales may not go missing.
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copyfile(out / "model.safetensors", alone / "model.safetensors")
    with pytest.raises(FileNotFoundError, match="activation scales"):
        norm_site_scales(alone)


def test_convert_rejects_awq_combinations(tmp_path):
    ids = calibration_ids()
    for scheme, clip in (("int8_pc", "minmax"), ("int4_g", "mse")):
        with pytest.raises(ValueError, match="inconsistent"):
            convert(FIXTURE_DIR, tmp_path / "q", scheme, GROUP, clip, device="cpu", awq=ids)
    with pytest.raises(ValueError, match="inconsistent"):
        convert(FIXTURE_DIR, tmp_path / "q", "int4_g", GROUP, "awq", device="cpu")
    with pytest.raises(ValueError, match="calib_meta"):
        convert(FIXTURE_DIR, tmp_path / "q", "int4_g", GROUP, device="cpu",
                calib_meta={KEY_CALIB_SEED: "0"})  # fmt: skip
    with pytest.raises(ValueError, match="calib_meta"):
        convert(FIXTURE_DIR, tmp_path / "q", "int4_g", GROUP, device="cpu", awq=ids,
                calib_meta={KEY_SOURCE: "x"})  # fmt: skip
    assert not (tmp_path / "q").exists()


# Tensors of a Whisper checkpoint that are not matrices: stored FP16 whatever the scheme.
WHISPER_KEPT = (
    "model.encoder.conv1.weight",
    "model.encoder.conv2.weight",
    "model.encoder.embed_positions.weight",
    "model.decoder.embed_positions.weight",
    "model.encoder.layers.0.self_attn.k_proj.bias",
    "model.decoder.layers.1.encoder_attn_layer_norm.weight",
)
# Measured on the fixture's three prompts (rel-RMS of the logits against the FP16 reference):
# int8_pc 0.010 to 0.011, int4_g (group 32) 0.155 to 0.165. The limits leave about 3x of margin.
WHISPER_LOGIT_REL_RMS_MAX = {"int8_pc": 0.03, "int4_g": 0.5}


@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_convert_whisper_quantizes_only_its_matrices(tmp_path, scheme):
    out = tmp_path / scheme
    manifest = convert(WHISPER_FIXTURE_DIR, out, scheme, GROUP, device="cpu")
    source = load_file(str(WHISPER_FIXTURE_DIR / "model.safetensors"))
    stored = load_file(str(out / "model.safetensors"))
    fmt, loaded = load_dequantized(out)
    assert fmt == WeightFormat.make(scheme, GROUP) and set(loaded) == set(source)

    matrices = {n for n, w in source.items() if w.ndim == 2 and "embed_positions" not in n}
    assert set(manifest["tensors"]) == matrices
    assert not matrices & set(WHISPER_KEPT) and set(WHISPER_KEPT) <= set(source)
    assert {"model.decoder.embed_tokens.weight", "model.encoder.layers.0.fc1.weight"} <= matrices
    for name, w in source.items():
        if name in matrices:
            q, scale = quantize(w, fmt)
            assert torch.equal(stored[name], q) and torch.equal(stored[f"{name}.scale"], scale)
            assert torch.equal(loaded[name], dequantize(q, scale, fmt))
        else:
            # The fixture is FP16 already, so the tensor is stored unchanged.
            assert f"{name}.scale" not in stored
            assert stored[name].dtype == torch.float16 and torch.equal(stored[name], w)
            assert torch.equal(loaded[name], w.float())
    for file in ("config.json", GENERATION_FILE):
        assert (out / file).read_bytes() == (WHISPER_FIXTURE_DIR / file).read_bytes()

    # The reference model computes with the dequantized matrices.
    model, reference = load_whisper(out), load_whisper(WHISPER_FIXTURE_DIR)
    assert torch.equal(model.decoder.embed.weight, loaded["model.decoder.embed_tokens.weight"])
    assert torch.equal(model.encoder.conv1.weight, reference.encoder.conv1.weight)
    golden_manifest, goldens = golden.load(WHISPER_FIXTURE_DIR)
    for prompt in golden_manifest["prompts"]:
        features = goldens[f"{prompt['id']}/input_features"][None]
        ids = goldens[f"{prompt['id']}/input_ids"][None].long()
        actual, expected = model(features, ids)[0], reference(features, ids)[0]
        assert tolerance.rel_rms(actual, expected) <= WHISPER_LOGIT_REL_RMS_MAX[scheme]


def test_convert_refuses_awq_for_whisper(tmp_path):
    with pytest.raises(ValueError, match="Whisper checkpoint takes none"):
        convert(WHISPER_FIXTURE_DIR, tmp_path / "q", "int4_g", GROUP, device="cpu",
                awq=calibration_ids())  # fmt: skip
    assert not (tmp_path / "q").exists()


@pytest.mark.parametrize("name", ["whisper-base-ar-quran", "whisper-tiny-ar-quran", "whisper-tiny"])
def test_default_int4_groups_divide_every_whisper_row(name):
    arch = load_spec(name).arch
    fmt = WeightFormat.make("int4_g")
    for width in (arch["d_model"], arch["encoder_ffn_dim"], arch["decoder_ffn_dim"]):
        fmt.check_shape((1, width))
        WeightFormat.make("int8_pc").check_shape((1, width))
