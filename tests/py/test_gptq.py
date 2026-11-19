"""GPTQ rounding: the solver, the sequential pass over the fixture, and checkpoints it writes."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from kvbound.fixture import FIXTURE_DIR, HF_CONFIG
from kvbound.gptq import (
    GptqConfig,
    collect_hessians,
    inverse_cholesky,
    minmax_scales,
    quantize_model,
    solve,
)
from kvbound.quantize import MANIFEST_FILE, convert, load_dequantized, unpack_int4
from kvbound.ref import load_llama
from kvbound.weight_format import KEY_CALIB_DATASET, KEY_ROUNDING, VARIANTS, WeightFormat

# The fixture's hidden size is 64 and its intermediate size 128.
GROUP = 32
DEAD = 5


def correlated_hessian(n=128, tokens=4000, seed=0):
    """H of inputs whose channels are mixed and unevenly scaled, with channel DEAD never set."""
    gen = torch.Generator().manual_seed(seed)
    mix = 0.2 * torch.randn(n, n, generator=gen) + torch.eye(n)
    x = torch.randn(tokens, n, generator=gen) @ mix * 3 * torch.rand(n, generator=gen).pow(2)
    x[:, DEAD] = 0.0
    return (x.T @ x).double() / tokens


def damped(h, damp):
    """H as the solver factors it: diagonal 1 where it is 0, then damp * mean(diag) added."""
    h = h.clone()
    diag = torch.diagonal(h)
    diag[diag == 0] = 1.0
    diag += damp * diag.mean()
    return h


def output_error(w, w_hat, h):
    d = (w - w_hat).double()
    return ((d @ h) * d).sum().item()


def calibration_ids(n=16, length=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(HF_CONFIG["vocab_size"], (n, length), generator=gen, dtype=torch.int32)


def test_inverse_cholesky_gives_the_rows_of_every_reduced_inverse():
    h = correlated_hessian()
    u = inverse_cholesky(h, 0.01)
    hd = damped(h, 0.01)
    inv = torch.linalg.inv(hd)
    assert torch.equal(u, u.triu())
    assert (u.T @ u - inv).abs().max() <= 1e-6 * inv.abs().max()
    # Row j of the inverse over columns j.. is U[j, j] * U[j, j:]: no elimination is needed.
    for j in (1, 40, 100):
        reduced = torch.linalg.inv(hd[j:, j:])[0]
        assert (reduced - u[j, j] * u[j, j:]).abs().max() <= 1e-6 * reduced.abs().max()
    # The dead channel moves no other weight.
    assert u[DEAD, DEAD + 1 :].abs().max() < 1e-12 and u[:DEAD, DEAD].abs().max() < 1e-12


@pytest.mark.parametrize(
    "config",
    [GptqConfig(act_order=False), GptqConfig(act_order=False, static_groups=True), GptqConfig()],
    ids=["input_order", "input_order_static", "act_order"],
)
def test_solver_lowers_the_output_error_on_the_int4_grid(config):
    h = correlated_hessian()
    w = torch.randn(64, 128, generator=torch.Generator().manual_seed(1))
    q, scale, loss = solve(w, h, GROUP, None, config)
    assert q.dtype == torch.int8 and q.shape == w.shape and int(q.abs().max()) <= 7
    assert scale.dtype == torch.float16 and scale.shape == (64, 128 // GROUP)
    deq = q.float() * scale.float().repeat_interleave(GROUP, dim=1)

    rtn_scale = minmax_scales(w, GROUP)
    if config.static_groups or config.act_order:
        assert torch.equal(scale, rtn_scale)
    rtn = torch.clamp(torch.round(w.reshape(64, -1, GROUP) / rtn_scale.float()[..., None]), -7, 7)
    rtn_deq = (rtn * rtn_scale.float()[..., None]).reshape(w.shape)
    assert output_error(w, deq, h) < 0.7 * output_error(w, rtn_deq, h)
    # The summed e^2 is the final output error under the damped H the solver factors.
    assert output_error(w, deq, damped(h, config.damp)) == pytest.approx(loss, rel=1e-4)


def test_solver_keeps_given_scales_and_chooses_integers_against_them():
    h = correlated_hessian()
    w = torch.randn(16, 128, generator=torch.Generator().manual_seed(2))
    given = (0.8 * minmax_scales(w, GROUP).float()).half()
    q, scale, _ = solve(w, h, GROUP, given)
    assert torch.equal(scale, given)
    # Without correlations GPTQ moves nothing, and so rounds to nearest on those scales.
    q, _, _ = solve(w, torch.diag(torch.diagonal(h)), GROUP, given)
    rtn = torch.clamp(torch.round(w.reshape(16, -1, GROUP) / given.float()[..., None]), -7, 7)
    assert torch.equal(q.float(), rtn.reshape(w.shape))
    with pytest.raises(ValueError, match="F16"):
        solve(w, h, GROUP, given.float())


def test_matrices_that_share_an_input_share_a_hessian():
    lin_a, lin_b = torch.nn.Linear(8, 4), torch.nn.Linear(8, 4)
    x = torch.randn(4, 3, 8, generator=torch.Generator().manual_seed(3))
    hs = collect_hessians(
        lambda rows: (lin_a(x[rows]), lin_b(x[rows])), 4, {"a": lin_a, "b": lin_b}
    )
    flat = x.reshape(-1, 8).double()
    # Each chunk is multiplied in float32 and summed in float64.
    assert torch.allclose(hs["a"], flat.T @ flat / flat.shape[0], rtol=1e-5, atol=1e-6)
    assert torch.allclose(hs["a"], hs["b"])
    assert not lin_a._forward_pre_hooks and not lin_b._forward_pre_hooks


@pytest.mark.parametrize("act_order", [True, False])
def test_every_fixture_matrix_beats_rounding_to_nearest(act_order):
    model = load_llama(FIXTURE_DIR, "fp16")
    result = quantize_model(model, calibration_ids(), GROUP, "cpu", GptqConfig(act_order=act_order))
    n_layers = model.cfg.n_layers
    # Seven matrices per layer; the tied lm_head is the embedding, which rounds to nearest.
    assert len(result.quantized) == 7 * n_layers and len(result.report) == n_layers
    for layer in result.report:
        assert set(layer) == {"attn", "o", "mlp", "down"}
        for entry in layer.values():
            for name, m in entry["matrices"].items():
                assert m["error"] < m["error_rtn"], name
    # The model comes back computing with what is stored.
    packed, scale = result.quantized["model.layers.1.mlp.down_proj.weight"]
    deq = unpack_int4(packed).float() * scale.float().repeat_interleave(GROUP, dim=1)
    assert torch.equal(model.layers[1].mlp.down_proj.weight, deq)


def test_gptq_metadata():
    for clip, act_scales, variant in (
        ("minmax", "", "int4_g.gptq"),
        ("awq", "awq", "int4_g.awq-gptq"),
        ("minmax", "awq", "int4_g.awq-noclip-gptq"),
    ):
        fmt = WeightFormat.make("int4_g", GROUP, clip, act_scales, "gptq")
        assert fmt.to_metadata()[KEY_ROUNDING] == "gptq"
        assert WeightFormat.from_metadata(fmt.to_metadata()) == fmt
        assert fmt.variant == variant and variant in VARIANTS
    # Rounding to nearest writes no key.
    assert KEY_ROUNDING not in WeightFormat.make("int4_g", GROUP).to_metadata()
    for scheme, clip, rounding in (
        ("int8_pc", "minmax", "gptq"),
        ("int4_g", "mse", "gptq"),
        ("int4_g", "minmax", "other"),
    ):
        with pytest.raises(ValueError):
            WeightFormat.make(scheme, GROUP, clip, "", rounding)


@pytest.mark.parametrize("awq", [False, True], ids=["gptq", "awq-gptq"])
def test_convert_with_gptq(tmp_path, awq):
    ids = calibration_ids()
    clip = "awq" if awq else "minmax"
    calib_meta = {KEY_CALIB_DATASET: "random"}
    out = tmp_path / "q"
    convert(
        FIXTURE_DIR,
        out,
        "int4_g",
        GROUP,
        clip,
        device="cpu",
        calib=ids,
        awq=awq,
        gptq=GptqConfig(),
        calib_meta=calib_meta,
    )
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        metadata = f.metadata()
    fmt = WeightFormat.from_metadata(metadata)
    assert fmt == WeightFormat.make("int4_g", GROUP, clip, "awq" if awq else "", "gptq")
    assert calib_meta.items() <= metadata.items()
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["variant"] == fmt.variant and manifest["gptq"]["config"]["damp"] == 0.01

    # The reference model reads it like any int4_g file.
    model = load_llama(out)
    _, tensors = load_dequantized(out)
    assert torch.equal(
        model.layers[0].attn.o_proj.weight, tensors["model.layers.0.self_attn.o_proj.weight"]
    )
    stored = load_file(str(out / "model.safetensors"))
    plain_dir = tmp_path / "plain"
    convert(
        FIXTURE_DIR,
        plain_dir,
        "int4_g",
        GROUP,
        clip,
        device="cpu",
        calib=ids if awq else None,
        awq=awq,
    )
    plain = load_file(str(plain_dir / "model.safetensors"))
    # The embedding rounds to nearest; the layers' integers differ; AWQ's scales are kept.
    assert torch.equal(stored["model.embed_tokens.weight"], plain["model.embed_tokens.weight"])
    name = "model.layers.0.mlp.down_proj.weight"
    assert not torch.equal(stored[name], plain[name])
    if awq:
        assert torch.equal(stored[name + ".scale"], plain[name + ".scale"])
        norm = "model.layers.0.input_layernorm.weight"
        assert torch.equal(stored[norm], plain[norm])


def test_convert_takes_calibration_tokens_only_with_awq_or_gptq(tmp_path):
    with pytest.raises(ValueError, match="calib is given"):
        convert(FIXTURE_DIR, tmp_path / "q", "int4_g", GROUP, device="cpu", gptq=GptqConfig())
    with pytest.raises(ValueError, match="calib is given"):
        convert(FIXTURE_DIR, tmp_path / "q", "int4_g", GROUP, device="cpu", calib=calibration_ids())
    assert not (tmp_path / "q").exists()
