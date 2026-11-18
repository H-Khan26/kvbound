"""The speech engine: every capture site of the encoder and of the teacher-forced decoder against
the goldens."""

import numpy as np
import pytest
import torch

from kvbound import golden, quantize, tolerance
from kvbound.fixture import WHISPER_FIXTURE_DIR
from kvbound.models import load_spec
from kvbound.ref import load_whisper
from kvbound.ref import whisper as ref_whisper

pytestmark = pytest.mark.gpu


def decoder_sites(engine, ids, split, seq=0):
    """Feeds ids[:split] in one step, then the rest one token per step, capturing every site."""
    parts = [engine.feed(ids[:split], start=0, seq=seq, capture=True)]
    parts += [
        engine.feed([t], start=i, seq=seq, capture=True) for i, t in enumerate(ids) if i >= split
    ]
    return {site: np.concatenate([part[site] for part in parts]) for site in parts[0]}


def assert_sites_close(sites, expected, limits, label):
    """Per-site rel-RMS within the limits; returns the argmax mismatches of the logits, if any."""
    for site, value in sites.items():
        assert value.shape == tuple(expected[site].shape), (label, site)
        error = tolerance.rel_rms(torch.from_numpy(value), expected[site])
        assert error <= limits["rel_rms_max"][site], (label, site)
    if "logits" not in sites:
        return 0
    logits = torch.from_numpy(sites["logits"])
    assert tolerance.mean_kl(logits, expected["logits"]) <= limits["mean_kl_max"], label
    return int((logits.argmax(-1) != expected["logits"].argmax(-1)).sum())


def golden_sites(tensors, prompt_id):
    prefix = f"{prompt_id}/"
    return {name.removeprefix(prefix): t for name, t in tensors.items() if name.startswith(prefix)}


def assert_matches_goldens(engine, manifest, tensors, splits):
    """Every site of every prompt within the manifest's FP16 limits: the encoder's from one
    encoder step per clip, the decoder's for each way of splitting the ids into steps."""
    limits = manifest["thresholds"]["fp16"]
    names = set(limits["rel_rms_max"])
    for split_of in splits:
        mismatches = 0
        for prompt in manifest["prompts"]:
            expected = golden_sites(tensors, prompt["id"])
            ids = expected["input_ids"].tolist()
            sites = engine.encode(expected["input_features"].numpy(), capture=True)
            sites |= decoder_sites(engine, ids, split_of(len(ids)))
            assert set(sites) == names
            mismatches += assert_sites_close(sites, expected, limits, prompt["id"])
        assert mismatches <= limits["top1_mismatches_max"]


# Whole sequence as one step; one token per step; a multi-row step followed by decode steps.
SPLITS = (lambda n: n, lambda n: 1, lambda n: max(1, n // 2))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"fuse": False},
        {"plan": "naive"},
        {"kv_block_size": 4},
        {"kv_layout": "token_major"},
        {"kv_layout": "dim_major", "kv_block_size": 8},
        {"f16_gemv_rows": 0},
        {"fuse": False, "f16_gemv_rows": 0, "gemm": "fp16"},
        {"prefill_attention": False},
        {"wide_key_loads": False, "kv_block_size": 4},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "default",
)
def test_fixture_sites_match_goldens(kvb, options):
    engine = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), **options)
    manifest, tensors = golden.load(WHISPER_FIXTURE_DIR)
    assert set(manifest["thresholds"]["fp16"]["rel_rms_max"]) == set(
        ref_whisper.site_names(load_whisper(WHISPER_FIXTURE_DIR).cfg)
    )
    assert_matches_goldens(engine, manifest, tensors, SPLITS)


def test_clips_share_an_encoder_step(kvb):
    """Three clips through one encoder step, into block-table rows in another order: each
    clip's encoder sites are its goldens, and each row then decodes against its own clip."""
    manifest, tensors = golden.load(WHISPER_FIXTURE_DIR)
    limits = manifest["thresholds"]["fp16"]
    prompts = [golden_sites(tensors, prompt["id"]) for prompt in manifest["prompts"]]
    features = np.stack([p["input_features"].numpy() for p in prompts])
    rows = [2, 0, 1]
    for options in ({}, {"fuse": False}):
        engine = kvb.AsrEngine(
            str(WHISPER_FIXTURE_DIR), max_seqs=3, max_encode_batch=3, kv_block_size=8, **options
        )
        sites = engine.encode(features, seqs=rows, capture=True)
        for clip, (expected, row) in enumerate(zip(prompts, rows, strict=True)):
            own = {site: np.split(value, len(prompts))[clip] for site, value in sites.items()}
            assert_sites_close(own, expected, limits, clip)
            ids = expected["input_ids"].tolist()
            assert_sites_close(decoder_sites(engine, ids, len(ids), row), expected, limits, clip)
        # A later step of two clips replaces the clips of rows 0 and 2; row 1 keeps clip 2.
        engine.encode(features[[0, 1]], seqs=[0, 2])
        for clip, row in ((0, 0), (2, 1), (1, 2)):
            ids = prompts[clip]["input_ids"].tolist()
            assert_sites_close(decoder_sites(engine, ids, 1, row), prompts[clip], limits, clip)


def test_decoder_outputs_follow_the_suppress_rules(kvb):
    """Without capture, feed() returns the logits the argmax reads: the tokens the rules
    suppress are -inf, the begin list only in the row of the last prompt token."""
    manifest, tensors = golden.load(WHISPER_FIXTURE_DIR)
    limit = manifest["thresholds"]["fp16"]["rel_rms_max"]["logits"]
    engine = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR))
    rules = engine.rules()
    assert (
        rules
        == ref_whisper.WhisperRules.from_json(WHISPER_FIXTURE_DIR / "generation.json").to_dict()
    )
    for prompt in manifest["prompts"]:
        expected = golden_sites(tensors, prompt["id"])
        ids = expected["input_ids"].tolist()
        engine.encode(expected["input_features"].numpy())
        logits = torch.from_numpy(engine.feed(ids)["logits"])
        masked = expected["logits"].clone()
        masked[:, rules["suppress_tokens"]] = float("-inf")
        masked[len(rules["prompt"]) - 1, rules["begin_suppress_tokens"]] = float("-inf")
        assert torch.equal(logits == float("-inf"), masked == float("-inf"))
        kept = masked != float("-inf")
        assert tolerance.rel_rms(logits[kept], masked[kept]) <= limit
        assert engine.next_ids() == masked.argmax(-1).tolist()


@pytest.fixture(scope="module")
def quantized_fixture(tmp_path_factory):
    """The fixture checkpoint in each quantized variant. INT4 uses groups of 32, so that a row
    of the fixture's narrow matrices holds several."""
    dirs = {}
    for scheme in ("int8_pc", "int4_g"):
        dirs[scheme] = tmp_path_factory.mktemp(scheme)
        quantize.convert(WHISPER_FIXTURE_DIR, dirs[scheme], scheme, group_size=32)
    return dirs


@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_quantized_weights_match_the_dequantizing_reference(kvb, quantized_fixture, scheme):
    """The engine on quantized matrices against the reference model computing in FP32 on the
    same weights dequantized, for every way of multiplying: in place, on tensor cores
    (quantized_gemm_rows=0 takes that path on every step), and expanded to FP16 for cuBLAS,
    whole and two output channels at a time."""
    model_dir = quantized_fixture[scheme]
    model = load_whisper(model_dir)
    manifest, tensors = golden.load(WHISPER_FIXTURE_DIR)
    limits = manifest["thresholds"]["fp16"]
    wide = {"quantized_gemm_rows": 0}
    expand = {**wide, "expand_quantized": True}
    for options in ({}, {"fuse": False}, wide, expand, {**expand, "dequant_slice_bytes": 512}):
        engine = kvb.AsrEngine(str(model_dir), kv_block_size=8, **options)
        assert engine.weights() == scheme
        for prompt in manifest["prompts"]:
            features = tensors[f"{prompt['id']}/input_features"]
            ids = tensors[f"{prompt['id']}/input_ids"].tolist()
            expected = golden.capture_whisper_sites(model, features, ids)
            for split in (len(ids), 1):
                sites = engine.encode(features.numpy(), capture=True)
                sites |= decoder_sites(engine, ids, split)
                assert set(sites) == set(expected)
                label = (prompt["id"], split, tuple(options))
                assert assert_sites_close(sites, expected, limits, label) <= 1
    # A quantized checkpoint holds its matrices at the size the format promises; the conv stem,
    # the position tables, the biases and the norms stay FP16.
    fp16_bytes = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR)).memory()["weights"]
    assert engine.memory()["weights"] < (0.66 if scheme == "int8_pc" else 0.50) * fp16_bytes


def test_int8_cross_cache_is_close_to_fp16(kvb):
    """Quantizing the Cross cache moves the logits by a small error and shrinks its pool."""
    manifest, tensors = golden.load(WHISPER_FIXTURE_DIR)
    fp16 = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR))
    int8 = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), cross_kv_dtype="int8", kv_dtype="int8")
    assert int8.config()["cross_kv_dtype"] == "int8" and fp16.config()["cross_kv_dtype"] == "fp16"
    for prompt in manifest["prompts"]:
        features = tensors[f"{prompt['id']}/input_features"].numpy()
        ids = tensors[f"{prompt['id']}/input_ids"].tolist()
        outputs = []
        for engine in (fp16, int8):
            engine.encode(features)
            outputs.append(torch.from_numpy(engine.feed(ids, capture=True)["logits"]))
        assert tolerance.rel_rms(outputs[1], outputs[0]) < 0.02
        assert tolerance.top1_agreement(outputs[1], outputs[0]) == 1.0
    cfg = manifest["model"]["config"]
    head_dim = cfg["d_model"] // cfg["decoder_heads"]
    for pool in ("self_kv", "cross_kv"):
        ratio = int8.memory()[pool] / fp16.memory()[pool]
        assert ratio == pytest.approx((head_dim + 2) / (2 * head_dim))


def test_engine_options_and_memory_are_visible(kvb):
    cfg = load_whisper(WHISPER_FIXTURE_DIR).cfg
    engine = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), max_seqs=3, max_encode_batch=2)
    config = engine.config()
    assert config["max_ctx"] == cfg.max_target_positions and config["kv_block_size"] == 16
    assert config["kv_blocks"] == 3 * cfg.max_target_positions // 16
    assert config["max_encode_batch"] == 2 and config["weights"] == "fp16"
    memory = engine.memory()
    assert memory["arena"] == max(memory["encoder_arena"], memory["decoder_arena"])
    # One block of every encoder position per sequence: keys and values of each decoder layer.
    cross = 3 * cfg.decoder_layers * cfg.max_source_positions * cfg.d_model * 2 * 2
    assert memory["cross_kv"] == cross


def test_engine_rejects_what_it_cannot_run(kvb):
    cfg = load_whisper(WHISPER_FIXTURE_DIR).cfg
    features = np.zeros((cfg.n_mels, cfg.n_frames), dtype=np.float32)
    engine = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), max_seqs=2)
    with pytest.raises(RuntimeError, match="each clip needs features"):
        engine.encode(features[:, :-2])
    with pytest.raises(RuntimeError, match="between 1 and 1 clips"):
        engine.encode(np.stack([features, features]))
    with pytest.raises(RuntimeError, match="distinct block-table row"):
        engine.encode(features, seqs=[2])
    bad = features.copy()
    bad[3, 7] = np.inf
    with pytest.raises(RuntimeError, match="finite"):
        engine.encode(bad)
    with pytest.raises(RuntimeError, match="out of range"):
        engine.feed([cfg.vocab_size])
    with pytest.raises(RuntimeError, match="out of range"):
        engine.feed([1], start=cfg.max_target_positions)
    for options, message in (
        ({"max_ctx": cfg.max_target_positions + 1}, "target positions"),
        ({"max_encode_batch": 2}, "max_encode_batch"),
        ({"kv_block_size": 12}, "power of two"),
    ):
        with pytest.raises(RuntimeError, match=message):
            kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), **options)
    # The engine still runs after each refusal.
    engine.encode(features)
    assert engine.feed([1, 2, 3])["logits"].shape == (3, cfg.vocab_size)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["whisper-base-ar-quran", "whisper-tiny"])
def test_real_model_sites_match_goldens(kvb, name):
    spec = load_spec(name)
    if not (spec.golden_dir / golden.GOLDEN_FILE).exists():
        pytest.skip(f"{name} has no goldens; run `make fetch golden MODEL={name}`")
    manifest, tensors = golden.load_for_model(name)
    engine = kvb.AsrEngine(str(spec.checkpoint_dir))
    assert_matches_goldens(engine, manifest, tensors, SPLITS[:2])


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["int8_pc", "int4_g"])
def test_real_model_quantized_sites_match_the_dequantizing_reference(kvb, scheme):
    """The Tarteel base model with quantized matrices, on its first golden clip, against the
    reference model computing in FP32 on the same matrices dequantized."""
    spec = load_spec("whisper-base-ar-quran")
    model_dir = spec.variant_dir(scheme)
    if not (model_dir / "model.safetensors").exists():
        pytest.skip(f"run `python -m kvbound.quantize {spec.name} --scheme {scheme}`")
    if not (spec.golden_dir / golden.GOLDEN_FILE).exists():
        pytest.skip(f"{spec.name} has no goldens; run `make golden MODEL={spec.name}`")
    manifest, tensors = golden.load_for_model(spec.name)
    limits = manifest["thresholds"]["fp16"]
    prompt = manifest["prompts"][0]["id"]
    features = tensors[f"{prompt}/input_features"]
    ids = tensors[f"{prompt}/input_ids"].tolist()
    expected = golden.capture_whisper_sites(load_whisper(model_dir), features, ids)

    engine = kvb.AsrEngine(str(model_dir))
    assert engine.weights() == scheme
    sites = engine.encode(features.numpy(), capture=True)
    sites |= decoder_sites(engine, ids, len(ids))
    assert set(sites) == set(expected)
    mismatches = assert_sites_close(sites, expected, limits, (scheme, prompt))
    assert mismatches <= limits["top1_mismatches_max"]
