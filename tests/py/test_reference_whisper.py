"""The from-scratch Whisper reference against Hugging Face and against the committed goldens."""

import io
import json
import shutil
import wave
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import WhisperForConditionalGeneration

from kvbound import audio, golden, tolerance
from kvbound.evals import asr
from kvbound.fixture import GREEDY_MARGIN, WHISPER_FIXTURE_DIR
from kvbound.models import GENERATION_FILE, convert_whisper, load_spec
from kvbound.ref import WhisperConfig, WhisperRules, load_whisper
from kvbound.ref import whisper as ref_whisper


def load_hf(directory):
    return WhisperForConditionalGeneration.from_pretrained(
        directory, dtype=torch.float32, attn_implementation="eager"
    ).eval()


def hf_greedy(hf, features, **kwargs):
    """The whole sequence Hugging Face generates without sampling or beams, prompt included."""
    with torch.no_grad():
        out = hf.generate(
            features[None], do_sample=False, num_beams=1, return_dict_in_generate=True, **kwargs
        )
    return out.sequences[0].tolist()


def assert_matches_hf(ref, hf, features, input_ids, thresholds):
    """Compares every hidden state Hugging Face returns, and the logits, for one prompt."""
    ids = torch.tensor([input_ids])
    capture = {}
    ref(features[None], ids, capture)
    with torch.no_grad():
        out = hf(input_features=features[None], decoder_input_ids=ids, output_hidden_states=True)

    cfg = ref.cfg
    pairs = [("logits", out.logits), ("encoder.final_norm", out.encoder_last_hidden_state)]
    # HF hidden_states of each side: [embed, layer outputs..., final-norm output].
    for side, n_layers, states in (
        ("encoder", cfg.encoder_layers, out.encoder_hidden_states),
        ("decoder", cfg.decoder_layers, out.decoder_hidden_states),
    ):
        assert len(states) == n_layers + 1
        pairs += [(f"{side}.embed", states[0]), (f"{side}.final_norm", states[-1])]
        pairs += [(f"{side}.layers.{i}.resid_post", states[i + 1]) for i in range(n_layers - 1)]
    for site, expected in pairs:
        assert capture[site].shape == expected.shape, site
        assert tolerance.rel_rms(capture[site], expected) <= thresholds["rel_rms_max"], site
    assert tolerance.top1_agreement(capture["logits"], out.logits) == 1.0


@pytest.fixture(scope="module")
def fixture_goldens():
    return golden.load(WHISPER_FIXTURE_DIR)


@pytest.fixture(scope="module")
def fixture_rules():
    return WhisperRules.from_json(WHISPER_FIXTURE_DIR / GENERATION_FILE)


def test_reference_matches_hf_on_fixture(fixture_goldens):
    manifest, tensors = fixture_goldens
    ref = load_whisper(WHISPER_FIXTURE_DIR, weights="native")
    hf = load_hf(WHISPER_FIXTURE_DIR)
    for prompt in manifest["prompts"]:
        features = tensors[f"{prompt['id']}/input_features"]
        ids = tensors[f"{prompt['id']}/input_ids"].tolist()
        assert_matches_hf(ref, hf, features, ids, manifest["thresholds"]["fp32"])


def test_committed_goldens_match_reference(fixture_goldens):
    manifest, tensors = fixture_goldens
    fp32 = manifest["thresholds"]["fp32"]
    model = load_whisper(WHISPER_FIXTURE_DIR, weights=manifest["model"]["weights_precision"])
    cfg = model.cfg
    assert manifest["model"]["config"] == cfg.to_dict()
    assert list(manifest["thresholds"]["fp16"]["rel_rms_max"]) == ref_whisper.site_names(cfg)
    for prompt in manifest["prompts"]:
        features = tensors[f"{prompt['id']}/input_features"]
        ids = tensors[f"{prompt['id']}/input_ids"].tolist()
        assert features.shape == (cfg.n_mels, prompt["n_frames"]) == (cfg.n_mels, cfg.n_frames)
        assert len(ids) == prompt["n_tokens"]
        sites = golden.capture_whisper_sites(model, features, ids)
        assert list(sites) == ref_whisper.site_names(cfg)
        for site, value in sites.items():
            expected = tensors[f"{prompt['id']}/{site}"]
            assert value.shape == expected.shape, site
            assert tolerance.rel_rms(value, expected) <= fp32["rel_rms_max"], site
            assert tolerance.cosine(value, expected) >= fp32["cosine_min"], site


def test_golden_site_shapes(fixture_goldens):
    manifest, tensors = fixture_goldens
    cfg = WhisperConfig.from_json(WHISPER_FIXTURE_DIR / "config.json")
    n_ctx, d = cfg.max_source_positions, cfg.d_model
    for prompt in manifest["prompts"]:

        def shape(site, prompt=prompt):
            return tuple(tensors[f"{prompt['id']}/{site}"].shape)

        # Encoder sites have a row per frame, decoder sites a row per token.
        assert shape("encoder.conv1") == (2 * n_ctx, d)
        assert shape("encoder.conv2") == shape("encoder.final_norm") == (n_ctx, d)
        assert shape("decoder.embed") == (prompt["n_tokens"], d)
        assert shape("logits") == (prompt["n_tokens"], cfg.vocab_size)
        for j in range(cfg.decoder_layers):
            # Keys in the first d_model columns, values in the rest.
            assert shape(f"encoder.cross_kv.{j}") == (n_ctx, 2 * d)


def test_greedy_matches_hf_on_fixture(fixture_goldens, fixture_rules):
    manifest, tensors = fixture_goldens
    rules = fixture_rules
    ref = load_whisper(WHISPER_FIXTURE_DIR, weights="native")
    hf = load_hf(WHISPER_FIXTURE_DIR)
    paths = {}
    for prompt in manifest["prompts"]:
        features = tensors[f"{prompt['id']}/input_features"]
        scores = []
        ids = paths[prompt["id"]] = ref_whisper.greedy(ref, features, rules, scores)
        assert ids == hf_greedy(hf, features)
        assert ids == tensors[f"{prompt['id']}/greedy_ids"].tolist()
        assert ids[: len(rules.prompt)] == rules.prompt
        assert len(scores) == len(ids) - len(rules.prompt)
        assert ref_whisper.top2_gap(scores) >= GREEDY_MARGIN

    # Between them the prompts apply every rule (see fixture.whisper_prompts).
    def without(prompt_id, **lists):
        features = tensors[f"{prompt_id}/input_features"]
        return ref_whisper.greedy(ref, features, replace(rules, **lists))

    assert len(paths["p0"]) == rules.max_length and paths["p0"][-1] != rules.eos_token_id
    assert without("p0", suppress_tokens=[]) != paths["p0"]
    assert paths["p1"][-1] == rules.eos_token_id and len(paths["p1"]) < rules.max_length
    assert without("p2", begin_suppress_tokens=[]) != paths["p2"]
    assert not set(rules.suppress_tokens) & {t for ids in paths.values() for t in ids[4:]}


def test_greedy_stops_at_another_eos_like_hf(fixture_goldens, fixture_rules):
    tensors = fixture_goldens[1]
    ref = load_whisper(WHISPER_FIXTURE_DIR, weights="native")
    hf = load_hf(WHISPER_FIXTURE_DIR)
    features = tensors["p0/input_features"]
    path = tensors["p0/greedy_ids"].tolist()
    # Makes the seventh generated token the end of the sequence.
    eos = path[len(fixture_rules.prompt) + 6]
    ids = ref_whisper.greedy(ref, features, replace(fixture_rules, eos_token_id=eos))
    assert ids == path[: path.index(eos, len(fixture_rules.prompt)) + 1]
    assert ids == hf_greedy(hf, features, eos_token_id=eos)


def fixture_as_hf_checkpoint(directory):
    """Writes the fixture the way Hugging Face stores a Whisper: a pickle without key biases."""
    directory.mkdir()
    state = load_file(str(WHISPER_FIXTURE_DIR / "model.safetensors"))
    state = {name: w for name, w in state.items() if not name.endswith(".k_proj.bias")}
    shutil.copyfile(WHISPER_FIXTURE_DIR / "config.json", directory / "config.json")
    return state


def test_convert_whisper_reproduces_the_fixture(tmp_path, fixture_rules):
    source, out = tmp_path / "hf", tmp_path / "converted"
    state = fixture_as_hf_checkpoint(source)
    state["proj_out.weight"] = state["model.decoder.embed_tokens.weight"].clone()
    torch.save(state, source / "pytorch_model.bin")
    convert_whisper(source, out, fixture_rules.to_dict())

    expected = load_file(str(WHISPER_FIXTURE_DIR / "model.safetensors"))
    converted = load_file(str(out / "model.safetensors"))
    assert set(converted) == set(expected)
    assert all(torch.equal(converted[name], expected[name]) for name in expected)
    assert WhisperRules.from_json(out / GENERATION_FILE) == fixture_rules
    assert (out / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert not (out / "tokenizer.json").exists()
    load_whisper(out)


def test_convert_whisper_refuses_an_untied_output_projection(tmp_path, fixture_rules):
    source = tmp_path / "hf"
    state = fixture_as_hf_checkpoint(source)
    state["proj_out.weight"] = state["model.decoder.embed_tokens.weight"] + 1
    torch.save(state, source / "pytorch_model.bin")
    with pytest.raises(ValueError, match=r"proj_out\.weight is not tied"):
        convert_whisper(source, tmp_path / "converted", fixture_rules.to_dict())


def test_whisper_config_refuses_unsupported_variants():
    raw = json.loads((WHISPER_FIXTURE_DIR / "config.json").read_text(encoding="utf-8"))
    for change in ({"scale_embedding": True}, {"activation_function": "relu"}):
        with pytest.raises(ValueError, match="unsupported config fields"):
            WhisperConfig.from_hf_dict(raw | change)


def test_rules_must_fit_the_model(fixture_rules):
    cfg = WhisperConfig.from_json(WHISPER_FIXTURE_DIR / "config.json")
    fixture_rules.check(cfg)
    for change in (
        {"max_length": cfg.max_target_positions + 1},
        {"max_length": len(fixture_rules.prompt)},
        {"suppress_tokens": [cfg.vocab_size]},
        {"prompt": []},
    ):
        with pytest.raises(ValueError):
            replace(fixture_rules, **change).check(cfg)


def test_features_pad_a_clip_to_one_window():
    seconds = 2
    t = np.arange(seconds * audio.SAMPLE_RATE) / audio.SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    features = audio.features(tone)
    assert features.shape == (80, 3000) and features.dtype == torch.float32
    # The zero padding after the clip is the quietest value: 8 below the peak in log10, scaled.
    assert torch.all(features[:, 100 * seconds + 2 :] == features.min())
    assert features.max() - features.min() == pytest.approx(2.0)
    assert audio.features(tone, n_mels=16).shape == (16, 3000)


def wav_bytes(pcm, rate):
    data = io.BytesIO()
    with wave.open(data, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(pcm.dtype.itemsize)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())
    return data.getvalue()


def test_decode_wav_reads_16_bit_mono():
    pcm = np.array([0, 1, -1, 16384, -32768, 32767], dtype="<i2")
    samples = audio.decode_wav(wav_bytes(pcm, audio.SAMPLE_RATE))
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, pcm / 32768.0)
    with pytest.raises(ValueError, match="expected"):
        audio.decode_wav(wav_bytes(pcm, 8000))


# Keyword arguments that make Hugging Face `generate` follow each model's toml [generation].
REAL_MODELS = {
    "whisper-tiny": {"language": "ar", "task": "transcribe"},
    "whisper-base-ar-quran": {},
}
# Whether fine-tuning moved the encoder's position table away from Whisper's fixed sinusoids.
TRAINED_POSITIONS = {"whisper-tiny": False, "whisper-base-ar-quran": True}
N_CLIPS = 3


@pytest.fixture(scope="module", params=list(REAL_MODELS))
def real_model(request):
    """A fetched model: its spec and rules, the reference model, and the Hugging Face model
    loaded from the unconverted download."""
    spec = load_spec(request.param)
    if not (spec.checkpoint_dir / "model.safetensors").exists():
        pytest.skip(f"{spec.name} is not fetched; run `make fetch MODEL={spec.name}`")
    rules = WhisperRules.from_json(spec.checkpoint_dir / GENERATION_FILE)
    assert rules == WhisperRules.from_dict(spec.generation)
    ref = load_whisper(spec.checkpoint_dir, weights="native")
    return SimpleNamespace(spec=spec, rules=rules, ref=ref, hf=load_hf(spec.source_dir))


@pytest.fixture(scope="module")
def clip_features():
    """Log-mel features of N_CLIPS recitation clips, each from another reciter."""
    try:
        clips = asr.load_clips()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    first: dict[str, asr.Clip] = {}
    for clip in clips:
        first.setdefault(clip.reciter, clip)
    return [audio.features(clip.audio()) for clip in list(first.values())[:N_CLIPS]]


@pytest.mark.slow
def test_reference_matches_hf_on_real_model(real_model, clip_features):
    m = real_model
    for features in clip_features:
        ids = hf_greedy(m.hf, features, **REAL_MODELS[m.spec.name])
        assert_matches_hf(m.ref, m.hf, features, ids, golden.FP32_THRESHOLDS)


@pytest.mark.slow
def test_greedy_matches_hf_on_real_model(real_model, clip_features):
    spec, rules, ref = real_model.spec, real_model.rules, real_model.ref
    tokenizer = Tokenizer.from_file(str(spec.checkpoint_dir / "tokenizer.json"))
    for features in clip_features:
        ids = hf_greedy(real_model.hf, features, **REAL_MODELS[spec.name])
        # The prompt Hugging Face feeds is the toml's.
        assert ids[: len(rules.prompt)] == rules.prompt
        assert ids[-1] == rules.eos_token_id and len(ids) < rules.max_length
        assert ref_whisper.greedy(ref, features, rules) == ids
        # The converted tokenizer treats every control token as special.
        text = tokenizer.decode(ids, skip_special_tokens=True)
        assert text.strip() and "<|" not in text


@pytest.mark.slow
def test_hf_generation_settings_equal_the_toml(real_model):
    rules, config = real_model.rules, real_model.hf.generation_config
    assert (config.suppress_tokens or []) == rules.suppress_tokens
    assert config.begin_suppress_tokens == rules.begin_suppress_tokens
    assert config.eos_token_id == rules.eos_token_id
    assert config.decoder_start_token_id == rules.prompt[0]
    # generate() allows min(max_length + prompt length, max_target_positions) tokens in all.
    n_prompt = len(rules.prompt)
    max_target_positions = real_model.ref.cfg.max_target_positions
    assert min(config.max_length + n_prompt, max_target_positions) == rules.max_length


@pytest.mark.slow
def test_stored_encoder_positions_against_sinusoids(real_model):
    stored = load_file(str(real_model.spec.checkpoint_dir / "model.safetensors"))
    positions = stored["model.encoder.embed_positions.weight"]
    cfg = real_model.ref.cfg
    expected = ref_whisper.sinusoids(cfg.max_source_positions, cfg.d_model)
    assert positions.shape == expected.shape
    deviation = (positions - expected).abs().max().item()
    if TRAINED_POSITIONS[real_model.spec.name]:
        # Near the sinusoids but not them: the table must be read, not recomputed.
        assert 1e-3 < deviation < 0.05
    else:
        # The sinusoids as FP16 stores them: within one FP16 step below 1.
        assert deviation <= 2.0**-11
        assert torch.equal(positions, positions.half().float())


@pytest.mark.slow
def test_converted_checkpoint_layout(real_model):
    cfg = real_model.ref.cfg
    stored = load_file(str(real_model.spec.checkpoint_dir / "model.safetensors"))
    assert set(stored) == set(ref_whisper.hf_weight_names(cfg).values())
    key_biases = [name for name in stored if name.endswith(".k_proj.bias")]
    assert len(key_biases) == cfg.encoder_layers + 2 * cfg.decoder_layers
    assert all(not stored[name].any() for name in key_biases)
    # Every other tensor is the downloaded one, bit for bit.
    theirs = real_model.hf.state_dict()
    assert all(torch.equal(w, theirs[name]) for name, w in stored.items() if name in theirs)
    assert set(stored) - set(theirs) == set(key_biases)
