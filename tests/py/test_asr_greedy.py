"""Greedy transcription through the speech engine's scheduler, token for token against the
reference on the fixture and against Hugging Face `generate` on real checkpoints."""

import itertools

import numpy as np
import pytest
import torch

from kvbound import golden
from kvbound.fixture import WHISPER_FIXTURE_DIR
from kvbound.models import load_spec
from kvbound.ref import WhisperRules, load_whisper

pytestmark = pytest.mark.gpu

# A divergence from Hugging Face counts as a near-tie where the FP32 reference barely prefers
# one token: both tokens within NEAR_TIE log-probability units of its best. FP16 alone moves the
# reference's top-2 gap by more than that: on the whisper-tiny clips, both Hugging Face FP16 and
# the engine shift it by up to 0.033 at some step (both have logits within 0.6% rel-RMS of the
# reference). A divergence is therefore accepted up to FP16_GAP_NOISE, and those above NEAR_TIE
# are counted and limited to MAX_NOISE_DIVERGENCES clips of 20.
NEAR_TIE = 0.01
FP16_GAP_NOISE = 0.05
MAX_NOISE_DIVERGENCES = 2


def fixture_requests(copies=2):
    """Features [n, n_mels, frames] and the reference's greedy ids of each, every golden prompt
    `copies` times."""
    manifest, tensors = golden.load(WHISPER_FIXTURE_DIR)
    ids = [p["id"] for p in manifest["prompts"]] * copies
    features = np.stack([tensors[f"{i}/input_features"].numpy() for i in ids])
    return features, [tensors[f"{i}/greedy_ids"].tolist() for i in ids]


def transcribe(engine, features, policy):
    """Prompt plus generated ids of each clip, after checking the engine's per-clip times."""
    prompt = engine.rules()["prompt"]
    results = engine.transcribe(features, policy=policy)
    assert len(results) == len(features)
    for result in results:
        times = result["token_s"]
        assert len(times) == len(result["ids"]) and times == sorted(times)
        assert times[0] > 0 and times[-1] <= result["seconds"]
        assert 0 < result["encode_s"] < result["seconds"]
    return [prompt + result["ids"] for result in results]


@pytest.mark.parametrize(
    ("policy", "graph", "fuse", "encode_batch"),
    list(itertools.product(("static", "continuous"), (True, False), (True, False), (1, 4))),
)
def test_fixture_greedy_equals_reference(kvb, policy, graph, fuse, encode_batch):
    features, expected = fixture_requests()
    engine = kvb.AsrEngine(
        str(WHISPER_FIXTURE_DIR),
        max_seqs=4,
        max_encode_batch=encode_batch,
        kv_block_size=4,
        graph=graph,
        fuse=fuse,
    )
    assert transcribe(engine, features, policy) == expected
    stats, graphs = engine.transcribe_stats(), engine.graph_stats()
    assert stats["preemptions"] == 0 and stats["encodes"] == len(features)
    # Six clips over four rows: at least two encoder steps, and one step encodes several clips
    # when the batch allows it.
    if encode_batch > 1:
        assert stats["encoder_steps"] < len(features)
    else:
        assert stats["encoder_steps"] == len(features)
    assert (graphs["replayed"] > 0) == graph
    # The engine runs again after a run, with the same result.
    assert transcribe(engine, features[:1], policy) == expected[:1]


@pytest.mark.parametrize("encode_batch", [1, 4])
def test_fixture_greedy_survives_preemption(kvb, encode_batch):
    """A Self pool that holds one full sequence: sequences are preempted, encoded again when
    readmitted, and recomputed, and the tokens do not change."""
    features, expected = fixture_requests(copies=3)
    cfg = load_whisper(WHISPER_FIXTURE_DIR).cfg
    engine = kvb.AsrEngine(
        str(WHISPER_FIXTURE_DIR),
        max_seqs=4,
        max_encode_batch=encode_batch,
        kv_block_size=4,
        kv_blocks=cfg.max_target_positions // 4,
        max_graphs=2,
    )
    assert transcribe(engine, features, "continuous") == expected
    stats = engine.transcribe_stats()
    assert stats["preemptions"] > 0 and stats["encodes"] > len(features)
    assert engine.graph_stats()["evicted"] > 0


def test_fixture_greedy_quantized_matches_reference_on_its_weights(kvb, tmp_path):
    """INT8 weights: the engine's greedy ids are the reference's on the dequantized weights."""
    from kvbound import quantize
    from kvbound.ref import whisper as ref_whisper

    quantize.convert(WHISPER_FIXTURE_DIR, tmp_path, "int8_pc")
    model = load_whisper(tmp_path)
    rules = WhisperRules.from_json(tmp_path / "generation.json")
    features, _ = fixture_requests(copies=1)
    expected = [ref_whisper.greedy(model, torch.from_numpy(f), rules) for f in features]
    engine = kvb.AsrEngine(str(tmp_path), max_seqs=3, max_encode_batch=3)
    assert transcribe(engine, features, "continuous") == expected


def test_transcribe_rejects_what_it_cannot_run(kvb):
    cfg = load_whisper(WHISPER_FIXTURE_DIR).cfg
    engine = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR))
    with pytest.raises(RuntimeError, match="each clip needs features"):
        engine.transcribe(np.zeros((cfg.n_mels, cfg.n_frames - 1), dtype=np.float32))
    with pytest.raises(RuntimeError, match="unknown scheduling policy"):
        engine.transcribe(np.zeros((cfg.n_mels, cfg.n_frames), dtype=np.float32), policy="x")
    bad = np.zeros((cfg.n_mels, cfg.n_frames), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        engine.transcribe(bad)
    # Nothing of a failed run stays in the engine.
    features, expected = fixture_requests(copies=1)
    assert transcribe(engine, features[:1], "static") == expected[:1]


# --- Real checkpoints ----------------------------------------------------------------------------


def first_divergence(a, b):
    """Index of the first position where the two id lists differ, or None if they are equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def reference_gap(model, features, rules, ids, at, chosen):
    """How far below the reference's best each of `chosen` lies at position `at` of ids, in FP32
    log-probabilities after the suppress rules: the larger of the two shortfalls."""
    with torch.no_grad():
        logits = model(features[None], torch.tensor([ids[:at]]))[0, -1].float()
    logits[rules.suppress_tokens] = float("-inf")
    if at == len(rules.prompt):
        logits[rules.begin_suppress_tokens] = float("-inf")
    logprobs = torch.log_softmax(logits, -1)
    best = float(logprobs.max())
    return max(best - float(logprobs[token]) for token in chosen)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["whisper-base-ar-quran", "whisper-tiny"])
def test_real_model_greedy_equals_hf_or_near_tie(kvb, name):
    """The engine against Hugging Face `generate` (greedy, FP16 on the GPU) on the 20 local
    clips: the same ids, or a first divergence where the FP32 reference's margin between the two
    tokens is within FP16 noise."""
    from kvbound import asr, audio
    from kvbound.baselines import hf_whisper
    from kvbound.evals import asr as evals_asr

    spec = load_spec(name)
    if not (spec.checkpoint_dir / "model.safetensors").exists():
        pytest.skip(f"run `make fetch MODEL={name}`")
    try:
        clips = evals_asr.load_clips()
    except FileNotFoundError:
        pytest.skip("run `python -m kvbound.evals.asr fetch`")
    rules = WhisperRules.from_dict(spec.generation)
    features = [audio.features(clip.audio(), spec.arch["n_mels"]) for clip in clips]

    hf = hf_whisper.load(spec, "cuda", "fp16", 1)
    hf_ids = [hf(clip.audio()) for clip in clips]
    del hf
    torch.cuda.empty_cache()
    recognizer = asr.Recognizer(spec.checkpoint_dir, max_seqs=4, max_encode_batch=4)
    ours = recognizer.transcribe(np.stack([f.numpy() for f in features]), policy="continuous")

    model = load_whisper(spec.checkpoint_dir)
    exact, ties, noise = 0, [], []
    for clip, f, theirs, mine in zip(clips, features, hf_ids, ours, strict=True):
        assert theirs.prompt == mine.prompt == rules.prompt
        a, b = rules.prompt + mine.tokens, rules.prompt + theirs.tokens
        at = first_divergence(a, b)
        if at is None:
            exact += 1
            continue
        # A sequence that is a prefix of the other ended there: it chose end-of-sequence.
        chosen = {
            a[at] if at < len(a) else rules.eos_token_id,
            b[at] if at < len(b) else rules.eos_token_id,
        }
        gap = reference_gap(model, f, rules, a, at, chosen)
        assert gap < FP16_GAP_NOISE, (clip.id, at, gap, chosen)
        (ties if gap < NEAR_TIE else noise).append((clip.id, at, round(gap, 4)))
    print(f"\n{name}: {exact} of {len(clips)} equal to HF; near-ties {ties}; above {noise}")
    assert len(noise) <= MAX_NOISE_DIVERGENCES
