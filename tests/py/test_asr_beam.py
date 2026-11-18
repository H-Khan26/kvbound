"""Beam-search transcription through the speech engine: against the reference beam search and its
committed traces on the fixture, with blocks shared or private, and against Hugging Face
`generate(num_beams=5)` on a real checkpoint."""

import functools
import json

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from kvbound import golden
from kvbound.fixture import BEAM_CASES, BEAM_TRACE_FILE, WHISPER_FIXTURE_DIR
from kvbound.models import load_spec
from kvbound.ref import WhisperRules, load_whisper

pytestmark = pytest.mark.gpu

# How close the engine's hypothesis scores come to the reference's: within these of the same
# hypothesis's reference score. Penalty 0 leaves scores as sums of up to 28 log-probabilities,
# where the error of each token adds up: the fixture's largest, 4.2e-3 (FP16 Self cache) and
# 1.6e-2 (INT8), 2.5e-2 in the dim_major layout, whose INT8 keys are read another way.
SCORE_TOLERANCE = {"fp16": 5e-3, "int8": 3e-2}
# A reference decision closer than this, in score, may go the other way in the engine, so the
# engine's ids are compared only where the reference's margin is wider. Two continuations of one
# search share most of their tokens, and with them most of their error: at length penalty 1 the
# engine moves the difference between two of the fixture's kept hypotheses by up to 4.7e-4 (FP16
# Self cache) and 9.7e-4 (INT8).
DECISION_BOUND = {"fp16": 5e-4, "int8": 1e-3}


@functools.cache
def traces():
    """The committed reference searches: per "<prompt>/<case>", its tensors and BeamConfig."""
    path = WHISPER_FIXTURE_DIR / BEAM_TRACE_FILE
    tensors = load_file(str(path))
    with safe_open(str(path), framework="pt") as f:
        configs = json.loads(f.metadata()["configs"])
    result = {}
    for name, config in configs.items():
        prefix = f"{name}/"
        own = {
            k.removeprefix(prefix): v.numpy() for k, v in tensors.items() if k.startswith(prefix)
        }
        result[name] = (own, config)
    assert result.keys() == {f"{case.prompt}/{case.name}" for case in BEAM_CASES}
    return result


@functools.cache
def features(prompt):
    """A golden prompt's features [n_mels, n_frames]."""
    return golden.load(WHISPER_FIXTURE_DIR)[1][f"{prompt}/input_features"].numpy()


def reference_hypotheses(trace):
    """Kept hypotheses of a trace, best first, as {tuple of ids: score}."""
    return {
        tuple(int(t) for t in row if t >= 0): float(score)
        for row, score in zip(trace["hypotheses"], trace["hypothesis_scores"], strict=True)
    }


def reference_margin(trace, config):
    """The reference's closest decision, in score: the smallest change to one continuation's or
    hypothesis's score that could change what the search keeps, ends, returns or when it stops.

    Per step: the num_beams-th live continuation against the next one; every ended one against
    the num_beams-th best of the others; the kept hypotheses against the best dropped one; the
    best live beam, scored as if it ended, against the worst kept hypothesis. At the end: the
    best hypothesis against the second.
    """
    n_beams, eos = config["num_beams"], config["eos_token_id"]
    gaps = []
    live_scores = np.zeros(1, dtype=np.float32)
    kept: list[float] = []
    for step in range(trace["beam_scores"].shape[0]):
        length = step + 1
        at_limit = length == config["max_new_tokens"]
        continuations = []  # (score, ends)
        for beam, score in enumerate(live_scores):
            ids, log_probs = (
                trace["candidate_ids"][step, beam],
                trace["candidate_log_probs"][step, beam],
            )
            for token, log_prob in zip(ids, log_probs, strict=True):
                if np.isfinite(log_prob):
                    continuations.append((score + log_prob, token == eos or at_limit))
        continuations.sort(key=lambda c: -c[0])
        live = [score for score, ends in continuations if not ends]
        if len(live) > n_beams:
            gaps.append(live[n_beams - 1] - live[n_beams])
        for i, (score, ends) in enumerate(continuations):
            others = [s for j, (s, _) in enumerate(continuations) if j != i]
            if ends and len(others) >= n_beams:
                gaps.append(abs(score - others[n_beams - 1]))
        divisor = np.float32(length ** config["length_penalty"])
        ended = trace["finished"][:, 0] == step if len(trace["finished"]) else []
        kept = sorted([*kept, *trace["finished_scores"][ended].tolist()], reverse=True)
        if len(kept) > n_beams:
            gaps.append(kept[n_beams - 1] - kept[n_beams])
        kept = kept[:n_beams]
        scores = trace["beam_scores"][step]
        live_scores = scores[np.isfinite(scores)]
        if len(live_scores) and len(kept) == n_beams:
            gaps.append(abs(live_scores[0] / divisor - kept[-1]))
    scores = trace["hypothesis_scores"]
    if len(scores) > 1:
        gaps.append(scores[0] - scores[1])
    return float(min(gaps))


def run_case(engine, name):
    _, config = traces()[name]
    result = engine.transcribe_beam(
        features(name.split("/")[0]),
        num_beams=config["num_beams"],
        length_penalty=config["length_penalty"],
        eos_token_id=config["eos_token_id"],
    )[0]
    times = result["step_s"]
    assert len(times) == result["steps"] and times == sorted(times)
    assert 0 < result["encode_s"] < times[0] <= times[-1] <= result["seconds"]
    return result


def assert_blocks_returned(engine):
    usage = engine.block_usage()
    assert usage["self_free"] == usage["self_blocks"] and usage["self_refs"] == 0, usage
    assert usage["cross_free"] == usage["cross_blocks"] and usage["cross_refs"] == 0, usage
    assert usage["rows_free"] == usage["rows"], usage


def test_reference_margins_are_the_known_ones():
    """The margin measure finds the closest decisions the reference's tests name."""
    names = ("p0/beam5", "p1/beam5", "p2/beam5")
    margins = {name: reference_margin(*traces()[name]) for name in names}
    assert margins == pytest.approx(
        {"p0/beam5": 1.2e-5, "p1/beam5": 7.9e-4, "p2/beam5": 1.5e-2}, rel=0.05
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"kv_dtype": "int8"},
        {"kv_block_size": 4},
        {"kv_block_size": 4, "kv_dtype": "int8"},
        # Blocks copied in every layout.
        {"kv_block_size": 4, "kv_layout": "token_major"},
        {"kv_block_size": 8, "kv_layout": "dim_major", "kv_dtype": "int8"},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "default",
)
def test_fixture_beam_equals_reference(kvb, options):
    """Every traced search, beam 5 on each golden prompt among them: the engine's kept
    hypotheses score close to the reference's, and where the reference's margin is wider than
    the engine can move it, the engine returns the same ids and keeps the same hypotheses."""
    kv = options.get("kv_dtype", "fp16")
    engine = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), max_seqs=5, **options)
    prompt = engine.rules()["prompt"]
    unchecked = {}
    for name, (trace, config) in traces().items():
        result = run_case(engine, name)
        expected = reference_hypotheses(trace)
        ours = {tuple(h["ids"]): h["score"] for h in result["hypotheses"]}
        assert (
            next(iter(ours)) == tuple(result["ids"])
            and ours[tuple(result["ids"])] == result["score"]
        )
        for ids in ours.keys() & expected.keys():
            assert abs(ours[ids] - expected[ids]) <= SCORE_TOLERANCE[kv], (name, ids)
        margin = reference_margin(trace, config)
        same = prompt + result["ids"] == trace["ids"].tolist()
        best = float(trace["hypothesis_scores"][0])
        assert abs(result["score"] - best) <= SCORE_TOLERANCE[kv] + (0 if same else margin), name
        if margin > DECISION_BOUND[kv]:
            assert same, name
            assert ours.keys() == expected.keys(), name
        else:
            unchecked[name] = (margin, same, result["score"] - best)
    print(f"\n{options}: ids not compared (margin, same ids, score - reference): {unchecked}")
    # Only beam 5 on p0, and with an INT8 cache on p1, decide closer than the engine's bound.
    allowed = {"p0/beam5"} if kv == "fp16" else {"p0/beam5", "p1/beam5"}
    assert unchecked.keys() <= allowed and "p0/beam5" in unchecked
    assert_blocks_returned(engine)


@pytest.mark.parametrize(
    "options",
    [{}, {"kv_block_size": 4}, {"kv_block_size": 4, "kv_dtype": "int8", "graph": False}],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "default",
)
def test_private_blocks_give_the_same_search(kvb, options):
    """Forks that share blocks and copy on write against forks that copy every block: the same
    ids, scores and hypotheses, bit for bit, from fewer copies."""
    shared = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), max_seqs=5, **options)
    private = kvb.AsrEngine(
        str(WHISPER_FIXTURE_DIR), max_seqs=5, beam_share_blocks=False, **options
    )
    assert shared.config()["beam_share_blocks"] and not private.config()["beam_share_blocks"]
    copies = {"shared": 0, "private": 0}
    for name in traces():
        a, b = run_case(shared, name), run_case(private, name)
        for key in ("ids", "score", "hypotheses", "steps"):
            assert a[key] == b[key], (name, key)
        assert a["block_copies"] <= b["block_copies"], name
        copies["shared"] += a["block_copies"]
        copies["private"] += b["block_copies"]
    assert 0 < copies["shared"] < copies["private"]
    graphed = options.get("graph", True)
    assert (shared.graph_stats()["replayed"] > 0) == graphed
    for engine in (shared, private):
        assert_blocks_returned(engine)


def test_one_beam_is_greedy(kvb):
    """num_beams 1 returns the greedy ids, and beam and greedy runs share an engine."""
    manifest, tensors = golden.load(WHISPER_FIXTURE_DIR)
    engine = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), max_seqs=3)
    prompt = engine.rules()["prompt"]
    clips = np.stack([tensors[f"{p['id']}/input_features"].numpy() for p in manifest["prompts"]])
    expected = [tensors[f"{p['id']}/greedy_ids"].tolist() for p in manifest["prompts"]]
    for _ in range(2):
        beam = engine.transcribe_beam(clips, num_beams=1)
        assert [prompt + r["ids"] for r in beam] == expected
        assert all(r["block_copies"] == 0 for r in beam)
        greedy = engine.transcribe(clips, policy="continuous")
        assert [prompt + r["ids"] for r in greedy] == expected
        assert_blocks_returned(engine)


def test_beam_search_rejects_what_it_cannot_run(kvb):
    clip = features("p0")
    engine = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), max_seqs=5)
    with pytest.raises(RuntimeError, match="num_beams must be between 1 and max_seqs"):
        engine.transcribe_beam(clip, num_beams=6)
    with pytest.raises(RuntimeError, match="each clip needs features"):
        engine.transcribe_beam(np.ascontiguousarray(clip[:, :-1]))
    # The encoder refuses the clip after the search has taken its rows and Cross block.
    bad = clip.copy()
    bad[0, 0] = np.inf
    with pytest.raises(RuntimeError, match="finite"):
        engine.transcribe_beam(bad)
    assert_blocks_returned(engine)
    # Five beams of the fixture's 32 positions need 10 blocks of 16.
    small = kvb.AsrEngine(str(WHISPER_FIXTURE_DIR), max_seqs=5, kv_blocks=9)
    with pytest.raises(RuntimeError, match="need 10 Self blocks"):
        small.transcribe_beam(clip, num_beams=5)
    assert small.transcribe_beam(clip, num_beams=4)[0]["ids"]
    # The engine still runs after each refusal.
    trace, _ = traces()["p0/beam5"]
    assert engine.rules()["prompt"] + run_case(engine, "p0/beam5")["ids"] == trace["ids"].tolist()


# --- Real checkpoints ----------------------------------------------------------------------------

# Hypotheses of Hugging Face FP16 and the engine that differ count as near-ties where the FP32
# reference scores them within this of each other.
RESCORE_TOLERANCE = 1e-2


def reference_score(model, window, rules, tokens, ended, length_penalty=1.0):
    """The FP32 reference's beam score of prompt + tokens (+ the end token when it ended): the
    sum of the generated tokens' log-probabilities / their count ** length_penalty."""
    ids = rules.prompt + tokens + ([rules.eos_token_id] if ended else [])
    with torch.no_grad():
        logits = model(window[None], torch.tensor([ids[:-1]]))[0].float()
    log_probs = torch.log_softmax(logits, -1)
    n_prompt = len(rules.prompt)
    chosen = log_probs[torch.arange(n_prompt - 1, len(ids) - 1), torch.tensor(ids[n_prompt:])]
    return float(chosen.sum()) / (len(ids) - n_prompt) ** length_penalty


@pytest.mark.slow
def test_real_model_beam_equals_hf_or_near_tie(kvb):
    """Tarteel base at beam 5 against Hugging Face `generate(num_beams=5)`, FP16 on the GPU, on
    the 20 local clips: the same ids, or hypotheses the FP32 reference scores within
    RESCORE_TOLERANCE of each other."""
    from kvbound import asr, audio
    from kvbound.baselines import hf_whisper
    from kvbound.evals import asr as evals_asr

    spec = load_spec("whisper-base-ar-quran")
    if not (spec.checkpoint_dir / "model.safetensors").exists():
        pytest.skip(f"run `make fetch MODEL={spec.name}`")
    try:
        clips = evals_asr.load_clips()
    except FileNotFoundError:
        pytest.skip("run `python -m kvbound.evals.asr fetch`")
    rules = WhisperRules.from_dict(spec.generation)
    windows = [audio.features(clip.audio(), spec.arch["n_mels"]) for clip in clips]

    hf = hf_whisper.load(spec, "cuda", "fp16", 5)
    theirs = [hf(clip.audio()) for clip in clips]
    del hf
    torch.cuda.empty_cache()
    recognizer = asr.Recognizer(spec.checkpoint_dir, max_seqs=5)
    ours = [recognizer.transcribe_beam(w.numpy(), 5)[0] for w in windows]
    assert_blocks_returned(recognizer.engine)

    model = load_whisper(spec.checkpoint_dir)
    exact, ties = 0, []
    for clip, window, hf_t, mine in zip(clips, windows, theirs, ours, strict=True):
        assert hf_t.prompt == mine.prompt == rules.prompt
        if hf_t.tokens == mine.tokens:
            exact += 1
            continue
        # Hugging Face's transcript drops the end token; a hypothesis ended unless it is at
        # max_length.
        hf_ended = len(rules.prompt) + len(hf_t.tokens) < rules.max_length
        a = reference_score(model, window, rules, mine.tokens, mine.ended)
        b = reference_score(model, window, rules, hf_t.tokens, hf_ended)
        assert abs(a - b) <= RESCORE_TOLERANCE, (clip.id, a, b)
        ties.append((clip.id, round(a - b, 5)))
    print(f"\nbeam 5: {exact} of {len(clips)} equal to HF; near-ties (ours - HF, FP32): {ties}")
