"""The reference beam search against Hugging Face, against greedy decoding and against the traces
the C++ tests replay."""

import json
from dataclasses import replace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration

from kvbound import golden
from kvbound.fixture import (
    BEAM_CASES,
    BEAM_TRACE_FILE,
    WHISPER_FIXTURE_DIR,
    WHISPER_RULES,
    BeamCase,
    whisper_beam_traces,
)
from kvbound.ref import beam as ref_beam
from kvbound.ref import load_whisper
from kvbound.ref import whisper as ref_whisper


@pytest.fixture(scope="module")
def model():
    return load_whisper(WHISPER_FIXTURE_DIR, weights="native")


@pytest.fixture(scope="module")
def features():
    tensors = golden.load(WHISPER_FIXTURE_DIR)[1]
    return {case.prompt: tensors[f"{case.prompt}/input_features"] for case in BEAM_CASES}


@pytest.fixture(scope="module")
def results(model, features):
    return {case: search(model, features, case) for case in BEAM_CASES}


def search(model, features, case: BeamCase, top_k=None):
    return ref_beam.beam_search(
        model, features[case.prompt], case.rules(), case.num_beams, case.length_penalty, top_k
    )


def test_beam_matches_hf_on_fixture(features, results):
    # Between them the cases end on the length limit and on an end token, keep fewer hypotheses
    # than finish, and use length penalties 0, 0.6, 1 and 2. The ref's log-probabilities along
    # these paths equal Hugging Face's bit for bit, and its scores differ from generate's by at
    # most 6e-6 (cached decoding). The closest decision is p0/beam5's best hypothesis over the
    # second, 1.2e-5 apart; p1/beam5's is 7.9e-4 and p2/beam5's 1.5e-2.
    hf = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_FIXTURE_DIR, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cases = [case for case in BEAM_CASES if case.num_beams > 1]
    assert {case.prompt for case in cases if case.name == "beam5"} == {"p0", "p1", "p2"}
    for case in cases:
        with torch.no_grad():
            out = hf.generate(
                features[case.prompt][None],
                do_sample=False,
                num_beams=case.num_beams,
                eos_token_id=case.eos_token_id,
                length_penalty=case.length_penalty,
                return_dict_in_generate=True,
                output_scores=True,
            )
        result = results[case]
        assert result.ids == out.sequences[0].tolist(), case
        assert abs(result.score - float(out.sequences_scores[0])) < 1e-4, case


def test_one_beam_is_greedy(model, features):
    for prompt, tensor in features.items():
        for eos in (WHISPER_RULES.eos_token_id, 38, 72):
            rules = replace(WHISPER_RULES, eos_token_id=eos)
            result = ref_beam.beam_search(model, tensor, rules, 1)
            assert result.ids == ref_whisper.greedy(model, tensor, rules), (prompt, eos)
            assert len(result.hypotheses) == 1


def test_whole_vocabulary_changes_no_decision(model, features, results):
    vocab = model.cfg.vocab_size
    for case in BEAM_CASES:
        whole = search(model, features, case, top_k=vocab)
        top = results[case]
        assert [(s.beams, s.finished, s.done) for s in whole.steps] == [
            (s.beams, s.finished, s.done) for s in top.steps
        ], case
        assert all(len(c) == ref_beam.n_candidates(top.config) for c in top.steps[0].candidates)


def test_committed_traces_match_reference(model, features):
    path = WHISPER_FIXTURE_DIR / BEAM_TRACE_FILE
    committed = load_file(str(path))
    with safe_open(str(path), framework="pt") as f:
        committed_metadata = f.metadata()
    traces, metadata = whisper_beam_traces(model, features)
    assert committed_metadata == metadata
    assert committed.keys() == traces.keys()
    for name, tensor in traces.items():
        assert committed[name].dtype == tensor.dtype, name
        if tensor.is_floating_point():
            # Log-probabilities vary in their last bits with the platform's CPU math library;
            # the ids and beam decisions next to them are compared exactly.
            torch.testing.assert_close(
                committed[name], tensor, rtol=0, atol=1e-4, msg=lambda m, n=name: f"{n}: {m}"
            )
        else:
            assert torch.equal(committed[name], tensor), name
    configs = json.loads(metadata["configs"])
    assert configs == {
        f"{case.prompt}/{case.name}": {
            "num_beams": case.num_beams,
            "eos_token_id": case.eos_token_id,
            "max_new_tokens": WHISPER_RULES.max_length - len(WHISPER_RULES.prompt),
            "length_penalty": case.length_penalty,
        }
        for case in BEAM_CASES
    }


def test_search_rules():
    search = ref_beam.BeamSearch(ref_beam.BeamConfig(2, eos_token_id=0, max_new_tokens=3))
    assert search.n_candidates == 3
    # The end token ranks below the best num_beams: it is neither a hypothesis nor a beam.
    step = search.step([[(1, -0.5), (2, -0.75), (0, -1.0)]])
    assert [(b.parent, b.token) for b in step.beams] == [(0, 1), (0, 2)]
    assert step.finished == [] and not step.done
    with pytest.raises(ValueError):
        search.step([[(1, -0.5)]])
    with pytest.raises(ValueError):
        ref_beam.BeamSearch(ref_beam.BeamConfig(0, eos_token_id=0, max_new_tokens=3))
