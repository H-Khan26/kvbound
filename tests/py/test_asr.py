"""Speech evaluation: drawing the subset, its committed manifest, and the scoring driver."""

import copy
import json
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

from kvbound import record
from kvbound.evals import asr
from kvbound.evals.asr import Row, Transcript
from kvbound.models import load_spec, model_entry


def metadata(seed=0):
    """A synthetic test split: three shards of two pages each, reciters in runs as in the real
    one, some clips over 30 s, and one reciter that sits deep inside a single page."""
    rng = random.Random(seed)
    runs = [
        [("a", 300), ("b", 150)],  # shard 0, page 0
        [("b", 250)],
        [("c", 400), ("deep", 60)],  # "deep" starts hours into this page
        [("c", 100), ("d", 180)],
        [("d", 200), ("few", 4)],
        [("e", 350)],
    ]
    rows = []
    for page, run in enumerate(runs):
        shard, first = f"data/test-{page // 2:05d}-of-00003-abc.parquet", (page % 2) * 1000
        seconds = 0.0
        for reciter, n in run:
            for _ in range(n):
                duration = round(rng.uniform(2.0, 36.0), 3)
                rows.append(Row(shard, first, reciter, duration, f"text {first}", seconds))
                seconds += duration
                first += 1
    return rows


def test_apportion_by_largest_remainder():
    assert asr.apportion({"a": 50, "b": 30, "c": 20}, 10) == {"a": 5, "b": 3, "c": 2}
    # 3.5, 3.5 and 3: the two halves compete for one clip and the first name wins.
    assert asr.apportion({"b": 7, "a": 7, "c": 6}, 10) == {"a": 4, "b": 3, "c": 3}
    assert asr.apportion({"a": 1, "b": 1, "c": 98}, 10) == {"a": 0, "b": 0, "c": 10}
    assert sum(asr.apportion({"a": 13, "b": 29, "c": 31}, 1000).values()) == 1000


def test_selection_is_deterministic_and_ignores_the_order_of_the_rows():
    rows = metadata()
    chosen = asr.select_subset(rows, n_clips=200)
    assert chosen == asr.select_subset(rows, n_clips=200)
    shuffled = random.Random(1).sample(rows, len(rows))
    assert chosen == asr.select_subset(shuffled, n_clips=200)
    assert chosen != asr.select_subset(rows, n_clips=200, seed=1)
    # Pinned: a change to the draw changes the committed subset.
    assert [(row.shard[10:15], row.row) for row in chosen[::40]] == PINNED_ROWS


def test_selection_is_stratified_by_reciter_and_keeps_short_clips():
    rows = metadata()
    chosen = asr.select_subset(rows, n_clips=200)
    assert len(chosen) == len(set(chosen)) == 200
    assert chosen == sorted(chosen, key=lambda r: (r.shard, r.row))
    assert all(row.duration <= asr.SUBSET_MAX_SECONDS for row in chosen)
    eligible = Counter(row.reciter for row in rows if row.duration <= asr.SUBSET_MAX_SECONDS)
    assert Counter(row.reciter for row in chosen) == Counter(
        {reciter: n for reciter, n in asr.apportion(eligible, 200).items() if n}
    )
    short = asr.select_subset(rows, n_clips=200, max_seconds=10.0)
    assert len(short) == 200 and all(row.duration <= 10.0 for row in short)


def test_candidates_are_the_start_of_each_page_and_no_more_than_needed_beyond():
    rows = metadata()
    chosen = asr.select_subset(rows, n_clips=60, pool_seconds=900.0, pool_factor=2)
    shares = Counter(row.reciter for row in chosen)
    beyond = Counter(row.reciter for row in chosen if row.page_seconds > 900.0)
    # Only the reciter no page starts with is read past the first 900 s of a page.
    assert set(beyond) == {"deep"}
    own = [row for row in rows if row.reciter == "deep" and row.duration <= 30.0]
    pool = asr.candidate_pool(own, shares["deep"], 900.0, 2)
    assert len(pool) == 2 * shares["deep"]
    assert max(row.page_seconds for row in pool) < sorted(r.page_seconds for r in own)[len(pool)]
    # A reciter with fewer rows than its pool asks for offers them all.
    few = [row for row in rows if row.reciter == "few"]
    assert sorted(asr.candidate_pool(few, 3, 900.0, 2), key=lambda r: r.row) == few


def test_spread_takes_evenly_spaced_clips():
    assert asr.spread(list(range(10)), 3) == [0, 3, 6]
    assert asr.spread(list(range(10)), 0) == list(range(10))
    assert asr.spread(list(range(4)), 9) == list(range(4))


@pytest.fixture(scope="module")
def manifest():
    return json.loads(asr.SUBSET_MANIFEST.read_text(encoding="utf-8"))


def test_committed_manifest_is_well_formed(manifest):
    asr.check_manifest(manifest)
    assert manifest["name"] == asr.SUBSET
    assert (manifest["dataset"], manifest["revision"]) == (asr.DATASET_REPO, asr.DATASET_REVISION)
    assert manifest["selection"] == {
        "n_clips": asr.SUBSET_CLIPS,
        "seed": asr.SUBSET_SEED,
        "max_seconds": asr.SUBSET_MAX_SECONDS,
        "pool_seconds": asr.POOL_SECONDS,
        "pool_factor": asr.POOL_FACTOR,
    }
    population = manifest["population"]
    assert population["rows"] == population["eligible"] + population["over_max_seconds"]
    assert len({clip["reciter"] for clip in manifest["clips"]}) == population["reciters"]
    # The file is exactly what manifest_json writes.
    assert asr.SUBSET_MANIFEST.read_text(encoding="utf-8") == asr.manifest_json(manifest)


def edit(manifest, change):
    broken = copy.deepcopy(manifest)
    change(broken)
    return broken


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.pop("shards"),
        lambda m: m["selection"].update(seed="0"),
        lambda m: m["clips"].pop(),
        lambda m: m["clips"][0].update(audio_sha256="abc"),
        lambda m: m["clips"][0].pop("text_sha256"),
        lambda m: m["clips"][0].update(duration=30.5),
        lambda m: m["clips"][0].update(row=m["clips"][0]["row"] + 1),
        lambda m: m["clips"].__setitem__(1, m["clips"][0]),
        lambda m: m["clips"].reverse(),
        lambda m: m["shards"].pop(m["clips"][0]["shard"]),
        lambda m: m.update(seconds=m["seconds"] + 1.0),
    ],
)
def test_check_manifest_rejects_a_malformed_one(manifest, change):
    with pytest.raises(ValueError):
        asr.check_manifest(edit(manifest, change))


@dataclass(frozen=True)
class FakeClip(asr.Clip):
    def audio(self):
        return np.zeros(int(self.duration * 16000), dtype=np.float32)


class FakeStack:
    """Returns scripted texts with fixed timings."""

    engine = {"name": "fake"}
    params = {
        "backend": "fake",
        "device": "cpu",
        "compute_type": "fp32",
        "weights": "fp32",
        "kv": "fp32",
        "beams": 1,
    }

    def __init__(self, texts, first_token_s=0.03, prompts=None):
        self.texts, self.first_token_s = iter(texts), first_token_s
        self.prompts = iter(prompts or [[1]] * len(texts))
        self.calls = 0

    def __call__(self, samples):
        self.calls += 1
        return Transcript(
            text=next(self.texts),
            prompt=next(self.prompts),
            tokens=[7, 8, 9],
            e2e_s=0.5,
            model_s=0.4,
            encoder_s=0.1,
            first_token_s=self.first_token_s,
        )


CLIPS = [
    FakeClip("c0", "s", 0, "r", 10.0, "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"),
    FakeClip("c1", "s", 1, "r", 30.0, "قُلْ هُوَ اللَّهُ أَحَدٌ"),
]


def test_evaluate_scores_raw_and_normalized_text_and_real_time_factors():
    # The warm-up transcript is discarded. Then: one clip right but without diacritics, and one
    # with a wrong word and a missing one.
    stack = FakeStack(["warm-up", "بسم الله الرحمن الرحيم", "قل هو الاحد"])
    metrics, samples, transcripts = asr.evaluate(stack, CLIPS)
    assert stack.calls == 3 and len(transcripts) == 2
    assert metrics["wer"] == pytest.approx(2 / 8)
    assert metrics["edits"]["wer"] == {
        "substitutions": 1,
        "insertions": 0,
        "deletions": 1,
        "reference": 8,
    }
    assert metrics["wer_raw"] == 1.0  # every written word differs from its diacritized reference
    assert 0 < metrics["cer"] < metrics["cer_raw"] < 1
    assert metrics["rtf_model"] == pytest.approx(0.8 / 40)
    assert metrics["rtf_e2e"] == pytest.approx(1.0 / 40)
    assert metrics["encoder_ms"]["p50"] == pytest.approx(100.0)
    assert metrics["first_token_ms"]["mean"] == pytest.approx(30.0)
    # 0.37 s from the first token to the last, the end token, which is 3 tokens later.
    assert metrics["decode_ms_per_token"]["mean"] == pytest.approx(370.0 / 3)
    # 0.3 s after the encoder for 3 tokens and the end token.
    assert metrics["generate_ms_per_token"]["mean"] == pytest.approx(75.0)
    assert metrics["audio_seconds"] == 40.0
    assert samples["clip"] == ["c0", "c1"]
    assert (samples["word_errors"], samples["words"]) == ([0, 2], [4, 4])
    json.dumps(metrics), json.dumps(samples)


def test_evaluate_leaves_out_the_first_token_time_a_stack_does_not_report():
    stack = FakeStack(["x", "y"], first_token_s=None)
    metrics, samples, _ = asr.evaluate(stack, CLIPS, warmup=0)
    assert "first_token_ms" not in metrics and "first_token_s" not in samples
    assert "decode_ms_per_token" not in metrics
    assert metrics["generate_ms_per_token"]["n"] == 2
    assert metrics["wer"] == 1.0


def test_evaluate_rejects_a_prompt_that_changes_between_clips():
    with pytest.raises(ValueError, match="prompts"):
        asr.evaluate(FakeStack(["x", "y"], prompts=[[1], [1, 2]]), CLIPS, warmup=0)
    with pytest.raises(ValueError, match="no clips"):
        asr.evaluate(FakeStack([]), [])


def test_model_entry_adds_the_family_to_the_three_older_keys():
    spec = load_spec("tinyllama-1.1b")
    assert list(model_entry(spec).items()) == [
        ("name", spec.name),
        ("hf_repo", spec.hf_repo),
        ("revision", spec.revision),
        ("family", "llama"),
    ]
    assert model_entry(load_spec("whisper-base-ar-quran"))["family"] == "whisper"


def test_a_run_writes_an_asr_record_of_a_whisper_model(monkeypatch, tmp_path):
    stack = FakeStack(["warm-up", "بسم الله الرحمن الرحيم", "قل هو الله احد"])
    monkeypatch.setattr(asr, "load_clips", lambda: CLIPS)
    monkeypatch.setattr(asr, "load_transcriber", lambda *args: stack)
    result, clips, transcripts = asr.run(
        "whisper-base-ar-quran", "fake", device="cpu", subset=asr.CLIPS
    )
    assert (len(clips), len(transcripts)) == (2, 2)

    # A machine without a GPU has no fingerprint; the record needs these fields of one.
    fingerprint = {"device_id": "test", "git": {"dirty": False}, "software": {"os_class": "linux"}}
    monkeypatch.setattr(record.fp, "fingerprint", lambda gpu_index: fingerprint)
    monkeypatch.setattr(record, "RESULTS_RAW", tmp_path)
    written = json.loads(record.write_record(**result).read_text(encoding="utf-8"))
    assert written["kind"] == "asr" and record.model_family(written) == "whisper"
    assert written["model"] == model_entry(load_spec("whisper-base-ar-quran"))
    assert written["metrics"]["wer"] == 0.0 and written["metrics"]["wer_raw"] == 1.0
    assert written["nvml"] is None
    assert {
        "prompt",
        "beams",
        "weights",
        "kv",
        "subset",
        "n_clips",
        "backend",
        "compute_type",
        "normalization",
    } <= set(written["params"])
    assert {
        "wer",
        "cer",
        "wer_raw",
        "cer_raw",
        "rtf_model",
        "rtf_e2e",
        "encoder_ms",
        "first_token_ms",
        "decode_ms_per_token",
    } <= set(written["metrics"])
    with pytest.raises(ValueError, match="whisper"):
        asr.run("tinyllama-1.1b", "fake", device="cpu", subset=asr.CLIPS)


# Every 40th row of select_subset(metadata(), n_clips=200), as (shard number, row).
PINNED_ROWS = [("00000", 1), ("00000", 1042), ("00001", 44), ("00001", 1071), ("00002", 95)]
