"""Perplexity arithmetic: windows, next-token targets and the summary, on toy logits."""

import math

import pytest
import torch

from kvbound.evals.perplexity import score, split_windows, summarize, window_nll


def test_split_windows_edge_cases():
    assert split_windows(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert split_windows(8, 4) == [(0, 4), (4, 8)]
    assert split_windows(9, 4) == [(0, 4), (4, 8)]  # a 1-token tail predicts nothing
    assert split_windows(3, 4) == [(0, 3)]
    assert split_windows(1, 4) == []
    assert split_windows(10, 4, max_windows=2) == [(0, 4), (4, 8)]
    assert split_windows(10, 4, max_windows=0) == split_windows(10, 4)
    assert split_windows(10, 4, max_windows=9) == split_windows(10, 4)
    with pytest.raises(ValueError):
        split_windows(10, 1)


def test_uniform_logits_give_the_vocabulary_size():
    vocab = 50
    ids = torch.randint(vocab, (100,), generator=torch.Generator().manual_seed(0)).tolist()
    spans = split_windows(len(ids), 16)
    sums, counts, _ = score(ids, spans, lambda window: torch.zeros(len(window), vocab))
    assert counts == [15] * 6 + [3]  # the last window holds tokens 96..99
    assert summarize(sums, counts)["perplexity"] == pytest.approx(vocab, rel=1e-6)


def test_known_distribution_gives_its_analytic_perplexity():
    # The same next-token distribution at every position, shifted by a constant that cancels.
    probs = torch.tensor([0.5, 0.25, 0.125, 0.125])
    logits_fn = lambda window: probs.log().expand(len(window), -1) + 3.0  # noqa: E731
    # One window: tokens 0, 0, 1, 2 are predicted, NLL = (1 + 1 + 2 + 3) ln 2.
    ids = [3, 0, 0, 1, 2]
    sums, counts, _ = score(ids, split_windows(len(ids), 5), logits_fn)
    assert summarize(sums, counts)["perplexity"] == pytest.approx(2 ** (7 / 4), rel=1e-6)
    # Windows of 3: token 1 opens the second window and is context only, so tokens 0, 0, 2, 3
    # are predicted, NLL = (1 + 1 + 3 + 3) ln 2.
    ids = [3, 0, 0, 1, 2, 3]
    sums, counts, _ = score(ids, split_windows(len(ids), 3), logits_fn)
    assert counts == [2, 2]
    assert summarize(sums, counts)["perplexity"] == pytest.approx(4.0, rel=1e-6)


def test_row_i_predicts_token_i_plus_1():
    # Row i puts probability 1/2 on ids[i + 1] and spreads the rest evenly over the other tokens.
    vocab, ids = 5, torch.tensor([0, 1, 2, 3, 4, 0])
    probs = torch.full((len(ids), vocab), 0.5 / (vocab - 1))
    probs[torch.arange(len(ids) - 1), ids[1:]] = 0.5
    # Scoring row i against ids[i] instead would give each token probability 1/8.
    assert window_nll(probs.log(), ids) == pytest.approx(5 * math.log(2), rel=1e-6)


def test_summary_weights_tokens_and_reports_the_standard_error():
    # Per-window mean NLL 1 and 3, over 10 and 30 tokens.
    summary = summarize([10.0, 90.0], [10, 30])
    assert summary["nll_mean"] == pytest.approx(100 / 40)
    assert summary["perplexity"] == pytest.approx(math.exp(2.5))
    assert summary["nll_stderr"] == pytest.approx(1.0)  # stdev(1, 3) / sqrt(2)
    assert summarize([4.0], [2])["nll_stderr"] == 0.0
