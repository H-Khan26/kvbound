"""Reference beam search: the algorithm of Hugging Face `generate(num_beams=N, do_sample=False)`.

BeamSearch is a state machine that knows no model. Each step it is handed, for every live beam,
that beam's best next tokens with their log-probabilities, and it answers with the beams of the
next step as (parent beam, token). The engine's kvb::BeamSearch (engine/runtime/beam_search.cpp)
is the same machine; kvbound.fixture writes traces of this one for the C++ tests to replay.

The rules, from transformers generation/utils.py `_beam_search`:
  * A beam's score is the sum of its tokens' log-probabilities. They come from a log-softmax over
    the whole vocabulary; suppressed tokens are set to -inf after it, so they still count in the
    normalisation.
  * Every step ranks all continuations of all live beams by score and keeps the best
    2 * num_beams. Those that end (the end-of-sequence token, or the length limit) leave the
    search; the best num_beams of the others are the next live beams. No beam needs to offer
    more than n_candidates of its tokens for this.
  * A continuation that ends becomes a finished hypothesis only if it ranks among the best
    num_beams of the step. Its final score is score / length ** length_penalty, where length
    counts the generated tokens, end-of-sequence included and the prompt excluded. The best
    num_beams hypotheses are kept.
  * The search stops when no beam is live, or when num_beams hypotheses are kept and the best
    live beam, scored as if it ended now without another token, does not beat the worst of them
    (`early_stopping=False`).
  * The search starts from one live beam: Hugging Face starts the others at -1e9, which keeps
    them out of the first step.

All score arithmetic is FP32, as in Hugging Face. Ties, which torch.topk leaves unspecified, go
to the lower beam, then the larger log-probability, then the lower token id; among hypotheses to
the one that finished first.

With num_beams 1 the result is the greedy path: a step's best continuation either ends the
search or is the only beam kept.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from kvbound.ref.whisper import Whisper, WhisperRules

# (token id, log-probability)
Candidate = tuple[int, float]


@dataclass(frozen=True)
class BeamConfig:
    num_beams: int
    eos_token_id: int
    # Tokens after which every beam ends, whatever its last token.
    max_new_tokens: int
    length_penalty: float = 1.0


@dataclass(frozen=True)
class Beam:
    """A continuation: the tokens of beam `parent` of the step before, then `token`."""

    parent: int
    token: int
    # Sum of log-probabilities for a live beam; the final score for a finished one.
    score: float


@dataclass(frozen=True)
class Hypothesis:
    # Generated tokens, the end-of-sequence token included when it ended on one.
    tokens: list[int]
    score: float


@dataclass(frozen=True)
class BeamStep:
    # What the step consumed: one list per live beam.
    candidates: list[list[Candidate]]
    # Live beams of the next step, best first.
    beams: list[Beam]
    # Continuations that ended in this step, best first; not all need be kept as hypotheses.
    finished: list[Beam]
    done: bool


def n_candidates(config: BeamConfig) -> int:
    """Candidates per beam that give every step the decisions the whole vocabulary would.

    A step keeps the best num_beams continuations that go on and, of those that end, the ones
    among the best num_beams overall. Within one beam the ranking follows (log-probability
    descending, token ascending), since adding the beam's score in FP32 keeps that order or
    makes scores equal, and equal scores of one beam fall back to it. A candidate behind the
    beam's best num_beams + 1 thus has at least num_beams ahead of it in the step's ranking, so
    it does not end among the best num_beams; and at least num_beams that go on ahead of it, as
    only one of them can be the end token, so it is not kept either. Hugging Face ranks the best
    2 * num_beams over all beams, which hold every continuation these decisions use.
    """
    return config.num_beams + 1


class BeamSearch:
    def __init__(self, config: BeamConfig) -> None:
        if config.num_beams < 1 or config.max_new_tokens < 1:
            raise ValueError(f"beam search needs num_beams > 0 and max_new_tokens > 0: {config}")
        self.config = config
        self.done = False
        # Kept hypotheses, best first: at most num_beams.
        self.hypotheses: list[Hypothesis] = []
        self._tokens: list[list[int]] = [[]]
        self._scores = [np.float32(0.0)]
        self._n_generated = 0

    @property
    def n_live(self) -> int:
        """Beams the next step takes candidates for."""
        return len(self._tokens)

    @property
    def n_candidates(self) -> int:
        """Candidates per beam that are always enough (see n_candidates)."""
        return n_candidates(self.config)

    def tokens(self, beam: int) -> list[int]:
        """Generated tokens of a live beam."""
        return list(self._tokens[beam])

    def _length_divisor(self, length: int) -> np.float32:
        return np.float32(length**self.config.length_penalty)

    def step(self, candidates: list[list[Candidate]]) -> BeamStep:
        """Advances every live beam by one token.

        candidates[b] holds next tokens of live beam b. It must include the beam's n_candidates
        best by (log-probability descending, token id ascending), or every token with a finite
        log-probability when there are fewer; more are allowed and entries that are not finite
        are ignored.
        """
        cfg = self.config
        if self.done or len(candidates) != self.n_live:
            raise ValueError("step() needs one candidate list per live beam of a running search")

        ranked = []
        for beam, (score, beam_candidates) in enumerate(zip(self._scores, candidates, strict=True)):
            for token, value in beam_candidates:
                log_prob = np.float32(value)
                if math.isfinite(log_prob):
                    ranked.append((score + log_prob, beam, log_prob, token))
        ranked.sort(key=lambda c: (-c[0], c[1], -c[2], c[3]))
        del ranked[2 * cfg.num_beams :]

        length = self._n_generated + 1
        at_limit = length == cfg.max_new_tokens
        live: list[Beam] = []
        finished: list[Beam] = []
        for rank, (score, beam, _, token) in enumerate(ranked):
            if token == cfg.eos_token_id or at_limit:
                if rank < cfg.num_beams:
                    final = score / self._length_divisor(length)
                    finished.append(Beam(beam, token, float(final)))
            elif len(live) < cfg.num_beams:
                live.append(Beam(beam, token, float(score)))

        ended = [Hypothesis([*self._tokens[b.parent], b.token], b.score) for b in finished]
        # The sort is stable, so on a tie the hypothesis kept longest stays ahead.
        merged = sorted(self.hypotheses + ended, key=lambda h: -h.score)
        self.hypotheses = merged[: cfg.num_beams]

        self._tokens = [[*self._tokens[b.parent], b.token] for b in live]
        self._scores = [np.float32(b.score) for b in live]
        self._n_generated = length
        if live and len(self.hypotheses) == cfg.num_beams:
            attainable = self._scores[0] / self._length_divisor(length)
            self.done = not attainable > np.float32(self.hypotheses[-1].score)
        else:
            self.done = not live
        return BeamStep(candidates, live, finished, self.done)


@dataclass(frozen=True)
class BeamResult:
    config: BeamConfig
    # The best hypothesis as a whole sequence: the prompt, then its tokens.
    ids: list[int]
    score: float
    # Every kept hypothesis, best first.
    hypotheses: list[Hypothesis]
    steps: list[BeamStep]


def top_candidates(log_probs: Tensor, k: int) -> list[Candidate]:
    """The k largest finite entries of log_probs [vocab], largest first, lower id first on a tie."""
    values, ids = torch.sort(log_probs, descending=True, stable=True)
    keep = values[:k] > -math.inf
    return list(zip(ids[:k][keep].tolist(), values[:k][keep].tolist(), strict=True))


@torch.no_grad()
def beam_search(
    model: Whisper,
    features: Tensor,
    rules: WhisperRules,
    num_beams: int,
    length_penalty: float = 1.0,
    top_k: int | None = None,
) -> BeamResult:
    """Beam-search transcription of one window of features [n_mels, frames].

    The suppress and begin-suppress lists and the length limit apply as in ref.whisper.greedy.
    Each beam offers its top_k best tokens to the search: by default n_candidates(config),
    which gives the result the whole vocabulary would.
    """
    rules.check(model.cfg)
    device = model.decoder.embed.weight.device
    cross = model.encode(features[None])
    n_prompt = len(rules.prompt)
    search = BeamSearch(
        BeamConfig(num_beams, rules.eos_token_id, rules.max_length - n_prompt, length_penalty)
    )
    k = search.n_candidates if top_k is None else top_k
    steps = []
    while not search.done:
        candidates = []
        for beam in range(search.n_live):
            # The whole prefix runs again for each beam: no cache, so no cache to get wrong.
            ids = rules.prompt + search.tokens(beam)
            last = model.hidden(torch.tensor([ids], device=device), cross)[0, -1]
            log_probs = F.log_softmax(model.project(last).float(), dim=-1)
            log_probs[rules.suppress_tokens] = float("-inf")
            if len(ids) == n_prompt:
                log_probs[rules.begin_suppress_tokens] = float("-inf")
            candidates.append(top_candidates(log_probs.cpu(), k))
        steps.append(search.step(candidates))
    best = search.hypotheses[0]
    return BeamResult(
        search.config, rules.prompt + best.tokens, best.score, search.hypotheses, steps
    )
