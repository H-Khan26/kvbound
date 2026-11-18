"""Arabic text normalization and word and character error rates for speech evaluation.

normalize_arabic gives the words of the Sawt Sakinah app's arabic_words (detection/_arabic.py),
joined by spaces, so a score here is over the words the app matches against the Quran text: bare
letters, with no diacritics and one spelling for each letter the script writes several ways.

Error rates are edit distances over a reference: (substitutions + insertions + deletions) /
reference length. WER counts whitespace-separated words. CER counts characters, with every run of
whitespace collapsed to one space that counts as a character. Over several clips the rate is the
total edits over the total reference length, not the mean of the clips' rates.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

ALEF, WAW, HA, YA = 0x0627, 0x0648, 0x0647, 0x064A
# Code point to its replacement; None removes it.
_FOLD: dict[int, int | None] = {
    # Harakat, tanwin, sukun and shadda (U+064B..U+0652), the combining madda, hamza and other
    # marks (U+0653..U+065F), the superscript alef (U+0670) and the tatweel (U+0640) are removed.
    **dict.fromkeys([*range(0x064B, 0x0660), 0x0670, 0x0640]),
    # One form per letter.
    0x0623: ALEF,  # alef with hamza above
    0x0625: ALEF,  # alef with hamza below
    0x0622: ALEF,  # alef with madda
    0x0671: ALEF,  # alef wasla
    0x0649: YA,  # alef maqsura
    0x0629: HA,  # ta marbuta
    0x0624: WAW,  # waw with hamza
    0x0626: YA,  # ya with hamza
    0x0621: None,  # hamza on its own
}
# Anything but a base Arabic letter (hamza to ghain, fa to ya): punctuation, Quranic annotation
# marks, digits, other scripts.
_NON_LETTERS = re.compile("[^ء-غف-ي]+")


def normalize_arabic(text: str) -> str:
    """The words of `text` as bare Arabic letters, joined by single spaces.

    Marks vanish and every other character that is not a letter separates words, so a diacritic
    never splits a word and punctuation always does.
    """
    return " ".join(_NON_LETTERS.sub(" ", text.translate(_FOLD)).split())


@dataclass(frozen=True)
class Edits:
    """Edit operations that turn a reference into a hypothesis, and the reference length."""

    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    reference: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def rate(self) -> float:
        """errors / reference; with an empty reference, 0 for no errors and infinity otherwise."""
        if self.reference == 0:
            return float("inf") if self.errors else 0.0
        return self.errors / self.reference

    def __add__(self, other: "Edits") -> "Edits":
        return Edits(
            self.substitutions + other.substitutions,
            self.insertions + other.insertions,
            self.deletions + other.deletions,
            self.reference + other.reference,
        )


def total(edits: Iterable[Edits]) -> Edits:
    """The edits of a corpus: its rate is total errors over total reference length."""
    return sum(edits, Edits())


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Edits:
    """Levenshtein distance between two token sequences, split into the three operations.

    Several alignments can reach the minimum; the one counted prefers, from the end of both
    sequences backwards, a match, then a substitution, then a deletion, then an insertion.
    """
    n, m = len(reference), len(hypothesis)
    ids: dict[str, int] = {}
    ref = np.array([ids.setdefault(token, len(ids)) for token in reference], dtype=np.int64)
    hyp = np.array([ids.setdefault(token, len(ids)) for token in hypothesis], dtype=np.int64)
    # dist[i, j]: edits between the first i reference tokens and the first j hypothesis tokens.
    columns = np.arange(m + 1)
    dist = np.empty((n + 1, m + 1), dtype=np.int64)
    dist[0] = columns
    for i in range(1, n + 1):
        # Best of a deletion and a match or substitution; insertions then extend along the row:
        # dist[i, j] = min over k <= j of (step[k] + j - k).
        step = np.empty(m + 1, dtype=np.int64)
        step[0] = i
        step[1:] = np.minimum(dist[i - 1, 1:] + 1, dist[i - 1, :-1] + (hyp != ref[i - 1]))
        dist[i] = np.minimum.accumulate(step - columns) + columns

    substitutions = insertions = deletions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i, j] == dist[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            substitutions += int(ref[i - 1] != hyp[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and dist[i, j] == dist[i - 1, j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return Edits(substitutions, insertions, deletions, n)


def word_edits(reference: str, hypothesis: str) -> Edits:
    """Edits between the whitespace-separated words of two texts."""
    return align(reference.split(), hypothesis.split())


def char_edits(reference: str, hypothesis: str) -> Edits:
    """Edits between the characters of two texts, each run of whitespace being one space."""
    return align(" ".join(reference.split()), " ".join(hypothesis.split()))


def wer(references: Sequence[str], hypotheses: Sequence[str]) -> Edits:
    """Word edits of a corpus; .rate is its word error rate."""
    return total(word_edits(r, h) for r, h in zip(references, hypotheses, strict=True))


def cer(references: Sequence[str], hypotheses: Sequence[str]) -> Edits:
    """Character edits of a corpus; .rate is its character error rate."""
    return total(char_edits(r, h) for r, h in zip(references, hypotheses, strict=True))
