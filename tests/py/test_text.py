"""Arabic normalization and error rates, on cases worked by hand."""

import random

import pytest

from kvbound.evals.text import (
    Edits,
    align,
    cer,
    char_edits,
    normalize_arabic,
    total,
    wer,
    word_edits,
)


@pytest.mark.parametrize(
    ("written", "normalized"),
    [
        # Harakat, shadda, tanwin and sukun go; the superscript alef (U+0670) goes with them.
        ("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ", "بسم الله الرحمن الرحيم"),
        ("كِتَابًا عَلِيمٌ بِسَبْعٍ", "كتابا عليم بسبع"),
        # Alef with hamza above or below, with madda, and alef wasla are a bare alef.
        ("أَحْمَدُ إِبْرَاهِيمُ آمَنَ ٱلَّذِينَ", "احمد ابراهيم امن الذين"),
        # Alef maqsura is ya and ta marbuta is ha.
        ("إِلَى هُدًى رَحْمَةً", "الي هدي رحمه"),
        # Hamza on waw or ya leaves its seat; a standalone hamza is dropped.
        ("مُؤْمِنُونَ أُولَئِكَ شَيْءٍ مَاءً سَأَلَ", "مومنون اوليك شي ما سال"),
        # A combining hamza or madda (U+0654, U+0653) leaves its carrier.
        ("أمر آمن", "امر امن"),
        # Tatweel only stretches a word.
        ("اللـــه رحـيـم", "الله رحيم"),
        # Punctuation, Quranic marks, digits and other scripts separate words and vanish.
        ("قَالَ: نَعَمْ، ۚ (123) abc ۝ ٣", "قال نعم"),
        ("عبد،الله", "عبد الله"),
        ("  الحمد \n\t لله  ", "الحمد لله"),
        ("", ""),
        ("... 42 !", ""),
    ],
)
def test_normalize_arabic(written, normalized):
    assert normalize_arabic(written) == normalized
    assert normalize_arabic(normalized) == normalized


def test_normalization_makes_spelling_variants_equal():
    assert normalize_arabic("الرَّحْمَٰنِ") == normalize_arabic("الرحمن")
    assert normalize_arabic("ٱلْهُدَىٰ") == normalize_arabic("الهدي")
    assert normalize_arabic("شَيْئًا") == normalize_arabic("شييا")


@pytest.mark.parametrize(
    ("reference", "hypothesis", "edits"),
    [
        ("a b c", "a b c", Edits(0, 0, 0, 3)),
        ("a b c", "a x c", Edits(1, 0, 0, 3)),
        ("a b c", "a c", Edits(0, 0, 1, 3)),
        ("a b c", "a b x c", Edits(0, 1, 0, 3)),
        ("a b c d", "a x c d e", Edits(1, 1, 0, 4)),
        ("a b c d", "b c", Edits(0, 0, 2, 4)),
        # Swapped words are two substitutions, not an insertion and a deletion.
        ("a b", "b a", Edits(2, 0, 0, 2)),
        ("a", "x y z", Edits(1, 2, 0, 1)),
        ("a b c", "", Edits(0, 0, 3, 3)),
        ("", "a b", Edits(0, 2, 0, 0)),
        ("", "", Edits(0, 0, 0, 0)),
        ("  a   b ", "a b", Edits(0, 0, 0, 2)),
    ],
)
def test_word_edits(reference, hypothesis, edits):
    assert word_edits(reference, hypothesis) == edits


def test_rates():
    assert Edits(1, 1, 0, 4).rate == 0.5
    assert word_edits("a b c", "").rate == 1.0
    assert word_edits("a", "x y z").rate == 3.0
    # An empty reference: nothing to get wrong, or errors without a denominator.
    assert word_edits("", "").rate == 0.0
    assert word_edits("", "a b").rate == float("inf")


def test_char_edits_count_one_space_between_words():
    assert char_edits("abc def", "abd  def") == Edits(1, 0, 0, 7)
    assert char_edits("ab cd", "abcd") == Edits(0, 0, 1, 5)
    assert char_edits("abc", "") == Edits(0, 0, 3, 3)
    assert char_edits("", " ") == Edits(0, 0, 0, 0)
    # Diacritics are characters until the text is normalized.
    assert char_edits("بِسْمِ", "بسم") == Edits(0, 0, 3, 6)
    assert char_edits(normalize_arabic("بِسْمِ"), "بسم").errors == 0


def test_corpus_rate_is_total_edits_over_total_reference():
    references, hypotheses = ["a b c d", "e"], ["a b c d", "x"]
    assert wer(references, hypotheses) == Edits(1, 0, 0, 5)
    assert wer(references, hypotheses).rate == 0.2  # the mean of the two clips' rates is 0.5
    assert cer(references, hypotheses) == Edits(1, 0, 0, 8)
    assert total([]) == Edits()
    with pytest.raises(ValueError):
        wer(["a"], [])


def levenshtein(a, b):
    """Textbook dynamic program, for the distance only."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def test_align_matches_the_textbook_distance_on_random_strings():
    rng = random.Random(0)
    for _ in range(300):
        a = "".join(rng.choices("abc", k=rng.randrange(0, 12)))
        b = "".join(rng.choices("abc", k=rng.randrange(0, 12)))
        edits = align(a, b)
        assert edits.errors == levenshtein(a, b)
        assert edits.reference == len(a)
        # Insertions and deletions account for the difference in length.
        assert len(a) + edits.insertions - edits.deletions == len(b)
