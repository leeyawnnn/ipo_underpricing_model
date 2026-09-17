"""Tests for Loughran-McDonald ratios, tokenisation and readability.

These functions decide every text-based number in the study, and they fail
silently: a tokeniser that keeps punctuation or a case-sensitivity slip in the
dictionary lookup produces plausible-looking ratios that are simply wrong. The
expected values below are computed by hand in the test, not copied from a run.
"""

from __future__ import annotations

import math

import pytest

from src.text_features import (
    _count_syllables,
    compute_lm_ratios,
    gunning_fog_index,
    tokenise,
)

TOY_DICT = {
    "lm_negative": {"ADVERSE", "LOSS", "DECLINE"},
    "lm_positive": {"PROFITABLE", "GAIN"},
    "lm_litigious": {"PLAINTIFF", "LAWSUIT"},
}


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenise_strips_punctuation_and_splits_contractions():
    # The apostrophe is neither \w nor \s, so it becomes a space and the
    # contraction splits. The old docstring claimed "firm's" survived intact.
    assert tokenise("The firm's revenue grew.") == ["the", "firm", "s", "revenue", "grew"]


def test_tokenise_drops_standalone_numbers_but_keeps_alphanumerics():
    assert tokenise("revenue of 15 million in Q3 2024") == [
        "revenue",
        "of",
        "million",
        "in",
        "q3",
    ]


def test_tokenise_lowercases():
    assert tokenise("ADVERSE Adverse adverse") == ["adverse", "adverse", "adverse"]


def test_tokenise_empty_and_whitespace():
    assert tokenise("") == []
    assert tokenise("   \n\t  ") == []


def test_tokenise_punctuation_only():
    assert tokenise("... --- ,,, !!!") == []


# ---------------------------------------------------------------------------
# LM ratios
# ---------------------------------------------------------------------------


def test_compute_lm_ratios_hand_computed():
    # 8 tokens: adverse loss the company reported a profitable quarter
    text = "Adverse loss: the company reported a profitable quarter."
    ratios = compute_lm_ratios(text, TOY_DICT)
    assert ratios["word_count"] == 8
    assert ratios["lm_negative_ratio"] == pytest.approx(2 / 8)
    assert ratios["lm_positive_ratio"] == pytest.approx(1 / 8)
    assert ratios["lm_litigious_ratio"] == 0.0


def test_compute_lm_ratios_is_case_insensitive():
    lower = compute_lm_ratios("adverse loss decline", TOY_DICT)
    upper = compute_lm_ratios("ADVERSE LOSS DECLINE", TOY_DICT)
    mixed = compute_lm_ratios("Adverse LoSs decline", TOY_DICT)
    assert lower["lm_negative_ratio"] == upper["lm_negative_ratio"] == mixed["lm_negative_ratio"]
    assert lower["lm_negative_ratio"] == pytest.approx(1.0)


def test_compute_lm_ratios_empty_document_returns_nan_not_zero():
    ratios = compute_lm_ratios("", TOY_DICT)
    assert set(ratios) == {f"{k}_ratio" for k in TOY_DICT}
    assert all(math.isnan(v) for v in ratios.values())


def test_compute_lm_ratios_all_numeric_document_returns_nan():
    # Every token is a bare number, so tokenise leaves nothing to score.
    ratios = compute_lm_ratios("1 2 3 456 7890", TOY_DICT)
    assert all(math.isnan(v) for v in ratios.values())


def test_compute_lm_ratios_punctuation_does_not_block_a_match():
    # "loss," and "(adverse)" must still match LOSS and ADVERSE.
    ratios = compute_lm_ratios("loss, (adverse) result", TOY_DICT)
    assert ratios["word_count"] == 3
    assert ratios["lm_negative_ratio"] == pytest.approx(2 / 3)


def test_compute_lm_ratios_sum_bounded_by_one_per_category():
    ratios = compute_lm_ratios("adverse loss decline gain profitable lawsuit", TOY_DICT)
    for key, value in ratios.items():
        if key == "word_count":
            continue
        assert 0.0 <= value <= 1.0


def test_real_dictionary_category_sizes(lm_dict):
    # Guards the loader against a silently empty category, which is what the
    # StrongModal/Strong_Modal column-name mismatch used to produce.
    assert set(lm_dict) == {
        "lm_negative",
        "lm_positive",
        "lm_uncertainty",
        "lm_litigious",
        "lm_constraining",
        "lm_modal_strong",
        "lm_modal_weak",
    }
    for name, words in lm_dict.items():
        assert len(words) > 0, f"{name} loaded empty"
        assert all(w == w.upper() for w in list(words)[:50])
    # LM's negative list is an order of magnitude larger than its positive
    # one; if that inverts, the wrong columns were read.
    assert len(lm_dict["lm_negative"]) > 5 * len(lm_dict["lm_positive"])


# ---------------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------------


def test_count_syllables_examples():
    assert _count_syllables("cat") == 1
    assert _count_syllables("water") == 2
    assert _count_syllables("beautiful") == 3
    assert _count_syllables("make") == 1  # silent e


def test_gunning_fog_worked_example():
    # Gunning's formula: 0.4 * (words/sentences + 100 * complex/words).
    # Two sentences, ten tokens, one word of three or more syllables
    # ("company" -> com-pa-ny).
    text = "The dog ran fast today. A company must pay tax."
    tokens = tokenise(text)
    assert len(tokens) == 10
    expected = 0.4 * (10 / 2 + 100 * (1 / 10))
    assert gunning_fog_index(text) == pytest.approx(expected, abs=0.3)


def test_gunning_fog_rises_with_complexity():
    simple = gunning_fog_index("The dog ran. The cat sat. The bird flew.")
    complex_text = gunning_fog_index(
        "Notwithstanding the aforementioned considerations, the organisation "
        "systematically implemented comprehensive administrative modifications "
        "affecting numerous operational responsibilities."
    )
    assert complex_text > simple


def test_gunning_fog_empty_is_nan():
    assert math.isnan(gunning_fog_index(""))
    assert math.isnan(gunning_fog_index("   "))
