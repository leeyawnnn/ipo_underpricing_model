"""Tests for deal-size extraction, uniqueness and the derived features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_derived_features,
    expanding_prospectus_uniqueness,
    filed_shares_offered,
)


# ---------------------------------------------------------------------------
# Deal size from the cover page
# ---------------------------------------------------------------------------

def test_filed_shares_reads_the_banner_count():
    cover = "Subject to completion, dated May 11, 2022 Prospectus 16,000,000 shares ProFrac Holding"
    assert filed_shares_offered(cover) == 16_000_000


def test_filed_shares_prefers_the_banner_over_the_over_allotment_line():
    cover = (
        "SUBJECT TO COMPLETION, DATED DECEMBER 10, 2020 PRELIMINARY PROSPECTUS "
        "23,000,000 Shares Common Shares. "
        "(1) Includes 3,450,000 shares that the underwriters have the option to purchase."
    )
    assert filed_shares_offered(cover) == 23_000_000


def test_filed_shares_handles_adss_and_units():
    assert filed_shares_offered("PROSPECTUS 30,400,000 American Depositary Shares") == 30_400_000
    assert filed_shares_offered("Prospectus 2,000,000 Units consisting of") == 2_000_000


def test_filed_shares_is_nan_when_the_cover_states_none():
    assert math.isnan(filed_shares_offered("PROSPECTUS Common Stock. This is an offering."))
    assert math.isnan(filed_shares_offered(""))


def test_filed_shares_ignores_a_count_far_from_the_banner():
    cover = "PROSPECTUS Common Stock. " + "filler " * 80 + "101,611,736 shares of outstanding stock"
    assert math.isnan(filed_shares_offered(cover))


# ---------------------------------------------------------------------------
# Expanding-window prospectus uniqueness
# ---------------------------------------------------------------------------

def _corpus(n: int) -> tuple[list[str], list[str], list[pd.Timestamp]]:
    boilerplate = "risk factors the company may not achieve profitability market conditions "
    texts = [boilerplate + f"unique token{i} " * 3 for i in range(n)]
    sectors = ["Healthcare"] * n
    dates = list(pd.date_range("2020-01-01", periods=n, freq="7D"))
    return texts, sectors, dates


def test_uniqueness_is_nan_until_enough_prior_filings_exist():
    texts, sectors, dates = _corpus(20)
    scores = expanding_prospectus_uniqueness(texts, sectors, dates, min_prior=5)
    # The first five filings in the sector have no historical comparison set.
    assert np.isnan(scores[:5]).all()
    assert np.isfinite(scores[5:]).all()


def test_uniqueness_never_uses_a_later_filing():
    """Reversing the tail of the corpus must not change earlier scores."""
    texts, sectors, dates = _corpus(20)
    baseline = expanding_prospectus_uniqueness(texts, sectors, dates, min_prior=5)

    shuffled_texts = texts[:12] + texts[12:][::-1]
    shuffled = expanding_prospectus_uniqueness(shuffled_texts, sectors, dates, min_prior=5)
    # Scores for filings 5..11 depend only on filings 0..10, which are untouched.
    np.testing.assert_allclose(baseline[5:11], shuffled[5:11], rtol=1e-9)


def test_uniqueness_is_bounded_and_low_for_boilerplate():
    identical = ["the same boilerplate risk language repeated exactly here"] * 15
    sectors = ["Technology"] * 15
    dates = list(pd.date_range("2021-01-01", periods=15, freq="5D"))
    scores = expanding_prospectus_uniqueness(identical, sectors, dates, min_prior=5)
    finite = scores[np.isfinite(scores)]
    assert len(finite) > 0
    assert (finite < 0.05).all()


def test_uniqueness_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="same length"):
        expanding_prospectus_uniqueness(["a", "b"], ["X"], [pd.Timestamp("2020-01-01")])


def test_uniqueness_returns_all_nan_on_a_tiny_corpus():
    scores = expanding_prospectus_uniqueness(
        ["a", "b"], ["X", "X"], list(pd.date_range("2020-01-01", periods=2))
    )
    assert np.isnan(scores).all()


def test_uniqueness_is_computed_within_sector():
    texts, _, dates = _corpus(24)
    sectors = ["Healthcare"] * 12 + ["Energy"] * 12
    scores = expanding_prospectus_uniqueness(texts, sectors, dates, min_prior=5)
    # Each sector needs its own five-filing warm-up.
    assert np.isnan(scores[:5]).all()
    assert np.isnan(scores[12:17]).all()
    assert np.isfinite(scores[17:]).all()


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "offer_price": [10.0, 20.0, 0.0],
            "filed_shares_offered": [1_000_000.0, 5_000_000.0, np.nan],
            "word_count": [50_000.0, 120_000.0, np.nan],
            "lm_negative_ratio": [0.02, 0.0, 0.03],
            "rf_lm_negative_ratio": [0.03, 0.01, np.nan],
            "max_underwriter_rank": [9.0, 5.0, np.nan],
        }
    )


def test_log_transforms(raw):
    out = add_derived_features(raw)
    assert out["log_offer_price"].iloc[0] == pytest.approx(np.log(10.0))
    assert out["log_offer_size"].iloc[0] == pytest.approx(np.log1p(10_000_000.0))
    assert out["log_prospectus_words"].iloc[1] == pytest.approx(np.log1p(120_000.0))


def test_zero_offer_price_is_clipped_not_infinite(raw):
    out = add_derived_features(raw)
    assert np.isfinite(out["log_offer_price"].iloc[2])


def test_risk_concentration_ratio_guards_division_by_zero(raw):
    out = add_derived_features(raw)
    assert out["risk_concentration_ratio"].iloc[0] == pytest.approx(0.03 / 0.02)
    assert np.isnan(out["risk_concentration_ratio"].iloc[1])  # denominator is zero
    assert np.isnan(out["risk_concentration_ratio"].iloc[2])  # numerator missing


def test_top_tier_uses_the_carter_manaster_cut_off(raw):
    out = add_derived_features(raw)
    assert out["top_tier_underwriter"].iloc[0] == 1.0   # rank 9
    assert out["top_tier_underwriter"].iloc[1] == 0.0   # rank 5
    assert np.isnan(out["top_tier_underwriter"].iloc[2])  # unranked stays missing


def test_add_derived_features_does_not_mutate_its_input(raw):
    before = raw.copy()
    add_derived_features(raw)
    pd.testing.assert_frame_equal(raw, before)
