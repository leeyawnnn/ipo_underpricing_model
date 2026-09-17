"""Tests for first-day price reconstruction.

The target of the whole study is one number per IPO, and getting it wrong is
invisible downstream: a plausible-looking return series produced every result
in the previous version of this project. The ground-truth cases below are
first-day returns that are a matter of public record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scraper_prices import cumulative_split_factor, split_is_applied

# Offer price and first-day close for IPOs whose first day is public record.
# Sources: the pricing press release and the first close reported on the day.
KNOWN_FIRST_DAYS = [
    ("ABNB", 68.00, 144.71, 1.1281),
    ("DASH", 102.00, 189.51, 0.8579),
    ("CRWD", 34.00, 58.00, 0.7059),
    ("RBLX", 45.00, 69.50, 0.5444),
    ("PLTR", 7.25, 9.50, 0.3103),
    ("LYFT", 72.00, 78.29, 0.0874),
    ("UBER", 45.00, 41.57, -0.0762),
    ("SNOW", 120.00, 253.93, 1.1161),
]


@pytest.mark.parametrize(("ticker", "offer", "close", "expected"), KNOWN_FIRST_DAYS)
def test_underpricing_formula_matches_public_record(ticker, offer, close, expected):
    assert (close - offer) / offer == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# Split correction
# ---------------------------------------------------------------------------


def _series(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=pd.DatetimeIndex(list(values)))


def test_no_splits_leaves_the_price_alone():
    assert cumulative_split_factor(pd.Series(dtype=float), pd.Timestamp("2020-01-01")) == 1.0
    assert cumulative_split_factor(None, pd.Timestamp("2020-01-01")) == 1.0


def test_forward_split_after_listing_scales_the_price_up():
    # Yahoo divides the pre-split history by 4 for a 4-for-1 split, so undoing
    # it multiplies by 4. CrowdStrike: reported $14.50, actually $58.00.
    splits = _series({"2026-07-02": 4.0})
    factor = cumulative_split_factor(splits, pd.Timestamp("2019-06-12"))
    assert factor == pytest.approx(4.0)
    assert 14.50 * factor == pytest.approx(58.00)


def test_reverse_split_after_listing_scales_the_price_down():
    splits = _series({"2024-01-15": 0.05})  # 1-for-20
    assert cumulative_split_factor(splits, pd.Timestamp("2021-01-01")) == pytest.approx(0.05)


def test_multiple_splits_compound():
    splits = _series({"2022-01-01": 2.0, "2023-01-01": 3.0})
    assert cumulative_split_factor(splits, pd.Timestamp("2021-01-01")) == pytest.approx(6.0)


def test_splits_before_listing_are_ignored():
    splits = _series({"2018-01-01": 10.0, "2023-01-01": 2.0})
    assert cumulative_split_factor(splits, pd.Timestamp("2020-01-01")) == pytest.approx(2.0)


def test_split_on_the_listing_day_itself_is_ignored():
    splits = _series({"2020-06-01": 5.0})
    assert cumulative_split_factor(splits, pd.Timestamp("2020-06-01")) == 1.0


def test_timezone_aware_split_index_is_handled():
    index = pd.DatetimeIndex(["2023-01-01"]).tz_localize("America/New_York")
    splits = pd.Series([2.0], index=index)
    assert cumulative_split_factor(splits, pd.Timestamp("2021-01-01")) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Detecting whether Yahoo has actually restated the series
# ---------------------------------------------------------------------------


def _closes(before: float, after: float, split_date: str) -> pd.Series:
    dates = pd.bdate_range(pd.Timestamp(split_date) - pd.Timedelta(days=20), periods=30)
    split = pd.Timestamp(split_date)
    values = [before if d < split else after for d in dates]
    return pd.Series(values, index=dates)


def test_restated_series_is_detected_as_applied():
    # Continuous across the split: Yahoo has already folded it in.
    closes = _closes(100.0, 98.0, "2024-06-10")
    assert split_is_applied(closes, pd.Timestamp("2024-06-10"), 10.0) is True


def test_unrestated_series_is_detected_as_not_applied():
    # A 1-for-20 reverse split Yahoo has announced but not applied: the series
    # steps by 20x at the split date. Applying the factor anyway turned a
    # -33% first-day return into -97%.
    closes = _closes(5.0, 100.0, "2026-09-14")
    assert split_is_applied(closes, pd.Timestamp("2026-09-14"), 0.05) is False


def test_split_with_no_prices_after_it_is_assumed_applied():
    dates = pd.bdate_range("2024-01-01", periods=10)
    closes = pd.Series(np.full(10, 50.0), index=dates)
    assert split_is_applied(closes, pd.Timestamp("2030-01-01"), 2.0) is True


def test_a_unit_ratio_is_never_applied():
    closes = _closes(100.0, 100.0, "2024-06-10")
    assert split_is_applied(closes, pd.Timestamp("2024-06-10"), 1.0) is False


def test_cumulative_factor_skips_unapplied_splits():
    closes = _closes(5.0, 100.0, "2026-09-14")
    splits = _series({"2026-09-14": 0.05})
    assert cumulative_split_factor(splits, pd.Timestamp("2021-01-01"), closes=closes) == 1.0


# ---------------------------------------------------------------------------
# The committed sample must look like an IPO sample
# ---------------------------------------------------------------------------


def test_sample_first_day_returns_are_plausible(analysis_sample):
    """A guard against the defect that produced the previous results.

    The old target was a multi-year holding return: its median was -4.5% and
    its 25th percentile -68%. A first-day-return sample looks nothing like
    that.
    """
    returns = analysis_sample["underpricing"].dropna()
    assert len(returns) > 500
    assert returns.min() >= -1.0, "a first-day return cannot be below -100%"
    assert 0.0 <= returns.median() <= 0.25
    assert returns.quantile(0.25) > -0.30
    assert 0.5 < (returns > 0).mean() < 0.8


def test_sample_has_no_scrape_date_return_column(analysis_sample):
    for column in ["first_day_return_pct", "return_to_scrape_date", "price_at_scrape_date"]:
        assert column not in analysis_sample.columns
