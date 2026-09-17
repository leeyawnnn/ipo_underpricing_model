"""Temporal-leakage guards.

Every feature must be knowable before the stock's first trade. These tests
encode that as an assertion rather than a claim, because leakage is invisible
in the output: a leaky model just looks good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset import expanding_hot_market, market_as_of_prior_day
from src.models import (
    EXCLUDED_EXACT,
    NUMERIC_CANDIDATES,
    POST_LISTING_PREFIXES,
    select_features,
)


@pytest.fixture
def market() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=40)
    return pd.DataFrame(
        {
            "vix_close": np.arange(len(dates), dtype=float),
            "nasdaq_close": 100.0 + np.arange(len(dates), dtype=float),
            "nasdaq_30d_return": np.arange(len(dates), dtype=float) / 100,
            "nasdaq_30d_volatility": np.arange(len(dates), dtype=float) / 200,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Market features must be strictly lagged
# ---------------------------------------------------------------------------


def test_market_features_use_the_prior_trading_day(market):
    ipo_date = market.index[10]
    out = market_as_of_prior_day(pd.Series([ipo_date]), market)
    assert out["market_data_date"].iloc[0] == market.index[9]
    assert out["vix_at_pricing"].iloc[0] == market["vix_close"].iloc[9]
    # The listing day's own close must not leak in.
    assert out["vix_at_pricing"].iloc[0] != market["vix_close"].iloc[10]


def test_market_features_never_use_a_date_on_or_after_the_ipo(market):
    dates = pd.Series(market.index[5:25])
    out = market_as_of_prior_day(dates, market)
    assert (out["market_data_date"].to_numpy() < dates.to_numpy()).all()


def test_market_features_bridge_a_weekend(market):
    # Monday's IPO must read Friday's close, not the previous Monday's.
    monday = pd.Timestamp("2021-01-11")
    friday = pd.Timestamp("2021-01-08")
    out = market_as_of_prior_day(pd.Series([monday]), market)
    assert out["market_data_date"].iloc[0] == friday


def test_market_features_are_nan_before_the_series_starts(market):
    out = market_as_of_prior_day(pd.Series([pd.Timestamp("2020-01-01")]), market)
    assert np.isnan(out["vix_at_pricing"].iloc[0])


# ---------------------------------------------------------------------------
# The hot-market threshold must not see the future
# ---------------------------------------------------------------------------


def test_hot_market_threshold_is_expanding_not_full_sample():
    # A quiet first year followed by a boom. With a full-sample threshold the
    # early deals are all "cold" by construction; with an expanding threshold
    # some early deals are hot relative to what had been seen by then.
    quiet = pd.date_range("2019-01-01", periods=150, freq="7D")
    boom = pd.date_range("2021-01-01", periods=150, freq="1D")
    dates = pd.Series(list(quiet) + list(boom))

    flags = expanding_hot_market(dates, min_history=50)
    assert flags.isna().sum() == 50
    assert set(flags.dropna().unique()) <= {0.0, 1.0}
    # The boom period must be flagged hot.
    assert flags.iloc[-50:].mean() == 1.0


def test_hot_market_is_order_invariant():
    dates = pd.date_range("2019-01-01", periods=300, freq="3D").to_series().reset_index(drop=True)
    straight = expanding_hot_market(dates, min_history=40)
    shuffled_index = np.random.default_rng(0).permutation(len(dates))
    shuffled = expanding_hot_market(
        dates.iloc[shuffled_index].reset_index(drop=True), min_history=40
    )
    # Re-align the shuffled result back to the original order.
    realigned = shuffled.to_numpy()[np.argsort(shuffled_index)]
    np.testing.assert_allclose(straight.to_numpy(), realigned, equal_nan=True)


# ---------------------------------------------------------------------------
# The feature set must exclude anything observable at or after the first trade
# ---------------------------------------------------------------------------


def test_no_candidate_feature_is_a_post_listing_quantity():
    for column in NUMERIC_CANDIDATES:
        assert not column.startswith(POST_LISTING_PREFIXES), column
        assert column not in EXCLUDED_EXACT, column


def test_select_features_rejects_a_post_listing_column(monkeypatch):
    monkeypatch.setattr("src.models.NUMERIC_CANDIDATES", ["first_day_close"])
    frame = pd.DataFrame({"first_day_close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="post-listing"):
        select_features(frame)


def test_selected_features_are_all_pre_listing(analysis_sample):
    features = select_features(analysis_sample)
    banned = {
        "underpricing",
        "first_day_close",
        "first_day_open",
        "first_week_return",
        "first_month_return",
        "split_factor",
        "split_adjusted",
        "first_trade_date",
        "sector_encoded",
        "lead_underwriter_encoded",
        "winsorized_underpricing",
    }
    assert banned.isdisjoint(set(features.all_columns))


def test_no_target_encoded_column_survives_in_the_dataset(analysis_sample):
    # sector_encoded and lead_underwriter_encoded were the per-category mean of
    # the target over the whole sample. Their presence anywhere is leakage.
    offenders = [c for c in analysis_sample.columns if c.endswith("_encoded")]
    assert offenders == []


def test_synthetic_row_features_all_predate_listing():
    """Every source date attached to one IPO must be strictly before listing."""
    listing = pd.Timestamp("2022-06-15")
    market = pd.DataFrame(
        {
            "vix_close": [20.0, 21.0, 22.0],
            "nasdaq_close": [100.0, 101.0, 102.0],
            "nasdaq_30d_return": [0.01, 0.02, 0.03],
            "nasdaq_30d_volatility": [0.1, 0.2, 0.3],
        },
        index=pd.to_datetime(["2022-06-13", "2022-06-14", "2022-06-15"]),
    )
    out = market_as_of_prior_day(pd.Series([listing]), market)
    assert out["market_data_date"].iloc[0] < listing
