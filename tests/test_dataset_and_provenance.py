"""Tests for dataset assembly, the sample funnel and provenance stamping."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.dataset import load_market, sample_funnel, selection_comparison
from src.provenance import git_commit, sha256, write_table

# ---------------------------------------------------------------------------
# Market series
# ---------------------------------------------------------------------------


def test_load_market_adds_rolling_statistics(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=80)
    path = tmp_path / "market_indices.csv"
    pd.DataFrame(
        {
            "date": dates,
            "vix_close": np.linspace(15, 30, len(dates)),
            "nasdaq_close": np.linspace(8000, 9000, len(dates)),
        }
    ).to_csv(path, index=False)

    market = load_market(path)
    assert market.index.is_monotonic_increasing
    assert {"nasdaq_30d_return", "nasdaq_30d_volatility"} <= set(market.columns)
    # The first 30 observations cannot have a 30-day rolling statistic.
    assert market["nasdaq_30d_return"].iloc[:29].isna().all()
    assert market["nasdaq_30d_return"].iloc[35] > 0  # the series rises


def test_committed_market_series_covers_the_study_window():
    market = load_market()
    assert market.index.min() <= pd.Timestamp("2018-12-01")
    assert market.index.max() >= pd.Timestamp("2024-12-01")
    assert market["vix_close"].between(5, 90).all()


# ---------------------------------------------------------------------------
# Sample funnel
# ---------------------------------------------------------------------------


@pytest.fixture
def toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "offer_price": [10.0, 20.0, np.nan, 15.0],
            "first_day_close": [12.0, 18.0, np.nan, np.nan],
            "underpricing": [0.2, -0.1, np.nan, np.nan],
            "sic": ["2834", None, "6770", "7372"],
            "has_filing": [1, 0, 1, 1],
            "risk_factors_path": [None, None, None, None],
            "ipo_year": [2020, 2021, 2022, 2023],
            "vix_at_pricing": [20.0, 25.0, 30.0, 18.0],
            "is_spac": [0.0, 0.0, 1.0, 0.0],
        }
    )


def test_funnel_is_monotone_non_increasing(toy):
    funnel = sample_funnel(toy)
    assert funnel["n"].is_monotonic_decreasing
    assert funnel["n"].iloc[0] == len(toy)


def test_funnel_drops_reconcile(toy):
    funnel = sample_funnel(toy)
    for i in range(1, len(funnel)):
        assert funnel["n"].iloc[i - 1] - funnel["n"].iloc[i] == funnel["dropped"].iloc[i]


def test_funnel_percentages_are_of_the_universe(toy):
    funnel = sample_funnel(toy)
    assert funnel["pct_of_universe"].iloc[0] == 100.0
    assert funnel["pct_of_universe"].iloc[-1] <= 100.0


def test_committed_funnel_matches_the_committed_sample(analysis_sample):
    funnel = pd.read_csv("reports/tables/sample_funnel.csv")
    target_row = funnel[funnel["step"].str.contains("target")]
    assert int(target_row["n"].iloc[0]) == len(analysis_sample)


# ---------------------------------------------------------------------------
# Selection comparison
# ---------------------------------------------------------------------------


def test_selection_comparison_reports_both_groups(toy):
    out = selection_comparison(toy)
    assert {"observable", "in_sample", "out_of_sample", "difference", "p_value"} <= set(out.columns)
    assert len(out) == 5


def test_selection_comparison_detects_a_planted_difference():
    rng = np.random.default_rng(2)
    n = 400
    frame = pd.DataFrame(
        {
            "has_filing": [1] * n + [0] * n,
            "offer_price": np.concatenate([rng.normal(18, 3, n), rng.normal(10, 3, n)]),
            "ipo_year": rng.integers(2019, 2025, 2 * n),
            "underpricing": rng.normal(0.1, 0.3, 2 * n),
            "vix_at_pricing": rng.normal(20, 5, 2 * n),
            "is_spac": np.zeros(2 * n),
        }
    )
    out = selection_comparison(frame).set_index("observable")
    assert out.loc["Offer price (USD)", "p_value"] < 0.001
    assert out.loc["Offer price (USD)", "difference"] > 5
    assert out.loc["First-day return", "p_value"] > 0.05


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_write_table_emits_a_provenance_sidecar(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    frame = pd.DataFrame({"x": [1, 2, 3]})

    path = write_table(
        frame, "demo", "a demo table", command="pytest", inputs=[source], directory=tmp_path
    )
    assert path.exists()

    meta = json.loads((tmp_path / "demo.meta.json").read_text())
    assert meta["description"] == "a demo table"
    assert meta["command"] == "pytest"
    assert meta["rows"] == 3
    assert meta["input_sha256"] == sha256(source)
    assert meta["generated_utc"].endswith("+00:00")
    assert meta["git_commit"]


def test_sha256_of_a_missing_file_is_empty(tmp_path):
    assert sha256(tmp_path / "nope.csv") == ""


def test_sha256_is_stable_and_content_dependent(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    assert sha256(first) == sha256(second)
    second.write_bytes(b"world")
    assert sha256(first) != sha256(second)


def test_git_commit_returns_a_string():
    assert isinstance(git_commit(), str)
    assert git_commit()
