"""Tests for the hypothesis tests and the cross-validation protocol.

The hypothesis functions are checked against synthetic data with a planted
effect and against synthetic null data, so we know they detect what is there
and do not invent what is not. The CV tests pin the property that makes the
out-of-sample numbers meaningful: no test observation may precede a training
observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

# Aliased on import: the functions are named test_* in src, and pytest would
# otherwise collect them as test cases and fail on their required arguments.
from src.hypothesis_tests import adjust_family
from src.hypothesis_tests import test_h1_litigious_tone as run_h1
from src.hypothesis_tests import test_h3_disclosure_concentration as run_h3
from src.hypothesis_tests import test_h4_vix_variance as run_h4
from src.models import TARGET, build_pipeline, cross_validate, evaluate, select_features


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def test_bonferroni_multiplies_by_the_family_size():
    out = adjust_family({"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.5})
    row = out.set_index("test")
    assert row.loc["a", "p_bonferroni"] == pytest.approx(0.004)
    assert row.loc["c", "p_bonferroni"] == pytest.approx(0.16)


def test_bonferroni_is_capped_at_one():
    out = adjust_family({"a": 0.5, "b": 0.6, "c": 0.7})
    assert (out["p_bonferroni"] <= 1.0).all()


def test_bh_is_never_stricter_than_bonferroni():
    out = adjust_family({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.2, "e": 0.9})
    assert (out["p_bh"] <= out["p_bonferroni"] + 1e-12).all()


def test_bh_is_monotone_in_the_raw_p_value():
    out = adjust_family({f"t{i}": p for i, p in enumerate([0.001, 0.008, 0.02, 0.04, 0.3])})
    assert out["p_bh"].is_monotonic_increasing


def test_nan_p_values_are_excluded_from_the_family_size():
    out = adjust_family({"a": 0.01, "b": float("nan")})
    row = out.set_index("test")
    assert row.loc["a", "p_bonferroni"] == pytest.approx(0.01)   # family of one
    assert np.isnan(row.loc["b", "p_bonferroni"])
    assert not bool(row.loc["b", "significant_raw"])


# ---------------------------------------------------------------------------
# Planted effect and null data
# ---------------------------------------------------------------------------

def _synthetic(n: int = 400, effect: float = 0.0, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    litigious = rng.uniform(0.005, 0.04, n)
    noise = rng.normal(0, 0.20, n)
    return pd.DataFrame(
        {
            "underpricing": effect * (litigious - litigious.mean()) / litigious.std() + noise,
            "lm_litigious_ratio": litigious,
            "vix_at_pricing": rng.uniform(12, 40, n),
            "ipo_year": rng.integers(2019, 2025, n),
            "is_spac": 0.0,
            "sector": rng.choice(["Healthcare", "Technology"], n),
            "split_adjusted": 0,
        }
    )


def test_h1_detects_a_planted_negative_association():
    result = run_h1(_synthetic(effect=-0.30))
    assert result["p_value"] < 0.01
    assert result["statistic"] < 0
    assert result["reject_h0"] is True


def test_h1_detects_a_planted_positive_association():
    result = run_h1(_synthetic(effect=0.30))
    assert result["p_value"] < 0.01
    assert result["statistic"] > 0


def test_h1_does_not_reject_on_null_data():
    # Ten independent null draws: at alpha = 0.05 we expect at most a couple
    # of rejections by chance, never most of them.
    rejections = sum(
        run_h1(_synthetic(effect=0.0, seed=s))["reject_h0"]
        for s in range(10)
    )
    assert rejections <= 2


def test_h1_confidence_interval_covers_the_estimate():
    result = run_h1(_synthetic(effect=-0.30))
    low, high = result["ci_95"]
    assert low <= result["statistic"] <= high
    assert high < 0.0  # a real effect: the interval excludes zero


def test_h1_null_interval_contains_zero():
    result = run_h1(_synthetic(effect=0.0, seed=3))
    low, high = result["ci_95"]
    assert low < 0 < high


def test_h1_reports_every_quintile_with_a_count():
    result = run_h1(_synthetic(effect=-0.20))
    quintiles = result["quintiles"]
    assert len(quintiles) == 5
    assert (quintiles["n"] > 0).all()
    assert (quintiles["ci_low"] <= quintiles["median"]).all()
    assert (quintiles["median"] <= quintiles["ci_high"]).all()


def test_h3_returns_a_structured_result_when_the_column_is_absent():
    result = run_h3(_synthetic())
    assert result["reject_h0"] is False
    assert np.isnan(result["p_value"])


def test_h4_detects_planted_heteroscedasticity():
    rng = np.random.default_rng(5)
    n = 600
    vix = rng.uniform(10, 45, n)
    frame = pd.DataFrame({
        "vix_at_pricing": vix,
        # Dispersion rises with VIX; the centre does not move.
        "underpricing": rng.normal(0.05, 0.05 + vix / 100, n),
    })
    result = run_h4(frame)
    assert result["p_value"] < 0.01
    assert result["fligner_p_value"] < 0.01


def test_h4_does_not_reject_under_homoscedasticity():
    rng = np.random.default_rng(6)
    n = 600
    frame = pd.DataFrame({
        "vix_at_pricing": rng.uniform(10, 45, n),
        "underpricing": rng.normal(0.05, 0.3, n),
    })
    result = run_h4(frame)
    assert result["p_value"] > 0.05
    assert result["fligner_p_value"] > 0.05


# ---------------------------------------------------------------------------
# Cross-validation ordering
# ---------------------------------------------------------------------------

def test_time_series_split_never_trains_on_the_future():
    dates = pd.Series(pd.date_range("2019-01-01", periods=400, freq="3D"))
    X = pd.DataFrame({"x": np.arange(400, dtype=float)})
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
        assert dates.iloc[train_idx].max() < dates.iloc[test_idx].min()
        assert train_idx.max() < test_idx.min()


def test_cross_validate_folds_are_chronological(analysis_sample):
    per_fold, _ = cross_validate(analysis_sample.head(300), n_splits=3)
    ends = per_fold.drop_duplicates("fold").sort_values("fold")
    assert ends["train_end"].is_monotonic_increasing
    assert ends["test_end"].is_monotonic_increasing
    assert (ends["n_train"].diff().dropna() > 0).all()
    for _, row in ends.iterrows():
        assert row["train_end"] <= row["test_end"]


def test_baselines_are_present_in_every_fold(analysis_sample):
    per_fold, _ = cross_validate(analysis_sample.head(300), n_splits=3)
    for fold in per_fold["fold"].unique():
        models = set(per_fold.loc[per_fold["fold"] == fold, "model"])
        assert "Baseline: train mean" in models
        assert "Baseline: train median" in models


def test_preprocessing_is_fitted_inside_the_pipeline(analysis_sample):
    """A scaler fitted on the full dataset leaks test-fold moments into
    training. The transformer must live inside the estimator pipeline."""
    from sklearn.linear_model import Ridge

    features = select_features(analysis_sample)
    pipeline = build_pipeline(features, Ridge(alpha=1.0))
    assert list(pipeline.named_steps) == ["prep", "model"]

    train = analysis_sample.head(200)
    pipeline.fit(train[features.all_columns], train[TARGET].to_numpy(dtype=float))
    scaler = pipeline.named_steps["prep"].named_transformers_["numeric"].named_steps["scale"]
    fitted_on_train = scaler.mean_.copy()

    pipeline.fit(
        analysis_sample[features.all_columns],
        analysis_sample[TARGET].to_numpy(dtype=float),
    )
    fitted_on_all = pipeline.named_steps["prep"].named_transformers_["numeric"].named_steps["scale"].mean_
    assert not np.allclose(fitted_on_train, fitted_on_all)


def test_evaluate_on_perfect_and_constant_predictions():
    y = np.array([0.1, -0.2, 0.5, 0.0, 0.3])
    perfect = evaluate(y, y.copy())
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["rmse"] == pytest.approx(0.0)

    constant = evaluate(y, np.full_like(y, y.mean()))
    assert constant["r2"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(constant["spearman"])


# ---------------------------------------------------------------------------
# The remaining family members, on the real sample
# ---------------------------------------------------------------------------

def test_h2_h5_h6_return_complete_results(analysis_sample):
    from src.hypothesis_tests import (
        test_h2_underwriter_translation as run_h2,
    )
    from src.hypothesis_tests import (
        test_h5_underwriter_variance as run_h5,
    )
    from src.hypothesis_tests import (
        test_h6_text_features as run_h6,
    )

    for runner in (run_h2, run_h5, run_h6):
        result = runner(analysis_sample)
        assert result["hypothesis"]
        assert result["n"] > 0
        assert isinstance(result["reject_h0"], bool)
        assert "interpretation" in result


def test_h5_reports_both_a_robust_and_a_non_robust_test(analysis_sample):
    from src.hypothesis_tests import test_h5_underwriter_variance as run_h5

    result = run_h5(analysis_sample)
    assert "fligner_p_value" in result
    assert "ex_split_adjusted" in result
    # Spread is reported three ways so a single outlier cannot carry the claim.
    assert {"std", "iqr", "mad"} <= set(result["top_spread"])


def test_h6_compares_models_on_one_complete_case_sample(analysis_sample):
    from src.hypothesis_tests import test_h6_text_features as run_h6

    result = run_h6(analysis_sample)
    assert result["df_diff"] == len(result["text_features"])
    assert result["full_adj_r2"] is not None
    assert result["n"] <= len(analysis_sample)


def test_tests_with_missing_columns_degrade_gracefully():
    empty = pd.DataFrame({"underpricing": [0.1, 0.2, 0.3]})
    from src.hypothesis_tests import (
        test_h2_underwriter_translation as run_h2,
    )
    from src.hypothesis_tests import (
        test_h5_underwriter_variance as run_h5,
    )

    for runner in (run_h2, run_h5):
        result = runner(empty)
        assert result["reject_h0"] is False
        assert np.isnan(result["p_value"])


def test_h1_robustness_covers_the_required_specifications(analysis_sample):
    from src.hypothesis_tests import h1_robustness

    table = h1_robustness(analysis_sample)
    specifications = set(table["specification"])
    assert "Baseline (all filings)" in specifications
    assert "Excluding SPACs (SIC 6770)" in specifications
    assert "Excluding split-adjusted prices" in specifications
    assert any(s.startswith("Listing year") for s in specifications)
    assert "OLS: + log document length" in specifications
    assert "OLS: + sector and year fixed effects" in specifications
    assert set(table["kind"]) == {"spearman", "ols_beta"}
    finite = table.dropna(subset=["estimate"])
    assert (finite["ci_low"] <= finite["estimate"]).all()
    assert (finite["estimate"] <= finite["ci_high"]).all()


def test_run_all_publishes_both_correction_families(analysis_sample):
    from src.hypothesis_tests import run_all

    results = run_all(analysis_sample.head(400))
    assert {"H1", "H2", "H3", "H4", "H5", "H6"} <= set(results)
    for key in ("adjusted", "adjusted_robust", "h1_robustness"):
        assert isinstance(results[key], pd.DataFrame)
    assert len(results["adjusted"]) == 6


def test_report_prints_without_raising(analysis_sample, capsys):
    from src.hypothesis_tests import report

    report(run_h1(analysis_sample))
    captured = capsys.readouterr().out
    assert "Decision" in captured
    assert "95% CI" in captured
