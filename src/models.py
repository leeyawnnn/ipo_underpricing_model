"""
Out-of-sample prediction of first-day IPO returns.

The project is named for a prediction model and the previous version published
no predictive metric at all. This module produces them, under a protocol fixed
before the numbers were looked at.

**Protocol.**

* Evaluation is a five-fold expanding-window :class:`TimeSeriesSplit` over the
  sample sorted by listing date, plus one held-out period (listings from
  2024-01-01) that no model sees during selection.
* Every fold is scored against two baselines: predict the training mean and
  predict the training median. A model that cannot beat the training mean has
  negative out-of-sample R-squared, and that is the number we report.
* All preprocessing — median imputation, standardisation, sector one-hot —
  lives inside a :class:`~sklearn.pipeline.Pipeline` fitted on the training
  fold only. Nothing is fitted on the full dataset.
* Hyperparameters come from a small fixed grid chosen by an *inner*
  three-fold time-series split on the training fold, by MAE. The outer test
  fold is never used for selection. There is no Optuna search: with an
  out-of-sample R-squared near zero, a 60-trial search selects noise, and
  running one until a fold looks good is fitting the test set.
* Seeds are fixed. Re-running gives the same table.

**Why so few features survive.** ``select_features`` drops anything measured
at or after the first trade, anything derived from the target, and any column
constant on the training fold. The surviving count is reported, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import spearmanr

from src.utils import setup_logging

log = setup_logging(__name__)

RANDOM_STATE = 20240517
TARGET = "underpricing"
DATE_COLUMN = "ipo_date"
HOLDOUT_START = "2024-01-01"

# Columns that are the target, encode it, or are only observable once trading
# has begun. Anything matching these cannot enter the feature set.
POST_LISTING_PREFIXES = ("first_day_", "first_week", "first_month")
EXCLUDED_EXACT = frozenset(
    {
        "underpricing",
        "winsorized_underpricing",
        "offer_price",          # enters as log_offer_price
        "split_factor",
        "split_adjusted",
        "trading_day_lag",
        "has_filing",
        "cik",
        "sic",
        "n_underwriters_matched",
    }
)

NUMERIC_CANDIDATES = [
    "log_offer_price",
    "log_offer_size",
    "filed_shares_offered",
    "vix_at_pricing",
    "nasdaq_30d_return",
    "nasdaq_30d_volatility",
    "hot_market_dummy",
    "is_spac",
    "max_underwriter_rank",
    "lead_underwriter_rank",
    "top_tier_underwriter",
    "ipo_year",
    "ipo_quarter",
    "ipo_month",
    "ipo_dayofweek",
    "lm_negative_ratio",
    "lm_positive_ratio",
    "lm_uncertainty_ratio",
    "lm_litigious_ratio",
    "lm_constraining_ratio",
    "lm_modal_strong_ratio",
    "lm_modal_weak_ratio",
    "rf_lm_negative_ratio",
    "rf_lm_uncertainty_ratio",
    "rf_lm_litigious_ratio",
    "risk_concentration_ratio",
    "gunning_fog",
    "prospectus_uniqueness",
    "log_prospectus_words",
]
CATEGORICAL_CANDIDATES = ["sector"]


@dataclass
class FeatureSet:
    """The columns a model is allowed to see, and why others were dropped."""

    numeric: list[str]
    categorical: list[str]
    dropped_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)

    @property
    def all_columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]


def select_features(
    df: pd.DataFrame,
    min_coverage: float = 0.30,
) -> FeatureSet:
    """Choose modelling columns, excluding anything not knowable before trading.

    Args:
        df: Analysis sample.
        min_coverage: Minimum non-null share for a column to be kept.

    Returns:
        A :class:`FeatureSet` recording what was kept and what was dropped.

    Raises:
        ValueError: If a post-listing column reaches the candidate list, which
            would mean the exclusion rules have drifted.
    """
    numeric, categorical = [], []
    dropped_missing, dropped_constant = [], []

    for column in NUMERIC_CANDIDATES:
        if column not in df.columns:
            continue
        if column in EXCLUDED_EXACT or column.startswith(POST_LISTING_PREFIXES):
            raise ValueError(f"{column!r} is a post-listing or target column")
        series = df[column]
        if series.notna().mean() < min_coverage:
            dropped_missing.append(column)
        elif series.nunique(dropna=True) < 2:
            dropped_constant.append(column)
        else:
            numeric.append(column)

    for column in CATEGORICAL_CANDIDATES:
        if column in df.columns and df[column].nunique(dropna=True) >= 2:
            categorical.append(column)

    return FeatureSet(numeric, categorical, dropped_missing, dropped_constant)


def build_pipeline(features: FeatureSet, estimator: Any) -> Pipeline:
    """Wrap *estimator* behind per-fold imputation, scaling and one-hot encoding.

    Fitting the preprocessing inside the pipeline is what keeps the training
    fold's medians and category set out of the test fold.
    """
    # keep_empty_features keeps the matrix width stable when an early inner
    # fold happens to contain no observed value for a column (hot_market_dummy
    # is undefined for the first 100 deals, before its expanding threshold has
    # any history). Without it the column is dropped and the feature space
    # changes between folds.
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unclassified")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                     sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_steps, features.numeric),
            ("categorical", categorical_steps, features.categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", transformer), ("model", estimator)])


# ---------------------------------------------------------------------------
# Model grid
# ---------------------------------------------------------------------------

def candidate_models() -> dict[str, list[tuple[dict, Any]]]:
    """Return the fixed hyperparameter grid, declared before any evaluation.

    Returns:
        Model name to a list of ``(params, estimator)`` pairs. The inner
        time-series split picks one pair per outer training fold.
    """
    import lightgbm as lgb

    grid: dict[str, list[tuple[dict, Any]]] = {
        "Ridge": [
            ({"alpha": a}, Ridge(alpha=a, random_state=None))
            for a in (0.1, 1.0, 10.0, 100.0, 1000.0)
        ],
        "RandomForest": [
            (
                {"n_estimators": n, "max_depth": d, "min_samples_leaf": 10},
                RandomForestRegressor(
                    n_estimators=n, max_depth=d, min_samples_leaf=10,
                    random_state=RANDOM_STATE, n_jobs=-1,
                ),
            )
            for n in (300,)
            for d in (3, 5, None)
        ],
        "LightGBM": [
            (
                {"n_estimators": n, "learning_rate": lr, "num_leaves": leaves,
                 "min_child_samples": 20},
                lgb.LGBMRegressor(
                    n_estimators=n, learning_rate=lr, num_leaves=leaves,
                    min_child_samples=20, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
                ),
            )
            for n in (200, 600)
            for lr in (0.03, 0.1)
            for leaves in (15, 31)
        ],
    }
    return grid


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return R-squared, RMSE, MAE and Spearman rho for one set of predictions."""
    rho = float("nan")
    if len(np.unique(y_pred)) > 1 and len(y_true) > 2:
        rho = float(spearmanr(y_true, y_pred).statistic)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": rho,
    }


def _select_on_inner_split(
    X: pd.DataFrame,
    y: np.ndarray,
    features: FeatureSet,
    grid: list[tuple[dict, Any]],
    n_inner: int = 3,
) -> tuple[dict, Any]:
    """Pick one grid point using an inner time-series split of the training fold."""
    if len(grid) == 1 or len(X) < (n_inner + 1) * 20:
        return grid[0]

    inner = TimeSeriesSplit(n_splits=n_inner)
    best_score, best = np.inf, grid[0]
    for params, estimator in grid:
        scores = []
        for train_idx, valid_idx in inner.split(X):
            pipeline = build_pipeline(features, _clone(estimator))
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            scores.append(mean_absolute_error(y[valid_idx], pipeline.predict(X.iloc[valid_idx])))
        score = float(np.mean(scores))
        if score < best_score:
            best_score, best = score, (params, estimator)
    return best


def _clone(estimator: Any) -> Any:
    from sklearn.base import clone

    return clone(estimator)


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------

def cross_validate(
    df: pd.DataFrame,
    target_col: str = TARGET,
    n_splits: int = 5,
    winsorise: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model and both baselines on an expanding-window split.

    Args:
        df: Analysis sample. Sorted by listing date internally.
        target_col: Target column.
        n_splits: Outer folds.
        winsorise: Clip the target at the *training fold's* 1st and 99th
            percentiles before fitting and scoring. Fitted on the training
            fold only, so it is not look-ahead. Reported as a separate table
            because OLS-style losses are not robust to the raw right tail.

    Returns:
        ``(per_fold, summary)``. ``per_fold`` has one row per model and fold;
        ``summary`` averages across folds and reports the spread, which on IPO
        data is large enough that an average alone is misleading.
    """
    data = df.dropna(subset=[target_col]).sort_values(DATE_COLUMN).reset_index(drop=True)
    features = select_features(data)
    X = data[features.all_columns]
    y = data[target_col].to_numpy(dtype=float)

    log.info(
        "Cross-validating on n=%d with %d numeric + %d categorical features "
        "(dropped %d for coverage, %d constant)",
        len(data), len(features.numeric), len(features.categorical),
        len(features.dropped_missing), len(features.dropped_constant),
    )

    grid = candidate_models()
    rows: list[dict] = []
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        y_train, y_test = y[train_idx], y[test_idx]
        if winsorise:
            low, high = np.percentile(y_train, [1, 99])
            y_train = np.clip(y_train, low, high)
            y_test = np.clip(y_test, low, high)

        common = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_end": str(data[DATE_COLUMN].iloc[train_idx[-1]].date()),
            "test_end": str(data[DATE_COLUMN].iloc[test_idx[-1]].date()),
        }

        for name, value in [("Baseline: train mean", float(np.mean(y_train))),
                            ("Baseline: train median", float(np.median(y_train)))]:
            rows.append({**common, "model": name,
                         **evaluate(y_test, np.full(len(y_test), value))})

        for name, model_grid in grid.items():
            params, estimator = _select_on_inner_split(
                X.iloc[train_idx], y_train, features, model_grid
            )
            pipeline = build_pipeline(features, _clone(estimator))
            pipeline.fit(X.iloc[train_idx], y_train)
            rows.append({
                **common,
                "model": name,
                **evaluate(y_test, pipeline.predict(X.iloc[test_idx])),
                "params": str(params),
            })

    per_fold = pd.DataFrame(rows)
    summary = (
        per_fold.groupby("model")[["r2", "rmse", "mae", "spearman"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("r2_mean", ascending=False)
    return per_fold, summary


def holdout_evaluation(
    df: pd.DataFrame,
    target_col: str = TARGET,
    holdout_start: str = HOLDOUT_START,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Train on everything before *holdout_start* and score the period after it.

    Args:
        df: Analysis sample.
        target_col: Target column.
        holdout_start: ISO date opening the held-out period.

    Returns:
        ``(metrics, predictions, test_frame)``. ``predictions`` maps model name
        to its held-out predictions, aligned with ``test_frame``.
    """
    data = df.dropna(subset=[target_col]).sort_values(DATE_COLUMN).reset_index(drop=True)
    features = select_features(data)
    cutoff = pd.Timestamp(holdout_start)
    train = data[data[DATE_COLUMN] < cutoff]
    test = data[data[DATE_COLUMN] >= cutoff]

    X_train, y_train = train[features.all_columns], train[target_col].to_numpy(dtype=float)
    X_test, y_test = test[features.all_columns], test[target_col].to_numpy(dtype=float)

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    common = {"n_train": len(train), "n_test": len(test),
              "holdout_start": holdout_start}

    for name, value in [("Baseline: train mean", float(np.mean(y_train))),
                        ("Baseline: train median", float(np.median(y_train)))]:
        pred = np.full(len(y_test), value)
        predictions[name] = pred
        rows.append({**common, "model": name, **evaluate(y_test, pred)})

    for name, model_grid in candidate_models().items():
        params, estimator = _select_on_inner_split(X_train, y_train, features, model_grid)
        pipeline = build_pipeline(features, _clone(estimator))
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        predictions[name] = pred
        rows.append({**common, "model": name, **evaluate(y_test, pred), "params": str(params)})

    return pd.DataFrame(rows).sort_values("r2", ascending=False), predictions, test


def fit_final_lightgbm(df: pd.DataFrame, target_col: str = TARGET) -> tuple[Pipeline, pd.DataFrame, FeatureSet]:
    """Fit LightGBM on the whole sample, for SHAP inspection only.

    The returned model is not evaluated: it has seen every row. It exists so
    that :func:`shap_summary` can report what the model keys on, which is a
    statement about the model and not about the world.
    """
    import lightgbm as lgb

    data = df.dropna(subset=[target_col]).sort_values(DATE_COLUMN).reset_index(drop=True)
    features = select_features(data)
    estimator = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    pipeline = build_pipeline(features, estimator)
    pipeline.fit(data[features.all_columns], data[target_col].to_numpy(dtype=float))
    return pipeline, data, features


def shap_summary(pipeline: Pipeline, data: pd.DataFrame, features: FeatureSet) -> tuple[Any, pd.DataFrame, np.ndarray]:
    """Compute SHAP values for the fitted LightGBM pipeline.

    SHAP explains the model, not the data-generating process. When the model's
    out-of-sample R-squared is approximately zero, a large mean absolute SHAP
    value identifies what the model leaned on while failing to generalise. It
    is not evidence that the feature drives underpricing.

    Returns:
        ``(shap_values, importance, transformed)`` where ``importance`` ranks
        features by mean absolute SHAP value.
    """
    import shap

    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    transformed = prep.transform(data[features.all_columns])
    names = list(prep.get_feature_names_out())
    frame = pd.DataFrame(transformed, columns=names)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(frame)

    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return values, importance, frame
