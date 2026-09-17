"""
Hypothesis tests on the cross-section of first-day IPO returns.

Six hypotheses are tested as a family. Three concern prospectus text, three
are stylised-fact replications:

  H1  Litigious tone. Does the share of Loughran-McDonald *Litigious* words in
      the S-1 predict the first-day return?
  H2  Underwriter translation. Is any H1 effect weaker when a top-tier bank
      certifies the deal (Carter & Manaster 1990)?
  H3  Disclosure concentration. Does it matter whether negative tone is
      compartmentalised in Risk Factors or spread across the prospectus?
  H4  Does VIX change the *variance* of first-day returns?
  H5  Do top-tier underwriters reduce that variance?
  H6  Do text features add explanatory power over deal and market features?

**Pre-registration.** These were not pre-registered. H1 and H3 were specified
after exploratory work on the same data, so their p-values are optimistic in a
way no correction fully repairs; they are reported as exploratory. H2, H4, H5
and H6 are directional predictions taken from the prior literature.

**Multiple testing.** Six tests at alpha = 0.05 carry a family-wise error rate
of about 26%. :func:`adjust_family` reports Bonferroni and
Benjamini-Hochberg adjusted p-values alongside the raw ones, and the README
quotes both.

**Controls.** The previous version used ``sector_encoded`` — the mean of the
target within each sector, computed over the whole sample — as a regression
control. Conditioning on a function of the dependent variable inflates fit and
distorts every other coefficient. Sector now enters as fixed effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.plotting.style import bootstrap_ci, spearman_ci
from src.utils import setup_logging

log = setup_logging(__name__)

ALPHA = 0.05
TARGET = "underpricing"
LITIGIOUS = "lm_litigious_ratio"

# The seven Loughran-McDonald categories, for the within-family comparison
# that H1 leans on.
LM_CATEGORIES = [
    "lm_negative_ratio",
    "lm_positive_ratio",
    "lm_uncertainty_ratio",
    "lm_litigious_ratio",
    "lm_constraining_ratio",
    "lm_modal_strong_ratio",
    "lm_modal_weak_ratio",
]


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------

def adjust_family(p_values: dict[str, float], alpha: float = ALPHA) -> pd.DataFrame:
    """Apply Bonferroni and Benjamini-Hochberg across a family of tests.

    Args:
        p_values: Test label to raw p-value. ``nan`` entries are carried
            through and excluded from the family size.
        alpha: Family-wise / false-discovery rate.

    Returns:
        DataFrame with ``test``, ``p_raw``, ``p_bonferroni``, ``p_bh``,
        ``significant_raw``, ``significant_bonferroni`` and
        ``significant_bh``, ordered by raw p-value.

    Example:
        >>> out = adjust_family({"a": 0.001, "b": 0.04})
        >>> float(out.loc[out.test == "a", "p_bonferroni"].iloc[0])
        0.002
    """
    frame = pd.DataFrame(
        {"test": list(p_values), "p_raw": [p_values[k] for k in p_values]}
    ).sort_values("p_raw", na_position="last").reset_index(drop=True)

    testable = frame["p_raw"].notna()
    m = int(testable.sum())

    frame["p_bonferroni"] = np.where(testable, np.minimum(frame["p_raw"] * m, 1.0), np.nan)

    # Benjamini-Hochberg, with the standard monotonicity enforcement.
    frame["p_bh"] = np.nan
    if m > 0:
        ranks = np.arange(1, m + 1)
        raw = frame.loc[testable, "p_raw"].to_numpy()
        bh = np.minimum.accumulate((raw * m / ranks)[::-1])[::-1]
        frame.loc[testable, "p_bh"] = np.minimum(bh, 1.0)

    for column, source in [
        ("significant_raw", "p_raw"),
        ("significant_bonferroni", "p_bonferroni"),
        ("significant_bh", "p_bh"),
    ]:
        frame[column] = frame[source] < alpha
    return frame


# ---------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------

def test_h1_litigious_tone(
    df: pd.DataFrame,
    target_col: str = TARGET,
    lit_col: str = LITIGIOUS,
    alpha: float = ALPHA,
    n_quintiles: int = 5,
) -> dict[str, Any]:
    """Test whether litigious prospectus language predicts the first-day return.

    Spearman's rho on the full text sample, a Kruskal-Wallis across quintiles
    of the litigious ratio, and bootstrap intervals on every quintile median so
    the step pattern can be read against its own sampling error.

    The previous implementation dropped observations below -50%, describing
    them as delisting artifacts. They were: the target was a multi-year holding
    return, so a -100% row meant the company had collapsed since listing. With
    a genuine first-day return there is nothing to filter, and no filter is
    applied here.

    Args:
        df: Analysis dataset.
        target_col: First-day return column.
        lit_col: LM litigious ratio column.
        alpha: Significance level.
        n_quintiles: Number of quantile bins.

    Returns:
        Results dict with rho and its bootstrap CI, the Kruskal-Wallis
        statistic, per-quintile medians with CIs and counts, and the
        per-category Spearman comparison across all seven LM categories.
    """
    data = df[[target_col, lit_col]].dropna()
    n = len(data)
    if n < 30:
        return {
            "hypothesis": "H1: litigious tone and first-day return",
            "test": "insufficient data",
            "n": n,
            "p_value": float("nan"),
            "reject_h0": False,
        }

    rho, rho_lo, rho_hi = spearman_ci(data[lit_col], data[target_col])
    _, p_value = stats.spearmanr(data[lit_col], data[target_col])

    labels = [f"Q{i}" for i in range(1, n_quintiles + 1)]
    data = data.assign(
        _bin=pd.qcut(data[lit_col], q=n_quintiles, labels=labels, duplicates="drop")
    )

    quintiles = []
    groups = []
    for label, group in data.groupby("_bin", observed=True):
        point, lo, hi = bootstrap_ci(group[target_col].to_numpy())
        quintiles.append(
            {
                "quintile": str(label),
                "n": int(len(group)),
                "median": point,
                "ci_low": lo,
                "ci_high": hi,
                "litigious_median": float(group[lit_col].median()),
            }
        )
        groups.append(group[target_col].to_numpy())

    kw_stat, kw_p = stats.kruskal(*groups) if len(groups) > 1 else (float("nan"), float("nan"))

    # Is Litigious special, or does every LM category behave this way?
    by_category = []
    for column in LM_CATEGORIES:
        if column not in df.columns:
            continue
        pair = df[[target_col, column]].dropna()
        if len(pair) < 30:
            continue
        category_rho, lo, hi = spearman_ci(pair[column], pair[target_col])
        _, category_p = stats.spearmanr(pair[column], pair[target_col])
        by_category.append(
            {
                "category": column,
                "rho": category_rho,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": float(category_p),
                "n": int(len(pair)),
            }
        )
    category_frame = pd.DataFrame(by_category)
    if not category_frame.empty:
        category_frame = category_frame.merge(
            adjust_family(dict(zip(category_frame["category"], category_frame["p_value"])))
            .rename(columns={"test": "category"})[["category", "p_bonferroni", "p_bh"]],
            on="category",
            how="left",
        )

    first, last = quintiles[0], quintiles[-1]
    return {
        "hypothesis": "H1: litigious tone and first-day return",
        "h0": "No monotone association between the LM litigious ratio and the first-day return",
        "h1": "A monotone association exists",
        "test": "Spearman rho with a percentile bootstrap CI, plus Kruskal-Wallis across quintiles",
        "statistic": rho,
        "p_value": float(p_value),
        "effect_size": rho,
        "ci_95": (rho_lo, rho_hi),
        "kw_statistic": float(kw_stat),
        "kw_p_value": float(kw_p),
        "quintiles": pd.DataFrame(quintiles),
        "lm_categories": category_frame,
        "n": n,
        "reject_h0": bool(p_value < alpha),
        "interpretation": (
            f"Spearman rho = {rho:+.3f}, 95% CI [{rho_lo:+.3f}, {rho_hi:+.3f}], "
            f"p = {p_value:.4g} on n = {n}. Median first-day return moves from "
            f"{first['median']:+.1%} in {first['quintile']} (n = {first['n']}) to "
            f"{last['median']:+.1%} in {last['quintile']} (n = {last['n']}); "
            f"Kruskal-Wallis H = {kw_stat:.2f}, p = {kw_p:.4g}."
        ),
    }


def h1_robustness(
    df: pd.DataFrame,
    target_col: str = TARGET,
    lit_col: str = LITIGIOUS,
) -> pd.DataFrame:
    """Re-estimate the H1 association under specifications that could break it.

    A bivariate rank correlation is not evidence of a mechanism. Each row is a
    specification designed to remove a competing explanation; rows where the
    estimate collapses are as informative as rows where it survives.

    Args:
        df: Analysis dataset.
        target_col: First-day return column.
        lit_col: LM litigious ratio column.

    Returns:
        DataFrame with ``specification``, ``n``, ``estimate``, ``ci_low``,
        ``ci_high``, ``p_value`` and ``kind`` (``spearman`` or ``ols_beta``).
    """
    rows: list[dict] = []

    def add_spearman(label: str, frame: pd.DataFrame) -> None:
        pair = frame[[target_col, lit_col]].dropna()
        if len(pair) < 30:
            rows.append({"specification": label, "n": len(pair), "estimate": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                         "kind": "spearman"})
            return
        rho, lo, hi = spearman_ci(pair[lit_col], pair[target_col])
        _, p = stats.spearmanr(pair[lit_col], pair[target_col])
        rows.append({"specification": label, "n": len(pair), "estimate": rho,
                     "ci_low": lo, "ci_high": hi, "p_value": float(p), "kind": "spearman"})

    add_spearman("Baseline (all filings)", df)

    # SIC 6770 ("Blank Checks") identifies SPACs. None survive into the
    # analysis sample - Yahoo purges the shell ticker once a SPAC de-SPACs or
    # liquidates - so these rows document the absence rather than test it.
    if "is_spac" in df.columns:
        add_spearman("Excluding SPACs (SIC 6770)", df[df["is_spac"] != 1])
        add_spearman("SPACs only", df[df["is_spac"] == 1])

    if "split_adjusted" in df.columns:
        add_spearman("Excluding split-adjusted prices", df[df["split_adjusted"] == 0])

    for year in sorted(pd.to_numeric(df["ipo_year"], errors="coerce").dropna().unique()):
        add_spearman(f"Listing year {int(year)}", df[df["ipo_year"] == year])

    # Multivariate specifications. Each adds one layer of control.
    specs: list[tuple[str, list[str], bool, bool]] = [
        ("OLS: litigious only", [], False, False),
        ("OLS: + deal and market controls", ["log_offer_price", "log_offer_size",
                                             "vix_at_pricing", "nasdaq_30d_return",
                                             "is_spac"], False, False),
        ("OLS: + log document length", ["log_offer_price", "log_offer_size",
                                        "vix_at_pricing", "nasdaq_30d_return",
                                        "is_spac", "log_prospectus_words"], False, False),
        ("OLS: + sector fixed effects", ["log_offer_price", "log_offer_size",
                                         "vix_at_pricing", "nasdaq_30d_return",
                                         "is_spac", "log_prospectus_words"], True, False),
        ("OLS: + sector and year fixed effects", ["log_offer_price", "log_offer_size",
                                                  "vix_at_pricing", "nasdaq_30d_return",
                                                  "is_spac", "log_prospectus_words"], True, True),
    ]
    for label, controls, sector_fe, year_fe in specs:
        result = _ols_litigious(df, target_col, lit_col, controls, sector_fe, year_fe)
        rows.append({"specification": label, **result, "kind": "ols_beta"})

    return pd.DataFrame(rows)


def _ols_litigious(
    df: pd.DataFrame,
    target_col: str,
    lit_col: str,
    controls: list[str],
    sector_fe: bool,
    year_fe: bool,
) -> dict[str, float]:
    """Fit one OLS specification and return the litigious coefficient."""
    usable = [c for c in controls if c in df.columns and df[c].nunique(dropna=True) > 1]
    columns = [target_col, lit_col, *usable]
    frame = df[columns].copy()
    if sector_fe:
        frame["sector"] = df["sector"]
    if year_fe:
        frame["ipo_year"] = df["ipo_year"]
    frame = frame.dropna()
    if len(frame) < 40:
        return {"n": len(frame), "estimate": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "p_value": np.nan}

    design = frame[[lit_col, *usable]].astype(float)
    if sector_fe:
        dummies = pd.get_dummies(frame["sector"], prefix="sector", drop_first=True, dtype=float)
        design = pd.concat([design, dummies.loc[:, dummies.nunique() > 1]], axis=1)
    if year_fe:
        design = pd.concat(
            [design, pd.get_dummies(frame["ipo_year"].astype(int), prefix="year",
                                    drop_first=True, dtype=float)],
            axis=1,
        )
    design = sm.add_constant(design)
    # Heteroscedasticity-robust: first-day returns are strongly non-normal.
    model = sm.OLS(frame[target_col].astype(float), design).fit(cov_type="HC3")
    ci = model.conf_int().loc[lit_col]
    return {
        "n": int(model.nobs),
        "estimate": float(model.params[lit_col]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(model.pvalues[lit_col]),
    }


# ---------------------------------------------------------------------------
# H2
# ---------------------------------------------------------------------------

def test_h2_underwriter_translation(
    df: pd.DataFrame,
    target_col: str = TARGET,
    lit_col: str = LITIGIOUS,
    tier_col: str = "top_tier_underwriter",
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Test whether top-tier certification attenuates the litigious-tone effect.

    Carter & Manaster (1990): a prestigious underwriter certifies the issue, so
    the marginal information in the prospectus language should matter less. The
    prediction is a positive ``litigious x top_tier`` interaction and a
    less-negative Spearman rho inside the top-tier subsample.

    Args:
        df: Analysis dataset.
        target_col: First-day return column.
        lit_col: LM litigious ratio column.
        tier_col: Binary top-tier flag from the Carter-Manaster rank.
        alpha: Significance level.

    Returns:
        Results dict with the interaction coefficient, both subsample rhos and
        a bootstrap CI on their difference.
    """
    needed = [target_col, lit_col, tier_col]
    if any(c not in df.columns for c in needed):
        return {"hypothesis": "H2: underwriter translation", "test": "columns missing",
                "p_value": float("nan"), "reject_h0": False, "n": 0}

    data = df[needed + ["log_offer_price", "log_offer_size", "vix_at_pricing",
                        "nasdaq_30d_return", "is_spac", "sector", "ipo_year"]].dropna(
        subset=needed
    )
    top = data[data[tier_col] == 1]
    other = data[data[tier_col] == 0]
    if len(top) < 30 or len(other) < 30:
        return {
            "hypothesis": "H2: underwriter translation",
            "test": f"insufficient subsample (top n={len(top)}, other n={len(other)})",
            "p_value": float("nan"),
            "reject_h0": False,
            "n": len(data),
        }

    top_rho, top_lo, top_hi = spearman_ci(top[lit_col], top[target_col])
    other_rho, other_lo, other_hi = spearman_ci(other[lit_col], other[target_col])

    rng = np.random.default_rng(20240517)
    diffs = np.empty(5_000)
    top_x, top_y = top[lit_col].to_numpy(), top[target_col].to_numpy()
    other_x, other_y = other[lit_col].to_numpy(), other[target_col].to_numpy()
    for i in range(diffs.size):
        ti = rng.integers(0, top_x.size, top_x.size)
        oi = rng.integers(0, other_x.size, other_x.size)
        diffs[i] = (
            stats.spearmanr(top_x[ti], top_y[ti]).statistic
            - stats.spearmanr(other_x[oi], other_y[oi]).statistic
        )
    diff_lo, diff_hi = np.nanpercentile(diffs, [2.5, 97.5])

    frame = data.dropna(subset=["log_offer_price", "log_offer_size", "vix_at_pricing",
                                "nasdaq_30d_return", "is_spac"]).copy()
    frame["_interaction"] = frame[lit_col] * frame[tier_col]
    design = frame[[lit_col, tier_col, "_interaction", "log_offer_price", "log_offer_size",
                    "vix_at_pricing", "nasdaq_30d_return", "is_spac"]].astype(float)
    design = pd.concat(
        [design, pd.get_dummies(frame["sector"], prefix="sector", drop_first=True, dtype=float)],
        axis=1,
    )
    design = sm.add_constant(design)
    model = sm.OLS(frame[target_col].astype(float), design).fit(cov_type="HC3")

    interaction_p = float(model.pvalues["_interaction"])
    return {
        "hypothesis": "H2: underwriter translation",
        "h0": "The litigious x top_tier interaction coefficient is zero",
        "h1": "Positive interaction: top-tier certification weakens the main effect",
        "test": "OLS interaction with HC3 errors, plus a bootstrap difference of Spearman rho",
        "statistic": float(model.params["_interaction"]),
        "p_value": interaction_p,
        "main_coefficient": float(model.params[lit_col]),
        "main_p_value": float(model.pvalues[lit_col]),
        "top_rho": top_rho,
        "top_ci": (top_lo, top_hi),
        "top_n": int(len(top)),
        "other_rho": other_rho,
        "other_ci": (other_lo, other_hi),
        "other_n": int(len(other)),
        "rho_difference": top_rho - other_rho,
        "rho_difference_ci": (float(diff_lo), float(diff_hi)),
        "n": int(model.nobs),
        "reject_h0": bool(interaction_p < alpha),
        "interpretation": (
            f"Interaction beta = {model.params['_interaction']:+.2f} "
            f"(p = {interaction_p:.4g}) on n = {int(model.nobs)}. "
            f"Top-tier rho = {top_rho:+.3f} [{top_lo:+.3f}, {top_hi:+.3f}] (n = {len(top)}) "
            f"versus other rho = {other_rho:+.3f} [{other_lo:+.3f}, {other_hi:+.3f}] "
            f"(n = {len(other)}); difference {top_rho - other_rho:+.3f}, "
            f"95% CI [{diff_lo:+.3f}, {diff_hi:+.3f}]."
        ),
    }


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------

def test_h3_disclosure_concentration(
    df: pd.DataFrame,
    target_col: str = TARGET,
    ratio_col: str = "risk_concentration_ratio",
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Test whether the *location* of negative tone predicts the first-day return.

    ``risk_concentration_ratio`` is the LM-negative intensity of the Risk
    Factors section over that of the whole prospectus. High means the negative
    language is compartmentalised where the SEC asks for it; low means it
    bleeds through the document.

    Args:
        df: Analysis dataset.
        target_col: First-day return column.
        ratio_col: Concentration ratio column.
        alpha: Significance level.

    Returns:
        Results dict with Spearman rho and CI, Kruskal-Wallis across terciles,
        and per-tercile medians with bootstrap intervals.
    """
    if ratio_col not in df.columns:
        return {"hypothesis": "H3: disclosure concentration", "test": "column missing",
                "p_value": float("nan"), "reject_h0": False, "n": 0}

    data = df[[target_col, ratio_col]].dropna()
    data = data[np.isfinite(data[ratio_col])]
    n = len(data)
    if n < 30:
        return {"hypothesis": "H3: disclosure concentration", "test": "insufficient data",
                "p_value": float("nan"), "reject_h0": False, "n": n}

    rho, lo, hi = spearman_ci(data[ratio_col], data[target_col])
    _, p_value = stats.spearmanr(data[ratio_col], data[target_col])

    labels = ["T1 pervasive", "T2 mid", "T3 compartmentalised"]
    data = data.assign(_bin=pd.qcut(data[ratio_col], q=3, labels=labels, duplicates="drop"))

    terciles, groups = [], []
    for label, group in data.groupby("_bin", observed=True):
        point, tlo, thi = bootstrap_ci(group[target_col].to_numpy())
        terciles.append({"tercile": str(label), "n": int(len(group)), "median": point,
                         "ci_low": tlo, "ci_high": thi,
                         "concentration_median": float(group[ratio_col].median())})
        groups.append(group[target_col].to_numpy())
    kw_stat, kw_p = stats.kruskal(*groups) if len(groups) > 1 else (float("nan"), float("nan"))

    return {
        "hypothesis": "H3: disclosure concentration",
        "h0": "No monotone association between risk_concentration_ratio and the first-day return",
        "h1": "A monotone association exists",
        "test": "Spearman rho with a percentile bootstrap CI, plus Kruskal-Wallis across terciles",
        "statistic": rho,
        "p_value": float(p_value),
        "effect_size": rho,
        "ci_95": (lo, hi),
        "kw_statistic": float(kw_stat),
        "kw_p_value": float(kw_p),
        "terciles": pd.DataFrame(terciles),
        "n": n,
        "reject_h0": bool(p_value < alpha),
        "interpretation": (
            f"Spearman rho = {rho:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], "
            f"p = {p_value:.4g} on n = {n}. Kruskal-Wallis H = {kw_stat:.2f}, p = {kw_p:.4g}."
        ),
    }


# ---------------------------------------------------------------------------
# H4 and H5
# ---------------------------------------------------------------------------


def _dispersion(values: np.ndarray) -> dict[str, float]:
    """Return several spread measures, from outlier-sensitive to robust."""
    return {
        "n": int(values.size),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "mad": float(stats.median_abs_deviation(values)),
    }


def test_h4_vix_variance(
    df: pd.DataFrame,
    target_col: str = TARGET,
    vix_col: str = "vix_at_pricing",
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Levene's test for equal first-day-return variance across VIX terciles."""
    data = df[[target_col, vix_col]].dropna()
    if len(data) < 60:
        return {"hypothesis": "H4: VIX and return variance", "test": "insufficient data",
                "p_value": float("nan"), "reject_h0": False, "n": len(data)}

    data = data.assign(_bin=pd.qcut(data[vix_col], q=3, labels=["Low", "Mid", "High"]))
    groups, spreads = [], {}
    for label, group in data.groupby("_bin", observed=True):
        values = group[target_col].to_numpy()
        groups.append(values)
        spreads[str(label)] = {**_dispersion(values), "vix_median": float(group[vix_col].median())}
    statistic, p_value = stats.levene(*groups, center="median")
    # Levene remains sensitive to extreme values even when median-centred, and
    # first-day returns have a very heavy right tail. Fligner-Killeen works on
    # ranks and is reported alongside it, not instead of it.
    fk_statistic, fk_p = stats.fligner(*groups)

    return {
        "hypothesis": "H4: VIX and return variance",
        "h0": "First-day-return variance is equal across VIX terciles",
        "h1": "Variance differs across VIX terciles",
        "test": "Levene's test, median-centred",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "fligner_statistic": float(fk_statistic),
        "fligner_p_value": float(fk_p),
        "groups": spreads,
        "n": int(len(data)),
        "reject_h0": bool(p_value < alpha),
        "robust_reject_h0": bool(fk_p < alpha),
        "interpretation": (
            f"Levene W = {statistic:.2f}, p = {p_value:.4g}; rank-based Fligner-Killeen "
            f"chi-square = {fk_statistic:.2f}, p = {fk_p:.4g}; n = {len(data)}. "
            + "; ".join(
                f"{k}: sd = {v['std']:.1%}, IQR = {v['iqr']:.1%} (n = {v['n']})"
                for k, v in spreads.items()
            )
        ),
    }


def test_h5_underwriter_variance(
    df: pd.DataFrame,
    target_col: str = TARGET,
    tier_col: str = "top_tier_underwriter",
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Levene's test for equal variance between top-tier and other underwriters."""
    if tier_col not in df.columns:
        return {"hypothesis": "H5: underwriter tier and return variance",
                "test": "column missing", "p_value": float("nan"),
                "reject_h0": False, "n": 0}

    data = df[[target_col, tier_col]].dropna()
    top = data.loc[data[tier_col] == 1, target_col].to_numpy()
    other = data.loc[data[tier_col] == 0, target_col].to_numpy()
    if len(top) < 30 or len(other) < 30:
        return {
            "hypothesis": "H5: underwriter tier and return variance",
            "test": f"insufficient subsample (top n={len(top)}, other n={len(other)})",
            "p_value": float("nan"), "reject_h0": False, "n": len(data),
        }

    statistic, p_value = stats.levene(top, other, center="median")
    fk_statistic, fk_p = stats.fligner(top, other)
    top_spread, other_spread = _dispersion(top), _dispersion(other)

    # The same comparison with split-adjusted prices removed. Reverse splits
    # concentrate in the weakest micro-cap deals, which are also the rows whose
    # first-day price is least reliable and are almost all non-top-tier, so a
    # variance gap can be manufactured entirely by those observations.
    subsample: dict[str, float] = {}
    if "split_adjusted" in df.columns:
        clean = df[df["split_adjusted"] == 0][[target_col, tier_col]].dropna()
        clean_top = clean.loc[clean[tier_col] == 1, target_col].to_numpy()
        clean_other = clean.loc[clean[tier_col] == 0, target_col].to_numpy()
        if len(clean_top) >= 10 and len(clean_other) >= 10:
            clean_stat, clean_p = stats.levene(clean_top, clean_other, center="median")
            subsample = {
                "levene_statistic": float(clean_stat),
                "levene_p_value": float(clean_p),
                "top_n": int(len(clean_top)),
                "other_n": int(len(clean_other)),
                "top_std": float(np.std(clean_top, ddof=1)),
                "other_std": float(np.std(clean_other, ddof=1)),
            }

    return {
        "hypothesis": "H5: underwriter tier and return variance",
        "h0": "Variance is equal for top-tier and other underwriters",
        "h1": "Top-tier underwriters produce lower variance (certification)",
        "test": "Levene's test, median-centred, with Fligner-Killeen as a rank-based check",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "fligner_statistic": float(fk_statistic),
        "fligner_p_value": float(fk_p),
        "top_spread": top_spread,
        "other_spread": other_spread,
        "top_std": top_spread["std"],
        "other_std": other_spread["std"],
        "top_n": int(len(top)),
        "other_n": int(len(other)),
        "ex_split_adjusted": subsample,
        "n": int(len(data)),
        "reject_h0": bool(p_value < alpha),
        "robust_reject_h0": bool(fk_p < alpha),
        "interpretation": (
            f"Top-tier sd = {top_spread['std']:.1%}, IQR = {top_spread['iqr']:.1%} "
            f"(n = {len(top)}) versus other sd = {other_spread['std']:.1%}, "
            f"IQR = {other_spread['iqr']:.1%} (n = {len(other)}). "
            f"Levene W = {statistic:.2f}, p = {p_value:.4g}; rank-based Fligner-Killeen "
            f"chi-square = {fk_statistic:.2f}, p = {fk_p:.4g}."
            + (
                f" Excluding split-adjusted prices: Levene p = "
                f"{subsample['levene_p_value']:.4g} (n = {subsample['top_n']} versus "
                f"{subsample['other_n']})."
                if subsample else ""
            )
        ),
    }


# ---------------------------------------------------------------------------
# H6
# ---------------------------------------------------------------------------

def test_h6_text_features(
    df: pd.DataFrame,
    target_col: str = TARGET,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Likelihood-ratio test: do text features add fit over deal and market features?

    Both models are fitted on the identical complete-case sample, so the
    comparison is not confounded by differing coverage.

    Args:
        df: Analysis dataset.
        target_col: First-day return column.
        alpha: Significance level.

    Returns:
        Results dict with the LR statistic, adjusted R-squared for both models,
        AIC/BIC deltas and the feature lists used.
    """
    base_features = [
        c for c in ["log_offer_price", "log_offer_size", "vix_at_pricing",
                    "nasdaq_30d_return", "hot_market_dummy", "is_spac",
                    "max_underwriter_rank"]
        if c in df.columns and df[c].notna().any()
    ]
    text_feature_names = [
        c for c in ["lm_negative_ratio", "lm_positive_ratio", "lm_uncertainty_ratio",
                    "lm_litigious_ratio", "gunning_fog", "prospectus_uniqueness",
                    "log_prospectus_words"]
        if c in df.columns and df[c].notna().any()
    ]

    frame = df[[target_col, "sector", *base_features, *text_feature_names]].dropna()
    if len(frame) < 60:
        return {"hypothesis": "H6: incremental value of text features",
                "test": "insufficient data", "p_value": float("nan"),
                "reject_h0": False, "n": len(frame)}

    sector_dummies = pd.get_dummies(frame["sector"], prefix="sector", drop_first=True, dtype=float)
    y = frame[target_col].astype(float)

    x_base = sm.add_constant(pd.concat([frame[base_features].astype(float), sector_dummies], axis=1))
    x_full = sm.add_constant(
        pd.concat([frame[base_features + text_feature_names].astype(float), sector_dummies], axis=1)
    )
    base_model = sm.OLS(y, x_base).fit()
    full_model = sm.OLS(y, x_full).fit()

    lr_statistic = 2 * (full_model.llf - base_model.llf)
    degrees = int(full_model.df_model - base_model.df_model)
    p_value = float(stats.chi2.sf(lr_statistic, df=degrees)) if degrees > 0 else float("nan")

    return {
        "hypothesis": "H6: incremental value of text features",
        "h0": "Text features add no explanatory power over deal, market and sector controls",
        "h1": "Text features improve fit",
        "test": f"Likelihood-ratio test, chi-square({degrees})",
        "statistic": float(lr_statistic),
        "p_value": p_value,
        "df_diff": degrees,
        "base_adj_r2": float(base_model.rsquared_adj),
        "full_adj_r2": float(full_model.rsquared_adj),
        "delta_aic": float(base_model.aic - full_model.aic),
        "delta_bic": float(base_model.bic - full_model.bic),
        "base_features": base_features,
        "text_features": text_feature_names,
        "n": int(len(frame)),
        "reject_h0": bool(p_value < alpha) if np.isfinite(p_value) else False,
        "interpretation": (
            f"LR = {lr_statistic:.2f} on {degrees} df, p = {p_value:.4g}, n = {len(frame)}. "
            f"Adjusted R-squared {base_model.rsquared_adj:.3f} without text features versus "
            f"{full_model.rsquared_adj:.3f} with them; delta BIC = "
            f"{base_model.bic - full_model.bic:+.1f}."
        ),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(df: pd.DataFrame, alpha: float = ALPHA) -> dict[str, Any]:
    """Run the six-test family and apply multiple-testing correction.

    Args:
        df: Analysis dataset.
        alpha: Family-wise / false-discovery rate.

    Returns:
        Dict with each test's results under ``H1`` .. ``H6``, an ``adjusted``
        DataFrame of raw and corrected p-values, and ``h1_robustness``.
    """
    results = {
        "H1": test_h1_litigious_tone(df, alpha=alpha),
        "H2": test_h2_underwriter_translation(df, alpha=alpha),
        "H3": test_h3_disclosure_concentration(df, alpha=alpha),
        "H4": test_h4_vix_variance(df, alpha=alpha),
        "H5": test_h5_underwriter_variance(df, alpha=alpha),
        "H6": test_h6_text_features(df, alpha=alpha),
    }
    results["adjusted"] = adjust_family(
        {key: value.get("p_value", float("nan")) for key, value in results.items()},
        alpha=alpha,
    )

    # H4 and H5 are variance tests on a target with a very heavy right tail.
    # Levene, even median-centred, is driven by a handful of extreme micro-cap
    # observations; Fligner-Killeen works on ranks. The two disagree in
    # opposite directions here - H4 is significant only under the robust test,
    # H5 only under the non-robust one - so both families are reported and
    # neither is presented as the answer.
    robust_p = {}
    for key, value in results.items():
        if not isinstance(value, dict):
            continue
        robust_p[key] = value.get("fligner_p_value", value.get("p_value", float("nan")))
    results["adjusted_robust"] = adjust_family(robust_p, alpha=alpha)

    results["h1_robustness"] = h1_robustness(df)
    return results


def report(result: dict[str, Any]) -> None:
    """Print one test result in a fixed format."""
    print("=" * 78)
    print(result.get("hypothesis", ""))
    print("-" * 78)
    for label, key in [("H0", "h0"), ("H1", "h1"), ("Test", "test")]:
        if key in result:
            print(f"  {label:<12} {result[key]}")
    for label, key in [("Statistic", "statistic"), ("p-value", "p_value"), ("n", "n")]:
        value = result.get(key)
        if value is not None:
            print(f"  {label:<12} {value:.4g}" if isinstance(value, (int, float)) else
                  f"  {label:<12} {value}")
    if "ci_95" in result:
        lo, hi = result["ci_95"]
        print(f"  {'95% CI':<12} [{lo:.4g}, {hi:.4g}]")
    print(f"  {'Decision':<12} {'Reject H0' if result.get('reject_h0') else 'Fail to reject H0'}")
    if "interpretation" in result:
        print(f"  {'Reading':<12} {result['interpretation']}")
    print()
