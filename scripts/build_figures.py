#!/usr/bin/env python3
"""Regenerate every figure in reports/figures/ from committed data.

Deterministic: fixed seeds, sorted iteration, no wall-clock in the output.
Style, colour, uncertainty and the provenance footer all come from
src/plotting/style.py so the set reads as one piece of work.

Usage::

    python scripts/build_figures.py
    python scripts/build_figures.py --out-dir /tmp/ci-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import style as st

SAMPLE_PATH = Path("data/processed/analysis_sample.parquet")
TABLES = Path("reports/tables")
DEFAULT_OUT = Path("reports/figures")

AS_OF = "2026-09-17"
SOURCE_PRICES = (
    "Source: Yahoo Finance first-day closes, split-corrected; offer prices from "
    "the stockanalysis.com IPO calendar."
)
SOURCE_TEXT = (
    "Sources: SEC EDGAR S-1/F-1 prospectuses; Loughran-McDonald master dictionary "
    "(1993-2025); Yahoo Finance first-day closes."
)
SOURCE_MODELS = "Source: expanding-window TimeSeriesSplit over the 709-IPO analysis sample."

MIN_CELL = 5  # cells thinner than this are shown grey, not coloured


# ---------------------------------------------------------------------------
# 01 Distribution of first-day returns
# ---------------------------------------------------------------------------


def fig_distribution(df: pd.DataFrame, out: Path) -> Path:
    values = df["underpricing"].dropna()
    median, mean = float(values.median()), float(values.mean())
    # The 99th percentile is above +500%, so a 1-99 trim still leaves the top
    # panel unreadable. The central 90% is where the mass is.
    low, high = values.quantile([0.05, 0.95])
    typical = values[(values >= low) & (values <= high)]
    top = df.loc[values.idxmax()]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))

    ax_top.hist(typical, bins=60, color=st.OKABE_ITO["blue"], alpha=0.85)
    ax_top.axvline(median, color=st.OKABE_ITO["vermillion"], linewidth=2)
    ax_top.axvline(mean, color=st.OKABE_ITO["orange"], linewidth=2, linestyle="--")
    ax_top.annotate(
        f"median {median:+.1%}",
        xy=(median, ax_top.get_ylim()[1] * 0.95),
        xytext=(8, 0),
        textcoords="offset points",
        color=st.OKABE_ITO["vermillion"],
        fontsize=st.SIZE_ANNOTATION,
        fontweight="semibold",
    )
    ax_top.annotate(
        f"mean {mean:+.1%}",
        xy=(mean, ax_top.get_ylim()[1] * 0.78),
        xytext=(8, 0),
        textcoords="offset points",
        color=st.OKABE_ITO["orange"],
        fontsize=st.SIZE_ANNOTATION,
        fontweight="semibold",
    )
    ax_top.set_title(f"Central 90% of the sample ({low:+.0%} to {high:+.0%})")
    ax_top.set_xlabel("First-day return, 5th to 95th percentile (%)")
    ax_top.set_ylabel("IPOs")
    ax_top.xaxis.set_major_formatter(st.percent_formatter(0))

    ax_bottom.hist(values, bins=80, color=st.OKABE_ITO["sky"], alpha=0.9)
    ax_bottom.set_yscale("symlog", linthresh=1)
    ax_bottom.set_title("Full range, log-scaled count (log axis, minor ticks shown)")
    ax_bottom.set_xlabel("First-day return (%)")
    ax_bottom.set_ylabel("IPOs (log)")
    ax_bottom.xaxis.set_major_formatter(st.percent_formatter(0))
    ax_bottom.minorticks_on()
    st.callout(
        ax_bottom,
        f"{top['ticker']} {top['underpricing']:+.0%}\n(Yahoo split history\nis unreliable here)",
        xy=(float(top["underpricing"]), 1.0),
        xytext=(float(top["underpricing"]) * 0.55, 60),
    )

    st.titles(
        fig,
        "First-day IPO returns are right-skewed with a median near zero",
        f"US listings 2019-2024 with a recoverable first-day close. n = {len(values)}. "
        f"{(values > 0).mean():.0%} closed above the offer price.",
    )
    return st.finish(
        fig, out / "01_first_day_return_distribution.png", source=SOURCE_PRICES, as_of=AS_OF
    )


# ---------------------------------------------------------------------------
# 02 Sector composition
# ---------------------------------------------------------------------------


def fig_sector_counts(df: pd.DataFrame, out: Path) -> Path:
    counts = df["sector"].value_counts().sort_values()
    unclassified = int(counts.get("Unclassified", 0))

    fig, ax = plt.subplots(figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))
    colours = [
        st.MUTED if name == "Unclassified" else st.OKABE_ITO["blue"] for name in counts.index
    ]
    bars = ax.barh(counts.index, counts.to_numpy(), color=colours, height=0.72)
    for bar, value in zip(bars, counts.to_numpy()):
        ax.annotate(
            f"{value}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=st.SIZE_ANNOTATION,
            color=st.INK,
        )

    ax.set_xlabel("IPOs in the analysis sample (count)")
    ax.set_xlim(0, counts.max() * 1.12)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    st.callout(
        ax,
        f"{unclassified} rows have no SEC-assigned SIC code.\n"
        "They are labelled Unclassified, not folded\ninto a bucket to hide the gap.",
        xy=(unclassified, list(counts.index).index("Unclassified")),
        xytext=(counts.max() * 0.42, 2.0),
    )

    st.titles(
        fig,
        "Healthcare and Technology dominate the sample; nothing is a fallback bucket",
        f"Sector from the registrant's SEC-assigned SIC code via a committed crosswalk. "
        f"n = {len(df)} IPOs, {len(df) - unclassified} classified ({1 - unclassified / len(df):.1%}).",
    )
    return st.finish(
        fig,
        out / "02_sector_composition.svg",
        source="Source: SEC company submissions API (sic field); "
        "data/external/sic_gics_crosswalk.csv.",
        as_of=AS_OF,
    )


# ---------------------------------------------------------------------------
# 03 Sector x year median return
# ---------------------------------------------------------------------------


def fig_sector_year(df: pd.DataFrame, out: Path) -> Path:
    pivot_median = df.pivot_table(
        index="sector", columns="ipo_year", values="underpricing", aggfunc="median"
    )
    pivot_count = df.pivot_table(
        index="sector", columns="ipo_year", values="underpricing", aggfunc="size"
    )
    order = pivot_count.sum(axis=1).sort_values(ascending=False).index
    pivot_median, pivot_count = pivot_median.loc[order], pivot_count.loc[order]

    thin = pivot_count.fillna(0) < MIN_CELL
    shown = pivot_median.mask(thin)

    fig, ax = plt.subplots(figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))
    limit = float(np.nanmax(np.abs(shown.to_numpy()))) if shown.notna().any().any() else 1.0
    mesh = ax.imshow(
        shown.to_numpy(),
        cmap=st.DIVERGING,
        aspect="auto",
        norm=st.diverging_norm(-limit, limit, 0.0),
    )

    # Thin and empty cells are drawn grey so they cannot be read as a colour.
    grey = np.where(thin.to_numpy() | pivot_median.isna().to_numpy(), 1.0, np.nan)
    ax.imshow(grey, cmap="Greys", vmin=0, vmax=1.6, aspect="auto")

    ax.set_xticks(range(len(pivot_median.columns)), [str(int(c)) for c in pivot_median.columns])
    ax.set_yticks(range(len(pivot_median.index)), list(pivot_median.index))
    ax.grid(visible=False)

    for i in range(pivot_median.shape[0]):
        for j in range(pivot_median.shape[1]):
            count = pivot_count.iat[i, j]
            if pd.isna(count) or count == 0:
                continue
            value = pivot_median.iat[i, j]
            if thin.iat[i, j]:
                label, colour = f"n={int(count)}", st.MUTED
            else:
                label = f"{value:+.0%}\nn={int(count)}"
                colour = "white" if abs(value) > limit * 0.55 else st.INK
            ax.text(
                j, i, label, ha="center", va="center", fontsize=st.SIZE_ANNOTATION - 1, color=colour
            )

    bar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.035)
    bar.set_label("Median first-day return (%)", fontsize=st.SIZE_LABEL)
    bar.ax.yaxis.set_major_formatter(st.percent_formatter(0))

    st.titles(
        fig,
        "The 2020-21 window is the only one where most sectors priced below market",
        f"Median first-day return by sector and listing year; cell label shows the count. "
        f"Cells with fewer than {MIN_CELL} IPOs are grey, not coloured. n = {len(df)}.",
    )
    return st.finish(
        fig, out / "03_sector_year_median_return.png", source=SOURCE_PRICES, as_of=AS_OF
    )


# ---------------------------------------------------------------------------
# 04 Sentiment deciles
# ---------------------------------------------------------------------------


def _decile_medians(frame: pd.DataFrame, column: str, target: str, q: int = 10) -> pd.DataFrame:
    frame = frame[[column, target]].dropna()
    if len(frame) < q * 4:
        return pd.DataFrame()
    frame = frame.assign(_bin=pd.qcut(frame[column], q=q, labels=False, duplicates="drop"))
    rows = []
    for index, group in frame.groupby("_bin", observed=True):
        point, low, high = st.bootstrap_ci(group[target].to_numpy())
        rows.append(
            {
                "bin": int(index) + 1,
                "n": len(group),
                "median": point,
                "ci_low": low,
                "ci_high": high,
                "x": float(group[column].median()),
            }
        )
    return pd.DataFrame(rows)


def fig_sentiment(df: pd.DataFrame, out: Path) -> Path:
    categories = [
        ("lm_negative_ratio", "Negative"),
        ("lm_positive_ratio", "Positive"),
        ("lm_uncertainty_ratio", "Uncertainty"),
        ("lm_litigious_ratio", "Litigious"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN), sharey=True)

    for ax, (column, label) in zip(axes.ravel(), categories):
        table = _decile_medians(df, column, "underpricing")
        if table.empty:
            ax.set_visible(False)
            continue
        ax.errorbar(
            table["bin"],
            table["median"],
            yerr=[table["median"] - table["ci_low"], table["ci_high"] - table["median"]],
            fmt="o-",
            color=st.OKABE_ITO["blue"],
            ecolor=st.OKABE_ITO["sky"],
            elinewidth=1.6,
            capsize=3,
            markersize=5,
        )
        ax.axhline(0, color=st.MUTED, linewidth=0.9)
        rho, low, high = st.spearman_ci(df[column], df["underpricing"])
        ax.set_title(
            f"{label}   rho = {rho:+.3f} [{low:+.2f}, {high:+.2f}]", fontsize=st.SIZE_SUBTITLE + 1
        )
        ax.set_xlabel(f"{label}-word ratio, decile (1 = lowest)")
        ax.yaxis.set_major_formatter(st.percent_formatter(0))
        ax.set_xticks(range(1, 11))
        n_per = int(table["n"].median())
        ax.annotate(
            f"n = {n_per} per decile",
            xy=(0.98, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=st.SIZE_FOOTER,
            color=st.MUTED,
        )

    axes[0, 0].set_ylabel("Median first-day return (%)")
    axes[1, 0].set_ylabel("Median first-day return (%)")

    n = int(df["lm_litigious_ratio"].notna().sum())
    st.titles(
        fig,
        "No Loughran-McDonald tone category orders the first-day return",
        f"Median first-day return by decile of each tone ratio, with 95% bootstrap "
        f"intervals (5,000 resamples). n = {n} IPOs with a recovered prospectus.",
    )
    return st.finish(fig, out / "04_tone_deciles.svg", source=SOURCE_TEXT, as_of=AS_OF)


# ---------------------------------------------------------------------------
# 05 VIX and dispersion
# ---------------------------------------------------------------------------


def fig_vix(df: pd.DataFrame, out: Path) -> Path:
    frame = df[["vix_at_pricing", "underpricing"]].dropna()
    frame = frame.assign(
        tercile=pd.qcut(frame["vix_at_pricing"], q=3, labels=["Low VIX", "Mid VIX", "High VIX"])
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))

    table = _decile_medians(frame, "vix_at_pricing", "underpricing")
    ax_left.errorbar(
        table["x"],
        table["median"],
        yerr=[table["median"] - table["ci_low"], table["ci_high"] - table["median"]],
        fmt="o-",
        color=st.OKABE_ITO["blue"],
        ecolor=st.OKABE_ITO["sky"],
        elinewidth=1.6,
        capsize=3,
        markersize=5,
    )
    ax_left.axhline(0, color=st.MUTED, linewidth=0.9)
    ax_left.set_title("The level does not move with VIX", fontsize=st.SIZE_SUBTITLE + 1)
    ax_left.set_xlabel("VIX close on the trading day before listing (index points)")
    ax_left.set_ylabel("Median first-day return (%)")
    ax_left.yaxis.set_major_formatter(st.percent_formatter(0))

    labels, spreads, counts = [], [], []
    for label, group in frame.groupby("tercile", observed=True):
        values = group["underpricing"].to_numpy()
        labels.append(str(label))
        spreads.append(float(np.percentile(values, 75) - np.percentile(values, 25)))
        counts.append(len(values))

    bars = ax_right.bar(labels, spreads, color=st.OKABE_ITO["orange"], width=0.55)
    for bar, spread, count in zip(bars, spreads, counts):
        ax_right.annotate(
            f"{spread:.1%}\nn = {count}",
            xy=(bar.get_x() + bar.get_width() / 2, spread),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=st.SIZE_ANNOTATION,
            color=st.INK,
        )
    ax_right.set_title("The dispersion does", fontsize=st.SIZE_SUBTITLE + 1)
    ax_right.set_xlabel("VIX tercile on the day before listing")
    ax_right.set_ylabel("Interquartile range of the first-day return (%)")
    ax_right.yaxis.set_major_formatter(st.percent_formatter(0))
    ax_right.set_ylim(0, max(spreads) * 1.25)

    st.titles(
        fig,
        "Volatility widens the spread of first-day returns without shifting their centre",
        f"n = {len(frame)}. Left: decile medians with 95% bootstrap intervals. "
        "Right: interquartile range by VIX tercile (Fligner-Killeen p < 1e-8).",
    )
    return st.finish(fig, out / "05_vix_and_dispersion.svg", source=SOURCE_PRICES, as_of=AS_OF)


# ---------------------------------------------------------------------------
# 06 Correlation heatmap
# ---------------------------------------------------------------------------


def fig_correlations(df: pd.DataFrame, out: Path) -> Path:
    columns = [
        "underpricing",
        "log_offer_price",
        "log_offer_size",
        "max_underwriter_rank",
        "vix_at_pricing",
        "nasdaq_30d_return",
        "nasdaq_30d_volatility",
        "hot_market_dummy",
        "lm_negative_ratio",
        "lm_positive_ratio",
        "lm_uncertainty_ratio",
        "lm_litigious_ratio",
        "risk_concentration_ratio",
        "gunning_fog",
        "prospectus_uniqueness",
        "log_prospectus_words",
    ]
    columns = [c for c in columns if c in df.columns]
    corr = df[columns].corr(method="pearson")

    # Order by hierarchical clustering so related blocks sit together, but keep
    # underpricing first: it is the only row anyone reads.
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    others = [c for c in columns if c != "underpricing"]
    distance = 1 - corr.loc[others, others].abs().to_numpy()
    np.fill_diagonal(distance, 0.0)
    ordered = [
        others[i]
        for i in leaves_list(linkage(squareform(distance, checks=False), method="average"))
    ]
    order = ["underpricing", *ordered]
    corr = corr.loc[order, order]

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    shown = corr.mask(mask)

    fig, ax = plt.subplots(figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))
    mesh = ax.imshow(
        shown.to_numpy(), cmap=st.DIVERGING, norm=st.diverging_norm(-1.0, 1.0, 0.0), aspect="auto"
    )
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=st.SIZE_FOOTER)
    ax.set_yticks(range(len(order)), order, fontsize=st.SIZE_FOOTER)
    ax.grid(visible=False)

    for i in range(len(order)):
        for j in range(i + 1):
            value = corr.iat[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=st.SIZE_FOOTER - 1,
                color="white" if abs(value) > 0.55 else st.INK,
            )

    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5),
            len(order),
            1,
            fill=False,
            edgecolor=st.OKABE_ITO["vermillion"],
            linewidth=2.2,
        )
    )
    strongest = corr.loc["underpricing"].drop("underpricing").abs().max()

    bar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.035)
    bar.set_label("Pearson correlation", fontsize=st.SIZE_LABEL)

    st.titles(
        fig,
        f"Nothing correlates with the first-day return above |{strongest:.2f}|",
        f"Pearson correlations on pairwise-complete observations, n up to {len(df)}. "
        "Upper triangle masked; features ordered by hierarchical clustering. "
        "The boxed row is the target.",
    )
    return st.finish(fig, out / "06_feature_correlations.svg", source=SOURCE_TEXT, as_of=AS_OF)


# ---------------------------------------------------------------------------
# 07 Litigious quintiles, the former headline
# ---------------------------------------------------------------------------


def fig_litigious(df: pd.DataFrame, out: Path) -> Path:
    quintiles = pd.read_csv(TABLES / "h1_quintiles.csv")
    robustness = pd.read_csv(TABLES / "h1_robustness.csv")
    subsample = df[df["split_adjusted"] == 0]

    clean_rows = []
    frame = subsample[["lm_litigious_ratio", "underpricing"]].dropna()
    if len(frame) >= 50:
        frame = frame.assign(
            _bin=pd.qcut(
                frame["lm_litigious_ratio"],
                q=5,
                labels=[f"Q{i}" for i in range(1, 6)],
                duplicates="drop",
            )
        )
        for label, group in frame.groupby("_bin", observed=True):
            point, low, high = st.bootstrap_ci(group["underpricing"].to_numpy())
            clean_rows.append(
                {
                    "quintile": str(label),
                    "n": len(group),
                    "median": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    clean = pd.DataFrame(clean_rows)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    positions = np.arange(len(quintiles))
    width = 0.38
    ax_left.bar(
        positions - width / 2,
        quintiles["median"],
        width=width,
        color=st.OKABE_ITO["blue"],
        label="All filings",
    )
    ax_left.errorbar(
        positions - width / 2,
        quintiles["median"],
        yerr=[
            quintiles["median"] - quintiles["ci_low"],
            quintiles["ci_high"] - quintiles["median"],
        ],
        fmt="none",
        ecolor=st.INK,
        elinewidth=1.4,
        capsize=4,
    )
    if not clean.empty:
        ax_left.bar(
            positions + width / 2,
            clean["median"],
            width=width,
            color=st.OKABE_ITO["orange"],
            hatch="//",
            label="Excluding split-corrected prices",
        )
        ax_left.errorbar(
            positions + width / 2,
            clean["median"],
            yerr=[clean["median"] - clean["ci_low"], clean["ci_high"] - clean["median"]],
            fmt="none",
            ecolor=st.INK,
            elinewidth=1.4,
            capsize=4,
        )

    headroom = float(
        max(quintiles["ci_high"].max(), clean["ci_high"].max() if not clean.empty else 0)
    )
    for x, row in zip(positions, quintiles.itertuples()):
        ax_left.annotate(
            f"n = {row.n}",
            xy=(x, headroom),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            fontsize=st.SIZE_FOOTER,
            color=st.MUTED,
        )
    ax_left.set_ylim(0, headroom * 1.30)

    ax_left.set_xticks(positions, quintiles["quintile"])
    ax_left.set_xlabel("Litigious-word ratio, quintile (Q1 = least legal language)")
    ax_left.set_ylabel("Median first-day return (%)")
    ax_left.yaxis.set_major_formatter(st.percent_formatter(0))
    ax_left.axhline(0, color=st.MUTED, linewidth=0.9)
    ax_left.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.13))
    ax_left.set_title(
        "Medians by quintile, with 95% bootstrap intervals", fontsize=st.SIZE_SUBTITLE + 1
    )

    specs = robustness[robustness["kind"] == "spearman"].dropna(subset=["estimate"])
    specs = specs.iloc[::-1]
    y = np.arange(len(specs))
    ax_right.errorbar(
        specs["estimate"],
        y,
        xerr=[specs["estimate"] - specs["ci_low"], specs["ci_high"] - specs["estimate"]],
        fmt="o",
        color=st.OKABE_ITO["blue"],
        ecolor=st.MUTED,
        elinewidth=1.4,
        capsize=3,
        markersize=5,
    )
    ax_right.axvline(0, color=st.OKABE_ITO["vermillion"], linewidth=1.4)
    ax_right.set_yticks(
        y,
        [f"{s}  (n={n})" for s, n in zip(specs["specification"], specs["n"])],
        fontsize=st.SIZE_FOOTER,
    )
    ax_right.set_xlabel("Spearman rho with the first-day return")
    ax_right.grid(axis="x")
    ax_right.grid(axis="y", visible=False)
    ax_right.set_title("Every subsample interval spans zero", fontsize=st.SIZE_SUBTITLE + 1)

    st.titles(
        fig,
        "The litigious-tone effect does not survive a correct first-day return",
        f"Spearman rho = {robustness['estimate'].iloc[0]:+.3f} "
        f"(95% CI [{robustness['ci_low'].iloc[0]:+.3f}, {robustness['ci_high'].iloc[0]:+.3f}], "
        f"p = {robustness['p_value'].iloc[0]:.2f}) on n = {int(robustness['n'].iloc[0])} "
        "IPOs with a recovered prospectus.",
    )
    return st.finish(fig, out / "07_litigious_tone.svg", source=SOURCE_TEXT, as_of=AS_OF)


# ---------------------------------------------------------------------------
# 08 Disclosure concentration
# ---------------------------------------------------------------------------


def fig_disclosure(df: pd.DataFrame, out: Path) -> Path:
    terciles = pd.read_csv(TABLES / "h3_terciles.csv")
    frame = df[["risk_concentration_ratio", "underpricing"]].dropna()
    frame = frame[np.isfinite(frame["risk_concentration_ratio"])]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))

    positions = np.arange(len(terciles))
    ax_left.bar(positions, terciles["median"], width=0.55, color=st.OKABE_ITO["green"])
    ax_left.errorbar(
        positions,
        terciles["median"],
        yerr=[terciles["median"] - terciles["ci_low"], terciles["ci_high"] - terciles["median"]],
        fmt="none",
        ecolor=st.INK,
        elinewidth=1.4,
        capsize=4,
    )
    tercile_headroom = float(terciles["ci_high"].max())
    for x, row in zip(positions, terciles.itertuples()):
        ax_left.annotate(
            f"n = {row.n}",
            xy=(x, tercile_headroom),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=st.SIZE_FOOTER,
            color=st.MUTED,
        )
    ax_left.set_ylim(0, tercile_headroom * 1.28)
    ax_left.set_xticks(positions, [t.replace(" ", "\n", 1) for t in terciles["tercile"]])
    ax_left.set_ylabel("Median first-day return (%)")
    ax_left.yaxis.set_major_formatter(st.percent_formatter(0))
    ax_left.axhline(0, color=st.MUTED, linewidth=0.9)
    ax_left.set_title("Medians overlap across terciles", fontsize=st.SIZE_SUBTITLE + 1)

    ax_right.scatter(
        frame["risk_concentration_ratio"],
        frame["underpricing"],
        s=14,
        alpha=0.45,
        color=st.OKABE_ITO["blue"],
        edgecolors="none",
    )
    ax_right.set_yscale("symlog", linthresh=1)
    ax_right.set_xlabel("Risk-factor share of prospectus negative tone (ratio)")
    ax_right.set_ylabel("First-day return (%, log scale)")
    ax_right.yaxis.set_major_formatter(st.percent_formatter(0))
    ax_right.axhline(0, color=st.MUTED, linewidth=0.9)
    ax_right.minorticks_on()
    ax_right.set_title("and the scatter shows no slope", fontsize=st.SIZE_SUBTITLE + 1)

    rho, low, high = st.spearman_ci(frame["risk_concentration_ratio"], frame["underpricing"])
    st.titles(
        fig,
        "Where negative tone sits in the prospectus does not predict the first-day return",
        f"Spearman rho = {rho:+.3f} (95% CI [{low:+.3f}, {high:+.3f}]) on n = {len(frame)}. "
        "Bars show 95% bootstrap intervals on each tercile median.",
    )
    return st.finish(fig, out / "08_disclosure_concentration.png", source=SOURCE_TEXT, as_of=AS_OF)


# ---------------------------------------------------------------------------
# 09 Model performance
# ---------------------------------------------------------------------------


def fig_models(out: Path) -> Path:
    per_fold = pd.read_csv(TABLES / "model_cv_per_fold.csv")
    summary = pd.read_csv(TABLES / "model_cv_summary.csv")

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN),
        gridspec_kw={"width_ratios": [1, 1.25]},
    )

    order = summary.sort_values("r2_mean")["model"].tolist()
    y = np.arange(len(order))
    means = [summary.loc[summary["model"] == m, "r2_mean"].iloc[0] for m in order]
    colours = [
        st.OKABE_ITO["green"] if m.startswith("Baseline") else st.OKABE_ITO["blue"] for m in order
    ]
    ax_left.barh(y, means, color=colours, height=0.6)
    for yi, model in zip(y, order):
        low = per_fold.loc[per_fold["model"] == model, "r2"].min()
        high = per_fold.loc[per_fold["model"] == model, "r2"].max()
        ax_left.plot([low, high], [yi, yi], color=st.INK, linewidth=1.4)
        ax_left.plot([low, high], [yi, yi], "|", color=st.INK, markersize=7)
    ax_left.axvline(0, color=st.OKABE_ITO["vermillion"], linewidth=1.6)
    ax_left.set_yticks(y, order, fontsize=st.SIZE_ANNOTATION)
    ax_left.set_xlabel("Out-of-sample R-squared (mean across folds; bar spans fold min to max)")
    ax_left.grid(axis="x")
    ax_left.grid(axis="y", visible=False)
    ax_left.set_title(
        "Every model is worse than predicting the training median", fontsize=st.SIZE_SUBTITLE + 1
    )
    ax_left.annotate(
        "R² = 0 means\n'as good as the\ntraining mean'",
        xy=(0, len(order) - 0.5),
        xytext=(4, -6),
        textcoords="offset points",
        fontsize=st.SIZE_FOOTER,
        color=st.OKABE_ITO["vermillion"],
    )

    models = [m for m in summary["model"] if not m.startswith("Baseline")]
    for colour, model in zip(st.QUALITATIVE, sorted(models)):
        rows = per_fold[per_fold["model"] == model].sort_values("fold")
        ax_right.plot(rows["fold"], rows["r2"], "o-", color=colour, markersize=5)
        st.direct_label(ax_right, rows["fold"].iloc[-1], rows["r2"].iloc[-1], f"  {model}", colour)
    baseline = per_fold[per_fold["model"] == "Baseline: train median"].sort_values("fold")
    ax_right.plot(baseline["fold"], baseline["r2"], "--", color=st.MUTED, linewidth=1.6)
    st.direct_label(
        ax_right,
        baseline["fold"].iloc[0],
        baseline["r2"].iloc[0],
        "median baseline  ",
        st.MUTED,
        ha="right",
    )

    ax_right.axhline(0, color=st.OKABE_ITO["vermillion"], linewidth=1.2)
    ax_right.set_xticks(sorted(per_fold["fold"].unique()))
    ax_right.set_xlabel("Expanding-window fold (1 = earliest listings)")
    ax_right.set_ylabel("Out-of-sample R-squared")
    ax_right.set_xlim(0.6, per_fold["fold"].max() + 1.1)
    ax_right.set_title("Fold-to-fold variation dwarfs the average", fontsize=st.SIZE_SUBTITLE + 1)

    n_train = int(per_fold["n_train"].max())
    n_test = int(per_fold["n_test"].max())
    st.titles(
        fig,
        "First-day IPO returns are not forecastable from pre-listing information",
        f"Five-fold expanding-window TimeSeriesSplit; final fold trains on {n_train} "
        f"listings and tests on {n_test}. Hyperparameters chosen by an inner split "
        "on the training fold only.",
    )
    return st.finish(fig, out / "09_model_performance.svg", source=SOURCE_MODELS, as_of=AS_OF)


# ---------------------------------------------------------------------------
# 10 Holdout predictions and residuals
# ---------------------------------------------------------------------------


def fig_holdout(df: pd.DataFrame, out: Path) -> Path:
    from src.models import holdout_evaluation

    metrics, predictions, test = holdout_evaluation(df)
    actual = test["underpricing"].to_numpy(dtype=float)
    predicted = predictions["LightGBM"]
    residuals = actual - predicted
    r2 = float(metrics.loc[metrics["model"] == "LightGBM", "r2"].iloc[0])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))

    ax_left.scatter(
        predicted, actual, s=22, alpha=0.6, color=st.OKABE_ITO["blue"], edgecolors="none"
    )
    limit = float(max(np.abs(actual).max(), np.abs(predicted).max())) * 1.08
    ax_left.plot(
        [-limit, limit],
        [-limit, limit],
        color=st.OKABE_ITO["vermillion"],
        linewidth=1.5,
        linestyle="--",
    )
    ax_left.set_xlim(-limit, limit)
    ax_left.set_ylim(-limit, limit)
    ax_left.set_xlabel("Predicted first-day return (%)")
    ax_left.set_ylabel("Actual first-day return (%)")
    ax_left.xaxis.set_major_formatter(st.percent_formatter(0))
    ax_left.yaxis.set_major_formatter(st.percent_formatter(0))
    ax_left.grid(axis="both")
    ax_left.set_title(
        f"Predictions cluster near the mean (R² = {r2:.2f})", fontsize=st.SIZE_SUBTITLE + 1
    )
    ax_left.legend(
        handles=[
            Line2D(
                [], [], color=st.OKABE_ITO["vermillion"], linestyle="--", label="perfect prediction"
            )
        ],
        loc="upper left",
    )

    ax_right.scatter(
        pd.to_datetime(test["ipo_date"]),
        residuals,
        s=22,
        alpha=0.6,
        color=st.OKABE_ITO["orange"],
        edgecolors="none",
    )
    ax_right.axhline(0, color=st.OKABE_ITO["vermillion"], linewidth=1.4)
    ax_right.set_xlabel("Listing date")
    ax_right.set_ylabel("Residual, actual minus predicted (%)")
    ax_right.yaxis.set_major_formatter(st.percent_formatter(0))
    ax_right.set_title(
        "Residuals are as large as the returns themselves", fontsize=st.SIZE_SUBTITLE + 1
    )
    fig.autofmt_xdate(rotation=30)

    st.titles(
        fig,
        "On unseen 2024 listings the model adds nothing over a constant",
        f"Trained on {int(metrics['n_train'].iloc[0])} listings before 2024-01-01, "
        f"scored on {int(metrics['n_test'].iloc[0])} listings after it. "
        "LightGBM; the training-median baseline scores R² = "
        f"{float(metrics.loc[metrics['model'] == 'Baseline: train median', 'r2'].iloc[0]):.3f}.",
    )
    return st.finish(fig, out / "10_holdout_predictions.svg", source=SOURCE_MODELS, as_of=AS_OF)


# ---------------------------------------------------------------------------
# 11 SHAP
# ---------------------------------------------------------------------------


def fig_shap(out: Path) -> Path:
    importance = pd.read_csv(TABLES / "shap_importance.csv").head(12).iloc[::-1]
    labels = [
        f.replace("numeric__", "").replace("categorical__", "") for f in importance["feature"]
    ]

    fig, ax = plt.subplots(figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))
    y = np.arange(len(importance))
    bars = ax.barh(y, importance["mean_abs_shap"], color=st.OKABE_ITO["purple"], height=0.66)
    for bar, value in zip(bars, importance["mean_abs_shap"]):
        ax.annotate(
            f"{value:.3f}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=st.SIZE_ANNOTATION,
            color=st.INK,
        )
    ax.set_yticks(y, labels, fontsize=st.SIZE_ANNOTATION)
    ax.set_xlabel("Mean |SHAP value| (percentage points of predicted first-day return)")
    ax.set_xlim(0, importance["mean_abs_shap"].max() * 1.16)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    st.callout(
        ax,
        "This ranks what the model leaned on.\n"
        "That model has negative out-of-sample R².\n"
        "It is not evidence about what drives underpricing.",
        xy=(importance["mean_abs_shap"].iloc[-1], len(importance) - 1),
        xytext=(importance["mean_abs_shap"].max() * 0.42, len(importance) - 5.0),
    )

    st.titles(
        fig,
        "The model keys on market regime and deal size, not on prospectus text",
        "Mean absolute SHAP value, LightGBM fitted on all 709 listings for inspection only. "
        "Market-regime features lead; every tone ratio ranks below them.",
    )
    return st.finish(fig, out / "11_shap_importance.svg", source=SOURCE_MODELS, as_of=AS_OF)


# ---------------------------------------------------------------------------
# 12 Sample funnel
# ---------------------------------------------------------------------------


def fig_funnel(out: Path) -> Path:
    full = pd.read_csv(TABLES / "sample_funnel.csv")
    # Steps that remove nothing add a bar without adding information.
    funnel = full[(full["dropped"] > 0) | (full.index == 0)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(st.FIG_WIDTH_IN, st.FIG_HEIGHT_IN))
    y = np.arange(len(funnel))[::-1]
    bars = ax.barh(y, funnel["n"], color=st.OKABE_ITO["blue"], height=0.62)
    for bar, row in zip(bars, funnel.itertuples()):
        ax.annotate(
            f"{row.n:,}   ({row.pct_of_universe:.0f}% of the calendar)",
            xy=(row.n, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=st.SIZE_ANNOTATION,
            color=st.INK,
        )
        if row.dropped:
            ax.annotate(
                f"-{row.dropped:,}",
                xy=(row.n, bar.get_y() + bar.get_height()),
                xytext=(-6, 6),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=st.SIZE_FOOTER,
                color=st.OKABE_ITO["vermillion"],
            )

    ax.set_yticks(y, funnel["step"], fontsize=st.SIZE_ANNOTATION)
    ax.set_xlabel("IPOs remaining (count)")
    ax.set_xlim(0, funnel["n"].max() * 1.42)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    biggest = int(funnel["dropped"].idxmax())
    st.callout(
        ax,
        f"Largest single loss: -{funnel['dropped'].iloc[biggest]:,}.\n"
        "Yahoo purges delisted tickers, so this is\nsurvivorship, not a modelling choice.",
        xy=(funnel["n"].iloc[biggest], y[biggest]),
        xytext=(funnel["n"].max() * 0.60, y[biggest] - 1.6),
    )

    st.titles(
        fig,
        "Two of every five listed IPOs reach the analysis sample",
        f"From {full['n'].iloc[0]:,} calendar entries to "
        f"{int(full.loc[full['step'].str.contains('target'), 'n'].iloc[0]):,} with a first-day "
        f"return and {full['n'].iloc[-1]:,} with a prospectus. "
        "Steps that remove nothing are omitted; red labels show what each requirement costs.",
    )
    return st.finish(
        fig,
        out / "12_sample_funnel.svg",
        source="Source: reports/tables/sample_funnel.csv.",
        as_of=AS_OF,
    )


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not SAMPLE_PATH.exists():
        print(f"{SAMPLE_PATH} not found. Run scripts/build_dataset.py first.", file=sys.stderr)
        return 1

    st.apply_style()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SAMPLE_PATH)
    df["ipo_date"] = pd.to_datetime(df["ipo_date"])

    written = [
        fig_distribution(df, out),
        fig_sector_counts(df, out),
        fig_sector_year(df, out),
        fig_sentiment(df, out),
        fig_vix(df, out),
        fig_correlations(df, out),
        fig_litigious(df, out),
        fig_disclosure(df, out),
        fig_models(out),
        fig_holdout(df, out),
        fig_shap(out),
        fig_funnel(out),
    ]
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"\n{len(written)} figures written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
