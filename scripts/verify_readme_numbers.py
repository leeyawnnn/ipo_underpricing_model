#!/usr/bin/env python3
"""Check that every headline number in the prose still matches its table.

The failure mode this guards against is the one that produced the previous
version of this repository: a number written into prose once, then left behind
when the pipeline that produced it changed. Each entry below names a literal
string in a Markdown file and the table cell it must agree with. If a result
moves and the prose does not, CI fails.

Both README.md and FIXES.md are covered. FIXES.md quotes the same results in
order to say which of them changed, so it drifts for exactly the same reason
and is checked the same way.

Usage::

    python scripts/verify_readme_numbers.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = "README.md"
FIXES = "FIXES.md"
TABLES = ROOT / "reports" / "tables"


@dataclass(frozen=True)
class Claim:
    """One number asserted in the README, and where it comes from."""

    literal: str  # the exact text that must appear in the document
    table: str  # CSV under reports/tables/, without the extension
    key_column: str  # column to look the row up by
    key: str | int  # row key
    value_column: str  # column holding the number
    tolerance: float = 5e-3
    document: str = README  # Markdown file the literal must appear in


CLAIMS: list[Claim] = [
    # Out-of-sample prediction
    Claim("−0.026", "model_cv_summary", "model", "Baseline: train median", "r2_mean"),
    Claim("−0.217", "model_cv_summary", "model", "Baseline: train mean", "r2_mean"),
    Claim("−0.557", "model_cv_summary", "model", "Ridge", "r2_mean"),
    Claim("−2.154", "model_cv_summary", "model", "LightGBM", "r2_mean"),
    Claim("−1.918", "model_cv_summary", "model", "RandomForest", "r2_mean"),
    Claim("−6.931", "model_cv_summary", "model", "LightGBM", "r2_min"),
    Claim("−5.277", "model_cv_summary", "model", "RandomForest", "r2_min"),
    Claim("−7.947", "model_holdout_2024", "model", "LightGBM", "r2"),
    Claim("−8.675", "model_holdout_2024", "model", "RandomForest", "r2"),
    Claim("−1.859", "model_holdout_2024", "model", "Ridge", "r2"),
    Claim("−0.918", "model_holdout_2024", "model", "Baseline: train mean", "r2"),
    # Hypothesis family
    Claim("ρ = −0.112", "hypothesis_summary", "test", "H1", "statistic"),
    Claim("ρ = +0.060", "hypothesis_summary", "test", "H3", "statistic"),
    Claim("Levene W = 2.39", "hypothesis_summary", "test", "H4", "statistic"),
    Claim("Levene W = 7.34", "hypothesis_summary", "test", "H5", "statistic"),
    Claim("= 7.14 ", "hypothesis_summary", "test", "H6", "statistic"),
    Claim("[−0.188, −0.037]", "hypothesis_summary", "test", "H1", "ci_low"),
    Claim("[−0.034, +0.152]", "hypothesis_summary", "test", "H3", "ci_low"),
    Claim("p = 0.004", "hypothesis_summary", "test", "H1", "p_raw"),
    Claim(
        "Bonferroni p = 0.024",
        "lm_category_correlations",
        "category",
        "lm_positive_ratio",
        "p_bonferroni",
    ),
    Claim("ρ = +0.113", "lm_category_correlations", "category", "lm_positive_ratio", "rho"),
    # Descriptive
    Claim("+7.9%", "descriptive_statistics", "group", "All", "median"),
    Claim("+40.3%", "descriptive_statistics", "group", "All", "mean"),
    Claim("62.1%", "descriptive_statistics", "group", "All", "share_positive"),
    Claim("+27.7% (2020)", "descriptive_statistics", "group", "Listing year 2020", "median"),
    Claim("+11.9% (2019)", "descriptive_statistics", "group", "Listing year 2019", "median"),
    # Sample funnel
    Claim("1,773", "sample_funnel", "step", "IPOs in the 2019-2024 calendar", "n"),
    Claim("| 709 |", "sample_funnel", "step", "with a recoverable first-day close", "n"),
    Claim("1,020", "sample_funnel", "step", "with a recoverable first-day close", "dropped"),
    Claim("| 666 |", "sample_funnel", "step", "and a recovered S-1/F-1 prospectus", "n"),
    Claim("| 408 |", "sample_funnel", "step", "and a Risk Factors section located", "n"),
    # Selection
    Claim("+2.55", "selection_comparison", "observable", "Offer price (USD)", "difference"),
    Claim("−2.13", "selection_comparison", "observable", "VIX at pricing", "difference"),
    # H1 robustness
    Claim(
        "−0.140", "h1_robustness", "specification", "Excluding split-adjusted prices", "estimate"
    ),
    Claim("−0.155", "h1_robustness", "specification", "Listing year 2021", "estimate"),
    Claim("**+0.181**", "h1_robustness", "specification", "Listing year 2022", "estimate"),
    Claim(
        "−0.096",
        "h1_robustness",
        "specification",
        "Partial rank: controlling for log document length",
        "estimate",
    ),
    Claim(
        "−0.015",
        "h1_robustness",
        "specification",
        "Partial rank: + deal and market controls",
        "estimate",
    ),
    Claim(
        "−0.117",
        "h1_robustness",
        "specification",
        "Baseline on the multivariate subsample",
        "estimate",
    ),
    Claim(
        "−8.98",
        "h1_robustness",
        "specification",
        "OLS: + sector and year fixed effects",
        "estimate",
        tolerance=0.01,
    ),
    # Legacy classifier audit
    Claim("| **518** |", "legacy_sector_classifier_layers", "layer", "7_fallback", "n"),
    Claim(
        "| **29.2%** |",
        "legacy_sector_classifier_layers",
        "layer",
        "7_fallback",
        "pct",
        tolerance=0.05,
    ),
]

# FIXES.md quotes the corrected results in order to say which claims moved.
# Those are the same numbers, so they go stale the same way; the EDGAR fixes
# left the first draft of that file contradicting the README it is linked from.
CLAIMS += [
    Claim(
        "ρ = −0.112, p = 0.0039, n = 666",
        "hypothesis_summary",
        "test",
        "H1",
        "statistic",
        document=FIXES,
    ),
    Claim(
        "Bonferroni p = 0.023", "hypothesis_summary", "test", "H1", "p_bonferroni", document=FIXES
    ),
    Claim(
        "partial rank correlation to −0.015",
        "h1_robustness",
        "specification",
        "Partial rank: + deal and market controls",
        "estimate",
        document=FIXES,
    ),
    Claim(
        "bivariate estimate is −0.117",
        "h1_robustness",
        "specification",
        "Baseline on the multivariate subsample",
        "estimate",
        document=FIXES,
    ),
    Claim(
        "ρ = +0.113, Bonferroni p = 0.024",
        "lm_category_correlations",
        "category",
        "lm_positive_ratio",
        "rho",
        document=FIXES,
    ),
    Claim(
        "Bonferroni p = 0.027",
        "lm_category_correlations",
        "category",
        "lm_litigious_ratio",
        "p_bonferroni",
        document=FIXES,
    ),
    Claim(
        "ρ = +0.060, p = 0.22, n = 408",
        "hypothesis_summary",
        "test",
        "H3",
        "statistic",
        document=FIXES,
    ),
    Claim("β = +7.49, p = 0.83", "hypothesis_summary", "test", "H2", "statistic", document=FIXES),
    Claim("W = 7.34, p = 0.0070", "hypothesis_summary", "test", "H5", "statistic", document=FIXES),
    Claim(
        "= 7.14, p = 0.41, n = 284", "hypothesis_summary", "test", "H6", "statistic", document=FIXES
    ),
    Claim("W = 2.39, p = 0.093", "hypothesis_summary", "test", "H4", "statistic", document=FIXES),
    Claim(
        "| **666** |",
        "sample_funnel",
        "step",
        "and a recovered S-1/F-1 prospectus",
        "n",
        document=FIXES,
    ),
    Claim(
        "| **408** |",
        "sample_funnel",
        "step",
        "and a Risk Factors section located",
        "n",
        document=FIXES,
    ),
    Claim(
        "**−0.112, p = 0.0039**", "hypothesis_summary", "test", "H1", "statistic", document=FIXES
    ),
    Claim("+0.060, p = 0.22 |", "hypothesis_summary", "test", "H3", "statistic", document=FIXES),
    Claim("| −0.557 |", "model_cv_summary", "model", "Ridge", "r2_mean", document=FIXES),
    # Quintile medians, quoted in the retraction of the "monotone step" claim.
    Claim("+17.3%", "h1_quintiles", "quintile", "Q1", "median", document=FIXES),
    Claim("+20.0%", "h1_quintiles", "quintile", "Q2", "median", document=FIXES),
    Claim("+7.6%", "h1_quintiles", "quintile", "Q3", "median", document=FIXES),
]


# Numbers quoted in prose that are not single table cells. Each names the
# command that regenerates the artifact it came from, so nothing is orphaned.
DERIVED_CLAIMS = [
    (README, "79.0%", "underwriter match rate", "python scripts/build_dataset.py"),
    (README, "76.9%", "Ritter rank match rate", "python scripts/build_dataset.py"),
    (README, "3.4%", "unclassified sector share", "python scripts/build_dataset.py"),
    (
        README,
        "96.3%",
        "share of Industrials reached by fallback",
        "reports/tables/legacy_sector_classifier_layers.csv",
    ),
    (FIXES, "79.0%", "underwriter match rate", "python scripts/build_dataset.py"),
    (FIXES, "76.9%", "Ritter rank match rate", "python scripts/build_dataset.py"),
]


def _to_float(text: str) -> float | None:
    """Parse the first signed number out of *text*, accepting a Unicode minus."""
    cleaned = text.replace("−", "-").replace("−", "-").replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def main() -> int:
    documents = {name: (ROOT / name).read_text(encoding="utf-8") for name in (README, FIXES)}
    failures: list[str] = []
    checked = 0

    for claim in CLAIMS:
        path = TABLES / f"{claim.table}.csv"
        if not path.exists():
            failures.append(f"{claim.table}.csv is missing")
            continue

        frame = pd.read_csv(path)
        rows = frame[frame[claim.key_column].astype(str) == str(claim.key)]
        if rows.empty:
            failures.append(f"{claim.table}.csv has no row {claim.key!r}")
            continue
        actual = float(rows[claim.value_column].iloc[0])

        if claim.literal not in documents[claim.document]:
            failures.append(
                f"{claim.document} no longer contains {claim.literal!r} "
                f"({claim.table}.{claim.value_column} is now {actual:.4f})"
            )
            continue

        stated = _to_float(claim.literal)
        if stated is None:
            failures.append(f"could not parse a number out of {claim.literal!r}")
            continue

        # Percentages in the prose correspond to fractions in the tables.
        candidates = [stated, stated / 100.0]
        if not any(abs(actual - c) <= claim.tolerance for c in candidates):
            failures.append(
                f"{claim.literal!r} in {claim.document} vs {actual:.4f} in "
                f"{claim.table}.csv[{claim.key}][{claim.value_column}]"
            )
            continue
        checked += 1

    for document, literal, description, source in DERIVED_CLAIMS:
        if literal not in documents[document]:
            failures.append(f"{document} no longer contains {literal!r} ({description}, {source})")
        else:
            checked += 1

    if failures:
        print("Prose number check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nRegenerate with: python scripts/run_analysis.py, then update the prose.",
            file=sys.stderr,
        )
        return 1

    print(f"All {checked} numbers in README.md and FIXES.md agree with reports/tables/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
