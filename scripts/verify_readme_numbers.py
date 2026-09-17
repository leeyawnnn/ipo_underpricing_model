#!/usr/bin/env python3
"""Check that every headline number in the README still matches its table.

The failure mode this guards against is the one that produced the previous
version of this repository: a number written into prose once, then left behind
when the pipeline that produced it changed. Each entry below names a literal
string in README.md and the table cell it must agree with. If a result moves
and the prose does not, CI fails.

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
README = ROOT / "README.md"
TABLES = ROOT / "reports" / "tables"


@dataclass(frozen=True)
class Claim:
    """One number asserted in the README, and where it comes from."""

    literal: str  # the exact text that must appear in README.md
    table: str  # CSV under reports/tables/, without the extension
    key_column: str  # column to look the row up by
    key: str | int  # row key
    value_column: str  # column holding the number
    tolerance: float = 5e-3


CLAIMS: list[Claim] = [
    # Out-of-sample prediction
    Claim("−0.026", "model_cv_summary", "model", "Baseline: train median", "r2_mean"),
    Claim("−0.217", "model_cv_summary", "model", "Baseline: train mean", "r2_mean"),
    Claim("−0.426", "model_cv_summary", "model", "Ridge", "r2_mean"),
    Claim("−1.571", "model_cv_summary", "model", "LightGBM", "r2_mean"),
    Claim("−1.579", "model_cv_summary", "model", "RandomForest", "r2_mean"),
    Claim("−5.442", "model_cv_summary", "model", "LightGBM", "r2_min"),
    Claim("−5.327", "model_cv_summary", "model", "RandomForest", "r2_min"),
    Claim("−5.801", "model_holdout_2024", "model", "LightGBM", "r2"),
    Claim("−6.701", "model_holdout_2024", "model", "RandomForest", "r2"),
    Claim("−1.521", "model_holdout_2024", "model", "Ridge", "r2"),
    Claim("−0.918", "model_holdout_2024", "model", "Baseline: train mean", "r2"),
    # Hypothesis family
    Claim("ρ = −0.047", "hypothesis_summary", "test", "H1", "statistic"),
    Claim("ρ = −0.012", "hypothesis_summary", "test", "H3", "statistic"),
    Claim("Levene W = 2.39", "hypothesis_summary", "test", "H4", "statistic"),
    Claim("Levene W = 9.16", "hypothesis_summary", "test", "H5", "statistic"),
    Claim("= 7.30 ", "hypothesis_summary", "test", "H6", "statistic"),
    Claim("[−0.136, +0.040]", "hypothesis_summary", "test", "H1", "ci_low"),
    Claim("[−0.106, +0.086]", "hypothesis_summary", "test", "H3", "ci_low"),
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
    Claim("| 474 |", "sample_funnel", "step", "and a recovered S-1/F-1 prospectus", "n"),
    # Selection
    Claim("+3.51", "selection_comparison", "observable", "Offer price (USD)", "difference"),
    Claim("−1.34", "selection_comparison", "observable", "VIX at pricing", "difference"),
    # H1 robustness
    Claim(
        "−0.067", "h1_robustness", "specification", "Excluding split-adjusted prices", "estimate"
    ),
    Claim("−0.163", "h1_robustness", "specification", "Listing year 2021", "estimate"),
    Claim(
        "+13.54",
        "h1_robustness",
        "specification",
        "OLS: + sector and year fixed effects",
        "estimate",
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

# Numbers quoted in prose that are not single table cells. Each names the
# command that regenerates the artifact it came from, so nothing is orphaned.
DERIVED_CLAIMS = [
    ("87.6%", "underwriter match rate", "python scripts/build_dataset.py"),
    ("85.4%", "Ritter rank match rate", "python scripts/build_dataset.py"),
    ("3.4%", "unclassified sector share", "python scripts/build_dataset.py"),
    (
        "96.3%",
        "share of Industrials reached by fallback",
        "reports/tables/legacy_sector_classifier_layers.csv",
    ),
]


def _to_float(text: str) -> float | None:
    """Parse the first signed number out of *text*, accepting a Unicode minus."""
    cleaned = text.replace("−", "-").replace("−", "-").replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def main() -> int:
    readme = README.read_text(encoding="utf-8")
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

        if claim.literal not in readme:
            failures.append(
                f"README no longer contains {claim.literal!r} "
                f"({claim.table}.{claim.value_column} is now {actual:.4f})"
            )
            continue

        stated = _to_float(claim.literal)
        if stated is None:
            failures.append(f"could not parse a number out of {claim.literal!r}")
            continue

        # Percentages in the README correspond to fractions in the tables.
        candidates = [stated, stated / 100.0]
        if not any(abs(actual - c) <= claim.tolerance for c in candidates):
            failures.append(
                f"{claim.literal!r} in the README vs {actual:.4f} in "
                f"{claim.table}.csv[{claim.key}][{claim.value_column}]"
            )
            continue
        checked += 1

    for literal, description, source in DERIVED_CLAIMS:
        if literal not in readme:
            failures.append(f"README no longer contains {literal!r} ({description}, {source})")
        else:
            checked += 1

    if failures:
        print("README number check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nRegenerate with: python scripts/run_analysis.py, then update README.md.",
            file=sys.stderr,
        )
        return 1

    print(f"All {checked} README numbers agree with reports/tables/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
