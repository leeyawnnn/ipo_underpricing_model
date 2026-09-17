#!/usr/bin/env python3
"""Assemble the analysis dataset from the committed and fetched sources.

Runs the join, the feature engineering and the sample funnel, and writes:

  data/processed/ipo_analysis.parquet   one row per IPO in the calendar
  data/processed/analysis_sample.parquet  the rows with a computable target
  reports/tables/sample_funnel.csv
  reports/tables/selection_comparison.csv
  reports/tables/sector_coverage.csv

Usage::

    python scripts/build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import provenance
from src.dataset import assemble, sample_funnel, selection_comparison
from src.feature_engineering import build_all_features
from src.utils import setup_logging

COMMAND = "python scripts/build_dataset.py"
SOURCES = [
    Path("data/raw/ipo_calendar.csv"),
    Path("data/raw/first_day_prices.csv"),
    Path("data/external/company_sic_codes.csv"),
    Path("data/raw/market_indices.csv"),
]

log = setup_logging(__name__)

FULL_PATH = Path("data/processed/ipo_analysis.parquet")
SAMPLE_PATH = Path("data/processed/analysis_sample.parquet")
TABLES = Path("reports/tables")


def main() -> int:
    log.info("Assembling sources …")
    df = assemble()

    log.info("Engineering features …")
    df = build_all_features(df)

    # A split after listing means Yahoo restated the first-day price, and its
    # split calendar is demonstrably incomplete for some micro-caps. The flag
    # travels with the row so every result can be re-run without them.
    df["split_adjusted"] = (
        pd.to_numeric(df.get("split_factor"), errors="coerce").fillna(1.0) != 1.0
    ).astype(int)

    funnel = sample_funnel(df)
    provenance.write_table(
        funnel,
        "sample_funnel",
        "IPOs surviving each data requirement, in the order applied",
        command=COMMAND,
        inputs=SOURCES,
        directory=TABLES,
    )
    log.info("Sample funnel:\n%s", funnel.to_string(index=False))

    comparison = selection_comparison(df)
    provenance.write_table(
        comparison,
        "selection_comparison",
        "IPOs with a recovered prospectus compared against those without, on observables",
        command=COMMAND,
        inputs=SOURCES,
        directory=TABLES,
    )
    log.info(
        "Selection comparison (filing recovered vs not):\n%s", comparison.to_string(index=False)
    )

    coverage = df["sector"].value_counts(dropna=False).rename_axis("sector").reset_index(name="n")
    coverage["pct"] = (coverage["n"] / len(df) * 100).round(1)
    provenance.write_table(
        coverage,
        "sector_coverage",
        "Sector distribution over the full calendar, from SEC-assigned SIC codes",
        command=COMMAND,
        inputs=SOURCES,
        directory=TABLES,
    )

    FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FULL_PATH, index=False)

    sample = df[df["underpricing"].notna()].reset_index(drop=True)
    sample.to_parquet(SAMPLE_PATH, index=False)

    log.info("Wrote %s (%d rows) and %s (%d rows)", FULL_PATH, len(df), SAMPLE_PATH, len(sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
