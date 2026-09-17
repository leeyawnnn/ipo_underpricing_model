#!/usr/bin/env python3
"""Run every hypothesis test and model evaluation, and write the result tables.

Reads data/processed/analysis_sample.parquet and writes to reports/tables/.
Each table carries a sibling .meta.json recording the command, the git commit,
the UTC timestamp and the input file's SHA-256, so any number in the README can
be traced back to the code and data that produced it.

Usage::

    python scripts/run_analysis.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import hypothesis_tests as ht  # noqa: E402
from src import models  # noqa: E402
from src.utils import setup_logging  # noqa: E402

log = setup_logging(__name__)

SAMPLE_PATH = Path("data/processed/analysis_sample.parquet")
TABLES = Path("reports/tables")


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True, timeout=10)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_table(frame: pd.DataFrame, name: str, description: str) -> None:
    """Write *frame* to reports/tables/<name>.csv with a provenance sidecar."""
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    frame.to_csv(path, index=False)
    (TABLES / f"{name}.meta.json").write_text(
        json.dumps(
            {
                "description": description,
                "command": "python scripts/run_analysis.py",
                "git_commit": _git_commit(),
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "input": str(SAMPLE_PATH),
                "input_sha256": _sha256(SAMPLE_PATH),
                "rows": int(len(frame)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %s (%d rows)", path, len(frame))


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the first-day return overall, by year and by split status."""
    rows = []

    def add(label: str, values: pd.Series) -> None:
        values = values.dropna()
        if values.empty:
            return
        rows.append(
            {
                "group": label,
                "n": int(len(values)),
                "median": round(float(values.median()), 4),
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std()), 4),
                "p25": round(float(values.quantile(0.25)), 4),
                "p75": round(float(values.quantile(0.75)), 4),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "share_positive": round(float((values > 0).mean()), 4),
            }
        )

    add("All", df["underpricing"])
    for year, group in df.groupby("ipo_year"):
        add(f"Listing year {int(year)}", group["underpricing"])
    add("Prices needing no split correction", df.loc[df["split_adjusted"] == 0, "underpricing"])
    add("Prices split-corrected", df.loc[df["split_adjusted"] == 1, "underpricing"])
    add("With a recovered prospectus", df.loc[df["lm_litigious_ratio"].notna(), "underpricing"])
    return pd.DataFrame(rows)


def hypothesis_summary(results: dict) -> pd.DataFrame:
    """Flatten the six test results into one publishable row each."""
    rows = []
    for key in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        result = results[key]
        ci = result.get("ci_95", (float("nan"), float("nan")))
        rows.append(
            {
                "test": key,
                "hypothesis": result.get("hypothesis", ""),
                "method": result.get("test", ""),
                "n": result.get("n", 0),
                "statistic": result.get("statistic", float("nan")),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_raw": result.get("p_value", float("nan")),
                "p_robust": result.get("fligner_p_value", float("nan")),
                "reject_at_005_raw": result.get("reject_h0", False),
            }
        )
    frame = pd.DataFrame(rows)
    adjusted = results["adjusted"][["test", "p_bonferroni", "p_bh",
                                    "significant_bonferroni", "significant_bh"]]
    return frame.merge(adjusted, on="test", how="left")


def main() -> int:
    if not SAMPLE_PATH.exists():
        print(f"{SAMPLE_PATH} not found. Run scripts/build_dataset.py first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(SAMPLE_PATH)
    log.info("Analysis sample: %d rows, %d with a recovered prospectus",
             len(df), int(df["lm_litigious_ratio"].notna().sum()))

    write_table(descriptive_table(df), "descriptive_statistics",
                "First-day return distribution overall, by listing year and by split status")

    log.info("Running the hypothesis family …")
    results = ht.run_all(df)
    write_table(hypothesis_summary(results), "hypothesis_summary",
                "Six hypothesis tests with raw, Bonferroni and Benjamini-Hochberg p-values")
    write_table(results["adjusted"], "multiple_testing_levene",
                "Family-wise correction using Levene for the two variance tests")
    write_table(results["adjusted_robust"], "multiple_testing_rank_based",
                "Family-wise correction using Fligner-Killeen for the two variance tests")
    write_table(results["h1_robustness"], "h1_robustness",
                "H1 re-estimated across subsamples and control sets")
    write_table(results["H1"]["quintiles"], "h1_quintiles",
                "Median first-day return by litigious-ratio quintile, with bootstrap CIs")
    write_table(results["H1"]["lm_categories"], "lm_category_correlations",
                "Spearman rho between each LM category and the first-day return")
    if isinstance(results["H3"].get("terciles"), pd.DataFrame):
        write_table(results["H3"]["terciles"], "h3_terciles",
                    "Median first-day return by disclosure-concentration tercile, with CIs")

    log.info("Cross-validating models …")
    per_fold, summary = models.cross_validate(df)
    write_table(per_fold, "model_cv_per_fold",
                "Per-fold out-of-sample metrics, expanding-window TimeSeriesSplit")
    write_table(summary, "model_cv_summary",
                "Out-of-sample metrics averaged across folds, with the spread")

    per_fold_w, summary_w = models.cross_validate(df, winsorise=True)
    write_table(summary_w, "model_cv_summary_winsorised",
                "As model_cv_summary, target clipped at the training fold's 1st/99th percentile")

    holdout, _, _ = models.holdout_evaluation(df)
    write_table(holdout, "model_holdout_2024",
                "Trained on listings before 2024-01-01, scored on 2024 listings")

    log.info("Computing SHAP on the full-sample LightGBM …")
    pipeline, fitted_data, features = models.fit_final_lightgbm(df)
    _, importance, _ = models.shap_summary(pipeline, fitted_data, features)
    write_table(importance.head(25), "shap_importance",
                "Mean absolute SHAP value per feature for the full-sample LightGBM; "
                "describes the model, not the data-generating process")

    write_table(
        pd.DataFrame(
            {
                "role": (["numeric"] * len(features.numeric)
                         + ["categorical"] * len(features.categorical)
                         + ["dropped_low_coverage"] * len(features.dropped_missing)
                         + ["dropped_constant"] * len(features.dropped_constant)),
                "feature": (features.numeric + features.categorical
                            + features.dropped_missing + features.dropped_constant),
            }
        ),
        "feature_set",
        "Columns offered to the models, and those dropped for coverage or zero variance",
    )

    print("\nAll tables written to reports/tables/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
