#!/usr/bin/env python3
"""Assert that the figure pipeline produced every expected figure, intact.

CI runs scripts/build_figures.py into a scratch directory and then this, so a
figure that silently fails to render, renders empty, or grows past GitHub's
comfortable size fails the build rather than reaching the README.

Usage::

    python scripts/check_figures.py reports/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

# Expected filename -> (minimum bytes, maximum bytes). The upper bound on PNGs
# is the 250 KB budget for a README image; SVGs are text and compress, but a
# runaway one usually means a scatter was drawn as vector by mistake.
EXPECTED: dict[str, tuple[int, int]] = {
    "01_first_day_return_distribution.png": (20_000, 256_000),
    "02_sector_composition.svg": (5_000, 400_000),
    "03_sector_year_median_return.png": (20_000, 256_000),
    "04_tone_deciles.svg": (5_000, 400_000),
    "05_vix_and_dispersion.svg": (5_000, 400_000),
    "06_feature_correlations.svg": (5_000, 400_000),
    "07_litigious_tone.svg": (5_000, 400_000),
    "08_disclosure_concentration.png": (20_000, 256_000),
    "09_model_performance.svg": (5_000, 400_000),
    "10_holdout_predictions.svg": (5_000, 400_000),
    "11_shap_importance.svg": (5_000, 400_000),
    "12_sample_funnel.svg": (5_000, 400_000),
}


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/figures")
    failures: list[str] = []

    for name, (minimum, maximum) in sorted(EXPECTED.items()):
        path = directory / name
        if not path.exists():
            failures.append(f"missing: {name}")
            continue
        size = path.stat().st_size
        if size < minimum:
            failures.append(f"suspiciously small ({size:,} B < {minimum:,} B): {name}")
        elif size > maximum:
            failures.append(f"over the size budget ({size:,} B > {maximum:,} B): {name}")
        else:
            print(f"ok  {name}  {size / 1024:.0f} KB")

    unexpected = sorted(
        p.name
        for p in directory.glob("*")
        if p.is_file() and p.suffix in {".png", ".svg"} and p.name not in EXPECTED
    )
    for name in unexpected:
        failures.append(f"unexpected file left in {directory}: {name}")

    if failures:
        print("\nFigure check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\nAll {len(EXPECTED)} figures present and within budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
