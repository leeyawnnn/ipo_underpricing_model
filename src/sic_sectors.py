"""
Sector classification from SEC-assigned SIC codes.

The previous classifier inferred a GICS sector by matching regular expressions
against the *company name*, with a final rule that sent anything unmatched to
Industrials. That produced a published finding ("Industrials = 494") which was
in substantial part the fallback rate, and it could never be audited because no
record was kept of which rule fired.

This module replaces the guess with the registrant's own Standard Industrial
Classification code, which the SEC assigns and publishes at
``https://data.sec.gov/submissions/CIK##########.json``. The SIC is mapped to a
GICS-like sector through a committed crosswalk,
``data/external/sic_gics_crosswalk.csv``, which a reader can inspect and
disagree with line by line.

Two consequences worth stating:

* SIC 6770 ("Blank Checks") is the SEC's own code for a special-purpose
  acquisition company, so SPAC identification stops being a name regex.
* A company with no recoverable SIC is labelled ``Unclassified`` and counted.
  Nothing is forced into a bucket to avoid an empty one.

The SIC returned by the submissions endpoint is the registrant's *current*
assignment, not necessarily the one in force on the IPO date. Reassignment is
uncommon, and sector enters this study only as a control, but the caveat is
real and is recorded in the README's leakage audit.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.utils import setup_logging

log = setup_logging(__name__)

CROSSWALK_PATH = Path("data/external/sic_gics_crosswalk.csv")
SIC_CODES_PATH = Path("data/external/company_sic_codes.csv")

UNCLASSIFIED = "Unclassified"


@lru_cache(maxsize=1)
def load_crosswalk(path: str = str(CROSSWALK_PATH)) -> pd.DataFrame:
    """Load the SIC-range to sector crosswalk.

    Args:
        path: Path to the crosswalk CSV.

    Returns:
        DataFrame with ``sic_low``, ``sic_high``, ``sector`` and ``note``,
        sorted so that narrower ranges are tested first.

    Raises:
        FileNotFoundError: If the crosswalk is missing.
        ValueError: If two ranges overlap, which would make the mapping
            order-dependent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SIC crosswalk not found at {p}")

    walk = pd.read_csv(p)
    walk = walk.sort_values(["sic_low", "sic_high"]).reset_index(drop=True)

    highs = walk["sic_high"].shift(1)
    overlap = walk["sic_low"] <= highs
    if overlap.any():
        bad = walk.loc[overlap, "sic_low"].tolist()
        raise ValueError(f"Overlapping SIC ranges in {p} starting at {bad}")

    return walk


def sector_for_sic(sic: int | str | float | None) -> str:
    """Map a SIC code to a GICS-like sector.

    Args:
        sic: Four-digit SIC code, as int, str or float. ``None`` and values
            outside every crosswalk range return ``"Unclassified"``.

    Returns:
        Sector name, or ``"Unclassified"``.

    Example:
        >>> sector_for_sic(6770)
        'SPAC'
        >>> sector_for_sic("2834")
        'Healthcare'
        >>> sector_for_sic(None)
        'Unclassified'
    """
    if sic is None or (isinstance(sic, float) and pd.isna(sic)):
        return UNCLASSIFIED
    try:
        # float() first, because a SIC read from a pandas column containing
        # nulls arrives as "2834.0", which int() rejects outright.
        code = int(float(str(sic).strip()))
    except (TypeError, ValueError):
        return UNCLASSIFIED

    walk = load_crosswalk()
    hit = walk[(walk["sic_low"] <= code) & (code <= walk["sic_high"])]
    if hit.empty:
        return UNCLASSIFIED
    return str(hit["sector"].iloc[0])


def load_sic_codes(path: str | Path = SIC_CODES_PATH) -> pd.DataFrame:
    """Load the ticker-to-SIC table produced by ``scripts/fetch_sic_codes.py``.

    Args:
        path: Path to ``company_sic_codes.csv``.

    Returns:
        DataFrame with ``ticker``, ``cik``, ``sic``, ``sic_description`` and a
        derived ``sector`` column.

    Raises:
        FileNotFoundError: If the table has not been fetched.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"SIC codes not found at {p}. Run: python scripts/fetch_sic_codes.py"
        )
    codes = pd.read_csv(p, dtype={"sic": "string", "cik": "string"})
    codes["sector"] = codes["sic"].map(sector_for_sic)
    return codes


def coverage_report(codes: pd.DataFrame) -> pd.DataFrame:
    """Summarise SIC coverage and the resulting sector distribution.

    Args:
        codes: DataFrame as returned by :func:`load_sic_codes`.

    Returns:
        DataFrame with one row per sector: ``sector``, ``n``, ``pct``.
    """
    counts = codes["sector"].value_counts(dropna=False).rename_axis("sector").reset_index(name="n")
    counts["pct"] = (counts["n"] / len(codes) * 100).round(1)
    return counts
