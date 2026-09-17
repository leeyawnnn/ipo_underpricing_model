#!/usr/bin/env python3
"""Fetch Jay Ritter's Carter-Manaster underwriter reputation rankings.

Source: https://site.warrington.ufl.edu/ritter/ipo-data/ -> "Underwriter-Rank.xls".
The workbook holds one column per ranking period (Rank8084 ... Rank26), scored
on the Carter-Manaster 0-9 scale. Ritter encodes a missing rank as -9 and adds
0.001 to every score to mark the entry as his own; both are undone here.

Ranks are period-specific, so an IPO must be matched against the column
covering its listing year. The output is a long CSV with one row per
(underwriter, year).

Usage::

    python scripts/fetch_underwriter_ranks.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests

SOURCE_PAGE = "https://site.warrington.ufl.edu/ritter/ipo-data/"
SOURCE_URL = "https://site.warrington.ufl.edu/ritter/files/Underwriter-Rank.xls"
SOURCE_SHA256 = "bd10e6908faabf3f5304b78b9a7761fc52922d08133e8ef81d795998937340c3"
ACCESS_DATE = "2026-09-17"

XLS_PATH = Path("data/external/ritter_underwriter_rank.xls")
OUT_PATH = Path("data/external/underwriter_ranks.csv")

# Ritter's ranking-period columns mapped to the listing years they cover.
# Only the periods overlapping this study's 2019-2024 window are emitted.
PERIOD_YEARS: dict[str, tuple[int, ...]] = {
    "Rank1820": (2018, 2019, 2020),
    "Rank2122": (2021, 2022),
    "Rank23": (2023,),
    "Rank24": (2024,),
}

MISSING_SENTINEL = -9


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path = XLS_PATH) -> Path:
    """Download the workbook to *dest* and verify its SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256(dest) == SOURCE_SHA256:
        print(f"{dest} already present and matches the pinned hash.")
        return dest

    print(f"Downloading {SOURCE_URL} …")
    resp = requests.get(SOURCE_URL, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)

    digest = sha256(dest)
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {dest}.\n"
            f"  expected {SOURCE_SHA256}\n"
            f"  got      {digest}\n"
            f"Ritter updates this file periodically. Check {SOURCE_PAGE}, then update "
            "SOURCE_SHA256 and regenerate every downstream artifact."
        )
    print(f"Verified SHA-256 {digest}")
    return dest


def to_long(xls: Path = XLS_PATH, dest: Path = OUT_PATH) -> pd.DataFrame:
    """Reshape the workbook into a long (underwriter, year, rank) CSV."""
    wide = pd.read_excel(xls, sheet_name="Sheet1", header=0)
    wide = wide.rename(columns={wide.columns[0]: "underwriter"})

    rows = []
    for column, years in PERIOD_YEARS.items():
        if column not in wide.columns:
            raise RuntimeError(f"{xls} is missing expected column {column!r}")
        ranks = pd.to_numeric(wide[column], errors="coerce")
        ranks = ranks.where(ranks != MISSING_SENTINEL)
        # Ritter adds 0.001 to every score as a provenance marker.
        ranks = (ranks - 0.001).round(2)
        for year in years:
            block = pd.DataFrame(
                {
                    "underwriter": wide["underwriter"].astype(str).str.strip(),
                    "year": year,
                    "rank": ranks,
                    "source_column": column,
                }
            )
            rows.append(block.dropna(subset=["rank"]))

    long = pd.concat(rows, ignore_index=True)
    long = long[long["underwriter"].str.len() > 0]
    long = long.sort_values(["underwriter", "year"]).reset_index(drop=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(dest, index=False)

    meta = dest.with_suffix(".meta.json")
    n_uw = long["underwriter"].nunique()
    meta.write_text(
        "{\n"
        f'  "source_page": "{SOURCE_PAGE}",\n'
        f'  "source_url": "{SOURCE_URL}",\n'
        f'  "source_sha256": "{SOURCE_SHA256}",\n'
        f'  "access_date": "{ACCESS_DATE}",\n'
        f'  "derived_by": "python scripts/fetch_underwriter_ranks.py",\n'
        f'  "n_rows": {len(long)},\n'
        f'  "n_underwriters": {n_uw},\n'
        '  "scale": "Carter-Manaster 0-9, Ritter updates of Loughran-Ritter (2004)",\n'
        '  "note": "Ranks are period-specific; rows are expanded to one per listing year."\n'
        "}\n",
        encoding="utf-8",
    )
    print(f"Wrote {dest} — {len(long):,} rows, {n_uw} underwriters, {ACCESS_DATE}")
    print(f"Wrote {meta}")
    return long


if __name__ == "__main__":
    download()
    to_long()
