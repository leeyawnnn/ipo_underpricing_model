#!/usr/bin/env python3
"""Fetch the Loughran-McDonald Master Dictionary and derive the word list we use.

The full master dictionary is a 9 MB, 86,553-row CSV carrying per-word corpus
statistics we never touch. Committing it made this repository 23 MB. Instead we
commit the derived seven-category word list (``data/external/lm_sentiment_words.csv``,
about 100 KB) and keep the master a download step.

Licence: the LM dictionary and sentiment lists are free for use in academic
research; commercial use requires a licence from the authors
(loughranmcdonald@gmail.com). See https://sraf.nd.edu/loughranmcdonald-master-dictionary/.

Usage::

    python scripts/fetch_lm_dictionary.py             # download, verify, derive
    python scripts/fetch_lm_dictionary.py --check     # verify the committed subset only
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import requests

# Version pinned by content hash. The SRAF landing page links to a Google Drive
# copy; the id below is the "CSV Format" link as of the access date. If SRAF
# publishes a new release the hash check will fail loudly rather than silently
# changing every number downstream.
SRAF_PAGE = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
MASTER_FILENAME = "Loughran-McDonald_MasterDictionary_1993-2025.csv"
MASTER_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
MASTER_SHA256 = "e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d"
ACCESS_DATE = "2026-09-17"

MASTER_PATH = Path("data/external/lm_master_dictionary.csv")
SUBSET_PATH = Path("data/external/lm_sentiment_words.csv")

# Column name in the master CSV -> column name in our derived subset.
# The master has used both a single tri-valued ``Modal`` column (2018 and
# earlier releases) and separate ``Strong_Modal`` / ``Weak_Modal`` columns
# (current release), so we accept either.
CATEGORY_COLUMNS = [
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Constraining",
    "Strong_Modal",
    "Weak_Modal",
]


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_master(dest: Path = MASTER_PATH) -> Path:
    """Download the master dictionary to *dest* and verify its SHA-256.

    Raises:
        RuntimeError: If the downloaded file does not match ``MASTER_SHA256``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256(dest) == MASTER_SHA256:
        print(f"{dest} already present and matches the pinned hash.")
        return dest

    print(f"Downloading {MASTER_FILENAME} …")
    resp = requests.get(MASTER_URL, timeout=180)
    resp.raise_for_status()
    dest.write_bytes(resp.content)

    digest = sha256(dest)
    if digest != MASTER_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {dest}.\n"
            f"  expected {MASTER_SHA256}\n"
            f"  got      {digest}\n"
            f"SRAF may have published a new release. Check {SRAF_PAGE}, confirm the "
            "new file, then update MASTER_SHA256 and regenerate every downstream "
            "artifact — the sentiment ratios will change."
        )
    print(f"Verified SHA-256 {digest}")
    return dest


def derive_subset(master: Path = MASTER_PATH, dest: Path = SUBSET_PATH) -> pd.DataFrame:
    """Write the seven-category word list derived from *master* to *dest*."""
    df = pd.read_csv(master, low_memory=False)

    word_col = "Word" if "Word" in df.columns else next(
        c for c in df.columns if c.lower() == "word"
    )
    out = pd.DataFrame({"word": df[word_col].astype(str).str.upper()})

    if "Modal" in df.columns and "Strong_Modal" not in df.columns:
        # Older layout: 1 = strong, 2 = moderate, 3 = weak.
        df = df.assign(
            Strong_Modal=(df["Modal"] == 1).astype(int),
            Weak_Modal=(df["Modal"] == 3).astype(int),
        )

    for col in CATEGORY_COLUMNS:
        if col not in df.columns:
            raise RuntimeError(f"Master dictionary is missing expected column {col!r}")
        out[col.lower()] = (df[col] != 0).astype("int8")

    # Keep only words that belong to at least one category we use.
    out = out[out[[c.lower() for c in CATEGORY_COLUMNS]].sum(axis=1) > 0]
    out = out.drop_duplicates(subset="word").sort_values("word").reset_index(drop=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    counts = {c.lower(): int(out[c.lower()].sum()) for c in CATEGORY_COLUMNS}
    print(f"Wrote {dest} — {len(out):,} words, {dest.stat().st_size / 1024:.0f} KB")
    for k, v in counts.items():
        print(f"  {k:<14} {v:>5}")

    meta = dest.with_suffix(".meta.json")
    meta.write_text(
        "{\n"
        f'  "source_page": "{SRAF_PAGE}",\n'
        f'  "source_file": "{MASTER_FILENAME}",\n'
        f'  "source_sha256": "{MASTER_SHA256}",\n'
        f'  "access_date": "{ACCESS_DATE}",\n'
        f'  "derived_by": "python scripts/fetch_lm_dictionary.py",\n'
        f'  "n_words": {len(out)},\n'
        '  "category_counts": {\n'
        + ",\n".join(f'    "{k}": {v}' for k, v in counts.items())
        + "\n  },\n"
        '  "licence": "Free for academic research; commercial use requires a licence '
        'from the authors (loughranmcdonald@gmail.com)."\n'
        "}\n",
        encoding="utf-8",
    )
    print(f"Wrote {meta}")
    return out


def check_subset(dest: Path = SUBSET_PATH) -> int:
    """Validate the committed subset without touching the network."""
    if not dest.exists():
        print(f"{dest} is missing. Run: python scripts/fetch_lm_dictionary.py", file=sys.stderr)
        return 1
    df = pd.read_csv(dest)
    expected = {"word", *(c.lower() for c in CATEGORY_COLUMNS)}
    missing = expected - set(df.columns)
    if missing:
        print(f"{dest} is missing columns: {sorted(missing)}", file=sys.stderr)
        return 1
    print(f"{dest}: {len(df):,} words, columns OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed subset offline instead of downloading",
    )
    args = parser.parse_args()

    if args.check:
        return check_subset()

    download_master()
    derive_subset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
