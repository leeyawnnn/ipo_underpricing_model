#!/usr/bin/env python3
"""Re-extract Risk Factors and MD&A from prospectuses already on disk.

Section extraction is a text operation on the saved full text, so improving it
does not require re-downloading anything. This rewrites the two section files
beside every ``data/raw/s1_filings/{ticker}_{date}.txt`` and reports coverage.

Usage::

    python scripts/extract_sections.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scraper_edgar import extract_sections
from src.utils import setup_logging

log = setup_logging(__name__)

S1_DIR = Path("data/raw/s1_filings")


def main() -> int:
    if not S1_DIR.exists():
        print(f"{S1_DIR} not found. Run: python -m src.scraper_edgar", file=sys.stderr)
        return 1

    stems = sorted(
        path.stem
        for path in S1_DIR.glob("*.txt")
        if not path.stem.endswith(("_mda", "_risk_factors"))
    )

    counts = {"documents": 0, "risk_factors": 0, "mda": 0, "no_page_breaks": 0}
    for stem in stems:
        text = (S1_DIR / f"{stem}.txt").read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        counts["documents"] += 1

        sections = extract_sections(text)
        if not sections:
            counts["no_page_breaks"] += 1

        risk = sections.get("risk_factors", "")
        mda = sections.get("mda", "")
        (S1_DIR / f"{stem}_risk_factors.txt").write_text(risk, encoding="utf-8")
        (S1_DIR / f"{stem}_mda.txt").write_text(mda, encoding="utf-8")

        counts["risk_factors"] += bool(risk)
        counts["mda"] += bool(mda)

    total = counts["documents"]
    log.info(
        "Re-extracted %d documents: Risk Factors in %d (%.0f%%), MD&A in %d (%.0f%%); "
        "%d had too few page-break markers to anchor on",
        total,
        counts["risk_factors"],
        100 * counts["risk_factors"] / max(total, 1),
        counts["mda"],
        100 * counts["mda"] / max(total, 1),
        counts["no_page_breaks"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
