#!/usr/bin/env python3
"""Fetch each IPO registrant's SEC-assigned SIC code.

Resolves every ticker in the IPO calendar to a CIK via the SEC's published
ticker map, then reads the ``sic`` field from
``https://data.sec.gov/submissions/CIK##########.json``.

SEC fair access requires a descriptive User-Agent naming a real contact, and
caps traffic at 10 requests per second. This script declares the contact from
the ``SEC_EDGAR_USER_AGENT`` environment variable and throttles to 8 r/s.

Usage::

    export SEC_EDGAR_USER_AGENT="Your Name your.email@example.com"
    python scripts/fetch_sic_codes.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import setup_logging  # noqa: E402

log = setup_logging(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

CACHE_DIR = Path("data/raw/.cache")
OUT_PATH = Path("data/external/company_sic_codes.csv")

REQUESTS_PER_SECOND = 8.0
_MIN_INTERVAL = 1.0 / REQUESTS_PER_SECOND


def user_agent() -> str:
    """Return the SEC fair-access User-Agent string.

    Raises:
        SystemExit: If ``SEC_EDGAR_USER_AGENT`` is unset. The SEC requires a
            real contact address, and a placeholder gets the whole project
            rate-limited or blocked, so we refuse to guess one.
    """
    import os

    value = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not value or "@" not in value:
        raise SystemExit(
            "SEC_EDGAR_USER_AGENT is not set to a contactable address.\n"
            'Set it, for example:\n'
            '  export SEC_EDGAR_USER_AGENT="Jane Doe jane@example.com"\n'
            "SEC fair access (https://www.sec.gov/os/webmaster-faq#developers) "
            "requires a descriptive User-Agent with a contact."
        )
    return value


def ticker_to_cik(session: requests.Session) -> dict[str, str]:
    """Return a ticker -> zero-padded CIK map from the SEC's published file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "company_tickers.json"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        resp = session.get(TICKER_MAP_URL, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        cache.write_text(json.dumps(payload), encoding="utf-8")
    return {
        str(entry["ticker"]).upper(): f'{int(entry["cik_str"]):010d}'
        for entry in payload.values()
    }


def fetch_sic(session: requests.Session, cik: str) -> tuple[str | None, str | None]:
    """Return ``(sic, sic_description)`` for a CIK, or ``(None, None)``."""
    try:
        resp = session.get(SUBMISSIONS_URL.format(cik=cik), timeout=60)
        if resp.status_code != 200:
            return None, None
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None, None
    sic = str(payload.get("sic") or "").strip() or None
    return sic, str(payload.get("sicDescription") or "").strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ipo-csv",
        type=Path,
        default=Path("data/raw/ipo_calendar.csv"),
        help="IPO calendar providing the ticker universe",
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if not args.ipo_csv.exists():
        print(f"{args.ipo_csv} not found. Run scripts/fetch_ipo_calendar.py first.",
              file=sys.stderr)
        return 1

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})

    ipos = pd.read_csv(args.ipo_csv)
    tickers = sorted({str(t).strip().upper() for t in ipos["ticker"].dropna()})
    log.info("Resolving %d tickers", len(tickers))

    cik_map = ticker_to_cik(session)

    rows = []
    last = 0.0
    for i, ticker in enumerate(tickers, start=1):
        cik = cik_map.get(ticker)
        if cik is None:
            rows.append({"ticker": ticker, "cik": None, "sic": None,
                         "sic_description": None, "status": "cik_not_found"})
            continue

        elapsed = time.monotonic() - last
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        last = time.monotonic()

        sic, description = fetch_sic(session, cik)
        rows.append({
            "ticker": ticker,
            "cik": cik,
            "sic": sic,
            "sic_description": description,
            "status": "ok" if sic else "no_sic",
        })
        if i % 200 == 0:
            log.info("Progress: %d / %d", i, len(tickers))

    codes = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    codes.to_csv(args.out, index=False)

    log.info("Wrote %s", args.out)
    log.info("Status:\n%s", codes["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
