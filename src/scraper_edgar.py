"""
SEC EDGAR S-1 filing scraper.

For each ticker in the IPO calendar, this module:
1. Searches the EDGAR full-text index for S-1 / S-1/A filings filed before
   the IPO date.
2. Downloads the primary filing document (HTML → plain text).
3. Extracts the "Risk Factors" and "Management's Discussion and Analysis"
   sections by locating the section headings.
4. Saves plain text to ``data/raw/s1_filings/{ticker}_{date}.txt`` and
   section extracts to ``…/{ticker}_{date}_risk_factors.txt`` and
   ``…/{ticker}_{date}_mda.txt``.

EDGAR fair-access policy caps traffic at 10 requests/second and requires a
descriptive User-Agent naming a contact. We throttle to 8 r/s, back off
exponentially, and read the contact from the ``SEC_EDGAR_USER_AGENT``
environment variable — the module refuses to run without one.

Usage (CLI)::

    python -m src.scraper_edgar
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils import retry, setup_logging, throttle

log = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
S1_DIR = RAW_DIR / "s1_filings"
CACHE_DIR = RAW_DIR / ".cache"
IPO_CSV = RAW_DIR / "ipo_calendar.csv"

def _user_agent() -> str:
    """Return the SEC fair-access User-Agent, from the environment.

    SEC fair access (https://www.sec.gov/os/webmaster-faq#developers) requires
    a descriptive User-Agent naming a contactable address. This module used to
    ship a placeholder, "student@university.edu", which is not a real contact
    and risks the SEC blocking the traffic. We refuse to guess and require the
    operator to declare their own.

    Raises:
        RuntimeError: If ``SEC_EDGAR_USER_AGENT`` is unset or has no address.
    """
    value = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not value or "@" not in value:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT is not set to a contactable address.\n"
            'Set it, for example:\n'
            '  export SEC_EDGAR_USER_AGENT="Jane Doe jane@example.com"'
        )
    return value


EDGAR_HEADERS = {"Accept-Encoding": "gzip, deflate"}

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index?"
    "q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}"
    "&forms=S-1,F-1"
)
EDGAR_COMPANY_SEARCH = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "company={name}&CIK=&type=S-1&dateb=&owner=include&count=10"
    "&search_text=&action=getcompany&output=atom"
)
EDGAR_FILING_IDX = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_dashes}/"


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update(EDGAR_HEADERS)


def _ensure_user_agent() -> None:
    """Attach the fair-access User-Agent to the shared session, once."""
    if "User-Agent" not in SESSION.headers or "@" not in str(SESSION.headers["User-Agent"]):
        SESSION.headers["User-Agent"] = _user_agent()


@retry(max_attempts=6, backoff_factor=2.5, initial_wait=1.5,
       exceptions=(requests.RequestException, OSError))
@throttle(calls_per_second=8.0)
def _get(url: str, **kwargs) -> requests.Response:
    """GET *url* with retry/throttle applied.

    Args:
        url: Target URL.
        **kwargs: Additional kwargs forwarded to :func:`requests.Session.get`.

    Returns:
        :class:`requests.Response` with a 2xx status code.

    Raises:
        requests.HTTPError: On persistent 4xx/5xx errors.
    """
    _ensure_user_agent()
    resp = SESSION.get(url, timeout=30, **kwargs)
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 10))
        log.warning("Rate-limited by EDGAR; sleeping %ds", retry_after)
        time.sleep(retry_after)
        resp.raise_for_status()
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

def lookup_cik(ticker: str) -> Optional[str]:
    """Look up the SEC CIK number for a given ticker.

    Uses the EDGAR company-facts tickers.json mapping, which maps ticker
    symbols to CIKs.

    Args:
        ticker: Stock ticker symbol (case-insensitive).

    Returns:
        Zero-padded 10-digit CIK string, or ``None`` if not found.
    """
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        cache_file = CACHE_DIR / "company_tickers.json"

        if cache_file.exists():
            data = json.loads(cache_file.read_text())
        else:
            resp = _get(url)
            data = resp.json()
            cache_file.write_text(json.dumps(data))

        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return f"{entry['cik_str']:010d}"
    except Exception as exc:
        log.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# Filing search
# ---------------------------------------------------------------------------

def find_s1_filing(cik: str, before_date: str) -> Optional[dict]:
    """Find the most recent S-1 or S-1/A filing for a company before a date.

    Args:
        cik: Zero-padded 10-digit CIK string.
        before_date: ISO date string (``YYYY-MM-DD``); only filings before
            this date are considered.

    Returns:
        Dict with keys ``accession_number``, ``filing_date``, ``form_type``,
        and ``primary_doc`` — or ``None`` if no filing is found.
    """
    try:
        url = EDGAR_SUBMISSIONS_URL.format(cik=int(cik))
        resp = _get(url)
        submissions = resp.json()

        filings = submissions.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        candidates = []
        for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
            if form in ("S-1", "S-1/A", "F-1", "F-1/A") and date < before_date:
                candidates.append(
                    {
                        "form_type": form,
                        "filing_date": date,
                        "accession_number": acc,
                        "primary_doc": doc,
                    }
                )

        if not candidates:
            return None

        # Return the most recent S-1/A or S-1 before the IPO date
        candidates.sort(key=lambda x: x["filing_date"], reverse=True)
        return candidates[0]

    except Exception as exc:
        log.warning("Filing search failed for CIK %s: %s", cik, exc)
        return None


# ---------------------------------------------------------------------------
# Document download and text extraction
# ---------------------------------------------------------------------------

def _build_filing_url(cik: str, accession_number: str, primary_doc: str) -> str:
    """Construct the direct URL to a filing's primary HTML document.

    Args:
        cik: CIK (digits only, no leading zeros required here).
        accession_number: Accession number with dashes, e.g.
            ``0001234567-23-000001``.
        primary_doc: Filename of the primary document, e.g. ``forms-1.htm``.

    Returns:
        Full EDGAR archives URL.
    """
    acc_nodash = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{acc_nodash}/{primary_doc}"
    )


def _html_to_text(html: str) -> str:
    """Strip HTML tags and normalise whitespace.

    Args:
        html: Raw HTML string.

    Returns:
        Plain text with excessive whitespace collapsed.
    """
    soup = BeautifulSoup(html, "lxml")
    # Remove script/style
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Section heading patterns (case-insensitive)
_RISK_FACTORS_RE = re.compile(
    r"(?:ITEM\s+1A\.?\s*)?RISK\s+FACTORS", re.IGNORECASE
)
_MDA_RE = re.compile(
    r"MANAGEMENT.{0,10}S?\s+DISCUSSION\s+AND\s+ANALYSIS", re.IGNORECASE
)
_NEXT_ITEM_RE = re.compile(
    r"ITEM\s+\d+[A-Z]?\.", re.IGNORECASE
)


def _extract_section(text: str, start_pattern: re.Pattern, end_pattern: re.Pattern) -> str:
    """Extract the text between two section headings, skipping TOC.

    Args:
        text: Full plain-text prospectus.
        start_pattern: Regex marking the beginning of the desired section.
        end_pattern: Regex marking the start of the next section.

    Returns:
        Extracted section text, or empty string if not found.
    """
    # TOC is usually in the first 10k-20k characters.
    # We find all matches and pick the one that occurs after 10k chars if possible,
    # or the one that is followed by the most text.
    matches = list(start_pattern.finditer(text))
    if not matches:
        return ""

    # Heuristic: pick the first match after character 10,000 (likely after TOC)
    # If no match after 10k, pick the last match.
    start_match = None
    for m in matches:
        if m.start() > 10000:
            start_match = m
            break
    if not start_match:
        start_match = matches[-1]

    start_pos = start_match.end()

    # Find the end pattern after the start
    end_match = end_pattern.search(text, start_pos + 1000)
    end_pos = end_match.start() if end_match else len(text)

    return text[start_pos:end_pos].strip()


def extract_sections(text: str) -> dict[str, str]:
    """Extract Risk Factors and MD&A sections from a prospectus.

    Args:
        text: Full plain-text prospectus.

    Returns:
        Dict with keys ``risk_factors`` and ``mda``, each containing the
        extracted section text (or empty string if not found).
    """
    # Risk Factors: between "Risk Factors" and the next "Item X."
    risk_end_re = re.compile(
        r"ITEM\s+[2-9]|MANAGEMENT.{0,10}S?\s+DISCUSSION", re.IGNORECASE
    )
    risk = _extract_section(text, _RISK_FACTORS_RE, risk_end_re)

    # MD&A: between heading and next Item
    mda_end_re = re.compile(r"ITEM\s+[3-9]", re.IGNORECASE)
    mda = _extract_section(text, _MDA_RE, mda_end_re)

    return {"risk_factors": risk, "mda": mda}


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------

def process_ticker(
    ticker: str,
    ipo_date: str,
    force: bool = False,
) -> dict[str, str]:
    """Download and parse the S-1 filing for one ticker.

    Args:
        ticker: Stock ticker symbol.
        ipo_date: IPO date as ``YYYY-MM-DD`` string.
        force: Re-download even if cached files exist.

    Returns:
        Dict with keys ``ticker``, ``cik``, ``filing_date``, ``form_type``,
        ``full_text_path``, ``risk_factors_path``, ``mda_path``, and
        ``status`` (``"ok"`` or an error description).
    """
    S1_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"{ticker}_{ipo_date}"
    full_path = S1_DIR / f"{stem}.txt"
    risk_path = S1_DIR / f"{stem}_risk_factors.txt"
    mda_path = S1_DIR / f"{stem}_mda.txt"

    if full_path.exists() and not force:
        log.debug("Skipping %s — already on disk", ticker)
        return {
            "ticker": ticker,
            "status": "cached",
            "full_text_path": str(full_path),
            "risk_factors_path": str(risk_path),
            "mda_path": str(mda_path),
        }

    # 1. Lookup CIK
    cik = lookup_cik(ticker)
    if not cik:
        log.warning("%s: CIK not found", ticker)
        return {"ticker": ticker, "status": "cik_not_found"}

    # 2. Find S-1 filing
    filing = find_s1_filing(cik, ipo_date)
    if not filing:
        log.warning("%s: No S-1 filing found before %s", ticker, ipo_date)
        return {"ticker": ticker, "cik": cik, "status": "no_filing"}

    # 3. Download HTML document
    doc_url = _build_filing_url(cik, filing["accession_number"], filing["primary_doc"])
    try:
        resp = _get(doc_url)
        html = resp.text
    except Exception as exc:
        log.error("%s: Download failed (%s)", ticker, exc)
        return {"ticker": ticker, "cik": cik, "status": f"download_error: {exc}"}

    # 4. Convert to plain text
    text = _html_to_text(html)
    full_path.write_text(text, encoding="utf-8")

    # 5. Extract sections
    sections = extract_sections(text)
    risk_path.write_text(sections["risk_factors"], encoding="utf-8")
    mda_path.write_text(sections["mda"], encoding="utf-8")

    log.info(
        "%s: saved full=%d chars, risk=%d chars, mda=%d chars",
        ticker,
        len(text),
        len(sections["risk_factors"]),
        len(sections["mda"]),
    )

    return {
        "ticker": ticker,
        "cik": cik,
        "filing_date": filing["filing_date"],
        "form_type": filing["form_type"],
        "full_text_path": str(full_path),
        "risk_factors_path": str(risk_path),
        "mda_path": str(mda_path),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_edgar_scraper(
    ipo_csv: Path = IPO_CSV,
    max_tickers: Optional[int] = None,
) -> pd.DataFrame:
    """Scrape S-1 filings for all tickers in the IPO calendar CSV.

    Args:
        ipo_csv: Path to the IPO calendar CSV produced by
            :mod:`src.scraper_ipo_calendar`.
        max_tickers: Optional cap on the number of tickers to process (useful
            for test runs).

    Returns:
        DataFrame summarising the scraping results (one row per ticker).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df_ipos = pd.read_csv(ipo_csv, parse_dates=["ipo_date"])
    df_ipos = df_ipos.dropna(subset=["ticker", "ipo_date"])

    if max_tickers:
        df_ipos = df_ipos.head(max_tickers)

    results = []
    for _, row in df_ipos.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        ipo_date = str(row["ipo_date"])[:10]
        result = process_ticker(ticker, ipo_date)
        results.append(result)

    summary = pd.DataFrame(results)
    summary_path = RAW_DIR / "edgar_scrape_summary.csv"
    summary.to_csv(summary_path, index=False)

    ok = (summary["status"] == "ok").sum() if "status" in summary.columns else 0
    log.info("EDGAR scrape complete: %d/%d succeeded", ok, len(summary))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = run_edgar_scraper()
    print(summary["status"].value_counts().to_string())
