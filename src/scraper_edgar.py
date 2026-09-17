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
            "Set it, for example:\n"
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


@retry(
    max_attempts=6,
    backoff_factor=2.5,
    initial_wait=1.5,
    exceptions=(requests.RequestException, OSError),
)
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


def lookup_cik(ticker: str) -> str | None:
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


def _filing_rows(block: dict) -> list[tuple[str, str, str, str]]:
    """Return (form, date, accession, primary document) tuples from a filings block."""
    return list(
        zip(
            block.get("form", []),
            block.get("filingDate", []),
            block.get("accessionNumber", []),
            block.get("primaryDocument", []),
        )
    )


def find_s1_filing(cik: str, before_date: str) -> dict | None:
    """Find the latest S-1 / F-1 or amendment filed before *before_date*.

    The submissions endpoint puts only the most recent filings in
    ``filings.recent`` and pages the rest into separate JSON shards listed
    under ``filings.files``. An earlier version read ``recent`` alone, so a
    registrant that has filed prolifically since its IPO — which is most of
    them by 2026 — could have its S-1 pushed out of the window and be recorded
    as having no filing at all. The shards are now followed when ``recent``
    yields nothing.

    The *latest* qualifying filing is returned, which is the final
    pre-effective amendment: the version of the prospectus closest to the one
    investors actually priced against.

    Args:
        cik: Zero-padded 10-digit CIK string.
        before_date: ISO date string (``YYYY-MM-DD``); only filings strictly
            before this date are considered.

    Returns:
        Dict with ``accession_number``, ``filing_date``, ``form_type`` and
        ``primary_doc``, or ``None`` if nothing qualifies.
    """
    wanted = ("S-1", "S-1/A", "F-1", "F-1/A")

    def qualifying(rows: list[tuple[str, str, str, str]]) -> list[dict]:
        return [
            {
                "form_type": form,
                "filing_date": date,
                "accession_number": accession,
                "primary_doc": document,
            }
            for form, date, accession, document in rows
            if form in wanted and date < before_date and document
        ]

    try:
        submissions = _get(EDGAR_SUBMISSIONS_URL.format(cik=int(cik))).json()
        filings = submissions.get("filings", {})
        candidates = qualifying(_filing_rows(filings.get("recent", {})))

        if not candidates:
            for shard in filings.get("files", []):
                name = shard.get("name")
                if not name:
                    continue
                shard_url = f"https://data.sec.gov/submissions/{name}"
                candidates.extend(qualifying(_filing_rows(_get(shard_url).json())))

        if not candidates:
            return None

        candidates.sort(key=lambda row: row["filing_date"], reverse=True)
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
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary_doc}"


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


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------
#
# A prospectus is not a 10-K. It has no "Item 1A." numbering, so the obvious
# approach - slice from a heading to the next "Item N." - has nothing to stop
# at. The previous implementation ended Risk Factors at the first in-text
# cross-reference to "Management's Discussion and Analysis", which appears a
# few paragraphs in, and ended MD&A at "Item [3-9]", which never matches, so
# MD&A ran to the end of the filing. Measured over 400 saved prospectuses,
# Risk Factors came out at a median 0.3% of the document and MD&A at 58.6%,
# with 122 of 400 above 80%.
#
# The signal that does work is the page-break marker. EDGAR's HTML paginates
# with a "Table of Contents" link at the top of every page, which survives the
# text conversion, and a section always starts at the top of a page. So a
# heading that appears within a few dozen characters after a page break is a
# real heading; the same words elsewhere are prose or a cross-reference.
# Against that rule Risk Factors comes out at a median 20.6% of the document
# and MD&A at 26.1%, which are plausible proportions for an S-1.
#
# Roughly 30% of the saved filings carry too few page-break markers for the
# rule to apply. Their sections are reported as missing rather than guessed;
# the full-prospectus tone ratios, which are the primary text signal, do not
# depend on this at all.

# Section headings distinctive enough to anchor a boundary. Single common
# words - "Business", "Management", "Dilution", "Capitalization",
# "Underwriting" - are deliberately excluded: they appear at the top of a page
# mid-sentence often enough to cut a section short.
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("prospectus_summary", re.compile(r"PROSPECTUS\s+SUMMARY", re.IGNORECASE)),
    ("risk_factors", re.compile(r"(?:ITEM\s+1A\.?\s*)?RISK\s+FACTORS", re.IGNORECASE)),
    (
        "forward_looking",
        re.compile(
            r"(?:SPECIAL\s+NOTE|CAUTIONARY\s+(?:NOTE|STATEMENT))[^.\n]{0,40}"
            r"FORWARD[\s-]LOOKING",
            re.IGNORECASE,
        ),
    ),
    (
        "market_data",
        re.compile(r"(?:INDUSTRY\s+AND\s+MARKET|MARKET\s+AND\s+INDUSTRY)\s+DATA", re.IGNORECASE),
    ),
    ("use_of_proceeds", re.compile(r"USE\s+OF\s+PROCEEDS", re.IGNORECASE)),
    ("dividend_policy", re.compile(r"DIVIDEND\s+POLICY", re.IGNORECASE)),
    (
        "selected_financial",
        re.compile(r"SELECTED\s+(?:CONSOLIDATED\s+)?(?:HISTORICAL\s+)?FINANCIAL", re.IGNORECASE),
    ),
    ("mda", re.compile(r"MANAGEMENT.{0,3}S?\s+DISCUSSION\s+AND\s+ANALYSIS", re.IGNORECASE)),
    (
        "market_risk",
        re.compile(r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES", re.IGNORECASE),
    ),
    ("executive_compensation", re.compile(r"EXECUTIVE\s+COMPENSATION", re.IGNORECASE)),
    (
        "related_party",
        re.compile(r"(?:CERTAIN\s+)?RELATIONSHIPS\s+AND\s+RELATED", re.IGNORECASE),
    ),
    (
        "principal_holders",
        re.compile(r"PRINCIPAL\s+(?:AND\s+SELLING\s+)?(?:STOCK|SHARE)HOLDERS", re.IGNORECASE),
    ),
    (
        "capital_stock",
        re.compile(
            r"DESCRIPTION\s+OF\s+(?:OUR\s+)?(?:CAPITAL\s+STOCK|SECURITIES|SHARE\s+CAPITAL)",
            re.IGNORECASE,
        ),
    ),
    ("future_sale", re.compile(r"SHARES\s+ELIGIBLE\s+FOR\s+FUTURE\s+SALE", re.IGNORECASE)),
    (
        "taxation",
        re.compile(r"MATERIAL\s+U\.?S\.?\s+FEDERAL\s+INCOME\s+TAX", re.IGNORECASE),
    ),
    ("legal_matters", re.compile(r"LEGAL\s+MATTERS", re.IGNORECASE)),
    (
        "financial_statements",
        re.compile(
            r"INDEX\s+TO\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS", re.IGNORECASE
        ),
    ),
]

_PAGE_BREAK_RE = re.compile(r"Table\s+of\s+Contents", re.IGNORECASE)

# Characters after a page break within which a heading still counts as one.
_HEADING_OFFSET_CHARS = 40
# Below this many page breaks the document was not paginated in a way we can
# use, and no section is claimed.
_MIN_PAGE_BREAKS = 5
# A slice shorter than this is a cross-reference, not a section.
_MIN_SECTION_CHARS = 500


def extract_sections(text: str) -> dict[str, str]:
    """Split a prospectus into its named sections.

    Args:
        text: Full plain-text prospectus, as produced by :func:`_html_to_text`.

    Returns:
        Dict keyed by section name (``risk_factors``, ``mda`` and the others
        in ``_SECTION_PATTERNS``). Sections that could not be located, and
        every section when the document carries too few page-break markers to
        anchor on, are absent. Callers should use ``.get(name, "")``.
    """
    page_breaks = [match.end() for match in _PAGE_BREAK_RE.finditer(text)]
    if len(page_breaks) < _MIN_PAGE_BREAKS:
        return {}

    def at_page_top(position: int) -> bool:
        return any(0 <= position - end <= _HEADING_OFFSET_CHARS for end in page_breaks)

    headings = sorted(
        (match.start(), name)
        for name, pattern in _SECTION_PATTERNS
        for match in pattern.finditer(text)
        if at_page_top(match.start())
    )
    if not headings:
        return {}

    first_seen: dict[str, int] = {}
    for position, name in headings:
        first_seen.setdefault(name, position)
    ordered = sorted(first_seen.items(), key=lambda item: item[1])

    sections: dict[str, str] = {}
    for index, (name, position) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(text)
        chunk = text[position:end].strip()
        if len(chunk) >= _MIN_SECTION_CHARS:
            sections[name] = chunk
    return sections


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
    risk_path.write_text(sections.get("risk_factors", ""), encoding="utf-8")
    mda_path.write_text(sections.get("mda", ""), encoding="utf-8")

    log.info(
        "%s: saved full=%d chars, risk=%d chars, mda=%d chars",
        ticker,
        len(text),
        len(sections.get("risk_factors", "")),
        len(sections.get("mda", "")),
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
    max_tickers: int | None = None,
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
