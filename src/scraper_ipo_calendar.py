"""
IPO calendar scraper - stockanalysis.com.

Fetches the ticker, company name, listing date and offer price for US IPOs in
2019-2024 and persists them to ``data/raw/ipo_calendar.csv``.

**What this source does not provide.** The year pages carry the columns
``[IPO Date, Symbol, Company Name, IPO Price, Current, Return]``. ``Return``
runs from the offer price to the price *on the day the page was fetched*, not
to the first-day close. An earlier version of this scraper mapped it to
``first_day_return_pct`` and the whole project was built on it; Palantir was
recorded as +1873.66% underpriced against a true first-day return of +31.03%.
It is now named ``return_to_scrape_date`` and nothing downstream reads it.
The first-day close comes from :mod:`src.scraper_prices` instead.

The pages also carry no shares offered, no proceeds and no underwriter. Deal
size and the syndicate are recovered from the prospectus cover page in
:mod:`src.feature_engineering` and :mod:`src.underwriters`.

**Conduct.** https://stockanalysis.com/robots.txt disallows only /e/ and /p/,
not /ipos/. Requests are throttled to 2 per second and year pages are cached
on disk so a re-run costs no traffic. Their terms permit snippets with
attribution but not republishing content in full, so the scrape is not
committed to this repository.

Usage (CLI)::

    python -m src.scraper_ipo_calendar

The script is idempotent: already-downloaded year pages are served from a
disk cache in ``data/raw/.cache/``.
"""

from __future__ import annotations

import os
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
CACHE_DIR = RAW_DIR / ".cache"
OUTPUT_CSV = RAW_DIR / "ipo_calendar.csv"

# A descriptive User-Agent with a reachable contact, so the site operator can
# get in touch rather than block an anonymous crawler. The contact comes from
# SCRAPER_CONTACT; the placeholder that shipped here ("student@university.edu")
# was not a real address.
_CONTACT = os.environ.get("SCRAPER_CONTACT", "").strip()
HEADERS = {
    "User-Agent": (
        "ipo-underpricing-research/1.0 (academic study of US IPO first-day returns"
        + (f"; contact: {_CONTACT}" if _CONTACT else "")
        + ")"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

YEARS = list(range(2019, 2025))  # 2019 through 2024

# stockanalysis.com exposes per-year IPO pages
BASE_URL = "https://stockanalysis.com/ipos/{year}/"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


@retry(
    max_attempts=5, backoff_factor=2.0, initial_wait=2.0, exceptions=(requests.RequestException,)
)
@throttle(calls_per_second=2.0)
def _fetch(url: str) -> str:
    """Fetch *url* and return the response text.

    Args:
        url: Full URL to GET.

    Returns:
        Response body as a UTF-8 string.

    Raises:
        requests.HTTPError: On 4xx/5xx responses after exhausting retries.
    """
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_ipo_table(html: str, year: int) -> pd.DataFrame:
    """Extract the IPO table from a stockanalysis.com year page.

    Args:
        html: Raw HTML string of the page.
        year: Calendar year being parsed (used only for logging).

    Returns:
        DataFrame with one row per IPO.  Columns may vary slightly between
        years; downstream code handles renaming.
    """
    soup = BeautifulSoup(html, "lxml")

    # The main data table has id="main-table" on newer pages; fall back to
    # the first <table> element if the id is absent.
    table = soup.find("table", {"id": "main-table"}) or soup.find("table")
    if table is None:
        log.warning("No table found on page for year %d", year)
        return pd.DataFrame()

    rows: list[dict] = []
    headers: list[str] = []

    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]

    for tr in table.find_all("tr")[1:]:  # skip header row
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        if headers and len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
        else:
            rows.append({f"col_{i}": v for i, v in enumerate(cells)})

    df = pd.DataFrame(rows)
    df["scrape_year"] = year
    log.info("Parsed %d IPOs for year %d", len(df), year)
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_COLUMN_MAP = {
    # stockanalysis.com column name variants → canonical names
    "Symbol": "ticker",
    "Ticker": "ticker",
    "Company": "company_name",
    "Company Name": "company_name",
    "IPO Date": "ipo_date",
    "Date": "ipo_date",
    "Price": "offer_price",
    "IPO Price": "offer_price",
    "Offer Price": "offer_price",
    "Shares": "shares_offered",
    "Shares Offered": "shares_offered",
    "$ Raised": "offer_size_m",
    "Amount Raised ($M)": "offer_size_m",
    # NOT the first-day return: this is the return to the date the page was
    # fetched. Named so that it cannot be mistaken for the target again.
    "Return": "return_to_scrape_date",
    "Current": "price_at_scrape_date",
    "Exchange": "exchange",
    "Underwriter": "lead_underwriter",
    "Lead Underwriter": "lead_underwriter",
    "Sector": "sector",
    "Industry": "industry",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names and coerce dtypes.

    Args:
        df: Raw parsed DataFrame.

    Returns:
        Normalised DataFrame.
    """
    df = df.rename(columns={k: v for k, v in _COLUMN_MAP.items() if k in df.columns})

    # Coerce numeric columns
    for col in ("offer_price", "shares_offered", "offer_size_m", "return_to_scrape_date"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[$,%\s]", "", regex=True)
                .replace("", float("nan"))
                .pipe(pd.to_numeric, errors="coerce")
            )

    if "ipo_date" in df.columns:
        df["ipo_date"] = pd.to_datetime(df["ipo_date"], errors="coerce")

    # Compute offer_size_m from price × shares if missing
    if "offer_size_m" not in df.columns and {"offer_price", "shares_offered"}.issubset(df.columns):
        df["offer_size_m"] = (df["offer_price"] * df["shares_offered"]) / 1e6

    # Drop rows with no ticker or no date (structural failures)
    if "ticker" in df.columns:
        df = df[df["ticker"].notna() & (df["ticker"] != "")]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def scrape_ipo_calendar(years: list[int] = YEARS) -> pd.DataFrame:
    """Scrape IPO calendar data for the given years.

    Args:
        years: List of calendar years to fetch.

    Returns:
        Combined, normalised DataFrame of all IPOs.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for year in years:
        url = BASE_URL.format(year=year)
        cache_file = CACHE_DIR / f"ipo_calendar_{year}.html"

        if cache_file.exists():
            log.info("Loading %d page from cache", year)
            html = cache_file.read_text(encoding="utf-8")
        else:
            log.info("Fetching %d from %s", year, url)
            try:
                html = _fetch(url)
                cache_file.write_text(html, encoding="utf-8")
            except Exception as exc:
                log.error("Failed to fetch %d: %s", year, exc)
                continue

        df_year = _parse_ipo_table(html, year)
        if not df_year.empty:
            frames.append(df_year)

    if not frames:
        log.error("No IPO data collected — check network access and cache.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = _normalise(combined)

    # Deduplicate on (ticker, ipo_date) — S-1/A amendments can appear twice
    if {"ticker", "ipo_date"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["ticker", "ipo_date"])

    combined.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved %d IPO records → %s", len(combined), OUTPUT_CSV)
    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = scrape_ipo_calendar()
    print(df.head(10).to_string())
    print(f"\nTotal IPOs collected: {len(df)}")
