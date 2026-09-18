"""
Assembly of the analysis dataset, and the sample funnel that documents it.

Every row that enters the analysis is built here from five independent
sources, and every row that drops out is counted. The funnel
(:func:`sample_funnel`) is the honest answer to "how many IPOs is this
actually about", and it is published rather than described.

Source            | Supplies                          | Coverage on the universe
------------------|-----------------------------------|------------------------
IPO calendar      | ticker, date, offer price, name   | all
Yahoo Finance     | first-day close -> the target     | partial, survivorship
SEC submissions   | SIC code -> sector, SPAC flag     | partial
SEC EDGAR S-1/F-1 | prospectus text, underwriters     | partial
Yahoo indices     | VIX and NASDAQ, lagged one day    | all

**Market features are lagged.** The old implementation took the VIX close
using ``index <= ipo_date``, which on a listing day is the close of the day
the stock started trading. That is not information an underwriter had when
pricing the deal the previous evening. Everything market-derived here uses the
last observation strictly before the IPO date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import setup_logging

log = setup_logging(__name__)

CALENDAR_PATH = Path("data/raw/ipo_calendar.csv")
PRICES_PATH = Path("data/raw/first_day_prices.csv")
MARKET_PATH = Path("data/raw/market_indices.csv")
SIC_PATH = Path("data/external/company_sic_codes.csv")
S1_DIR = Path("data/raw/s1_filings")

OUTPUT_PATH = Path("data/processed/ipo_analysis.parquet")
FUNNEL_PATH = Path("reports/tables/sample_funnel.csv")

# Columns the IPO calendar contributes. The scraped "Current" price and the
# "Return" derived from it are deliberately excluded: that return runs from the
# offer price to the scrape date, not to the first-day close, and using it as
# the target was the central error in the previous version of this project.
CALENDAR_COLUMNS = ["ipo_date", "ticker", "company_name", "offer_price"]


# ---------------------------------------------------------------------------
# Market regime, lagged
# ---------------------------------------------------------------------------


def load_market(path: Path = MARKET_PATH) -> pd.DataFrame:
    """Load daily VIX and NASDAQ closes with derived rolling statistics.

    Args:
        path: Path to ``market_indices.csv``.

    Returns:
        DataFrame indexed by date with ``vix_close``, ``nasdaq_close``,
        ``nasdaq_30d_return`` and ``nasdaq_30d_volatility``.
    """
    market = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    log_ret = np.log(market["nasdaq_close"] / market["nasdaq_close"].shift(1))
    market["nasdaq_30d_return"] = log_ret.rolling(30).sum()
    market["nasdaq_30d_volatility"] = log_ret.rolling(30).std() * np.sqrt(252)
    return market


def market_as_of_prior_day(dates: pd.Series, market: pd.DataFrame) -> pd.DataFrame:
    """Return market state as of the last trading day *strictly before* each date.

    Args:
        dates: IPO dates.
        market: Output of :func:`load_market`.

    Returns:
        DataFrame aligned to *dates* with ``vix_at_pricing``,
        ``nasdaq_30d_return``, ``nasdaq_30d_volatility`` and
        ``market_data_date`` (the observation date actually used, so the lag
        is auditable rather than asserted).
    """
    wanted = ["vix_close", "nasdaq_30d_return", "nasdaq_30d_volatility"]
    ordered = pd.to_datetime(dates).reset_index(drop=True)

    # searchsorted on the left edge gives the first index >= the IPO date;
    # stepping back one gives the last strictly-earlier trading day.
    positions = market.index.searchsorted(ordered.to_numpy(), side="left") - 1
    valid = positions >= 0

    out = pd.DataFrame(index=ordered.index, columns=[*wanted, "market_data_date"], dtype="object")
    if valid.any():
        rows = market.iloc[positions[valid]]
        for column in wanted:
            out.loc[valid, column] = rows[column].to_numpy()
        out.loc[valid, "market_data_date"] = market.index[positions[valid]]

    out = out.rename(columns={"vix_close": "vix_at_pricing"})
    for column in ["vix_at_pricing", "nasdaq_30d_return", "nasdaq_30d_volatility"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["market_data_date"] = pd.to_datetime(out["market_data_date"])
    return out


def expanding_hot_market(
    dates: pd.Series,
    window_days: int = 90,
    quantile: float = 2 / 3,
    min_history: int = 100,
) -> pd.Series:
    """Flag IPOs priced into a crowded new-issue market, without look-ahead.

    The count of IPOs in the trailing *window_days* is backward-looking and
    fine. The threshold that turns it into a dummy was previously the
    full-sample tercile, which is not knowable at pricing time. Here the
    threshold is the same quantile of the counts observed *so far*.

    Args:
        dates: IPO dates, in any order.
        window_days: Trailing window for the issue count.
        quantile: Quantile of the historical count distribution that defines
            "hot". The default two-thirds keeps the original top-tercile idea.
        min_history: Deals required before a threshold is estimated at all;
            earlier rows get ``nan``.

    Returns:
        Series aligned to *dates*: 1.0 hot, 0.0 not, ``nan`` before history.
    """
    parsed = pd.to_datetime(dates)
    order = np.argsort(parsed.to_numpy(), kind="stable")
    sorted_dates = parsed.to_numpy()[order]

    counts = np.empty(len(sorted_dates), dtype=float)
    window = np.timedelta64(window_days, "D")
    for i, day in enumerate(sorted_dates):
        counts[i] = np.count_nonzero(
            (sorted_dates[: i + 1] >= day - window) & (sorted_dates[: i + 1] <= day)
        )

    flags = np.full(len(sorted_dates), np.nan)
    for i in range(len(sorted_dates)):
        if i < min_history:
            continue
        threshold = np.quantile(counts[:i], quantile)
        flags[i] = float(counts[i] >= threshold)

    out = np.empty(len(sorted_dates))
    out[order] = flags
    return pd.Series(out, index=parsed.index, name="hot_market_dummy")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def load_calendar(path: Path = CALENDAR_PATH) -> pd.DataFrame:
    """Load and de-duplicate the IPO calendar."""
    calendar = pd.read_csv(path, parse_dates=["ipo_date"])
    calendar["ticker"] = calendar["ticker"].astype(str).str.strip().str.upper()
    calendar = calendar[CALENDAR_COLUMNS].copy()
    calendar["offer_price"] = pd.to_numeric(calendar["offer_price"], errors="coerce")
    before = len(calendar)
    calendar = calendar.drop_duplicates(subset=["ticker", "ipo_date"]).reset_index(drop=True)
    if before != len(calendar):
        log.info("Dropped %d duplicate (ticker, date) rows", before - len(calendar))
    return calendar


def filing_paths(s1_dir: Path = S1_DIR) -> pd.DataFrame:
    """Return one row per recovered S-1, with paths to its three extracts."""
    if not s1_dir.exists():
        return pd.DataFrame(
            columns=["ticker", "ipo_date", "full_text_path", "risk_factors_path", "mda_path"]
        )
    rows = []
    for path in sorted(s1_dir.glob("*.txt")):
        stem = path.stem
        if stem.endswith(("_mda", "_risk_factors")):
            continue
        ticker, _, date_str = stem.rpartition("_")
        if not ticker:
            continue
        rows.append(
            {
                "ticker": ticker.upper(),
                "ipo_date": pd.Timestamp(date_str),
                "full_text_path": str(path),
                "risk_factors_path": str(path.with_name(f"{stem}_risk_factors.txt")),
                "mda_path": str(path.with_name(f"{stem}_mda.txt")),
            }
        )
    return pd.DataFrame(rows)


def assemble(
    calendar_path: Path = CALENDAR_PATH,
    prices_path: Path = PRICES_PATH,
    sic_path: Path = SIC_PATH,
    market_path: Path = MARKET_PATH,
    s1_dir: Path = S1_DIR,
) -> pd.DataFrame:
    """Join every source into one row per IPO, before feature engineering.

    Returns:
        DataFrame with identifiers, the target, sector, lagged market state,
        and paths to any recovered filing text.
    """
    df = load_calendar(calendar_path)

    prices = pd.read_csv(prices_path)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["ipo_date"] = pd.to_datetime(prices["ipo_date"])
    keep = [
        "ticker",
        "ipo_date",
        "first_trade_date",
        "first_day_open",
        "first_day_close",
        "split_factor",
        "underpricing",
        "first_week_return",
        "first_month_return",
        "status",
    ]
    prices = prices[[c for c in keep if c in prices.columns]]
    df = df.merge(
        prices.rename(columns={"status": "price_status"}), on=["ticker", "ipo_date"], how="left"
    )

    sic = pd.read_csv(sic_path, dtype={"sic": "string", "cik": "string"})
    sic["ticker"] = sic["ticker"].astype(str).str.upper()
    from src.sic_sectors import UNCLASSIFIED, sector_for_sic

    sic["sector"] = sic["sic"].map(sector_for_sic)
    df = df.merge(
        sic[["ticker", "cik", "sic", "sic_description", "sector"]].drop_duplicates("ticker"),
        on="ticker",
        how="left",
    )
    df["sector"] = df["sector"].fillna(UNCLASSIFIED)
    # SIC 6770 is the SEC's own code for a blank-cheque company.
    df["is_spac"] = (df["sic"] == "6770").astype("float")
    df.loc[df["sic"].isna(), "is_spac"] = np.nan

    market = load_market(market_path)
    df = pd.concat([df, market_as_of_prior_day(df["ipo_date"], market)], axis=1)
    df["hot_market_dummy"] = expanding_hot_market(df["ipo_date"])

    filings = filing_paths(s1_dir)
    df = df.merge(filings, on=["ticker", "ipo_date"], how="left")
    df["has_filing"] = df["full_text_path"].notna().astype(int)

    df["ipo_year"] = df["ipo_date"].dt.year
    df["ipo_quarter"] = df["ipo_date"].dt.quarter
    df["ipo_month"] = df["ipo_date"].dt.month
    df["ipo_dayofweek"] = df["ipo_date"].dt.dayofweek

    return df.sort_values("ipo_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------


def sample_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Count what survives each requirement, in the order they are applied.

    Args:
        df: Output of :func:`assemble`.

    Returns:
        DataFrame with ``step``, ``n``, ``dropped`` and ``pct_of_universe``.
    """
    steps: list[tuple[str, int]] = [
        ("IPOs in the 2019-2024 calendar", len(df)),
        ("with an offer price", int(df["offer_price"].notna().sum())),
        (
            "with a recoverable first-day close",
            int((df["offer_price"].notna() & df["first_day_close"].notna()).sum()),
        ),
        (
            "with a computable first-day return (the target)",
            int(df["underpricing"].notna().sum()),
        ),
        (
            "and a SEC-assigned SIC code",
            int((df["underpricing"].notna() & df["sic"].notna()).sum()),
        ),
        (
            "and a recovered S-1/F-1 prospectus",
            int((df["underpricing"].notna() & (df["has_filing"] == 1)).sum()),
        ),
        (
            # A non-empty section, not merely a file: extract_sections writes
            # an empty placeholder when a prospectus carries too few
            # page-break markers to locate its headings.
            "and a Risk Factors section located",
            int(
                (
                    df["underpricing"].notna()
                    & df["risk_factors_path"]
                    .fillna("")
                    .map(lambda p: bool(p) and Path(p).exists() and Path(p).stat().st_size > 0)
                ).sum()
            ),
        ),
    ]

    funnel = pd.DataFrame(steps, columns=["step", "n"])
    funnel["dropped"] = funnel["n"].shift(1).sub(funnel["n"]).fillna(0).astype(int)
    funnel["pct_of_universe"] = (funnel["n"] / len(df) * 100).round(1)
    return funnel


def selection_comparison(df: pd.DataFrame, flag: str = "has_filing") -> pd.DataFrame:
    """Compare IPOs that survive a selection against those that do not.

    Text-based findings are conditional on a prospectus having been
    recoverable, and that subset is not random. This tabulates the difference
    on observables so the conditioning can be judged rather than assumed.

    Args:
        df: Output of :func:`assemble`.
        flag: Binary column naming the selection.

    Returns:
        DataFrame with one row per observable and columns for each group, the
        difference, and a two-sided p-value (Welch t-test for means, chi-square
        for the sector distribution is reported separately).
    """
    from scipy import stats

    rows = []
    included = df[df[flag] == 1]
    excluded = df[df[flag] != 1]

    for label, column in [
        ("Listing year", "ipo_year"),
        ("Offer price (USD)", "offer_price"),
        ("First-day return", "underpricing"),
        ("VIX at pricing", "vix_at_pricing"),
        ("SPAC share", "is_spac"),
    ]:
        a = pd.to_numeric(included[column], errors="coerce").dropna()
        b = pd.to_numeric(excluded[column], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            rows.append(
                {
                    "observable": label,
                    "in_sample": np.nan,
                    "out_of_sample": np.nan,
                    "difference": np.nan,
                    "p_value": np.nan,
                    "n_in": len(a),
                    "n_out": len(b),
                }
            )
            continue
        test = stats.ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "observable": label,
                "in_sample": round(float(a.mean()), 4),
                "out_of_sample": round(float(b.mean()), 4),
                "difference": round(float(a.mean() - b.mean()), 4),
                "p_value": round(float(test.pvalue), 6),
                "n_in": len(a),
                "n_out": len(b),
            }
        )
    return pd.DataFrame(rows)
