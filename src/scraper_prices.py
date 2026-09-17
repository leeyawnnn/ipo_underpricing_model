"""
First-day price collection from Yahoo Finance.

Underpricing is ``(first_day_close - offer_price) / offer_price``, where
``first_day_close`` is the *unadjusted* closing price on the first trading day.
Getting that right is the whole job of this module, and there are two traps:

1. **Yahoo's "raw" OHLC is already split-adjusted.** ``auto_adjust=False`` only
   turns off the *dividend* adjustment; the ``Close`` column is still restated
   for every split after the date requested. NVDA closed at $481.68 on
   2024-01-02, and Yahoo reports $48.17 because of the 10-for-1 split in June
   2024. For an IPO that later reverse-split 1-for-20, the reported first-day
   close is 20x the price anyone actually paid. We undo this by multiplying by
   the product of every split ratio strictly after the first trading day.

2. **Yahoo publishes split events before it restates prices.** A split dated
   within the last few days appears in the split calendar while the history is
   still on the old scale, so blindly applying the factor destroys the price.
   :func:`split_is_applied` decides per split, by measuring whether the series
   actually jumps at the split date, and only applied splits enter the factor.

3. **Tickers get reused.** ASPL's Yahoo history starts in 2007 at a flat $0.34
   and has nothing to do with the 2020 SPAC that listed under that symbol. A
   genuine IPO's series begins at the IPO; one that begins materially earlier
   belongs to a different issuer, or to an OTC company uplisting rather than
   floating, and either way its "first-day return" is not underpricing.

4. **Delisted tickers vanish.** Yahoo purges history for deregistered symbols,
   so coverage falls over time and the survivors are a biased subsample. The
   status column records why each ticker failed so the selection can be
   measured rather than assumed.

Usage (CLI)::

    python -m src.scraper_prices
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import setup_logging

log = setup_logging(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
IPO_CSV = RAW_DIR / "ipo_calendar.csv"
OUTPUT_CSV = RAW_DIR / "first_day_prices.csv"

# A first trading day more than this many calendar days after the scheduled IPO
# date is treated as a different event (re-IPO, ticker reuse) and rejected.
MAX_TRADING_DAY_LAG = 7

# Price history starting more than this many calendar days before the IPO date
# means the symbol carries a previous issuer's series.
MAX_PRE_IPO_HISTORY_DAYS = 5

# Trading days either side of a split date used to test whether the price
# series was actually restated for it.
SPLIT_PROBE_DAYS = 5


# ---------------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------------

def split_is_applied(
    closes: pd.Series,
    split_date: pd.Timestamp,
    ratio: float,
    probe: int = SPLIT_PROBE_DAYS,
) -> bool:
    """Return True if *closes* has already been restated for this split.

    Yahoo lists a split as soon as it is announced or effective, but the
    historical series is restated on its own schedule. The two states are
    distinguishable from the prices alone: a restated series is continuous
    across the split date, while an un-restated one steps by the split ratio.

    Args:
        closes: Close prices indexed by a timezone-naive DatetimeIndex.
        split_date: Effective date of the split.
        ratio: Split ratio as yfinance reports it (4.0 for 4-for-1, 0.05 for
            1-for-20).
        probe: Trading days either side of *split_date* to median.

    Returns:
        True when the series is continuous at the split (already restated),
        False when it steps by roughly the split ratio, and True when the
        evidence is missing, since an unverifiable split is more often one
        Yahoo has already folded in than one it has not.
    """
    if ratio <= 0 or not np.isfinite(ratio) or abs(ratio - 1.0) < 1e-9:
        return False

    before = closes[closes.index < split_date].tail(probe)
    after = closes[closes.index >= split_date].head(probe)
    if before.empty or after.empty:
        return True

    before_level = float(before.median())
    after_level = float(after.median())
    if before_level <= 0 or after_level <= 0:
        return True

    step = after_level / before_level
    # Restated: step near 1. Not restated: step near 1/ratio.
    restated_error = abs(np.log(step))
    raw_error = abs(np.log(step * ratio))
    # bool(), not the numpy scalar: callers and tests treat this as a plain bool.
    return bool(restated_error <= raw_error)


def cumulative_split_factor(
    splits: pd.Series,
    after: pd.Timestamp,
    closes: pd.Series | None = None,
) -> float:
    """Return the product of split ratios that Yahoo has applied after *after*.

    Yahoo restates historical prices for later splits, dividing by the ratio of
    a forward split and multiplying by the inverse of a reverse split. Undoing
    that means multiplying the restated price by this factor.

    Args:
        splits: Split ratios indexed by split date (2.0 for 2-for-1, 0.05 for
            1-for-20).
        after: Only splits strictly after this date are considered.
        closes: Close prices used by :func:`split_is_applied` to check whether
            each split has in fact been folded into the series. When omitted,
            every split is assumed applied.

    Returns:
        The product of applicable ratios, or 1.0 when there are none.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([10.0], index=pd.DatetimeIndex(["2024-06-10"]))
        >>> cumulative_split_factor(s, pd.Timestamp("2024-01-02"))
        10.0
    """
    if splits is None or len(splits) == 0:
        return 1.0

    idx = pd.DatetimeIndex(splits.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    values = pd.Series(np.asarray(splits, dtype=float), index=idx)

    applicable = values[(idx > after) & (values > 0)]
    if applicable.empty:
        return 1.0

    if closes is not None:
        applicable = applicable[
            [split_is_applied(closes, date, ratio) for date, ratio in applicable.items()]
        ]
    if applicable.empty:
        return 1.0
    return float(np.prod(applicable.to_numpy()))


def _naive_index(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return *frame* with a timezone-naive DatetimeIndex."""
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    return out


# ---------------------------------------------------------------------------
# Per-ticker fetch
# ---------------------------------------------------------------------------

def fetch_first_day_prices(
    ticker: str,
    ipo_date: pd.Timestamp,
    offer_price: float,
) -> dict:
    """Fetch and split-correct the first-day price record for one IPO.

    Args:
        ticker: Stock ticker symbol.
        ipo_date: Scheduled IPO date.
        offer_price: IPO offer (or reference) price in USD.

    Returns:
        Dict with ``ticker``, ``first_trade_date``, ``first_day_open``,
        ``first_day_close``, ``split_factor``, ``underpricing``,
        ``first_week_return``, ``first_month_return`` and ``status``.
        ``status == "ok"`` iff ``underpricing`` is populated.
    """
    result: dict = {"ticker": ticker, "status": "unknown"}

    try:
        # One request for the whole series. Splits that happen *after* the IPO
        # are what we need to undo, so a window around the IPO date would not
        # show them; ``period="max"`` with ``actions=True`` returns prices and
        # the split calendar together and halves the request count.
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=True)
    except Exception as exc:  # network, JSON, delisting - all terminal here
        result["status"] = f"fetch_error: {type(exc).__name__}"
        return result

    if hist is None or hist.empty:
        result["status"] = "no_data"
        return result

    hist = _naive_index(hist)

    history_start = hist.index.min()
    lead_days = (ipo_date.normalize() - history_start).days
    result["history_start"] = history_start.date().isoformat()
    if lead_days > MAX_PRE_IPO_HISTORY_DAYS:
        result["status"] = f"pre_ipo_history_{lead_days}d"
        return result

    on_or_after = hist[hist.index >= ipo_date.normalize()]
    if on_or_after.empty:
        result["status"] = "no_close_on_or_after_ipo_date"
        return result

    first_trade_date = on_or_after.index[0]
    lag_days = (first_trade_date - ipo_date.normalize()).days
    if lag_days > MAX_TRADING_DAY_LAG:
        result["status"] = f"first_trade_{lag_days}d_after_ipo_date"
        result["first_trade_date"] = first_trade_date.date().isoformat()
        return result

    splits = hist["Stock Splits"] if "Stock Splits" in hist.columns else pd.Series(dtype=float)
    splits = splits[splits > 0]
    factor = cumulative_split_factor(splits, first_trade_date, closes=hist["Close"])

    close = float(on_or_after["Close"].iloc[0]) * factor
    open_ = float(on_or_after["Open"].iloc[0]) * factor

    result.update(
        {
            "first_trade_date": first_trade_date.date().isoformat(),
            "trading_day_lag": lag_days,
            "first_day_open": open_,
            "first_day_close": close,
            "split_factor": factor,
        }
    )

    if offer_price and offer_price > 0 and np.isfinite(offer_price):
        result["underpricing"] = (close - offer_price) / offer_price
        result["status"] = "ok"
    else:
        result["status"] = "no_offer_price"

    # Post-listing horizons, measured from the first-day close. Split factors
    # cancel in a ratio of two prices inside the same restated series, so these
    # need no correction.
    base = float(on_or_after["Close"].iloc[0])
    for label, days in (("first_week_return", 7), ("first_month_return", 30)):
        later = on_or_after[on_or_after.index >= first_trade_date + pd.Timedelta(days=days)]
        if not later.empty and base != 0:
            result[label] = float(later["Close"].iloc[0]) / base - 1.0

    return result


# ---------------------------------------------------------------------------
# Market index downloads (VIX + NASDAQ)
# ---------------------------------------------------------------------------

def download_market_indices(
    start: str = "2018-10-01",
    end: str = "2025-03-31",
) -> pd.DataFrame:
    """Download daily VIX and NASDAQ Composite closing prices.

    Args:
        start: Start date string ``YYYY-MM-DD``.
        end: End date string ``YYYY-MM-DD``.

    Returns:
        DataFrame indexed by date with columns ``vix_close`` and
        ``nasdaq_close``.
    """
    log.info("Downloading VIX and NASDAQ data …")
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)["Close"]
    nasdaq = yf.download("^IXIC", start=start, end=end, progress=False, auto_adjust=False)["Close"]

    frame = pd.DataFrame(
        {
            "vix_close": np.asarray(vix).ravel(),
            "nasdaq_close": np.asarray(nasdaq).ravel(),
        },
        index=pd.DatetimeIndex(vix.index).tz_localize(None),
    )
    frame.index.name = "date"
    frame = frame.dropna(how="all")

    out_path = RAW_DIR / "market_indices.csv"
    frame.to_csv(out_path)
    log.info("Saved market indices → %s (%d rows)", out_path, len(frame))
    return frame


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_price_scraper(
    ipo_csv: Path = IPO_CSV,
    output_csv: Path = OUTPUT_CSV,
    max_tickers: Optional[int] = None,
    sleep_between: float = 0.15,
) -> pd.DataFrame:
    """Collect first-day prices for every ticker in the IPO calendar.

    Args:
        ipo_csv: Path to the IPO calendar CSV.
        output_csv: Destination CSV.
        max_tickers: Optional cap, for smoke runs.
        sleep_between: Courtesy pause between Yahoo calls, in seconds.

    Returns:
        One row per ticker with the fields described in
        :func:`fetch_first_day_prices`.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ipos = pd.read_csv(ipo_csv, parse_dates=["ipo_date"])
    ipos = ipos.dropna(subset=["ticker", "ipo_date"]).drop_duplicates(subset=["ticker", "ipo_date"])
    if max_tickers:
        ipos = ipos.head(max_tickers)

    records = []
    for i, (_, row) in enumerate(ipos.iterrows(), start=1):
        offer = pd.to_numeric(row.get("offer_price"), errors="coerce")
        rec = fetch_first_day_prices(
            str(row["ticker"]).strip().upper(),
            pd.Timestamp(row["ipo_date"]),
            float(offer) if pd.notna(offer) else float("nan"),
        )
        rec["ipo_date"] = pd.Timestamp(row["ipo_date"]).date().isoformat()
        records.append(rec)
        if i % 100 == 0:
            ok = sum(r["status"] == "ok" for r in records)
            log.info("Progress: %d / %d (%d ok)", i, len(ipos), ok)
        time.sleep(sleep_between)

    prices = pd.DataFrame(records)
    prices.to_csv(output_csv, index=False)
    log.info(
        "Price collection complete: %d / %d ok → %s",
        (prices["status"] == "ok").sum(), len(prices), output_csv,
    )
    return prices


if __name__ == "__main__":
    df = run_price_scraper()
    print(df["status"].value_counts().to_string())
