"""Fetch and validate daily adjusted prices; default window is five years."""

import time
import warnings
import numpy as np
import yfinance as yf
import pandas as pd
from pathlib import Path

# Weights are transcribed from data/GPP2026_Complete_Analysis_exCAT.xlsx,
# sheet "Position PL Tracker". CAT was never held (see "CAT Removal Log").
EQUITY_TICKERS = ["AAPL", "MSFT", "MU", "WMT", "DAL", "IAG"]
BROAD_ETF_TICKERS = ["SPY", "VT", "XLV", "XLF", "EEM"]
FIXED_INCOME_TICKERS = ["VGIT", "VTIP", "JPIE", "MINT"]

ASSET_CLASSES = {
    "Equities":          EQUITY_TICKERS,
    "Broad ETFs":        BROAD_ETF_TICKERS,
    "Fixed Income ETFs": FIXED_INCOME_TICKERS,
}

TICKERS = EQUITY_TICKERS + BROAD_ETF_TICKERS + FIXED_INCOME_TICKERS

# Benchmark proxies, downloaded alongside the positions but never held:
#   URTH  iShares MSCI World ETF        -> 60% MSCI World leg
#   BNDW  Vanguard Total World Bond ETF -> 40% Bloomberg Global Agg leg
BENCHMARK_TICKERS = ["URTH", "BNDW"]
DOWNLOAD_TICKERS = TICKERS + BENCHMARK_TICKERS

ORIGINAL_WEIGHTS = {
    'SPY': 0.14, 'VT': 0.10, 'VGIT': 0.10,
    'MU': 0.10, 'XLV': 0.06, 'JPIE': 0.06,
    'MSFT': 0.06, 'VTIP': 0.06, 'AAPL': 0.05,
    'XLF': 0.05, 'MINT': 0.045, 'WMT': 0.04,
    'EEM': 0.03, 'SPCX': 0.03,
    'DAL': 0.015, 'IAG': 0.005
}

# Revised simulation allocation approved 2026-09-23, not live market-value weights.
WEIGHTS_DATE = "2026-09-23"
CASH_WEIGHT = 0.01
CASH_RATE = 0.0  # annual nominal rate, accrued at rate / 252 per observation
WEIGHTS = {t: w * (1 - CASH_WEIGHT) / sum(
    v for k, v in ORIGINAL_WEIGHTS.items() if k != "SPCX"
) for t, w in ORIGINAL_WEIGHTS.items() if t != "SPCX"}


def validate_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Select required assets; allow only leading pre-inception missing prices."""
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Prices require valid datetime dates")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted")
    if prices.columns.has_duplicates:
        raise ValueError("Price columns must be unique")
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    frame = prices.loc[:, tickers].astype(float)
    first_dates = []
    for ticker in tickers:
        series = frame[ticker]
        first = series.first_valid_index()
        if first is None:
            raise ValueError(f"No prices for {ticker}")
        active = series.loc[first:]
        if not np.isfinite(active).all() or (active <= 0).any():
            raise ValueError(f"Invalid, internal or trailing missing prices for {ticker}")
        first_dates.append(first)
    frame = frame.loc[max(first_dates):]
    if len(frame) < 2:
        raise ValueError("At least two common price observations are required")
    return frame


def warn_if_stale(prices: pd.DataFrame, today=None) -> None:
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).date()
    last = prices.index[-1].date()
    age = int(np.busday_count(last, today)) if today >= last else 0
    if age > 3:
        warnings.warn(f"Stale price cache: latest date {last}, {age} weekdays old. "
                      "Run with --refresh for updated daily prices.", UserWarning, stacklevel=2)


DATA_DIR = Path(__file__).parent


def fetch_prices(period: str = "5y") -> pd.DataFrame:
    """
    Download adjusted closing prices for all positions and benchmark proxies
    via yfinance.

    Parameters
    ----------
    period : str
        yfinance period string (default '5y').

    Returns
    -------
    pd.DataFrame
        Daily adjusted close prices, columns = tickers, index = date.
        Leading pre-inception gaps determine the common start date.
        Internal/trailing missing prices are rejected, never forward-filled.
    """
    required_cols = DOWNLOAD_TICKERS
    max_attempts = 3
    missing = required_cols

    for attempt in range(1, max_attempts + 1):
        raw = yf.download(
            DOWNLOAD_TICKERS,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            prices = pd.DataFrame(index=raw.index)
        missing = [c for c in required_cols if c not in prices or prices[c].isna().all()]
        if not missing:
            break

        if attempt < max_attempts:
            print(f"  Retry {attempt}/{max_attempts - 1}: no data for {', '.join(missing)} "
                  f"(likely Yahoo Finance rate-limiting) — retrying in {attempt * 3}s...")
            time.sleep(attempt * 3)

    if missing:
        raise RuntimeError(
            f"No price data returned for: {', '.join(missing)} after {max_attempts} attempts. "
            "Yahoo Finance may be rate-limiting or temporarily unavailable "
            "— wait a moment and retry. If it persists, check the ticker symbols."
        )

    return validate_prices(prices, required_cols)


def load_prices(refresh: bool = False, period: str = "5y") -> pd.DataFrame:
    """
    Load prices from local CSV cache, fetching fresh data if cache is absent
    or refresh is requested.

    Parameters
    ----------
    refresh : bool
        If True, always re-download even if cache exists.
    period : str
        yfinance period string passed to fetch_prices (default '5y').
        5 years is used so historical stress scenarios (2022 rate shock) are
        within the available price history.

    Returns
    -------
    pd.DataFrame
        Daily adjusted close prices.
    """
    cache_path = DATA_DIR / "prices.csv"

    if cache_path.exists() and not refresh:
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded prices from cache ({cache_path})")
    else:
        print("Downloading prices from Yahoo Finance...")
        prices = fetch_prices(period=period)
        prices.to_csv(cache_path)
        print(f"Saved prices to {cache_path}")

    prices = validate_prices(prices, DOWNLOAD_TICKERS)
    warn_if_stale(prices)
    return prices


if __name__ == "__main__":
    df = load_prices(refresh=True)
    print(f"\nPrice data shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(df.tail())
