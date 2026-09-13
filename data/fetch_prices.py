"""
Fetch and cache 2 years of daily adjusted closing prices for all portfolio tickers.
"""

import time
import yfinance as yf
import pandas as pd
from pathlib import Path

# Weights are transcribed from data/GPP2026_Complete_Analysis_exCAT.xlsx,
# sheet "Position PL Tracker". CAT was never held (see "CAT Removal Log").
EQUITY_TICKERS = ["AAPL", "MSFT", "MU", "WMT", "DAL", "IAG", "SPCX"]
BROAD_ETF_TICKERS = ["SPY", "VT", "XLV", "XLF", "EEM"]
FIXED_INCOME_TICKERS = ["VGIT", "VTIP", "JPIE", "MINT"]

ASSET_CLASSES = {
    "Equities":          EQUITY_TICKERS,
    "Broad ETFs":        BROAD_ETF_TICKERS,
    "Fixed Income ETFs": FIXED_INCOME_TICKERS,
}

TICKERS = EQUITY_TICKERS + BROAD_ETF_TICKERS + FIXED_INCOME_TICKERS

WEIGHTS = {
    'SPY': 0.14, 'VT': 0.10, 'VGIT': 0.10,
    'MU': 0.10, 'XLV': 0.06, 'JPIE': 0.06,
    'MSFT': 0.06, 'VTIP': 0.06, 'AAPL': 0.05,
    'XLF': 0.05, 'MINT': 0.045, 'WMT': 0.04,
    'EEM': 0.03, 'SPCX': 0.03,
    'DAL': 0.015, 'IAG': 0.005
}

# SPCX listed recently and has only weeks of trading history. Including it in
# a straight dropna() would truncate the ENTIRE price table (all 16 tickers)
# down to just its short window. It's kept in WEIGHTS for documentation but
# excluded here so the rest of the portfolio retains full multi-year history.
SHORT_HISTORY_TICKERS = ["SPCX"]
BACKTEST_WEIGHTS = {t: w for t, w in WEIGHTS.items() if t not in SHORT_HISTORY_TICKERS}

DATA_DIR = Path(__file__).parent


def fetch_prices(period: str = "2y") -> pd.DataFrame:
    """
    Download adjusted closing prices for all tickers via yfinance.

    Parameters
    ----------
    period : str
        yfinance period string (default '2y' for two years).

    Returns
    -------
    pd.DataFrame
        Daily adjusted close prices, columns = tickers, index = date.
        Rows are forward-filled, then dropped only where a long-history
        ticker is still missing — SHORT_HISTORY_TICKERS (e.g. SPCX, which
        listed recently) are left with leading NaNs rather than truncating
        everyone else's history down to their short window.
    """
    required_cols = [t for t in TICKERS if t not in SHORT_HISTORY_TICKERS]
    max_attempts = 3
    missing = required_cols

    for attempt in range(1, max_attempts + 1):
        raw = yf.download(
            TICKERS,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        # yfinance returns MultiIndex columns when multiple tickers requested
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = TICKERS

        prices = prices.ffill()
        missing = [c for c in required_cols if prices[c].isna().all()]
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

    prices = prices.dropna(subset=required_cols)
    return prices


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

    return prices


if __name__ == "__main__":
    df = load_prices(refresh=True)
    print(f"\nPrice data shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(df.tail())
