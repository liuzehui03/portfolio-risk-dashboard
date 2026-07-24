"""
Fetch and cache 2 years of daily adjusted closing prices for all portfolio tickers.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = [
    # Equities
    "AAPL", "MSFT", "MU", "WMT", "DAL", "IAG", "CAT", "SPCX",
    # Broad ETFs
    "SPY", "VT", "XLV", "XLF", "EEM",
    # Fixed Income ETFs
    "VGIT", "VTIP", "JPIE", "MINT",
]

WEIGHTS = {
    'SPY': 0.10, 'VT': 0.10, 'VGIT': 0.10,
    'MU': 0.09, 'XLV': 0.06, 'JPIE': 0.06,
    'MSFT': 0.06, 'VTIP': 0.06, 'AAPL': 0.05,
    'XLF': 0.05, 'MINT': 0.045, 'WMT': 0.04,
    'CAT': 0.04, 'EEM': 0.03, 'SPCX': 0.03,
    'DAL': 0.015, 'IAG': 0.005
}

FIXED_INCOME_TICKERS = ["VGIT", "VTIP", "JPIE", "MINT"]

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
        Rows with any NaN are forward-filled then dropped.
    """
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

    prices = prices.ffill().dropna()
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
