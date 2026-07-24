"""
Return attribution and correlation analysis.

Decomposes total portfolio return into each position's contribution,
and computes the full correlation matrix across all holdings.
"""

import pandas as pd
import numpy as np


def return_attribution(prices: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Compute each position's contribution to total portfolio return.

    Contribution = position weight × asset total return over the period.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices (columns = tickers).
    weights : dict
        {ticker: weight} mapping.

    Returns
    -------
    pd.DataFrame
        Sorted by contribution descending. Columns:
        Ticker, Weight (%), Total Return (%), Contribution (%), Contribution (bps).
    """
    tickers = [t for t in weights if t in prices.columns]
    total_returns = (prices[tickers].iloc[-1] / prices[tickers].iloc[0]) - 1

    rows = []
    for ticker in tickers:
        weight = weights[ticker]
        asset_ret = float(total_returns[ticker])
        contribution = weight * asset_ret
        rows.append({
            "Ticker": ticker,
            "Weight (%)": round(weight * 100, 2),
            "Total Return (%)": round(asset_ret * 100, 2),
            "Contribution (%)": round(contribution * 100, 2),
            "Contribution (bps)": round(contribution * 10_000, 1),
        })

    df = pd.DataFrame(rows).sort_values("Contribution (%)", ascending=False)
    df = df.reset_index(drop=True)

    # Append totals row
    totals = {
        "Ticker": "TOTAL",
        "Weight (%)": round(sum(r["Weight (%)"] for r in rows), 2),
        "Total Return (%)": "",
        "Contribution (%)": round(sum(r["Contribution (%)"] for r in rows), 2),
        "Contribution (bps)": round(sum(r["Contribution (bps)"] for r in rows), 1),
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    return df


def correlation_matrix(prices: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation of daily returns across all positions.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices.
    weights : dict
        Used to select and order the tickers consistently.

    Returns
    -------
    pd.DataFrame
        Correlation matrix (tickers × tickers), values in [-1, 1].
    """
    tickers = [t for t in weights if t in prices.columns]
    daily_returns = prices[tickers].pct_change().dropna()
    return daily_returns.corr()


def rolling_correlation(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 60,
) -> pd.Series:
    """
    Rolling Pearson correlation between two tickers.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices.
    ticker_a : str
        First ticker.
    ticker_b : str
        Second ticker.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        Rolling correlation series indexed by date.
    """
    ret_a = prices[ticker_a].pct_change()
    ret_b = prices[ticker_b].pct_change()
    return ret_a.rolling(window).corr(ret_b)


def asset_class_attribution(prices: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Aggregate return attribution by asset class sleeve.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices.
    weights : dict
        {ticker: weight} mapping.

    Returns
    -------
    pd.DataFrame
        Aggregated contribution by asset class.
    """
    asset_classes = {
        "Equities":          ["AAPL", "MSFT", "MU", "WMT", "DAL", "IAG", "CAT", "SPCX"],
        "Broad ETFs":        ["SPY", "VT", "XLV", "XLF", "EEM"],
        "Fixed Income ETFs": ["VGIT", "VTIP", "JPIE", "MINT"],
    }

    attr = return_attribution(prices, weights)
    # Drop totals row for aggregation
    attr = attr[attr["Ticker"] != "TOTAL"].copy()

    rows = []
    for asset_class, tickers in asset_classes.items():
        subset = attr[attr["Ticker"].isin(tickers)]
        rows.append({
            "Asset Class": asset_class,
            "Weight (%)": round(subset["Weight (%)"].sum(), 2),
            "Contribution (%)": round(subset["Contribution (%)"].sum(), 2),
            "Contribution (bps)": round(subset["Contribution (bps)"].sum(), 1),
        })

    return pd.DataFrame(rows)
