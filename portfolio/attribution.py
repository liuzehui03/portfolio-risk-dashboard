"""
Return attribution and correlation analysis.

Decomposes total portfolio return into each position's contribution,
and computes the full correlation matrix across all holdings.
"""

import pandas as pd
import numpy as np

from data.fetch_prices import ASSET_CLASSES


def return_attribution(prices: pd.DataFrame, weights: dict,
                       cash_weight: float | None = None, cash_rate: float = 0.0) -> pd.DataFrame:
    """Wealth-linked daily P&L contributions; sums to compounded portfolio return.

    Returned values retain precision; round only when displaying them.
    """
    from data.fetch_prices import validate_prices
    from risk.metrics import portfolio_returns, validate_weights, daily_rate
    cash = validate_weights(weights, cash_weight)
    clean = validate_prices(prices, list(weights))
    returns = clean.pct_change(fill_method=None).iloc[1:]
    port = portfolio_returns(clean, weights, cash, cash_rate)
    prior_wealth = (1 + port).cumprod().shift(1, fill_value=1.0)
    contributions = returns.mul(pd.Series(weights)).mul(prior_wealth, axis=0).sum()
    total_returns = clean.iloc[-1] / clean.iloc[0] - 1
    all_weights = dict(weights)
    if cash > 0:
        all_weights["CASH"] = cash
        contributions["CASH"] = (prior_wealth * cash * daily_rate(cash_rate)).sum()
        total_returns["CASH"] = (1 + daily_rate(cash_rate)) ** len(port) - 1
    rows = [{
        "Ticker": ticker, "Weight (%)": weight * 100,
        "Total Return (%)": total_returns[ticker] * 100,
        "Contribution (%)": contributions[ticker] * 100,
        "Contribution (bps)": contributions[ticker] * 10_000,
    } for ticker, weight in all_weights.items()]
    df = pd.DataFrame(rows).sort_values("Contribution (%)", ascending=False)
    total = {"Ticker": "TOTAL", "Weight (%)": sum(all_weights.values()) * 100,
             "Total Return (%)": np.nan,
             "Contribution (%)": contributions.sum() * 100,
             "Contribution (bps)": contributions.sum() * 10_000}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


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
    from data.fetch_prices import validate_prices
    daily_returns = validate_prices(prices, list(weights)).pct_change(fill_method=None).iloc[1:]
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
    from data.fetch_prices import validate_prices
    clean = validate_prices(prices, [ticker_a, ticker_b])
    ret_a = clean[ticker_a].pct_change(fill_method=None)
    ret_b = clean[ticker_b].pct_change(fill_method=None)
    return ret_a.rolling(window).corr(ret_b)


def asset_class_attribution(prices: pd.DataFrame, weights: dict,
                            cash_weight: float | None = None, cash_rate: float = 0.0) -> pd.DataFrame:
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
    attr = return_attribution(prices, weights, cash_weight, cash_rate)
    # Drop totals row for aggregation
    attr = attr[attr["Ticker"] != "TOTAL"].copy()

    rows = []
    for asset_class, tickers in {**ASSET_CLASSES, "Cash": ["CASH"]}.items():
        subset = attr[attr["Ticker"].isin(tickers)]
        rows.append({
            "Asset Class": asset_class,
            "Weight (%)": subset["Weight (%)"].sum(),
            "Contribution (%)": subset["Contribution (%)"].sum(),
            "Contribution (bps)": subset["Contribution (bps)"].sum(),
        })

    return pd.DataFrame(rows)
