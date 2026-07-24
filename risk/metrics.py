"""
Core portfolio risk metrics: VaR, CVaR, Sharpe, Sortino, max drawdown,
and rolling volatility.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(prices: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Compute daily portfolio returns as the weighted sum of asset returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices (columns = tickers).
    weights : dict
        {ticker: weight} mapping. Weights should sum to 1.

    Returns
    -------
    pd.Series
        Daily portfolio returns indexed by date.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    asset_returns = prices[tickers].pct_change().dropna()
    port_ret = asset_returns.values @ w
    return pd.Series(port_ret, index=asset_returns.index, name="portfolio")


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (non-parametric) Value at Risk.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
        Confidence level, e.g. 0.95 for 95% VaR.

    Returns
    -------
    float
        VaR as a positive number (loss expressed as a fraction of portfolio value).
    """
    return float(-np.percentile(returns, (1 - confidence) * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Conditional Value at Risk (Expected Shortfall).

    CVaR is the mean of losses that exceed the VaR threshold, giving
    a more complete picture of tail risk than VaR alone.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
        Confidence level, e.g. 0.95.

    Returns
    -------
    float
        CVaR as a positive number.
    """
    var = historical_var(returns, confidence)
    tail_losses = returns[returns < -var]
    return float(-tail_losses.mean()) if len(tail_losses) > 0 else var


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    risk_free_rate : float
        Annual risk-free rate (default 4%, approximate current T-bill yield).

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Annualised Sortino ratio — like Sharpe but penalises only downside volatility.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    float
        Annualised Sortino ratio.
    """
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    downside = excess[excess < 0]
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return np.nan
    return float((excess.mean() / downside_std) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> tuple[float, pd.Series]:
    """
    Maximum drawdown of a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.

    Returns
    -------
    tuple[float, pd.Series]
        (max_drawdown_fraction, drawdown_series)
        max_drawdown is expressed as a positive fraction (e.g. 0.15 = 15% loss).
        drawdown_series shows the drawdown at each point in time.
    """
    cumulative = (1 + returns).cumprod()
    rolling_peak = cumulative.cummax()
    drawdown = (cumulative - rolling_peak) / rolling_peak
    return float(drawdown.min()), drawdown


def rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling annualised volatility.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    window : int
        Rolling window in trading days (default 30).

    Returns
    -------
    pd.Series
        Annualised rolling volatility.
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def annualised_volatility(returns: pd.Series) -> float:
    """
    Full-period annualised volatility.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.

    Returns
    -------
    float
        Annualised standard deviation.
    """
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def annualised_return(returns: pd.Series) -> float:
    """
    Compound annualised return (CAGR).

    Parameters
    ----------
    returns : pd.Series
        Daily return series.

    Returns
    -------
    float
        Annualised return.
    """
    n_years = len(returns) / TRADING_DAYS
    if n_years == 0:
        return 0.0
    total = (1 + returns).prod()
    return float(total ** (1 / n_years) - 1)


def risk_summary(
    port_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Compute all key risk metrics for the portfolio and benchmark.

    Parameters
    ----------
    port_returns : pd.Series
        Daily portfolio returns.
    benchmark_returns : pd.Series
        Daily benchmark (SPY) returns aligned to the same dates.
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    dict
        Nested dict with keys 'portfolio' and 'benchmark', each containing
        all computed risk metrics.
    """
    def _metrics(ret, label):
        mdd, _ = max_drawdown(ret)
        return {
            "label": label,
            "ann_return": annualised_return(ret),
            "ann_vol": annualised_volatility(ret),
            "sharpe": sharpe_ratio(ret, risk_free_rate),
            "sortino": sortino_ratio(ret, risk_free_rate),
            "var_95": historical_var(ret, 0.95),
            "var_99": historical_var(ret, 0.99),
            "cvar_95": historical_cvar(ret, 0.95),
            "cvar_99": historical_cvar(ret, 0.99),
            "max_drawdown": mdd,
        }

    aligned = port_returns.align(benchmark_returns, join="inner")
    return {
        "portfolio": _metrics(aligned[0], "Portfolio"),
        "benchmark": _metrics(aligned[1], "SPY Benchmark"),
    }
