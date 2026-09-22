"""
Core portfolio risk metrics: VaR, CVaR, Sharpe, Sortino, max drawdown,
and rolling volatility.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def validate_weights(weights: dict, cash_weight: float | None = None) -> float:
    """Long-only weights; omitted cash is the zero-return residual for compatibility."""
    values = np.array(list(weights.values()), dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Weights must be nonempty, finite and nonnegative")
    cash = 1 - values.sum() if cash_weight is None else cash_weight
    if not np.isfinite(cash) or cash < -1e-12 or not np.isclose(values.sum() + cash, 1, atol=1e-12, rtol=0):
        raise ValueError("Security weights plus cash must equal 1")
    return max(0.0, float(cash))


def daily_rate(annual_rate: float) -> float:
    """Nominal annual assumption, consistently divided by 252 observations."""
    if not np.isfinite(annual_rate) or annual_rate <= -1:
        raise ValueError("Annual rate must be finite and greater than -1")
    return annual_rate / TRADING_DAYS


def validate_returns(returns: pd.Series) -> None:
    if returns.empty or not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError("Returns must be nonempty, finite and at least -100%")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be unique and sorted")


def portfolio_returns(prices: pd.DataFrame, weights: dict,
                      cash_weight: float | None = None, cash_rate: float = 0.0) -> pd.Series:
    """Historical daily scenarios at fixed weights; compounding implies daily rebalancing."""
    from data.fetch_prices import validate_prices
    cash = validate_weights(weights, cash_weight)
    clean = validate_prices(prices, list(weights))
    asset_returns = clean.pct_change(fill_method=None).iloc[1:]
    result = asset_returns @ pd.Series(weights) + cash * daily_rate(cash_rate)
    result.name = "portfolio"
    return result


# Proxy for the GPP2026 mandate benchmark (60% MSCI World / 40% Bloomberg
# Global Agg): 60% URTH (MSCI World ETF) + 40% BNDW (Total World Bond ETF),
# rebalanced daily. Both are total-return series, like the portfolio.
BLEND_WEIGHTS = {"URTH": 0.60, "BNDW": 0.40}


def blended_benchmark_returns(prices: pd.DataFrame, weights: dict = BLEND_WEIGHTS) -> pd.Series:
    """
    Daily returns of a fixed-weight blended benchmark.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices; must contain every ticker in `weights`.
    weights : dict
        {ticker: weight} mapping, default 0.60 × URTH + 0.40 × BNDW.

    Returns
    -------
    pd.Series
        Daily blended returns, named "60/40 Blend".
    """
    blend = portfolio_returns(prices, weights)
    blend.name = "60/40 Blend"
    return blend


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
        Loss quantile as a fraction; negative values indicate a gain at the quantile.
    """
    validate_returns(returns)
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between 0 and 1")
    return float(-np.percentile(returns, (1 - confidence) * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Conditional Value at Risk (Expected Shortfall).

    CVaR averages exactly the worst (1-confidence) fraction of observations,
    including fractional probability mass at the boundary.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
        Confidence level, e.g. 0.95.

    Returns
    -------
    float
        Average tail loss as a fraction; may be negative for all-gain samples.
    """
    historical_var(returns, confidence)  # validate sample and confidence
    losses = -np.sort(returns.to_numpy(dtype=float))
    mass = (1 - confidence) * len(losses)
    whole = int(np.floor(mass))
    fraction = mass - whole
    tail_sum = losses[:whole].sum()
    if fraction > 0 and whole < len(losses):
        tail_sum += fraction * losses[whole]
    return float(tail_sum / mass)


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
    validate_returns(returns)
    daily_rf = daily_rate(risk_free_rate)
    excess = returns - daily_rf
    std = excess.std()
    if not np.isfinite(std) or std <= np.finfo(float).eps:
        return np.nan
    return float((excess.mean() / std) * np.sqrt(TRADING_DAYS))


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
    validate_returns(returns)
    daily_rf = daily_rate(risk_free_rate)
    excess = returns - daily_rf
    downside = excess.clip(upper=0)
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
        max_drawdown is a negative fraction (e.g. -0.15 = 15% loss).
        drawdown_series shows the drawdown at each point in time.
    """
    validate_returns(returns)
    cumulative = (1 + returns).cumprod()
    rolling_peak = cumulative.cummax().clip(lower=1.0)
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
    validate_returns(returns)
    if window < 2:
        raise ValueError("Rolling window must be at least 2")
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
    validate_returns(returns)
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
    validate_returns(returns)
    n_years = len(returns) / TRADING_DAYS
    total = (1 + returns).prod()
    return float(total ** (1 / n_years) - 1)


def risk_summary(
    port_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.04,
    blend_returns: pd.Series | None = None,
) -> dict:
    """
    Compute all key risk metrics for the portfolio and its benchmarks.

    Parameters
    ----------
    port_returns : pd.Series
        Daily portfolio returns.
    benchmark_returns : pd.Series
        Daily SPY returns.
    risk_free_rate : float
        Annual risk-free rate.
    blend_returns : pd.Series, optional
        Daily 60/40 blended benchmark returns (see `blended_benchmark_returns`).

    Returns
    -------
    dict
        Nested dict with keys 'portfolio' and 'benchmark', plus 'blend' when
        `blend_returns` is given. All series are aligned to their common dates
        before any metric is computed.
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

    series = {"portfolio": port_returns, "benchmark": benchmark_returns}
    if blend_returns is not None:
        series["blend"] = blend_returns

    for ret in series.values():
        validate_returns(ret)
    common = pd.concat(series, axis=1, join="inner")
    labels = {"portfolio": "Portfolio", "benchmark": "SPY", "blend": "60/40 Blend"}
    return {key: _metrics(common[key], labels[key]) for key in series}
