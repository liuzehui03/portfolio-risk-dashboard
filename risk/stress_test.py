"""
Historical stress scenario analysis.

Replays actual market return periods (2022 rate shock, COVID crash) against
current portfolio weights to estimate what the portfolio would have lost.
"""

import pandas as pd
import numpy as np

# Historical stress periods defined as (start, end) date strings.
# Returns over these windows are extracted from the price history and applied
# to current weights to simulate the portfolio's hypothetical loss.
STRESS_SCENARIOS = {
    "2022 Rate Shock (Jan–Oct 2022)": ("2022-01-03", "2022-10-13"),
    "COVID Crash (Feb–Mar 2020)":     ("2020-02-19", "2020-03-23"),
    "2020 Recovery (Mar–Aug 2020)":   ("2020-03-23", "2020-08-31"),
    "GFC Echo / Q4 2018":             ("2018-10-01", "2018-12-24"),
}


def scenario_returns(
    prices: pd.DataFrame,
    start: str,
    end: str,
    weights: dict,
    cash_weight: float | None = None,
    cash_rate: float = 0.0,
) -> dict:
    """
    Compute per-asset and portfolio-level return over a stress window.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history (all tickers, daily frequency).
    start : str
        Start date of stress window (YYYY-MM-DD). Requires that exact observation date.
    end : str
        End date of stress window (YYYY-MM-DD). Requires that exact observation date.
    weights : dict
        {ticker: weight} portfolio weights.

    Returns
    -------
    dict
        {ticker: total_return_over_period} plus 'portfolio' key.
    """
    from data.fetch_prices import validate_prices
    from risk.metrics import validate_weights, daily_rate
    cash = validate_weights(weights, cash_weight)
    clean = validate_prices(prices, list(weights))
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date >= end_date or start_date not in clean.index or end_date not in clean.index:
        raise ValueError(f"Full scenario endpoints unavailable: {start} to {end}")
    window = clean.loc[start:end]
    asset_returns = window.iloc[-1] / window.iloc[0] - 1
    result = {t: float(asset_returns[t]) for t in weights}
    result["CASH"] = (1 + daily_rate(cash_rate)) ** (len(window) - 1) - 1
    result["portfolio"] = float(sum(weights[t] * result[t] for t in weights)
                                + cash * result["CASH"])
    return result


def run_all_scenarios(prices: pd.DataFrame, weights: dict,
                      cash_weight: float | None = None, cash_rate: float = 0.0) -> pd.DataFrame:
    """
    Run all defined stress scenarios and return a summary table.

    For each scenario, computes the portfolio return and each position's
    individual return contribution.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history.
    weights : dict
        {ticker: weight} portfolio weights.

    Returns
    -------
    pd.DataFrame
        Rows = scenarios, columns = tickers + 'Portfolio Return'.
        Values are percentage returns (multiplied by 100).
    """
    tickers = list(weights.keys())
    rows = []

    for scenario_name, (start, end) in STRESS_SCENARIOS.items():
        try:
            ret = scenario_returns(prices, start, end, weights, cash_weight, cash_rate)
        except ValueError as exc:
            rows.append({"Scenario": scenario_name, "Portfolio Return (%)": np.nan,
                         "Status": f"Unavailable: {exc}"})
            continue

        row = {"Scenario": scenario_name, "Status": "Available"}
        for t in tickers:
            row[t] = round(ret.get(t, np.nan) * 100, 2)
        row["Portfolio Return (%)"] = round(ret.get("portfolio", np.nan) * 100, 2)
        rows.append(row)

    if not rows:
        print("  Warning: no stress scenario periods found in price history.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Scenario")
    return df


def scenario_contribution_table(
    prices: pd.DataFrame,
    weights: dict,
    scenario_name: str,
    cash_weight: float | None = None,
    cash_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Break down a single scenario's portfolio return into per-position contributions.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history.
    weights : dict
        {ticker: weight} portfolio weights.
    scenario_name : str
        Key from STRESS_SCENARIOS.

    Returns
    -------
    pd.DataFrame
        Columns: Ticker, Weight (%), Asset Return (%), Contribution (bps).
    """
    if scenario_name not in STRESS_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Valid: {list(STRESS_SCENARIOS)}")

    start, end = STRESS_SCENARIOS[scenario_name]
    ret = scenario_returns(prices, start, end, weights, cash_weight, cash_rate)
    from risk.metrics import validate_weights
    weights = {**weights, "CASH": validate_weights(weights, cash_weight)}

    rows = []
    for ticker, weight in weights.items():
        asset_ret = ret.get(ticker, np.nan)
        contribution_bps = weight * asset_ret * 10_000 if not np.isnan(asset_ret) else np.nan
        rows.append({
            "Ticker": ticker,
            "Weight (%)": round(weight * 100, 2),
            "Asset Return (%)": round(asset_ret * 100, 2) if not np.isnan(asset_ret) else np.nan,
            "Contribution (bps)": round(contribution_bps, 1) if not np.isnan(contribution_bps) else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("Contribution (bps)")
    return df
