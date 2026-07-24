"""
Fixed income risk metrics: modified duration, DV01, convexity, and
parallel yield curve shock P&L estimation for the portfolio's bond ETFs.

Note: ETF-level duration/convexity figures are approximated from published
fund data since intraday NAV decomposition is not available via yfinance.
These are point-in-time estimates and should be refreshed against fund
fact sheets periodically.
"""

import numpy as np
import pandas as pd

# Published approximate effective duration (years) and convexity for each
# fixed income ETF as of mid-2024. Update from fund fact sheets as needed.
# Sources: Vanguard, JPMorgan, PIMCO fund pages.
FI_PROFILES = {
    "VGIT": {"duration": 5.2,  "convexity": 0.35, "ytm": 0.046},  # Vanguard Intermediate Treasury
    "VTIP": {"duration": 2.6,  "convexity": 0.10, "ytm": 0.024},  # Vanguard Short-Term TIPS
    "JPIE": {"duration": 3.8,  "convexity": 0.20, "ytm": 0.065},  # JPMorgan Income ETF
    "MINT": {"duration": 0.35, "convexity": 0.01, "ytm": 0.053},  # PIMCO Enhanced Short Maturity
}


def price_change_from_shock(duration: float, convexity: float, dy: float) -> float:
    """
    Estimate percentage price change of a bond/ETF from a parallel yield shift.

    Uses the standard duration-convexity approximation:
        ΔP/P ≈ -D * Δy + 0.5 * C * (Δy)²

    Parameters
    ----------
    duration : float
        Modified duration in years.
    convexity : float
        Convexity (unitless, scaled so Δy is in decimal form).
    dy : float
        Yield change in decimal form (e.g. 0.01 for +100 bps).

    Returns
    -------
    float
        Estimated fractional price change (negative = price falls when yields rise).
    """
    return -duration * dy + 0.5 * convexity * dy**2


def dv01(duration: float, price: float = 100.0) -> float:
    """
    Dollar Value of a 1 basis point (0.01%) move in yield per $100 face value.

    DV01 = Duration × Price × 0.0001

    Parameters
    ----------
    duration : float
        Modified duration in years.
    price : float
        Current clean price (default 100 for par).

    Returns
    -------
    float
        DV01 in dollars per $100 face.
    """
    return duration * price * 0.0001


def yield_shock_pnl(
    weights: dict,
    portfolio_value: float = 1_000_000,
    shocks_bps: list[int] | None = None,
) -> pd.DataFrame:
    """
    Estimate P&L impact of parallel yield curve shocks on fixed income positions.

    Parameters
    ----------
    weights : dict
        Full portfolio weights {ticker: weight}.
    portfolio_value : float
        Total portfolio value in dollars (default $1,000,000).
    shocks_bps : list[int]
        Basis point shocks to apply (default: [50, 100, 200]).

    Returns
    -------
    pd.DataFrame
        Table with columns: ticker, weight, allocation_$, duration,
        DV01_per_$1M, and one column per shock scenario showing P&L in $.
    """
    if shocks_bps is None:
        shocks_bps = [50, 100, 200]

    rows = []
    for ticker, profile in FI_PROFILES.items():
        weight = weights.get(ticker, 0.0)
        allocation = weight * portfolio_value
        dur = profile["duration"]
        conv = profile["convexity"]
        dv01_val = dv01(dur) * allocation / 100  # scaled to actual allocation

        row = {
            "Ticker": ticker,
            "Weight (%)": round(weight * 100, 2),
            "Allocation ($)": round(allocation, 0),
            "Duration (yrs)": dur,
            "DV01 ($)": round(dv01_val, 2),
        }

        for bps in shocks_bps:
            dy = bps / 10_000  # convert bps to decimal
            pct_chg = price_change_from_shock(dur, conv, dy)
            pnl = pct_chg * allocation
            row[f"+{bps}bps P&L ($)"] = round(pnl, 0)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Append a totals row for each shock column
    totals = {"Ticker": "TOTAL", "Weight (%)": "", "Allocation ($)": "", "Duration (yrs)": "", "DV01 ($)": ""}
    for bps in shocks_bps:
        col = f"+{bps}bps P&L ($)"
        totals[col] = round(df[col].sum(), 0)
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    return df


def portfolio_fi_duration(weights: dict) -> float:
    """
    Compute the weighted average duration of the fixed income sleeve.

    Parameters
    ----------
    weights : dict
        Full portfolio weights {ticker: weight}.

    Returns
    -------
    float
        Weighted average duration of FI positions in years.
    """
    total_fi_weight = sum(weights.get(t, 0) for t in FI_PROFILES)
    if total_fi_weight == 0:
        return 0.0
    weighted_dur = sum(
        weights.get(t, 0) * FI_PROFILES[t]["duration"] for t in FI_PROFILES
    )
    return weighted_dur / total_fi_weight
