"""
Portfolio Risk Dashboard — main entry point.

Runs the full risk pipeline end-to-end:
  1. Fetch / load price data
  2. Compute portfolio and benchmark returns
  3. Calculate all risk metrics (VaR, CVaR, Sharpe, Sortino, drawdown, vol)
  4. Run fixed income yield shock analysis
  5. Run historical stress scenarios
  6. Compute return attribution and correlation
  7. Generate all charts
  8. Print a formatted risk summary report

Usage:
    python main.py              # uses cached prices if available
    python main.py --refresh    # forces fresh download from Yahoo Finance
"""

import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate


# ── Project imports ──────────────────────────────────────────────────────────
from data.fetch_prices import load_prices, WEIGHTS, CASH_WEIGHT, CASH_RATE, WEIGHTS_DATE
from risk.metrics import (
    portfolio_returns,
    blended_benchmark_returns,
    rolling_volatility,
    max_drawdown,
    risk_summary,
)
from risk.fixed_income import yield_shock_pnl, portfolio_fi_duration, FI_PROFILES
from risk.stress_test import run_all_scenarios, scenario_contribution_table
from portfolio.attribution import (
    return_attribution,
    correlation_matrix,
    asset_class_attribution,
)
from visualisation.charts import (
    plot_correlation_heatmap,
    plot_rolling_volatility,
    plot_drawdown,
    plot_var_distribution,
    plot_return_attribution,
    plot_yield_shock,
)

PORTFOLIO_VALUE = 1_000_000  # $1M notional for dollar P&L calculations
RISK_FREE_RATE = 0.04        # fixed annual nominal assumption, not a live T-bill yield


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(x, decimals=2):
    """Format a decimal as a percentage string."""
    if pd.isna(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x * 100:.{decimals}f}%"
    return str(x)


def _separator(char="═", width=72):
    print(char * width)


def _section(title: str):
    print()
    _separator()
    print(f"  {title}")
    _separator()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(refresh: bool = False, cash_rate: float = CASH_RATE) -> None:
    print("\n" + "═" * 72)
    print("  PORTFOLIO RISK ANALYTICS DASHBOARD")
    print(f"  {len(WEIGHTS)}-Position Multi-Asset Portfolio | GIC Bootcamp Simulation")
    print("═" * 72)

    # ── 1. Load prices ────────────────────────────────────────────────────────
    prices = load_prices(refresh=refresh)

    # ── 2. Compute returns ────────────────────────────────────────────────────
    print(f"  Revised simulation weights as of {WEIGHTS_DATE}: 99% securities + 1% cash")
    print("  Coverage: all 15 held securities; no holdings excluded")
    print(f"  Prices: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Cash interest: {cash_rate:.2%}; risk-free assumption: {RISK_FREE_RATE:.2%} annually")
    print("  Nominal rates accrue at annual rate / 252 per daily observation")
    print("  Historical fixed-weight scenarios; compounded performance assumes daily rebalancing")
    print("  USD simulation; no transaction costs; not actual account performance")
    port_ret = portfolio_returns(prices, WEIGHTS, CASH_WEIGHT, cash_rate)
    bench_ret = prices["SPY"].pct_change(fill_method=None).dropna()
    blend_ret = blended_benchmark_returns(prices)   # 0.60 × URTH + 0.40 × BNDW

    # Align all three to common dates
    common = pd.concat([port_ret, bench_ret, blend_ret], axis=1, join="inner")
    port_ret, bench_ret, blend_ret = (common.iloc[:, i] for i in range(3))

    print(f"  Common daily return observations: {len(port_ret)}")

    # ── 3. Risk metrics ───────────────────────────────────────────────────────
    summary = risk_summary(port_ret, bench_ret, RISK_FREE_RATE, blend_returns=blend_ret)
    p = summary["portfolio"]
    bl = summary["blend"]
    b = summary["benchmark"]

    _section("RISK METRICS SUMMARY")
    cols = [p, bl, b]
    def _row(label, key, fmt=_pct):
        return [label, *(fmt(c[key]) for c in cols)]

    metrics_table = [
        ["Metric", "Portfolio", "60/40 Blend", "SPY"],
        ["─" * 25, "─" * 14, "─" * 14, "─" * 14],
        _row("Annualised Return",     "ann_return"),
        _row("Annualised Volatility", "ann_vol"),
        _row("Sharpe Ratio",          "sharpe",  lambda x: "n/a" if not np.isfinite(x) else f"{x:.3f}"),
        _row("Sortino Ratio",         "sortino", lambda x: "n/a" if not np.isfinite(x) else f"{x:.3f}"),
        _row("VaR 95% (1-day)",       "var_95"),
        _row("VaR 99% (1-day)",       "var_99"),
        _row("CVaR 95% (1-day)",      "cvar_95"),
        _row("CVaR 99% (1-day)",      "cvar_99"),
        _row("Max Drawdown",          "max_drawdown"),
    ]
    print(tabulate(metrics_table, headers="firstrow", tablefmt="plain"))
    print("\n  60/40 Blend = 0.60 × URTH + 0.40 × BNDW daily returns "
          "(MSCI World ETF / Total World Bond ETF, proxy for 60% MSCI World / 40% Global Agg)")

    # Dollar-equivalent VaR/CVaR
    print(f"\n  Portfolio value assumed: ${PORTFOLIO_VALUE:,.0f}")
    print(f"  VaR 95% ($):  ${p['var_95'] * PORTFOLIO_VALUE:,.0f}")
    print(f"  VaR 99% ($):  ${p['var_99'] * PORTFOLIO_VALUE:,.0f}")
    print(f"  CVaR 95% ($): ${p['cvar_95'] * PORTFOLIO_VALUE:,.0f}")

    # ── 4. Return attribution ─────────────────────────────────────────────────
    _section("RETURN ATTRIBUTION BY POSITION")
    attr_df = return_attribution(prices, WEIGHTS, CASH_WEIGHT, cash_rate)
    print(tabulate(attr_df.fillna(""), floatfmt=".2f", headers="keys", tablefmt="plain", showindex=False))

    _section("RETURN ATTRIBUTION BY ASSET CLASS")
    ac_df = asset_class_attribution(prices, WEIGHTS, CASH_WEIGHT, cash_rate)
    print(tabulate(ac_df, floatfmt=".2f", headers="keys", tablefmt="plain", showindex=False))

    # ── 5. Fixed income analysis ──────────────────────────────────────────────
    _section("BOND SLEEVE — DURATION-ONLY YIELD SHOCK SENSITIVITY")
    fi_df = yield_shock_pnl(WEIGHTS, portfolio_value=PORTFOLIO_VALUE)
    print(tabulate(fi_df, floatfmt=".2f", headers="keys", tablefmt="plain", showindex=False))

    print("  Approximation only; convexity unavailable and omitted. Not total portfolio stress.")
    print("  VTIP uses real-yield duration; shocks are not a single nominal-curve scenario.")
    print("  Credit spreads, inflation accrual and nonlinear effects are not modeled.")
    for ticker, profile in FI_PROFILES.items():
        print(f"  {ticker}: {profile['as_of']} — {profile['status']}; source: {profile['source']}")
    avg_dur = portfolio_fi_duration(WEIGHTS)
    fi_weight = sum(WEIGHTS[t] for t in ["VGIT", "VTIP", "JPIE", "MINT"])
    print(f"\n  Fixed income sleeve weight: {fi_weight*100:.1f}%")
    print(f"  Weighted average duration (FI sleeve): {avg_dur:.2f} years")

    # ── 6. Historical stress scenarios ────────────────────────────────────────
    _section("HISTORICAL STRESS SCENARIOS — BUY-AND-HOLD FROM TARGET WEIGHTS")
    stress_df = run_all_scenarios(prices, WEIGHTS, CASH_WEIGHT, cash_rate)
    if not stress_df.empty and "Portfolio Return (%)" in stress_df.columns:
        print(tabulate(
            stress_df[["Portfolio Return (%)", "Status"]].fillna("n/a"),
            headers="keys",
            tablefmt="plain",
        ))
        if ("2022 Rate Shock (Jan–Oct 2022)" in stress_df.index
                and stress_df.loc["2022 Rate Shock (Jan–Oct 2022)", "Status"] == "Available"):
            print("\n  2022 Rate Shock — Position-Level Breakdown:")
            contrib_df = scenario_contribution_table(prices, WEIGHTS, "2022 Rate Shock (Jan–Oct 2022)", CASH_WEIGHT, cash_rate)
            print(tabulate(contrib_df, headers="keys", tablefmt="plain", showindex=False))
    else:
        print("  No historical stress periods available in the current price history.")
        print("  Run with --refresh and a longer period (e.g. '5y') to include 2022 data.")

    # ── 7. Charts ─────────────────────────────────────────────────────────────
    _section("GENERATING CHARTS  →  charts/")

    _, port_dd = max_drawdown(port_ret)
    _, bench_dd = max_drawdown(bench_ret)
    port_vol = rolling_volatility(port_ret, window=30)
    bench_vol = rolling_volatility(bench_ret, window=30)
    corr = correlation_matrix(prices, WEIGHTS)

    plot_correlation_heatmap(corr)
    plot_rolling_volatility(port_vol, bench_vol, window=30)
    plot_drawdown(port_dd, bench_dd)
    plot_var_distribution(port_ret, p["var_95"], p["var_99"], p["cvar_95"])
    plot_return_attribution(attr_df)
    plot_yield_shock(fi_df)

    # ── 8. Footer ─────────────────────────────────────────────────────────────
    _separator()
    print("  Analysis complete. Charts saved to charts/")
    print(f"  Data period: {prices.index[0].date()} → {prices.index[-1].date()}")
    print("  Risk-free rate assumed: {:.0f}bps".format(RISK_FREE_RATE * 10_000))
    _separator()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Risk Analytics Dashboard")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of price data from Yahoo Finance",
    )
    parser.add_argument("--cash-rate", type=float, default=CASH_RATE,
                        help="Annual nominal cash interest as decimal (default 0; example 0.04)")
    args = parser.parse_args()
    main(refresh=args.refresh, cash_rate=args.cash_rate)
