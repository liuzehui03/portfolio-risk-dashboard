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

import sys
import argparse
import warnings
import pandas as pd
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ── Project imports ──────────────────────────────────────────────────────────
from data.fetch_prices import load_prices, WEIGHTS, BACKTEST_WEIGHTS
from risk.metrics import (
    portfolio_returns,
    rolling_volatility,
    max_drawdown,
    risk_summary,
)
from risk.fixed_income import yield_shock_pnl, portfolio_fi_duration
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
RISK_FREE_RATE = 0.04        # ~4% annualised, approximate T-bill yield


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(x, decimals=2):
    """Format a decimal as a percentage string."""
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

def main(refresh: bool = False) -> None:
    print("\n" + "═" * 72)
    print("  PORTFOLIO RISK ANALYTICS DASHBOARD")
    print(f"  {len(WEIGHTS)}-Position Multi-Asset Portfolio | GIC Bootcamp Simulation")
    print("═" * 72)

    # ── 1. Load prices ────────────────────────────────────────────────────────
    prices = load_prices(refresh=refresh)

    # ── 2. Compute returns ────────────────────────────────────────────────────
    # BACKTEST_WEIGHTS excludes SPCX (recently listed, ~weeks of history) so
    # the rest of the portfolio keeps its full multi-year backtest window.
    port_ret = portfolio_returns(prices, BACKTEST_WEIGHTS)
    bench_ret = prices["SPY"].pct_change().dropna()

    # Align to common dates
    port_ret, bench_ret = port_ret.align(bench_ret, join="inner")

    # ── 3. Risk metrics ───────────────────────────────────────────────────────
    summary = risk_summary(port_ret, bench_ret, RISK_FREE_RATE)
    p = summary["portfolio"]
    b = summary["benchmark"]

    _section("RISK METRICS SUMMARY")
    metrics_table = [
        ["Metric",                  "Portfolio",          "SPY Benchmark"],
        ["─" * 25,                  "─" * 18,             "─" * 18],
        ["Annualised Return",       _pct(p["ann_return"]),  _pct(b["ann_return"])],
        ["Annualised Volatility",   _pct(p["ann_vol"]),     _pct(b["ann_vol"])],
        ["Sharpe Ratio",            f"{p['sharpe']:.3f}",   f"{b['sharpe']:.3f}"],
        ["Sortino Ratio",           f"{p['sortino']:.3f}",  f"{b['sortino']:.3f}"],
        ["VaR 95% (1-day)",         _pct(p["var_95"]),      _pct(b["var_95"])],
        ["VaR 99% (1-day)",         _pct(p["var_99"]),      _pct(b["var_99"])],
        ["CVaR 95% (1-day)",        _pct(p["cvar_95"]),     _pct(b["cvar_95"])],
        ["CVaR 99% (1-day)",        _pct(p["cvar_99"]),     _pct(b["cvar_99"])],
        ["Max Drawdown",            _pct(p["max_drawdown"]),_pct(b["max_drawdown"])],
    ]
    print(tabulate(metrics_table, headers="firstrow", tablefmt="plain"))

    # Dollar-equivalent VaR/CVaR
    print(f"\n  Portfolio value assumed: ${PORTFOLIO_VALUE:,.0f}")
    print(f"  VaR 95% ($):  ${p['var_95'] * PORTFOLIO_VALUE:,.0f}")
    print(f"  VaR 99% ($):  ${p['var_99'] * PORTFOLIO_VALUE:,.0f}")
    print(f"  CVaR 95% ($): ${p['cvar_95'] * PORTFOLIO_VALUE:,.0f}")

    # ── 4. Return attribution ─────────────────────────────────────────────────
    _section("RETURN ATTRIBUTION BY POSITION")
    attr_df = return_attribution(prices, BACKTEST_WEIGHTS)
    print(tabulate(attr_df, headers="keys", tablefmt="plain", showindex=False))

    _section("RETURN ATTRIBUTION BY ASSET CLASS")
    ac_df = asset_class_attribution(prices, BACKTEST_WEIGHTS)
    print(tabulate(ac_df, headers="keys", tablefmt="plain", showindex=False))

    # ── SPCX footnote (recent listing, excluded from the backtest above) ─────
    _section("SPCX — RECENT LISTING (EXCLUDED FROM BACKTEST ABOVE)")
    spcx_px = prices["SPCX"].dropna()
    if len(spcx_px) > 1:
        spcx_ret = spcx_px.iloc[-1] / spcx_px.iloc[0] - 1
        print(f"  Target weight: {WEIGHTS['SPCX']*100:.1f}% (not included in the metrics/attribution/stress sections above)")
        print(f"  Trading history: {spcx_px.index[0].date()} → {spcx_px.index[-1].date()} ({len(spcx_px)} sessions)")
        print(f"  Return since listing: {spcx_ret*100:.2f}%")
        print("  Too little history for annualised vol/Sharpe/VaR — revisit once >1yr of data exists.")
    else:
        print("  No SPCX price history available.")

    # ── 5. Fixed income analysis ──────────────────────────────────────────────
    _section("FIXED INCOME — YIELD SHOCK ANALYSIS (Parallel +bps Shifts)")
    fi_df = yield_shock_pnl(WEIGHTS, portfolio_value=PORTFOLIO_VALUE)
    print(tabulate(fi_df, headers="keys", tablefmt="plain", showindex=False))

    avg_dur = portfolio_fi_duration(WEIGHTS)
    fi_weight = sum(WEIGHTS[t] for t in ["VGIT", "VTIP", "JPIE", "MINT"])
    print(f"\n  Fixed income sleeve weight: {fi_weight*100:.1f}%")
    print(f"  Weighted average duration (FI sleeve): {avg_dur:.2f} years")

    # ── 6. Historical stress scenarios ────────────────────────────────────────
    _section("HISTORICAL STRESS SCENARIOS")
    stress_df = run_all_scenarios(prices, BACKTEST_WEIGHTS)
    if not stress_df.empty and "Portfolio Return (%)" in stress_df.columns:
        print(tabulate(
            stress_df[["Portfolio Return (%)"]],
            headers="keys",
            tablefmt="plain",
        ))
        if "2022 Rate Shock (Jan–Oct 2022)" in stress_df.index:
            print("\n  2022 Rate Shock — Position-Level Breakdown:")
            contrib_df = scenario_contribution_table(prices, BACKTEST_WEIGHTS, "2022 Rate Shock (Jan–Oct 2022)")
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
    corr = correlation_matrix(prices, BACKTEST_WEIGHTS)

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
    args = parser.parse_args()
    main(refresh=args.refresh)
