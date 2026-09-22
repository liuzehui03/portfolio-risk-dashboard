"""
Visualisation module: heatmap, rolling volatility, drawdown, VaR,
and return attribution charts. All charts are saved to the charts/ directory
and optionally displayed inline.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

CHARTS_DIR = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Consistent style for all charts
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = {"portfolio": "#1f77b4", "benchmark": "#ff7f0e"}


def _save(fig: plt.Figure, filename: str, show: bool = False) -> None:
    """Save figure to charts directory and optionally display it."""
    path = CHARTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    show: bool = False,
) -> None:
    """
    Plot and save a correlation heatmap for all portfolio positions.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Pairwise correlation matrix (output of attribution.correlation_matrix).
    show : bool
        If True, display the chart interactively.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # show lower triangle

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(f"Return Correlation Matrix — {len(corr_matrix)}-Position Portfolio", fontsize=14, pad=15)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, "correlation_heatmap.png", show)


def plot_rolling_volatility(
    port_vol: pd.Series,
    bench_vol: pd.Series,
    window: int = 30,
    show: bool = False,
) -> None:
    """
    Plot rolling annualised volatility for the portfolio vs SPY benchmark.

    Parameters
    ----------
    port_vol : pd.Series
        Portfolio rolling volatility (output of metrics.rolling_volatility).
    bench_vol : pd.Series
        Benchmark rolling volatility.
    window : int
        Rolling window used (for title label only).
    show : bool
        If True, display interactively.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(port_vol.index, port_vol * 100, label="Portfolio", color=PALETTE["portfolio"], linewidth=1.5)
    ax.plot(bench_vol.index, bench_vol * 100, label="SPY Benchmark", color=PALETTE["benchmark"],
            linewidth=1.5, linestyle="--")

    ax.set_title(f"{window}-Day Rolling Annualised Volatility (%)", fontsize=13)
    ax.set_ylabel("Volatility (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "rolling_volatility.png", show)


def plot_drawdown(
    port_drawdown: pd.Series,
    bench_drawdown: pd.Series,
    show: bool = False,
) -> None:
    """
    Plot the drawdown series for the portfolio and SPY benchmark.

    Parameters
    ----------
    port_drawdown : pd.Series
        Portfolio drawdown series (output of metrics.max_drawdown).
    bench_drawdown : pd.Series
        Benchmark drawdown series.
    show : bool
        If True, display interactively.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(
        port_drawdown.index,
        port_drawdown * 100,
        0,
        alpha=0.4,
        color=PALETTE["portfolio"],
        label="Portfolio",
    )
    ax.fill_between(
        bench_drawdown.index,
        bench_drawdown * 100,
        0,
        alpha=0.25,
        color=PALETTE["benchmark"],
        label="SPY Benchmark",
    )
    ax.plot(port_drawdown.index, port_drawdown * 100, color=PALETTE["portfolio"], linewidth=1)
    ax.plot(bench_drawdown.index, bench_drawdown * 100, color=PALETTE["benchmark"],
            linewidth=1, linestyle="--")

    ax.set_title("Portfolio Drawdown vs SPY Benchmark", fontsize=13)
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "drawdown.png", show)


def plot_var_distribution(
    port_returns: pd.Series,
    var_95: float,
    var_99: float,
    cvar_95: float,
    show: bool = False,
) -> None:
    """
    Plot the daily return distribution with VaR and CVaR thresholds marked.

    Parameters
    ----------
    port_returns : pd.Series
        Daily portfolio returns.
    var_95, var_99 : float
        95% and 99% VaR as positive fractions.
    cvar_95 : float
        95% CVaR as a positive fraction.
    show : bool
        If True, display interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(port_returns * 100, bins=80, color="#4c72b0", edgecolor="white",
            linewidth=0.3, alpha=0.85, density=True, label="Daily Returns")

    ax.axvline(-var_95 * 100, color="#e67e22", linestyle="--", linewidth=1.5,
               label=f"VaR 95% ({-var_95*100:.2f}%)")
    ax.axvline(-var_99 * 100, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"VaR 99% ({-var_99*100:.2f}%)")
    ax.axvline(-cvar_95 * 100, color="#8e44ad", linestyle=":", linewidth=1.5,
               label=f"CVaR 95% ({-cvar_95*100:.2f}%)")

    ax.set_title("Portfolio Daily Return Distribution with VaR / CVaR", fontsize=13)
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "var_distribution.png", show)


def plot_return_attribution(attribution_df: pd.DataFrame, show: bool = False) -> None:
    """
    Horizontal bar chart showing each position's return contribution in basis points.

    Parameters
    ----------
    attribution_df : pd.DataFrame
        Output of attribution.return_attribution (excluding TOTAL row).
    show : bool
        If True, display interactively.
    """
    df = attribution_df[attribution_df["Ticker"] != "TOTAL"].copy()
    df = df.sort_values("Contribution (bps)", ascending=True)

    colors = ["#c0392b" if v < 0 else "#27ae60" for v in df["Contribution (bps)"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(df["Ticker"], df["Contribution (bps)"], color=colors, edgecolor="white", height=0.6)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Wealth-Linked Return Attribution (basis points)", fontsize=13)
    ax.set_xlabel("Contribution (bps)")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, df["Contribution (bps)"]):
        x = bar.get_width()
        ax.text(
            x + (2 if x >= 0 else -2),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
        )

    _save(fig, "return_attribution.png", show)


def plot_yield_shock(fi_shock_df: pd.DataFrame, show: bool = False) -> None:
    """
    Grouped bar chart showing fixed income P&L under different yield shock scenarios.

    Parameters
    ----------
    fi_shock_df : pd.DataFrame
        Output of fixed_income.yield_shock_pnl (excluding TOTAL row).
    show : bool
        If True, display interactively.
    """
    df = fi_shock_df[fi_shock_df["Ticker"] != "TOTAL"].copy()

    shock_cols = [c for c in df.columns if "P&L" in c]
    x = np.arange(len(df))
    width = 0.25
    colors = ["#f39c12", "#e67e22", "#c0392b"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (col, color) in enumerate(zip(shock_cols, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[col], width, label=col, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Ticker"], fontsize=10)
    ax.set_title("Bond Sleeve — Duration-Only Yield Sensitivity ($1M Portfolio)", fontsize=12)
    ax.set_ylabel("Estimated P&L ($)")
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "yield_shock_pnl.png", show)
