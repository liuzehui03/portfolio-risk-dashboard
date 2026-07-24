# Portfolio Risk Analytics Dashboard

A Python-based risk analytics engine for a 17-position multi-asset portfolio, built to replicate the kind of quantitative risk reporting used in institutional asset management. Covers market risk (VaR/CVaR, volatility, drawdown), return attribution, fixed income duration/DV01 analysis, and historical stress testing.

---

## Portfolio

| Asset Class | Tickers | Weight |
|---|---|---|
| Equities | AAPL, MSFT, MU, WMT, DAL, IAG, CAT, SPCX | 33.0% |
| Broad / Sector ETFs | SPY, VT, XLV, XLF, EEM | 34.0% |
| Fixed Income ETFs | VGIT, VTIP, JPIE, MINT | 26.5% |

> Weights sum to 93.5% reflecting the GIC bootcamp simulation allocation.

---

## Features

**Market Risk**
- Historical VaR at 95% and 99% confidence, and CVaR (Expected Shortfall)
- Dollar-equivalent risk figures on a $1M notional portfolio
- Daily return distribution chart with VaR/CVaR thresholds marked

**Performance Analytics**
- Annualised return (CAGR), Sharpe ratio, and Sortino ratio vs SPY benchmark
- 30-day rolling annualised volatility chart (portfolio vs SPY)
- Maximum drawdown calculation and time-series chart

**Return Attribution**
- Per-position contribution to total portfolio return (in % and basis points)
- Asset class sleeve attribution (equities / broad ETFs / fixed income)
- Full 17×17 correlation heatmap across all positions

**Fixed Income Risk**
- Modified duration, DV01, and convexity for the bond ETF sleeve (VGIT, VTIP, JPIE, MINT)
- Parallel yield curve shock scenarios: +50, +100, +200 bps
- Estimated P&L impact per position and in aggregate

**Historical Stress Testing**
- 2022 rate shock (Jan–Oct 2022): portfolio return and per-position breakdown
- Additional scenarios defined for COVID crash, 2020 recovery, Q4 2018

---

## Results (5-year backtest, Nov 2021 – Jul 2026)

> ⚠️ The figures below were generated under the previous portfolio composition (VWRA.L/XLE instead of VT/SPCX). Re-run `python main.py --refresh` to regenerate this section for the current weights.

| Metric | Portfolio | SPY Benchmark |
|---|---|---|
| Annualised Return | 12.85% | 11.95% |
| Annualised Volatility | 10.45% | 17.25% |
| Sharpe Ratio | 0.83 | 0.51 |
| Sortino Ratio | 0.82 | 0.51 |
| VaR 95% (1-day) | 0.92% / $9,244 | 1.67% |
| VaR 99% (1-day) | 1.73% / $17,346 | 2.94% |
| CVaR 95% (1-day) | 1.46% / $14,629 | 2.49% |
| Max Drawdown | -16.2% | -24.5% |

**2022 Rate Shock** — Portfolio lost **-14.0%** (Jan–Oct 2022) under the previous composition; fixed income cost ~-420 bps in a rising-rate environment. *(Per-position contributors will shift now that XLE is no longer held — pending re-run.)*

**Fixed income +200 bps shock** — Estimated portfolio P&L: **-$18,385** on $1M, with VGIT the largest risk at -$10,393.

---

## Project Structure

```
portfolio-risk-dashboard/
├── main.py                    # End-to-end pipeline; prints full risk report
├── requirements.txt
├── data/
│   └── fetch_prices.py        # yfinance download + CSV cache
├── risk/
│   ├── metrics.py             # VaR, CVaR, Sharpe, Sortino, drawdown, rolling vol
│   ├── fixed_income.py        # Duration, DV01, convexity, yield shock P&L
│   └── stress_test.py         # Historical scenario replay
├── portfolio/
│   └── attribution.py         # Return contribution, correlation matrix
├── visualisation/
│   └── charts.py              # Heatmap, rolling vol, drawdown, VaR, attribution charts
└── charts/                    # Generated PNG outputs (auto-created on first run)
```

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone https://github.com/your-username/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
pip install -r requirements.txt

# 2. Run the full pipeline (downloads 5 years of price data on first run)
python3 main.py

# 3. Force a fresh data download
python3 main.py --refresh
```

Prices are cached to `data/prices.csv` after the first download to avoid repeated API calls.

---

## Methodology Notes

**VaR / CVaR**: Historical simulation (non-parametric). No distributional assumptions — losses are ranked directly from the empirical return series. CVaR is the mean of all days that breach the VaR threshold.

**Sharpe / Sortino**: Annualised using 252 trading days. Risk-free rate assumed at 4% (approximate T-bill yield). Sortino uses downside deviation only, making it more appropriate for asymmetric return profiles.

**Fixed income duration/convexity**: Point-in-time estimates sourced from fund fact sheets (Vanguard, JPMorgan, PIMCO). These are not derived from underlying bond cashflows. Yield shock P&L uses the standard duration-convexity approximation: `ΔP/P ≈ -D·Δy + ½·C·(Δy)²`.

**Stress scenarios**: Historical period returns are applied to current weights. This assumes the portfolio composition is fixed — it does not account for rebalancing or position changes that may have occurred during the stress period.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical price data via Yahoo Finance |
| `pandas` / `numpy` | Data manipulation and numerical computation |
| `matplotlib` / `seaborn` | Charting and visualisation |
| `scipy` | Statistical utilities |
| `tabulate` | Formatted terminal output |
