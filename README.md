# Portfolio Risk Analytics Dashboard

A Python-based risk analytics engine for a 16-position multi-asset portfolio, built to replicate the kind of quantitative risk reporting used in institutional asset management. Covers market risk (VaR/CVaR, volatility, drawdown), return attribution, fixed income duration/DV01 analysis, and historical stress testing.

---

## Portfolio

| Asset Class | Tickers | Weight |
|---|---|---|
| Equities | AAPL, MSFT, MU, WMT, DAL, IAG, SPCX | 30.0% |
| Broad / Sector ETFs | SPY, VT, XLV, XLF, EEM | 38.0% |
| Fixed Income ETFs | VGIT, VTIP, JPIE, MINT | 26.5% |

> Weights sum to 94.5% reflecting the GIC bootcamp simulation allocation, transcribed from `data/GPP2026_Complete_Analysis_exCAT.xlsx` (sheet *Position PL Tracker*). CAT was never held; the workbook's *CAT Removal Log* sheet records that correction. SPCX listed recently and has only a few weeks of trading history; it's kept in the target weights above but excluded from the historical backtest (returns, VaR/CVaR, attribution, stress tests) so the other 15 positions retain their full multi-year window. SPCX is reported standalone in the console output instead.

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
- Full 15×15 correlation heatmap across all backtested positions (SPCX excluded for lack of history)

**Fixed Income Risk**
- Modified duration, DV01, and convexity for the bond ETF sleeve (VGIT, VTIP, JPIE, MINT)
- Parallel yield curve shock scenarios: +50, +100, +200 bps
- Estimated P&L impact per position and in aggregate

**Historical Stress Testing**
- 2022 rate shock (Jan–Oct 2022): portfolio return and per-position breakdown
- Additional scenarios defined for COVID crash, 2020 recovery, Q4 2018

---

## Results (5-year backtest, Nov 2021 – Jul 2026)

Generated 13 Sep 2026 with `python main.py` on the cached prices (2021-11-02 → 2026-07-23) for the ex-CAT weights above. The backtest excludes SPCX, so the backtested book is 91.5% invested; the uninvested remainder is treated as earning zero.

| Metric | Portfolio | SPY Benchmark |
|---|---|---|
| Annualised Return | 15.09% | 12.03% |
| Annualised Volatility | 12.94% | 17.47% |
| Sharpe Ratio | 0.842 | 0.509 |
| Sortino Ratio | 0.851 | 0.500 |
| VaR 95% (1-day) | 1.21% / $12,060 | 1.68% |
| VaR 99% (1-day) | 2.11% / $21,071 | 2.95% |
| CVaR 95% (1-day) | 1.77% / $17,656 | 2.50% |
| Max Drawdown | -19.76% | -24.50% |

**2022 Rate Shock** — Portfolio lost **-18.1%** (Jan–Oct 2022). MU was the largest single drag at -424 bps; the fixed income sleeve cost ~-204 bps in a rising-rate environment, led by VGIT at -116 bps.

**Fixed income +200 bps shock** — Estimated portfolio P&L: **-$18,385** on $1M, with VGIT the largest risk at -$10,393.

---

## Project Structure

```
portfolio-risk-dashboard/
├── main.py                    # End-to-end pipeline; prints full risk report
├── requirements.txt
├── pytest.ini
├── data/
│   ├── fetch_prices.py        # Ticker list, target weights, yfinance download + CSV cache
│   ├── prices.csv             # Cached adjusted closes
│   └── GPP2026_Complete_Analysis_exCAT.xlsx   # Source workbook for the weights
├── tests/
│   ├── test_portfolio_config.py   # Weights match the workbook; CAT absent
│   └── test_attribution.py        # Stray price columns are ignored
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

# 4. Run the tests (no network needed)
pip install pytest
pytest
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
