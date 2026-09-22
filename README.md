# Portfolio Risk Analytics Dashboard

A Python command-line risk dashboard for a **15-security multi-asset simulation portfolio with 1% cash**. It measures historical market risk at specified weights, compares two benchmarks, reconciles return attribution, and estimates bond-sleeve yield sensitivity.

It answers: **“How would this chosen allocation have behaved under historical market moves?”** It does not infer current holdings from purchase prices, provide live intraday risk, optimize allocations, or report actual account performance.

## Portfolio

Revised simulation allocation dated **23 September 2026**. The original weights were transcribed from `data/GPP2026_Complete_Analysis_exCAT.xlsx`, sheet *Position PL Tracker*. CAT was not held. SPCX is removed from the revised portfolio.

The other original weights total 91.5%. Each is multiplied by `0.99 / 0.915`, preserving their relative proportions. The remaining 1% is explicit cash. These are chosen target weights, not verified current market-value weights. The workbook remains an unchanged historical reference.

| Holding | Revised weight |
|---|---:|
| SPY | 15.1475% |
| VT | 10.8197% |
| VGIT | 10.8197% |
| MU | 10.8197% |
| XLV | 6.4918% |
| JPIE | 6.4918% |
| MSFT | 6.4918% |
| VTIP | 6.4918% |
| AAPL | 5.4098% |
| XLF | 5.4098% |
| MINT | 4.8689% |
| WMT | 4.3279% |
| EEM | 3.2459% |
| DAL | 1.6230% |
| IAG | 0.5410% |
| Cash | 1.0000% |

Full precision is retained internally; displayed weights are rounded. Equities total 29.21%, broad/sector ETFs 41.11%, fixed-income ETFs 28.67%, and cash 1%.

## Quickstart

Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest

python main.py                  # existing cached daily prices
python main.py --refresh        # download updated five-year daily history
python main.py --cash-rate 0.04 # optional 4% nominal annual cash assumption
python -m pytest -q             # offline numerical and integration tests
```

`--cash-rate` defaults to **0%**, independent of the **4% risk-free assumption** used for Sharpe/Sortino. Both annual nominal rates are divided by 252 per daily observation. Dollar risk uses a disclosed **$1 million USD simulation notional**.

Price dates and the number of observations appear in the report. Cached data older than three weekdays produces a warning but remains usable for reproducible historical analysis. Weekday age excludes weekends, not exchange holidays. No automatic network refresh occurs when a cache exists.

## Features and methodology

### Historical risk and performance

Yahoo Finance adjusted closes (`auto_adjust=True`) supply split/dividend-adjusted daily return proxies. Each historical daily portfolio scenario is `sum(weight × asset_return) + cash_weight × daily_cash_rate`.

- **VaR:** negative empirical return percentile at the chosen confidence, using NumPy's linear percentile interpolation.
- **CVaR / expected shortfall:** average of exactly the worst 5% or 1% probability mass, including a fractional boundary observation. Tied outcomes are handled consistently. VaR and CVaR are signed losses; an all-gain sample can produce negative loss figures.
- **Volatility:** sample daily standard deviation multiplied by `sqrt(252)`; rolling window defaults to 30 observations.
- **Sharpe:** mean excess daily return divided by sample daily standard deviation, annualized by `sqrt(252)`.
- **Sortino:** mean excess daily return divided by the root mean squared negative excess return **over all observations**, with positive excess returns contributing zero. See [Sortino methodology](https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf).
- **CAGR:** compounded return annualized using `observations / 252`.
- **Drawdown:** loss relative to the running wealth peak, including initial wealth of 1; displayed as a negative percentage.

Compounding constant-weight daily returns assumes daily rebalancing without trading costs or taxes. It is hypothetical historical performance, not an achieved investment return or forecast. Rescaling the existing mix to 99% increases exposure; it is not an optimization or a claim of improved future Sharpe.

### Benchmarks

SPY is the pure-equity reference. The second benchmark is a daily-rebalanced **60% URTH / 40% BNDW** ETF blend, used as a proxy for the MSCI World / Bloomberg Global Aggregate mandate mix. ETF fees, tracking differences, and bond currency hedging can cause differences from the intended indices. All portfolio and benchmark metrics use the same dates.

### Reconciled attribution

Daily asset contributions are multiplied by prior-day portfolio wealth and then summed. Contributions, including cash, reconcile to total compounded portfolio return before rounding. Asset-class attribution aggregates these same contributions.

The separate asset total-return column is each asset's buy-and-hold return. Multiplying it by a fixed target weight does **not** reproduce a daily-rebalanced portfolio's contribution. Correlation covers the 15 securities; constant cash has no defined correlation and is omitted from that heatmap.

### Historical stress scenarios

The 2022 rate shock, COVID crash/recovery, and Q4 2018 scenarios apply each asset's complete period return to the revised starting weights. These are **buy-and-hold scenario shocks**, distinct from the daily-rebalanced performance series. Cash compounds at its configured rate over the scenario's observed return intervals.

Both exact scenario endpoints must exist for every holding. Missing holdings, incomplete windows, or invalid prices produce an explicit unavailable status. Earlier crises cannot be reconstructed for this entire portfolio merely by downloading more data: JPIE's shorter history limits the common window.

### Bond-sleeve sensitivity

The active calculation is **duration-only**: `price_change ≈ −duration × yield_change`, with yield changes in decimals; `DV01 = duration × allocation × 0.0001`. Scenarios are +50, +100, and +200 basis points. Totals are calculated before display rounding.

| ETF | Duration (years) | Input date | Source/status |
|---|---:|---|---|
| VGIT | 4.9 | 2026-07-31 | [Vanguard average duration](https://advisors.vanguard.com/investments/products/vgit/vanguard-intermediate-term-treasury-etf.html) |
| VTIP | 2.5 | 2026-07-31 | [Vanguard real-yield duration](https://advisors.vanguard.com/investments/products/vtip/vanguard-short-term-inflation-protected-securities-etf) |
| JPIE | 2.55 | 2026-08-31 | [JPMorgan fact sheet, portfolio analysis](https://am.jpmorgan.com/content/dam/jpm-am-aem/americas/us/en/literature/fact-sheet/etfs/FS-JPIE.PDF) |
| MINT | 0.35 | mid-2024 | **Unverified legacy assumption**; current [issuer document](https://www.pimco.com/us/en/investments/etf/pimco-enhanced-short-maturity-active-exchange-traded-fund/nyse) was inaccessible during review |

Sources reviewed 23 September 2026; these inputs do not refresh with prices. Unsupported legacy convexities have been removed. The reusable formula supports a supplied convexity in years squared, but none of the active profiles has a verified value.

VTIP's sensitivity is to real yields. Adding the sleeve's sensitivities is a stylized simultaneous yield-shift exercise, not a single nominal Treasury-curve shock. Credit spreads, inflation accrual, nonlinear effects, and equity reactions are not modeled. MINT's assumption also limits precision. Results are not a complete portfolio stress loss.

## Reproduced results

Generated with the revised code on **23 September 2026**, using the existing cache **2021-11-02 through 2026-09-11**, **1,218 daily returns**. Cash earns 0%; risk-free assumption is 4%. These replace the previous SPCX-excluded, 91.5%-invested results. The cache is stale relative to the generation date.

| Metric | Portfolio | 60/40 blend | SPY |
|---|---:|---:|---:|
| Annualized return | 16.73% | 6.83% | 12.47% |
| Annualized volatility | 13.99% | 10.58% | 17.32% |
| Sharpe | 0.890 | 0.299 | 0.534 |
| Sortino | 1.323 | 0.427 | 0.768 |
| Daily VaR 95% | 1.30% | 1.00% | 1.66% |
| Daily VaR 99% | 2.25% | 1.84% | 2.94% |
| Daily CVaR 95% | 1.91% | 1.49% | 2.49% |
| Daily CVaR 99% | 3.02% | 2.30% | 3.94% |
| Maximum drawdown | -21.27% | -21.32% | -24.50% |

On the $1 million notional, daily 95% VaR is **$12,956** and 95% expected shortfall is **$19,089**. Reconciled cumulative attribution is **111.23%**. The complete 2022 scenario returns **−19.57%**. The bond-sleeve +200bp duration-only estimate is **−$17,501**, subject to the assumptions above.

## Validation and structure

- `data/fetch_prices.py`: original reference allocation, revised weights/cash, downloads, cache validation and age warning.
- `risk/metrics.py`: fixed-weight returns and risk measures.
- `risk/stress_test.py`: complete-window historical shocks and unavailable statuses.
- `risk/fixed_income.py`: dated duration inputs and bond-sleeve sensitivity.
- `portfolio/attribution.py`: reconciled contributions and correlations.
- `main.py`: console report; `visualisation/charts.py`: six PNG charts in `charts/`.
- `tests/`: configuration, benchmark alignment, independent numerical examples, data failure cases, and integration checks.

Required prices must be positive and finite, with unique sorted dates. Leading pre-inception gaps determine the common start; internal/trailing missing values are rejected, not forward-filled. Extra cached columns, including old SPCX/CAT data, are ignored. Previously forward-filled cached observations cannot be identified retroactively; refresh to download a new history. The validator does not independently certify vendor prices or detect every missing exchange session.

Undefined ratios display `n/a`; invalid samples raise errors. The test suite runs offline. This project does not implement allocation optimization or the separate research described in the unrelated planning documents.
