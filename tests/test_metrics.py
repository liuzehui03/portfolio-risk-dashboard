"""
Blended 60/40 benchmark (0.60 × URTH + 0.40 × BNDW daily returns) and the
three-way risk summary: Portfolio | 60/40 Blend | SPY.
"""

import numpy as np
import pandas as pd
import pytest

from risk.metrics import (
    BLEND_WEIGHTS,
    blended_benchmark_returns,
    portfolio_returns,
    risk_summary,
)

METRIC_KEYS = {"ann_return", "ann_vol", "sharpe", "sortino",
               "var_95", "var_99", "cvar_95", "cvar_99", "max_drawdown"}


@pytest.fixture
def prices() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=60)
    rng = np.random.default_rng(1)
    cols = {t: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 60))
            for t in ["URTH", "BNDW", "SPY", "AAPL", "MSFT"]}
    return pd.DataFrame(cols, index=dates)


def test_blend_weights_are_60_40_urth_bndw():
    assert BLEND_WEIGHTS == {"URTH": 0.60, "BNDW": 0.40}


def test_blended_benchmark_is_weighted_daily_returns(prices):
    expected = 0.60 * prices["URTH"].pct_change() + 0.40 * prices["BNDW"].pct_change()
    expected = expected.dropna()

    result = blended_benchmark_returns(prices)

    pd.testing.assert_index_equal(result.index, expected.index)
    np.testing.assert_allclose(result.values, expected.values)
    assert result.name == "60/40 Blend"


def test_blended_benchmark_ignores_other_columns(prices):
    full = blended_benchmark_returns(prices)
    subset = blended_benchmark_returns(prices[["URTH", "BNDW"]])
    pd.testing.assert_series_equal(full, subset)


def test_risk_summary_reports_all_three_series(prices):
    port = portfolio_returns(prices, {"AAPL": 0.5, "MSFT": 0.5})
    spy = prices["SPY"].pct_change().dropna()
    blend = blended_benchmark_returns(prices)

    summary = risk_summary(port, spy, risk_free_rate=0.04, blend_returns=blend)

    assert set(summary) == {"portfolio", "benchmark", "blend"}
    assert summary["portfolio"]["label"] == "Portfolio"
    assert summary["blend"]["label"] == "60/40 Blend"
    assert summary["benchmark"]["label"] == "SPY"
    for block in summary.values():
        assert METRIC_KEYS <= set(block)
        assert all(np.isfinite(block[k]) for k in METRIC_KEYS)


def test_risk_summary_without_blend_is_unchanged(prices):
    port = portfolio_returns(prices, {"AAPL": 0.5, "MSFT": 0.5})
    spy = prices["SPY"].pct_change().dropna()

    summary = risk_summary(port, spy, risk_free_rate=0.04)

    assert set(summary) == {"portfolio", "benchmark"}


def test_risk_summary_aligns_all_series_to_common_dates(prices):
    port = portfolio_returns(prices, {"AAPL": 0.5, "MSFT": 0.5})
    spy = prices["SPY"].pct_change().dropna()
    blend = blended_benchmark_returns(prices).iloc[10:]   # shorter window

    full = risk_summary(port, spy, 0.04, blend_returns=blend)
    trimmed = risk_summary(port.iloc[10:], spy.iloc[10:], 0.04, blend_returns=blend)

    for key in ("portfolio", "benchmark", "blend"):
        assert full[key]["ann_return"] == pytest.approx(trimmed[key]["ann_return"])
        assert full[key]["max_drawdown"] == pytest.approx(trimmed[key]["max_drawdown"])
