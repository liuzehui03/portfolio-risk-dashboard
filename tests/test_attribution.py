"""
Behavioural checks: a stale CAT column in the price cache must not affect any output.
"""

import numpy as np
import pandas as pd
import pytest

from data.fetch_prices import WEIGHTS
from portfolio.attribution import asset_class_attribution
from risk.metrics import portfolio_returns


@pytest.fixture
def prices() -> pd.DataFrame:
    """Five days of synthetic prices for every held ticker plus a stray CAT column."""
    dates = pd.bdate_range("2026-06-01", periods=5)
    rng = np.random.default_rng(0)
    frame = {t: 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 5)) for t in WEIGHTS}
    frame["CAT"] = 100 * np.cumprod(1 + np.full(5, 0.05))  # strong drift, must be ignored
    return pd.DataFrame(frame, index=dates)


def test_asset_class_attribution_ignores_extra_columns(prices):
    with_cat = asset_class_attribution(prices, WEIGHTS)
    without_cat = asset_class_attribution(prices.drop(columns="CAT"), WEIGHTS)

    assert with_cat["Weight (%)"].sum() == pytest.approx(94.5)
    pd.testing.assert_frame_equal(with_cat, without_cat)


def test_portfolio_returns_uses_only_weighted_tickers(prices):
    tickers = list(WEIGHTS)
    expected = prices[tickers].pct_change().dropna() @ np.array([WEIGHTS[t] for t in tickers])

    result = portfolio_returns(prices, WEIGHTS)

    np.testing.assert_allclose(result.values, expected.values)
    assert "CAT" not in tickers
