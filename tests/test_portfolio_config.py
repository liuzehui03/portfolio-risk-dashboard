"""
Portfolio configuration must match the ex-CAT analysis workbook.

Reference: data/GPP2026_Complete_Analysis_exCAT.xlsx, sheet "Position PL Tracker",
column C (Weight). CAT was never held; see sheet "CAT Removal Log".
"""

import pytest

from data.fetch_prices import (
    ASSET_CLASSES,
    ORIGINAL_WEIGHTS,
    CASH_WEIGHT,
    BENCHMARK_TICKERS,
    DOWNLOAD_TICKERS,
    FIXED_INCOME_TICKERS,
    TICKERS,
    WEIGHTS,
)
from risk.metrics import BLEND_WEIGHTS

# Transcribed from the workbook, 16 positions, sum 0.945.
EXPECTED_WEIGHTS = {
    "AAPL": 0.05, "MSFT": 0.06, "MU": 0.10, "DAL": 0.015, "WMT": 0.04, "IAG": 0.005,
    "SPY": 0.14, "VT": 0.10, "SPCX": 0.03, "EEM": 0.03, "XLV": 0.06, "XLF": 0.05,
    "VGIT": 0.10, "VTIP": 0.06, "JPIE": 0.06, "MINT": 0.045,
}


def test_cat_is_not_held():
    assert "CAT" not in TICKERS
    assert "CAT" not in WEIGHTS


def test_tickers_and_weights_agree():
    assert set(TICKERS) == set(WEIGHTS)
    assert len(TICKERS) == 15


def test_weights_match_excat_workbook():
    assert ORIGINAL_WEIGHTS == EXPECTED_WEIGHTS
    assert sum(ORIGINAL_WEIGHTS.values()) == pytest.approx(0.945)


def test_active_weights_are_proportional_with_explicit_cash():
    assert set(WEIGHTS) == set(ORIGINAL_WEIGHTS) - {"SPCX"}
    assert "SPCX" not in DOWNLOAD_TICKERS
    assert CASH_WEIGHT == 0.01
    assert sum(WEIGHTS.values()) == pytest.approx(0.99)
    assert sum(WEIGHTS.values()) + CASH_WEIGHT == pytest.approx(1)
    for ticker, weight in WEIGHTS.items():
        assert weight == pytest.approx(EXPECTED_WEIGHTS[ticker] * 0.99 / 0.915)


def test_benchmark_tickers_are_downloaded_but_not_held():
    assert BENCHMARK_TICKERS == ["URTH", "BNDW"]
    assert set(BLEND_WEIGHTS) <= set(BENCHMARK_TICKERS)
    assert not set(BENCHMARK_TICKERS) & set(WEIGHTS)
    assert DOWNLOAD_TICKERS == TICKERS + BENCHMARK_TICKERS


def test_asset_classes_partition_tickers():
    sleeves = list(ASSET_CLASSES.values())
    union = set().union(*sleeves)
    assert union == set(TICKERS)
    assert sum(len(s) for s in sleeves) == len(union), "a ticker appears in two sleeves"
    assert ASSET_CLASSES["Fixed Income ETFs"] == FIXED_INCOME_TICKERS
