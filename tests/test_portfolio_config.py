"""
Portfolio configuration must match the ex-CAT analysis workbook.

Reference: data/GPP2026_Complete_Analysis_exCAT.xlsx, sheet "Position PL Tracker",
column C (Weight). CAT was never held; see sheet "CAT Removal Log".
"""

import pytest

from data.fetch_prices import (
    ASSET_CLASSES,
    BACKTEST_WEIGHTS,
    FIXED_INCOME_TICKERS,
    TICKERS,
    WEIGHTS,
)

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
    assert len(TICKERS) == 16


def test_weights_match_excat_workbook():
    assert WEIGHTS == EXPECTED_WEIGHTS
    assert sum(WEIGHTS.values()) == pytest.approx(0.945)


def test_backtest_weights_exclude_only_spcx():
    assert set(BACKTEST_WEIGHTS) == set(WEIGHTS) - {"SPCX"}
    for ticker, weight in BACKTEST_WEIGHTS.items():
        assert weight == WEIGHTS[ticker]


def test_asset_classes_partition_tickers():
    sleeves = list(ASSET_CLASSES.values())
    union = set().union(*sleeves)
    assert union == set(TICKERS)
    assert sum(len(s) for s in sleeves) == len(union), "a ticker appears in two sleeves"
    assert ASSET_CLASSES["Fixed Income ETFs"] == FIXED_INCOME_TICKERS
