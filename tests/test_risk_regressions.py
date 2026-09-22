"""Independent numerical examples and failure cases for the risk report."""
import warnings

import numpy as np
import pandas as pd
import pytest

from data import fetch_prices as data
from portfolio.attribution import return_attribution, asset_class_attribution
from risk.metrics import (
    portfolio_returns, max_drawdown, sortino_ratio, sharpe_ratio,
    historical_var, historical_cvar, annualised_return, risk_summary,
)
from risk.stress_test import scenario_returns, run_all_scenarios
from risk.fixed_income import price_change_from_shock, yield_shock_pnl, FI_PROFILES


def frame():
    return pd.DataFrame({'A': [100., 110., 99.], 'B': [100., 90., 99.]},
                        index=pd.bdate_range('2022-01-03', periods=3))


def test_drawdown_includes_initial_wealth():
    value, series = max_drawdown(pd.Series([-.1, -.1]))
    assert value == pytest.approx(-.19)
    np.testing.assert_allclose(series, [-.1, -.19])
    assert max_drawdown(pd.Series([.1, .1]))[0] == 0


def test_sortino_full_population_hand_example():
    # Mean 2%; downside RMS sqrt((.02**2 + .04**2) / 4).
    r = pd.Series([.10, -.02, .04, -.04])
    assert sortino_ratio(r, 0) == pytest.approx(.02 / np.sqrt(.002 / 4) * np.sqrt(252))


def test_undefined_ratios():
    assert np.isnan(sortino_ratio(pd.Series([.01, .02]), 0))
    assert np.isnan(sharpe_ratio(pd.Series([.01, .01]), 0))
    assert np.isnan(sharpe_ratio(pd.Series([.01]), 0))


def test_expected_shortfall_ties_and_fractional_boundary():
    assert historical_cvar(pd.Series([-.1] + [-.05] * 9 + [.01] * 90), .95) == pytest.approx(.06)
    assert historical_cvar(pd.Series([-.1, -.04, .01]), .5) == pytest.approx(.08)
    assert historical_cvar(pd.Series([-.1, -.04, .01]), .99) == pytest.approx(.1)


@pytest.mark.parametrize('confidence', [0, 1, -1, np.nan])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        historical_cvar(pd.Series([-.1, .1]), confidence)


@pytest.mark.parametrize('values', [[], [np.nan], [np.inf], [-1.1]])
def test_invalid_return_samples(values):
    r = pd.Series(values, dtype=float)
    for fn in [historical_var, historical_cvar, max_drawdown, annualised_return, sharpe_ratio, sortino_ratio]:
        with pytest.raises(ValueError):
            fn(r)


def test_cash_return_and_attribution_reconcile_by_hand():
    p = frame()
    # 49.5% in each security and 1% cash with 1% interest per observation.
    weights = {'A': .495, 'B': .495}
    r = portfolio_returns(p, weights, .01, 2.52)
    np.testing.assert_allclose(r, [.0001, .0001], atol=1e-14)
    attr = return_attribution(p, weights, .01, 2.52).set_index('Ticker')
    # A: +.0495 on first day, -.0495 * 1.0001 on second day.
    assert attr.loc['A', 'Contribution (%)'] / 100 == pytest.approx(-.00000495)
    assert attr.loc['B', 'Contribution (%)'] / 100 == pytest.approx(.00000495)
    assert attr.loc['CASH', 'Contribution (%)'] / 100 == pytest.approx(.00020001)
    assert attr.loc['TOTAL', 'Contribution (%)'] / 100 == pytest.approx((1+r).prod()-1, abs=1e-10)
    assert attr.loc['TOTAL', 'Weight (%)'] == pytest.approx(100)


@pytest.mark.parametrize('weights,cash', [({'A': 1.1}, None), ({'A': -.1}, None),
                                           ({'A': np.nan}, None), ({'A': .9}, .01)])
def test_invalid_weights(weights, cash):
    with pytest.raises(ValueError):
        portfolio_returns(frame(), weights, cash)


def test_price_gaps_never_become_zero_returns():
    for row in [1, 2]:
        p = frame()
        p.iloc[row, 0] = np.nan
        with pytest.raises(ValueError, match='missing prices'):
            portfolio_returns(p, {'A': .5, 'B': .5})
    p = frame()
    p.iloc[0, 0] = np.nan
    assert len(portfolio_returns(p, {'A': .5, 'B': .5})) == 1


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'unsorted', 'zero', 'infinite'])
def test_bad_prices(mode):
    p = frame()
    if mode == 'missing':
        p = p.drop(columns='A')
    elif mode == 'duplicate':
        p.index = [p.index[0], p.index[0], p.index[2]]
    elif mode == 'unsorted':
        p = p.iloc[::-1]
    elif mode == 'zero':
        p.iloc[0, 0] = 0
    else:
        p.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        portfolio_returns(p, {'A': .5, 'B': .5})


def test_missing_holdings_cannot_disappear_from_attribution_or_stress():
    for fn in [lambda: return_attribution(frame(), {'C': 1}),
               lambda: scenario_returns(frame(), '2022-01-03', '2022-01-05', {'C': 1})]:
        with pytest.raises(ValueError, match='Missing price columns'):
            fn()


def test_partial_stress_window_is_unavailable():
    p = frame()
    with pytest.raises(ValueError, match='endpoints'):
        scenario_returns(p, '2022-01-03', '2022-10-13', {'A': 1})
    report = run_all_scenarios(p, {'A': 1})
    assert len(report) == 4
    assert report['Status'].str.startswith('Unavailable:').all()
    assert report['Portfolio Return (%)'].isna().all()


def test_buy_and_hold_stress_cash():
    result = scenario_returns(frame(), '2022-01-03', '2022-01-05', {'A': .99}, .01, 2.52)
    assert result['portfolio'] == pytest.approx(.99 * -.01 + .01 * .0201)


def test_staleness_threshold():
    with pytest.warns(UserWarning, match='Stale price cache'):
        data.warn_if_stale(frame(), today='2022-01-11')
    with warnings.catch_warnings(record=True) as captured:
        data.warn_if_stale(frame(), today='2022-01-10')
    assert not captured


def test_cache_validation_and_extra_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'DATA_DIR', tmp_path)
    p = pd.DataFrame({t: [100., 101., 102.] for t in data.DOWNLOAD_TICKERS},
                     index=pd.bdate_range(pd.Timestamp.today().normalize() - pd.offsets.BDay(3), periods=3))
    p['SPCX'] = np.nan
    p['CAT'] = -1
    p.to_csv(tmp_path / 'prices.csv')
    assert list(data.load_prices().columns) == data.DOWNLOAD_TICKERS
    p.iloc[1, 0] = np.nan
    p.to_csv(tmp_path / 'prices.csv')
    with pytest.raises(ValueError):
        data.load_prices()


def test_download_failure_does_not_replace_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(data.time, 'sleep', lambda _: None)
    monkeypatch.setattr(data.yf, 'download', lambda *a, **k: pd.DataFrame())
    cached = tmp_path / 'prices.csv'
    cached.write_text('original cache')
    with pytest.raises(RuntimeError, match='No price data'):
        data.load_prices(refresh=True)
    assert cached.read_text() == 'original cache'


def test_risk_summary_rejects_nan_and_disjoint_dates():
    p = pd.Series([.01, np.nan], index=[1, 2])
    with pytest.raises(ValueError):
        risk_summary(p, p)
    with pytest.raises(ValueError):
        risk_summary(pd.Series([.01], index=[1]), pd.Series([.02], index=[2]))


def test_duration_only_and_dollar_scaling():
    assert all(profile['convexity'] is None for profile in FI_PROFILES.values())
    assert price_change_from_shock(5, None, .01) == pytest.approx(-.05)
    assert price_change_from_shock(5, 30, .01) == pytest.approx(-.0485)
    df = yield_shock_pnl({'VGIT': .1}, portfolio_value=1_000_000).set_index('Ticker')
    assert df.loc['VGIT', 'DV01 ($)'] == pytest.approx(49)
    assert df.loc['TOTAL', '+100bps P&L ($)'] == pytest.approx(-4900)


def test_successful_download_validates_without_filling(monkeypatch):
    raw = pd.DataFrame({('Close', t): [100., 101., 102.] for t in data.DOWNLOAD_TICKERS},
                       index=pd.bdate_range('2026-01-01', periods=3))
    monkeypatch.setattr(data.yf, 'download', lambda *a, **k: raw)
    assert data.fetch_prices().shape == (3, len(data.DOWNLOAD_TICKERS))
    raw.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match='missing prices'):
        data.fetch_prices()


def test_real_cache_attribution_and_sleeves_reconcile():
    p = pd.read_csv(data.DATA_DIR / 'prices.csv', index_col=0, parse_dates=True)
    r = portfolio_returns(p, data.WEIGHTS, data.CASH_WEIGHT)
    attr = return_attribution(p, data.WEIGHTS, data.CASH_WEIGHT)
    sleeves = asset_class_attribution(p, data.WEIGHTS, data.CASH_WEIGHT)
    total = (1+r).prod()-1
    assert attr.iloc[-1]['Contribution (%)'] / 100 == pytest.approx(total, abs=1e-10)
    assert sleeves['Contribution (%)'].sum() / 100 == pytest.approx(total, abs=1e-10)
    assert sleeves['Weight (%)'].sum() == pytest.approx(100)


def test_cli_pipeline_includes_cash_and_generates_six_charts(tmp_path, monkeypatch, capsys):
    import matplotlib
    matplotlib.use('Agg')
    import main
    import visualisation.charts as charts
    monkeypatch.setattr(charts, 'CHARTS_DIR', tmp_path)
    rng = np.random.default_rng(7)
    p = pd.DataFrame({t: 100*np.cumprod(1+rng.normal(0, .01, 70)) for t in data.DOWNLOAD_TICKERS},
                     index=pd.bdate_range('2026-01-01', periods=70))
    monkeypatch.setattr(main, 'load_prices', lambda **kwargs: p)
    main.main(cash_rate=.04)
    output = capsys.readouterr().out
    assert 'CASH' in output
    assert '99% securities + 1% cash' in output
    assert 'SPCX' not in output
    assert 'Unavailable:' in output
    assert 'UNVERIFIED legacy assumption' in output
    assert len(list(tmp_path.glob('*.png'))) == 6
    assert all(path.stat().st_size > 1000 for path in tmp_path.glob('*.png'))
