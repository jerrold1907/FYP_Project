"""The backtest page's Sharpe interval must be formed on daily returns.

An earlier version applied Lo's (2002) standard error to the annualised ratio,
which made the interval over ten times too narrow: for AAPL, MSFT and MU
it showed [0.53, 0.70] where the per-ticker intervals all include zero.
"""

import numpy as np
import pytest

from src.statistics_tests import sharpe_ratio_interval

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def combined_sharpe():
    from app import _combined_sharpe
    return _combined_sharpe


def _curve(returns):
    """Cumulative return in percent, starting at 0, as the page builds it."""
    values = np.concatenate([[1.0], np.cumprod(1 + returns)])
    return list((values - 1) * 100)


def test_interval_is_lo_on_the_daily_returns(combined_sharpe):
    returns = np.random.default_rng(0).normal(0.0006, 0.01, 750)
    got = combined_sharpe(_curve(returns))
    expected = sharpe_ratio_interval(returns)
    assert got.estimate == pytest.approx(expected.estimate)
    assert got.lower == pytest.approx(expected.lower)
    assert got.upper == pytest.approx(expected.upper)


def test_interval_has_the_width_of_three_years_of_data(combined_sharpe):
    # About 750 trading days pin an annualised Sharpe ratio down only to
    # roughly +/-1.1; the old formula reported about +/-0.1.
    returns = np.random.default_rng(1).normal(0.0006, 0.01, 750)
    got = combined_sharpe(_curve(returns))
    assert got.upper - got.lower > 2.0


def test_short_or_flat_curves_have_no_ratio(combined_sharpe):
    assert combined_sharpe([0.0, 1.0]) is None
    assert combined_sharpe([0.0] * 10) is None
