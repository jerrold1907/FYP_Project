"""Unit tests for backtest cost accounting and the chance-matched baseline.

The model inside the walk-forward loop is replaced by a stub that always
predicts one signal, so the expected equity can be computed by hand from the
prices. That isolates the accounting from the classifier.
"""

import numpy as np
import pandas as pd
import pytest

import src.backtester as backtester
from src.backtester import (
    BacktestConfig,
    random_entry_win_probability,
    run_backtest,
)

#: Raw rows giving exactly one 500 + 60 day window: compute_features drops 49
#: warm-up rows and compute_target_labels drops the last 30.
ONE_WINDOW_ROWS = 560 + 49 + 30


class _ConstantModel:
    """Stand-in classifier that predicts the same signal for every row."""

    def __init__(self, signal):
        self.signal = signal
        self.training_labels = None

    def fit(self, X, y):
        self.training_labels = np.asarray(y)
        return self

    def predict(self, X):
        return np.array([self.signal] * len(X))


def _prices(n_rows=ONE_WINDOW_ROWS, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n_rows))
    index = pd.bdate_range("2020-01-01", periods=n_rows)
    return pd.DataFrame({"Close": close, "Volume": 1_000_000.0}, index=index)


def _stub(monkeypatch, signal):
    model = _ConstantModel(signal)
    monkeypatch.setattr(backtester, "_build_model", lambda model_type: model)
    return model


def _test_window(df):
    """Closing prices of the single test window, as run_backtest sees them."""
    from src.features import compute_features, compute_target_labels
    common = compute_features(df).index.intersection(
        compute_target_labels(df).index)
    return df["Close"].loc[common].iloc[500:560].to_numpy()


class TestHoldCostModel:

    def test_on_change_charges_only_entry_and_exit(self, monkeypatch):
        _stub(monkeypatch, "Hold")
        df = _prices()
        cost = 0.001
        result = run_backtest(df, BacktestConfig(transaction_cost_pct=cost,
                                                 hold_cost_model="on_change"))

        prices = _test_window(df)
        daily = prices[1:] / prices[:-1] - 1          # 59 tradable days
        charges = np.zeros_like(daily)
        charges[0] += cost                            # opened on the first day
        charges[-1] += cost                           # closed on the last
        expected = np.prod(1 + 0.5 * (daily - charges)) - 1

        assert result.total_return_pct == pytest.approx(expected * 100)

    def test_per_day_reproduces_the_original_accounting(self, monkeypatch):
        _stub(monkeypatch, "Hold")
        df = _prices()
        cost = 0.001
        result = run_backtest(df, BacktestConfig(transaction_cost_pct=cost,
                                                 hold_cost_model="per_day"))

        prices = _test_window(df)
        daily = prices[1:] / prices[:-1] - 1
        expected = np.prod(1 + 0.5 * (daily - 2 * cost)) - 1

        assert result.total_return_pct == pytest.approx(expected * 100)

    def test_models_agree_when_trading_is_free(self, monkeypatch):
        _stub(monkeypatch, "Hold")
        df = _prices()
        a = run_backtest(df, BacktestConfig(transaction_cost_pct=0.0,
                                            hold_cost_model="per_day"))
        b = run_backtest(df, BacktestConfig(transaction_cost_pct=0.0,
                                            hold_cost_model="on_change"))
        assert a.total_return_pct == pytest.approx(b.total_return_pct)

    def test_on_change_is_the_default(self):
        assert BacktestConfig().hold_cost_model == "on_change"

    def test_unknown_cost_model_rejected(self):
        with pytest.raises(ValueError, match="hold_cost_model"):
            run_backtest(_prices(), BacktestConfig(hold_cost_model="weekly"))


class TestBuyThresholdPassThrough:

    def test_window_models_train_on_the_configured_labels(self, monkeypatch):
        df = _prices()
        counts = {}
        for threshold in (0.05, 0.15):
            model = _stub(monkeypatch, "Avoid")
            run_backtest(df, BacktestConfig(buy_threshold=threshold))
            counts[threshold] = int((model.training_labels == "Buy").sum())
        assert counts[0.05] > counts[0.15]


class TestRandomEntryWinProbability:

    def test_rising_prices_always_profit_without_costs(self):
        prices = pd.Series(np.linspace(100, 130, 40))
        assert random_entry_win_probability(prices, 3, 0.0) == 1.0

    def test_falling_prices_never_profit(self):
        prices = pd.Series(np.linspace(130, 100, 40))
        assert random_entry_win_probability(prices, 3, 0.0) == 0.0

    def test_costs_can_turn_small_gains_into_losses(self):
        prices = pd.Series(100 * 1.0005 ** np.arange(40))   # +0.05% a day
        assert random_entry_win_probability(prices, 1, 0.0) == 1.0
        assert random_entry_win_probability(prices, 1, 0.002) == 0.0

    @pytest.mark.parametrize("holding_days", [0, 40, 41])
    def test_degenerate_holding_periods_return_zero(self, holding_days):
        prices = pd.Series(np.linspace(100, 130, 40))
        assert random_entry_win_probability(prices, holding_days, 0.0) == 0.0

    def test_every_trade_carries_a_probability(self, monkeypatch):
        _stub(monkeypatch, "Buy")
        result = run_backtest(_prices())
        assert result.trades
        for trade in result.trades:
            assert 0.0 <= trade.random_entry_win_prob <= 1.0
