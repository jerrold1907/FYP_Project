"""Unit tests for the feature engineering module.

Tests compute_features() and compute_target_labels() with known inputs
and hand-computed expected outputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import compute_features, compute_target_labels


class TestComputeFeatures:
    """Tests for the compute_features function."""

    def _make_price_df(self, n_days: int = 100, start_price: float = 100.0) -> pd.DataFrame:
        """Create a synthetic price DataFrame with n_days of data."""
        np.random.seed(42)
        # Generate a realistic price series using random walk
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = start_price * np.cumprod(1 + returns)
        volumes = np.random.randint(1_000_000, 10_000_000, n_days)

        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        return pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Adj Close": prices,
            "Volume": volumes,
        }, index=dates)

    def test_requires_minimum_80_days(self):
        """Fewer than 80 trading days raises ValueError."""
        df = self._make_price_df(n_days=79)
        with pytest.raises(ValueError, match="at least 80 are required"):
            compute_features(df)

    def test_accepts_exactly_80_days(self):
        """Exactly 80 trading days should work without error."""
        df = self._make_price_df(n_days=80)
        result = compute_features(df)
        assert len(result) > 0

    def test_output_columns(self):
        """Output DataFrame has all expected feature columns."""
        df = self._make_price_df(n_days=100)
        result = compute_features(df)
        expected_columns = [
            "close_price", "daily_return", "ma_5", "ma_20", "ma_50",
            "volatility", "volume", "RSI", "MACD",
        ]
        assert list(result.columns) == expected_columns

    def test_no_nan_values(self):
        """Output contains no NaN values after dropping."""
        df = self._make_price_df(n_days=100)
        result = compute_features(df)
        assert result.isna().sum().sum() == 0

    def test_rsi_range(self):
        """RSI values are bounded between 0 and 100."""
        df = self._make_price_df(n_days=200)
        result = compute_features(df)
        assert (result["RSI"] >= 0).all()
        assert (result["RSI"] <= 100).all()

    def test_ma_5_calculation(self):
        """5-day SMA is correctly computed as mean of last 5 close prices."""
        df = self._make_price_df(n_days=100)
        result = compute_features(df)
        close = df["Adj Close"]

        # Check the 50th row in the result (after NaN rows are dropped)
        # Get a row index that's definitely in the result
        # After dropping NaNs, the first valid row should be around index 49
        # (ma_50 needs 50 days, so first valid is day 49)
        idx = result.index[5]  # Pick a row well into the result
        pos = df.index.get_loc(idx)

        expected_ma5 = close.iloc[pos - 4: pos + 1].mean()
        assert np.isclose(result.loc[idx, "ma_5"], expected_ma5, rtol=1e-10)

    def test_daily_return_calculation(self):
        """Daily return matches (close[t] - close[t-1]) / close[t-1]."""
        df = self._make_price_df(n_days=100)
        result = compute_features(df)
        close = df["Adj Close"]

        idx = result.index[5]
        pos = df.index.get_loc(idx)

        expected_return = (close.iloc[pos] - close.iloc[pos - 1]) / close.iloc[pos - 1]
        assert np.isclose(result.loc[idx, "daily_return"], expected_return, rtol=1e-10)

    def test_macd_calculation(self):
        """MACD equals EMA(12) - EMA(26) of close prices."""
        df = self._make_price_df(n_days=100)
        result = compute_features(df)
        close = df["Adj Close"]

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        expected_macd = ema_12 - ema_26

        # Check several rows
        for idx in result.index[10:15]:
            assert np.isclose(
                result.loc[idx, "MACD"], expected_macd.loc[idx], rtol=1e-10
            )

    def test_uses_close_when_adj_close_missing(self):
        """Falls back to 'Close' column when 'Adj Close' is not present."""
        df = self._make_price_df(n_days=100)
        df = df.drop(columns=["Adj Close"])
        result = compute_features(df)
        assert len(result) > 0
        assert result.isna().sum().sum() == 0


class TestComputeTargetLabels:
    """Tests for the compute_target_labels function."""

    def _make_price_df(self, prices: list[float]) -> pd.DataFrame:
        """Create a DataFrame from a list of prices."""
        dates = pd.date_range("2023-01-01", periods=len(prices), freq="B")
        return pd.DataFrame({
            "Close": prices,
            "Adj Close": prices,
            "Volume": [1_000_000] * len(prices),
        }, index=dates)

    def test_buy_label_above_10_percent(self):
        """Future return > 10% is labeled as Buy."""
        # Create prices where day 0 = 100, day 30 = 115 (15% return)
        prices = [100.0] * 31
        prices[30] = 115.0  # 15% return from day 0
        df = self._make_price_df(prices)
        result = compute_target_labels(df)
        # First row: future return = (115 - 100) / 100 = 0.15 > 0.10 -> Buy
        assert result.iloc[0]["target"] == "Buy"

    def test_hold_label_between_0_and_10_percent(self):
        """Future return between 0% and 10% is labeled as Hold."""
        # Create prices where day 0 = 100, day 30 = 105 (5% return)
        prices = [100.0] * 31
        prices[30] = 105.0  # 5% return from day 0
        df = self._make_price_df(prices)
        result = compute_target_labels(df)
        assert result.iloc[0]["target"] == "Hold"

    def test_hold_label_at_zero_percent(self):
        """Future return of exactly 0% is labeled as Hold."""
        prices = [100.0] * 31  # no change
        df = self._make_price_df(prices)
        result = compute_target_labels(df)
        assert result.iloc[0]["target"] == "Hold"

    def test_hold_label_at_10_percent(self):
        """Future return of exactly 10% is labeled as Hold (inclusive)."""
        prices = [100.0] * 31
        prices[30] = 110.0  # exactly 10%
        df = self._make_price_df(prices)
        result = compute_target_labels(df)
        assert result.iloc[0]["target"] == "Hold"

    def test_avoid_label_below_0_percent(self):
        """Future return < 0% is labeled as Avoid."""
        prices = [100.0] * 31
        prices[30] = 95.0  # -5% return
        df = self._make_price_df(prices)
        result = compute_target_labels(df)
        assert result.iloc[0]["target"] == "Avoid"

    def test_drops_rows_without_future_data(self):
        """Last 30 rows are dropped since no 30-day future data is available."""
        prices = [100.0 + i for i in range(50)]
        df = self._make_price_df(prices)
        result = compute_target_labels(df)
        # Should have 50 - 30 = 20 rows
        assert len(result) == 20

    def test_output_columns(self):
        """Output has future_30_day_return and target columns."""
        prices = [100.0 + i * 0.5 for i in range(50)]
        df = self._make_price_df(prices)
        result = compute_target_labels(df)
        assert "future_30_day_return" in result.columns
        assert "target" in result.columns

    def test_no_nan_in_output(self):
        """Output contains no NaN values."""
        prices = [100.0 + i * 0.5 for i in range(50)]
        df = self._make_price_df(prices)
        result = compute_target_labels(df)
        assert result.isna().sum().sum() == 0

    def test_uses_close_when_adj_close_missing(self):
        """Falls back to 'Close' column when 'Adj Close' is not present."""
        prices = [100.0] * 31
        prices[30] = 115.0
        df = self._make_price_df(prices)
        df = df.drop(columns=["Adj Close"])
        result = compute_target_labels(df)
        assert result.iloc[0]["target"] == "Buy"
