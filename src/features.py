"""Feature Engineering Module.

Computes technical indicators from raw OHLCV stock data for use in
machine learning model training and prediction.

This module provides two main functions:
- compute_features(): Calculates technical indicators including moving averages
  (5, 20, 50-day SMA), daily returns, volatility, RSI (14-day with Wilder's
  smoothing), and MACD (EMA12 - EMA26) from raw price data.
- compute_target_labels(): Generates classification labels (Buy/Hold/Avoid)
  based on 30-day forward-looking returns.

Stocks with fewer than 80 trading days are excluded from computation because:
- The 50-day moving average requires at least 50 data points
- RSI calculation needs 14 days for initial smoothing
- EMA(26) needs approximately 26+ days to stabilize
- Combined with warm-up periods, 80 days ensures all indicators are meaningful
"""

import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features from raw stock price data.

    Takes a DataFrame containing at least 'Close' (or 'Adj Close') and 'Volume'
    columns with daily OHLCV data and returns a new DataFrame with computed
    technical indicators.

    Features computed:
        - close_price: Raw adjusted close price (or Close if Adj Close unavailable)
        - daily_return: Percentage change from previous day's close.
          Formula: (close[t] - close[t-1]) / close[t-1]
        - ma_5: 5-day Simple Moving Average of close price.
          Formula: mean(close[t-4:t+1])
        - ma_20: 20-day Simple Moving Average of close price.
          Formula: mean(close[t-19:t+1])
        - ma_50: 50-day Simple Moving Average of close price.
          Formula: mean(close[t-49:t+1])
        - volatility: 20-day rolling standard deviation of daily returns.
          Measures recent price dispersion/risk.
        - volume: Raw trading volume (number of shares traded).
        - RSI: 14-day Relative Strength Index using Wilder's smoothing method.
          RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss.
          Wilder's smoothing: avg = (prev_avg * 13 + current) / 14
        - MACD: Moving Average Convergence Divergence.
          Formula: EMA(12, close) - EMA(26, close)
          Uses standard exponential moving average with span parameter.

    Args:
        df: A pandas DataFrame with at least 'Close' (or 'Adj Close') and
            'Volume' columns, indexed or containing daily trading data.
            Must contain at least 80 trading days.

    Returns:
        A pandas DataFrame with columns: close_price, daily_return, ma_5,
        ma_20, ma_50, volatility, volume, RSI, MACD. All rows containing
        NaN values are dropped (caused by initial periods of rolling windows).

    Raises:
        ValueError: If the input DataFrame has fewer than 80 trading days.
    """
    # Require at least 80 trading days to ensure all rolling window indicators
    # (especially 50-day MA and EMA26) have enough data to stabilize
    if len(df) < 80:
        raise ValueError(
            f"Insufficient data: {len(df)} trading days provided, "
            f"but at least 80 are required. The 50-day moving average needs "
            f"50 days, RSI needs 14 days, and EMA(26) needs stabilization time."
        )

    # Handle MultiIndex columns (yfinance may return these)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Use Adj Close if available, otherwise fall back to Close
    close = df["Adj Close"].copy() if "Adj Close" in df.columns else df["Close"].copy()

    # Initialize result DataFrame
    result = pd.DataFrame(index=df.index)

    # close_price: Raw adjusted close price
    result["close_price"] = close.values

    # daily_return: percentage change from previous day
    # Formula: (close[t] - close[t-1]) / close[t-1]
    # First day will be NaN since there's no previous day
    result["daily_return"] = close.pct_change().values

    # ma_5: 5-day Simple Moving Average of close price
    # Averages the last 5 days of closing prices (including current day)
    result["ma_5"] = close.rolling(window=5).mean().values

    # ma_20: 20-day Simple Moving Average of close price
    # Averages the last 20 days of closing prices (including current day)
    result["ma_20"] = close.rolling(window=20).mean().values

    # ma_50: 50-day Simple Moving Average of close price
    # Averages the last 50 days of closing prices (including current day)
    # This is why we need at least 50+ days of data
    result["ma_50"] = close.rolling(window=50).mean().values

    # volatility: 20-day rolling standard deviation of daily returns
    # Measures the dispersion of returns over the last 20 trading days
    # Higher volatility indicates more price uncertainty/risk
    daily_returns = close.pct_change()
    result["volatility"] = daily_returns.rolling(window=20).std().values

    # volume: Raw trading volume (number of shares traded)
    result["volume"] = df["Volume"].values

    # RSI: 14-day Relative Strength Index using Wilder's smoothing
    # Step 1: Calculate price changes (gains and losses)
    delta = close.diff()

    # Step 2: Separate gains (positive changes) and losses (negative changes)
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Step 3: Apply Wilder's smoothing (exponential moving average with alpha=1/14)
    # Wilder's smoothing formula: avg[t] = (prev_avg * (n-1) + current) / n
    # This is equivalent to EWM with alpha = 1/n (or com = n-1)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # Step 4: Calculate RS (Relative Strength) and RSI
    # RS = average gain / average loss
    # RSI = 100 - (100 / (1 + RS))
    # When avg_loss is 0, RSI = 100 (all gains, no losses)
    rs = avg_gain / avg_loss
    result["RSI"] = (100.0 - (100.0 / (1.0 + rs))).values

    # MACD: Moving Average Convergence Divergence
    # Formula: EMA(12, close) - EMA(26, close)
    # EMA(12) is the fast-moving average, reacts quickly to price changes
    ema_12 = close.ewm(span=12, adjust=False).mean()
    # EMA(26) is the slow-moving average, smooths out short-term fluctuations
    ema_26 = close.ewm(span=26, adjust=False).mean()
    # MACD line is the difference: positive = bullish momentum, negative = bearish
    result["MACD"] = (ema_12 - ema_26).values

    # Drop all rows with NaN values
    # NaN values are caused by the initial periods of rolling window calculations:
    # - daily_return: first row is NaN (no previous day)
    # - ma_5: first 4 rows are NaN
    # - ma_20: first 19 rows are NaN
    # - ma_50: first 49 rows are NaN (largest contributor)
    # - volatility: first 20 rows of daily_return are NaN (needs 20 returns + 1 for pct_change)
    # - RSI: first 13 rows are NaN (min_periods=14)
    # After dropping, all remaining rows have complete, meaningful feature values
    result = result.dropna()

    return result


def compute_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target classification labels based on 30-day forward returns.

    Calculates the future 30-day return for each row and assigns a label:
        - "Buy": future_30_day_return > 10% (strong positive expected return)
        - "Hold": 0% <= future_30_day_return <= 10% (moderate/flat expected return)
        - "Avoid": future_30_day_return < 0% (negative expected return)

    The future_30_day_return is calculated as:
        (close[t+30] - close[t]) / close[t]

    This represents the percentage gain or loss if a stock is held for the
    next 30 trading days from the current day.

    Args:
        df: A pandas DataFrame with at least a 'Close' or 'Adj Close' column.
            The DataFrame should have sufficient rows to compute forward-looking
            returns (the last 30 rows will have NaN labels and be dropped).

    Returns:
        A pandas DataFrame with columns:
            - future_30_day_return: The percentage return over the next 30 trading days
            - target: The classification label ("Buy", "Hold", or "Avoid")
        Rows with NaN values (last 30 rows with insufficient future data) are dropped.
    """
    # Handle MultiIndex columns (yfinance may return these)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Use Adj Close if available, otherwise fall back to Close
    close = df["Adj Close"].copy() if "Adj Close" in df.columns else df["Close"].copy()

    result = pd.DataFrame(index=df.index)

    # Compute future 30-day return: percentage change from current close
    # to the close price 30 trading days ahead
    # Formula: (close[t+30] - close[t]) / close[t]
    # shift(-30) moves future values back to align with current row
    result["future_30_day_return"] = (close.shift(-30) - close) / close

    # Assign target labels based on future return thresholds:
    # - Buy: return exceeds 10% (strong growth expected)
    # - Hold: return between 0% and 10% inclusive (moderate/stable)
    # - Avoid: return below 0% (decline expected)
    # We use pd.cut-style logic with np.select; conditions are mutually exclusive
    # and exhaustive for non-NaN values, so default should never be reached
    conditions = [
        result["future_30_day_return"] > 0.10,   # Buy: > 10%
        result["future_30_day_return"] >= 0.0,    # Hold: 0% to 10% (inclusive)
        result["future_30_day_return"] < 0.0,     # Avoid: < 0%
    ]
    labels = ["Buy", "Hold", "Avoid"]

    # Use empty string as default for rows where future_30_day_return is NaN
    # (last 30 rows without sufficient future data)
    result["target"] = np.select(conditions, labels, default="")

    # Drop rows where future_30_day_return is NaN (last 30 rows with insufficient
    # future data). These rows cannot have a target label because we don't have
    # 30 days of future price data to compute the forward return
    result = result.dropna(subset=["future_30_day_return"])

    return result
