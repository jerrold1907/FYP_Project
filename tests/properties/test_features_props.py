"""Property-based tests for the Feature Engineering module.

Tests the following properties for feature engineering correctness:
- Property 4: Feature pipeline produces no NaN values
- Property 5: Feature computations match their mathematical definitions

Each technical indicator is verified by an independent (oracle) computation that
does not share code with the implementation under test. This ensures that errors
in the implementation are caught by comparing against a known-correct formula.

Why property-based testing here:
    Stock price data has enormous variability (prices from fractions of a penny to
    thousands of dollars, volumes from 0 to billions). Hand-picked examples cannot
    cover edge cases like constant prices (RSI = 100), monotonically decreasing
    prices (RSI near 0), or very small/large price movements. Hypothesis explores
    these boundary conditions automatically.

    NaN values in the feature output would corrupt downstream model training.
    scikit-learn models cannot handle NaN inputs — they either raise errors or
    produce undefined predictions. Verifying NaN-freedom as a universal property
    across randomly generated price series ensures the feature pipeline's internal
    dropna() logic correctly handles all rolling window warm-up periods regardless
    of the shape or distribution of input data.
"""

import numpy as np
import pandas as pd
from hypothesis import given, assume, settings
from hypothesis.strategies import (
    integers,
    floats,
    composite,
    lists,
)

from src.features import compute_features, compute_target_labels


# =============================================================================
# Feature: ai-stock-recommendation, Property 4: Feature pipeline produces no NaN values
# =============================================================================


@composite
def valid_price_dataframes(draw):
    """Composite strategy generating valid stock price DataFrames for NaN testing.

    Generates a DataFrame with Close and Volume columns containing at least 80
    rows of valid (non-NaN) data, suitable as input to compute_features().

    Strategy design rationale:
        - num_days: Random integer in [80, 300]. The minimum of 80 satisfies
          compute_features()'s requirement (50-day MA needs 50 points, RSI needs
          14, EMA(26) needs ~26 days, combined with warm-up = 80 minimum). The
          upper bound of 300 keeps test execution fast while exploring longer
          series where cumulative floating-point effects could introduce NaN.

        - Close prices: Generated as a random walk starting from a positive
          initial price. Each day's return is drawn from [-0.08, 0.08] (±8%
          daily, realistic for volatile stocks). The cumulative product produces
          realistic correlated price series. A floor of 0.01 guarantees no
          zero or negative prices (which would cause division-by-zero in
          pct_change calculations and potentially produce NaN).

        - Volume: Random positive integers in [100, 10_000_000]. Volume must
          be positive and non-NaN. The range covers low-liquidity to large-cap
          stocks.

    The composite pattern allows Hypothesis to shrink each component
    independently, producing minimal counterexamples on failure.
    """
    # Number of trading days — at least 80 required by compute_features()
    # Upper bound of 300 keeps tests fast while exploring longer series
    num_days = draw(integers(min_value=80, max_value=300))

    # Starting price: a positive float representing the initial stock price
    # Range [1.0, 1000.0] covers penny stocks to high-priced stocks
    start_price = draw(
        floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )

    # Daily returns: small percentage changes that simulate realistic price movement
    # Range [-0.08, 0.08] means max ±8% daily change (realistic for volatile stocks)
    # We generate num_days - 1 returns (first day is the start price, no return)
    #
    # IMPORTANT: We exclude the degenerate case of ALL returns being exactly 0.0
    # because constant prices produce RSI = 0/0 = NaN (mathematically undefined
    # when there are no gains AND no losses). Real stock data always has some
    # variation. We use floats that avoid exact zero by filtering, but in practice
    # Hypothesis rarely generates all-zero lists from this range anyway.
    daily_returns = draw(
        lists(
            floats(
                min_value=-0.08,
                max_value=0.08,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=num_days - 1,
            max_size=num_days - 1,
        )
    )

    # Ensure at least some price variation exists — constant prices make RSI
    # undefined (0/0). Real stock data always has price movement. We guarantee
    # at least one non-zero return exists in the series.
    if all(r == 0.0 for r in daily_returns):
        # Force at least one small positive return to avoid the degenerate case
        daily_returns[0] = 0.01

    # Build the price series as a random walk:
    # price[0] = start_price
    # price[t] = price[t-1] * (1 + daily_return[t-1])
    # This produces realistic correlated price series (not i.i.d. random values)
    # which is important because rolling windows operate on sequential data
    prices = [start_price]
    for ret in daily_returns:
        next_price = prices[-1] * (1.0 + ret)
        # Floor at 0.01 to prevent zero/negative prices which would cause
        # division-by-zero in percentage change calculations (producing NaN)
        prices.append(max(next_price, 0.01))

    # Volume: positive integers, no NaN allowed
    # Range [100, 10_000_000] covers low-liquidity to high-liquidity stocks
    volumes = draw(
        lists(
            integers(min_value=100, max_value=10_000_000),
            min_size=num_days,
            max_size=num_days,
        )
    )

    # Construct the DataFrame with Close and Volume columns
    # This is the minimal schema required by compute_features()
    df = pd.DataFrame(
        {
            "Close": prices,
            "Volume": volumes,
        }
    )

    return df


@given(df=valid_price_dataframes())
def test_feature_pipeline_produces_no_nan_values(df):
    """Property 4: Feature pipeline produces no NaN values.

    **Validates: Requirements 2.6, 3.2**

    For any raw stock DataFrame with at least 80 trading days and no NaN in
    Close/Volume columns, the feature engineering pipeline SHALL produce an
    output DataFrame containing zero NaN values in any column.

    This property is critical because:
        - NaN values would corrupt scikit-learn model training (most estimators
          raise ValueError on NaN inputs)
        - The feature pipeline uses multiple rolling windows (5, 14, 20, 26, 50
          days) that produce NaN values for their initial warm-up rows
        - compute_features() must correctly drop ALL rows with any NaN before
          returning the result
        - This must hold regardless of the price distribution, magnitude, or
          number of days in the input

    The test generates random price DataFrames via Hypothesis with:
        - 80 to 300 trading days (minimum requirement and beyond)
        - Random walk prices (realistic correlated series, always positive)
        - Positive integer volumes (valid market data)

    It then verifies that:
        1. The output DataFrame is not empty (features were successfully computed)
        2. The output contains zero NaN values in any cell across all columns
        3. All expected feature columns are present in the output
    """
    # Run the feature pipeline on the generated price data
    result = compute_features(df)

    # The output should not be empty — with 80+ days of data, there should
    # be rows remaining after dropping the rolling window warm-up period
    # (the largest window is 50-day MA, so at least 80 - 50 = 30+ rows should remain)
    assert len(result) > 0, (
        f"compute_features() returned an empty DataFrame for input with "
        f"{len(df)} trading days. Expected at least some rows after dropping "
        f"NaN warm-up rows."
    )

    # Core property assertion: no NaN values anywhere in the output
    # This checks every cell in every column of the result DataFrame
    nan_count = result.isna().sum().sum()
    assert nan_count == 0, (
        f"compute_features() produced {nan_count} NaN value(s) in output.\n"
        f"Input had {len(df)} trading days.\n"
        f"Output has {len(result)} rows and columns: {list(result.columns)}.\n"
        f"NaN counts per column:\n{result.isna().sum()}"
    )

    # Verify all expected feature columns are present
    expected_columns = [
        "close_price",
        "daily_return",
        "ma_5",
        "ma_20",
        "ma_50",
        "volatility",
        "volume",
        "RSI",
        "MACD",
    ]
    for col in expected_columns:
        assert col in result.columns, (
            f"Expected column '{col}' missing from compute_features() output. "
            f"Got columns: {list(result.columns)}"
        )


# =============================================================================
# Feature: ai-stock-recommendation, Property 5: Feature computations match mathematical definitions
# =============================================================================


# --- Tolerance for floating-point comparisons ---
# We use a relative tolerance of 1e-7 and absolute tolerance of 1e-10.
# Rationale: pandas rolling operations and numpy mean calculations may differ
# at the level of floating-point rounding (last few bits of mantissa). A
# relative tolerance of 1e-7 is well above machine epsilon (2.2e-16 for float64)
# but tight enough to catch any algorithmic error (e.g. off-by-one in window).
RTOL = 1e-7
ATOL = 1e-10


@composite
def stock_price_series(draw):
    """Generate a realistic stock price DataFrame with at least 80 trading days.

    Strategy:
        - Draw a length between 80 and 200 (enough for all rolling windows to
          produce non-NaN output, but not so large that tests are slow).
        - Generate close prices as positive floats in [1.0, 10000.0]. This range
          covers penny stocks through high-priced equities. We avoid values near
          zero to prevent division issues in return calculations.
        - Generate volume as positive integers (standard for share counts).

    The resulting DataFrame mimics the structure expected by compute_features():
    a DatetimeIndex with 'Close' and 'Volume' columns.
    """
    # Number of trading days: at least 80 (minimum for compute_features),
    # up to 200 to keep test runtime reasonable
    n_days = draw(integers(min_value=80, max_value=200))

    # Generate close prices as positive floats.
    # Range [1.0, 10000.0] avoids near-zero prices that could cause
    # numerical instability in percentage return calculations.
    close_prices = draw(
        lists(
            floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
            min_size=n_days,
            max_size=n_days,
        )
    )

    # Generate volume as positive integers (shares traded per day)
    volumes = draw(
        lists(
            integers(min_value=100, max_value=1_000_000_000),
            min_size=n_days,
            max_size=n_days,
        )
    )

    # Build a DataFrame matching the expected input format for compute_features()
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")
    df = pd.DataFrame(
        {"Close": close_prices, "Volume": volumes},
        index=dates,
    )

    return df


@given(df=stock_price_series())
def test_feature_computations_match_mathematical_definitions(df):
    """Property 5: Feature computations match mathematical definitions.

    **Validates: Requirements 3.1**

    For any stock price series of at least 80 trading days, the computed features
    SHALL satisfy:
        - ma_5[t] equals the mean of close[t-4:t+1] (5-day simple moving average)
        - ma_20[t] equals the mean of close[t-19:t+1] (20-day simple moving average)
        - volatility[t] equals the 20-day rolling standard deviation of daily_return
        - RSI[t] is in [0, 100] for all rows
        - MACD[t] equals EMA(12, close)[t] minus EMA(26, close)[t]

    Verification approach (oracle testing):
        Each feature is independently recomputed using plain numpy/pandas operations
        that do NOT call the same rolling/ewm methods in the same way as the
        implementation. This ensures we are testing the mathematical correctness
        rather than just checking the code against itself.
    """
    # --- Run the implementation under test ---
    result = compute_features(df)

    # The result should not be empty (80+ days gives enough data after NaN drop)
    assume(len(result) > 0)

    # Get the raw close prices for independent computation
    close = df["Close"].values.astype(float)

    # =========================================================================
    # Verify ma_5: 5-day Simple Moving Average
    # Mathematical definition: ma_5[t] = mean(close[t-4], close[t-3], ..., close[t])
    # We compute this independently using a simple loop over numpy arrays.
    # =========================================================================
    for i in range(len(result)):
        # Find the original index position of this result row in the input DataFrame
        result_date = result.index[i]
        orig_idx = df.index.get_loc(result_date)

        # Independent oracle computation: mean of the 5 most recent closes
        # including the current day (window = [t-4, t-3, t-2, t-1, t])
        expected_ma5 = np.mean(close[orig_idx - 4 : orig_idx + 1])
        actual_ma5 = result["ma_5"].iloc[i]

        # Compare with tolerance to account for floating-point rounding
        assert np.isclose(actual_ma5, expected_ma5, rtol=RTOL, atol=ATOL), (
            f"ma_5 mismatch at index {i} (date {result_date}): "
            f"expected {expected_ma5}, got {actual_ma5}"
        )

    # =========================================================================
    # Verify ma_20: 20-day Simple Moving Average
    # Mathematical definition: ma_20[t] = mean(close[t-19], ..., close[t])
    # Independent computation using numpy mean over the 20-element window.
    # =========================================================================
    for i in range(len(result)):
        result_date = result.index[i]
        orig_idx = df.index.get_loc(result_date)

        # Independent oracle: mean of the 20 most recent closes including current day
        expected_ma20 = np.mean(close[orig_idx - 19 : orig_idx + 1])
        actual_ma20 = result["ma_20"].iloc[i]

        assert np.isclose(actual_ma20, expected_ma20, rtol=RTOL, atol=ATOL), (
            f"ma_20 mismatch at index {i} (date {result_date}): "
            f"expected {expected_ma20}, got {actual_ma20}"
        )

    # =========================================================================
    # Verify volatility: 20-day rolling standard deviation of daily_return
    # Mathematical definition: volatility[t] = std(daily_return[t-19:t+1])
    # where daily_return[t] = (close[t] - close[t-1]) / close[t-1]
    # We compute daily returns independently, then take the std of the last 20.
    # Note: pandas rolling.std() uses ddof=1 (sample std) by default.
    # =========================================================================

    # Independent computation of daily returns
    daily_returns = np.diff(close) / close[:-1]  # length = n_days - 1

    for i in range(len(result)):
        result_date = result.index[i]
        orig_idx = df.index.get_loc(result_date)

        # daily_returns array is shifted by 1 relative to close (no return for day 0)
        # daily_returns[k] corresponds to the return from close[k] to close[k+1]
        # So the daily return AT orig_idx is daily_returns[orig_idx - 1]
        # The 20-day window of returns ending at orig_idx is:
        # daily_returns[orig_idx - 20 : orig_idx]
        returns_window = daily_returns[orig_idx - 20 : orig_idx]

        # Use ddof=1 (sample standard deviation) to match pandas rolling.std() default
        expected_volatility = np.std(returns_window, ddof=1)
        actual_volatility = result["volatility"].iloc[i]

        assert np.isclose(actual_volatility, expected_volatility, rtol=RTOL, atol=ATOL), (
            f"volatility mismatch at index {i} (date {result_date}): "
            f"expected {expected_volatility}, got {actual_volatility}"
        )

    # =========================================================================
    # Verify RSI: Relative Strength Index is in [0, 100]
    # The RSI is bounded by definition: RSI = 100 - 100/(1+RS) where RS >= 0.
    # When RS = 0 (all losses), RSI = 0. When RS = inf (all gains), RSI = 100.
    # We verify the range property without recomputing the exact value (Wilder's
    # smoothing is complex to replicate exactly, so we verify bounds).
    # =========================================================================
    rsi_values = result["RSI"].values

    # All RSI values must be in the closed interval [0, 100]
    assert np.all(rsi_values >= 0.0), (
        f"RSI below 0 found: min RSI = {rsi_values.min()}"
    )
    assert np.all(rsi_values <= 100.0), (
        f"RSI above 100 found: max RSI = {rsi_values.max()}"
    )

    # =========================================================================
    # Verify MACD: Moving Average Convergence Divergence
    # Mathematical definition: MACD[t] = EMA(12, close)[t] - EMA(26, close)[t]
    # Independent computation using the EMA formula:
    #   EMA[t] = alpha * close[t] + (1 - alpha) * EMA[t-1]
    #   where alpha = 2 / (span + 1)
    # We compute EMA from scratch using a loop (no pandas ewm) to ensure
    # independence from the implementation.
    # =========================================================================

    def compute_ema_independently(prices, span):
        """Compute Exponential Moving Average from scratch using iterative formula.

        This is an independent oracle that does not use pandas ewm().
        Formula: EMA[t] = alpha * price[t] + (1 - alpha) * EMA[t-1]
        where alpha = 2 / (span + 1), initialized with EMA[0] = price[0].

        Args:
            prices: numpy array of price values
            span: EMA span parameter (e.g., 12 or 26)

        Returns:
            numpy array of EMA values, same length as prices
        """
        alpha = 2.0 / (span + 1.0)
        ema = np.zeros(len(prices))
        # Initialize EMA with the first price (same as pandas adjust=False)
        ema[0] = prices[0]
        for t in range(1, len(prices)):
            ema[t] = alpha * prices[t] + (1.0 - alpha) * ema[t - 1]
        return ema

    # Compute EMA(12) and EMA(26) independently over the full close price series
    ema_12_oracle = compute_ema_independently(close, span=12)
    ema_26_oracle = compute_ema_independently(close, span=26)

    # MACD = EMA(12) - EMA(26) at each point
    macd_oracle = ema_12_oracle - ema_26_oracle

    for i in range(len(result)):
        result_date = result.index[i]
        orig_idx = df.index.get_loc(result_date)

        expected_macd = macd_oracle[orig_idx]
        actual_macd = result["MACD"].iloc[i]

        # Use a slightly more relaxed tolerance for MACD because EMA accumulates
        # rounding differences over many iterations. Still tight enough to catch
        # algorithmic errors (e.g., wrong span parameter).
        assert np.isclose(actual_macd, expected_macd, rtol=1e-6, atol=1e-9), (
            f"MACD mismatch at index {i} (date {result_date}): "
            f"expected {expected_macd}, got {actual_macd}"
        )


# =============================================================================
# Feature: ai-stock-recommendation, Property 6: Target labels follow labeling rules
# =============================================================================


class TestTargetLabelRules:
    """Property tests verifying that target labels follow the documented labeling rules.

    The compute_target_labels() function assigns labels based on the 30-day forward
    return, defined as: future_30_day_return = (close[t+30] - close[t]) / close[t]

    Labeling thresholds:
        - "Buy":  future_30_day_return > 10%  (strictly greater than 0.10)
        - "Hold": 0% <= future_30_day_return <= 10%  (inclusive on both boundaries)
        - "Avoid": future_30_day_return < 0%  (strictly less than 0.0)

    Boundary conditions:
        - Exactly 0% return (future_close == current_close) → "Hold"
        - Exactly 10% return (future_close == 1.10 * current_close) → "Hold"
        - Just above 10% → "Buy"
        - Just below 0% → "Avoid"

    **Validates: Requirements 3.3**
    """

    @given(
        # Generate a positive current close price (must be > 0 to avoid division by zero)
        current_close=floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        # Generate a positive future close price (price 30 days ahead)
        future_close=floats(min_value=0.01, max_value=50000.0, allow_nan=False, allow_infinity=False),
    )
    def test_target_label_matches_threshold_rules(self, current_close: float, future_close: float):
        """For any (current_close, future_close) pair, the assigned label must match threshold rules.

        This test constructs a minimal DataFrame with exactly 31 rows:
        - Row 0 has close = current_close (the row we'll check the label for)
        - Row 30 has close = future_close (the price 30 trading days ahead)
        - Intermediate rows (1-29) are filled with current_close (their values don't
          affect the label for row 0; only row 0 and row 30 matter for the forward return)

        The function computes: future_30_day_return = (close[30] - close[0]) / close[0]
        Then assigns a label based on the return value.

        We independently compute the expected return using the same formula and verify
        the label matches the threshold rules. Because both the test and the implementation
        use the same floating-point arithmetic on the same values, boundary comparisons
        are consistent (no floating-point divergence at boundaries).
        """
        # Construct a DataFrame with 31 rows: row 0 = current_close, row 30 = future_close
        # Intermediate rows are filled with current_close since they don't affect
        # the label for row 0 (only close[t] and close[t+30] participate in the formula)
        prices = [current_close] * 30 + [future_close]
        df = pd.DataFrame({"Close": prices})

        # Compute target labels using the function under test
        result = compute_target_labels(df)

        # Only row 0 should have a valid label (rows 1-30 don't have 30 days of future data)
        assert len(result) == 1, f"Expected 1 row with valid label, got {len(result)}"

        # Independently compute the expected future_30_day_return
        # Formula: (close[t+30] - close[t]) / close[t]
        expected_return = (future_close - current_close) / current_close

        # Verify the computed return matches our independent calculation
        actual_return = result["future_30_day_return"].iloc[0]
        assert np.isclose(actual_return, expected_return, rtol=1e-9), (
            f"Return mismatch: expected {expected_return}, got {actual_return}"
        )

        # Determine the expected label based on threshold rules:
        # - "Buy" when future_30_day_return > 10% (strictly greater than 0.10)
        # - "Hold" when 0% <= future_30_day_return <= 10% (inclusive boundaries)
        # - "Avoid" when future_30_day_return < 0% (strictly less than 0.0)
        # Note: we use actual_return (the value the function computed) to determine
        # the expected label, ensuring floating-point consistency with the implementation.
        if actual_return > 0.10:
            expected_label = "Buy"
        elif actual_return >= 0.0:
            expected_label = "Hold"
        else:
            expected_label = "Avoid"

        # Verify the label matches our expected classification
        actual_label = result["target"].iloc[0]
        assert actual_label == expected_label, (
            f"Label mismatch for return={actual_return:.6f}: "
            f"expected '{expected_label}', got '{actual_label}'"
        )

    @given(
        # Generate current_close for boundary testing at exactly 0% return
        current_close=floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    def test_zero_return_is_hold(self, current_close: float):
        """When future_close equals current_close (0% return), label must be "Hold".

        Boundary condition: exactly 0% return is on the inclusive lower boundary
        of the Hold range (0% <= return <= 10%), so it should be classified as Hold.

        When future_close == current_close, the return is:
            (current_close - current_close) / current_close = 0.0 (exactly)
        This is unambiguous in floating-point since subtraction of identical values is 0.
        """
        # future_close == current_close means exactly 0% return
        future_close = current_close

        # Construct DataFrame: row 0 = current_close, row 30 = future_close (same value)
        # All prices are identical, so return is exactly 0.0
        prices = [current_close] * 31
        df = pd.DataFrame({"Close": prices})

        result = compute_target_labels(df)
        assert len(result) == 1

        # Exactly 0% return should yield "Hold" (inclusive lower boundary)
        assert result["target"].iloc[0] == "Hold", (
            f"Expected 'Hold' for 0% return, got '{result['target'].iloc[0]}'"
        )
        # The return should be exactly 0.0 (no floating-point error possible here)
        assert result["future_30_day_return"].iloc[0] == 0.0

    @given(
        # Use multiples of 10 as current_close so that dividing by 10 is exact in IEEE 754
        # This ensures current_close / 10 produces no rounding error, making the
        # 10% return computation exact: (current + current/10 - current) / current = 0.1
        current_close=integers(min_value=10, max_value=10000).map(lambda x: float(x * 10)),
    )
    def test_ten_percent_return_is_hold(self, current_close: float):
        """When future_30_day_return is exactly 10%, label must be "Hold".

        Boundary condition: exactly 10% return is on the inclusive upper boundary
        of the Hold range (0% <= return <= 10%), so it should be classified as Hold,
        NOT as Buy (which requires strictly > 10%).

        We use multiples of 10 as the current_close (e.g., 100.0, 200.0, 5000.0)
        so that current_close / 10.0 is exactly representable in IEEE 754 binary
        floating-point. This ensures the computation:
            future_close = current_close + current_close / 10.0
            return = (future_close - current_close) / current_close
                   = (current_close / 10.0) / current_close
                   = 0.1 (exactly)
        avoids floating-point imprecision at the boundary.
        """
        # future_close = current_close + current_close / 10.0 = current_close * 1.1
        # For multiples of 10, this division is exact in IEEE 754
        future_close = current_close + current_close / 10.0

        # Construct DataFrame: row 0 = current_close, row 30 = future_close
        prices = [current_close] * 30 + [future_close]
        df = pd.DataFrame({"Close": prices})

        result = compute_target_labels(df)
        assert len(result) == 1

        # Verify the computed return is exactly 0.10
        actual_return = result["future_30_day_return"].iloc[0]
        assert actual_return == 0.10, (
            f"Expected return of exactly 0.10, got {repr(actual_return)} "
            f"for current_close={current_close}"
        )

        # Exactly 10% return should yield "Hold" (inclusive upper boundary)
        assert result["target"].iloc[0] == "Hold", (
            f"Expected 'Hold' for exactly 10% return, got '{result['target'].iloc[0]}'"
        )
