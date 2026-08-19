"""Integration tests for the model export pipeline.

This module tests the training pipeline's model export flow by mocking yfinance
and simulating the key steps that the training notebook performs:

1. **Data acquisition** (mocked yfinance download → raw DataFrame)
2. **Feature engineering** (compute technical indicators from raw data)
3. **Model training** (train classifiers on computed features)
4. **Model export** (serialize best model to disk as stock_model.pkl)

WHY WE MOCK YFINANCE:
    The training notebook downloads real stock data from Yahoo Finance via the
    yfinance library. We mock this because:
    - Tests should not depend on network connectivity
    - External API responses can change or timeout, causing flaky tests
    - We need deterministic, reproducible data for consistent test results
    - Rate limiting or API outages should not block CI/CD pipelines

Since we cannot execute a Jupyter notebook programmatically in a unit test
environment, these tests replicate the notebook's core logic steps:
    1. Download data (mocked) → raw DataFrame
    2. Compute features using src/features.py
    3. Train a model using scikit-learn
    4. Export model to disk using pickle

This verifies the end-to-end export flow produces a valid model file that
can be loaded back for inference.

Requirements validated: 1.1, 6.2, 7.1
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.features import compute_features, compute_target_labels
from src.stock_recommender import StockRecommender, FEATURE_COLUMNS


def _generate_mock_stock_data(ticker: str, n_days: int = 200) -> pd.DataFrame:
    """Generate realistic mock stock price data for testing.

    Creates a DataFrame that mimics yfinance output with columns:
    Date, Open, High, Low, Close, Adj Close, Volume, and Ticker.

    The price data follows a random walk with realistic daily returns
    to produce meaningful technical indicators when processed.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        n_days: Number of trading days to generate (default 200)

    Returns:
        DataFrame with OHLCV data matching yfinance output schema.
    """
    rng = np.random.default_rng(hash(ticker) % (2**31))

    # Start with a base price and simulate random walk
    base_price = rng.uniform(50, 500)
    # Daily returns drawn from normal distribution (realistic ~1% daily std)
    daily_returns = rng.normal(0.0005, 0.015, n_days)
    # Cumulative returns create the price path
    price_multipliers = np.cumprod(1 + daily_returns)
    close_prices = base_price * price_multipliers

    # Generate OHLCV data with realistic relationships
    # High is always >= Close, Low is always <= Close
    highs = close_prices * (1 + rng.uniform(0.001, 0.02, n_days))
    lows = close_prices * (1 - rng.uniform(0.001, 0.02, n_days))
    opens = close_prices * (1 + rng.uniform(-0.01, 0.01, n_days))
    volumes = rng.integers(10_000_000, 100_000_000, n_days)

    # Create date range (business days only, mimicking trading days)
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)

    df = pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": close_prices,
        "Adj Close": close_prices,  # Simplified: Adj Close = Close
        "Volume": volumes,
        "Ticker": ticker,
    })

    return df


@pytest.mark.integration
class TestMockYfinanceDataAcquisition:
    """Test data acquisition step with mocked yfinance.

    Verifies that when yfinance is mocked to return realistic data,
    the downstream pipeline (feature engineering → training → export)
    produces valid results.

    Mocking yfinance is necessary because:
    - Tests must be deterministic and not depend on network
    - Yahoo Finance API may rate-limit or return errors
    - We need consistent data for reproducible test results
    - yfinance may not be installed in CI/test environments

    We use sys.modules patching to create a fake yfinance module so that
    the test works regardless of whether yfinance is actually installed.
    """

    def _create_mock_yfinance(self):
        """Create a mock yfinance module and inject it into sys.modules.

        This allows `import yfinance` to succeed even when yfinance is not
        installed, which is common in CI/test environments where only core
        dependencies are present.

        Returns:
            The mock yfinance module object with a download() method.
        """
        import sys
        mock_yf = MagicMock()
        sys.modules["yfinance"] = mock_yf
        return mock_yf

    def _cleanup_mock_yfinance(self):
        """Remove the mocked yfinance from sys.modules after test."""
        import sys
        sys.modules.pop("yfinance", None)

    def test_mock_yfinance_returns_valid_dataframe(self):
        """Verify that mocked yfinance data has the expected schema.

        This simulates what the training notebook does in the Data Acquisition
        section: call yfinance.download() for each ticker and validate the
        resulting DataFrame structure.
        """
        try:
            # Setup mock: inject fake yfinance module into sys.modules
            mock_yf = self._create_mock_yfinance()

            # Configure mock to return synthetic data matching yfinance output
            ticker = "AAPL"
            mock_data = _generate_mock_stock_data(ticker, n_days=200)
            # yfinance.download typically returns data without a Ticker column
            # and uses Date as the index
            yf_output = mock_data.drop(columns=["Ticker"]).set_index("Date")
            mock_yf.download.return_value = yf_output

            # Simulate the notebook's download call
            import yfinance as yf
            result = yf.download(ticker, start="2020-01-01", end="2024-12-31")

            # Verify the data has expected columns (matching Requirement 2.4)
            expected_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            assert expected_cols.issubset(set(result.columns)), (
                f"Missing columns: {expected_cols - set(result.columns)}"
            )

            # Verify data has sufficient rows (>= 80 for feature engineering)
            assert len(result) >= 80, (
                f"Insufficient data: {len(result)} rows, need at least 80"
            )

            # Verify no NaN in Close or Volume (Requirement 2.6)
            assert result["Close"].notna().all()
            assert result["Volume"].notna().all()
        finally:
            self._cleanup_mock_yfinance()

    def test_mock_yfinance_handles_multiple_tickers(self):
        """Verify data acquisition handles multiple tickers from Stock_Universe.

        The training notebook downloads data for all 10 tickers in the
        Stock_Universe. This test mocks multiple downloads and verifies
        the aggregation step.
        """
        try:
            # Setup mock: inject fake yfinance module
            mock_yf = self._create_mock_yfinance()

            # Stock Universe as defined in Requirements
            stock_universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

            # Mock returns different data for each ticker call
            def side_effect(ticker, *args, **kwargs):
                data = _generate_mock_stock_data(ticker, n_days=150)
                return data.drop(columns=["Ticker"]).set_index("Date")

            mock_yf.download.side_effect = side_effect

            # Simulate the notebook's multi-ticker download loop
            all_data = []
            for ticker in stock_universe:
                import yfinance as yf
                df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
                df["Ticker"] = ticker
                all_data.append(df)

            # Combine all ticker data (as the notebook does)
            combined = pd.concat(all_data)

            # Verify all tickers are represented
            assert set(combined["Ticker"].unique()) == set(stock_universe)

            # Verify each ticker has sufficient data
            for ticker in stock_universe:
                ticker_data = combined[combined["Ticker"] == ticker]
                assert len(ticker_data) >= 80, (
                    f"{ticker} has only {len(ticker_data)} rows"
                )
        finally:
            self._cleanup_mock_yfinance()


@pytest.mark.integration
class TestFeatureEngineeringFromMockData:
    """Test feature engineering step using mock stock data.

    Verifies that compute_features() produces valid, NaN-free output
    when given data that simulates what yfinance would return.
    This is the second stage of the training pipeline.
    """

    def test_features_computed_from_mock_data(self):
        """Verify feature engineering produces valid features from mock data.

        Pipeline step: raw mock data → compute_features() → feature DataFrame.
        The output must have all 9 expected feature columns with no NaN values.
        """
        # Generate mock data simulating yfinance output (200 days)
        mock_data = _generate_mock_stock_data("AAPL", n_days=200)

        # compute_features expects a DataFrame with 'Adj Close' and 'Volume'
        features_df = compute_features(mock_data)

        # Verify all 9 feature columns are present
        expected_features = [
            "close_price", "daily_return", "ma_5", "ma_20", "ma_50",
            "volatility", "volume", "RSI", "MACD",
        ]
        for col in expected_features:
            assert col in features_df.columns, f"Missing feature column: {col}"

        # Verify no NaN values remain (Property 4 from design)
        assert features_df.isna().sum().sum() == 0, (
            "Feature DataFrame contains NaN values after compute_features()"
        )

        # Verify RSI is within valid range [0, 100]
        assert (features_df["RSI"] >= 0).all(), "RSI below 0 detected"
        assert (features_df["RSI"] <= 100).all(), "RSI above 100 detected"

    def test_target_labels_computed_from_mock_data(self):
        """Verify target label computation produces valid labels from mock data.

        Pipeline step: mock data → compute_target_labels() → labeled DataFrame.
        Labels must be one of {Buy, Hold, Avoid} based on 30-day forward returns.
        """
        # Generate enough data to compute 30-day forward returns
        mock_data = _generate_mock_stock_data("MSFT", n_days=200)

        labels_df = compute_target_labels(mock_data)

        # Verify target column exists
        assert "target" in labels_df.columns
        assert "future_30_day_return" in labels_df.columns

        # All labels must be one of the valid values
        valid_labels = {"Buy", "Hold", "Avoid"}
        actual_labels = set(labels_df["target"].unique())
        assert actual_labels.issubset(valid_labels), (
            f"Invalid labels found: {actual_labels - valid_labels}"
        )

        # Verify no NaN in target or future_30_day_return
        assert labels_df["target"].notna().all()
        assert labels_df["future_30_day_return"].notna().all()


@pytest.mark.integration
class TestModelExportPipeline:
    """Test the model training and export pipeline end-to-end.

    Simulates the training notebook's Model Training → Model Export flow:
    1. Generate mock data and compute features
    2. Train models (Logistic Regression, Decision Tree, Random Forest)
    3. Select best model by weighted F1-score
    4. Export to a temporary directory
    5. Verify the exported model can be loaded and produces valid predictions

    This verifies Requirement 6.2: the exported model produces valid predictions
    when loaded back from disk.
    """

    def _build_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Build a training dataset from mock stock data.

        Simulates the notebook's data pipeline:
            Mock data → compute_features() → compute_target_labels() → merge

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Generate mock data for multiple tickers (simulates Stock_Universe)
        tickers = ["AAPL", "MSFT", "NVDA"]
        all_features = []

        for ticker in tickers:
            mock_data = _generate_mock_stock_data(ticker, n_days=200)

            # Compute features (same as notebook Feature Engineering section)
            features = compute_features(mock_data)

            # Compute target labels
            labels = compute_target_labels(mock_data)

            # Merge features and labels on index (both have same index from raw data)
            # Only keep rows that exist in both (inner join)
            combined = features.join(labels[["target"]], how="inner")
            combined = combined.dropna()

            all_features.append(combined)

        # Combine all tickers into one training dataset
        full_dataset = pd.concat(all_features, ignore_index=True)

        # Separate features (X) and target (y)
        X = full_dataset[FEATURE_COLUMNS]
        y = full_dataset["target"]

        return X, y

    def test_full_export_pipeline_produces_model_file(self):
        """Verify the complete training → export pipeline produces a loadable model.

        This is the key integration test: it runs the full pipeline from
        mock data through training to model export, then verifies the exported
        model can be loaded by StockRecommender and produce valid predictions.

        Steps replicated from the training notebook:
        1. Build training data from mock stock data
        2. Split 80/20 stratified with seed=42
        3. Scale features with StandardScaler
        4. Train RandomForest, DecisionTree, LogisticRegression
        5. Select best by weighted F1-score
        6. Export model and scaler to disk
        7. Load back and verify predictions are valid
        """
        # Step 1: Build training data from mock stock data
        X, y = self._build_training_data()
        assert len(X) > 50, f"Insufficient training data: {len(X)} rows"

        # Step 2: 80/20 stratified split with seed=42 (Requirement 4.3, 4.4)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Step 3: Fit StandardScaler on training data (Requirement 4.5)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=FEATURE_COLUMNS,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=FEATURE_COLUMNS,
        )

        # Step 4: Train all three models (Requirement 4.1)
        models = {
            "LogisticRegression": LogisticRegression(
                random_state=42, max_iter=1000
            ),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(
                n_estimators=10, random_state=42
            ),
        }

        # Train each model and compute weighted F1-score
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, average="weighted")
            results[name] = {"model": model, "f1": f1}

        # Step 5: Select best model by highest F1 (Requirement 5.3)
        best_name = max(results, key=lambda k: results[k]["f1"])
        best_model = results[best_name]["model"]

        # Step 6: Export model and scaler to a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "stock_model.pkl")
            scaler_path = os.path.join(tmp_dir, "scaler.pkl")

            # Serialize model (Requirement 6.1)
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)

            # Serialize scaler (Requirement 6.3)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            # Verify files were created
            assert os.path.exists(model_path), "Model file was not created"
            assert os.path.exists(scaler_path), "Scaler file was not created"
            assert os.path.getsize(model_path) > 0, "Model file is empty"
            assert os.path.getsize(scaler_path) > 0, "Scaler file is empty"

            # Step 7: Load back via StockRecommender and verify predictions
            recommender = StockRecommender(
                model_path=model_path,
                scaler_path=scaler_path,
            )

            # Predict with sample features — must return valid Stock_Signal
            sample_features = {
                "close_price": 200.0,
                "daily_return": 0.01,
                "ma_5": 198.0,
                "ma_20": 195.0,
                "ma_50": 190.0,
                "volatility": 0.025,
                "volume": 50_000_000,
                "RSI": 55.0,
                "MACD": 2.0,
            }

            signal = recommender.predict(sample_features)
            assert signal in {"Buy", "Hold", "Avoid"}, (
                f"Exported model produced invalid signal: '{signal}'"
            )

    def test_exported_model_preserves_predictions(self):
        """Verify serialization round-trip preserves model predictions.

        Requirement 6.2: when the model is loaded back from disk, it SHALL
        produce identical predictions to the model before serialization.
        """
        # Build and train a simple model
        X, y = self._build_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit scaler and train model
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=FEATURE_COLUMNS,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=FEATURE_COLUMNS,
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Get predictions BEFORE serialization
        predictions_before = model.predict(X_test_scaled)

        # Serialize and deserialize
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "stock_model.pkl")
            scaler_path = os.path.join(tmp_dir, "scaler.pkl")

            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            # Load model back from disk
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                loaded_scaler = pickle.load(f)

        # Get predictions AFTER deserialization
        X_test_rescaled = pd.DataFrame(
            loaded_scaler.transform(X_test),
            columns=FEATURE_COLUMNS,
        )
        predictions_after = loaded_model.predict(X_test_rescaled)

        # Predictions must be identical (Requirement 6.2)
        assert np.array_equal(predictions_before, predictions_after), (
            "Model predictions differ after serialization round-trip"
        )

    def test_export_without_scaler(self):
        """Verify model export works when no scaler is needed.

        Tree-based models (DecisionTree, RandomForest) do not require
        feature scaling. The pipeline should still export correctly and
        StockRecommender should handle the missing scaler gracefully.
        """
        # Build training data
        X, y = self._build_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train a tree-based model (no scaling needed)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Export model only (no scaler)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "stock_model.pkl")
            scaler_path = os.path.join(tmp_dir, "scaler.pkl")
            # Intentionally do NOT create scaler.pkl

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # StockRecommender should load successfully without scaler
            recommender = StockRecommender(
                model_path=model_path,
                scaler_path=scaler_path,  # File doesn't exist — should be handled
            )

            # Scaler should be None when file doesn't exist
            assert recommender.scaler is None
            assert recommender.model is not None

            # Predictions should still work without scaler
            features = {col: 100.0 for col in FEATURE_COLUMNS}
            signal = recommender.predict(features)
            assert signal in {"Buy", "Hold", "Avoid"}

    def test_model_file_not_found_raises_error(self):
        """Verify descriptive error when model file doesn't exist.

        Requirement 6.4: if the model file cannot be loaded, a descriptive
        error message should be raised indicating which file was not found.
        """
        # Attempt to load from a non-existent path
        with pytest.raises(FileNotFoundError) as exc_info:
            StockRecommender(
                model_path="nonexistent/path/stock_model.pkl",
                scaler_path="nonexistent/path/scaler.pkl",
            )

        # Error message should identify the missing file path
        assert "nonexistent/path/stock_model.pkl" in str(exc_info.value)
