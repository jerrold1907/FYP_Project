"""Integration test fixtures for end-to-end pipeline verification.

This conftest creates synthetic model artifacts (stock_model.pkl, scaler.pkl)
in the models/ directory so that integration tests can exercise the full pipeline
without requiring the training notebook to have been run first.

The synthetic model is a small RandomForestClassifier trained on randomly generated
feature data with 3 classes (Buy, Hold, Avoid). It won't produce meaningful
predictions, but it validates that all components wire together correctly:
    InvestorProfile → RiskClassifier → Risk_Category
    Features → StockRecommender → Stock_Signal
    (Risk_Category, Stock_Signal) → SuitabilityEngine → Suitability_Rating
"""

import os
import pickle

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Path to the models directory (relative to project root)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


@pytest.fixture(autouse=True)
def synthetic_model_artifacts(tmp_path, monkeypatch):
    """Create synthetic model and scaler artifacts for integration tests.

    This fixture:
    1. Trains a small RandomForestClassifier on synthetic data (9 features, 3 classes)
    2. Fits a StandardScaler on the same synthetic data
    3. Saves both to the models/ directory as stock_model.pkl and scaler.pkl
    4. Cleans up after the test completes

    The model is trained to output string labels ("Buy", "Hold", "Avoid") directly,
    matching the expected behavior of the real trained model.
    """
    model_path = os.path.join(MODELS_DIR, "stock_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    # Track whether files existed before the test (don't overwrite real models)
    model_existed = os.path.exists(model_path)
    scaler_existed = os.path.exists(scaler_path)

    if not model_existed:
        # Generate synthetic training data matching the 9 feature columns:
        # close_price, daily_return, ma_5, ma_20, ma_50, volatility, volume, RSI, MACD
        rng = np.random.default_rng(42)
        n_samples = 300

        # Feature column names matching what StockRecommender expects
        feature_columns = [
            "close_price", "daily_return", "ma_5", "ma_20", "ma_50",
            "volatility", "volume", "RSI", "MACD",
        ]

        # Create synthetic features with realistic value ranges as a DataFrame
        # so the model is fitted with feature names (avoids sklearn warnings)
        import pandas as pd
        X_train = pd.DataFrame({
            "close_price": rng.uniform(50, 500, n_samples),
            "daily_return": rng.uniform(-0.05, 0.05, n_samples),
            "ma_5": rng.uniform(50, 500, n_samples),
            "ma_20": rng.uniform(50, 500, n_samples),
            "ma_50": rng.uniform(50, 500, n_samples),
            "volatility": rng.uniform(0.01, 0.05, n_samples),
            "volume": rng.uniform(1e6, 1e8, n_samples),
            "RSI": rng.uniform(20, 80, n_samples),
            "MACD": rng.uniform(-5, 5, n_samples),
        })

        # Create balanced target labels (Buy, Hold, Avoid)
        y_train = np.array(["Buy", "Hold", "Avoid"] * (n_samples // 3))

        # Fit scaler on training data (DataFrame preserves feature names)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_columns,
        )

        # Train a small RandomForest (few trees for speed)
        # Training on a DataFrame with column names ensures the model
        # expects named features during prediction (avoids sklearn warnings)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_scaled, y_train)

        # Save model artifact
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save scaler artifact
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    yield

    # Cleanup: remove synthetic artifacts only if we created them
    if not model_existed and os.path.exists(model_path):
        os.remove(model_path)
    if not scaler_existed and os.path.exists(scaler_path):
        os.remove(scaler_path)
