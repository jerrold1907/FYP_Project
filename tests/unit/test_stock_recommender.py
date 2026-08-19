"""Quick validation test for StockRecommender implementation."""
import os
import pickle
import tempfile

import numpy as np
import pytest


class MockModel:
    """A simple mock that mimics scikit-learn's predict interface."""

    def predict(self, X):
        return np.array(["Buy"])


class MockScaler:
    """A simple mock that mimics scikit-learn's StandardScaler interface."""

    def transform(self, X):
        return X


class IntLabelModel:
    """A mock model that returns integer class labels."""

    def predict(self, X):
        return np.array([1])  # 1 -> "Buy" in LABEL_MAP


def test_stock_recommender_loads_model_and_predicts():
    """Test basic load and predict flow."""
    from src.stock_recommender import StockRecommender

    # Create temp directory with mock model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(MockModel(), f)

        recommender = StockRecommender(
            model_path=model_path,
            scaler_path=os.path.join(tmpdir, "nonexistent_scaler.pkl"),
        )

        features = {
            "close_price": 150.0,
            "daily_return": 0.02,
            "ma_5": 148.0,
            "ma_20": 145.0,
            "ma_50": 140.0,
            "volatility": 0.03,
            "volume": 1000000,
            "RSI": 55.0,
            "MACD": 1.5,
        }

        result = recommender.predict(features)
        assert result in ("Buy", "Hold", "Avoid")
        assert result == "Buy"


def test_stock_recommender_loads_scaler():
    """Test that scaler is loaded and applied when file exists."""
    from src.stock_recommender import StockRecommender

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        scaler_path = os.path.join(tmpdir, "scaler.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(MockModel(), f)
        with open(scaler_path, "wb") as f:
            pickle.dump(MockScaler(), f)

        recommender = StockRecommender(
            model_path=model_path, scaler_path=scaler_path
        )

        assert recommender.scaler is not None

        features = {
            "close_price": 150.0,
            "daily_return": 0.02,
            "ma_5": 148.0,
            "ma_20": 145.0,
            "ma_50": 140.0,
            "volatility": 0.03,
            "volume": 1000000,
            "RSI": 55.0,
            "MACD": 1.5,
        }

        result = recommender.predict(features)
        assert result == "Buy"


def test_stock_recommender_missing_model_raises():
    """Test FileNotFoundError when model file doesn't exist."""
    from src.stock_recommender import StockRecommender

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        StockRecommender(
            model_path="/nonexistent/path/model.pkl",
            scaler_path="/nonexistent/path/scaler.pkl",
        )


def test_stock_recommender_no_scaler_file():
    """Test that missing scaler file results in scaler=None."""
    from src.stock_recommender import StockRecommender

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(MockModel(), f)

        recommender = StockRecommender(
            model_path=model_path,
            scaler_path=os.path.join(tmpdir, "no_scaler.pkl"),
        )

        assert recommender.scaler is None


def test_stock_recommender_integer_label_mapping():
    """Test that integer predictions are mapped via LABEL_MAP."""
    from src.stock_recommender import StockRecommender

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(IntLabelModel(), f)

        recommender = StockRecommender(
            model_path=model_path,
            scaler_path=os.path.join(tmpdir, "no_scaler.pkl"),
        )

        features = {
            "close_price": 150.0,
            "daily_return": 0.02,
            "ma_5": 148.0,
            "ma_20": 145.0,
            "ma_50": 140.0,
            "volatility": 0.03,
            "volume": 1000000,
            "RSI": 55.0,
            "MACD": 1.5,
        }

        result = recommender.predict(features)
        assert result == "Buy"
