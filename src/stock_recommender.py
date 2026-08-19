"""Stock Recommender Module.

This module provides the StockRecommender class which serves as the inference
component of the AI-Powered Stock Recommendation System. Its role is to:

1. Load a pre-trained scikit-learn classification model from disk (serialized
   via pickle or joblib during the training notebook's Model Export step).
2. Optionally load a fitted StandardScaler if one was saved during training
   (required when the best model was Logistic Regression).
3. Accept a dictionary of technical indicator features and produce a stock
   action prediction: "Buy", "Hold", or "Avoid".

The module bridges the gap between the training pipeline (which produces
model artifacts) and the Suitability Engine (which consumes stock signals).
"""

import os
import pickle

import pandas as pd


# Feature columns expected by the trained model, in the exact order used during training.
FEATURE_COLUMNS = [
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

# Mapping from numeric prediction labels to human-readable stock signals.
# The model outputs integer class indices which map to these labels.
LABEL_MAP = {0: "Avoid", 1: "Buy", 2: "Hold"}


class StockRecommender:
    """Loads a trained stock prediction model and produces Buy/Hold/Avoid signals.

    The StockRecommender wraps a serialized scikit-learn classifier and an
    optional StandardScaler. It provides a simple predict() interface that
    accepts raw technical indicator values as a dictionary and returns one
    of three stock action signals.

    Attributes:
        model: The deserialized scikit-learn classifier.
        scaler: The deserialized StandardScaler, or None if no scaler file exists.
    """

    def __init__(
        self,
        model_path: str = "models/stock_model.pkl",
        scaler_path: str = "models/scaler.pkl",
    ):
        """Load the trained model and optionally the feature scaler from disk.

        The constructor attempts to load the classification model from
        `model_path` using pickle deserialization. If the file does not exist,
        a FileNotFoundError is raised with a descriptive message indicating
        the missing path.

        The scaler is optional: if `scaler_path` exists on disk, it is loaded
        and will be applied to features before prediction. If the scaler file
        does not exist, prediction proceeds without scaling. This accommodates
        models like Decision Tree or Random Forest that do not require feature
        scaling, while supporting Logistic Regression which does.

        Args:
            model_path: Path to the serialized model file (pickle format).
                        Defaults to "models/stock_model.pkl".
            scaler_path: Path to the serialized scaler file (pickle format).
                         Defaults to "models/scaler.pkl". Loaded only if the
                         file exists on disk.

        Raises:
            FileNotFoundError: If `model_path` does not exist. The error
                message includes the path that was not found.
        """
        # Load the trained classification model from pickle.
        # The model file is required — raise an error if it's missing.
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: '{model_path}'. "
                "Ensure the training notebook has been run and the model exported."
            )

        with open(model_path, "rb") as f:
            # Deserialize the scikit-learn model using pickle
            self.model = pickle.load(f)

        # Optionally load the scaler if the file exists.
        # The scaler is only present when the best model required feature scaling
        # (e.g., Logistic Regression). Tree-based models don't need it.
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                # Deserialize the fitted StandardScaler
                self.scaler = pickle.load(f)
        else:
            # No scaler file found — prediction will use raw feature values.
            # This is expected for tree-based models (Decision Tree, Random Forest).
            self.scaler = None

    def predict(self, features: dict) -> str:
        """Predict a stock action signal from technical indicator features.

        This method takes a dictionary of feature values, constructs a
        single-row DataFrame in the correct column order expected by the
        model, optionally applies the scaler transform, runs the model's
        predict method, and maps the numeric output to a human-readable
        label ("Buy", "Hold", or "Avoid").

        The feature dictionary must contain all keys defined in FEATURE_COLUMNS:
        close_price, daily_return, ma_5, ma_20, ma_50, volatility, volume,
        RSI, and MACD.

        Args:
            features: Dictionary mapping feature names to their numeric values.
                      All keys in FEATURE_COLUMNS must be present.

        Returns:
            One of "Buy", "Hold", or "Avoid" representing the model's
            stock action prediction.

        Raises:
            KeyError: If any required feature is missing from the input dict.
        """
        # Construct a single-row DataFrame with columns in the exact order
        # the model was trained on. This ensures feature alignment.
        feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        # Apply the scaler transform if one was loaded.
        # StandardScaler normalizes features to zero mean and unit variance,
        # which is required for Logistic Regression but not for tree-based models.
        if self.scaler is not None:
            scaled_values = self.scaler.transform(feature_df)
        else:
            scaled_values = feature_df.values

        # Run model prediction — returns an array of class labels.
        # For a single input row, this gives a 1-element array.
        # Pass numpy array (not DataFrame) to avoid feature name warnings.
        prediction = self.model.predict(scaled_values)

        # Map the numeric prediction to a human-readable label.
        # If the model outputs string labels directly (e.g., "Buy", "Hold", "Avoid"),
        # return them as-is. Otherwise, use LABEL_MAP for integer class indices.
        result = prediction[0]
        if isinstance(result, str):
            return result

        # Integer label mapping: 0 → "Avoid", 1 → "Buy", 2 → "Hold"
        return LABEL_MAP.get(result, str(result))
