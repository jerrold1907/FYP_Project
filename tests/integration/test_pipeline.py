"""Integration tests for the full recommendation pipeline.

This module tests the end-to-end flows that connect all system components:

1. **Model Load → Predict → Valid Signal**: Verifies that StockRecommender can
   load a serialized model from disk and produce a valid Stock_Signal (Buy, Hold,
   or Avoid) for any set of technical indicator features.

2. **Full Pipeline (Investor Profile + Stock Features → Suitability Rating)**:
   Verifies the complete data flow:
       InvestorProfile → RiskClassifier → Risk_Category
       Stock Features → StockRecommender → Stock_Signal
       (Risk_Category, Stock_Signal) → SuitabilityEngine → Suitability_Rating + Explanation

These tests use the synthetic model artifacts created by the integration conftest
fixture, which trains a small RandomForestClassifier on random data. The model
produces arbitrary but structurally valid predictions, ensuring that component
wiring is correct without depending on a real trained model.

Requirements validated: 1.1, 6.2, 7.1
"""

import pytest

from src.risk_classifier import InvestorProfile, RiskClassifier
from src.stock_recommender import StockRecommender, FEATURE_COLUMNS
from src.suitability import SuitabilityEngine


# Valid Stock_Signal values that the recommender must output
VALID_SIGNALS = {"Buy", "Hold", "Avoid"}

# Valid Suitability_Rating values that the engine must output
VALID_RATINGS = {"Suitable", "Use Caution", "Not Suitable"}

# Valid Risk_Category values that the classifier must output
VALID_CATEGORIES = {"Conservative", "Moderate", "Aggressive"}


@pytest.mark.integration
class TestModelLoadAndPredict:
    """Test that StockRecommender loads a model and returns valid Stock_Signal.

    These tests verify Requirement 6.2: when stock_model.pkl is loaded back into
    memory, StockRecommender SHALL produce valid class label predictions.

    The synthetic model artifacts (stock_model.pkl, scaler.pkl) are created by
    the integration conftest autouse fixture before each test runs.
    """

    def test_model_loads_from_disk(self):
        """Verify StockRecommender successfully loads model and scaler from disk.

        This confirms that the serialization format produced during training
        (pickle) is correctly consumed by StockRecommender's __init__ method.
        """
        # Load model from the synthetic artifacts created by conftest
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Both model and scaler should be loaded (not None)
        assert recommender.model is not None, "Model failed to load from disk"
        assert recommender.scaler is not None, "Scaler failed to load from disk"

    def test_predict_returns_valid_stock_signal(self):
        """Verify predict() output is always one of Buy, Hold, or Avoid.

        This is the core contract of StockRecommender: given any valid feature
        dictionary, the output MUST be a string in {Buy, Hold, Avoid}.
        Requirement 6.2 mandates that loaded models produce valid predictions.
        """
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Neutral/typical feature values representing a mid-range stock
        features = {
            "close_price": 175.50,
            "daily_return": 0.005,
            "ma_5": 174.0,
            "ma_20": 170.0,
            "ma_50": 165.0,
            "volatility": 0.022,
            "volume": 45_000_000,
            "RSI": 52.0,
            "MACD": 1.2,
        }

        signal = recommender.predict(features)

        # The prediction must be a valid Stock_Signal string
        assert signal in VALID_SIGNALS, (
            f"StockRecommender.predict() returned '{signal}', "
            f"expected one of {VALID_SIGNALS}"
        )

    def test_predict_with_bullish_features(self):
        """Verify predict() handles bullish indicator values correctly.

        Bullish indicators: price above moving averages, high RSI, positive MACD.
        Regardless of what signal the synthetic model produces, it must be valid.
        """
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Bullish feature profile: price above MAs, strong RSI, positive MACD
        bullish_features = {
            "close_price": 300.0,
            "daily_return": 0.04,
            "ma_5": 295.0,
            "ma_20": 280.0,
            "ma_50": 260.0,
            "volatility": 0.018,
            "volume": 70_000_000,
            "RSI": 72.0,
            "MACD": 5.5,
        }

        signal = recommender.predict(bullish_features)
        assert signal in VALID_SIGNALS

    def test_predict_with_bearish_features(self):
        """Verify predict() handles bearish indicator values correctly.

        Bearish indicators: price below moving averages, low RSI, negative MACD.
        The model must still produce a valid Stock_Signal.
        """
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Bearish feature profile: price below MAs, weak RSI, negative MACD
        bearish_features = {
            "close_price": 90.0,
            "daily_return": -0.035,
            "ma_5": 95.0,
            "ma_20": 105.0,
            "ma_50": 115.0,
            "volatility": 0.045,
            "volume": 100_000_000,
            "RSI": 22.0,
            "MACD": -4.0,
        }

        signal = recommender.predict(bearish_features)
        assert signal in VALID_SIGNALS

    def test_predict_with_extreme_values(self):
        """Verify predict() handles extreme but valid feature values.

        Edge case: very high volume, extreme RSI, large MACD divergence.
        The model should not crash and must return a valid signal.
        """
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Extreme but technically valid feature values
        extreme_features = {
            "close_price": 2000.0,
            "daily_return": 0.0,
            "ma_5": 2000.0,
            "ma_20": 2000.0,
            "ma_50": 2000.0,
            "volatility": 0.001,
            "volume": 500_000_000,
            "RSI": 50.0,
            "MACD": 0.0,
        }

        signal = recommender.predict(extreme_features)
        assert signal in VALID_SIGNALS

    def test_predict_uses_all_feature_columns(self):
        """Verify that the model expects exactly the 9 documented feature columns.

        This ensures alignment between the feature engineering module's output
        and the model's expected input schema.
        """
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # All 9 feature columns as documented in the design
        expected_columns = [
            "close_price", "daily_return", "ma_5", "ma_20", "ma_50",
            "volatility", "volume", "RSI", "MACD",
        ]
        assert FEATURE_COLUMNS == expected_columns

        # Build features dict from the expected columns
        features = {col: 100.0 for col in FEATURE_COLUMNS}
        signal = recommender.predict(features)
        assert signal in VALID_SIGNALS


@pytest.mark.integration
class TestFullPipelineEndToEnd:
    """Test the complete pipeline: investor profile + stock features → suitability.

    This class tests the full end-to-end data flow across all three system
    components working together:

    1. RiskClassifier receives an InvestorProfile and outputs a Risk_Category
    2. StockRecommender receives stock features and outputs a Stock_Signal
    3. SuitabilityEngine combines both outputs into a Suitability_Rating

    These tests verify that:
    - Component outputs are compatible with downstream component inputs
    - The data types flow correctly between stages
    - The final output contains both a rating and an explanation
    - Invalid inputs are caught at the appropriate stage

    Requirements validated: 1.1, 6.2, 7.1
    """

    def _run_full_pipeline(self, profile: InvestorProfile, features: dict) -> dict:
        """Helper: runs the full pipeline and returns the suitability result.

        This encapsulates the three-stage pipeline:
            Stage 1: Validate and classify investor profile
            Stage 2: Predict stock signal from features
            Stage 3: Combine into suitability recommendation

        Args:
            profile: InvestorProfile to classify
            features: Dictionary of 9 technical indicator features

        Returns:
            Dictionary with 'rating' and 'explanation' on success,
            or 'error' on validation failure.
        """
        # Stage 1: Risk Classification
        classifier = RiskClassifier()
        errors = classifier.validate(profile)
        if errors:
            return {"error": f"Validation failed: {errors}"}
        risk_category = classifier.classify(profile)

        # Stage 2: Stock Recommendation
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )
        stock_signal = recommender.predict(features)

        # Stage 3: Suitability Engine
        engine = SuitabilityEngine()
        return engine.recommend(risk_category, stock_signal)

    def test_young_aggressive_investor_pipeline(self):
        """Full pipeline: young aggressive investor with stock features → rating.

        Scenario: A 25-year-old with high risk score, long horizon, and
        significant experience. Expected classification: Aggressive.
        """
        # Young, experienced, long-horizon, high risk tolerance → Aggressive
        profile = InvestorProfile(
            age=25, income=130000.0, horizon=30, experience=6, risk_score=9
        )

        # Stock features representing a growth stock
        features = {
            "close_price": 450.0,
            "daily_return": 0.025,
            "ma_5": 445.0,
            "ma_20": 430.0,
            "ma_50": 400.0,
            "volatility": 0.03,
            "volume": 55_000_000,
            "RSI": 62.0,
            "MACD": 3.8,
        }

        result = self._run_full_pipeline(profile, features)

        # Pipeline should succeed without errors
        assert "error" not in result, f"Pipeline error: {result.get('error')}"

        # Verify output structure
        assert "rating" in result, "Missing 'rating' in pipeline output"
        assert "explanation" in result, "Missing 'explanation' in pipeline output"

        # Rating must be one of the three valid suitability ratings
        assert result["rating"] in VALID_RATINGS

        # Explanation must be a non-empty descriptive string
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 20

    def test_elderly_conservative_investor_pipeline(self):
        """Full pipeline: elderly conservative investor with stock features → rating.

        Scenario: A 72-year-old retiree with low risk score, short horizon,
        and minimal experience. Expected classification: Conservative.
        All adjustment factors apply (age > 60, horizon < 3, experience < 2).
        """
        # Elderly, low experience, short horizon, low risk → Conservative
        profile = InvestorProfile(
            age=72, income=35000.0, horizon=2, experience=1, risk_score=2
        )

        # Stock features representing a defensive stock
        features = {
            "close_price": 55.0,
            "daily_return": 0.001,
            "ma_5": 54.5,
            "ma_20": 54.0,
            "ma_50": 53.0,
            "volatility": 0.01,
            "volume": 20_000_000,
            "RSI": 48.0,
            "MACD": 0.3,
        }

        result = self._run_full_pipeline(profile, features)

        # Pipeline should succeed
        assert "error" not in result

        # Verify the investor was classified as Conservative
        classifier = RiskClassifier()
        category = classifier.classify(profile)
        assert category == "Conservative"

        # Final rating must be valid
        assert result["rating"] in VALID_RATINGS
        assert len(result["explanation"]) > 20

    def test_middle_aged_moderate_investor_pipeline(self):
        """Full pipeline: middle-aged moderate investor with stock features → rating.

        Scenario: A 45-year-old with moderate risk score, medium horizon,
        and moderate experience. Expected classification: Moderate.
        """
        # Middle-aged, moderate experience, medium horizon → Moderate
        profile = InvestorProfile(
            age=45, income=85000.0, horizon=12, experience=8, risk_score=5
        )

        # Stock features representing a blue-chip stock
        features = {
            "close_price": 180.0,
            "daily_return": 0.008,
            "ma_5": 178.0,
            "ma_20": 175.0,
            "ma_50": 170.0,
            "volatility": 0.02,
            "volume": 40_000_000,
            "RSI": 55.0,
            "MACD": 1.5,
        }

        result = self._run_full_pipeline(profile, features)

        # Pipeline should succeed
        assert "error" not in result

        # Verify the investor was classified as Moderate
        classifier = RiskClassifier()
        category = classifier.classify(profile)
        assert category == "Moderate"

        # Final rating must be valid
        assert result["rating"] in VALID_RATINGS
        assert len(result["explanation"]) > 20

    def test_pipeline_output_rating_matches_suitability_table(self):
        """Verify pipeline output matches the suitability mapping for known inputs.

        Uses a fixed Risk_Category and Stock_Signal (bypassing model randomness)
        to confirm the SuitabilityEngine lookup is correct within the pipeline.
        """
        engine = SuitabilityEngine()

        # Directly test the suitability mapping (independent of model output)
        # This verifies Requirement 7.1 in an integration context
        test_cases = [
            ("Conservative", "Buy", "Use Caution"),
            ("Moderate", "Buy", "Suitable"),
            ("Aggressive", "Avoid", "Use Caution"),
        ]

        for risk_cat, signal, expected_rating in test_cases:
            result = engine.recommend(risk_cat, signal)
            assert "error" not in result
            assert result["rating"] == expected_rating, (
                f"For ({risk_cat}, {signal}): "
                f"expected '{expected_rating}', got '{result['rating']}'"
            )

    def test_pipeline_catches_invalid_profile_early(self):
        """Verify that invalid investor profiles are caught at Stage 1.

        The pipeline should not proceed to stock prediction or suitability
        if the investor profile fails validation.
        """
        # Profile with multiple invalid fields
        invalid_profile = InvestorProfile(
            age=10,        # Below minimum (18)
            income=-5000,  # Below minimum (0)
            horizon=0,     # Below minimum (1)
            experience=0,  # Valid
            risk_score=5,  # Valid
        )

        features = {
            "close_price": 100.0,
            "daily_return": 0.01,
            "ma_5": 99.0,
            "ma_20": 98.0,
            "ma_50": 97.0,
            "volatility": 0.02,
            "volume": 30_000_000,
            "RSI": 50.0,
            "MACD": 1.0,
        }

        result = self._run_full_pipeline(invalid_profile, features)

        # Pipeline should report validation error rather than proceed
        assert "error" in result
        assert "Validation failed" in result["error"]

    def test_pipeline_produces_consistent_category_for_same_profile(self):
        """Verify the risk classification is deterministic for the same profile.

        Since RiskClassifier uses a rule-based algorithm (not ML), the same
        profile should always produce the same Risk_Category.
        """
        profile = InvestorProfile(
            age=35, income=90000.0, horizon=15, experience=5, risk_score=6
        )

        classifier = RiskClassifier()

        # Classify the same profile multiple times
        categories = [classifier.classify(profile) for _ in range(10)]

        # All classifications should be identical (deterministic)
        assert all(c == categories[0] for c in categories), (
            f"Non-deterministic classification: {set(categories)}"
        )
        assert categories[0] in VALID_CATEGORIES
