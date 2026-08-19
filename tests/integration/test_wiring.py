"""End-to-End Wiring Verification Tests.

This module verifies that all src components wire together correctly in the
full pipeline flow:

    ┌─────────────────┐        ┌────────────────┐        ┌─────────────────┐
    │ InvestorProfile │───────►│ RiskClassifier │───────►│  Risk_Category  │
    └─────────────────┘        └────────────────┘        └────────┬────────┘
                                                                  │
    ┌─────────────────┐        ┌──────────────────┐              │
    │  Stock Features │───────►│ StockRecommender │──────┐       │
    └─────────────────┘        └──────────────────┘      │       │
                                                          ▼       ▼
                                                   ┌──────────────────────┐
                                                   │  SuitabilityEngine   │
                                                   └──────────┬───────────┘
                                                              │
                                                              ▼
                                                   ┌──────────────────────┐
                                                   │  Suitability_Rating  │
                                                   │  + Explanation       │
                                                   └──────────────────────┘

Components under test:
    - src/risk_classifier.py   — InvestorProfile, RiskClassifier
    - src/stock_recommender.py — StockRecommender (loads model from pickle)
    - src/suitability.py       — SuitabilityEngine
    - src/features.py          — compute_features, compute_target_labels
    - src/validation.py        — validate_investor_profile, validate_risk_category,
                                 validate_stock_signal

Requirements validated: 1.1, 6.2, 7.1
"""

import pytest

# Step 1: Verify all src modules import correctly
from src.risk_classifier import InvestorProfile, RiskClassifier
from src.stock_recommender import StockRecommender, FEATURE_COLUMNS
from src.suitability import SuitabilityEngine
from src.features import compute_features, compute_target_labels
from src.validation import (
    validate_investor_profile,
    validate_risk_category,
    validate_stock_signal,
)


@pytest.mark.integration
class TestModuleImports:
    """Verify that all src modules import correctly and expose expected interfaces."""

    def test_risk_classifier_exports(self):
        """InvestorProfile and RiskClassifier are importable and instantiable."""
        # InvestorProfile is a dataclass — verify it accepts all required fields
        profile = InvestorProfile(
            age=30, income=75000.0, horizon=10, experience=5, risk_score=5
        )
        assert profile.age == 30
        assert profile.risk_score == 5

        # RiskClassifier exposes validate() and classify() methods
        classifier = RiskClassifier()
        assert hasattr(classifier, "validate")
        assert hasattr(classifier, "classify")

    def test_stock_recommender_exports(self):
        """StockRecommender is importable and has the expected interface."""
        # FEATURE_COLUMNS lists all 9 expected technical indicators
        assert len(FEATURE_COLUMNS) == 9
        assert "close_price" in FEATURE_COLUMNS
        assert "RSI" in FEATURE_COLUMNS
        assert "MACD" in FEATURE_COLUMNS

        # StockRecommender exposes predict() method
        # (instantiation requires model files — tested in pipeline test below)
        assert hasattr(StockRecommender, "predict")

    def test_suitability_engine_exports(self):
        """SuitabilityEngine is importable and has the expected interface."""
        engine = SuitabilityEngine()

        # Verify the MAPPING covers all valid combinations
        # 6 risk categories × 3 stock signals = 18 entries
        assert len(engine.MAPPING) == 18

        # Verify EXPLANATIONS covers all valid combinations
        assert len(engine.EXPLANATIONS) == 18

        # Verify exposed methods
        assert hasattr(engine, "recommend")
        assert hasattr(engine, "generate_explanation")

    def test_features_exports(self):
        """Feature engineering functions are importable."""
        assert callable(compute_features)
        assert callable(compute_target_labels)

    def test_validation_exports(self):
        """Validation functions are importable."""
        assert callable(validate_investor_profile)
        assert callable(validate_risk_category)
        assert callable(validate_stock_signal)


@pytest.mark.integration
class TestRiskClassifierFlow:
    """Verify the InvestorProfile → RiskClassifier → Risk_Category flow.

    This tests that:
    1. InvestorProfile accepts valid investor data
    2. RiskClassifier.validate() accepts valid profiles
    3. RiskClassifier.classify() produces a valid Risk_Category
    4. The output is one of {Conservative, Moderate, Aggressive}
    """

    def test_conservative_investor_flow(self):
        """Low risk_score investor is classified as Conservative."""
        # Step 1: Create an investor profile with low risk tolerance
        profile = InvestorProfile(
            age=65, income=50000.0, horizon=2, experience=1, risk_score=2
        )

        # Step 2: Validate the profile — should return no errors
        classifier = RiskClassifier()
        errors = classifier.validate(profile)
        assert errors == [], f"Unexpected validation errors: {errors}"

        # Step 3: Classify — should be Conservative (low score + adjustments)
        risk_category = classifier.classify(profile)
        assert risk_category == "Conservative"
        assert risk_category in {"Conservative", "Moderate", "Aggressive"}

    def test_moderate_investor_flow(self):
        """Mid-range risk_score investor is classified as Moderate."""
        # Step 1: Create an investor profile with moderate risk tolerance
        profile = InvestorProfile(
            age=35, income=90000.0, horizon=10, experience=5, risk_score=5
        )

        # Step 2: Validate — no errors expected
        classifier = RiskClassifier()
        errors = classifier.validate(profile)
        assert errors == []

        # Step 3: Classify — should be Moderate (mid score, no adjustments)
        risk_category = classifier.classify(profile)
        assert risk_category == "Moderate"

    def test_aggressive_investor_flow(self):
        """High risk_score investor is classified as Aggressive."""
        # Step 1: Create an investor profile with high risk tolerance
        profile = InvestorProfile(
            age=28, income=120000.0, horizon=20, experience=8, risk_score=9
        )

        # Step 2: Validate — no errors expected
        classifier = RiskClassifier()
        errors = classifier.validate(profile)
        assert errors == []

        # Step 3: Classify — should be Aggressive (high score, no adjustments)
        risk_category = classifier.classify(profile)
        assert risk_category == "Aggressive"


@pytest.mark.integration
class TestStockRecommenderFlow:
    """Verify that StockRecommender loads the exported model and produces predictions.

    This tests that:
    1. StockRecommender can load model artifacts from models/ directory
    2. The predict() method accepts a features dictionary
    3. The output is one of {Buy, Hold, Avoid}

    Requirements validated: 6.2
    """

    def test_model_loads_successfully(self):
        """StockRecommender loads the synthetic model without errors."""
        # The conftest creates synthetic model artifacts in models/
        # StockRecommender.__init__ loads model and scaler from disk
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Verify model and scaler were loaded
        assert recommender.model is not None
        assert recommender.scaler is not None

    def test_predict_returns_valid_signal(self):
        """StockRecommender.predict() returns a valid Stock_Signal."""
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Create a sample feature dictionary with all 9 required indicators
        sample_features = {
            "close_price": 150.0,
            "daily_return": 0.02,
            "ma_5": 148.0,
            "ma_20": 145.0,
            "ma_50": 140.0,
            "volatility": 0.025,
            "volume": 50_000_000,
            "RSI": 55.0,
            "MACD": 2.5,
        }

        # Predict — should return one of the valid stock signals
        stock_signal = recommender.predict(sample_features)
        assert stock_signal in {"Buy", "Hold", "Avoid"}, (
            f"Expected one of Buy/Hold/Avoid, got '{stock_signal}'"
        )

    def test_predict_with_different_feature_values(self):
        """StockRecommender handles various feature value ranges."""
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Test with different feature profiles (bearish indicators)
        bearish_features = {
            "close_price": 80.0,
            "daily_return": -0.03,
            "ma_5": 82.0,
            "ma_20": 90.0,
            "ma_50": 100.0,
            "volatility": 0.04,
            "volume": 80_000_000,
            "RSI": 25.0,
            "MACD": -3.0,
        }

        stock_signal = recommender.predict(bearish_features)
        assert stock_signal in {"Buy", "Hold", "Avoid"}


@pytest.mark.integration
class TestFullPipelineFlow:
    """Verify the complete end-to-end flow wiring all components together.

    Full pipeline:
        InvestorProfile → RiskClassifier → Risk_Category
        Features → StockRecommender → Stock_Signal
        (Risk_Category, Stock_Signal) → SuitabilityEngine → Suitability_Rating

    This is the critical integration test that proves all components connect
    correctly. Each step feeds its output to the next component.

    Requirements validated: 1.1, 6.2, 7.1
    """

    def test_full_pipeline_conservative_investor(self):
        """End-to-end: Conservative investor gets a suitability recommendation."""
        # === STAGE 1: Risk Classification ===
        # Create a conservative investor profile
        profile = InvestorProfile(
            age=70, income=40000.0, horizon=2, experience=1, risk_score=3
        )

        # Validate and classify the investor
        classifier = RiskClassifier()
        errors = classifier.validate(profile)
        assert errors == [], f"Validation failed: {errors}"

        # RiskClassifier output → Risk_Category
        risk_category = classifier.classify(profile)
        assert risk_category in {"Conservative", "Moderate", "Aggressive"}

        # === STAGE 2: Stock Recommendation ===
        # Load the model and get a stock signal from technical indicators
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        # Sample stock features (technical indicators)
        stock_features = {
            "close_price": 200.0,
            "daily_return": 0.01,
            "ma_5": 198.0,
            "ma_20": 195.0,
            "ma_50": 190.0,
            "volatility": 0.02,
            "volume": 30_000_000,
            "RSI": 60.0,
            "MACD": 1.5,
        }

        # StockRecommender output → Stock_Signal
        stock_signal = recommender.predict(stock_features)
        assert stock_signal in {"Buy", "Hold", "Avoid"}

        # === STAGE 3: Suitability Recommendation ===
        # SuitabilityEngine combines Risk_Category + Stock_Signal → Rating
        engine = SuitabilityEngine()
        result = engine.recommend(risk_category, stock_signal)

        # Verify the result structure: must have 'rating' and 'explanation'
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "rating" in result
        assert "explanation" in result

        # Verify the rating is one of the valid suitability ratings
        assert result["rating"] in {"Suitable", "Use Caution", "Not Suitable"}

        # Verify the explanation is a non-empty string
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    def test_full_pipeline_moderate_investor(self):
        """End-to-end: Moderate investor gets a suitability recommendation."""
        # === STAGE 1: Risk Classification ===
        profile = InvestorProfile(
            age=40, income=95000.0, horizon=15, experience=7, risk_score=5
        )

        classifier = RiskClassifier()
        errors = classifier.validate(profile)
        assert errors == []

        # RiskClassifier output feeds into SuitabilityEngine
        risk_category = classifier.classify(profile)
        assert risk_category == "Moderate"

        # === STAGE 2: Stock Recommendation ===
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        stock_features = {
            "close_price": 350.0,
            "daily_return": 0.03,
            "ma_5": 345.0,
            "ma_20": 330.0,
            "ma_50": 310.0,
            "volatility": 0.03,
            "volume": 60_000_000,
            "RSI": 65.0,
            "MACD": 4.0,
        }

        # StockRecommender output feeds into SuitabilityEngine
        stock_signal = recommender.predict(stock_features)
        assert stock_signal in {"Buy", "Hold", "Avoid"}

        # === STAGE 3: Suitability Recommendation ===
        engine = SuitabilityEngine()
        result = engine.recommend(risk_category, stock_signal)

        # Full pipeline produces a valid recommendation
        assert "error" not in result
        assert result["rating"] in {"Suitable", "Use Caution", "Not Suitable"}
        assert len(result["explanation"]) > 0

    def test_full_pipeline_aggressive_investor(self):
        """End-to-end: Aggressive investor gets a suitability recommendation."""
        # === STAGE 1: Risk Classification ===
        profile = InvestorProfile(
            age=25, income=150000.0, horizon=25, experience=10, risk_score=9
        )

        classifier = RiskClassifier()
        errors = classifier.validate(profile)
        assert errors == []

        # RiskClassifier output feeds into SuitabilityEngine
        risk_category = classifier.classify(profile)
        assert risk_category == "Aggressive"

        # === STAGE 2: Stock Recommendation ===
        recommender = StockRecommender(
            model_path="models/stock_model.pkl",
            scaler_path="models/scaler.pkl",
        )

        stock_features = {
            "close_price": 500.0,
            "daily_return": -0.02,
            "ma_5": 510.0,
            "ma_20": 520.0,
            "ma_50": 480.0,
            "volatility": 0.04,
            "volume": 90_000_000,
            "RSI": 35.0,
            "MACD": -2.0,
        }

        # StockRecommender output feeds into SuitabilityEngine
        stock_signal = recommender.predict(stock_features)
        assert stock_signal in {"Buy", "Hold", "Avoid"}

        # === STAGE 3: Suitability Recommendation ===
        engine = SuitabilityEngine()
        result = engine.recommend(risk_category, stock_signal)

        # Full pipeline produces a valid recommendation
        assert "error" not in result
        assert result["rating"] in {"Suitable", "Use Caution", "Not Suitable"}
        assert len(result["explanation"]) > 0

    def test_full_pipeline_all_suitability_combinations(self):
        """Verify all 9 suitability combinations can be produced by the engine.

        This test directly exercises the SuitabilityEngine with all valid
        (Risk_Category, Stock_Signal) pairs to confirm the wiring between
        RiskClassifier output and StockRecommender output is correctly consumed.
        """
        engine = SuitabilityEngine()

        # All valid Risk Categories (output from RiskClassifier)
        risk_categories = ["Conservative", "Moderate", "Aggressive"]

        # All valid Stock Signals (output from StockRecommender)
        stock_signals = ["Buy", "Hold", "Avoid"]

        # Expected mapping from the requirements (Requirement 7.1)
        expected_mapping = {
            ("Conservative", "Buy"):   "Use Caution",
            ("Conservative", "Hold"):  "Not Suitable",
            ("Conservative", "Avoid"): "Not Suitable",
            ("Moderate", "Buy"):       "Suitable",
            ("Moderate", "Hold"):      "Use Caution",
            ("Moderate", "Avoid"):     "Not Suitable",
            ("Aggressive", "Buy"):     "Suitable",
            ("Aggressive", "Hold"):    "Suitable",
            ("Aggressive", "Avoid"):   "Use Caution",
        }

        # Exercise every combination through the engine
        for risk_cat in risk_categories:
            for signal in stock_signals:
                result = engine.recommend(risk_cat, signal)

                # Verify no errors
                assert "error" not in result, (
                    f"Error for ({risk_cat}, {signal}): {result.get('error')}"
                )

                # Verify rating matches the expected mapping
                expected_rating = expected_mapping[(risk_cat, signal)]
                assert result["rating"] == expected_rating, (
                    f"For ({risk_cat}, {signal}): "
                    f"expected '{expected_rating}', got '{result['rating']}'"
                )

                # Verify explanation is present and meaningful
                assert "explanation" in result
                assert len(result["explanation"]) > 20  # Not trivially short


@pytest.mark.integration
class TestValidationIntegration:
    """Verify validation functions work correctly with the classification pipeline.

    Tests that invalid inputs are caught before reaching the classifier,
    and valid inputs flow through without errors.
    """

    def test_invalid_profile_stops_before_classification(self):
        """Invalid profiles are caught by validation before classification."""
        # Create a profile with invalid fields
        profile = InvestorProfile(
            age=15,       # Below minimum (18)
            income=-100,  # Below minimum (0)
            horizon=0,    # Below minimum (1)
            experience=0, # Valid (min is 0)
            risk_score=5, # Valid
        )

        classifier = RiskClassifier()
        errors = classifier.validate(profile)

        # Should catch the 3 invalid fields (age, income, horizon)
        assert len(errors) == 3
        # Each error should name the invalid field
        field_names_in_errors = " ".join(errors)
        assert "age" in field_names_in_errors
        assert "income" in field_names_in_errors
        assert "horizon" in field_names_in_errors

    def test_suitability_rejects_invalid_risk_category(self):
        """SuitabilityEngine returns error for invalid Risk_Category."""
        engine = SuitabilityEngine()

        # Pass an invalid risk category (would never come from RiskClassifier)
        result = engine.recommend("InvalidCategory", "Buy")
        assert "error" in result
        assert "InvalidCategory" in result["error"]

    def test_suitability_rejects_invalid_stock_signal(self):
        """SuitabilityEngine returns error for invalid Stock_Signal."""
        engine = SuitabilityEngine()

        # Pass an invalid stock signal (would never come from StockRecommender)
        result = engine.recommend("Moderate", "InvalidSignal")
        assert "error" in result
        assert "InvalidSignal" in result["error"]
