"""Unit tests for the Suitability Engine module."""

import pytest
from src.suitability import SuitabilityEngine


@pytest.fixture
def engine():
    """Create a SuitabilityEngine instance for testing."""
    return SuitabilityEngine()


class TestSuitabilityMapping:
    """Tests verifying the suitability mapping table is correct."""

    def test_conservative_buy(self, engine):
        result = engine.recommend("Conservative", "Buy")
        assert result["rating"] == "Use Caution"

    def test_conservative_hold(self, engine):
        result = engine.recommend("Conservative", "Hold")
        assert result["rating"] == "Not Suitable"

    def test_conservative_avoid(self, engine):
        result = engine.recommend("Conservative", "Avoid")
        assert result["rating"] == "Not Suitable"

    def test_moderate_buy(self, engine):
        result = engine.recommend("Moderate", "Buy")
        assert result["rating"] == "Suitable"

    def test_moderate_hold(self, engine):
        result = engine.recommend("Moderate", "Hold")
        assert result["rating"] == "Use Caution"

    def test_moderate_avoid(self, engine):
        result = engine.recommend("Moderate", "Avoid")
        assert result["rating"] == "Not Suitable"

    def test_aggressive_buy(self, engine):
        result = engine.recommend("Aggressive", "Buy")
        assert result["rating"] == "Suitable"

    def test_aggressive_hold(self, engine):
        result = engine.recommend("Aggressive", "Hold")
        assert result["rating"] == "Suitable"

    def test_aggressive_avoid(self, engine):
        result = engine.recommend("Aggressive", "Avoid")
        assert result["rating"] == "Use Caution"


class TestExplanations:
    """Tests verifying explanations are returned and contain relevant info."""

    def test_explanation_returned_for_valid_inputs(self, engine):
        result = engine.recommend("Moderate", "Buy")
        assert "explanation" in result
        assert len(result["explanation"]) > 0

    def test_explanation_contains_risk_category(self, engine):
        result = engine.recommend("Aggressive", "Hold")
        assert "Aggressive" in result["explanation"]

    def test_explanation_contains_stock_signal(self, engine):
        result = engine.recommend("Conservative", "Buy")
        assert "Buy" in result["explanation"]

    def test_explanation_references_rating(self, engine):
        result = engine.recommend("Moderate", "Buy")
        assert "Suitable" in result["explanation"]

    def test_generate_explanation_directly(self, engine):
        explanation = engine.generate_explanation("Conservative", "Buy")
        assert "Conservative" in explanation
        assert "Buy" in explanation
        assert "Use Caution" in explanation


class TestInvalidInputs:
    """Tests verifying error handling for invalid inputs."""

    def test_none_risk_category(self, engine):
        result = engine.recommend(None, "Buy")
        assert "error" in result
        assert "Risk_Category" in result["error"]

    def test_empty_risk_category(self, engine):
        result = engine.recommend("", "Buy")
        assert "error" in result
        assert "Risk_Category" in result["error"]

    def test_none_stock_signal(self, engine):
        result = engine.recommend("Moderate", None)
        assert "error" in result
        assert "Stock_Signal" in result["error"]

    def test_empty_stock_signal(self, engine):
        result = engine.recommend("Moderate", "")
        assert "error" in result
        assert "Stock_Signal" in result["error"]

    def test_invalid_risk_category(self, engine):
        result = engine.recommend("Unknown", "Buy")
        assert "error" in result
        assert "Risk_Category" in result["error"]
        assert "Unknown" in result["error"]

    def test_invalid_stock_signal(self, engine):
        result = engine.recommend("Moderate", "Sell")
        assert "error" in result
        assert "Stock_Signal" in result["error"]
        assert "Sell" in result["error"]

    def test_case_sensitive_risk_category(self, engine):
        result = engine.recommend("conservative", "Buy")
        assert "error" in result

    def test_case_sensitive_stock_signal(self, engine):
        result = engine.recommend("Moderate", "buy")
        assert "error" in result

    def test_no_explanation_on_error(self, engine):
        result = engine.recommend("Invalid", "Buy")
        assert "explanation" not in result
        assert "rating" not in result
