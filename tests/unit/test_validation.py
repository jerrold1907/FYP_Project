"""Unit tests for src/validation.py."""

import pytest

from src.validation import (
    validate_investor_profile,
    validate_risk_category,
    validate_stock_signal,
)


class TestValidateInvestorProfile:
    """Tests for validate_investor_profile function."""

    def test_valid_profile_returns_empty_list(self):
        """A fully valid profile should produce no errors."""
        data = {
            "age": 30,
            "income": 50000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        assert validate_investor_profile(data) == []

    def test_valid_profile_boundary_min(self):
        """All fields at their minimum valid values."""
        data = {
            "age": 18,
            "income": 0.00,
            "horizon": 1,
            "experience": 0,
            "risk_score": 1,
        }
        assert validate_investor_profile(data) == []

    def test_valid_profile_boundary_max(self):
        """All fields at their maximum valid values."""
        data = {
            "age": 120,
            "income": 999_999_999.99,
            "horizon": 50,
            "experience": 50,
            "risk_score": 10,
        }
        assert validate_investor_profile(data) == []

    def test_missing_single_field(self):
        """Missing a single field produces exactly one error."""
        data = {
            "income": 50000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 1
        assert "age" in errors[0]

    def test_missing_all_fields(self):
        """Missing all fields produces five errors."""
        errors = validate_investor_profile({})
        assert len(errors) == 5

    def test_all_fields_out_of_range(self):
        """All fields out of range produces five errors."""
        data = {
            "age": 17,
            "income": -1.0,
            "horizon": 0,
            "experience": -1,
            "risk_score": 0,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 5

    def test_age_below_range(self):
        """Age below 18 is invalid."""
        data = {
            "age": 17,
            "income": 50000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 1
        assert "age" in errors[0]
        assert "18" in errors[0]
        assert "120" in errors[0]

    def test_age_above_range(self):
        """Age above 120 is invalid."""
        data = {
            "age": 121,
            "income": 50000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 1
        assert "age" in errors[0]

    def test_income_above_range(self):
        """Income above 999999999.99 is invalid."""
        data = {
            "age": 30,
            "income": 1_000_000_000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 1
        assert "income" in errors[0]

    def test_income_accepts_integer(self):
        """Income field should accept int values (treated as float)."""
        data = {
            "age": 30,
            "income": 50000,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        assert validate_investor_profile(data) == []

    def test_wrong_type_string(self):
        """String values for numeric fields are invalid."""
        data = {
            "age": "thirty",
            "income": 50000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 1
        assert "age" in errors[0]

    def test_boolean_rejected_for_int_field(self):
        """Boolean values should be rejected for int fields."""
        data = {
            "age": True,
            "income": 50000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 1
        assert "age" in errors[0]

    def test_none_value_treated_as_missing(self):
        """None values should be treated as missing."""
        data = {
            "age": None,
            "income": 50000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 1
        assert "age" in errors[0]

    def test_multiple_invalid_fields_all_reported(self):
        """Multiple invalid fields produce multiple errors, one per field."""
        data = {
            "age": 17,
            "income": -1.0,
            "horizon": 10,
            "experience": 5,
            "risk_score": 11,
        }
        errors = validate_investor_profile(data)
        assert len(errors) == 3
        field_names_in_errors = " ".join(errors)
        assert "age" in field_names_in_errors
        assert "income" in field_names_in_errors
        assert "risk_score" in field_names_in_errors

    def test_error_messages_include_range(self):
        """Error messages should include the accepted range."""
        data = {
            "age": 200,
            "income": 50000.00,
            "horizon": 10,
            "experience": 5,
            "risk_score": 5,
        }
        errors = validate_investor_profile(data)
        assert "18" in errors[0]
        assert "120" in errors[0]


class TestValidateRiskCategory:
    """Tests for validate_risk_category function."""

    def test_conservative_valid(self):
        assert validate_risk_category("Conservative") is None

    def test_moderate_valid(self):
        assert validate_risk_category("Moderate") is None

    def test_aggressive_valid(self):
        assert validate_risk_category("Aggressive") is None

    def test_lowercase_invalid(self):
        """Case-sensitive: lowercase is invalid."""
        result = validate_risk_category("conservative")
        assert result is not None
        assert "conservative" in result

    def test_uppercase_invalid(self):
        """Case-sensitive: uppercase is invalid."""
        result = validate_risk_category("MODERATE")
        assert result is not None
        assert "MODERATE" in result

    def test_empty_string_invalid(self):
        result = validate_risk_category("")
        assert result is not None

    def test_random_string_invalid(self):
        result = validate_risk_category("Unknown")
        assert result is not None
        assert "Unknown" in result


class TestValidateStockSignal:
    """Tests for validate_stock_signal function."""

    def test_buy_valid(self):
        assert validate_stock_signal("Buy") is None

    def test_hold_valid(self):
        assert validate_stock_signal("Hold") is None

    def test_avoid_valid(self):
        assert validate_stock_signal("Avoid") is None

    def test_lowercase_invalid(self):
        """Case-sensitive: lowercase is invalid."""
        result = validate_stock_signal("buy")
        assert result is not None
        assert "buy" in result

    def test_uppercase_invalid(self):
        """Case-sensitive: uppercase is invalid."""
        result = validate_stock_signal("HOLD")
        assert result is not None
        assert "HOLD" in result

    def test_empty_string_invalid(self):
        result = validate_stock_signal("")
        assert result is not None

    def test_sell_invalid(self):
        """'Sell' is not a valid stock signal."""
        result = validate_stock_signal("Sell")
        assert result is not None
        assert "Sell" in result
