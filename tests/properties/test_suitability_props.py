"""Property-based tests for the Suitability Engine module.

Uses Hypothesis to verify universal properties of SuitabilityEngine.recommend():
- Property 11: Invalid suitability inputs produce descriptive errors.

Why property-based testing here:
    The space of invalid inputs is unbounded — any string not in {Conservative,
    Moderate, Aggressive} or {Buy, Hold, Avoid} should trigger a descriptive
    error. Example-based tests can only cover a handful of cases. Hypothesis
    generates diverse random strings (unicode, empty-ish, case variants, garbage)
    ensuring the error handling is robust across the entire invalid input space.
"""

from hypothesis import given, assume, settings
from hypothesis.strategies import (
    text,
    one_of,
    just,
    composite,
    sampled_from,
    booleans,
)

from src.suitability import SuitabilityEngine


# =============================================================================
# Feature: ai-stock-recommendation, Property 11: Invalid suitability inputs produce descriptive errors
# =============================================================================

# --- Valid sets (used to filter out valid values from generated strings) ---
VALID_RISK_CATEGORIES = {
    "Conservative", "Moderately Conservative", "Moderate",
    "Balanced", "Growth", "Aggressive",
}
VALID_STOCK_SIGNALS = {"Buy", "Hold", "Avoid"}


@composite
def invalid_risk_category_strings(draw):
    """Hypothesis strategy for generating strings NOT in the valid Risk_Category set.

    Strategy approach:
    - Uses text() to generate arbitrary unicode strings of varying lengths.
    - Uses assume() to reject any string that happens to match a valid category.
    - Also includes case variants (e.g., "conservative", "MODERATE") which are
      invalid because validation is case-sensitive per Requirement 7.2.
    - Includes common near-miss strings to increase the chance of finding edge cases.

    The text strategy generates a wide range of strings including empty strings,
    single characters, unicode, and long strings — all of which should be rejected
    by the validator as invalid Risk_Category values.
    """
    # Draw from a mix of random text and common "near miss" case variants
    value = draw(
        one_of(
            # Random text: covers unicode, empty-adjacent strings, garbage
            text(min_size=1, max_size=50),
            # Case variants that are explicitly invalid (case-sensitive matching)
            sampled_from([
                "conservative", "moderate", "aggressive",
                "CONSERVATIVE", "MODERATE", "AGGRESSIVE",
                "Conservative ", " Moderate", "Agressive",  # typo
                "Buy", "Hold", "Avoid",  # valid signals, not categories
                "None", "null", "undefined",
            ]),
        )
    )
    # Filter out any accidentally-generated valid categories
    assume(value not in VALID_RISK_CATEGORIES)
    return value


@composite
def invalid_stock_signal_strings(draw):
    """Hypothesis strategy for generating strings NOT in the valid Stock_Signal set.

    Strategy approach:
    - Uses text() to generate arbitrary unicode strings of varying lengths.
    - Uses assume() to reject any string that happens to match a valid signal.
    - Also includes case variants (e.g., "buy", "HOLD") which are invalid
      because validation is case-sensitive per Requirement 7.2.
    - Includes common near-miss strings like "Sell" or "Wait" that a user
      might mistakenly provide.

    This strategy ensures we test the full breadth of invalid signal inputs,
    from obvious garbage to plausible-but-wrong values.
    """
    # Draw from a mix of random text and common "near miss" values
    value = draw(
        one_of(
            # Random text: covers unicode, short strings, long strings
            text(min_size=1, max_size=50),
            # Case variants and common mistakes that are explicitly invalid
            sampled_from([
                "buy", "hold", "avoid",
                "BUY", "HOLD", "AVOID",
                "Buy ", " Hold", "Avoi",  # typo / whitespace
                "Sell", "Wait", "Pass",
                "Conservative", "Moderate", "Aggressive",  # valid categories, not signals
                "None", "null", "undefined",
            ]),
        )
    )
    # Filter out any accidentally-generated valid signals
    assume(value not in VALID_STOCK_SIGNALS)
    return value


# --- Instantiate the engine once for all property tests ---
_engine = SuitabilityEngine()


@settings(max_examples=100)
@given(
    invalid_risk=invalid_risk_category_strings(),
    stock_signal=sampled_from(sorted(VALID_STOCK_SIGNALS)),
)
def test_invalid_risk_category_produces_descriptive_error(invalid_risk, stock_signal):
    """Property 11 (Part A): Invalid Risk_Category with valid Stock_Signal produces error.

    **Validates: Requirements 7.2, 7.3**

    This property verifies that when an invalid Risk_Category string is provided
    (any string not in {Conservative, Moderate, Aggressive}, matched case-sensitively),
    the Suitability_Engine SHALL return an error message that:
    1. Identifies which input was invalid (mentions "Risk_Category")
    2. States the value that was received (contains the invalid string)
    3. Does NOT return an explanation key (error responses have no explanation)

    The test uses Hypothesis to generate diverse invalid strings including:
    - Random unicode text
    - Case variants (e.g., "conservative") that fail case-sensitive matching
    - Common near-miss values and typos
    """
    # Call the recommend method with an invalid risk category
    result = _engine.recommend(invalid_risk, stock_signal)

    # Assertion 1: The result must contain an "error" key indicating validation failure
    assert "error" in result, (
        f"Expected 'error' key in result for invalid Risk_Category '{invalid_risk}', "
        f"but got: {result}"
    )

    error_msg = result["error"]

    # Assertion 2: The error message must identify WHICH input is invalid.
    # It should mention "Risk_Category" so the user knows which field failed.
    assert "Risk_Category" in error_msg, (
        f"Error message should identify 'Risk_Category' as the invalid input, "
        f"but got: '{error_msg}'"
    )

    # Assertion 3: The error message must state the received value.
    # This allows the user to see what they passed and understand why it failed.
    assert invalid_risk in error_msg, (
        f"Error message should contain the received value '{invalid_risk}', "
        f"but got: '{error_msg}'"
    )

    # Assertion 4: No explanation should be returned for invalid inputs.
    # Error responses must NOT contain a rating or explanation — only the error key.
    assert "explanation" not in result, (
        f"Invalid input should not produce an 'explanation', "
        f"but result contains: {result}"
    )

    # Assertion 5: No rating should be returned for invalid inputs either.
    assert "rating" not in result, (
        f"Invalid input should not produce a 'rating', "
        f"but result contains: {result}"
    )


@settings(max_examples=100)
@given(
    risk_category=sampled_from(sorted(VALID_RISK_CATEGORIES)),
    invalid_signal=invalid_stock_signal_strings(),
)
def test_invalid_stock_signal_produces_descriptive_error(risk_category, invalid_signal):
    """Property 11 (Part B): Valid Risk_Category with invalid Stock_Signal produces error.

    **Validates: Requirements 7.2, 7.3**

    This property verifies that when an invalid Stock_Signal string is provided
    (any string not in {Buy, Hold, Avoid}, matched case-sensitively),
    the Suitability_Engine SHALL return an error message that:
    1. Identifies which input was invalid (mentions "Stock_Signal")
    2. States the value that was received (contains the invalid string)
    3. Does NOT return an explanation key (error responses have no explanation)

    The test uses Hypothesis to generate diverse invalid signal strings including:
    - Random unicode text
    - Case variants (e.g., "buy", "HOLD") that fail case-sensitive matching
    - Common mistakes like "Sell" or "Wait"
    """
    # Call the recommend method with a valid risk category but invalid signal
    result = _engine.recommend(risk_category, invalid_signal)

    # Assertion 1: The result must contain an "error" key indicating validation failure
    assert "error" in result, (
        f"Expected 'error' key in result for invalid Stock_Signal '{invalid_signal}', "
        f"but got: {result}"
    )

    error_msg = result["error"]

    # Assertion 2: The error message must identify WHICH input is invalid.
    # It should mention "Stock_Signal" so the user knows which field failed.
    assert "Stock_Signal" in error_msg, (
        f"Error message should identify 'Stock_Signal' as the invalid input, "
        f"but got: '{error_msg}'"
    )

    # Assertion 3: The error message must state the received value.
    # This allows the user to see what they passed and understand why it failed.
    assert invalid_signal in error_msg, (
        f"Error message should contain the received value '{invalid_signal}', "
        f"but got: '{error_msg}'"
    )

    # Assertion 4: No explanation should be returned for invalid inputs.
    # Error responses must NOT contain a rating or explanation — only the error key.
    assert "explanation" not in result, (
        f"Invalid input should not produce an 'explanation', "
        f"but result contains: {result}"
    )

    # Assertion 5: No rating should be returned for invalid inputs either.
    assert "rating" not in result, (
        f"Invalid input should not produce a 'rating', "
        f"but result contains: {result}"
    )


@settings(max_examples=100)
@given(
    invalid_risk=invalid_risk_category_strings(),
    invalid_signal=invalid_stock_signal_strings(),
)
def test_both_inputs_invalid_produces_error(invalid_risk, invalid_signal):
    """Property 11 (Part C): Both inputs invalid still produces a descriptive error.

    **Validates: Requirements 7.2, 7.3**

    This property verifies that when BOTH Risk_Category and Stock_Signal are
    invalid, the Suitability_Engine still returns an error response (not a crash
    or unexpected behavior). The error should at minimum identify the first
    invalid input encountered during validation.

    Note: The current implementation validates Risk_Category first, so the error
    will reference Risk_Category. This test ensures the engine is robust when
    receiving completely garbage inputs for both parameters.
    """
    # Call the recommend method with both inputs invalid
    result = _engine.recommend(invalid_risk, invalid_signal)

    # Assertion 1: The result must contain an "error" key
    assert "error" in result, (
        f"Expected 'error' key when both inputs are invalid "
        f"(risk='{invalid_risk}', signal='{invalid_signal}'), but got: {result}"
    )

    error_msg = result["error"]

    # Assertion 2: The error should identify at least one invalid input.
    # Since validation checks Risk_Category first, it should mention Risk_Category.
    assert "Risk_Category" in error_msg or "Stock_Signal" in error_msg, (
        f"Error message should identify which input is invalid, "
        f"but got: '{error_msg}'"
    )

    # Assertion 3: The error should contain the received invalid value
    # (at least the first one detected: the invalid risk category)
    assert invalid_risk in error_msg or invalid_signal in error_msg, (
        f"Error message should state the received value, "
        f"but got: '{error_msg}'"
    )

    # Assertion 4: No explanation should be returned for invalid inputs
    assert "explanation" not in result, (
        f"Invalid inputs should not produce an 'explanation', "
        f"but result contains: {result}"
    )

    # Assertion 5: No rating should be returned for invalid inputs
    assert "rating" not in result, (
        f"Invalid inputs should not produce a 'rating', "
        f"but result contains: {result}"
    )
