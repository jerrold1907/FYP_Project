"""Property-based tests for the validation module.

Uses Hypothesis to verify universal properties of validate_investor_profile():
- Property 2: Valid profiles always produce zero errors.
- Property 3: Profiles with N invalid/missing fields produce exactly N errors,
  each identifying the offending field by name and stating the accepted range.

Why property-based testing here:
    A handful of hand-picked examples cannot cover the enormous input space
    (age: 18-120, income: 0.00-999999999.99, horizon: 1-50, experience: 0-50,
    risk_score: 1-10). Hypothesis explores boundary values, midpoints, and
    random combinations that example-based tests would miss.
"""

from hypothesis import given, assume
from hypothesis.strategies import (
    integers,
    floats,
    composite,
    sampled_from,
    lists,
    just,
    one_of,
    booleans,
)

from src.validation import validate_investor_profile, INVESTOR_PROFILE_SCHEMA


# =============================================================================
# Feature: ai-stock-recommendation, Property 2: Valid investor profiles are accepted without errors
# =============================================================================


@composite
def valid_investor_profiles(draw):
    """Composite strategy generating investor profiles within all valid ranges.

    Each field is constrained to its schema-defined range:
        - age: integer in [18, 120]
        - income: float in [0.00, 999_999_999.99], no NaN/Inf
        - horizon: integer in [1, 50]
        - experience: integer in [0, 50]
        - risk_score: integer in [1, 10]

    NaN and Infinity are explicitly excluded from the income strategy because
    they are not valid numeric values and would fail type/range checks. The
    composite pattern is used so all fields are drawn independently, allowing
    Hypothesis to shrink each field individually when a failure is found.
    """
    # Generate age within the valid range for an investor (18 to 120 inclusive)
    age = draw(integers(min_value=18, max_value=120))

    # Generate income as a float within the valid monetary range.
    # allow_nan=False and allow_infinity=False ensure only finite numbers.
    income = draw(
        floats(
            min_value=0.00,
            max_value=999_999_999.99,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    # Generate investment horizon in years (1 to 50 inclusive)
    horizon = draw(integers(min_value=1, max_value=50))

    # Generate investment experience in years (0 to 50 inclusive, 0 = no experience)
    experience = draw(integers(min_value=0, max_value=50))

    # Generate risk tolerance score on the 1-10 scale
    risk_score = draw(integers(min_value=1, max_value=10))

    return {
        "age": age,
        "income": income,
        "horizon": horizon,
        "experience": experience,
        "risk_score": risk_score,
    }


@given(profile=valid_investor_profiles())
def test_valid_profiles_accepted_without_errors(profile):
    """Property 2: Valid investor profiles are accepted without errors.

    **Validates: Requirements 1.2**

    This property asserts that for any investor profile where:
        - age is an integer in [18, 120]
        - income is a float in [0.00, 999_999_999.99]
        - horizon is an integer in [1, 50]
        - experience is an integer in [0, 50]
        - risk_score is an integer in [1, 10]

    the validate_investor_profile() function SHALL return an empty error list,
    meaning the profile is fully accepted with no validation failures.
    """
    # Call the validation function with the generated valid profile
    errors = validate_investor_profile(profile)

    # A valid profile must produce zero errors - any non-empty list is a bug
    assert errors == [], (
        f"Expected no errors for valid profile {profile}, got: {errors}"
    )


# =============================================================================
# Feature: ai-stock-recommendation, Property 3: Invalid investor profiles report all errors with field names and ranges
# =============================================================================


# --- Strategies for generating invalid field values ---
# Each strategy below produces a value that is OUTSIDE the valid range for its
# field, or is of the wrong type, ensuring the validator flags it as invalid.


@composite
def invalid_age(draw):
    """Generate an age value outside valid range [18, 120] or of wrong type.

    We use one_of to choose between: too low, too high, or wrong type
    (float/string/bool). This covers all ways the field can be invalid.
    """
    return draw(
        one_of(
            # Age below minimum (could be negative or 0-17)
            integers(max_value=17),
            # Age above maximum
            integers(min_value=121),
            # Wrong type: float value
            just(25.5),
            # Wrong type: string
            just("thirty"),
            # Wrong type: boolean (bool is subclass of int, but should be rejected)
            booleans(),
        )
    )


@composite
def invalid_income(draw):
    """Generate an income value outside valid range [0.00, 999999999.99] or wrong type.

    Negative floats or floats above max are out of range; strings/bools are wrong type.
    """
    return draw(
        one_of(
            # Negative income (below minimum)
            floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
            # Income above maximum
            floats(
                min_value=1_000_000_000.00,
                max_value=1e15,
                allow_nan=False,
                allow_infinity=False,
            ),
            # Wrong type: string
            just("rich"),
            # Wrong type: boolean
            booleans(),
        )
    )


@composite
def invalid_horizon(draw):
    """Generate an investment horizon outside valid range [1, 50] or wrong type."""
    return draw(
        one_of(
            # Below minimum (0 or negative)
            integers(max_value=0),
            # Above maximum
            integers(min_value=51),
            # Wrong type: float
            just(5.5),
            # Wrong type: string
            just("long"),
            # Wrong type: boolean
            booleans(),
        )
    )


@composite
def invalid_experience(draw):
    """Generate experience outside valid range [0, 50] or wrong type."""
    return draw(
        one_of(
            # Below minimum (negative)
            integers(max_value=-1),
            # Above maximum
            integers(min_value=51),
            # Wrong type: float
            just(3.5),
            # Wrong type: string
            just("novice"),
            # Wrong type: boolean
            booleans(),
        )
    )


@composite
def invalid_risk_score(draw):
    """Generate risk_score outside valid range [1, 10] or wrong type."""
    return draw(
        one_of(
            # Below minimum (0 or negative)
            integers(max_value=0),
            # Above maximum
            integers(min_value=11),
            # Wrong type: float
            just(5.5),
            # Wrong type: string
            just("high"),
            # Wrong type: boolean
            booleans(),
        )
    )


# Map each field name to its invalid value strategy.
# Used by the composite strategy below to pick an invalid value for each chosen field.
INVALID_STRATEGIES = {
    "age": invalid_age(),
    "income": invalid_income(),
    "horizon": invalid_horizon(),
    "experience": invalid_experience(),
    "risk_score": invalid_risk_score(),
}

# Map each field name to a strategy that generates valid values for that field.
# Used to fill in the remaining fields that should NOT trigger errors.
VALID_STRATEGIES = {
    "age": integers(min_value=18, max_value=120),
    "income": floats(
        min_value=0.00, max_value=999_999_999.99, allow_nan=False, allow_infinity=False
    ),
    "horizon": integers(min_value=1, max_value=50),
    "experience": integers(min_value=0, max_value=50),
    "risk_score": integers(min_value=1, max_value=10),
}

ALL_FIELDS = list(INVESTOR_PROFILE_SCHEMA.keys())


@composite
def profile_with_n_invalid_fields(draw):
    """Strategy that generates an investor profile with exactly N invalid or missing fields.

    How N is chosen:
        N is drawn uniformly from [1, 5] representing 1 to all 5 fields being invalid.
        This tests the requirement that ALL invalid fields are reported simultaneously
        (Requirement 1.5: report all invalid fields, not just the first).

    How invalid fields are selected:
        We randomly sample N distinct field names from the 5 available fields.
        Each selected field is either assigned an invalid value OR omitted entirely
        (simulating a missing field). The remaining (5 - N) fields get valid values.

    Returns:
        Tuple of (profile_dict, set_of_invalid_field_names) for assertion use.
    """
    # Choose how many fields will be invalid: between 1 and 5
    n = draw(integers(min_value=1, max_value=5))

    # Randomly select which N fields will be invalid
    invalid_field_names = draw(
        lists(
            sampled_from(ALL_FIELDS),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )

    profile = {}

    for field in ALL_FIELDS:
        if field in invalid_field_names:
            # Decide whether to make the field invalid-valued or missing (omitted).
            # True = assign an out-of-range/wrong-type value, False = omit the field.
            use_invalid_value = draw(booleans())
            if use_invalid_value:
                # Assign a value that is out-of-range or wrong type
                profile[field] = draw(INVALID_STRATEGIES[field])
            # else: field is simply not present in the dict (missing field)
        else:
            # This field should be valid - draw a value within accepted range
            profile[field] = draw(VALID_STRATEGIES[field])

    return profile, set(invalid_field_names)


@given(data=profile_with_n_invalid_fields())
def test_invalid_profiles_report_all_errors_with_field_names(data):
    """Property 3: Invalid investor profiles report all errors with field names and ranges.

    **Validates: Requirements 1.4, 1.5**

    For any InvestorProfile with N fields (where N >= 1) that are missing or
    outside their valid ranges, validate_investor_profile() SHALL return exactly
    N error messages, each identifying the invalid field by name and stating the
    accepted range.

    Assertion logic explained:
    - We check len(errors) == N to confirm ALL invalid fields are reported
      (not just the first one - this validates Requirement 1.5).
    - We check each error message contains the field name to confirm the error
      identifies WHICH field is invalid (validates Requirement 1.4).
    - We check each error message contains the range boundaries to confirm the
      accepted range is communicated to the user (validates Requirement 1.4).
    """
    profile, invalid_fields = data
    n = len(invalid_fields)

    # Call the validation function under test
    errors = validate_investor_profile(profile)

    # Assert exactly N errors are returned - one per invalid/missing field.
    # This validates Requirement 1.5: a single response listing ALL invalid fields.
    assert len(errors) == n, (
        f"Expected {n} errors for {n} invalid fields {invalid_fields}, "
        f"but got {len(errors)} errors: {errors}\n"
        f"Profile: {profile}"
    )

    # Assert each error message names the corresponding invalid field.
    # This validates Requirement 1.4: error identifies each invalid field by name.
    for field_name in invalid_fields:
        # Find the error message that corresponds to this field
        matching_errors = [e for e in errors if field_name in e]
        assert len(matching_errors) == 1, (
            f"Expected exactly one error mentioning field '{field_name}', "
            f"found {len(matching_errors)} in errors: {errors}"
        )

        # Verify the error message includes range information.
        # The schema defines min and max for each field - the error should state them.
        schema = INVESTOR_PROFILE_SCHEMA[field_name]
        error_msg = matching_errors[0]
        assert str(schema["min"]) in error_msg, (
            f"Error for '{field_name}' should state minimum value "
            f"{schema['min']}: {error_msg}"
        )
        assert str(schema["max"]) in error_msg, (
            f"Error for '{field_name}' should state maximum value "
            f"{schema['max']}: {error_msg}"
        )
