"""Input Validation Module.

Provides shared validation logic for investor profiles, risk categories,
and stock signals used across the system.
"""

# Schema defining valid ranges for each investor profile field
INVESTOR_PROFILE_SCHEMA = {
    "age": {"type": int, "min": 18, "max": 120, "required": True},
    "income": {"type": float, "min": 0.00, "max": 999_999_999.99, "required": True},
    "horizon": {"type": int, "min": 1, "max": 50, "required": True},
    "experience": {"type": int, "min": 0, "max": 50, "required": True},
    "risk_score": {"type": int, "min": 1, "max": 10, "required": True},
}

VALID_RISK_CATEGORIES = {
    "Conservative",
    "Moderately Conservative",
    "Moderate",  # Keep for backward compatibility
    "Balanced",
    "Growth",
    "Aggressive",
}
VALID_STOCK_SIGNALS = {"Buy", "Hold", "Avoid"}


def validate_investor_profile(data: dict) -> list[str]:
    """Validates all investor profile fields against the schema.

    Checks each field for presence, correct type, and valid range.
    Returns a list of all error messages (one per invalid/missing field),
    each identifying the field name and accepted range.

    Args:
        data: Dictionary containing investor profile fields.

    Returns:
        List of error message strings. Empty list if all fields are valid.
    """
    errors: list[str] = []

    for field_name, schema in INVESTOR_PROFILE_SCHEMA.items():
        expected_type = schema["type"]
        min_val = schema["min"]
        max_val = schema["max"]

        # Check if field is missing
        if field_name not in data or data[field_name] is None:
            errors.append(
                f"Missing field '{field_name}': "
                f"must be {expected_type.__name__} in range [{min_val}, {max_val}]"
            )
            continue

        value = data[field_name]

        # Type check: for int fields, reject floats; for float fields, accept int as well
        if expected_type is int:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(
                    f"Invalid field '{field_name}': "
                    f"must be {expected_type.__name__} in range [{min_val}, {max_val}], "
                    f"got {repr(value)}"
                )
                continue
        elif expected_type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    f"Invalid field '{field_name}': "
                    f"must be {expected_type.__name__} in range [{min_val}, {max_val}], "
                    f"got {repr(value)}"
                )
                continue
            # Convert int to float for range check
            value = float(value)

        # Range check
        if value < min_val or value > max_val:
            errors.append(
                f"Invalid field '{field_name}': "
                f"must be {expected_type.__name__} in range [{min_val}, {max_val}], "
                f"got {repr(value)}"
            )

    return errors


def validate_risk_category(value: str) -> str | None:
    """Validates a risk category value (case-sensitive).

    Args:
        value: The risk category string to validate.

    Returns:
        Error message string if invalid, None if valid.
    """
    if value not in VALID_RISK_CATEGORIES:
        return (
            f"Invalid Risk_Category: must be one of "
            f"{sorted(VALID_RISK_CATEGORIES)}, got '{value}'"
        )
    return None


def validate_stock_signal(value: str) -> str | None:
    """Validates a stock signal value (case-sensitive).

    Args:
        value: The stock signal string to validate.

    Returns:
        Error message string if invalid, None if valid.
    """
    if value not in VALID_STOCK_SIGNALS:
        return (
            f"Invalid Stock_Signal: must be one of "
            f"{sorted(VALID_STOCK_SIGNALS)}, got '{value}'"
        )
    return None
