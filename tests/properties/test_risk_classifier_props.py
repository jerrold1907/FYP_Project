"""Property-based tests for the RiskClassifier module.

Tests Property 1: Risk classification follows documented rules.

This module verifies that for ANY valid InvestorProfile, the RiskClassifier.classify()
method returns a classification that exactly matches the documented algorithm:

    1. Base category from risk_score:
       - 1–3  → Conservative
       - 4–6  → Moderate
       - 7–10 → Aggressive
    2. Adjustment factors (each shifts one level toward Conservative):
       - age > 60
       - horizon < 3
       - experience < 2
    3. Category levels: Aggressive(0) → Moderate(1) → Conservative(2),
       floor at Conservative.

Why property-based testing here:
    The classification algorithm has a combinatorial input space (age × income ×
    horizon × experience × risk_score). There are boundary interactions between
    the base category and adjustment factors that example-based tests can miss —
    for instance, a Moderate base with exactly one adjustment factor active vs.
    two factors active. Hypothesis explores these combinations systematically,
    including shrunk counterexamples that pinpoint failures precisely.
"""

from hypothesis import given
from hypothesis.strategies import integers, floats, composite

from src.risk_classifier import InvestorProfile, RiskClassifier


# Feature: ai-stock-recommendation, Property 1: Risk classification follows documented rules


# =============================================================================
# Strategies
# =============================================================================


@composite
def valid_investor_profiles(draw):
    """Composite strategy generating valid InvestorProfile instances.

    Each field is constrained to its schema-defined valid range so that the
    profile passes validation and can be classified without error:
        - age: integer in [18, 120]
        - income: float in [0.00, 999_999_999.99], finite (no NaN/Inf)
        - horizon: integer in [1, 50]
        - experience: integer in [0, 50]
        - risk_score: integer in [1, 10]

    The composite pattern lets Hypothesis shrink each field independently,
    producing minimal counterexamples when a failure is detected.
    """
    # Age within valid investor range — covers both adjustment-active (>60) and inactive (<=60) regions
    age = draw(integers(min_value=18, max_value=120))

    # Income is required for a valid profile but does NOT affect classification.
    # We still generate it to ensure the full profile is valid.
    income = draw(
        floats(
            min_value=0.00,
            max_value=999_999_999.99,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    # Horizon in years — covers both adjustment-active (<3) and inactive (>=3) regions
    horizon = draw(integers(min_value=1, max_value=50))

    # Experience in years — covers both adjustment-active (<2) and inactive (>=2) regions
    experience = draw(integers(min_value=0, max_value=50))

    # Risk score determines the base category: 1-3, 4-6, 7-10
    risk_score = draw(integers(min_value=1, max_value=10))

    return InvestorProfile(
        age=age,
        income=income,
        horizon=horizon,
        experience=experience,
        risk_score=risk_score,
    )


# =============================================================================
# Oracle (independent reference implementation)
# =============================================================================


def expected_classification(profile: InvestorProfile) -> str:
    """Independently compute the expected risk classification.

    This is an oracle — a standalone reimplementation of the documented algorithm
    used to verify that the production code in RiskClassifier.classify() produces
    correct results. The oracle is intentionally written in a different style from
    the production code to avoid copy-paste equivalence.

    Algorithm (from design.md):
        Step 1: Determine base category index from risk_score.
                risk_score 1–3  → Conservative (index 2)
                risk_score 4–6  → Moderate     (index 1)
                risk_score 7–10 → Aggressive   (index 0)

        Step 2: Count how many adjustment factors apply.
                Each active factor shifts the category one level toward Conservative
                (i.e., increments the index by 1).
                - age > 60: older investors have less recovery time
                - horizon < 3: short horizons can't absorb volatility
                - experience < 2: novice investors need protection

        Step 3: Apply adjustments, capping at Conservative (index 2).

    Returns:
        One of 'Conservative', 'Moderate', or 'Aggressive'.
    """
    # The ordered category levels from most aggressive to most conservative
    categories = ["Aggressive", "Moderate", "Conservative"]

    # Step 1: Base category from risk_score
    # Low scores (1-3) indicate low risk tolerance → Conservative
    if profile.risk_score <= 3:
        base_index = 2  # Conservative
    # Medium scores (4-6) indicate moderate tolerance → Moderate
    elif profile.risk_score <= 6:
        base_index = 1  # Moderate
    # High scores (7-10) indicate high tolerance → Aggressive
    else:
        base_index = 0  # Aggressive

    # Step 2: Count adjustment factors that shift toward Conservative
    adjustments = 0

    # Age adjustment: investors over 60 have shorter recovery horizons
    if profile.age > 60:
        adjustments += 1

    # Horizon adjustment: less than 3 years is too short to recover from downturns
    if profile.horizon < 3:
        adjustments += 1

    # Experience adjustment: less than 2 years of experience means novice investor
    if profile.experience < 2:
        adjustments += 1

    # Step 3: Apply adjustments with floor at Conservative (index 2)
    # min() ensures we never exceed the maximum valid index
    final_index = min(base_index + adjustments, 2)

    return categories[final_index]


# =============================================================================
# Property Test
# =============================================================================


@given(profile=valid_investor_profiles())
def test_risk_classification_follows_documented_rules(profile):
    """Property 1: Risk classification follows documented rules.

    **Validates: Requirements 1.1, 1.3**

    For any valid InvestorProfile, the RiskClassifier SHALL return a classification
    that matches the documented algorithm:
        - Base category from risk_score (1–3 → Conservative, 4–6 → Moderate,
          7–10 → Aggressive)
        - Shifted one level toward Conservative for each of:
          age > 60, horizon < 3, or experience < 2
        - Conservative is the floor (cannot shift below it)

    This test generates random valid profiles via Hypothesis and independently
    computes the expected classification using a separate oracle function. The
    oracle reimplements the same algorithm documented in the design spec. If the
    production code and the oracle ever disagree, it indicates a deviation from
    the documented rules.

    The property covers:
        - All three base categories (Conservative, Moderate, Aggressive)
        - All combinations of 0, 1, 2, or 3 active adjustment factors
        - The Conservative floor behavior (adjustments cannot go below Conservative)
        - Boundary values of risk_score (3/4 and 6/7 transitions)
        - Boundary values of adjustment thresholds (age=60/61, horizon=2/3, experience=1/2)
    """
    classifier = RiskClassifier()

    # Get the actual classification from the production code
    actual = classifier.classify(profile)

    # Compute the expected classification independently using the oracle
    expected = expected_classification(profile)

    # The production code must agree with the independently computed result
    assert actual == expected, (
        f"Classification mismatch for profile:\n"
        f"  age={profile.age}, income={profile.income}, horizon={profile.horizon}, "
        f"experience={profile.experience}, risk_score={profile.risk_score}\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {actual}\n"
        f"  Base category: risk_score={profile.risk_score} → "
        f"{'Conservative' if profile.risk_score <= 3 else 'Moderate' if profile.risk_score <= 6 else 'Aggressive'}\n"
        f"  Active adjustments: age>60={profile.age > 60}, "
        f"horizon<3={profile.horizon < 3}, experience<2={profile.experience < 2}"
    )
