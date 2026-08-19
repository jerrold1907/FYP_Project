"""Investor Risk Classification Module.

Classifies investors into risk categories (Conservative, Moderate, Aggressive)
based on their profile attributes using a deterministic rule-based algorithm.

Classification Algorithm:
    1. Determine base category from risk_score:
       - 1–3  → Conservative
       - 4–6  → Moderate
       - 7–10 → Aggressive
    2. Apply adjustment factors — each of the following shifts the category
       one level toward Conservative:
       - age > 60 (older investors tend toward lower risk tolerance)
       - horizon < 3 years (short horizon limits recovery from losses)
       - experience < 2 years (inexperienced investors need protection)
    3. Category levels are ordered: Aggressive → Moderate → Conservative.
       Conservative is the floor — no adjustment can go below it.
"""

from dataclasses import dataclass

from src.validation import validate_investor_profile


@dataclass
class InvestorProfile:
    """Represents an investor's profile for risk classification.

    Each field captures a dimension of the investor's financial situation
    and risk capacity. All fields are validated before classification.

    Attributes:
        age: Investor's age in years. Valid range: 18–120.
             Used as an adjustment factor (age > 60 shifts toward Conservative).
        income: Annual income in currency units. Valid range: 0.00–999,999,999.99.
                Captured for profile completeness; not used in classification logic.
        horizon: Investment time horizon in years. Valid range: 1–50.
                 Used as an adjustment factor (horizon < 3 shifts toward Conservative).
        experience: Years of investment experience. Valid range: 0–50.
                    Used as an adjustment factor (experience < 2 shifts toward Conservative).
        risk_score: Self-assessed risk tolerance on a 1–10 scale. Valid range: 1–10.
                    Primary determinant of base risk category.
    """

    age: int
    income: float
    horizon: int
    experience: int
    risk_score: int


class RiskClassifier:
    """Classifies investors into risk categories based on their profile.

    Uses a rule-based algorithm that starts with a base category derived from
    the investor's risk_score and then applies adjustments based on age,
    investment horizon, and experience level.

    The three risk categories in order from most aggressive to most conservative:
        Aggressive → Moderate → Conservative

    Conservative is the floor — adjustments can never push below it.
    """

    # Category levels ordered from most aggressive to most conservative.
    # Index 0 = Aggressive (highest risk), Index 2 = Conservative (lowest risk).
    # Moving toward Conservative means incrementing the index (floor at index 2).
    CATEGORY_LEVELS = ["Aggressive", "Moderate", "Conservative"]

    def validate(self, profile: InvestorProfile) -> list[str]:
        """Validates an investor profile by delegating to the validation module.

        Converts the InvestorProfile dataclass to a dictionary and passes it
        to validate_investor_profile() which checks each field for presence,
        correct type, and valid range.

        Args:
            profile: An InvestorProfile instance to validate.

        Returns:
            A list of error message strings, one per invalid field.
            Returns an empty list if all fields are valid.
        """
        # Convert dataclass to dict for the shared validation function
        profile_data = {
            "age": profile.age,
            "income": profile.income,
            "horizon": profile.horizon,
            "experience": profile.experience,
            "risk_score": profile.risk_score,
        }
        return validate_investor_profile(profile_data)

    def classify(self, profile: InvestorProfile) -> str:
        """Classifies an investor into a risk category.

        Algorithm Steps:
            1. Determine base category from risk_score:
               - risk_score 1–3  → Conservative (index 2)
               - risk_score 4–6  → Moderate (index 1)
               - risk_score 7–10 → Aggressive (index 0)
            2. Count adjustment factors that apply:
               - age > 60: shifts one level toward Conservative
               - horizon < 3: shifts one level toward Conservative
               - experience < 2: shifts one level toward Conservative
            3. Apply adjustments by moving index toward Conservative (higher index),
               capped at index 2 (Conservative floor).

        Args:
            profile: A valid InvestorProfile instance. Should be validated
                     before calling this method.

        Returns:
            One of 'Conservative', 'Moderate', or 'Aggressive'.
        """
        # Step 1: Determine base category from risk_score
        if profile.risk_score <= 3:
            # Low risk tolerance (1–3) → Conservative base
            base_index = 2  # Conservative
        elif profile.risk_score <= 6:
            # Medium risk tolerance (4–6) → Moderate base
            base_index = 1  # Moderate
        else:
            # High risk tolerance (7–10) → Aggressive base
            base_index = 0  # Aggressive

        # Step 2: Count adjustment factors that shift toward Conservative.
        # Each factor that applies adds 1 to the shift count.
        adjustments = 0

        # Age adjustment: investors over 60 have shorter time to recover from losses
        if profile.age > 60:
            adjustments += 1

        # Horizon adjustment: short investment horizons limit recovery potential
        if profile.horizon < 3:
            adjustments += 1

        # Experience adjustment: inexperienced investors need more protection
        if profile.experience < 2:
            adjustments += 1

        # Step 3: Apply adjustments by shifting index toward Conservative.
        # Higher index = more conservative. Floor at index 2 (Conservative).
        # Using min() ensures we never exceed the maximum index (Conservative floor).
        adjusted_index = min(base_index + adjustments, 2)

        return self.CATEGORY_LEVELS[adjusted_index]
