"""Unit tests for Risk Classifier edge cases.

Tests boundary values of risk_score for base category determination,
the cumulative effect of multiple adjustment factors, and the Conservative
floor behavior that prevents classifications from going below Conservative.

These tests validate Requirements 1.1 and 1.3:
- 1.1: Risk_Classifier outputs exactly one Risk_Category
- 1.3: Classification logic with base categories and adjustments
"""

import pytest

from src.risk_classifier import InvestorProfile, RiskClassifier


@pytest.fixture
def classifier():
    """Provides a RiskClassifier instance for all tests."""
    return RiskClassifier()


class TestBoundaryValues:
    """Tests for risk_score boundary values that determine base category transitions.

    The risk_score ranges are:
      - 1–3  → Conservative (base)
      - 4–6  → Moderate (base)
      - 7–10 → Aggressive (base)

    These tests target the exact transition points between categories.
    """

    def test_risk_score_3_is_conservative_boundary(self, classifier):
        """Test that risk_score=3 classifies as Conservative (upper bound of Conservative range).

        risk_score=3 is the last value in the Conservative range (1–3).
        With no adjustment factors active (age=30, horizon=10, experience=5),
        the classification should remain at its base: Conservative.
        """
        # risk_score=3 is the upper boundary of the Conservative base range (1–3)
        # No adjustments: age=30 (≤60), horizon=10 (≥3), experience=5 (≥2)
        profile = InvestorProfile(
            age=30,
            income=50000.0,
            horizon=10,
            experience=5,
            risk_score=3,  # Upper boundary of Conservative range
        )
        result = classifier.classify(profile)
        assert result == "Conservative", (
            f"risk_score=3 should classify as Conservative (base range 1–3), got {result}"
        )

    def test_risk_score_4_is_moderate_boundary(self, classifier):
        """Test that risk_score=4 classifies as Moderate (lower bound of Moderate range).

        risk_score=4 is the first value in the Moderate range (4–6).
        This is the exact transition point from Conservative to Moderate.
        With no adjustment factors active, the classification should be Moderate.
        """
        # risk_score=4 is the lower boundary of the Moderate base range (4–6)
        # This is the exact transition from Conservative → Moderate
        # No adjustments: age=30 (≤60), horizon=10 (≥3), experience=5 (≥2)
        profile = InvestorProfile(
            age=30,
            income=50000.0,
            horizon=10,
            experience=5,
            risk_score=4,  # Lower boundary of Moderate range
        )
        result = classifier.classify(profile)
        assert result == "Moderate", (
            f"risk_score=4 should classify as Moderate (base range 4–6), got {result}"
        )

    def test_risk_score_7_is_aggressive_boundary(self, classifier):
        """Test that risk_score=7 classifies as Aggressive (lower bound of Aggressive range).

        risk_score=7 is the first value in the Aggressive range (7–10).
        This is the exact transition point from Moderate to Aggressive.
        With no adjustment factors active, the classification should be Aggressive.
        """
        # risk_score=7 is the lower boundary of the Aggressive base range (7–10)
        # This is the exact transition from Moderate → Aggressive
        # No adjustments: age=30 (≤60), horizon=10 (≥3), experience=5 (≥2)
        profile = InvestorProfile(
            age=30,
            income=50000.0,
            horizon=10,
            experience=5,
            risk_score=7,  # Lower boundary of Aggressive range
        )
        result = classifier.classify(profile)
        assert result == "Aggressive", (
            f"risk_score=7 should classify as Aggressive (base range 7–10), got {result}"
        )


class TestMaximumAdjustments:
    """Tests for cumulative adjustment factor effects on classification.

    Each adjustment factor shifts the category one level toward Conservative:
      - age > 60
      - horizon < 3
      - experience < 2

    When all three are active simultaneously on an Aggressive base,
    the result shifts 3 levels: Aggressive → Moderate → Conservative (floor).
    The third adjustment has no further effect due to the Conservative floor.
    """

    def test_aggressive_with_all_adjustments_becomes_conservative(self, classifier):
        """Test that an Aggressive profile with all three adjustments becomes Conservative.

        Starting from Aggressive (risk_score=10, base index 0):
          - age=65 (>60): shifts +1 → index 1 (Moderate)
          - horizon=1 (<3): shifts +1 → index 2 (Conservative)
          - experience=0 (<2): shifts +1 → capped at index 2 (Conservative floor)

        The cumulative effect of 3 adjustments on an Aggressive base results in
        Conservative, demonstrating the maximum possible downward shift.
        """
        # All three adjustment factors active simultaneously:
        # age=65 > 60, horizon=1 < 3, experience=0 < 2
        # Base: Aggressive (index 0) + 3 adjustments → min(0+3, 2) = index 2 = Conservative
        profile = InvestorProfile(
            age=65,           # > 60: triggers age adjustment
            income=100000.0,
            horizon=1,        # < 3: triggers horizon adjustment
            experience=0,     # < 2: triggers experience adjustment
            risk_score=10,    # Aggressive base (highest possible)
        )
        result = classifier.classify(profile)
        assert result == "Conservative", (
            f"Aggressive base with all 3 adjustments should become Conservative, got {result}"
        )

    def test_aggressive_with_two_adjustments_becomes_conservative(self, classifier):
        """Test that an Aggressive profile with two adjustments becomes Conservative.

        Starting from Aggressive (risk_score=8, base index 0):
          - age=70 (>60): shifts +1 → index 1 (Moderate)
          - horizon=2 (<3): shifts +1 → index 2 (Conservative)

        Two adjustments are sufficient to shift Aggressive all the way to Conservative.
        """
        # Two adjustment factors: age and horizon
        # Base: Aggressive (index 0) + 2 adjustments → min(0+2, 2) = index 2 = Conservative
        profile = InvestorProfile(
            age=70,           # > 60: triggers age adjustment
            income=75000.0,
            horizon=2,        # < 3: triggers horizon adjustment
            experience=5,     # ≥ 2: no adjustment
            risk_score=8,     # Aggressive base
        )
        result = classifier.classify(profile)
        assert result == "Conservative", (
            f"Aggressive base with 2 adjustments should become Conservative, got {result}"
        )

    def test_moderate_with_all_adjustments_becomes_conservative(self, classifier):
        """Test that a Moderate profile with all three adjustments becomes Conservative.

        Starting from Moderate (risk_score=5, base index 1):
          - age=61 (>60): shifts +1 → index 2 (Conservative)
          - horizon=2 (<3): would shift +1 → capped at index 2 (Conservative floor)
          - experience=1 (<2): would shift +1 → capped at index 2 (Conservative floor)

        Only one adjustment is needed to reach Conservative from Moderate,
        but additional adjustments have no further effect due to the floor.
        """
        # All three factors active, but Moderate only needs 1 shift to reach Conservative
        # The extra 2 adjustments are absorbed by the Conservative floor
        # Base: Moderate (index 1) + 3 adjustments → min(1+3, 2) = index 2 = Conservative
        profile = InvestorProfile(
            age=61,           # > 60: triggers age adjustment
            income=60000.0,
            horizon=2,        # < 3: triggers horizon adjustment
            experience=1,     # < 2: triggers experience adjustment
            risk_score=5,     # Moderate base
        )
        result = classifier.classify(profile)
        assert result == "Conservative", (
            f"Moderate base with 3 adjustments should become Conservative, got {result}"
        )


class TestConservativeFloor:
    """Tests for the Conservative floor behavior.

    Conservative is the lowest possible classification (floor).
    When a profile already has a Conservative base (risk_score 1–3),
    any active adjustment factors should have no effect — the classification
    must remain Conservative regardless of how many adjustments apply.

    This explicitly documents that the floor prevents any classification
    from going below Conservative, even with maximum adjustments.
    """

    def test_conservative_base_with_no_adjustments_stays_conservative(self, classifier):
        """Test that a Conservative base with no adjustments remains Conservative.

        Baseline behavior: risk_score=1 (lowest) with no adjustment factors
        should simply classify as Conservative with no further changes.
        """
        # Conservative base, no adjustments active
        # Base: Conservative (index 2) + 0 adjustments → index 2 = Conservative
        profile = InvestorProfile(
            age=25,           # ≤ 60: no adjustment
            income=40000.0,
            horizon=10,       # ≥ 3: no adjustment
            experience=5,     # ≥ 2: no adjustment
            risk_score=1,     # Conservative base (lowest score)
        )
        result = classifier.classify(profile)
        assert result == "Conservative", (
            f"Conservative base with no adjustments should stay Conservative, got {result}"
        )

    def test_conservative_base_with_all_adjustments_stays_conservative(self, classifier):
        """Test that Conservative floor holds even when all three adjustments are active.

        Starting from Conservative (risk_score=2, base index 2):
          - age=75 (>60): would shift +1 → capped at index 2 (Conservative floor)
          - horizon=1 (<3): would shift +1 → capped at index 2 (Conservative floor)
          - experience=0 (<2): would shift +1 → capped at index 2 (Conservative floor)

        The Conservative floor explicitly prevents the classification from going
        below Conservative. All adjustments are absorbed with no effect.
        min(2 + 3, 2) = 2 → Conservative. The floor holds.
        """
        # Conservative base with ALL adjustment factors active
        # This is the key test for the Conservative floor:
        # Base: Conservative (index 2) + 3 adjustments → min(2+3, 2) = index 2 = Conservative
        # The floor absorbs all adjustments without changing the result
        profile = InvestorProfile(
            age=75,           # > 60: triggers age adjustment (absorbed by floor)
            income=30000.0,
            horizon=1,        # < 3: triggers horizon adjustment (absorbed by floor)
            experience=0,     # < 2: triggers experience adjustment (absorbed by floor)
            risk_score=2,     # Conservative base
        )
        result = classifier.classify(profile)
        assert result == "Conservative", (
            f"Conservative floor must hold: Conservative base + 3 adjustments "
            f"should remain Conservative, got {result}"
        )

    def test_conservative_base_with_single_adjustment_stays_conservative(self, classifier):
        """Test that Conservative floor holds with a single active adjustment.

        Even one adjustment on a Conservative base should not change the result.
        This confirms the floor works for partial adjustment scenarios too.
        """
        # Conservative base with just one adjustment factor (age > 60)
        # Base: Conservative (index 2) + 1 adjustment → min(2+1, 2) = index 2 = Conservative
        profile = InvestorProfile(
            age=65,           # > 60: triggers age adjustment (absorbed by floor)
            income=45000.0,
            horizon=10,       # ≥ 3: no adjustment
            experience=10,    # ≥ 2: no adjustment
            risk_score=3,     # Conservative base (upper bound of range)
        )
        result = classifier.classify(profile)
        assert result == "Conservative", (
            f"Conservative floor must hold: Conservative base + 1 adjustment "
            f"should remain Conservative, got {result}"
        )
