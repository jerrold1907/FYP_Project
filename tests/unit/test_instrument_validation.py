"""Unit tests for questionnaire reliability and suitability matrix validation."""

import numpy as np
import pytest

from src.instrument_validation import (
    RISK_ORDER,
    SIGNAL_ORDER,
    analyse_reliability,
    band_sensitivity,
    cronbach_alpha,
    simulate_responses,
    validate_suitability_matrix,
)
from src.suitability import SuitabilityEngine


class TestCronbachAlpha:
    """Alpha must behave correctly at the extremes of item agreement."""

    def test_identical_items_give_alpha_near_one(self):
        """Perfectly correlated items are maximally consistent."""
        column = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
        matrix = np.column_stack([column] * 4)
        assert cronbach_alpha(matrix) == pytest.approx(1.0, abs=1e-9)

    def test_independent_items_give_low_alpha(self):
        rng = np.random.default_rng(0)
        alpha = cronbach_alpha(rng.integers(1, 5, size=(300, 5)).astype(float))
        assert alpha < 0.3

    def test_alpha_increases_with_coherence(self):
        low = cronbach_alpha(simulate_responses(300, seed=1, coherence=0.2))
        high = cronbach_alpha(simulate_responses(300, seed=1, coherence=0.95))
        assert high > low

    def test_single_item_rejected(self):
        with pytest.raises(ValueError, match="at least 2 items"):
            cronbach_alpha(np.array([[1.0], [2.0], [3.0]]))

    def test_single_respondent_rejected(self):
        with pytest.raises(ValueError, match="at least 2 respondents"):
            cronbach_alpha(np.array([[1.0, 2.0, 3.0]]))

    def test_zero_total_variance_rejected(self):
        """Every respondent scoring the same total makes alpha undefined."""
        matrix = np.array([[1.0, 4.0], [4.0, 1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="undefined"):
            cronbach_alpha(matrix)

    def test_one_dimensional_input_rejected(self):
        with pytest.raises(ValueError, match="2-D matrix"):
            cronbach_alpha(np.array([1.0, 2.0, 3.0]))


class TestReliabilityReport:
    """The report must surface item-level diagnostics, not just alpha."""

    def test_reports_one_entry_per_item(self):
        report = analyse_reliability(simulate_responses(200, seed=2))
        assert report.n_items == 5
        assert len(report.items) == 5
        assert report.n_respondents == 200

    def test_uses_questionnaire_item_ids_by_default(self):
        report = analyse_reliability(simulate_responses(100, seed=3))
        assert "loss_reaction" in [i.item_id for i in report.items]

    def test_interpretation_labels_track_alpha(self):
        coherent = analyse_reliability(simulate_responses(400, seed=4,
                                                         coherence=0.95))
        incoherent = analyse_reliability(simulate_responses(400, seed=4,
                                                           coherence=0.05))
        assert coherent.alpha > incoherent.alpha
        assert coherent.interpretation in {"excellent", "good", "acceptable"}

    def test_acceptable_flag_matches_threshold(self):
        report = analyse_reliability(simulate_responses(300, seed=5,
                                                       coherence=0.95))
        assert report.acceptable == (report.alpha >= 0.70)

    def test_coherent_items_all_discriminate(self):
        report = analyse_reliability(simulate_responses(400, seed=6,
                                                       coherence=0.9))
        assert report.weak_items == []

    def test_random_items_fail_discrimination(self):
        rng = np.random.default_rng(7)
        report = analyse_reliability(rng.integers(1, 5, size=(300, 5)).astype(float))
        assert len(report.weak_items) > 0

    def test_constant_item_gets_zero_correlation(self):
        """A no-variance item cannot correlate; must not raise."""
        responses = simulate_responses(100, seed=8).astype(float)
        responses[:, 0] = 3.0
        report = analyse_reliability(responses)
        assert report.items[0].item_total_correlation == 0.0


class TestBandSensitivity:
    """Shifting the cut-offs shows how arbitrary the bands are."""

    def test_covers_every_attainable_score(self):
        for result in band_sensitivity():
            assert result.total == 16  # scores 5..20 inclusive

    def test_shifting_boundaries_reclassifies_some_scores(self):
        for result in band_sensitivity(shifts=(1, -1)):
            assert result.reclassified > 0

    def test_larger_shifts_change_at_least_as_many(self):
        results = {r.shift: r.reclassified for r in band_sensitivity((1, 2))}
        assert results[2] >= results[1]

    def test_proportion_is_a_fraction(self):
        for result in band_sensitivity():
            assert 0.0 <= result.proportion_changed <= 1.0


class TestSuitabilityMatrixValidation:
    """Structural properties the matrix must satisfy to be coherent."""

    def test_production_matrix_is_valid(self):
        result = validate_suitability_matrix()
        assert result.complete, f"missing pairs: {result.missing_pairs}"
        assert result.risk_monotonic, result.risk_violations
        assert result.signal_monotonic, result.signal_violations
        assert result.explanations_present, result.missing_explanations
        assert result.valid

    def test_covers_all_fifteen_documented_combinations(self):
        engine = SuitabilityEngine()
        for risk in RISK_ORDER:
            for signal in SIGNAL_ORDER:
                assert (risk, signal) in engine.MAPPING

    def test_detects_risk_monotonicity_violation(self):
        """Making Conservative more permissive than Aggressive must be caught."""
        engine = SuitabilityEngine()
        engine.MAPPING = dict(engine.MAPPING)
        engine.MAPPING[("Conservative", "Buy")] = "Suitable"
        engine.MAPPING[("Aggressive", "Buy")] = "Not Suitable"

        result = validate_suitability_matrix(engine)
        assert not result.risk_monotonic
        assert not result.valid
        assert any("Aggressive" in v for v in result.risk_violations)

    def test_detects_signal_monotonicity_violation(self):
        """Rating Avoid above Buy for the same investor must be caught."""
        engine = SuitabilityEngine()
        engine.MAPPING = dict(engine.MAPPING)
        engine.MAPPING[("Balanced", "Avoid")] = "Suitable"
        engine.MAPPING[("Balanced", "Buy")] = "Not Suitable"

        result = validate_suitability_matrix(engine)
        assert not result.signal_monotonic
        assert not result.valid

    def test_detects_missing_pair(self):
        engine = SuitabilityEngine()
        engine.MAPPING = dict(engine.MAPPING)
        del engine.MAPPING[("Growth", "Hold")]

        result = validate_suitability_matrix(engine)
        assert not result.complete
        assert ("Growth", "Hold") in result.missing_pairs

    def test_detects_missing_explanation(self):
        engine = SuitabilityEngine()
        engine.EXPLANATIONS = dict(engine.EXPLANATIONS)
        del engine.EXPLANATIONS[("Balanced", "Buy")]

        result = validate_suitability_matrix(engine)
        assert not result.explanations_present
        assert ("Balanced", "Buy") in result.missing_explanations


class TestResponseSimulation:
    """The simulator must produce valid, reproducible response matrices."""

    def test_shape_and_value_range(self):
        responses = simulate_responses(150, seed=9)
        assert responses.shape == (150, 5)
        assert responses.min() >= 1 and responses.max() <= 4

    def test_reproducible_with_same_seed(self):
        assert np.array_equal(simulate_responses(50, seed=10),
                              simulate_responses(50, seed=10))

    def test_different_seeds_differ(self):
        assert not np.array_equal(simulate_responses(50, seed=11),
                                  simulate_responses(50, seed=12))

    def test_too_few_respondents_rejected(self):
        with pytest.raises(ValueError, match="at least 2 respondents"):
            simulate_responses(1)

    @pytest.mark.parametrize("coherence", [-0.1, 1.1])
    def test_invalid_coherence_rejected(self, coherence):
        with pytest.raises(ValueError, match="coherence"):
            simulate_responses(50, coherence=coherence)
