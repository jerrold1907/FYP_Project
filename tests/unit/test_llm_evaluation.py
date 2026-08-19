"""Unit tests for the LLM hallucination filter evaluation."""

import pytest

from src.llm_evaluation import (
    GuardCase,
    GuardPerformance,
    build_labelled_cases,
    evaluate_guard,
)

STOCKS = [
    {"ticker": "AAPL", "name": "Apple Inc.", "signal": "Avoid",
     "suitability": {"rating": "Not Suitable", "explanation": "..."}},
    {"ticker": "META", "name": "Meta Platforms Inc.", "signal": "Buy",
     "suitability": {"rating": "Suitable", "explanation": "..."}},
]


class TestGuardPerformanceMetrics:
    """Derived rates must follow from the confusion counts."""

    def test_perfect_filter_scores_one(self):
        p = GuardPerformance(true_positives=10, true_negatives=10)
        assert p.precision == 1.0
        assert p.recall == 1.0
        assert p.f1 == 1.0
        assert p.accuracy == 1.0
        assert p.false_accept_rate == 0.0
        assert p.false_reject_rate == 0.0

    def test_recall_reflects_missed_contradictions(self):
        p = GuardPerformance(true_positives=6, false_negatives=4)
        assert p.recall == pytest.approx(0.6)
        assert p.false_accept_rate == pytest.approx(0.4)

    def test_precision_reflects_needless_rejections(self):
        p = GuardPerformance(true_positives=6, false_positives=2)
        assert p.precision == pytest.approx(0.75)

    def test_false_reject_rate_uses_sound_summaries(self):
        p = GuardPerformance(true_negatives=8, false_positives=2)
        assert p.false_reject_rate == pytest.approx(0.2)

    def test_metrics_are_zero_when_no_data(self):
        p = GuardPerformance()
        assert p.total == 0
        assert p.precision == 0.0 and p.recall == 0.0
        assert p.f1 == 0.0 and p.accuracy == 0.0

    def test_total_sums_all_cells(self):
        p = GuardPerformance(true_positives=1, false_positives=2,
                             true_negatives=3, false_negatives=4)
        assert p.total == 10


class TestEvaluateGuard:
    """The evaluator must classify each case into the right cell."""

    def test_correct_claim_is_a_true_negative(self):
        cases = [GuardCase("META is rated Suitable for your profile.",
                           False, "correct")]
        p = evaluate_guard(cases, STOCKS)
        assert p.true_negatives == 1
        assert p.failures == []

    def test_swapped_rating_is_a_true_positive(self):
        cases = [GuardCase("META is rated Not Suitable for your profile.",
                           True, "swapped")]
        p = evaluate_guard(cases, STOCKS)
        assert p.true_positives == 1
        assert p.failures == []

    def test_missed_contradiction_is_recorded_as_false_accept(self):
        """A contradiction the filter cannot see must be reported, not hidden."""
        cases = [GuardCase("The outlook for this stock is broadly negative.",
                           True, "unattributed contradiction")]
        p = evaluate_guard(cases, STOCKS)
        assert p.false_negatives == 1
        assert any("FALSE ACCEPT" in f for f in p.failures)

    def test_multi_ticker_sentence_is_accepted(self):
        cases = [GuardCase("Both AAPL and META are volatile.", False,
                           "two tickers")]
        assert evaluate_guard(cases, STOCKS).true_negatives == 1

    def test_counts_sum_to_number_of_cases(self):
        cases = [
            GuardCase("META is rated Suitable.", False, "ok"),
            GuardCase("AAPL is rated Suitable.", True, "swapped"),
            GuardCase("Volatility is moderate.", False, "neutral"),
        ]
        assert evaluate_guard(cases, STOCKS).total == 3

    def test_empty_case_list_rejected(self):
        with pytest.raises(ValueError, match="at least one labelled case"):
            evaluate_guard([], STOCKS)


class TestBuildLabelledCases:
    """The generated evaluation set must be correct by construction."""

    def test_produces_both_positive_and_negative_cases(self):
        cases = build_labelled_cases(STOCKS)
        assert any(c.contradicts for c in cases)
        assert any(not c.contradicts for c in cases)

    def test_every_case_has_a_description(self):
        assert all(c.description for c in build_labelled_cases(STOCKS))

    def test_includes_the_observed_model_failure(self):
        cases = build_labelled_cases(STOCKS)
        assert any("llama3.2:1b failure" in c.description for c in cases)

    def test_filter_performs_well_on_generated_set(self):
        """The production filter should catch attributable contradictions."""
        cases = build_labelled_cases(STOCKS)
        p = evaluate_guard(cases, STOCKS)
        # No sound summary may be discarded.
        assert p.false_positives == 0, p.failures
        # Every contradiction naming one stock must be caught.
        assert p.recall == 1.0, p.failures

    def test_requires_at_least_two_stocks(self):
        with pytest.raises(ValueError, match="at least 2 stocks"):
            build_labelled_cases(STOCKS[:1])
