"""Unit tests for the statistical inference layer.

Where possible, results are checked against values that can be derived
independently rather than against the implementation's own output.
"""

import numpy as np
import pytest

from src.statistics_tests import (
    Interval,
    bootstrap_interval,
    kruskal_wallis,
    mcnemar_test,
    mean_return_test,
    one_way_anova,
    paired_return_test,
    proportion_test,
    required_sample_size,
    sharpe_ratio_interval,
    tukey_posthoc,
    wilson_interval,
)


class TestWilsonInterval:
    """Wilson bounds must stay in [0,1] and behave correctly at extremes."""

    def test_matches_published_value(self):
        """9/13 successes at 95% -> approximately [0.428, 0.868].

        Cross-checked against the standard Wilson formula; this is the
        project's actual backtest win rate (69.2% over 13 trades).
        """
        ci = wilson_interval(9, 13)
        assert ci.estimate == pytest.approx(0.6923, abs=1e-4)
        assert ci.lower == pytest.approx(0.428, abs=0.01)
        assert ci.upper == pytest.approx(0.868, abs=0.01)

    def test_win_rate_interval_includes_coin_flip(self):
        """The headline 69.2% win rate is not distinguishable from chance."""
        ci = wilson_interval(9, 13)
        assert not ci.excludes(0.5), "n=13 should not exclude p=0.5"

    def test_bounds_stay_within_unit_interval(self):
        for successes, n in [(0, 10), (10, 10), (1, 3), (0, 1), (1, 1)]:
            ci = wilson_interval(successes, n)
            assert 0.0 <= ci.lower <= ci.upper <= 1.0

    def test_interval_narrows_as_n_grows(self):
        narrow = wilson_interval(500, 1000)
        wide = wilson_interval(5, 10)
        assert narrow.width < wide.width

    def test_higher_confidence_widens_interval(self):
        assert wilson_interval(9, 13, 0.99).width > wilson_interval(9, 13, 0.90).width

    def test_excludes_detects_values_outside(self):
        ci = wilson_interval(95, 100)
        assert ci.excludes(0.5)
        assert not ci.excludes(0.95)

    @pytest.mark.parametrize("successes,n", [(-1, 10), (11, 10)])
    def test_invalid_success_count_rejected(self, successes, n):
        with pytest.raises(ValueError, match="successes"):
            wilson_interval(successes, n)

    def test_non_positive_n_rejected(self):
        with pytest.raises(ValueError, match="n must be positive"):
            wilson_interval(0, 0)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
    def test_invalid_confidence_rejected(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            wilson_interval(5, 10, confidence)


class TestProportionTest:
    """Exact binomial testing of observed proportions."""

    def test_backtest_win_rate_not_significant(self):
        """9/13 wins gives p > 0.05 against a fair coin."""
        result = proportion_test(9, 13, 0.5)
        assert not result.significant()
        assert result.statistic == pytest.approx(0.6923, abs=1e-4)

    def test_large_clear_effect_is_significant(self):
        assert proportion_test(90, 100, 0.5).significant()

    def test_exactly_null_gives_p_of_one(self):
        assert proportion_test(50, 100, 0.5).p_value == pytest.approx(1.0)

    def test_one_sided_is_more_powerful(self):
        two = proportion_test(9, 13, 0.5, "two-sided").p_value
        one = proportion_test(9, 13, 0.5, "greater").p_value
        assert one < two


class TestMcNemar:
    """Paired classifier comparison on a shared test set."""

    def test_identical_predictions_give_p_of_one(self):
        y = ["a", "b", "c", "a"]
        result = mcnemar_test(y, y, y)
        assert result.p_value == 1.0
        assert "agree on every instance" in result.detail

    def test_detects_clear_one_sided_difference(self):
        """A correct on 30 discordant cases, B on none -> significant."""
        y_true = ["a"] * 30 + ["b"] * 30
        pred_a = ["a"] * 30 + ["b"] * 30          # perfect
        pred_b = ["a"] * 30 + ["a"] * 30          # wrong on all 'b'
        result = mcnemar_test(y_true, pred_a, pred_b, "A", "B")
        assert result.significant()
        assert "favours A" in result.detail

    def test_uses_exact_test_for_small_discordant_count(self):
        y_true = ["a"] * 10
        pred_a = ["a"] * 10
        pred_b = ["a"] * 8 + ["b"] * 2  # only 2 discordant
        result = mcnemar_test(y_true, pred_a, pred_b)
        assert "exact binomial" in result.detail

    def test_uses_chi_square_for_large_discordant_count(self):
        y_true = ["a"] * 100
        pred_a = ["a"] * 100
        pred_b = ["a"] * 60 + ["b"] * 40  # 40 discordant
        result = mcnemar_test(y_true, pred_a, pred_b)
        assert "chi-square" in result.detail

    def test_symmetric_disagreement_not_significant(self):
        y_true = ["a"] * 20 + ["b"] * 20
        pred_a = ["b"] * 10 + ["a"] * 10 + ["b"] * 20
        pred_b = ["a"] * 20 + ["a"] * 10 + ["b"] * 10
        result = mcnemar_test(y_true, pred_a, pred_b)
        assert not result.significant()

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            mcnemar_test(["a", "b"], ["a"], ["a", "b"])

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            mcnemar_test([], [], [])


class TestGroupComparisons:
    """ANOVA, Kruskal-Wallis and post-hoc comparisons."""

    def test_anova_detects_separated_groups(self):
        groups = {
            "low": list(np.random.default_rng(0).normal(0, 1, 40)),
            "high": list(np.random.default_rng(1).normal(6, 1, 40)),
        }
        assert one_way_anova(groups).significant()

    def test_anova_accepts_identical_groups(self):
        rng = np.random.default_rng(3)
        groups = {f"g{i}": list(rng.normal(0, 1, 50)) for i in range(3)}
        assert not one_way_anova(groups).significant()

    def test_kruskal_agrees_on_clear_separation(self):
        groups = {
            "a": list(np.random.default_rng(4).normal(0, 1, 40)),
            "b": list(np.random.default_rng(5).normal(8, 1, 40)),
        }
        assert kruskal_wallis(groups).significant()

    def test_kruskal_robust_to_extreme_outlier(self):
        """A single huge outlier can sway ANOVA but not the rank test."""
        rng = np.random.default_rng(6)
        base = list(rng.normal(0, 1, 30))
        contaminated = list(rng.normal(0, 1, 30)) + [10_000.0]
        assert not kruskal_wallis({"a": base, "b": contaminated}).significant()

    def test_single_group_rejected(self):
        with pytest.raises(ValueError, match="at least 2 groups"):
            one_way_anova({"only": [1.0, 2.0]})

    def test_group_with_one_observation_rejected(self):
        with pytest.raises(ValueError, match="need >= 2"):
            one_way_anova({"a": [1.0, 2.0], "b": [3.0]})

    def test_tukey_returns_all_pairs(self):
        rng = np.random.default_rng(7)
        groups = {n: list(rng.normal(i * 4, 1, 25)) for i, n in enumerate("abc")}
        results = tukey_posthoc(groups)
        assert len(results) == 3  # 3 choose 2
        assert all(r.significant() for r in results)


class TestReturnStatistics:
    """Sharpe intervals and return tests."""

    def test_sharpe_point_estimate_is_annualised(self):
        rng = np.random.default_rng(8)
        daily = rng.normal(0.0005, 0.01, 500)
        ci = sharpe_ratio_interval(daily, periods_per_year=252)
        expected = (daily.mean() / daily.std(ddof=1)) * np.sqrt(252)
        assert ci.estimate == pytest.approx(expected, rel=1e-9)

    def test_sharpe_interval_brackets_estimate(self):
        rng = np.random.default_rng(9)
        ci = sharpe_ratio_interval(rng.normal(0.001, 0.02, 300))
        assert ci.lower < ci.estimate < ci.upper

    def test_sharpe_interval_narrows_with_more_data(self):
        rng = np.random.default_rng(10)
        short = sharpe_ratio_interval(rng.normal(0.001, 0.02, 60))
        long = sharpe_ratio_interval(rng.normal(0.001, 0.02, 2000))
        assert long.width < short.width

    def test_zero_variance_returns_rejected(self):
        with pytest.raises(ValueError, match="zero variance"):
            sharpe_ratio_interval([0.01] * 50)

    def test_too_few_returns_rejected(self):
        with pytest.raises(ValueError, match="at least 3 returns"):
            sharpe_ratio_interval([0.01, 0.02])

    def test_mean_return_test_detects_positive_drift(self):
        rng = np.random.default_rng(11)
        assert mean_return_test(rng.normal(0.05, 0.01, 100), 0.0).significant()

    def test_mean_return_test_accepts_zero_drift(self):
        rng = np.random.default_rng(12)
        assert not mean_return_test(rng.normal(0.0, 0.02, 100), 0.0).significant()

    def test_paired_test_removes_common_market_movement(self):
        """Strategy beats benchmark by a small edge plus idiosyncratic noise."""
        rng = np.random.default_rng(13)
        market = rng.normal(0.001, 0.02, 100)
        strategy = market + 0.002 + rng.normal(0, 0.0005, 100)
        assert paired_return_test(strategy, market).significant()

    def test_paired_test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            paired_return_test([0.1, 0.2], [0.1])


class TestBootstrap:
    """Distribution-free interval estimation."""

    def test_recovers_known_mean(self):
        rng = np.random.default_rng(14)
        ci = bootstrap_interval(rng.normal(5.0, 1.0, 500), n_resamples=2000)
        assert ci.lower < 5.0 < ci.upper

    def test_is_reproducible_with_fixed_seed(self):
        data = list(np.random.default_rng(15).normal(0, 1, 100))
        a = bootstrap_interval(data, n_resamples=1000, seed=99)
        b = bootstrap_interval(data, n_resamples=1000, seed=99)
        assert (a.lower, a.upper) == (b.lower, b.upper)

    def test_works_for_median(self):
        rng = np.random.default_rng(16)
        ci = bootstrap_interval(rng.normal(3.0, 1.0, 300),
                               statistic=np.median, n_resamples=2000)
        assert ci.lower < 3.0 < ci.upper

    def test_too_few_observations_rejected(self):
        with pytest.raises(ValueError, match="at least 2 observations"):
            bootstrap_interval([1.0])


class TestPowerAnalysis:
    """Sample-size requirements make the project's limits explicit."""

    def test_detecting_69_vs_50_percent_needs_far_more_than_13_trades(self):
        n = required_sample_size(0.50, 0.692)
        assert n > 13, "the observed effect is undetectable at n=13"
        assert 20 < n < 100

    def test_smaller_effects_require_larger_samples(self):
        assert required_sample_size(0.5, 0.55) > required_sample_size(0.5, 0.8)

    def test_higher_power_requires_larger_sample(self):
        assert (required_sample_size(0.5, 0.6, power=0.95)
                > required_sample_size(0.5, 0.6, power=0.80))

    def test_equal_proportions_rejected(self):
        with pytest.raises(ValueError, match="must differ"):
            required_sample_size(0.5, 0.5)

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.2])
    def test_out_of_range_proportion_rejected(self, p):
        with pytest.raises(ValueError, match="must be in"):
            required_sample_size(p, 0.6)


class TestIntervalHelpers:
    """Behaviour of the Interval container."""

    def test_width_and_string_format(self):
        interval = Interval(0.5, 0.4, 0.6, 0.95, 100)
        assert interval.width == pytest.approx(0.2)
        text = str(interval)
        assert "95% CI" in text and "n=100" in text


class TestChanceMatchedTest:
    """Win counts judged against each trial's own chance of success."""

    def test_equal_chances_reduce_to_the_binomial_normal_approximation(self):
        from src.statistics_tests import chance_matched_test
        result = chance_matched_test(60, [0.5] * 100)
        assert result.statistic == pytest.approx(2.0)      # (60 - 50) / 5
        assert result.p_value == pytest.approx(0.0455, abs=1e-4)

    def test_observing_the_expected_count_is_not_significant(self):
        from src.statistics_tests import chance_matched_test
        chances = [0.2, 0.4, 0.6, 0.8] * 25                # expects 50 wins
        result = chance_matched_test(50, chances)
        assert result.statistic == pytest.approx(0.0)
        assert not result.significant()

    def test_a_high_bar_makes_the_same_count_look_worse(self):
        from src.statistics_tests import chance_matched_test
        low = chance_matched_test(60, [0.45] * 100)
        high = chance_matched_test(60, [0.60] * 100)
        assert low.statistic > 0 and high.statistic == pytest.approx(0.0)

    def test_needs_uncertain_trials(self):
        from src.statistics_tests import chance_matched_test
        with pytest.raises(ValueError):
            chance_matched_test(0, [])
        with pytest.raises(ValueError):
            chance_matched_test(2, [0.0, 1.0, 1.0])
