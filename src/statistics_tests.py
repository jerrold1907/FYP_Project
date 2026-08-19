"""Statistical inference for model and strategy evaluation.

Every headline figure in this project was originally a bare point estimate: a
69.2% win rate over 13 trades, a weighted F1 of 0.41, a Sharpe ratio of 0.44.
None of those numbers say whether the result could have arisen by chance, and
with samples this small that question dominates the interpretation.

This module supplies the inference layer:

- Interval estimates for proportions (Wilson), so accuracy and win rate carry
  uncertainty bounds rather than implying false precision.
- Paired comparison of classifiers on the same test set (McNemar), which is the
  correct test when two models see identical data — an unpaired test would
  ignore that dependence and overstate significance.
- Group comparison across models or market regimes (one-way ANOVA, with the
  Kruskal-Wallis rank test as a distribution-free alternative because financial
  returns are heavy-tailed and fail ANOVA's normality assumption).
- Sharpe ratio standard errors following Lo (2002), and a bootstrap for
  statistics with no closed-form interval.
- Power analysis, used here to show honestly that n=13 trades cannot support
  strong claims.

References:
    Wilson, E.B. (1927) 'Probable inference, the law of succession, and
        statistical inference', JASA 22(158), pp. 209-212.
    McNemar, Q. (1947) 'Note on the sampling error of the difference between
        correlated proportions or percentages', Psychometrika 12(2), pp. 153-157.
    Lo, A.W. (2002) 'The statistics of Sharpe ratios', Financial Analysts
        Journal 58(4), pp. 36-52.
    Dietterich, T.G. (1998) 'Approximate statistical tests for comparing
        supervised classification learning algorithms', Neural Computation
        10(7), pp. 1895-1923.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats

#: Trading days per year, used to annualise return statistics.
TRADING_DAYS_PER_YEAR = 252


@dataclass
class Interval:
    """A point estimate with a confidence interval."""
    estimate: float
    lower: float
    upper: float
    confidence: float
    n: int

    @property
    def width(self) -> float:
        """Total width of the interval."""
        return self.upper - self.lower

    def excludes(self, value: float) -> bool:
        """True if `value` falls outside the interval.

        Used to judge whether a result is distinguishable from a reference
        level, e.g. whether a win rate interval excludes 0.5.
        """
        return value < self.lower or value > self.upper

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return (f"{self.estimate:.4f} "
                f"[{pct}% CI: {self.lower:.4f}, {self.upper:.4f}], n={self.n}")


@dataclass
class TestResult:
    """Outcome of a hypothesis test."""
    name: str
    statistic: float
    p_value: float
    detail: str = ""

    def significant(self, alpha: float = 0.05) -> bool:
        """True if the null hypothesis is rejected at level `alpha`."""
        return self.p_value < alpha

    def __str__(self) -> str:
        verdict = "significant" if self.significant() else "not significant"
        base = f"{self.name}: stat={self.statistic:.4f}, p={self.p_value:.4f} ({verdict})"
        return f"{base}\n  {self.detail}" if self.detail else base


def wilson_interval(successes: int, n: int,
                    confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) approximation because Wald behaves badly
    at small n or proportions near 0 or 1 — it can even produce bounds outside
    [0, 1]. Wilson stays inside the unit interval and holds its nominal
    coverage far better in exactly the small-sample regime this project is in.

    Args:
        successes: Number of successes observed.
        n: Number of trials.
        confidence: Coverage level, e.g. 0.95.

    Returns:
        Interval for the underlying success probability.

    Raises:
        ValueError: If n <= 0, successes is outside [0, n], or confidence is
            outside (0, 1).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n

    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(
        p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

    return Interval(
        estimate=p_hat,
        lower=max(0.0, centre - margin),
        upper=min(1.0, centre + margin),
        confidence=confidence,
        n=n,
    )


def proportion_test(successes: int, n: int, p_null: float = 0.5,
                    alternative: str = "two-sided") -> TestResult:
    """Exact binomial test of an observed proportion against a reference.

    Used to ask, for example, whether a 69.2% win rate over 13 trades is
    distinguishable from a coin flip. The exact test is used rather than a
    normal approximation because n is small.

    Args:
        successes: Number of successes.
        n: Number of trials.
        p_null: Reference proportion under the null hypothesis.
        alternative: 'two-sided', 'greater' or 'less'.

    Returns:
        TestResult carrying the observed proportion and p-value.
    """
    result = stats.binomtest(successes, n, p_null, alternative=alternative)
    return TestResult(
        name=f"Binomial test (H0: p = {p_null})",
        statistic=successes / n,
        p_value=float(result.pvalue),
        detail=(f"{successes}/{n} successes = {successes / n:.1%}; "
                f"alternative='{alternative}'"),
    )


def mcnemar_test(y_true: Sequence, pred_a: Sequence, pred_b: Sequence,
                 name_a: str = "A", name_b: str = "B",
                 exact_threshold: int = 25) -> TestResult:
    """McNemar's test comparing two classifiers on the same test set.

    When two models are evaluated on identical data their errors are
    correlated, so an unpaired test (such as a two-sample proportion test on
    the accuracies) violates independence and inflates significance. McNemar
    conditions on the discordant cases only — instances where exactly one model
    is correct — which is the appropriate paired comparison and is the test
    Dietterich (1998) recommends for a single held-out test set.

    The exact binomial form is used when the discordant count is small, where
    the chi-square approximation is unreliable; otherwise the continuity-
    corrected chi-square is used.

    Args:
        y_true: Ground-truth labels.
        pred_a: Predictions from the first model.
        pred_b: Predictions from the second model.
        name_a: Display name of the first model.
        name_b: Display name of the second model.
        exact_threshold: Use the exact test when discordant count is below this.

    Returns:
        TestResult. A significant result means the two models' error rates
        differ; the detail field reports which model won more discordant cases.

    Raises:
        ValueError: If the three sequences differ in length or are empty.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, "
            f"pred_a={len(pred_a)}, pred_b={len(pred_b)}")
    if len(y_true) == 0:
        raise ValueError("Cannot test on empty predictions")

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    # Discordant cells: only these carry information about the difference.
    only_a = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    only_b = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    discordant = only_a + only_b

    if discordant == 0:
        return TestResult(
            name=f"McNemar ({name_a} vs {name_b})",
            statistic=0.0,
            p_value=1.0,
            detail="Models agree on every instance; no discordant pairs.",
        )

    if discordant < exact_threshold:
        p_value = float(stats.binomtest(only_a, discordant, 0.5).pvalue)
        statistic = float(min(only_a, only_b))
        method = f"exact binomial (discordant n={discordant} < {exact_threshold})"
    else:
        statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "chi-square with continuity correction"

    leader = name_a if only_a > only_b else name_b if only_b > only_a else "neither"
    return TestResult(
        name=f"McNemar ({name_a} vs {name_b})",
        statistic=statistic,
        p_value=p_value,
        detail=(f"{name_a}-only correct: {only_a}, {name_b}-only correct: {only_b}; "
                f"{method}; favours {leader}"),
    )


def one_way_anova(groups: dict[str, Sequence[float]]) -> TestResult:
    """One-way ANOVA testing whether group means differ.

    Applied here to compare mean returns across models or across market
    regimes. ANOVA assumes normally distributed residuals and equal variances;
    financial returns typically satisfy neither, so `kruskal_wallis` should be
    reported alongside it and given precedence when the two disagree.

    Args:
        groups: Mapping of group name to that group's observations. At least
            two groups, each with at least two observations.

    Returns:
        TestResult with the F statistic and p-value.

    Raises:
        ValueError: If fewer than two groups, or any group has < 2 observations.
    """
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups, got {len(groups)}")
    for label, values in groups.items():
        if len(values) < 2:
            raise ValueError(
                f"Group '{label}' has {len(values)} observations; need >= 2")

    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    f_stat, p_value = stats.f_oneway(*arrays)
    means = ", ".join(f"{k}={np.mean(v):.4f}" for k, v in groups.items())

    return TestResult(
        name=f"One-way ANOVA across {len(groups)} groups",
        statistic=float(f_stat),
        p_value=float(p_value),
        detail=f"Group means: {means}",
    )


def kruskal_wallis(groups: dict[str, Sequence[float]]) -> TestResult:
    """Kruskal-Wallis H test: distribution-free alternative to ANOVA.

    Compares medians via ranks, so it tolerates the heavy tails and outliers
    characteristic of return series. Where ANOVA and this test disagree, this
    result is the more trustworthy for financial data.

    Args:
        groups: Mapping of group name to observations.

    Returns:
        TestResult with the H statistic and p-value.

    Raises:
        ValueError: If fewer than two groups, or any group has < 2 observations.
    """
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups, got {len(groups)}")
    for label, values in groups.items():
        if len(values) < 2:
            raise ValueError(
                f"Group '{label}' has {len(values)} observations; need >= 2")

    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    h_stat, p_value = stats.kruskal(*arrays)
    medians = ", ".join(f"{k}={np.median(v):.4f}" for k, v in groups.items())

    return TestResult(
        name=f"Kruskal-Wallis across {len(groups)} groups",
        statistic=float(h_stat),
        p_value=float(p_value),
        detail=f"Group medians: {medians}",
    )


def tukey_posthoc(groups: dict[str, Sequence[float]]) -> list[TestResult]:
    """Tukey HSD post-hoc test for all pairwise group comparisons.

    ANOVA only reports that *some* group differs. This identifies which pairs,
    while controlling the family-wise error rate across the comparisons — a
    correction that naive repeated t-tests would omit.

    Args:
        groups: Mapping of group name to observations.

    Returns:
        One TestResult per pair of groups.
    """
    labels = list(groups.keys())
    arrays = [np.asarray(groups[k], dtype=float) for k in labels]
    result = stats.tukey_hsd(*arrays)

    comparisons = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            comparisons.append(TestResult(
                name=f"Tukey HSD ({labels[i]} vs {labels[j]})",
                statistic=float(result.statistic[i, j]),
                p_value=float(result.pvalue[i, j]),
                detail=(f"mean difference = "
                        f"{np.mean(arrays[i]) - np.mean(arrays[j]):+.4f}"),
            ))
    return comparisons


def sharpe_ratio_interval(returns: Sequence[float],
                          periods_per_year: int = TRADING_DAYS_PER_YEAR,
                          confidence: float = 0.95) -> Interval:
    """Annualised Sharpe ratio with a confidence interval (Lo, 2002).

    A Sharpe ratio computed from a short return series is a noisy estimate, yet
    it is almost always reported as a bare number. Lo (2002) gives the
    asymptotic standard error under IID returns:

        SE(SR) = sqrt((1 + SR^2 / 2) / n)

    The interval is formed on the per-period Sharpe ratio and then annualised
    by sqrt(periods_per_year), which scales the point estimate and both bounds
    consistently.

    The IID assumption is a simplification: real returns are autocorrelated,
    which tends to make the true interval wider than reported here, so this
    should be read as a lower bound on the uncertainty.

    Args:
        returns: Per-period (not annualised) returns.
        periods_per_year: Periods per year for annualisation.
        confidence: Coverage level.

    Returns:
        Interval for the annualised Sharpe ratio.

    Raises:
        ValueError: If fewer than 3 returns, or the returns have zero variance.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        raise ValueError(f"Need at least 3 returns, got {len(values)}")

    std = values.std(ddof=1)
    if std == 0:
        raise ValueError("Returns have zero variance; Sharpe ratio is undefined")

    n = len(values)
    sharpe_period = values.mean() / std
    standard_error = np.sqrt((1 + 0.5 * sharpe_period**2) / n)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    scale = np.sqrt(periods_per_year)

    return Interval(
        estimate=sharpe_period * scale,
        lower=(sharpe_period - z * standard_error) * scale,
        upper=(sharpe_period + z * standard_error) * scale,
        confidence=confidence,
        n=n,
    )


def mean_return_test(returns: Sequence[float],
                     mu_null: float = 0.0) -> TestResult:
    """One-sample t-test on mean return against a reference level.

    Answers whether a strategy's average return is distinguishable from zero
    (or, by passing the benchmark's mean, from the benchmark).

    Args:
        returns: Per-period returns.
        mu_null: Mean under the null hypothesis.

    Returns:
        TestResult with the t statistic and two-sided p-value.

    Raises:
        ValueError: If fewer than 2 finite returns are supplied.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError(f"Need at least 2 returns, got {len(values)}")

    t_stat, p_value = stats.ttest_1samp(values, mu_null)
    return TestResult(
        name=f"One-sample t-test (H0: mean = {mu_null})",
        statistic=float(t_stat),
        p_value=float(p_value),
        detail=(f"observed mean = {values.mean():.6f}, "
                f"sd = {values.std(ddof=1):.6f}, n = {len(values)}"),
    )


def paired_return_test(strategy: Sequence[float],
                       benchmark: Sequence[float]) -> TestResult:
    """Paired t-test comparing strategy returns against a benchmark.

    The series are paired by period, so this removes the shared market movement
    common to both and tests the excess return directly. That is more sensitive
    than comparing the two means independently.

    Args:
        strategy: Strategy returns per period.
        benchmark: Benchmark returns over the same periods.

    Returns:
        TestResult on the mean difference.

    Raises:
        ValueError: If lengths differ or fewer than 2 pairs are supplied.
    """
    a = np.asarray(strategy, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise ValueError(f"Need at least 2 paired observations, got {len(a)}")

    t_stat, p_value = stats.ttest_rel(a, b)
    difference = a - b
    return TestResult(
        name="Paired t-test (strategy vs benchmark)",
        statistic=float(t_stat),
        p_value=float(p_value),
        detail=(f"mean excess return = {difference.mean():+.6f} per period, "
                f"n = {len(a)} pairs"),
    )


def bootstrap_interval(data: Sequence[float], statistic=np.mean,
                       n_resamples: int = 10_000, confidence: float = 0.95,
                       seed: int = 42) -> Interval:
    """Percentile bootstrap confidence interval for any statistic.

    Makes no distributional assumption, so it suits statistics with no
    closed-form standard error and small samples where asymptotic formulae are
    unreliable.

    Args:
        data: Observations to resample.
        statistic: Callable reducing a sample to a scalar.
        n_resamples: Number of bootstrap resamples.
        confidence: Coverage level.
        seed: Seed for reproducibility.

    Returns:
        Interval for the statistic.

    Raises:
        ValueError: If fewer than 2 finite observations are supplied.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError(f"Need at least 2 observations, got {len(values)}")

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    estimates = np.apply_along_axis(statistic, 1, draws)

    alpha = 1 - confidence
    return Interval(
        estimate=float(statistic(values)),
        lower=float(np.percentile(estimates, 100 * alpha / 2)),
        upper=float(np.percentile(estimates, 100 * (1 - alpha / 2))),
        confidence=confidence,
        n=len(values),
    )


def required_sample_size(p_null: float, p_alt: float, alpha: float = 0.05,
                         power: float = 0.80) -> int:
    """Trials needed to detect a proportion difference at given power.

    Included to make the project's statistical limits explicit. A backtest
    producing 13 trades cannot support a claim about win rate; this quantifies
    how many trades would be required.

    Uses the normal approximation for a two-sided one-sample proportion test.

    Args:
        p_null: Proportion under the null hypothesis.
        p_alt: Proportion to be detected.
        alpha: Significance level.
        power: Desired statistical power.

    Returns:
        Required number of trials, rounded up.

    Raises:
        ValueError: If proportions are outside (0, 1) or are equal.
    """
    for label, value in (("p_null", p_null), ("p_alt", p_alt)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{label} must be in (0, 1), got {value}")
    if p_null == p_alt:
        raise ValueError("p_null and p_alt must differ")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    effect = abs(p_alt - p_null)
    variance = p_alt * (1 - p_alt)

    return int(np.ceil((z_alpha + z_power) ** 2 * variance / effect ** 2))
