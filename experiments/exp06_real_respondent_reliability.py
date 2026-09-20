"""
Experiment 6: reliability of the risk questionnaire on REAL respondents.

The reliability evidence in exp05 is computed on simulated responses. Simulated
data can only confirm that the scoring code works — it cannot tell us whether
real people answer these five items consistently, because the simulation
generates answers from a single latent trait by construction. Alpha is
guaranteed to be high there.

This script runs the identical Cronbach's alpha calculation on responses
collected from the usability-test participants, who completed the questionnaire
as Task 1 of their session. The sample is small, so the point estimate is
reported with a Feldt (1965) confidence interval, which is wide at this n and
is meant to be read that way.

HOW TO USE
----------
1. While each participant completes Task 1, record the option they chose for
   each of the five items as a score of 1-4 (the score is the option's position:
   first option = 1, fourth option = 4).
2. Add one row per participant to REAL_RESPONSES below, with the five scores in
   question order.
3. Run: python experiments/exp06_real_respondent_reliability.py

References:
    Cronbach, L.J. (1951) 'Coefficient alpha and the internal structure of
        tests', Psychometrika, 16(3), pp. 297-334.
    Feldt, L.S. (1965) 'The approximate sampling distribution of Kuder-
        Richardson reliability coefficient twenty', Psychometrika, 30(3),
        pp. 357-370.
    Nunnally, J.C. and Bernstein, I.H. (1994) Psychometric Theory. 3rd edn.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.instrument_validation import (
    analyse_reliability, cronbach_alpha, simulate_responses,
)
from src.risk_questionnaire import QUESTIONS, classify_risk_from_score


# ---------------------------------------------------------------------------
# PARTICIPANT RESPONSES — fill this in
# ---------------------------------------------------------------------------
# One tuple per participant. Five scores, each 1-4, in this column order:
#
#     (loss_reaction, investment_goal, time_horizon,
#      emotional_comfort, risk_return_preference)
#
# All five items already run in the same direction — 1 is the most risk-averse
# option and 4 the most risk-tolerant — so no reverse-scoring is needed.
#
# Example of the expected shape (delete these once real data is entered):
#
#     REAL_RESPONSES = [
#         (2, 1, 2, 2, 1),   # P1
#         (3, 3, 4, 3, 3),   # P2
#         (1, 2, 1, 1, 2),   # P3
#     ]

REAL_RESPONSES: list[tuple[int, ...]] = [
    # (loss, goal, horizon, comfort, tradeoff)
]


ITEM_IDS = [q["id"] for q in QUESTIONS]
N_ITEMS = len(ITEM_IDS)


def feldt_interval(alpha: float, n_respondents: int, n_items: int,
                   confidence: float = 0.95) -> tuple[float, float]:
    """Feldt's confidence interval for Cronbach's alpha.

    Alpha is a sample statistic, not a population value. At small n the
    sampling error is large, and reporting a bare point estimate implies a
    precision the data does not support. Feldt's method derives the interval
    from the F distribution:

        lower = 1 - (1 - alpha) * F(1 - c/2; n-1, (n-1)(k-1))
        upper = 1 - (1 - alpha) * F(c/2;     n-1, (n-1)(k-1))

    Args:
        alpha: The observed Cronbach's alpha.
        n_respondents: Number of respondents.
        n_items: Number of items in the scale.
        confidence: Coverage probability. Defaults to 0.95.

    Returns:
        (lower, upper) bounds. Returns (nan, nan) if scipy is unavailable or
        the degrees of freedom are too small to define the interval.
    """
    try:
        from scipy import stats
    except ImportError:
        return float("nan"), float("nan")

    df1 = n_respondents - 1
    df2 = (n_respondents - 1) * (n_items - 1)
    if df1 < 1 or df2 < 1:
        return float("nan"), float("nan")

    tail = (1 - confidence) / 2
    lower = 1 - (1 - alpha) * stats.f.ppf(1 - tail, df1, df2)
    upper = 1 - (1 - alpha) * stats.f.ppf(tail, df1, df2)
    return float(lower), float(upper)


def validate(responses: list[tuple[int, ...]]) -> np.ndarray:
    """Check the entered responses are well formed before any analysis.

    A silent shape or range error would produce a plausible-looking alpha from
    wrong data, which is worse than a crash.

    Args:
        responses: Rows of item scores as entered above.

    Returns:
        The responses as an (n_respondents, n_items) float array.

    Raises:
        SystemExit: With an explanatory message if the data cannot be analysed.
    """
    if not responses:
        raise SystemExit(
            "No responses entered.\n\n"
            "Open this file and fill in REAL_RESPONSES with one row per\n"
            "participant: five scores of 1-4 in question order.\n"
            f"Items, in order: {', '.join(ITEM_IDS)}"
        )

    for position, row in enumerate(responses, start=1):
        if len(row) != N_ITEMS:
            raise SystemExit(
                f"P{position} has {len(row)} scores, expected {N_ITEMS}. "
                f"Order: {', '.join(ITEM_IDS)}"
            )
        for item_id, score in zip(ITEM_IDS, row):
            if not isinstance(score, int) or not 1 <= score <= 4:
                raise SystemExit(
                    f"P{position}, item '{item_id}': score {score!r} is not an "
                    "integer in 1-4."
                )

    if len(responses) < 2:
        raise SystemExit(
            "Alpha needs at least 2 respondents — it is a measure of how "
            "scores covary across people."
        )

    matrix = np.asarray(responses, dtype=float)
    if matrix.sum(axis=1).var(ddof=1) == 0:
        raise SystemExit(
            "Every participant produced the same total score, so total "
            "variance is zero and alpha is undefined. This is a property of "
            "the sample, not an error — report it as such."
        )
    return matrix


def main() -> None:
    """Run the reliability analysis and print a report."""
    matrix = validate(REAL_RESPONSES)
    n = matrix.shape[0]

    print("=" * 72)
    print("QUESTIONNAIRE RELIABILITY — REAL RESPONDENTS")
    print("=" * 72)

    print(f"\nParticipants: {n}")
    print(f"Items:        {N_ITEMS}")

    print("\n--- Individual responses ---")
    print("      " + "".join(f"{i[:9]:>11}" for i in ITEM_IDS)
          + f"{'TOTAL':>8}  CATEGORY")
    for position, row in enumerate(REAL_RESPONSES, start=1):
        total = sum(row)
        scores = "".join(f"{s:>11}" for s in row)
        print(f"  P{position:<3}{scores}{total:>8}  {classify_risk_from_score(total)}")

    alpha = cronbach_alpha(matrix)
    report = analyse_reliability(matrix, item_ids=ITEM_IDS)
    low, high = feldt_interval(alpha, n, N_ITEMS)

    print("\n--- Cronbach's alpha ---")
    print("  alpha = (k / (k-1)) * (1 - sum(item variances) / total variance)")
    print(f"  k = {N_ITEMS}")
    print(f"  sum of item variances = {matrix.var(axis=0, ddof=1).sum():.4f}")
    print(f"  variance of totals    = {matrix.sum(axis=1).var(ddof=1):.4f}")
    print(f"\n  alpha = {alpha:.3f}   ({report.interpretation})")
    if not np.isnan(low):
        print(f"  95% CI [{low:.3f}, {high:.3f}]  (Feldt, 1965)")
    print(f"  Meets the conventional 0.70 threshold: "
          f"{'yes' if report.acceptable else 'no'}")

    print("\n--- Item statistics ---")
    print(f"  {'item':<24}{'mean':>7}{'sd':>7}{'r(item-rest)':>15}"
          f"{'alpha if dropped':>19}")
    for item in report.items:
        dropped = ("n/a" if np.isnan(item.alpha_if_deleted)
                   else f"{item.alpha_if_deleted:.3f}")
        print(f"  {item.item_id:<24}{item.mean:>7.2f}{item.std:>7.2f}"
              f"{item.item_total_correlation:>+15.3f}{dropped:>19}")

    weak = report.weak_items
    if weak:
        print("\n  Items with item-rest correlation below 0.30 (weak "
              "discrimination):")
        for item in weak:
            print(f"    - {item.item_id} (r = {item.item_total_correlation:+.3f})")
    else:
        print("\n  All items discriminate at r >= 0.30.")

    simulated = analyse_reliability(simulate_responses())
    print("\n--- Comparison with the simulated baseline ---")
    print(f"  Simulated (n={simulated.n_respondents}): alpha = {simulated.alpha:.3f}")
    print(f"  Real      (n={n}): alpha = {alpha:.3f}")
    print("\n  The simulated figure is an upper bound, not a prediction: the\n"
          "  simulation draws every item from one latent trait, so the items\n"
          "  are consistent by construction. The real figure is the one that\n"
          "  carries evidential weight.")

    print("\n--- Reporting note ---")
    print(f"  With n = {n} the interval above is wide, and alpha at this sample")
    print("  size is unstable. State the n alongside the coefficient and treat")
    print("  it as indicative rather than conclusive; a reliability estimate")
    print("  with narrow bounds would need roughly 100+ respondents.")
    print("=" * 72)


if __name__ == "__main__":
    main()
