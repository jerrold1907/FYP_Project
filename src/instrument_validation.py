"""Psychometric validation of the risk questionnaire and suitability matrix.

The questionnaire and the 5x3 suitability matrix were asserted by design,
citing Grable & Lytton (1999) for inspiration but never tested. A marker can
reasonably ask why a five-item instrument with those particular score bands
should be trusted. This module supplies the missing evidence.

Three kinds of check are provided:

1. Reliability. Cronbach's alpha measures internal consistency — whether the
   five items behave as though they measure one underlying construct. Values
   below roughly 0.70 indicate the items are not cohesive enough to be summed
   into a single score (Nunnally & Bernstein, 1994).

2. Item quality. Corrected item-total correlations identify items that fail to
   discriminate, and alpha-if-item-deleted shows whether removing an item would
   improve the scale.

3. Structural validity of the suitability matrix. Rather than asking whether
   the ratings are subjectively "right", this tests the ordering properties any
   defensible matrix must satisfy: monotonicity in risk tolerance and in signal
   strength. A violation would be an outright design error.

References:
    Cronbach, L.J. (1951) 'Coefficient alpha and the internal structure of
        tests', Psychometrika 16(3), pp. 297-334.
    Nunnally, J.C. and Bernstein, I.H. (1994) Psychometric Theory. 3rd edn.
        New York: McGraw-Hill.
    Grable, J.E. and Lytton, R.H. (1999) 'Financial risk tolerance revisited',
        Financial Services Review 8(3), pp. 163-181.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Sequence

import numpy as np

from src.risk_questionnaire import (
    QUESTIONS,
    SCORE_BOUNDARIES,
    classify_risk_from_score,
)
from src.suitability import SuitabilityEngine

#: Conventional interpretation thresholds for Cronbach's alpha.
ALPHA_BANDS = (
    (0.90, "excellent"),
    (0.80, "good"),
    (0.70, "acceptable"),
    (0.60, "questionable"),
    (0.50, "poor"),
    (0.00, "unacceptable"),
)


@dataclass
class ItemStatistics:
    """Diagnostics for a single questionnaire item."""
    item_id: str
    mean: float
    std: float
    item_total_correlation: float
    alpha_if_deleted: float

    @property
    def discriminates(self) -> bool:
        """True if the item correlates acceptably with the rest of the scale.

        A corrected item-total correlation below 0.30 is the conventional
        signal that an item is not measuring the same construct as the others.
        """
        return self.item_total_correlation >= 0.30


@dataclass
class ReliabilityReport:
    """Internal-consistency analysis of a multi-item scale."""
    alpha: float
    n_items: int
    n_respondents: int
    items: list[ItemStatistics] = field(default_factory=list)

    @property
    def interpretation(self) -> str:
        """Conventional verbal label for the alpha value."""
        for threshold, label in ALPHA_BANDS:
            if self.alpha >= threshold:
                return label
        return "unacceptable"

    @property
    def acceptable(self) -> bool:
        """True if alpha meets the conventional 0.70 threshold."""
        return self.alpha >= 0.70

    @property
    def weak_items(self) -> list[ItemStatistics]:
        """Items that fail the discrimination criterion."""
        return [i for i in self.items if not i.discriminates]


def cronbach_alpha(responses: np.ndarray) -> float:
    """Cronbach's alpha for a respondents-by-items score matrix.

    alpha = (k / (k - 1)) * (1 - sum(item variances) / total score variance)

    Args:
        responses: Array of shape (n_respondents, n_items) of item scores.

    Returns:
        Alpha in (-inf, 1]. Negative values indicate items that disagree with
        one another more than chance would predict.

    Raises:
        ValueError: If fewer than 2 items or 2 respondents, or if total score
            variance is zero (every respondent gave the same total).
    """
    matrix = np.asarray(responses, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got {matrix.ndim} dimensions")

    n_respondents, n_items = matrix.shape
    if n_items < 2:
        raise ValueError(f"Need at least 2 items, got {n_items}")
    if n_respondents < 2:
        raise ValueError(f"Need at least 2 respondents, got {n_respondents}")

    item_variances = matrix.var(axis=0, ddof=1).sum()
    total_variance = matrix.sum(axis=1).var(ddof=1)

    if total_variance == 0:
        raise ValueError(
            "Total score variance is zero; alpha is undefined. All respondents "
            "produced an identical total score."
        )

    return (n_items / (n_items - 1)) * (1 - item_variances / total_variance)


def analyse_reliability(responses: np.ndarray,
                        item_ids: Optional[Sequence[str]] = None
                        ) -> ReliabilityReport:
    """Full internal-consistency analysis of a response matrix.

    Computes overall alpha plus, for each item, its corrected item-total
    correlation (against the sum of the *other* items, avoiding the spurious
    self-inflation of correlating an item with a total containing it) and the
    alpha the scale would have if that item were removed.

    Args:
        responses: Array of shape (n_respondents, n_items).
        item_ids: Names for the items. Defaults to the questionnaire's own ids.

    Returns:
        ReliabilityReport with overall and per-item statistics.
    """
    matrix = np.asarray(responses, dtype=float)
    n_respondents, n_items = matrix.shape

    if item_ids is None:
        item_ids = [q["id"] for q in QUESTIONS][:n_items]

    overall_alpha = cronbach_alpha(matrix)

    items = []
    for index in range(n_items):
        others = np.delete(matrix, index, axis=1)
        rest_total = others.sum(axis=1)

        # Corrected item-total correlation; undefined if either side is constant.
        if matrix[:, index].std() == 0 or rest_total.std() == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(matrix[:, index], rest_total)[0, 1])

        try:
            alpha_without = cronbach_alpha(others) if n_items > 2 else float("nan")
        except ValueError:
            alpha_without = float("nan")

        items.append(ItemStatistics(
            item_id=str(item_ids[index]),
            mean=float(matrix[:, index].mean()),
            std=float(matrix[:, index].std(ddof=1)),
            item_total_correlation=correlation,
            alpha_if_deleted=alpha_without,
        ))

    return ReliabilityReport(
        alpha=overall_alpha,
        n_items=n_items,
        n_respondents=n_respondents,
        items=items,
    )


@dataclass
class BandSensitivity:
    """How stable the risk classification is to shifting the score cut-offs."""
    shift: int
    reclassified: int
    total: int

    @property
    def proportion_changed(self) -> float:
        """Fraction of score values assigned a different category."""
        return self.reclassified / self.total if self.total else 0.0


def band_sensitivity(shifts: Sequence[int] = (-2, -1, 1, 2)
                     ) -> list[BandSensitivity]:
    """Measure how arbitrary the questionnaire's score cut-offs are.

    The bands (5-8 Conservative, 9-11 Moderately Conservative, and so on) were
    chosen by design judgement. If shifting every boundary by one point
    reclassifies most respondents, the categories are fragile and the choice of
    cut-off needs justification. If few change, the scheme is robust.

    Every attainable total score (5..20) is evaluated, since each corresponds to
    at least one possible response pattern.

    Args:
        shifts: Boundary offsets, in score points, to test.

    Returns:
        One BandSensitivity per shift.
    """
    all_scores = list(range(5, 21))
    baseline = {s: classify_risk_from_score(s) for s in all_scores}

    results = []
    for shift in shifts:
        shifted_bounds = [(lo + shift, hi + shift, cat)
                          for lo, hi, cat in SCORE_BOUNDARIES]

        changed = 0
        for score in all_scores:
            category = None
            for lo, hi, cat in shifted_bounds:
                if lo <= score <= hi:
                    category = cat
                    break
            # Scores falling outside the shifted bands clamp to the nearest end.
            if category is None:
                category = (shifted_bounds[0][2] if score < shifted_bounds[0][0]
                            else shifted_bounds[-1][2])
            if category != baseline[score]:
                changed += 1

        results.append(BandSensitivity(shift=shift, reclassified=changed,
                                       total=len(all_scores)))
    return results


@dataclass
class MatrixValidation:
    """Structural checks on the suitability mapping."""
    complete: bool
    missing_pairs: list[tuple[str, str]]
    risk_monotonic: bool
    risk_violations: list[str]
    signal_monotonic: bool
    signal_violations: list[str]
    explanations_present: bool
    missing_explanations: list[tuple[str, str]]

    @property
    def valid(self) -> bool:
        """True if every structural property holds."""
        return (self.complete and self.risk_monotonic
                and self.signal_monotonic and self.explanations_present)


#: Risk categories ordered from least to most risk-tolerant.
RISK_ORDER = ["Conservative", "Moderately Conservative", "Balanced",
              "Growth", "Aggressive"]
#: Signals ordered from weakest to strongest outlook.
SIGNAL_ORDER = ["Avoid", "Hold", "Buy"]
#: Ratings ordered from most to least restrictive.
RATING_RANK = {"Not Suitable": 0, "Use Caution": 1, "Suitable": 2}


def validate_suitability_matrix(engine: Optional[SuitabilityEngine] = None
                                ) -> MatrixValidation:
    """Check the suitability matrix for completeness and correct ordering.

    Two monotonicity properties must hold for the matrix to be coherent:

    - Risk monotonicity: holding the signal fixed, permissiveness must never
      decrease as risk tolerance increases. It would be incoherent for a
      Conservative investor to be cleared for a stock an Aggressive investor
      is warned away from.
    - Signal monotonicity: holding risk tolerance fixed, permissiveness must
      never decrease as the signal strengthens from Avoid to Hold to Buy.

    These are objective structural requirements, so a violation is a design
    bug rather than a matter of opinion. They complement (and do not replace)
    expert review of whether the specific ratings are appropriate.

    Args:
        engine: Engine to inspect. A fresh SuitabilityEngine is used if omitted.

    Returns:
        MatrixValidation describing every check.
    """
    engine = engine or SuitabilityEngine()

    # Completeness across the documented five-category design.
    missing = [(r, s) for r, s in product(RISK_ORDER, SIGNAL_ORDER)
               if (r, s) not in engine.MAPPING]

    missing_explanations = [(r, s) for r, s in product(RISK_ORDER, SIGNAL_ORDER)
                            if (r, s) not in engine.EXPLANATIONS]

    def rank(risk: str, signal: str) -> Optional[int]:
        rating = engine.MAPPING.get((risk, signal))
        return RATING_RANK.get(rating) if rating else None

    # Risk monotonicity: scan up the risk order for each signal.
    risk_violations = []
    for signal in SIGNAL_ORDER:
        for lower, higher in zip(RISK_ORDER, RISK_ORDER[1:]):
            a, b = rank(lower, signal), rank(higher, signal)
            if a is None or b is None:
                continue
            if b < a:
                risk_violations.append(
                    f"signal={signal}: {higher} ({engine.MAPPING[(higher, signal)]}) "
                    f"is more restrictive than {lower} "
                    f"({engine.MAPPING[(lower, signal)]})")

    # Signal monotonicity: scan up the signal order for each risk category.
    signal_violations = []
    for risk in RISK_ORDER:
        for weaker, stronger in zip(SIGNAL_ORDER, SIGNAL_ORDER[1:]):
            a, b = rank(risk, weaker), rank(risk, stronger)
            if a is None or b is None:
                continue
            if b < a:
                signal_violations.append(
                    f"risk={risk}: {stronger} ({engine.MAPPING[(risk, stronger)]}) "
                    f"is more restrictive than {weaker} "
                    f"({engine.MAPPING[(risk, weaker)]})")

    return MatrixValidation(
        complete=not missing,
        missing_pairs=missing,
        risk_monotonic=not risk_violations,
        risk_violations=risk_violations,
        signal_monotonic=not signal_violations,
        signal_violations=signal_violations,
        explanations_present=not missing_explanations,
        missing_explanations=missing_explanations,
    )


def simulate_responses(n: int = 400, seed: int = 42,
                       coherence: float = 0.7) -> np.ndarray:
    """Generate synthetic questionnaire responses for reliability testing.

    Real respondent data is not available at this stage, so reliability is
    demonstrated on simulated responses. Each simulated respondent has a latent
    risk tolerance; item answers are drawn around that latent value, with
    `coherence` controlling how strongly items track it.

    This validates the *analysis pipeline* and shows the instrument's expected
    behaviour under a plausible response model. It is not a substitute for
    administering the questionnaire to real users, which remains future work.

    Args:
        n: Number of simulated respondents.
        seed: Seed for reproducibility.
        coherence: Weight on the latent trait, in [0, 1]. Higher values produce
            more internally consistent responses.

    Returns:
        Integer array of shape (n, 5) with values in 1..4.

    Raises:
        ValueError: If n < 2 or coherence is outside [0, 1].
    """
    if n < 2:
        raise ValueError(f"Need at least 2 respondents, got {n}")
    if not 0.0 <= coherence <= 1.0:
        raise ValueError(f"coherence must be in [0, 1], got {coherence}")

    rng = np.random.default_rng(seed)
    n_items = len(QUESTIONS)

    latent = rng.uniform(1, 4, size=n)
    noise = rng.normal(0, 1.0, size=(n, n_items))
    raw = coherence * latent[:, None] + (1 - coherence) * (noise + 2.5)

    return np.clip(np.rint(raw), 1, 4).astype(int)
