"""Evaluation of the LLM summary layer and its hallucination filter.

The chatbot generates its answers deterministically from model output, then
optionally passes them to a local language model that rewrites the key point as
a short paragraph. Because a small language model can misattribute facts —
during development llama3.2:1b stated that META was "not suitable" when that
rating belonged to AAPL — a filter (`verify_summary`) rejects any summary that
contradicts the structured data.

A filter is only worth having if its own error rate is known, so this module
measures it. Two questions are answered:

1. How reliable is the filter? Measured on a labelled set of summaries where
   ground truth is known by construction, giving precision, recall and the two
   error types:
     - False accept: a contradictory summary passed to the user. This is the
       costly error in a financial tool.
     - False reject: a correct summary discarded. Merely wastes a generation.

2. How often does the live model produce contradictions? Measured by generating
   summaries against real analysis data and recording the rejection rate.

References:
    Ji, Z. et al. (2023) 'Survey of hallucination in natural language
        generation', ACM Computing Surveys 55(12), pp. 1-38.
    Huang, L. et al. (2025) 'A survey on hallucination in large language
        models', ACM Transactions on Information Systems 43(2), pp. 1-55.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from src.chatbot import verify_summary


@dataclass
class GuardCase:
    """One labelled summary for evaluating the filter.

    Attributes:
        summary: Candidate summary text.
        contradicts: True if the text genuinely contradicts `stock_data`.
        description: What this case is designed to probe.
    """
    summary: str
    contradicts: bool
    description: str


@dataclass
class GuardPerformance:
    """Confusion counts and derived rates for the filter.

    The positive class is "contradictory summary", so a true positive means a
    contradiction was correctly caught and rejected.
    """
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (self.true_positives + self.false_positives
                + self.true_negatives + self.false_negatives)

    @property
    def precision(self) -> float:
        """Of the summaries rejected, the fraction that truly contradicted."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Of the truly contradictory summaries, the fraction caught.

        The critical metric: any contradiction missed here reaches the user.
        """
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        return ((self.true_positives + self.true_negatives) / self.total
                if self.total else 0.0)

    @property
    def false_accept_rate(self) -> float:
        """Fraction of contradictions that slipped through. Lower is better."""
        denominator = self.true_positives + self.false_negatives
        return self.false_negatives / denominator if denominator else 0.0

    @property
    def false_reject_rate(self) -> float:
        """Fraction of sound summaries needlessly discarded."""
        denominator = self.true_negatives + self.false_positives
        return self.false_positives / denominator if denominator else 0.0


def evaluate_guard(cases: Sequence[GuardCase],
                   stock_data: Sequence[dict]) -> GuardPerformance:
    """Score the filter against labelled cases.

    Args:
        cases: Labelled summaries, each marked as contradictory or not.
        stock_data: Authoritative analysis the summaries are checked against.

    Returns:
        GuardPerformance with confusion counts, derived rates, and a list of
        the specific cases that were misclassified.

    Raises:
        ValueError: If no cases are supplied.
    """
    if not cases:
        raise ValueError("Need at least one labelled case")

    performance = GuardPerformance()

    for case in cases:
        # verify_summary returns True to accept, so rejection is `not accepted`.
        accepted = verify_summary(case.summary, list(stock_data))
        rejected = not accepted

        if case.contradicts and rejected:
            performance.true_positives += 1
        elif case.contradicts and accepted:
            performance.false_negatives += 1
            performance.failures.append(
                f"FALSE ACCEPT ({case.description}): contradiction reached the "
                f"user -> {case.summary!r}")
        elif not case.contradicts and rejected:
            performance.false_positives += 1
            performance.failures.append(
                f"FALSE REJECT ({case.description}): sound summary discarded "
                f"-> {case.summary!r}")
        else:
            performance.true_negatives += 1

    return performance


def build_labelled_cases(stock_data: Sequence[dict]) -> list[GuardCase]:
    """Construct a labelled evaluation set from real analysis data.

    Cases are generated programmatically from the actual ratings and signals so
    the ground-truth label follows by construction rather than by judgement.
    Coverage spans:
      - correct single-stock claims (must be accepted)
      - swapped ratings and signals between stocks (must be rejected)
      - multi-stock and stock-free sentences (unattributable, so accepted)
      - neutral factual statements (accepted)

    Args:
        stock_data: At least two stock analysis dicts with differing ratings.

    Returns:
        Labelled cases for `evaluate_guard`.

    Raises:
        ValueError: If fewer than two stocks are supplied.
    """
    if len(stock_data) < 2:
        raise ValueError(
            f"Need at least 2 stocks to build swap cases, got {len(stock_data)}")

    first, second = stock_data[0], stock_data[1]
    t1, t2 = first["ticker"], second["ticker"]
    r1 = first["suitability"]["rating"]
    r2 = second["suitability"]["rating"]
    s1, s2 = first["signal"], second["signal"]

    cases = [
        # --- Correct attributions: must be accepted ---
        GuardCase(f"{t1} is rated {r1} for your profile.", False,
                  "correct rating"),
        GuardCase(f"The model gives {t1} a {s1} signal.", False,
                  "correct signal"),
        GuardCase(f"{t2} is rated {r2}, reflecting its {s2} signal.", False,
                  "correct rating and signal together"),
        GuardCase(f"Considering {t1}, the outlook is {s1} and the "
                  f"suitability is {r1}.", False,
                  "correct pair, verbose phrasing"),

        # --- Unattributable sentences: accepted, cannot be checked ---
        GuardCase(f"Both {t1} and {t2} show elevated volatility.", False,
                  "two tickers, no attribution possible"),
        GuardCase("Volatility across the universe is moderate.", False,
                  "no ticker named"),
        GuardCase(f"{t1} has an RSI of 41.3 and a negative MACD.", False,
                  "indicator statement, no rating claim"),
    ]

    # --- Swapped attributions: must be rejected ---
    if r1 != r2:
        cases.append(GuardCase(
            f"{t1} is rated {r2} for your profile.", True,
            "rating swapped between stocks"))
        cases.append(GuardCase(
            f"{t2} is rated {r1} for your profile.", True,
            "rating swapped, reverse direction"))
    if s1 != s2:
        cases.append(GuardCase(
            f"The model gives {t1} a {s2} signal.", True,
            "signal swapped between stocks"))
        cases.append(GuardCase(
            f"{t2} carries a {s1} signal this period.", True,
            "signal swapped, reverse direction"))

    # The specific failure observed during development.
    suitable = next((s for s in stock_data
                     if s["suitability"]["rating"] == "Suitable"), None)
    if suitable:
        cases.append(GuardCase(
            f"{suitable['ticker']} is considered not suitable for a balanced "
            "portfolio due to its high volatility.", True,
            "the observed llama3.2:1b failure"))

    return cases
