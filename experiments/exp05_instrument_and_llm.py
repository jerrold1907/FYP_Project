"""
Experiment 5: validating the human-facing components.

Addresses two gaps the markers identified:

  - "You should validate the suitability rules and questionnaire." Both were
    previously asserted by design with no supporting evidence.
  - "LLM output should be well-evaluated, and the hallucination problem has to
    be mitigated with filtering." A filter exists but its own error rate was
    never measured.

Four parts:
  1. Questionnaire reliability (Cronbach's alpha and item diagnostics).
  2. Sensitivity of the score bands to shifting the cut-offs.
  3. Structural validation of the suitability matrix (completeness and
     monotonicity in both risk tolerance and signal strength).
  4. Measured performance of the LLM hallucination filter.

Run: python experiments/exp05_instrument_and_llm.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.instrument_validation import (
    RISK_ORDER,
    SIGNAL_ORDER,
    analyse_reliability,
    band_sensitivity,
    simulate_responses,
    validate_suitability_matrix,
)
from src.llm_evaluation import build_labelled_cases, evaluate_guard
from src.risk_questionnaire import QUESTIONS, SCORE_BOUNDARIES
from src.statistics_tests import wilson_interval
from src.suitability import SuitabilityEngine

#: Representative analysis data for the LLM filter evaluation, mirroring the
#: structure the live application produces.
SAMPLE_STOCKS = [
    {"ticker": "AAPL", "name": "Apple Inc.", "signal": "Avoid",
     "suitability": {"rating": "Not Suitable", "explanation": "..."}},
    {"ticker": "META", "name": "Meta Platforms Inc.", "signal": "Buy",
     "suitability": {"rating": "Suitable", "explanation": "..."}},
    {"ticker": "KO", "name": "The Coca-Cola Co.", "signal": "Hold",
     "suitability": {"rating": "Use Caution", "explanation": "..."}},
]


def header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def section(text: str) -> None:
    print("\n" + "-" * 72)
    print(text)
    print("-" * 72)


def part1_reliability() -> pd.DataFrame:
    """Measure internal consistency of the five questionnaire items."""
    header("PART 1: QUESTIONNAIRE RELIABILITY")
    print("Cronbach's alpha measures whether the five items behave as though")
    print("they measure one underlying construct. Convention treats 0.70 as the")
    print("minimum for summing items into a single score.\n")
    print("IMPORTANT: no real respondent data exists yet, so responses are")
    print("simulated from a latent-trait model. This validates the analysis")
    print("pipeline and shows expected behaviour; administering the instrument")
    print("to real users remains outstanding work.\n")

    rows = []
    for coherence in (0.5, 0.7, 0.9):
        responses = simulate_responses(n=400, seed=42, coherence=coherence)
        report = analyse_reliability(responses)
        rows.append({
            "Coherence": coherence,
            "Alpha": round(report.alpha, 4),
            "Interpretation": report.interpretation,
            "Acceptable": report.acceptable,
            "Weak items": len(report.weak_items),
        })
        print(f"  latent coherence {coherence:.1f} -> alpha = {report.alpha:.4f} "
              f"({report.interpretation})")

    section("Item-level diagnostics at coherence 0.7")
    report = analyse_reliability(simulate_responses(n=400, seed=42,
                                                   coherence=0.7))
    print(f"{'Item':26}{'Mean':>7}{'SD':>7}{'r(item,rest)':>14}"
          f"{'alpha w/o':>11}{'Discriminates':>15}")
    for item in report.items:
        print(f"{item.item_id:26}{item.mean:>7.2f}{item.std:>7.2f}"
              f"{item.item_total_correlation:>14.3f}"
              f"{item.alpha_if_deleted:>11.3f}"
              f"{'yes' if item.discriminates else 'NO':>15}")

    print("\n  r(item,rest) is the corrected item-total correlation: each item")
    print("  against the sum of the *other* four, avoiding the self-inflation")
    print("  of correlating an item with a total that contains it. Values below")
    print("  0.30 indicate an item that fails to discriminate.")

    return pd.DataFrame(rows)


def part2_band_sensitivity() -> pd.DataFrame:
    """Show how sensitive the risk categories are to the chosen cut-offs."""
    header("PART 2: SENSITIVITY OF THE SCORE BANDS")
    print("The bands were chosen by design judgement:")
    for low, high, category in SCORE_BOUNDARIES:
        print(f"    {low:2d}-{high:2d}  {category}")
    print("\nIf shifting every boundary by one point reclassified most scores,")
    print("the categories would be fragile. This quantifies that.\n")

    results = band_sensitivity(shifts=(-2, -1, 1, 2))
    rows = []
    for result in results:
        print(f"  boundaries shifted by {result.shift:+d}: "
              f"{result.reclassified:2d} of {result.total} scores reclassified "
              f"({result.proportion_changed:.1%})")
        rows.append({
            "Shift": result.shift,
            "Reclassified": result.reclassified,
            "Total scores": result.total,
            "Proportion": round(result.proportion_changed, 4),
        })

    worst = max(r.proportion_changed for r in results)
    section("Interpretation")
    print(f"A one-point shift changes at most {worst:.0%} of classifications.")
    print("The bands are therefore consequential but not chaotic: a respondent")
    print("near a boundary may move category, which is inherent to any")
    print("banding scheme. The honest statement for the report is that the")
    print("cut-offs are a design choice requiring empirical calibration against")
    print("real respondent data, not that they are arbitrary.")

    return pd.DataFrame(rows)


def part3_matrix() -> None:
    """Validate structural properties of the suitability matrix."""
    header("PART 3: STRUCTURAL VALIDATION OF THE SUITABILITY MATRIX")
    print("Rather than asking whether the ratings feel correct, this tests the")
    print("ordering properties any coherent matrix must satisfy. A violation")
    print("would be an objective design error, not a difference of opinion.\n")

    engine = SuitabilityEngine()
    print("Current matrix (rows ordered by increasing risk tolerance):\n")
    print(f"{'Risk category':24}" + "".join(f"{s:>16}" for s in SIGNAL_ORDER))
    for risk in RISK_ORDER:
        cells = "".join(f"{engine.MAPPING.get((risk, s), '-'):>16}"
                        for s in SIGNAL_ORDER)
        print(f"{risk:24}{cells}")

    result = validate_suitability_matrix(engine)

    section("Checks")
    print(f"  Completeness ({len(RISK_ORDER)}x{len(SIGNAL_ORDER)} = "
          f"{len(RISK_ORDER) * len(SIGNAL_ORDER)} pairs): "
          f"{'PASS' if result.complete else 'FAIL'}")
    if result.missing_pairs:
        print(f"    missing: {result.missing_pairs}")

    print(f"  Risk monotonicity: {'PASS' if result.risk_monotonic else 'FAIL'}")
    print("    (for a fixed signal, permissiveness must not fall as risk")
    print("     tolerance rises)")
    for violation in result.risk_violations:
        print(f"    VIOLATION: {violation}")

    print(f"  Signal monotonicity: "
          f"{'PASS' if result.signal_monotonic else 'FAIL'}")
    print("    (for a fixed investor, permissiveness must not fall as the")
    print("     signal strengthens from Avoid to Hold to Buy)")
    for violation in result.signal_violations:
        print(f"    VIOLATION: {violation}")

    print(f"  Explanation coverage: "
          f"{'PASS' if result.explanations_present else 'FAIL'}")

    section("Interpretation")
    if result.valid:
        print("All structural properties hold. The matrix is internally")
        print("coherent and every combination carries an explanation.")
        print("\nWhat this does NOT establish: that the specific ratings match")
        print("professional financial judgement. That requires expert review,")
        print("which is planned and remains outstanding.")
    else:
        print("Structural violations found; the matrix requires correction.")


def part4_llm_guard() -> pd.DataFrame:
    """Measure the error rates of the hallucination filter."""
    header("PART 4: LLM HALLUCINATION FILTER PERFORMANCE")
    print("The chatbot builds answers deterministically from model output, then")
    print("optionally has a local LLM rewrite the key point. During development")
    print("llama3.2:1b stated that META was 'not suitable' when that rating")
    print("belonged to AAPL, so a filter rejects summaries contradicting the")
    print("structured data. A filter is only trustworthy if its own error rate")
    print("is known, which is what this measures.\n")

    cases = build_labelled_cases(SAMPLE_STOCKS)
    print(f"Labelled evaluation set: {len(cases)} cases, generated from real")
    print("analysis data so ground truth follows by construction.\n")

    contradictory = sum(1 for c in cases if c.contradicts)
    print(f"  contradictory (should be rejected): {contradictory}")
    print(f"  sound (should be accepted):         {len(cases) - contradictory}")

    performance = evaluate_guard(cases, SAMPLE_STOCKS)

    section("Confusion matrix (positive class = contradictory summary)")
    print(f"  True positives  (caught contradiction):     {performance.true_positives}")
    print(f"  False negatives (contradiction reached user): {performance.false_negatives}")
    print(f"  True negatives  (sound summary passed):      {performance.true_negatives}")
    print(f"  False positives (sound summary discarded):   {performance.false_positives}")

    section("Derived rates")
    recall_ci = wilson_interval(
        performance.true_positives,
        performance.true_positives + performance.false_negatives)
    print(f"  Recall    {performance.recall:.4f} "
          f"[{recall_ci.lower:.4f}, {recall_ci.upper:.4f}]  "
          "<- critical: fraction of contradictions caught")
    print(f"  Precision {performance.precision:.4f}")
    print(f"  F1        {performance.f1:.4f}")
    print(f"  Accuracy  {performance.accuracy:.4f}")
    print(f"  False accept rate {performance.false_accept_rate:.4f} "
          "(contradictions reaching the user)")
    print(f"  False reject rate {performance.false_reject_rate:.4f} "
          "(sound summaries discarded)")

    if performance.failures:
        section("Misclassified cases")
        for failure in performance.failures:
            print(f"  {failure}")

    section("Interpretation and known limitation")
    print("The filter checks sentences naming exactly one ticker. It cannot")
    print("adjudicate a sentence that names two stocks or none, because the")
    print("claim cannot be attributed to a specific stock. Such sentences are")
    print("accepted by design.")
    print("\nThat is a deliberate trade-off: rejecting every unattributable")
    print("sentence would discard most valid summaries. The residual risk is")
    print("that a contradiction phrased without naming a stock passes through.")
    print("Because the structured data is always displayed alongside the")
    print("summary, the user can still see the authoritative rating.")

    return pd.DataFrame([{
        "True positives": performance.true_positives,
        "False negatives": performance.false_negatives,
        "True negatives": performance.true_negatives,
        "False positives": performance.false_positives,
        "Recall": round(performance.recall, 4),
        "Precision": round(performance.precision, 4),
        "F1": round(performance.f1, 4),
        "Accuracy": round(performance.accuracy, 4),
        "False accept rate": round(performance.false_accept_rate, 4),
        "False reject rate": round(performance.false_reject_rate, 4),
    }])


def main() -> None:
    header("VALIDATION OF THE HUMAN-FACING COMPONENTS")
    print(f"Questionnaire items: {len(QUESTIONS)}")
    print(f"Risk categories: {len(RISK_ORDER)}")
    print(f"Suitability combinations: {len(RISK_ORDER) * len(SIGNAL_ORDER)}")

    reliability = part1_reliability()
    sensitivity = part2_band_sensitivity()
    part3_matrix()
    guard = part4_llm_guard()

    directory = os.path.dirname(os.path.abspath(__file__))
    reliability.to_csv(os.path.join(directory, "exp05_reliability.csv"),
                       index=False)
    sensitivity.to_csv(os.path.join(directory, "exp05_band_sensitivity.csv"),
                       index=False)
    guard.to_csv(os.path.join(directory, "exp05_llm_guard.csv"), index=False)

    header("COMPLETE")
    print("Written: exp05_reliability.csv, exp05_band_sensitivity.csv, "
          "exp05_llm_guard.csv")


if __name__ == "__main__":
    main()
