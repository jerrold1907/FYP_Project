"""Evaluation utilities for time-series classification.

This module exists because the project's original evaluation used a random
stratified train/test split, which is invalid for financial time series and
produced a severely optimistic score (weighted F1 0.833 vs 0.356 under a
correct split — see experiments/exp01_leakage_diagnostic.py).

Two distinct leakage mechanisms are addressed:

1. Adjacent-row leakage. Features such as ma_20, ma_50 and volatility are
   rolling aggregates, so consecutive rows are near-duplicates. A random split
   places day t in training and day t+1 in testing, letting the model match
   almost-identical rows instead of forecasting.

2. Label-horizon overlap. Targets are 30-trading-day forward returns, so a
   training row dated d encodes prices up to d+30. Without a purge, training
   labels reveal outcomes that fall inside the test period.

`purged_chronological_split` removes both: it orders data by date, tests only
on the most recent block, and discards ("purges") the training rows whose label
window would reach into that block. The approach follows the purging principle
described by De Prado (2018) for financial machine learning.

All reported metrics are accompanied by naive baselines, because weighted F1 on
an imbalanced three-class problem is not interpretable in isolation.
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


#: Default label horizon in trading days, matching compute_target_labels().
DEFAULT_LABEL_HORIZON = 30


@dataclass
class SplitResult:
    """Row-index masks describing one train/test division.

    Attributes:
        train_idx: Positional indices of training rows.
        test_idx: Positional indices of test rows.
        split_date: First date belonging to the test block.
        purge_start: First date excluded from training by the purge.
        n_purged: Number of rows removed by the purge.
    """
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_date: pd.Timestamp
    purge_start: pd.Timestamp
    n_purged: int


def purged_chronological_split(
    dates: Sequence,
    test_size: float = 0.2,
    label_horizon: int = DEFAULT_LABEL_HORIZON,
) -> SplitResult:
    """Split time-indexed rows chronologically, purging label overlap.

    The test block is the most recent `test_size` fraction of trading dates.
    Training rows are then restricted to dates strictly earlier than
    `label_horizon` trading days before the split, so that no training label —
    which depends on `label_horizon` days of future prices — can encode
    information from the test period.

    Rows from multiple tickers may share a date; all rows on a given date are
    assigned to the same side of the split, so no ticker can leak across.

    Args:
        dates: Date for each row. Need not be sorted.
        test_size: Fraction of unique trading dates held out for testing.
            Must lie in (0, 1).
        label_horizon: Forward window, in trading days, used to build labels.
            Set to 0 to disable purging (chronological split only).

    Returns:
        SplitResult with positional train/test indices.

    Raises:
        ValueError: If `test_size` is outside (0, 1), or if the split leaves
            either side empty.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if label_horizon < 0:
        raise ValueError(f"label_horizon must be >= 0, got {label_horizon}")

    date_series = pd.Series(pd.to_datetime(pd.Series(list(dates)).values))
    unique_dates = np.sort(date_series.unique())

    if len(unique_dates) < 2:
        raise ValueError(
            f"Need at least 2 distinct dates to split, got {len(unique_dates)}"
        )

    # Index of the first test date, guaranteed to leave >=1 date on each side.
    split_pos = int(round(len(unique_dates) * (1.0 - test_size)))
    split_pos = max(1, min(split_pos, len(unique_dates) - 1))
    split_date = pd.Timestamp(unique_dates[split_pos])

    # Purge: training must end label_horizon trading days before the split.
    purge_pos = max(0, split_pos - label_horizon)
    purge_start = pd.Timestamp(unique_dates[purge_pos])

    test_mask = date_series >= split_date
    train_mask = date_series < purge_start

    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())

    if len(train_idx) == 0:
        raise ValueError(
            "Purge removed all training rows. Reduce label_horizon "
            f"({label_horizon}) or test_size ({test_size})."
        )
    if len(test_idx) == 0:
        raise ValueError("Split produced an empty test set.")

    n_purged = int(((date_series >= purge_start) & (date_series < split_date)).sum())

    return SplitResult(
        train_idx=train_idx,
        test_idx=test_idx,
        split_date=split_date,
        purge_start=purge_start,
        n_purged=n_purged,
    )


@dataclass
class ClassificationScores:
    """Metric bundle for one classifier on one test set."""
    name: str
    accuracy: float
    balanced_accuracy: float
    f1_weighted: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    n_test: int

    def as_row(self) -> dict:
        """Flatten to a dict suitable for a DataFrame row."""
        return {
            "Model": self.name,
            "Accuracy": round(self.accuracy, 4),
            "Balanced Acc": round(self.balanced_accuracy, 4),
            "F1 (weighted)": round(self.f1_weighted, 4),
            "F1 (macro)": round(self.f1_macro, 4),
            "Precision (w)": round(self.precision_weighted, 4),
            "Recall (w)": round(self.recall_weighted, 4),
            "n_test": self.n_test,
        }


def score_predictions(y_true, y_pred, name: str) -> ClassificationScores:
    """Compute the standard metric bundle for a set of predictions.

    Both weighted and macro F1 are reported. Weighted F1 is dominated by the
    majority class on imbalanced data, so macro F1 and balanced accuracy are
    included to expose per-class behaviour that the weighted figure can hide.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        name: Label for this model in results tables.

    Returns:
        ClassificationScores for the predictions.
    """
    return ClassificationScores(
        name=name,
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        precision_weighted=precision_score(
            y_true, y_pred, average="weighted", zero_division=0),
        recall_weighted=recall_score(
            y_true, y_pred, average="weighted", zero_division=0),
        n_test=len(y_true),
    )


def baseline_scores(X_train, y_train, X_test, y_test,
                    random_state: int = 42) -> list[ClassificationScores]:
    """Score naive baselines that any useful model must beat.

    Three references are produced:

    - Majority class: always predicts the most frequent training label. Sets
      the accuracy floor on imbalanced data.
    - Stratified random: samples predictions from the training class
      distribution. Approximates chance-level weighted F1.
    - Uniform random: samples uniformly across classes, the reference for the
      1/3 chance level often quoted for three-class problems.

    Args:
        X_train, y_train: Training data (features unused by the baselines but
            required by the scikit-learn estimator API).
        X_test, y_test: Test data.
        random_state: Seed for the stochastic baselines.

    Returns:
        List of ClassificationScores, one per baseline.
    """
    specs = [
        ("Baseline: majority class", DummyClassifier(strategy="most_frequent")),
        ("Baseline: stratified random",
         DummyClassifier(strategy="stratified", random_state=random_state)),
        ("Baseline: uniform random",
         DummyClassifier(strategy="uniform", random_state=random_state)),
    ]
    results = []
    for name, model in specs:
        model.fit(X_train, y_train)
        results.append(score_predictions(y_test, model.predict(X_test), name))
    return results


def scores_to_frame(scores: Sequence[ClassificationScores]) -> pd.DataFrame:
    """Assemble scores into a comparison table."""
    return pd.DataFrame([s.as_row() for s in scores])


def confusion_frame(y_true, y_pred, labels: Sequence[str]) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows=true, cols=predicted)."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(
        matrix,
        index=[f"true:{c}" for c in labels],
        columns=[f"pred:{c}" for c in labels],
    )


def per_class_report(y_true, y_pred, labels: Sequence[str]) -> str:
    """Per-class precision/recall/F1 text report."""
    return classification_report(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
