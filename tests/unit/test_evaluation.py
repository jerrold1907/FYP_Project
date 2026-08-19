"""Unit tests for the leakage-free evaluation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    baseline_scores,
    confusion_frame,
    purged_chronological_split,
    score_predictions,
    scores_to_frame,
)


def make_dates(n_days: int, tickers: int = 1) -> pd.Series:
    """Build a pooled date column: every ticker observed on every date."""
    days = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.Series(np.tile(days, tickers))


class TestPurgedChronologicalSplit:
    """The split must be ordered in time and free of label overlap."""

    def test_test_block_is_strictly_later_than_train(self):
        dates = make_dates(200)
        split = purged_chronological_split(dates, test_size=0.2, label_horizon=30)
        assert dates.iloc[split.train_idx].max() < dates.iloc[split.test_idx].min()

    def test_purge_gap_matches_label_horizon(self):
        """The gap between train end and test start spans label_horizon days."""
        dates = make_dates(300)
        horizon = 30
        split = purged_chronological_split(dates, test_size=0.2,
                                           label_horizon=horizon)
        unique = np.sort(dates.unique())
        gap = np.sum((unique >= split.purge_start) & (unique < split.split_date))
        assert gap == horizon

    def test_no_purge_when_horizon_zero(self):
        dates = make_dates(200)
        split = purged_chronological_split(dates, test_size=0.2, label_horizon=0)
        assert split.n_purged == 0
        assert split.purge_start == split.split_date

    def test_test_fraction_is_approximately_requested(self):
        dates = make_dates(500)
        split = purged_chronological_split(dates, test_size=0.2, label_horizon=0)
        assert abs(len(split.test_idx) / len(dates) - 0.2) < 0.02

    def test_all_rows_sharing_a_date_land_on_same_side(self):
        """With 10 tickers per date, no date may straddle the split."""
        dates = make_dates(200, tickers=10)
        split = purged_chronological_split(dates, test_size=0.2, label_horizon=30)
        train_dates = set(dates.iloc[split.train_idx])
        test_dates = set(dates.iloc[split.test_idx])
        assert train_dates.isdisjoint(test_dates)

    def test_unsorted_input_is_handled(self):
        dates = make_dates(200).sample(frac=1.0, random_state=0).reset_index(drop=True)
        split = purged_chronological_split(dates, test_size=0.2, label_horizon=30)
        assert dates.iloc[split.train_idx].max() < dates.iloc[split.test_idx].min()

    def test_multi_ticker_indices_are_positional(self):
        """Returned indices address rows, not pandas labels."""
        dates = make_dates(150, tickers=4)
        split = purged_chronological_split(dates, test_size=0.2, label_horizon=10)
        assert split.train_idx.max() < len(dates)
        assert split.test_idx.max() < len(dates)
        assert set(split.train_idx).isdisjoint(set(split.test_idx))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_test_size_rejected(self, bad):
        with pytest.raises(ValueError, match="test_size"):
            purged_chronological_split(make_dates(100), test_size=bad)

    def test_negative_horizon_rejected(self):
        with pytest.raises(ValueError, match="label_horizon"):
            purged_chronological_split(make_dates(100), label_horizon=-1)

    def test_too_few_dates_rejected(self):
        with pytest.raises(ValueError, match="at least 2 distinct dates"):
            purged_chronological_split(pd.Series(pd.to_datetime(["2020-01-01"])))

    def test_oversized_purge_rejected(self):
        """A horizon that consumes all training data must fail loudly."""
        with pytest.raises(ValueError, match="Purge removed all training rows"):
            purged_chronological_split(make_dates(50), test_size=0.2,
                                       label_horizon=100)


class TestScoring:
    """Metric helpers must be correct and self-consistent."""

    def test_perfect_predictions_score_one(self):
        y = ["Buy", "Hold", "Avoid", "Buy"]
        s = score_predictions(y, y, "perfect")
        assert s.accuracy == 1.0
        assert s.f1_weighted == 1.0
        assert s.f1_macro == 1.0

    def test_scores_record_test_size_and_name(self):
        y_true = ["Buy", "Hold", "Avoid"]
        y_pred = ["Buy", "Buy", "Avoid"]
        s = score_predictions(y_true, y_pred, "partial")
        assert s.name == "partial"
        assert s.n_test == 3
        assert 0.0 < s.f1_weighted < 1.0

    def test_macro_f1_penalises_ignoring_minority_class(self):
        """Weighted F1 can look acceptable while macro F1 exposes the failure."""
        y_true = ["Hold"] * 90 + ["Buy"] * 10
        y_pred = ["Hold"] * 100  # minority class never predicted
        s = score_predictions(y_true, y_pred, "majority-only")
        assert s.f1_macro < s.f1_weighted
        assert s.balanced_accuracy == pytest.approx(0.5)


class TestBaselines:
    """Baselines provide the reference every model must beat."""

    def test_returns_three_named_baselines(self):
        rng = np.random.default_rng(0)
        X_train, X_test = rng.normal(size=(60, 3)), rng.normal(size=(30, 3))
        y_train = ["Hold"] * 40 + ["Buy"] * 20
        y_test = ["Hold"] * 20 + ["Buy"] * 10

        results = baseline_scores(X_train, y_train, X_test, y_test)
        assert len(results) == 3
        assert all(r.name.startswith("Baseline:") for r in results)

    def test_majority_baseline_matches_class_prevalence(self):
        rng = np.random.default_rng(1)
        X_train, X_test = rng.normal(size=(80, 2)), rng.normal(size=(40, 2))
        y_train = ["Hold"] * 60 + ["Buy"] * 20
        y_test = ["Hold"] * 30 + ["Buy"] * 10

        majority = next(r for r in baseline_scores(X_train, y_train, X_test, y_test)
                        if "majority" in r.name)
        assert majority.accuracy == pytest.approx(0.75)


class TestTables:
    """Reporting helpers produce well-formed frames."""

    def test_scores_frame_has_one_row_per_model(self):
        scores = [score_predictions(["Buy"] * 5, ["Buy"] * 5, f"m{i}")
                  for i in range(3)]
        frame = scores_to_frame(scores)
        assert len(frame) == 3
        assert "F1 (weighted)" in frame.columns

    def test_confusion_frame_is_labelled_and_square(self):
        labels = ["Buy", "Hold", "Avoid"]
        frame = confusion_frame(["Buy", "Hold", "Avoid"], ["Buy", "Buy", "Avoid"],
                                labels)
        assert frame.shape == (3, 3)
        assert list(frame.index) == [f"true:{c}" for c in labels]
        assert list(frame.columns) == [f"pred:{c}" for c in labels]
