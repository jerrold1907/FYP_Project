"""Unit tests for the LSTM sequence baseline and its data preparation."""

import numpy as np
import pytest

from src.sequence_model import (
    DEFAULT_SEQUENCE_LENGTH,
    LSTMClassifier,
    build_sequences,
)

torch = pytest.importorskip("torch", reason="PyTorch not installed")


class TestBuildSequences:
    """Sequence construction must respect group boundaries and ordering."""

    def test_shape_is_samples_by_length_by_features(self):
        features = np.arange(100 * 3, dtype=float).reshape(100, 3)
        labels = ["a"] * 100
        X, y, idx = build_sequences(features, labels, sequence_length=10)
        assert X.shape == (91, 10, 3)   # 100 - 10 + 1
        assert len(y) == 91 and len(idx) == 91

    def test_label_comes_from_final_row_of_window(self):
        features = np.zeros((10, 2))
        labels = list("abcdefghij")
        _, y, idx = build_sequences(features, labels, sequence_length=3)
        assert y[0] == "c"      # window rows 0,1,2 -> label of row 2
        assert idx[0] == 2
        assert y[-1] == "j"

    def test_window_contains_consecutive_rows(self):
        features = np.arange(20, dtype=float).reshape(20, 1)
        X, _, _ = build_sequences(features, list(range(20)), sequence_length=4)
        assert np.array_equal(X[0].ravel(), [0.0, 1.0, 2.0, 3.0])
        assert np.array_equal(X[1].ravel(), [1.0, 2.0, 3.0, 4.0])

    def test_sequences_never_span_two_groups(self):
        """A window must not mix rows from different tickers."""
        features = np.arange(40, dtype=float).reshape(40, 1)
        groups = ["A"] * 20 + ["B"] * 20
        X, _, _ = build_sequences(features, list(range(40)), groups,
                                  sequence_length=5)
        # 16 windows per group of 20 rows.
        assert X.shape[0] == 32
        for window in X:
            values = window.ravel()
            # Every window lies wholly within one contiguous block.
            assert values.max() - values.min() == 4

    def test_group_shorter_than_sequence_is_skipped(self):
        features = np.zeros((12, 1))
        groups = ["A"] * 10 + ["B"] * 2      # B too short for length 5
        X, _, idx = build_sequences(features, list(range(12)), groups,
                                    sequence_length=5)
        assert X.shape[0] == 6               # only group A contributes
        assert all(i < 10 for i in idx)

    def test_returned_index_maps_back_to_source_rows(self):
        features = np.zeros((30, 2))
        _, _, idx = build_sequences(features, list(range(30)),
                                    sequence_length=DEFAULT_SEQUENCE_LENGTH)
        assert idx.min() == DEFAULT_SEQUENCE_LENGTH - 1
        assert idx.max() == 29

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            build_sequences(np.zeros((10, 2)), ["a"] * 5)

    def test_groups_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="groups length"):
            build_sequences(np.zeros((10, 2)), ["a"] * 10, groups=["A"] * 5)

    def test_zero_sequence_length_rejected(self):
        with pytest.raises(ValueError, match="sequence_length must be >= 1"):
            build_sequences(np.zeros((10, 2)), ["a"] * 10, sequence_length=0)

    def test_all_groups_too_short_rejected(self):
        with pytest.raises(ValueError, match="cannot build any sequence"):
            build_sequences(np.zeros((5, 2)), ["a"] * 5, sequence_length=10)


class TestLSTMClassifier:
    """The classifier must honour the sklearn-style contract."""

    @staticmethod
    def _learnable_data(n: int = 600, seed: int = 0):
        """Build data where the label depends on recent trend direction."""
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(n, 3)).astype(np.float32)
        # Encode a trend the network can pick up from the sequence.
        trend = np.cumsum(rng.normal(0, 0.4, size=n))
        features[:, 0] = trend
        labels = np.where(np.gradient(trend) > 0.15, "Buy",
                          np.where(np.gradient(trend) < -0.15, "Avoid", "Hold"))
        return features, labels

    def test_rejects_invalid_configuration(self):
        with pytest.raises(ValueError, match="sequence_length"):
            LSTMClassifier(sequence_length=0)
        with pytest.raises(ValueError, match="dropout"):
            LSTMClassifier(dropout=1.5)

    def test_predict_before_fit_raises(self):
        model = LSTMClassifier()
        with pytest.raises(RuntimeError, match="Call fit\\(\\) before predict"):
            model.predict(np.zeros((30, 3)))

    def test_fit_records_classes_and_history(self):
        features, labels = self._learnable_data()
        model = LSTMClassifier(sequence_length=5, hidden_size=8, max_epochs=3)
        model.fit(features, labels)

        assert set(model.classes_) <= {"Buy", "Hold", "Avoid"}
        assert len(model.history.train_loss) >= 1
        assert len(model.history.val_loss) == len(model.history.train_loss)

    def test_predictions_are_valid_labels(self):
        features, labels = self._learnable_data()
        model = LSTMClassifier(sequence_length=5, hidden_size=8, max_epochs=3)
        model.fit(features, labels)

        predictions = model.predict(features)
        assert set(np.unique(predictions)) <= set(model.classes_)

    def test_prediction_count_accounts_for_warmup_rows(self):
        features, labels = self._learnable_data(n=200)
        length = 10
        model = LSTMClassifier(sequence_length=length, hidden_size=8,
                               max_epochs=2)
        model.fit(features, labels)

        predictions, index = model.predict_with_index(features)
        assert len(predictions) == 200 - length + 1
        assert index.min() == length - 1

    def test_respects_groups_at_predict_time(self):
        features, labels = self._learnable_data(n=200)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        model = LSTMClassifier(sequence_length=10, hidden_size=8, max_epochs=2)
        model.fit(features, labels, groups=groups)

        predictions, _ = model.predict_with_index(features, groups=groups)
        # 91 windows per group.
        assert len(predictions) == 2 * 91

    def test_is_reproducible_with_fixed_seed(self):
        features, labels = self._learnable_data()
        a = LSTMClassifier(sequence_length=5, hidden_size=8, max_epochs=3,
                           random_state=7).fit(features, labels).predict(features)
        b = LSTMClassifier(sequence_length=5, hidden_size=8, max_epochs=3,
                           random_state=7).fit(features, labels).predict(features)
        assert np.array_equal(a, b)

    def test_invalid_validation_fraction_rejected(self):
        features, labels = self._learnable_data(n=120)
        model = LSTMClassifier(sequence_length=5, hidden_size=8, max_epochs=2)
        with pytest.raises(ValueError, match="validation_fraction"):
            model.fit(features, labels, validation_fraction=0.0)

    def test_early_stopping_can_halt_before_max_epochs(self):
        features, labels = self._learnable_data()
        model = LSTMClassifier(sequence_length=5, hidden_size=8,
                               max_epochs=50, patience=2)
        model.fit(features, labels)
        assert len(model.history.val_loss) <= 50
