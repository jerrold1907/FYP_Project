"""Property-based tests for model selection and serialization.

This module contains property-based tests that verify critical properties of the
model training, selection, and serialization pipeline. Specifically, it tests that
the model serialization round-trip (save to pickle → load from pickle) preserves
predictions identically.

The serialization round-trip property is fundamental to the system's correctness:
after the training notebook exports the best model to disk, the StockRecommender
class loads it back for inference. If serialization altered the model's behavior,
predictions at inference time would differ from those validated during training,
silently breaking the system's guarantees.

Testing Framework: Hypothesis (property-based testing)
"""

# Feature: ai-stock-recommendation, Property 9: Model serialization round-trip preserves predictions

import pickle
import tempfile
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.ensemble import RandomForestClassifier


@pytest.mark.property
class TestModelSerializationRoundTrip:
    """Property tests verifying that model serialization preserves predictions.

    The serialization round-trip property states: for any trained scikit-learn model,
    serializing to pickle and deserializing SHALL produce a model that returns
    identical predictions on any input feature vector as the original model before
    serialization.

    Why this matters:
    - The training notebook saves the best model to models/stock_model.pkl
    - The StockRecommender loads this file at inference time
    - If pickle serialization corrupted the model's internal state (tree structures,
      learned weights, etc.), predictions would silently differ from training-time
      evaluation, invalidating the model comparison results.

    Why RandomForest is used for testing:
    - RandomForest is representative of the actual models used in the system
      (the training pipeline includes RandomForest as one of three candidates)
    - RandomForest produces deterministic predictions for a given input (no
      randomness at prediction time, unlike some ensemble methods with random
      tie-breaking)
    - RandomForest has complex internal state (multiple decision trees with
      splits, thresholds, and leaf values) that exercises pickle serialization
      thoroughly — if round-trip works for RF, simpler models (LogReg, DT)
      will also work correctly
    """

    @given(
        # Generate a random seed for reproducible synthetic data and model training.
        # Using a strategy for the seed allows Hypothesis to explore many different
        # model configurations (different random forests trained on different data).
        random_seed=st.integers(min_value=0, max_value=10000),
        # Generate the number of samples for synthetic training data.
        # We use a modest range (50-200) to keep tests fast while still being
        # enough for a RandomForest to learn meaningful splits.
        n_samples=st.integers(min_value=50, max_value=200),
        # Generate the number of test samples to verify predictions on.
        # Multiple test points increase confidence that predictions are truly identical.
        n_test_samples=st.integers(min_value=5, max_value=30),
    )
    def test_serialization_round_trip_preserves_predictions(
        self, random_seed: int, n_samples: int, n_test_samples: int
    ):
        """Verify that save→load produces identical predictions on test inputs.

        **Validates: Requirements 6.2**

        This property test performs the following steps:
        1. Generate synthetic training data with features resembling technical indicators
        2. Train a RandomForestClassifier on the synthetic data
        3. Generate synthetic test inputs
        4. Record predictions from the original (in-memory) model
        5. Serialize the model to a temporary file using pickle
        6. Deserialize the model from the file
        7. Record predictions from the deserialized model
        8. Assert that predictions are identical (element-wise equality)

        The property holds universally: regardless of the random seed, data size,
        or number of test points, serialization must never alter predictions.

        Args:
            random_seed: Seed for reproducible random data generation and model training.
            n_samples: Number of synthetic training samples to generate.
            n_test_samples: Number of test samples to verify predictions on.
        """
        # --- Step 1: Generate synthetic training data ---
        # We create data with 9 features matching the FEATURE_COLUMNS used by
        # the actual system (close_price, daily_return, ma_5, ma_20, ma_50,
        # volatility, volume, RSI, MACD). The exact values don't matter for
        # testing serialization — we just need a valid trained model.
        rng = np.random.RandomState(random_seed)

        # Synthetic feature matrix: 9 features, values in realistic-ish ranges
        X_train = rng.randn(n_samples, 9)

        # Synthetic target labels: 3 classes (0=Avoid, 1=Buy, 2=Hold)
        # matching the LABEL_MAP in stock_recommender.py
        y_train = rng.randint(0, 3, size=n_samples)

        # --- Step 2: Train a RandomForestClassifier ---
        # Using a small forest (10 trees) for speed. The n_estimators doesn't
        # affect whether serialization preserves predictions — even 1 tree
        # would suffice, but 10 gives a more representative test of RF's
        # multi-tree internal state being correctly serialized.
        model = RandomForestClassifier(
            n_estimators=10,
            random_state=random_seed,
            max_depth=5,  # Limit depth for fast training
        )
        model.fit(X_train, y_train)

        # --- Step 3: Generate synthetic test inputs ---
        # These are the inputs we'll compare predictions on before and after
        # serialization. Using a different seed offset ensures test data
        # differs from training data.
        X_test = rng.randn(n_test_samples, 9)

        # --- Step 4: Record predictions from the original model ---
        # These are the "ground truth" predictions that must be preserved.
        original_predictions = model.predict(X_test)

        # --- Step 5: Serialize the model to a temporary file using pickle ---
        # This mirrors what the training notebook does when saving to
        # models/stock_model.pkl — it uses pickle.dump to serialize.
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False
        ) as tmp_file:
            temp_path = tmp_file.name
            pickle.dump(model, tmp_file)

        try:
            # --- Step 6: Deserialize the model from the file ---
            # This mirrors what StockRecommender.__init__ does when it loads
            # the model using pickle.load.
            with open(temp_path, "rb") as f:
                loaded_model = pickle.load(f)

            # --- Step 7: Record predictions from the deserialized model ---
            loaded_predictions = loaded_model.predict(X_test)

            # --- Step 8: Assert predictions are identical ---
            # We use array_equal for exact element-wise comparison.
            # Predictions must be IDENTICAL, not just "close" — there is no
            # acceptable tolerance for classification label differences.
            np.testing.assert_array_equal(
                original_predictions,
                loaded_predictions,
                err_msg=(
                    "Model predictions differ after serialization round-trip. "
                    "This indicates pickle serialization corrupted the model's "
                    "internal state (tree structure, split thresholds, or leaf values)."
                ),
            )
        finally:
            # Clean up the temporary file to avoid polluting the filesystem
            os.unlink(temp_path)


# Feature: ai-stock-recommendation, Property 7: Stratified split preserves class proportions


@pytest.mark.property
class TestStratifiedSplitProportions:
    """Property tests verifying that stratified splitting preserves class proportions.

    Stratified splitting is a technique that ensures both the training and test sets
    maintain approximately the same class distribution as the original dataset. This
    is critical for multi-class classification because:

    1. If class proportions drift between splits, the model may be trained on a
       distribution that doesn't reflect reality (e.g., over-representing "Buy" signals
       and under-representing "Avoid" signals).
    2. Evaluation metrics on the test set become misleading if the test set has different
       class balance than the training set — accuracy would not reflect real-world
       performance.
    3. For the stock recommendation system, the target labels (Buy, Hold, Avoid) are
       typically imbalanced (markets trend upward historically), making stratification
       essential to avoid a split where one class is absent or severely under-represented.

    The implementation uses scikit-learn's `train_test_split` with `stratify=y` and
    `random_state=42`, which performs stratified sampling — it partitions the data so
    each split has approximately the same percentage of samples of each target class
    as the complete dataset.

    The 1 percentage point tolerance accounts for rounding effects in finite samples.
    When the dataset is small or class counts don't divide evenly into 80/20, minor
    deviations are mathematically unavoidable. For example, with 10 samples of a class
    in a dataset of 100, the expected count in the test set is 2 (10% of 20), but
    integer rounding could yield 1 or 3 samples, causing small proportion differences.
    A 1pp tolerance accommodates these finite-sample rounding artifacts while still
    catching genuine stratification failures.
    """

    @given(
        # Strategy for generating multi-class labeled datasets:
        # We generate a list of class labels where:
        # - n_classes: number of distinct classes (min 3 as required by the property,
        #   max 6 to cover cases beyond the system's 3 classes)
        # - samples_per_class: each class gets between 200 and 500 samples
        #
        # The minimum of 200 samples per class ensures the 1 percentage point
        # tolerance is mathematically achievable. With a 20% test split, the test
        # set needs ≥ 100 samples so each individual sample represents ≤ 1% —
        # making off-by-one rounding stay within 1pp. With 3+ classes × 200+ per
        # class = 600+ total → 120+ test samples, each representing ~0.83%.
        #
        # The varying list sizes (3-6 classes) test that stratification works
        # correctly regardless of the number of target classes, not just for
        # the system's 3 classes (Buy, Hold, Avoid).
        n_classes=st.integers(min_value=3, max_value=6),
        data=st.data(),
    )
    def test_stratified_split_preserves_class_proportions(
        self, n_classes: int, data
    ):
        """Verify that stratified 80/20 split preserves class proportions within 1pp.

        **Validates: Requirements 4.3**

        This property test performs the following steps:
        1. Generate a labeled dataset with the specified number of classes, ensuring
           each class has sufficient samples for stratification
        2. Compute the original class proportions in the full dataset
        3. Apply an 80/20 stratified split using scikit-learn's train_test_split
           with stratify=y and random_state=42
        4. Compute class proportions in both the training and test sets
        5. Assert that every class proportion in train and test differs from the
           original by no more than 1 percentage point (0.01 in decimal)

        The tolerance of 1 percentage point is used because:
        - Stratified splitting distributes samples proportionally, but with finite
          sample sizes, integer rounding means exact proportions cannot always be
          achieved (e.g., you can't put 2.3 samples in a split — it must be 2 or 3)
        - scikit-learn's implementation minimizes this deviation but cannot eliminate it
        - 1pp is strict enough to detect broken stratification while accommodating
          the unavoidable rounding in finite datasets

        Args:
            n_classes: Number of distinct classes in the generated dataset (3-6).
            data: Hypothesis data object for drawing additional values.
        """
        from sklearn.model_selection import train_test_split
        from collections import Counter

        # --- Step 1: Generate a multi-class labeled dataset ---
        # We need sufficient samples per class so that integer rounding effects
        # during the 80/20 split stay within 1 percentage point.
        #
        # Mathematical justification for the minimum of 200 per class:
        # With a 20% test split, the test set size = 0.2 * total. For 1pp tolerance,
        # each sample in the test set must represent ≤ 1% of that set, meaning the
        # test set needs ≥ 100 samples. With 3+ classes × 200+ per class = 600+
        # total → 120+ test samples. Each test sample represents ~0.83%, so
        # off-by-one rounding causes ≤ 0.83pp deviation — within tolerance.
        #
        # Drawing samples_per_class as a list allows Hypothesis to explore datasets
        # with varying class imbalance (e.g., [300, 200, 250] vs [200, 200, 200]).
        # This reflects realistic scenarios where Buy/Hold/Avoid labels are not
        # perfectly balanced in stock data.
        samples_per_class = data.draw(
            st.lists(
                # Each class gets between 200 and 500 samples. The minimum of 200
                # ensures the total dataset (600+ for 3 classes) produces a test
                # set of 120+ samples, making the 1pp tolerance achievable.
                # This mirrors realistic dataset sizes — the actual stock training
                # data has thousands of rows across 3 classes (Buy, Hold, Avoid).
                st.integers(min_value=200, max_value=500),
                min_size=n_classes,
                max_size=n_classes,
            ),
            label="samples_per_class",
        )

        # Build the label array: class 0 repeated samples_per_class[0] times,
        # class 1 repeated samples_per_class[1] times, etc.
        y = []
        for class_idx, count in enumerate(samples_per_class):
            y.extend([class_idx] * count)
        y = np.array(y)

        total_samples = len(y)

        # Create a dummy feature matrix (content doesn't matter for split testing,
        # only the labels determine stratification behavior)
        X = np.zeros((total_samples, 1))

        # --- Step 2: Compute original class proportions ---
        original_counts = Counter(y)
        original_proportions = {
            cls: count / total_samples
            for cls, count in original_counts.items()
        }

        # --- Step 3: Apply 80/20 stratified split with seed=42 ---
        # This mirrors the exact call used in the training notebook:
        # train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # --- Step 4: Compute class proportions in train and test sets ---
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)

        train_proportions = {
            cls: count / len(y_train)
            for cls, count in train_counts.items()
        }
        test_proportions = {
            cls: count / len(y_test)
            for cls, count in test_counts.items()
        }

        # --- Step 5: Assert proportions differ by ≤ 1 percentage point ---
        # The tolerance is 0.01 (1 percentage point expressed as a decimal fraction).
        # For example, if the original has 30% "Buy" labels, the train and test sets
        # must each have between 29% and 31% "Buy" labels.
        tolerance = 0.01  # 1 percentage point tolerance for finite sample rounding

        for cls in original_proportions:
            original_prop = original_proportions[cls]

            # Check training set proportion
            train_prop = train_proportions.get(cls, 0.0)
            train_diff = abs(train_prop - original_prop)
            assert train_diff <= tolerance, (
                f"Training set class {cls} proportion ({train_prop:.4f}) differs from "
                f"original ({original_prop:.4f}) by {train_diff:.4f}, which exceeds "
                f"the 1 percentage point tolerance ({tolerance}). "
                f"Dataset: {total_samples} samples, {n_classes} classes, "
                f"samples_per_class={samples_per_class}"
            )

            # Check test set proportion
            test_prop = test_proportions.get(cls, 0.0)
            test_diff = abs(test_prop - original_prop)
            assert test_diff <= tolerance, (
                f"Test set class {cls} proportion ({test_prop:.4f}) differs from "
                f"original ({original_prop:.4f}) by {test_diff:.4f}, which exceeds "
                f"the 1 percentage point tolerance ({tolerance}). "
                f"Dataset: {total_samples} samples, {n_classes} classes, "
                f"samples_per_class={samples_per_class}"
            )


# Feature: ai-stock-recommendation, Property 8: Best model selection by F1 with tiebreaker


def select_best_model(scores: dict[str, dict[str, float]]) -> str:
    """Select the best model by highest weighted F1-score, with accuracy as tiebreaker.

    This function implements the same selection logic used in the training notebook's
    Model Comparison section. The algorithm is:

    1. Find the maximum weighted F1-score across all models.
    2. Identify all models that share this maximum F1-score (candidates).
    3. If there is only one candidate, it is the best model.
    4. If there are multiple candidates (tie in F1), select the one with
       the highest accuracy as a tiebreaker.

    This is extracted here so it can be tested independently of the notebook context.
    The notebook uses the exact same logic inline:
        best_f1 = max(f1_scores.values())
        best_candidates = [name for name, score in f1_scores.items() if score == best_f1]
        if len(best_candidates) > 1:
            best_model_name = max(best_candidates, key=lambda n: results[n]['Accuracy'])
        else:
            best_model_name = best_candidates[0]

    Args:
        scores: Dictionary mapping model names to their metrics.
                Each value is a dict with at least 'f1' and 'accuracy' keys,
                where both values are floats in [0, 1].

    Returns:
        The name (string key) of the best model.
    """
    # Step 1: Find the highest F1-score among all models
    f1_scores = {name: metrics["f1"] for name, metrics in scores.items()}
    best_f1 = max(f1_scores.values())

    # Step 2: Find all models tied at the highest F1
    best_candidates = [name for name, f1 in f1_scores.items() if f1 == best_f1]

    # Step 3/4: If tie, break by accuracy; otherwise take the single winner
    if len(best_candidates) > 1:
        best_model_name = max(
            best_candidates, key=lambda n: scores[n]["accuracy"]
        )
    else:
        best_model_name = best_candidates[0]

    return best_model_name


@pytest.mark.property
class TestBestModelSelectionByF1WithTiebreaker:
    """Property tests verifying that best model selection uses F1-first, accuracy as tiebreaker.

    The training notebook compares three models (Logistic Regression, Decision Tree,
    Random Forest) and selects the best one for export. The selection criterion is:

    1. **Primary criterion — Highest weighted F1-score**: The model with the highest
       weighted F1-score on the test set is selected. Weighted F1 balances precision and
       recall across all target classes (Buy, Hold, Avoid), weighted by class support.
       This is preferred over raw accuracy because financial datasets are often imbalanced.

    2. **Tiebreaker — Highest accuracy**: If two or more models have identical F1-scores
       (e.g., due to similar precision/recall trade-offs), accuracy is used as the
       tiebreaker. Accuracy provides a simple, interpretable fallback metric.

    The property test generates random score triples (F1 and accuracy for 3 models) and
    verifies that the selection function always picks the correct model according to
    these rules. This provides confidence that the selection logic is correct regardless
    of what specific metric values arise during training.

    Why this matters:
    - If the selection logic has a bug (e.g., selects by accuracy instead of F1, or
      handles ties incorrectly), the wrong model could be exported to models/stock_model.pkl
    - The exported model is used for all future stock predictions via StockRecommender
    - An inferior model would silently degrade prediction quality without any error signal
    """

    @given(
        # Strategy for generating F1 scores for 3 models.
        # Each F1-score is a float in [0.0, 1.0] representing the weighted F1.
        # We use floats() with allow_nan=False and allow_infinity=False to ensure
        # only valid probability-like values are generated.
        # The range [0, 1] covers all possible F1-scores from worst to perfect.
        f1_model_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        f1_model_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        f1_model_c=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        # Strategy for generating accuracy scores for 3 models.
        # Accuracy is also in [0.0, 1.0]. Note that accuracy and F1 are generated
        # independently — in reality they're correlated, but independence is fine for
        # testing selection logic (we don't need realistic metric relationships, just
        # correct selection behavior for any possible combination of values).
        acc_model_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        acc_model_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        acc_model_c=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_best_model_selected_by_f1_then_accuracy(
        self,
        f1_model_a: float,
        f1_model_b: float,
        f1_model_c: float,
        acc_model_a: float,
        acc_model_b: float,
        acc_model_c: float,
    ):
        """Verify that the selection function picks the model with highest F1, accuracy as tiebreaker.

        **Validates: Requirements 5.3**

        This property test works by:
        1. Constructing a scores dictionary with randomly generated F1 and accuracy for 3 models
        2. Running the selection function (system under test)
        3. Independently computing the expected winner using an oracle implementation
        4. Asserting the system and oracle agree

        The oracle (independent implementation of selection logic):
        - Finds the maximum F1-score among all three models
        - Filters to only models with that max F1
        - Among those, picks the one with the highest accuracy
        - If there's still a tie (same F1 AND same accuracy), any of the tied models is acceptable

        This oracle is intentionally written differently from the function under test
        to avoid the test simply duplicating the implementation. The oracle uses explicit
        filtering and sorting rather than the max()/list comprehension approach used in
        select_best_model, providing genuine cross-validation of the logic.

        Args:
            f1_model_a: Weighted F1-score for Model A (Logistic Regression equivalent).
            f1_model_b: Weighted F1-score for Model B (Decision Tree equivalent).
            f1_model_c: Weighted F1-score for Model C (Random Forest equivalent).
            acc_model_a: Accuracy for Model A.
            acc_model_b: Accuracy for Model B.
            acc_model_c: Accuracy for Model C.
        """
        # --- Step 1: Construct the scores dictionary ---
        # Three models with randomly generated F1 and accuracy scores.
        # Model names mirror the training notebook's model set.
        scores = {
            "Logistic Regression": {"f1": f1_model_a, "accuracy": acc_model_a},
            "Decision Tree": {"f1": f1_model_b, "accuracy": acc_model_b},
            "Random Forest": {"f1": f1_model_c, "accuracy": acc_model_c},
        }

        # --- Step 2: Run the system under test ---
        selected = select_best_model(scores)

        # --- Step 3: Oracle — independently determine the expected winner ---
        # The oracle uses a different algorithmic approach (sort-based) to avoid
        # simply re-implementing the function under test:
        #   - Sort all models by (F1 descending, accuracy descending)
        #   - The first element after sorting is the expected winner
        #
        # This correctly handles:
        #   - Clear F1 winner: model with highest F1 is first regardless of accuracy
        #   - F1 tie: among tied F1 models, highest accuracy comes first
        #   - Full tie (same F1 AND accuracy): any tied model is acceptable
        models_sorted = sorted(
            scores.keys(),
            key=lambda name: (scores[name]["f1"], scores[name]["accuracy"]),
            reverse=True,  # Descending: highest F1 first, then highest accuracy
        )

        # The oracle's expected winner is the first model after sorting
        expected_winner = models_sorted[0]

        # --- Step 4: Handle the edge case of a full tie ---
        # If multiple models have BOTH the same F1 AND the same accuracy,
        # either one is a valid selection (the requirement only specifies
        # F1-first, accuracy-tiebreaker — it doesn't define a third tiebreaker).
        # In this case, we check that the selected model is among the valid choices.
        max_f1 = max(scores[name]["f1"] for name in scores)
        f1_tied_models = [name for name in scores if scores[name]["f1"] == max_f1]

        if len(f1_tied_models) > 1:
            # Multiple models tied on F1 — check accuracy among them
            max_acc_among_tied = max(
                scores[name]["accuracy"] for name in f1_tied_models
            )
            # Valid winners: all models with best F1 AND best accuracy among F1-tied set
            valid_winners = [
                name
                for name in f1_tied_models
                if scores[name]["accuracy"] == max_acc_among_tied
            ]
        else:
            # Only one model has the best F1 — it must be the winner
            valid_winners = f1_tied_models

        # --- Step 5: Assert the selected model is a valid winner ---
        assert selected in valid_winners, (
            f"Selection function chose '{selected}' but valid winner(s) are {valid_winners}. "
            f"Scores: {scores}. "
            f"Max F1={max_f1}, F1-tied models={f1_tied_models}, "
            f"max accuracy among tied={max_acc_among_tied if len(f1_tied_models) > 1 else 'N/A'}"
        )

        # Additionally verify the selected model has the best F1-score
        assert scores[selected]["f1"] == max_f1, (
            f"Selected model '{selected}' has F1={scores[selected]['f1']} "
            f"but max F1 is {max_f1}. The primary selection criterion (highest F1) was violated."
        )
