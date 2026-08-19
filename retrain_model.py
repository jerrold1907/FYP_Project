"""
Train and export the stock signal model using a leakage-free evaluation.

Methodology note
----------------
An earlier version of this script used sklearn's random stratified
train_test_split. That is invalid for financial time series and inflated the
reported weighted F1 from roughly 0.36 to 0.83. Two mechanisms caused it:

  1. Rolling features (ma_20, ma_50, volatility) make consecutive rows
     near-duplicates, so a random split lets the model match almost-identical
     rows rather than forecast.
  2. Labels are 30-trading-day forward returns, so training labels overlapped
     the test period and encoded its outcomes.

This script uses a purged chronological split instead: the test block is the
most recent 20% of trading dates, and training rows within 30 trading days of
the split are discarded. See experiments/exp01_leakage_diagnostic.py for the
side-by-side comparison and src/evaluation.py for the implementation.

Every model is reported alongside majority-class and random baselines, because
weighted F1 on an imbalanced three-class target is not interpretable alone.

Run: python retrain_model.py
"""
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import (
    DEFAULT_LABEL_HORIZON,
    baseline_scores,
    confusion_frame,
    per_class_report,
    purged_chronological_split,
    score_predictions,
    scores_to_frame,
)
from src.features import compute_features, compute_target_labels
from src.stock_universe import TICKERS

FEATURE_COLUMNS = ["close_price", "daily_return", "ma_5", "ma_20", "ma_50",
                   "volatility", "volume", "RSI", "MACD"]
CLASS_LABELS = ["Buy", "Hold", "Avoid"]
TEST_SIZE = 0.20
SEED = 42


def load_dataset() -> pd.DataFrame:
    """Download prices and build the pooled, date-tagged feature dataset.

    The date column is retained because the chronological split depends on it;
    the previous random-split pipeline discarded it.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("Installing yfinance...")
        os.system(f"{sys.executable} -m pip install yfinance -q")
        import yfinance as yf

    frames = []
    for ticker in TICKERS:
        print(f"  {ticker:6}", end=" ")
        try:
            df = yf.download(ticker, start="2020-01-01", end="2024-12-31",
                             progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < 200:
                print(f"skipped ({len(df)} rows)")
                continue

            features = compute_features(df)
            labels = compute_target_labels(df)
            shared = features.index.intersection(labels.index)

            part = features.loc[shared].copy()
            part["target"] = labels.loc[shared, "target"]
            part["date"] = shared
            part["ticker"] = ticker
            frames.append(part)
            print(f"ok ({len(part)} rows)")
        except Exception as exc:
            print(f"error: {exc}")

    if not frames:
        raise RuntimeError("No ticker data could be downloaded.")

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values("date").reset_index(drop=True)


def main() -> None:
    print("=" * 66)
    print(f"MODEL TRAINING - purged chronological split (Python {sys.version.split()[0]})")
    print("=" * 66)

    print("\n[1/6] Downloading data and engineering features...")
    data = load_dataset()
    print(f"\n  Pooled dataset: {len(data)} rows across {data['ticker'].nunique()} tickers")
    print(f"  Date range    : {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"  Class balance : {data['target'].value_counts().to_dict()}")

    print("\n[2/6] Applying purged chronological split...")
    split = purged_chronological_split(
        data["date"], test_size=TEST_SIZE, label_horizon=DEFAULT_LABEL_HORIZON)
    train = data.iloc[split.train_idx]
    test = data.iloc[split.test_idx]

    print(f"  Test block starts : {split.split_date.date()}")
    print(f"  Training ends      : {split.purge_start.date()} "
          f"(purged {DEFAULT_LABEL_HORIZON} trading days)")
    print(f"  Rows purged        : {split.n_purged}")
    print(f"  Train / test rows  : {len(train)} / {len(test)}")
    print(f"  Train class balance: {train['target'].value_counts().to_dict()}")
    print(f"  Test class balance : {test['target'].value_counts().to_dict()}")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    print("\n[3/6] Scaling features (fitted on training data only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[4/6] Establishing naive baselines...")
    all_scores = baseline_scores(X_train_scaled, y_train, X_test_scaled, y_test,
                                 random_state=SEED)
    for s in all_scores:
        print(f"  {s.name:32} F1(w)={s.f1_weighted:.4f}  Acc={s.accuracy:.4f}")

    print("\n[5/6] Training candidate models...")
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=-1),
    }

    fitted = {}
    predictions = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        fitted[name] = model
        predictions[name] = preds
        scores = score_predictions(y_test, preds, name)
        all_scores.append(scores)
        print(f"  {name:32} F1(w)={scores.f1_weighted:.4f}  "
              f"Acc={scores.accuracy:.4f}  F1(macro)={scores.f1_macro:.4f}")

    # --- Comparison table ---
    table = scores_to_frame(all_scores)
    print("\n" + "-" * 66)
    print("MODEL COMPARISON (purged chronological test set)")
    print("-" * 66)
    print(table.to_string(index=False))

    # --- Model selection: highest weighted F1, accuracy as tiebreaker ---
    model_scores = [s for s in all_scores if not s.name.startswith("Baseline")]
    best = max(model_scores, key=lambda s: (s.f1_weighted, s.accuracy))
    best_baseline = max((s for s in all_scores if s.name.startswith("Baseline")),
                        key=lambda s: s.f1_weighted)

    print("\n" + "-" * 66)
    print(f"Best model      : {best.name}")
    print(f"  weighted F1   : {best.f1_weighted:.4f}")
    print(f"  accuracy      : {best.accuracy:.4f}")
    print(f"  macro F1      : {best.f1_macro:.4f}")
    print(f"Strongest baseline: {best_baseline.name} (F1={best_baseline.f1_weighted:.4f})")
    margin = best.f1_weighted - best_baseline.f1_weighted
    print(f"Margin over baseline: {margin:+.4f}")
    if margin < 0.05:
        print("\n  INTERPRETATION: the margin over a naive baseline is negligible.")
        print("  Under leakage-free evaluation these features show little to no")
        print("  predictive skill on 30-day forward returns. This is the honest")
        print("  out-of-sample result and should be reported as such.")

    # --- Diagnostics for the selected model ---
    print("\n" + "-" * 66)
    print(f"CONFUSION MATRIX - {best.name}")
    print("-" * 66)
    print(confusion_frame(y_test, predictions[best.name], CLASS_LABELS).to_string())
    print(f"\nPER-CLASS REPORT - {best.name}")
    print(per_class_report(y_test, predictions[best.name], CLASS_LABELS))

    print("[6/6] Exporting model artefacts...")
    os.makedirs("models", exist_ok=True)
    with open("models/stock_model.pkl", "wb") as fh:
        pickle.dump(fitted[best.name], fh)
    with open("models/scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    print("  models/stock_model.pkl")
    print("  models/scaler.pkl")

    os.makedirs("experiments", exist_ok=True)
    table.to_csv("experiments/exp02_model_comparison.csv", index=False)
    print("  experiments/exp02_model_comparison.csv")

    # Round-trip check
    with open("models/stock_model.pkl", "rb") as fh:
        reloaded = pickle.load(fh)
    assert (reloaded.predict(X_test_scaled[:20])
            == predictions[best.name][:20]).all(), "serialisation changed predictions"
    print("\n  Serialisation round-trip verified.")

    print("=" * 66)
    print("DONE")
    print("=" * 66)


if __name__ == "__main__":
    main()
