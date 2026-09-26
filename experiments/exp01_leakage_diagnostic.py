"""
Diagnostic: does the random train/test split leak future information?

Compares three splitting strategies on identical data and features:
  A. Random stratified split (what the project currently reports: F1=0.84)
  B. Chronological split (train on past, test on future)
  C. Chronological split + purge/embargo (removes label-horizon overlap)

If A >> B, the reported F1 is inflated by temporal leakage, which would
explain the gap between 84% test F1 and 48.8% walk-forward signal accuracy.
"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.dummy import DummyClassifier

from src.features import compute_features, compute_target_labels
from src.stock_universe import TICKERS

FEATURES = ["close_price", "daily_return", "ma_5", "ma_20", "ma_50",
            "volatility", "volume", "RSI", "MACD"]
LABEL_HORIZON = 30  # forward-return window used to build targets

from src.market_data import load_prices

print("Loading data...")
frames = []
for t in TICKERS:
    df = load_prices(t)
    if len(df) < 200:
        continue
    f = compute_features(df)
    l = compute_target_labels(df)
    idx = f.index.intersection(l.index)
    part = f.loc[idx].copy()
    part["target"] = l.loc[idx, "target"]
    part["date"] = idx
    part["ticker"] = t
    frames.append(part)

data = pd.concat(frames, ignore_index=True)
data = data.sort_values("date").reset_index(drop=True)
print(f"Total samples: {len(data)}  |  classes: {data['target'].value_counts().to_dict()}\n")


def evaluate(X_tr, y_tr, X_te, y_te, label):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(Xtr, y_tr)
    pred = model.predict(Xte)

    # Baselines for context
    maj = DummyClassifier(strategy="most_frequent").fit(Xtr, y_tr).predict(Xte)
    strat = DummyClassifier(strategy="stratified", random_state=42).fit(Xtr, y_tr).predict(Xte)

    print(f"{label}")
    print(f"   n_train={len(X_tr):5d}  n_test={len(X_te):5d}")
    print(f"   RandomForest      F1={f1_score(y_te, pred, average='weighted'):.4f}  "
          f"Acc={accuracy_score(y_te, pred):.4f}")
    print(f"   Majority baseline F1={f1_score(y_te, maj, average='weighted'):.4f}  "
          f"Acc={accuracy_score(y_te, maj):.4f}")
    print(f"   Random baseline   F1={f1_score(y_te, strat, average='weighted'):.4f}  "
          f"Acc={accuracy_score(y_te, strat):.4f}\n")
    return f1_score(y_te, pred, average="weighted")


# --- A. Random stratified split (current method) ---
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
    data[FEATURES], data["target"], test_size=0.2,
    random_state=42, stratify=data["target"])
f1_a = evaluate(Xa_tr, ya_tr, Xa_te, ya_te,
                "A. RANDOM stratified split  (current reported method)")

# --- B. Chronological split ---
cut = int(len(data) * 0.8)
f1_b = evaluate(data[FEATURES].iloc[:cut], data["target"].iloc[:cut],
                data[FEATURES].iloc[cut:], data["target"].iloc[cut:],
                "B. CHRONOLOGICAL split      (train on past, test on future)")

# --- C. Chronological + purge/embargo ---
# Drop training rows whose 30-day label window overlaps the test period.
cut_date = data["date"].iloc[cut]
embargo_start = cut_date - pd.Timedelta(days=LABEL_HORIZON * 2)
train_mask = data["date"] < embargo_start
test_mask = data.index >= cut
f1_c = evaluate(data.loc[train_mask, FEATURES], data.loc[train_mask, "target"],
                data.loc[test_mask, FEATURES], data.loc[test_mask, "target"],
                "C. CHRONOLOGICAL + PURGE    (label overlap removed)")

print("=" * 62)
print(f"Random split F1        : {f1_a:.4f}")
print(f"Chronological F1       : {f1_b:.4f}   (drop of {f1_a - f1_b:+.4f})")
print(f"Chronological+purge F1 : {f1_c:.4f}   (drop of {f1_a - f1_c:+.4f})")
print("=" * 62)
if f1_a - f1_c > 0.15:
    print("VERDICT: severe temporal leakage in the random split.")
    print("The reported F1 is not a valid estimate of out-of-sample performance.")
else:
    print("VERDICT: leakage is modest.")
