"""
Experiment 8: how much do the results depend on the Buy label threshold?

The Buy class is a 30-day forward return above +10%, Hold is 0% to +10% and
Avoid is below 0%. The +10% threshold was fixed before any model was trained
and was never tuned, but the first report draft noted that it had not been
tested either. This repeats the core classification and backtest analyses on
the 20-stock universe with Buy thresholds of +5%, +7.5%, +10% and +15%. The
Avoid boundary stays at 0%, the natural line between a gain and a loss.

Accuracy is not comparable across thresholds, because each threshold changes
the class balance and therefore the task. The comparable quantities are the
margin over the majority-class baseline, macro F1 against the random
baseline, Buy recall, and the backtest's return relative to buy-and-hold.

Run: python experiments/exp08_label_sensitivity.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.backtester import BacktestConfig, run_backtest
from src.evaluation import (
    DEFAULT_LABEL_HORIZON,
    purged_chronological_split,
    score_predictions,
)
from src.features import compute_features, compute_target_labels
from src.market_data import load_prices
from src.statistics_tests import mcnemar_test, mean_return_test, wilson_interval
from src.stock_universe import TICKERS

FEATURES = ["close_price", "daily_return", "ma_5", "ma_20", "ma_50",
            "volatility", "volume", "RSI", "MACD"]
SEED = 42
THRESHOLDS = (0.05, 0.075, 0.10, 0.15)


def header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def build_dataset(prices: dict[str, pd.DataFrame],
                  buy_threshold: float) -> pd.DataFrame:
    """Pooled feature dataset labelled with the given Buy threshold."""
    frames = []
    for ticker, df in prices.items():
        feats = compute_features(df)
        labels = compute_target_labels(df, buy_threshold=buy_threshold)
        shared = feats.index.intersection(labels.index)
        part = feats.loc[shared].copy()
        part["target"] = labels.loc[shared, "target"]
        part["date"] = shared
        part["ticker"] = ticker
        frames.append(part)
    return (pd.concat(frames, ignore_index=True)
            .sort_values("date").reset_index(drop=True))


def classify(data: pd.DataFrame) -> dict:
    """Logistic regression against the baselines on the purged split."""
    split = purged_chronological_split(
        data["date"], test_size=0.2, label_horizon=DEFAULT_LABEL_HORIZON)
    train, test = data.iloc[split.train_idx], data.iloc[split.test_idx]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test = scaler.transform(test[FEATURES])
    y_train, y_test = train["target"], test["target"].to_numpy()

    predictions = {}
    for name, model in {
        "lr": LogisticRegression(max_iter=1000, random_state=SEED),
        "majority": DummyClassifier(strategy="most_frequent"),
        "random": DummyClassifier(strategy="stratified", random_state=SEED),
    }.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    lr = predictions["lr"]
    ci = wilson_interval(int((lr == y_test).sum()), len(y_test))
    majority_acc = float((predictions["majority"] == y_test).mean())
    buy = y_test == "Buy"
    return {
        "lr_accuracy": ci.estimate,
        "lr_acc_ci": f"[{ci.lower:.3f}, {ci.upper:.3f}]",
        "majority_accuracy": majority_acc,
        "margin_over_majority": ci.estimate - majority_acc,
        "lr_vs_majority_p": mcnemar_test(y_test, lr, predictions["majority"],
                                         "lr", "majority").p_value,
        "lr_f1_macro": score_predictions(y_test, lr, "lr").f1_macro,
        "random_f1_macro": score_predictions(
            y_test, predictions["random"], "random").f1_macro,
        "lr_buy_recall": float((lr[buy] == "Buy").mean()) if buy.any() else 0.0,
    }


def backtest(prices: dict[str, pd.DataFrame], buy_threshold: float) -> dict:
    """Walk-forward backtest with models trained on the given labels."""
    config = BacktestConfig(train_window_days=500, test_window_days=60,
                            step_days=60, model_type="lr",
                            buy_threshold=buy_threshold)
    rows = []
    for ticker, df in prices.items():
        result = run_backtest(df, config)
        rows.append({
            "trades": result.total_trades,
            "wins": result.profitable_trades,
            "excess": result.total_return_pct - result.benchmark_return_pct,
            "strategy": result.total_return_pct,
        })
    frame = pd.DataFrame(rows)
    wins, trades = int(frame["wins"].sum()), int(frame["trades"].sum())
    pooled = wilson_interval(wins, trades)
    return {
        "trades": trades,
        "win_rate": pooled.estimate,
        "win_ci": f"[{pooled.lower:.3f}, {pooled.upper:.3f}]",
        "mean_strategy_return": frame["strategy"].mean(),
        "mean_excess_return": frame["excess"].mean(),
        "excess_p": mean_return_test(frame["excess"].to_numpy(), 0.0).p_value,
    }


def main() -> None:
    header("SENSITIVITY OF THE RESULTS TO THE BUY LABEL THRESHOLD")
    prices = {ticker: load_prices(ticker) for ticker in TICKERS}

    rows = []
    for threshold in THRESHOLDS:
        data = build_dataset(prices, threshold)
        balance = data["target"].value_counts(normalize=True)
        row = {
            "buy_threshold": f"+{threshold:.1%}",
            "share_buy": balance.get("Buy", 0.0),
            "share_hold": balance.get("Hold", 0.0),
            "share_avoid": balance.get("Avoid", 0.0),
        }
        row.update(classify(data))
        row.update(backtest(prices, threshold))
        rows.append(row)
        print(f"\nBuy threshold +{threshold:.1%}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items() if k != "buy_threshold"))

    table = pd.DataFrame(rows)
    header("SUMMARY")
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "exp08_label_sensitivity.csv")
    table.to_csv(out, index=False)
    print(f"\nWritten: {os.path.basename(out)}")


if __name__ == "__main__":
    main()
