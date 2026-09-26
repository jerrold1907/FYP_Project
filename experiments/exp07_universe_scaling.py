"""
Experiment 7: do the 20-stock results hold on a 100-stock universe?

Supervisor feedback on the first report draft: twenty stocks are too few for a
real recommender. Nothing in the pipeline depends on the size of the universe,
so this script repeats the core analyses on the 100 companies of the S&P 100
(SP100 in src/stock_universe.py). The 20-stock core universe is re-run in the
same pass on the same data snapshot, so the two columns of the summary are
directly comparable:

  1. Leakage: Random Forest weighted F1 under a random stratified split and
     under the purged chronological split (the comparison made in exp01).
  2. Classifiers on the purged chronological split: logistic regression,
     Random Forest, a decision tree, the LSTM and two naive baselines, all
     scored on the rows the LSTM can predict (the protocol of exp04), with
     McNemar tests of logistic regression against each alternative.
  3. Walk-forward backtest of the logistic regression strategy on every
     ticker with enough history (the protocol of exp03).
  4. Accuracy by market regime, with a Kruskal-Wallis test (exp03).

Survivorship bias: the S&P 100 list is today's, so it favours companies that
did well over 2020-2024, which inflates the buy-and-hold benchmark. See the
note on SP100 in src/stock_universe.py.

Run: python experiments/exp07_universe_scaling.py
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.backtester import BacktestConfig, run_backtest
from src.evaluation import (
    DEFAULT_LABEL_HORIZON,
    purged_chronological_split,
    score_predictions,
)
from src.features import compute_features, compute_target_labels
from src.market_data import load_prices
from src.sequence_model import DEFAULT_SEQUENCE_LENGTH, LSTMClassifier
from src.statistics_tests import (
    chance_matched_test,
    kruskal_wallis,
    mcnemar_test,
    mean_return_test,
    proportion_test,
    sharpe_ratio_interval,
    wilson_interval,
)
from src.stock_universe import SP100_TICKERS, TICKERS

FEATURES = ["close_price", "daily_return", "ma_5", "ma_20", "ma_50",
            "volatility", "volume", "RSI", "MACD"]
SEED = 42
#: Tickers with fewer trading days in the study period are excluded, as in
#: every other experiment.
MIN_ROWS = 200

UNIVERSES = {
    "Core 20": TICKERS,
    "S&P 100": SP100_TICKERS,
}

#: Same regimes as exp03; 2020 cannot be tested because nothing precedes it.
REGIMES = {
    "2021 bull market": ("2021-01-01", "2021-12-31"),
    "2022 bear market": ("2022-01-01", "2022-12-31"),
    "2023-24 recovery": ("2023-01-01", "2024-11-30"),
}


def header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def section(text: str) -> None:
    print("\n" + "-" * 72)
    print(text)
    print("-" * 72)


def build_dataset(tickers: list[str]):
    """Pooled feature dataset plus the raw prices each backtest needs.

    Returns:
        (data in ticker order as built, {ticker: prices}, {ticker: reason}
        for every ticker that was excluded).
    """
    frames, prices, excluded = [], {}, {}
    for ticker in tickers:
        df = load_prices(ticker)
        if len(df) < MIN_ROWS:
            excluded[ticker] = f"only {len(df)} trading days in 2020-2024"
            continue
        feats = compute_features(df)
        labels = compute_target_labels(df)
        shared = feats.index.intersection(labels.index)
        part = feats.loc[shared].copy()
        part["target"] = labels.loc[shared, "target"]
        part["date"] = shared
        part["ticker"] = ticker
        frames.append(part)
        prices[ticker] = df

    return pd.concat(frames, ignore_index=True), prices, excluded


def random_forest_f1(X_train, y_train, X_test, y_test) -> float:
    """Weighted F1 of the exp01 Random Forest on one train/test division."""
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, random_state=SEED,
                                   n_jobs=-1)
    model.fit(scaler.fit_transform(X_train), y_train)
    predictions = model.predict(scaler.transform(X_test))
    return f1_score(y_test, predictions, average="weighted")


def leakage(pooled: pd.DataFrame) -> dict:
    """Random versus purged chronological split, holding everything else.

    Follows exp01 step for step (date-sorted rows, an 80% row cut, a purge of
    twice the label horizon in calendar days), so the Core 20 figures here
    reproduce exp01's exactly.
    """
    data = pooled.sort_values("date").reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["target"], test_size=0.2, random_state=SEED,
        stratify=data["target"])
    leaky = random_forest_f1(X_train, y_train, X_test, y_test)

    cut = int(len(data) * 0.8)
    purge_start = (data["date"].iloc[cut]
                   - pd.Timedelta(days=DEFAULT_LABEL_HORIZON * 2))
    train = data[data["date"] < purge_start]
    test = data[data.index >= cut]
    purged = random_forest_f1(train[FEATURES], train["target"],
                              test[FEATURES], test["target"])

    print(f"  Random stratified split  : weighted F1 = {leaky:.4f}")
    print(f"  Purged chronological     : weighted F1 = {purged:.4f}")
    print(f"  Inflation from leakage   : {leaky - purged:+.4f}")
    return {"f1_random_split": leaky, "f1_purged_split": purged}


def classifiers(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """All candidate models on the rows the LSTM can predict."""
    split = purged_chronological_split(
        data["date"], test_size=0.2, label_horizon=DEFAULT_LABEL_HORIZON)
    train, test = data.iloc[split.train_idx], data.iloc[split.test_idx]
    print(f"  Train n={len(train)}, test n={len(test)}; test block begins "
          f"{split.split_date.date()}; {split.n_purged} rows purged.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test = scaler.transform(test[FEATURES])
    y_train = train["target"].to_numpy()
    y_test = test["target"].to_numpy()

    start = time.time()
    lstm = LSTMClassifier(sequence_length=DEFAULT_SEQUENCE_LENGTH,
                          hidden_size=48, random_state=SEED)
    lstm.fit(X_train, y_train, groups=train["ticker"].to_numpy(),
             verbose=False)
    lstm_predictions, row_index = lstm.predict_with_index(
        X_test, groups=test["ticker"].to_numpy())
    print(f"  LSTM trained in {time.time() - start:.1f}s; scores "
          f"{len(row_index)} of {len(test)} test rows.")

    y_true = y_test[row_index]
    predictions = {"LSTM": lstm_predictions}
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                  random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100,
                                                random_state=SEED, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Baseline: majority": DummyClassifier(strategy="most_frequent"),
        "Baseline: random": DummyClassifier(strategy="stratified",
                                            random_state=SEED),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test[row_index])

    rows = []
    for name, preds in predictions.items():
        scores = score_predictions(y_true, preds, name)
        ci = wilson_interval(int((preds == y_true).sum()), len(y_true))
        buy = y_true == "Buy"
        rows.append({
            "model": name,
            "accuracy": ci.estimate,
            "acc_ci_lower": ci.lower,
            "acc_ci_upper": ci.upper,
            "f1_weighted": scores.f1_weighted,
            "f1_macro": scores.f1_macro,
            "balanced_accuracy": scores.balanced_accuracy,
            "buy_recall": float((preds[buy] == "Buy").mean()),
            "n_test": len(y_true),
        })
    table = pd.DataFrame(rows)
    print()
    print(table.round(4).to_string(index=False))

    tests = {}
    print("\n  McNemar tests:")
    pairs = [("Logistic Regression", other) for other in
             ["LSTM", "Random Forest", "Decision Tree",
              "Baseline: majority", "Baseline: random"]]
    pairs += [("LSTM", "Baseline: majority"), ("LSTM", "Baseline: random")]
    for a, b in pairs:
        result = mcnemar_test(y_true, predictions[a], predictions[b], a, b)
        short = "lr" if a == "Logistic Regression" else "lstm"
        tests[f"mcnemar_{short}_vs_{b}"] = result.p_value
        print(f"    {a} vs {b:20} p={result.p_value:.4f}  "
              f"({'significant' if result.significant() else 'not significant'})"
              f"; {result.detail.split(';')[-1].strip()}")
    return table, tests


def backtest(prices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Walk-forward backtest of the logistic regression strategy."""
    config = BacktestConfig(train_window_days=500, test_window_days=60,
                            step_days=60, model_type="lr")
    rows, skipped, chances = [], {}, []
    for ticker, df in prices.items():
        try:
            result = run_backtest(df, config)
        except ValueError as exc:
            skipped[ticker] = str(exc)
            continue
        chances.extend(t.random_entry_win_prob for t in result.trades)
        returns = pd.Series(result.equity_curve).pct_change().dropna()
        try:
            ci = sharpe_ratio_interval(returns.to_numpy())
            if not ci.excludes(0.0):
                sharpe = "includes 0"
            else:
                sharpe = "above 0" if ci.lower > 0 else "below 0"
        except ValueError:
            sharpe = "undefined"
        rows.append({
            "ticker": ticker,
            "trades": result.total_trades,
            "wins": result.profitable_trades,
            "strategy_return": result.total_return_pct,
            "benchmark_return": result.benchmark_return_pct,
            "sharpe_interval": sharpe,
        })

    frame = pd.DataFrame(rows)
    frame["excess"] = frame["strategy_return"] - frame["benchmark_return"]
    if skipped:
        print(f"  Too little history to backtest: {', '.join(skipped)}")

    wins, trades = int(frame["wins"].sum()), int(frame["trades"].sum())
    pooled = wilson_interval(wins, trades)
    excess_test = mean_return_test(frame["excess"].to_numpy(), 0.0)
    sharpe_counts = frame["sharpe_interval"].value_counts().to_dict()
    zero_trade = int((frame["trades"] == 0).sum())
    matched = chance_matched_test(wins, chances)

    print(f"  Tickers backtested    : {len(frame)} ({zero_trade} never traded)")
    print(f"  Pooled win rate       : {wins}/{trades} = {pooled.estimate:.1%} "
          f"[{pooled.lower:.3f}, {pooled.upper:.3f}]")
    print(f"  {proportion_test(wins, trades, 0.5)}")
    print(f"  {matched}")
    print(f"  Mean strategy return  : {frame['strategy_return'].mean():+.2f}%")
    print(f"  Mean benchmark return : {frame['benchmark_return'].mean():+.2f}%")
    print(f"  Mean excess return    : {frame['excess'].mean():+.2f}%  "
          f"(p={excess_test.p_value:.4f})")
    print(f"  Sharpe intervals      : {sharpe_counts}")

    summary = {
        "tickers_backtested": len(frame),
        "zero_trade_tickers": zero_trade,
        "trades": trades,
        "wins": wins,
        "win_rate": pooled.estimate,
        "win_ci_lower": pooled.lower,
        "win_ci_upper": pooled.upper,
        "win_rate_p": proportion_test(wins, trades, 0.5).p_value,
        "chance_matched_win_rate": float(np.mean(chances)),
        "chance_matched_z": matched.statistic,
        "chance_matched_p": matched.p_value,
        "mean_strategy_return": frame["strategy_return"].mean(),
        "mean_benchmark_return": frame["benchmark_return"].mean(),
        "mean_excess_return": frame["excess"].mean(),
        "excess_p": excess_test.p_value,
        "sharpe_includes_0": sharpe_counts.get("includes 0", 0),
        "sharpe_above_0": sharpe_counts.get("above 0", 0),
        "sharpe_below_0": sharpe_counts.get("below 0", 0),
    }
    return frame, summary


def regimes(data: pd.DataFrame) -> dict:
    """Out-of-sample accuracy by market regime (exp03's protocol)."""
    horizon_gap = pd.Timedelta(days=DEFAULT_LABEL_HORIZON * 2)
    groups, summary = {}, {}
    for label, (start, end) in REGIMES.items():
        regime_start = pd.Timestamp(start)
        train = data[data["date"] < regime_start - horizon_gap]
        window = data[(data["date"] >= regime_start)
                      & (data["date"] <= pd.Timestamp(end))]
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, random_state=SEED)
        model.fit(scaler.fit_transform(train[FEATURES]), train["target"])
        correct = (model.predict(scaler.transform(window[FEATURES]))
                   == window["target"].to_numpy()).astype(float)
        groups[label] = correct.tolist()
        ci = wilson_interval(int(correct.sum()), len(correct))
        summary[f"acc_{label}"] = ci.estimate
        print(f"  {label:18} train={len(train):6d} test={len(correct):6d}  "
              f"acc={ci.estimate:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]")
    result = kruskal_wallis(groups)
    summary["regime_kruskal_p"] = result.p_value
    print(f"  {result}")
    return summary


def main() -> None:
    header("UNIVERSE SCALING: CORE 20 AGAINST THE S&P 100")
    summaries = {}
    directory = os.path.dirname(os.path.abspath(__file__))

    for universe, tickers in UNIVERSES.items():
        header(f"{universe.upper()} ({len(tickers)} tickers listed)")
        pooled, prices, excluded = build_dataset(tickers)
        # Sequence construction for the LSTM needs each ticker's rows
        # contiguous and in date order; the purged split works on dates.
        data = pooled.sort_values(["ticker", "date"]).reset_index(drop=True)
        for ticker, reason in excluded.items():
            print(f"  Excluded {ticker}: {reason}")
        print(f"  {len(prices)} tickers, {len(data)} labelled rows, "
              f"{data['date'].min().date()} to {data['date'].max().date()}")
        print(f"  Class balance: "
              f"{(data['target'].value_counts(normalize=True) * 100).round(1).to_dict()}")

        summary = {"tickers_used": len(prices), "rows": len(data),
                   "excluded": ", ".join(excluded) or "none"}

        section("1. Leakage: random against purged chronological split")
        summary.update(leakage(pooled))

        section("2. Classifiers on the purged chronological split")
        table, tests = classifiers(data)
        summary.update(tests)
        for _, row in table.iterrows():
            key = row["model"].lower().replace(" ", "_").replace(":", "")
            summary[f"acc_{key}"] = row["accuracy"]
            summary[f"f1_macro_{key}"] = row["f1_macro"]
        lr = table[table["model"] == "Logistic Regression"].iloc[0]
        summary["lr_acc_ci"] = f"[{lr['acc_ci_lower']:.3f}, {lr['acc_ci_upper']:.3f}]"
        summary["lr_buy_recall"] = lr["buy_recall"]
        summary["n_test_comparable"] = int(lr["n_test"])

        section("3. Walk-forward backtest (logistic regression)")
        frame, backtest_summary = backtest(prices)
        summary.update(backtest_summary)

        section("4. Accuracy by market regime")
        summary.update(regimes(data))

        slug = "core20" if universe == "Core 20" else "sp100"
        table.to_csv(os.path.join(directory, f"exp07_classifiers_{slug}.csv"),
                     index=False)
        frame.to_csv(os.path.join(directory, f"exp07_backtest_{slug}.csv"),
                     index=False)
        summaries[universe] = summary

    header("SUMMARY")
    combined = pd.DataFrame(summaries)
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(combined.to_string())
    combined.to_csv(os.path.join(directory, "exp07_summary.csv"))
    print("\nWritten: exp07_summary.csv, exp07_classifiers_*.csv, "
          "exp07_backtest_*.csv")


if __name__ == "__main__":
    main()
