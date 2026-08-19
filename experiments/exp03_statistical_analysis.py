"""
Experiment 3: statistical inference on model and strategy performance.

Converts the project's point estimates into interval estimates and hypothesis
tests, answering questions the raw numbers cannot:

  1. Are the three classifiers actually different, or is the ranking noise?
     -> McNemar's paired test on the shared test set.
  2. Does the best model beat a naive baseline by a distinguishable margin?
     -> McNemar against the majority-class and random baselines.
  3. Is the backtest win rate distinguishable from a coin flip?
     -> Wilson interval and exact binomial test.
  4. Is the Sharpe ratio distinguishable from zero?
     -> Lo (2002) confidence interval.
  5. Does performance differ across market regimes?
     -> One-way ANOVA and Kruskal-Wallis across regime-partitioned returns.
  6. How much data would be needed to support the win-rate claim?
     -> Power analysis.

Run: python experiments/exp03_statistical_analysis.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.backtester import BacktestConfig, run_backtest
from src.evaluation import (
    DEFAULT_LABEL_HORIZON,
    purged_chronological_split,
    score_predictions,
)
from src.features import compute_features, compute_target_labels
from src.stock_universe import TICKERS
from src.statistics_tests import (
    kruskal_wallis,
    mcnemar_test,
    mean_return_test,
    one_way_anova,
    proportion_test,
    required_sample_size,
    sharpe_ratio_interval,
    tukey_posthoc,
    wilson_interval,
)

FEATURES = ["close_price", "daily_return", "ma_5", "ma_20", "ma_50",
            "volatility", "volume", "RSI", "MACD"]
SEED = 42

#: Market regimes over the study period, used for the regime comparison.
REGIMES = {
    "2020 crash & recovery": ("2020-03-01", "2020-12-31"),
    "2021 bull market": ("2021-01-01", "2021-12-31"),
    "2022 bear market": ("2022-01-01", "2022-12-31"),
    "2023-24 recovery": ("2023-01-01", "2024-11-30"),
}


def header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def section(text: str) -> None:
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


#: Study period, fixed so every part of the analysis sees identical data.
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"


def fetch_prices(ticker: str) -> pd.DataFrame:
    """Download one ticker's daily prices with flattened columns.

    Every download in this script goes through here so that the classifier
    analysis and the backtest cannot silently disagree about the data. In
    particular auto_adjust is pinned rather than left to the yfinance default,
    which has changed between releases and would otherwise make results
    irreproducible across environments.

    Args:
        ticker: Exchange symbol.

    Returns:
        DataFrame of daily OHLCV data with single-level column names.
    """
    import yfinance as yf

    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_pooled_data() -> pd.DataFrame:
    """Download prices and build the pooled feature dataset with dates."""
    frames = []
    for ticker in TICKERS:
        df = fetch_prices(ticker)
        if len(df) < 200:
            continue
        feats = compute_features(df)
        labels = compute_target_labels(df)
        shared = feats.index.intersection(labels.index)
        part = feats.loc[shared].copy()
        part["target"] = labels.loc[shared, "target"]
        part["date"] = shared
        part["ticker"] = ticker
        frames.append(part)

    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def analyse_classifiers(data: pd.DataFrame) -> None:
    """Compare classifiers with interval estimates and paired tests."""
    header("PART 1: CLASSIFIER COMPARISON")

    split = purged_chronological_split(
        data["date"], test_size=0.2, label_horizon=DEFAULT_LABEL_HORIZON)
    train, test = data.iloc[split.train_idx], data.iloc[split.test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test = scaler.transform(test[FEATURES])
    y_train, y_test = train["target"], test["target"]

    print(f"Purged chronological split: train n={len(train)}, test n={len(test)}")
    print(f"Test block begins {split.split_date.date()}; "
          f"{split.n_purged} rows purged.")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100,
                                               random_state=SEED, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "Baseline-Majority": DummyClassifier(strategy="most_frequent"),
        "Baseline-Random": DummyClassifier(strategy="stratified",
                                           random_state=SEED),
    }

    predictions = {}
    section("Accuracy with 95% Wilson confidence intervals")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds

        scores = score_predictions(y_test, preds, name)
        n_correct = int((preds == y_test.to_numpy()).sum())
        ci = wilson_interval(n_correct, len(y_test))
        print(f"  {name:20} acc={ci.estimate:.4f} "
              f"[{ci.lower:.4f}, {ci.upper:.4f}]  F1(w)={scores.f1_weighted:.4f}")

    section("McNemar paired tests: candidate models against each other")
    print("H0: the two models have equal error rates on this test set.\n")
    candidates = ["LogisticRegression", "RandomForest", "DecisionTree"]
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            result = mcnemar_test(y_test, predictions[a], predictions[b], a, b)
            print(f"  {result}\n")

    section("McNemar paired tests: best model against naive baselines")
    print("H0: the model has the same error rate as the baseline.\n")
    for baseline in ["Baseline-Majority", "Baseline-Random"]:
        result = mcnemar_test(y_test, predictions["LogisticRegression"],
                              predictions[baseline],
                              "LogisticRegression", baseline)
        print(f"  {result}\n")

    return predictions, y_test


def analyse_backtest() -> None:
    """Interval estimates and hypothesis tests on backtest performance."""
    header("PART 2: BACKTEST PERFORMANCE INFERENCE")

    print(f"Running walk-forward backtest on all {len(TICKERS)} tickers...\n")
    config = BacktestConfig(train_window_days=500, test_window_days=60,
                            step_days=60, model_type="lr")

    per_ticker = {}
    failed = {}
    for ticker in TICKERS:
        df = fetch_prices(ticker)
        try:
            per_ticker[ticker] = run_backtest(df, config)
        except ValueError as exc:
            # Recorded rather than silently dropped: a ticker excluded from the
            # sample must be visible, otherwise the pooled statistics below
            # describe a different population than the stated universe.
            failed[ticker] = str(exc)

    if failed:
        section("Tickers excluded because the backtest could not run")
        for ticker, reason in failed.items():
            print(f"  {ticker:8} {reason}")

    section("Per-ticker results with win-rate confidence intervals")
    print("A ticker with zero trades is reported, not dropped: it means the")
    print("model never issued a Buy signal, so the strategy held cash. Those")
    print("rows carry no win rate but they do carry a return, and excluding")
    print("them would flatter the strategy.\n")
    print(f"{'Ticker':8}{'Trades':>7}{'Wins':>6}{'WinRate':>9}"
          f"{'95% CI':>20}{'Strategy%':>11}{'Bench%':>9}")
    rows = []
    no_trade_tickers = []
    for ticker, result in per_ticker.items():
        wins = result.profitable_trades
        n = result.total_trades
        if n == 0:
            no_trade_tickers.append(ticker)
            print(f"{ticker:8}{n:>7}{'-':>6}{'n/a':>9}"
                  f"{'no Buy signals':>20}"
                  f"{result.total_return_pct:>11.2f}"
                  f"{result.benchmark_return_pct:>9.2f}")
        else:
            ci = wilson_interval(wins, n)
            print(f"{ticker:8}{n:>7}{wins:>6}{ci.estimate:>8.1%}"
                  f"   [{ci.lower:.3f}, {ci.upper:.3f}]"
                  f"{result.total_return_pct:>11.2f}"
                  f"{result.benchmark_return_pct:>9.2f}")
        rows.append({
            "ticker": ticker,
            "trades": n,
            "wins": wins,
            "win_rate": (wins / n) if n else float("nan"),
            "strategy_return": result.total_return_pct,
            "benchmark_return": result.benchmark_return_pct,
            "sharpe": result.sharpe_ratio,
            "excess": result.total_return_pct - result.benchmark_return_pct,
        })

    frame = pd.DataFrame(rows)
    traded = frame[frame["trades"] > 0]

    print(f"\n  Backtested {len(frame)} of {len(TICKERS)} tickers; "
          f"{len(traded)} produced at least one trade.")
    if no_trade_tickers:
        print(f"  Zero-trade tickers: {', '.join(no_trade_tickers)}")
        print("  These are held in the return comparison below but cannot")
        print("  contribute to the win rate, which is undefined at n=0.")

    section("Pooled win rate across tickers that traded")
    total_wins = int(traded["wins"].sum())
    total_trades = int(traded["trades"].sum())
    pooled_ci = wilson_interval(total_wins, total_trades)
    print(f"  Pooled: {total_wins}/{total_trades} = {pooled_ci.estimate:.1%}")
    print(f"  95% CI: [{pooled_ci.lower:.4f}, {pooled_ci.upper:.4f}]")
    print(f"  {proportion_test(total_wins, total_trades, 0.5)}")
    verdict = ("distinguishable from chance"
               if pooled_ci.excludes(0.5) else "NOT distinguishable from chance")
    print(f"  Interpretation: pooled win rate is {verdict}.")
    print("\n  Caveat: this pools trades the model chose to take. It measures")
    print("  the quality of the signals acted upon, not the profitability of")
    print("  the strategy as a whole, which the return comparison addresses.")

    section("Single-ticker illustration: AAPL (the figure quoted in the report)")
    if "AAPL" in per_ticker:
        aapl = per_ticker["AAPL"]
        ci = wilson_interval(aapl.profitable_trades, aapl.total_trades)
        print(f"  Win rate: {ci}")
        print(f"  {proportion_test(aapl.profitable_trades, aapl.total_trades, 0.5)}")
        print(f"  The interval spans {ci.width:.1%}, so a single-ticker win rate")
        print("  at this sample size carries almost no information.")

    section("Strategy versus benchmark returns across tickers")
    print(f"  Tickers included     : {len(frame)} (all that ran, including "
          f"any that never traded)")
    print(f"  Mean strategy return : {frame['strategy_return'].mean():+.2f}%")
    print(f"  Mean benchmark return: {frame['benchmark_return'].mean():+.2f}%")
    print(f"  Mean excess return   : {frame['excess'].mean():+.2f}%")
    print(f"\n  {mean_return_test(frame['excess'].to_numpy(), 0.0)}")
    print("\n  H0 states the strategy matches buy-and-hold. Rejection with a")
    print("  negative mean excess return means it reliably underperforms.")
    print("\n  This is the headline result. It is measured per ticker and")
    print("  paired against that ticker's own buy-and-hold return, so it is")
    print("  not confounded by which stocks happened to rise.")

    section("Sharpe ratio interval estimates")
    for ticker in list(per_ticker)[:5]:
        curve = per_ticker[ticker].equity_curve
        if len(curve) < 10:
            continue
        returns = pd.Series(curve).pct_change().dropna().to_numpy()
        try:
            ci = sharpe_ratio_interval(returns)
            excl = "excludes 0" if ci.excludes(0.0) else "includes 0"
            print(f"  {ticker:8} Sharpe={ci.estimate:+.3f} "
                  f"[{ci.lower:+.3f}, {ci.upper:+.3f}]  ({excl})")
        except ValueError as exc:
            print(f"  {ticker:8} skipped: {exc}")
    print("\n  Intervals that include zero mean the risk-adjusted return is")
    print("  not distinguishable from no skill at all.")

    return frame, per_ticker


def analyse_regimes(data: pd.DataFrame) -> None:
    """Test whether predictive accuracy differs across market regimes."""
    header("PART 3: MARKET REGIME COMPARISON (ANOVA)")

    print("For each regime a fresh model is trained on all data preceding it,")
    print("purged by the label horizon. This keeps every evaluation strictly")
    print("out-of-sample while allowing more regimes to be compared than a")
    print("single fixed split would permit. The earliest regime cannot be")
    print("tested because no prior data exists to train on.\n")

    # Trading days needed before a regime to train a model at all.
    min_train_rows = 1000
    horizon_gap = pd.Timedelta(days=DEFAULT_LABEL_HORIZON * 2)

    groups: dict[str, list[float]] = {}
    for label, (start, end) in REGIMES.items():
        regime_start = pd.Timestamp(start)
        train = data[data["date"] < regime_start - horizon_gap]
        window = data[(data["date"] >= regime_start)
                      & (data["date"] <= pd.Timestamp(end))]

        if len(train) < min_train_rows:
            print(f"  {label:24} skipped (only {len(train)} prior rows "
                  f"available to train on)")
            continue
        if len(window) < 50:
            print(f"  {label:24} skipped ({len(window)} rows in regime)")
            continue

        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, random_state=SEED)
        model.fit(scaler.fit_transform(train[FEATURES]), train["target"])

        preds = model.predict(scaler.transform(window[FEATURES]))
        correct = (preds == window["target"].to_numpy()).astype(float)
        groups[label] = correct.tolist()

        ci = wilson_interval(int(correct.sum()), len(correct))
        print(f"  {label:24} train={len(train):5d} test={len(correct):5d}  "
              f"acc={ci.estimate:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]")

    if len(groups) < 2:
        print("\n  Too few regimes with out-of-sample data for a group test.")
        return

    section("Do accuracy rates differ across regimes?")
    print("H0: mean accuracy is equal in every regime.\n")
    print(f"  {one_way_anova(groups)}\n")
    print(f"  {kruskal_wallis(groups)}\n")
    print("  Accuracy here is a 0/1 indicator per row, which is binary rather")
    print("  than normal, so the Kruskal-Wallis rank test is the more")
    print("  defensible of the two and should be given precedence.")

    section("Pairwise regime comparisons (Tukey HSD, family-wise corrected)")
    for result in tukey_posthoc(groups):
        marker = "*" if result.significant() else " "
        print(f" {marker} {result.name}: p={result.p_value:.4f}, {result.detail}")


def _power_effect(rate: float) -> tuple[float, str]:
    """Pull a proportion away from the 0 and 1 boundaries for power maths.

    The normal-approximation sample size formula divides by the variance
    p(1-p), which collapses to zero at a boundary proportion. A win rate of
    exactly 100% from a handful of trades therefore produces a nonsensically
    small required sample size. Capping the effect size keeps the answer
    conservative and is flagged in the output rather than hidden.

    Args:
        rate: Observed proportion.

    Returns:
        The proportion to use, and a note to append if it was adjusted.
    """
    if rate >= 0.99:
        return 0.95, "*"
    if rate <= 0.01:
        return 0.05, "*"
    return rate, ""


def analyse_power(frame: pd.DataFrame, n_test: int, accuracy: float) -> None:
    """State the sample sizes the project's claims would require.

    Figures are taken from the run that has just completed rather than
    hardcoded, so this section cannot drift out of step with the results
    above when the universe or date range changes.

    Args:
        frame: Per-ticker backtest results from analyse_backtest.
        n_test: Size of the classification test set.
        accuracy: Observed accuracy of the best classifier.
    """
    header("PART 4: STATISTICAL POWER")

    traded = frame[frame["trades"] > 0]
    pooled_n = int(traded["trades"].sum())
    pooled_rate = float(traded["wins"].sum()) / pooled_n if pooled_n else 0.0

    # The single best-looking ticker, which is the figure most likely to be
    # quoted selectively and therefore the one most in need of a power check.
    best = traded.loc[traded["win_rate"].idxmax()]
    best_n = int(best["trades"])
    best_rate = float(best["win_rate"])

    best_ci = wilson_interval(int(best["wins"]), best_n)
    print(f"The most favourable single-ticker win rate in this run is "
          f"{best['ticker']} at")
    print(f"{best_rate:.1%} from {best_n} trades, 95% CI "
          f"[{best_ci.lower:.3f}, {best_ci.upper:.3f}], an interval "
          f"{best_ci.width:.1%} wide.")
    print("Headline figures like that are what this section is guarding "
          "against.\n")

    print(f"{'Claim':44}{'Required n':>12}{'Have':>7}{'Adequate':>10}")
    scenarios = [
        (f"Win rate {best_rate:.1%} differs from 50% ({best['ticker']})",
         0.50, best_rate, best_n),
        (f"Pooled win rate {pooled_rate:.1%} differs from 50%",
         0.50, pooled_rate, pooled_n),
        ("Win rate 55% differs from 50%", 0.50, 0.55, pooled_n),
        (f"Accuracy {accuracy:.1%} differs from 33.3%", 0.333, accuracy, n_test),
    ]
    capped = False
    for label, p_null, p_alt, have in scenarios:
        effect, note = _power_effect(p_alt)
        capped = capped or bool(note)
        need = required_sample_size(p_null, effect)
        ok = "yes" if have >= need else "NO"
        print(f"{label + note:44}{need:>12}{have:>7}{ok:>10}")

    if capped:
        print("\n  * Effect size capped at 95% because the sample size formula")
        print("    divides by p(1-p), which is zero at a boundary proportion.")
        print("    The true requirement is larger than the figure shown.")

    print(f"\n  The classification test set (n={n_test}) is large enough to")
    print("  detect the observed accuracy margin, so that result is properly")
    print(f"  powered. The pooled backtest (n={pooled_n} trades) supports the")
    print(f"  {pooled_rate:.1%} win rate, but no individual ticker does: "
          f"{best['ticker']}")
    print(f"  has only {best_n} trades. This is why the report quotes the "
          "pooled")
    print("  figure and treats per-ticker rates as illustrative only.")
    print("  Detecting a genuine but small edge near 55% would need several")
    print("  hundred trades, which the study period cannot supply.")


def main() -> None:
    header("STATISTICAL ANALYSIS OF MODEL AND STRATEGY PERFORMANCE")
    print("All tests use a 5% significance level unless stated otherwise.")

    print("\nLoading data...")
    data = load_pooled_data()
    print(f"Loaded {len(data)} rows, {data['ticker'].nunique()} tickers, "
          f"{data['date'].min().date()} to {data['date'].max().date()}")

    predictions, y_test = analyse_classifiers(data)
    best_accuracy = float(
        (predictions["LogisticRegression"] == y_test.to_numpy()).mean())

    frame, _ = analyse_backtest()
    analyse_regimes(data)
    analyse_power(frame, len(y_test), best_accuracy)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "exp03_backtest_by_ticker.csv")
    frame.to_csv(out, index=False)
    header("COMPLETE")
    print(f"Per-ticker backtest results written to {os.path.basename(out)}")


if __name__ == "__main__":
    main()
