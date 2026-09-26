"""Backtesting Module for the AI Stock Recommendation System.

Evaluates the advisor bot's recommendations against historical data using
walk-forward (temporal) validation. This provides evidence that the system's
advice generates value compared to a baseline (buy-and-hold) strategy.

The backtester:
1. Splits historical data into rolling train/test windows
2. Trains a model on each training window
3. Generates signals (Buy/Hold/Avoid) on the test window
4. Simulates portfolio performance following those signals
5. Compares against a buy-and-hold benchmark

Metrics reported:
- Total return (strategy vs. benchmark)
- Annualised return
- Sharpe ratio
- Maximum drawdown
- Win rate (% of trades that were profitable)
- Signal accuracy (% of signals correctly predicting direction)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.features import compute_features, compute_target_labels


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        train_window_days: Number of trading days for the training window.
        test_window_days: Number of trading days for each test window.
        step_days: How many days to advance between windows.
        initial_capital: Starting capital for the simulation.
        model_type: Which model to use ('lr', 'rf', 'dt'). Defaults to 'lr'
            because logistic regression outperformed the tree ensembles once
            the training split was corrected for temporal leakage
            (see experiments/exp02_results.txt).
        transaction_cost_pct: Cost per trade as a fraction (e.g., 0.001 = 0.1%).
        hold_cost_model: How costs are charged on the half-size position a
            Hold signal takes. "on_change" (the default) charges entry only
            when the half position is opened and exit only when it is closed,
            which is how a real position incurs costs. "per_day" reproduces
            the original behaviour, which treated every Hold day as a separate
            round trip and so charged a run of consecutive Hold days entry and
            exit costs every day; it overstated costs and is kept only so the
            figures in the first report draft can be reproduced.
        buy_threshold: Forward 30-day return above which a row is labelled
            Buy when training each window (see compute_target_labels).
    """
    train_window_days: int = 500
    test_window_days: int = 60
    step_days: int = 60
    initial_capital: float = 10000.0
    model_type: str = "lr"
    transaction_cost_pct: float = 0.001
    hold_cost_model: str = "on_change"
    buy_threshold: float = 0.10


@dataclass
class TradeRecord:
    """Record of a single trade executed during backtesting.

    random_entry_win_prob is the share of random entries in the same test
    window, held for the same number of days and charged the same costs, that
    would have been profitable. In a rising market most long positions make
    money, so 50% is too low a bar for "better than chance"; summing this
    field over trades gives the number of wins chance alone would produce.
    """
    date: str
    ticker: str
    signal: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    random_entry_win_prob: float = float("nan")


def random_entry_win_probability(prices: pd.Series, holding_days: int,
                                 round_trip_cost: float) -> float:
    """Share of entries in `prices` that profit over `holding_days` after costs.

    Args:
        prices: Closing prices of one test window, in date order.
        holding_days: Days each position is held.
        round_trip_cost: Entry plus exit cost as a fraction of price.

    Returns:
        Fraction of possible entry days whose position would have ended in
        profit. A zero-day holding can only lose its costs, so it returns 0.
    """
    values = np.asarray(prices, dtype=float)
    if holding_days <= 0 or holding_days >= len(values):
        return 0.0
    returns = (values[holding_days:] - values[:-holding_days]) / values[:-holding_days]
    return float(((returns - round_trip_cost) > 0).mean())


@dataclass
class BacktestResult:
    """Results from a single backtest run.

    Contains both summary metrics and detailed trade records
    for analysis and reporting.
    """
    # Summary metrics
    total_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    annualised_return_pct: float = 0.0
    annualised_benchmark_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    signal_accuracy_pct: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0

    # Detailed data
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    benchmark_curve: list[float] = field(default_factory=list)
    window_results: list[dict] = field(default_factory=list)

    # Configuration used
    config: Optional[BacktestConfig] = None


def _build_model(model_type: str):
    """Create a fresh model instance based on type string."""
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    elif model_type == "lr":
        return LogisticRegression(
            max_iter=1000, random_state=42
        )
    elif model_type == "dt":
        return DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


FEATURE_COLUMNS = [
    "close_price", "daily_return", "ma_5", "ma_20", "ma_50",
    "volatility", "volume", "RSI", "MACD",
]


def run_backtest(
    stock_data: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Run a walk-forward backtest on historical stock data.

    The walk-forward approach prevents look-ahead bias by training only on
    data that would have been available at each decision point.

    Process for each window:
    1. Train model on [t - train_window : t]
    2. Generate signals on [t : t + test_window]
    3. Simulate following signals: Buy → hold position, Avoid → exit/skip
    4. Record performance vs. buy-and-hold benchmark

    Args:
        stock_data: DataFrame with at least 'Close' (or 'Adj Close') and
                    'Volume' columns. Must have enough data for at least
                    one train + test window.
        config: Backtest configuration. Uses defaults if None.

    Returns:
        BacktestResult with summary metrics and detailed trade records.

    Raises:
        ValueError: If stock_data is too short for even one backtest window.
    """
    if config is None:
        config = BacktestConfig()
    if config.hold_cost_model not in ("per_day", "on_change"):
        raise ValueError(
            f"Unknown hold_cost_model: {config.hold_cost_model}")

    # Handle MultiIndex columns from yfinance
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = stock_data.copy()
        stock_data.columns = stock_data.columns.get_level_values(0)

    # Compute features for the full dataset
    features_df = compute_features(stock_data)
    labels_df = compute_target_labels(
        stock_data, buy_threshold=config.buy_threshold)

    # Align features and labels on common index
    common_idx = features_df.index.intersection(labels_df.index)
    features_df = features_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    total_rows = len(features_df)
    min_required = config.train_window_days + config.test_window_days

    if total_rows < min_required:
        raise ValueError(
            f"Insufficient data for backtest: {total_rows} rows available, "
            f"but need at least {min_required} "
            f"(train={config.train_window_days} + test={config.test_window_days})"
        )

    # Extract close prices aligned with features for return calculation
    close_col = "Adj Close" if "Adj Close" in stock_data.columns else "Close"
    close_prices = stock_data[close_col].loc[common_idx]

    # Walk-forward loop
    equity = config.initial_capital
    benchmark_start_price = close_prices.iloc[config.train_window_days]
    benchmark_equity = config.initial_capital

    equity_curve = []
    benchmark_curve = []
    all_trades = []
    window_results = []
    all_signals = []
    all_actuals = []

    start_idx = 0
    while start_idx + min_required <= total_rows:
        train_end = start_idx + config.train_window_days
        test_end = min(train_end + config.test_window_days, total_rows)

        # Split data
        X_train = features_df.iloc[start_idx:train_end][FEATURE_COLUMNS]
        y_train = labels_df.iloc[start_idx:train_end]["target"]
        X_test = features_df.iloc[train_end:test_end][FEATURE_COLUMNS]
        y_test = labels_df.iloc[train_end:test_end]["target"]
        test_prices = close_prices.iloc[train_end:test_end]

        if len(X_test) == 0:
            break

        # Train model
        model = _build_model(config.model_type)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)

        # Generate predictions
        predictions = model.predict(X_test_scaled)

        # Track signal accuracy
        all_signals.extend(predictions)
        all_actuals.extend(y_test.values)

        # Simulate trading for this window
        window_return = 0.0
        window_trades = 0
        window_wins = 0

        i = 0
        holding_half = False  # a Hold half position is open (on_change only)
        while i < len(predictions):
            signal = predictions[i]
            entry_price = test_prices.iloc[i]

            if signal == "Buy":
                # Hold until signal changes or window ends
                j = i + 1
                while j < len(predictions) and predictions[j] == "Buy":
                    j += 1
                exit_idx = min(j, len(test_prices) - 1)
                exit_price = test_prices.iloc[exit_idx]

                trade_return = (exit_price - entry_price) / entry_price
                trade_return -= config.transaction_cost_pct * 2  # entry + exit

                trade = TradeRecord(
                    date=str(test_prices.index[i]),
                    ticker="",  # filled by caller if needed
                    signal="Buy",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=trade_return * 100,
                    holding_days=exit_idx - i,
                    random_entry_win_prob=random_entry_win_probability(
                        test_prices, exit_idx - i,
                        config.transaction_cost_pct * 2),
                )
                all_trades.append(trade)
                window_trades += 1
                if trade_return > 0:
                    window_wins += 1

                equity *= (1 + trade_return)
                window_return += trade_return
                i = exit_idx + 1

            elif signal == "Hold":
                # Hold for one period, small exposure
                if i + 1 < len(test_prices):
                    next_price = test_prices.iloc[i + 1]
                    trade_return = (next_price - entry_price) / entry_price
                    if config.hold_cost_model == "per_day":
                        trade_return -= config.transaction_cost_pct * 2
                    else:
                        # The half position stays open into tomorrow only if
                        # tomorrow is also Hold and can itself earn a return.
                        stays_open = (i + 2 < len(test_prices)
                                      and predictions[i + 1] == "Hold")
                        if not holding_half:
                            trade_return -= config.transaction_cost_pct
                        if not stays_open:
                            trade_return -= config.transaction_cost_pct
                        holding_half = stays_open
                    equity *= (1 + trade_return * 0.5)  # half position
                    window_return += trade_return * 0.5
                i += 1

            else:  # Avoid
                # Stay in cash, no action
                i += 1

            equity_curve.append(equity)

        # Benchmark: buy and hold for this test window
        if len(test_prices) >= 2:
            bench_return = (
                (test_prices.iloc[-1] - test_prices.iloc[0]) / test_prices.iloc[0]
            )
            benchmark_equity *= (1 + bench_return)
        benchmark_curve.append(benchmark_equity)

        window_results.append({
            "window_start": start_idx,
            "window_end": test_end,
            "strategy_return_pct": window_return * 100,
            "trades": window_trades,
            "wins": window_wins,
        })

        # Advance to next window
        start_idx += config.step_days

    # Calculate summary metrics
    total_return = (equity - config.initial_capital) / config.initial_capital
    bench_total_return = (
        (benchmark_equity - config.initial_capital) / config.initial_capital
    )

    # Annualise (approximate based on number of trading days used)
    total_days = min(start_idx + min_required, total_rows) - config.train_window_days
    years = total_days / 252.0 if total_days > 0 else 1.0
    ann_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0
    ann_bench = ((1 + bench_total_return) ** (1 / years) - 1) if years > 0 else 0

    # Sharpe ratio (annualised)
    if equity_curve:
        returns_series = pd.Series(equity_curve).pct_change().dropna()
        if len(returns_series) > 1 and returns_series.std() > 0:
            sharpe = (
                returns_series.mean() / returns_series.std()
            ) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    if equity_curve:
        peak = pd.Series(equity_curve).expanding().max()
        drawdown = (pd.Series(equity_curve) - peak) / peak
        max_dd = drawdown.min() * 100
    else:
        max_dd = 0.0

    # Win rate
    total_trades_count = len(all_trades)
    profitable = sum(1 for t in all_trades if t.return_pct > 0)
    win_rate = (profitable / total_trades_count * 100) if total_trades_count > 0 else 0

    # Signal accuracy
    correct_signals = sum(
        1 for s, a in zip(all_signals, all_actuals) if s == a
    )
    signal_acc = (
        correct_signals / len(all_signals) * 100
    ) if all_signals else 0

    return BacktestResult(
        total_return_pct=total_return * 100,
        benchmark_return_pct=bench_total_return * 100,
        annualised_return_pct=ann_return * 100,
        annualised_benchmark_pct=ann_bench * 100,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate_pct=win_rate,
        signal_accuracy_pct=signal_acc,
        total_trades=total_trades_count,
        profitable_trades=profitable,
        trades=all_trades,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        window_results=window_results,
        config=config,
    )


def run_multi_stock_backtest(
    stock_dataframes: dict[str, pd.DataFrame],
    config: Optional[BacktestConfig] = None,
) -> dict:
    """Run backtests across multiple stocks and aggregate results.

    Args:
        stock_dataframes: Dict mapping ticker symbol to its DataFrame.
        config: Backtest configuration. Uses defaults if None.

    Returns:
        Dict with:
        - 'per_stock': Dict of ticker -> BacktestResult
        - 'aggregate': Dict with averaged metrics across all stocks
        - 'summary_table': DataFrame suitable for display
    """
    if config is None:
        config = BacktestConfig()

    results = {}
    for ticker, df in stock_dataframes.items():
        try:
            result = run_backtest(df, config)
            results[ticker] = result
        except ValueError as e:
            # Skip stocks with insufficient data
            print(f"Skipping {ticker}: {e}")
            continue

    if not results:
        return {
            "per_stock": {},
            "aggregate": {},
            "summary_table": pd.DataFrame(),
        }

    # Aggregate metrics
    metrics = [
        "total_return_pct", "benchmark_return_pct",
        "annualised_return_pct", "annualised_benchmark_pct",
        "sharpe_ratio", "max_drawdown_pct",
        "win_rate_pct", "signal_accuracy_pct",
        "total_trades",
    ]

    aggregate = {}
    for m in metrics:
        values = [getattr(r, m) for r in results.values()]
        aggregate[m] = np.mean(values) if values else 0.0

    # Build summary table
    rows = []
    for ticker, r in results.items():
        rows.append({
            "Ticker": ticker,
            "Strategy Return %": f"{r.total_return_pct:.2f}",
            "Benchmark Return %": f"{r.benchmark_return_pct:.2f}",
            "Sharpe Ratio": f"{r.sharpe_ratio:.2f}",
            "Max Drawdown %": f"{r.max_drawdown_pct:.2f}",
            "Win Rate %": f"{r.win_rate_pct:.1f}",
            "Signal Accuracy %": f"{r.signal_accuracy_pct:.1f}",
            "Total Trades": r.total_trades,
        })

    summary_df = pd.DataFrame(rows)

    return {
        "per_stock": results,
        "aggregate": aggregate,
        "summary_table": summary_df,
    }


def format_backtest_report(result: BacktestResult) -> str:
    """Format a backtest result into a human-readable report string.

    Args:
        result: A completed BacktestResult.

    Returns:
        Multi-line string suitable for display or logging.
    """
    lines = [
        "=" * 60,
        "BACKTEST RESULTS",
        "=" * 60,
        f"Strategy Total Return:   {result.total_return_pct:+.2f}%",
        f"Benchmark Total Return:  {result.benchmark_return_pct:+.2f}%",
        f"Excess Return:           {result.total_return_pct - result.benchmark_return_pct:+.2f}%",
        "-" * 40,
        f"Annualised Return:       {result.annualised_return_pct:+.2f}%",
        f"Annualised Benchmark:    {result.annualised_benchmark_pct:+.2f}%",
        f"Sharpe Ratio:            {result.sharpe_ratio:.2f}",
        f"Maximum Drawdown:        {result.max_drawdown_pct:.2f}%",
        "-" * 40,
        f"Total Trades:            {result.total_trades}",
        f"Profitable Trades:       {result.profitable_trades}",
        f"Win Rate:                {result.win_rate_pct:.1f}%",
        f"Signal Accuracy:         {result.signal_accuracy_pct:.1f}%",
        "=" * 60,
    ]
    return "\n".join(lines)
