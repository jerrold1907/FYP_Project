"""
AI-Powered Stock Advisor - Flask Web Application

A financial advisor bot that:
1. Assesses investor risk tolerance via scenario-based questionnaire
2. Automatically analyses a stock universe using the trained ML model
3. Recommends suitable stocks based on risk profile with price charts
4. Provides walk-forward backtesting evaluation

Run with: python app.py
Then open: http://127.0.0.1:5000
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from src.risk_questionnaire import (
    QUESTIONS, QuestionnaireResponse, classify_questionnaire,
    classify_risk_from_score,
)
from src.stock_recommender import StockRecommender
from src.suitability import SuitabilityEngine
from src.validation import VALID_RISK_CATEGORIES
from src.features import compute_features


def _safe_float(v, default=0.0):
    """Convert a value to float, replacing NaN/Inf with a default.

    yfinance can return NaN for missing data points. Python's json module
    serialises NaN as the invalid token `NaN`, which breaks JSON parsing on
    the client. This helper ensures every numeric field in an API response is
    a finite float.
    """
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default
from src.backtester import (
    BacktestConfig, run_backtest, run_multi_stock_backtest,
    format_backtest_report,
)
from src.chatbot import (
    classify_intent, generate_response, summarise_with_llm, verify_summary,
    is_ollama_available, OLLAMA_MODEL,
)
from src.stock_universe import STOCK_INFO, STOCK_NAMES, TICKERS, UNIVERSE, sectors
from src.statistics_tests import sharpe_ratio_interval


def _combined_sharpe(curve_pct):
    """Sharpe ratio of a cumulative-return curve (in %) with a 95% interval.

    Lo's (2002) standard error applies to the per-period ratio, so the interval
    is formed on daily returns and then annualised; applying it to an already
    annualised ratio makes the interval over ten times too narrow.

    Returns:
        An Interval, or None when the curve is too short or flat for a ratio.
    """
    values = 1 + np.asarray(curve_pct, dtype=float) / 100
    if len(values) < 4:
        return None
    try:
        return sharpe_ratio_interval(values[1:] / values[:-1] - 1)
    except ValueError:
        return None


app = Flask(__name__, template_folder="templates", static_folder="static")

# The advisor analyses every stock in the shared universe (see
# src/stock_universe.py). Kept as a name mapping for the recommendation loop.
STOCK_UNIVERSE = STOCK_NAMES

# Load model once at startup
recommender = None
try:
    recommender = StockRecommender(
        model_path="models/stock_model.pkl",
        scaler_path="models/scaler.pkl",
    )
    print("Stock model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load stock model: {e}")

engine = SuitabilityEngine()

# Detect the optional local LLM once at startup. The chatbot works without it,
# using deterministic template responses instead.
OLLAMA_READY = is_ollama_available()
if OLLAMA_READY:
    print(f"Ollama detected — chatbot responses will be polished by {OLLAMA_MODEL}.")
else:
    print("Ollama not detected — chatbot will use template responses.")


@app.route("/")
def index():
    """Main page with the questionnaire."""
    return render_template("index.html", questions=QUESTIONS)


@app.route("/compare")
def compare_page():
    """Stock comparison page.

    The universe is passed to the template so the dropdowns cannot drift out of
    step with the stocks the advisor actually analyses.
    """
    return render_template("compare.html", universe=UNIVERSE,
                           sectors=sectors())


@app.route("/backtest")
def backtest_page():
    """Backtesting evaluation page."""
    return render_template("backtest.html")


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """API: Takes questionnaire answers, analyses all stocks, returns recommendations.

    Flow:
    1. Classify user risk profile from questionnaire
    2. Download recent data for each stock in universe
    3. Compute features and predict signal (Buy/Hold/Avoid) for each
    4. Run through suitability engine for user's risk category
    5. Return all stocks with their signals, suitability, and price history
    """
    data = request.get_json()

    # 1. Process questionnaire
    answers = data.get("answers", {})
    response = QuestionnaireResponse(answers=answers)

    if not response.is_complete:
        return jsonify({"error": "Please answer all 5 questions."}), 400

    risk_result = classify_questionnaire(response)
    risk_category = risk_result["category"]

    if not recommender:
        return jsonify({"error": "ML model not loaded."}), 500

    # 2. Download and analyse each stock
    import yfinance as yf

    stock_results = []
    for ticker, name in STOCK_UNIVERSE.items():
        try:
            # Download last 6 months of data
            df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Drop rows where Close is NaN (e.g. today before market close)
            if "Close" in df.columns:
                df = df.dropna(subset=["Close"])

            if len(df) < 80:
                continue

            # Compute features
            features_df = compute_features(df)
            if len(features_df) == 0:
                continue

        # Get latest features (most recent trading day)
            latest = features_df.iloc[-1]
            feature_dict = {
                "close_price": _safe_float(latest["close_price"]),
                "daily_return": _safe_float(latest["daily_return"]),
                "ma_5": _safe_float(latest["ma_5"]),
                "ma_20": _safe_float(latest["ma_20"]),
                "ma_50": _safe_float(latest["ma_50"]),
                "volatility": _safe_float(latest["volatility"]),
                "volume": _safe_float(latest["volume"]),
                "RSI": _safe_float(latest["RSI"]),
                "MACD": _safe_float(latest["MACD"]),
            }

            # 3. Predict signal
            signal = recommender.predict(feature_dict)

            # 4. Get suitability for this user
            suitability = engine.recommend(risk_category, signal)

            # Price history for chart (last 90 days)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            price_history = df[close_col].tail(90).dropna()
            prices = [_safe_float(p) for p in price_history.values]
            dates = [d.strftime("%Y-%m-%d") for d in price_history.index]

            # Key indicators for display
            current_price = _safe_float(df[close_col].iloc[-1])
            price_change_30d = _safe_float(
                (df[close_col].iloc[-1] - df[close_col].iloc[-22]) /
                df[close_col].iloc[-22] * 100
            ) if len(df) > 22 else 0.0

            stock_results.append({
                "ticker": ticker,
                "name": name,
                "signal": signal,
                "suitability": suitability,
                "current_price": round(current_price, 2),
                "price_change_30d": round(price_change_30d, 2),
                "rsi": round(_safe_float(latest["RSI"]), 1),
                "macd": round(_safe_float(latest["MACD"]), 2),
                "volatility": round(_safe_float(latest["volatility"]) * 100, 2),
                "prices": prices,
                "dates": dates,
            })

        except Exception as e:
            print(f"Error analysing {ticker}: {e}")
            continue

    # Sort: Suitable first, then Use Caution, then Not Suitable
    rating_order = {"Suitable": 0, "Use Caution": 1, "Not Suitable": 2}
    stock_results.sort(
        key=lambda x: rating_order.get(x["suitability"].get("rating", ""), 3)
    )

    return jsonify({
        "risk_profile": risk_result,
        "stocks": stock_results,
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """API endpoint: chatbot interaction about stocks and recommendations.

    Two-stage pipeline:
      1. Deterministic stage — intent classification and data retrieval build a
         factual answer from the ML model output and suitability engine.
      2. Presentation stage (optional) — a local Ollama model rephrases that
         answer conversationally without altering any figures.
    """
    data = request.get_json()
    message = data.get("message", "").strip()
    stock_data = data.get("stock_data", [])

    # A risk category is only valid if it came from a completed questionnaire.
    # Never fall back to a default — inventing a risk profile would make the
    # advice unsound. None signals "assessment not yet completed".
    risk_category = data.get("risk_category") or None
    if risk_category not in VALID_RISK_CATEGORIES:
        risk_category = None

    if not message:
        return jsonify({"error": "No message provided."}), 400

    # Stage 1: deterministic, data-grounded answer
    intent, tickers = classify_intent(message)
    factual = generate_response(
        intent=intent,
        tickers=tickers,
        stock_data=stock_data,
        risk_category=risk_category,
    )

    # Stage 2: optional LLM summary shown above the structured answer.
    # The factual content is never replaced, so figures stay auditable, and any
    # summary that contradicts the model output is discarded.
    summary = None
    if OLLAMA_READY:
        candidate = summarise_with_llm(message, factual)
        if candidate and verify_summary(candidate, stock_data):
            summary = candidate

    return jsonify({
        "response": factual,
        "summary": summary,
        "intent": intent,
        "tickers_detected": tickers,
        "llm_used": summary is not None,
        "profile_assessed": risk_category is not None,
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """API endpoint: compare two stocks side by side with chart data."""
    data = request.get_json()
    tickers = data.get("tickers", [])

    if len(tickers) < 2:
        return jsonify({"error": "Please select at least 2 stocks to compare."}), 400

    if not recommender:
        return jsonify({"error": "Model not loaded."}), 500

    import yfinance as yf

    results = []
    for ticker in tickers[:3]:  # Max 3 stocks
        try:
            df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" in df.columns:
                df = df.dropna(subset=["Close"])
            if len(df) < 80:
                continue

            features_df = compute_features(df)
            if len(features_df) == 0:
                continue

            latest = features_df.iloc[-1]
            feature_dict = {col: _safe_float(latest[col]) for col in features_df.columns}
            signal = recommender.predict(feature_dict)

            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            price_history = df[close_col].tail(90).dropna()
            prices = [_safe_float(p) for p in price_history.values]
            dates = [d.strftime("%Y-%m-%d") for d in price_history.index]

            current_price = _safe_float(df[close_col].iloc[-1])
            price_change = _safe_float(
                (df[close_col].iloc[-1] - df[close_col].iloc[-22]) /
                df[close_col].iloc[-22] * 100
            ) if len(df) > 22 else 0.0

            results.append({
                "ticker": ticker,
                "name": STOCK_INFO.get(ticker, {}).get("name", ticker),
                "sector": STOCK_INFO.get(ticker, {}).get("sector", "Unknown"),
                "signal": signal,
                "current_price": round(current_price, 2),
                "price_change_30d": round(price_change, 2),
                "rsi": round(_safe_float(latest["RSI"]), 1),
                "macd": round(_safe_float(latest["MACD"]), 2),
                "volatility": round(_safe_float(latest["volatility"]) * 100, 2),
                "prices": prices,
                "dates": dates,
            })
        except Exception as e:
            print(f"Compare error for {ticker}: {e}")
            continue

    if len(results) < 2:
        return jsonify({"error": "Could not retrieve data for enough stocks."}), 400

    return jsonify({"stocks": results})


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """API endpoint: run walk-forward backtest on selected stocks."""
    data = request.get_json()

    tickers = data.get("tickers", ["AAPL", "MSFT", "NVDA"])
    start_date = data.get("start_date", "2020-01-01")
    end_date = data.get("end_date", "2024-12-31")
    model_type = data.get("model_type", "rf")
    train_window = data.get("train_window", 500)
    test_window = data.get("test_window", 60)

    config = BacktestConfig(
        train_window_days=train_window,
        test_window_days=test_window,
        step_days=test_window,
        model_type=model_type,
    )

    try:
        import yfinance as yf

        stock_dfs = {}
        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date,
                            auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            min_required = config.train_window_days + config.test_window_days + 80
            if len(df) >= min_required:
                stock_dfs[ticker] = df

        if not stock_dfs:
            return jsonify({"error": "No stocks had sufficient data."}), 400

        results = run_multi_stock_backtest(stock_dfs, config)

        # Convert to JSON-safe format
        aggregate = {
            k: _safe_float(v)
            for k, v in results["aggregate"].items()
        }

        # Compute combined equity and benchmark curves across all tickers
        # Normalise each ticker's curve to start at 1.0 and average them
        all_strategy_curves = []
        all_benchmark_curves = []
        total_trades = 0
        all_returns = []

        for ticker, res in results["per_stock"].items():
            total_trades += res.total_trades
            all_returns.append(res.total_return_pct)

            # Normalise equity curve to cumulative return percentage
            if res.equity_curve and len(res.equity_curve) > 0:
                start_val = res.equity_curve[0] if res.equity_curve[0] != 0 else 1
                normalised = [(v / start_val - 1) * 100 for v in res.equity_curve]
                all_strategy_curves.append(normalised)

            if res.benchmark_curve and len(res.benchmark_curve) > 0:
                start_val = res.benchmark_curve[0] if res.benchmark_curve[0] != 0 else 1
                normalised = [(v / start_val - 1) * 100 for v in res.benchmark_curve]
                all_benchmark_curves.append(normalised)

        # Average the curves (pad shorter ones with their last value)
        def average_curves(curves):
            if not curves:
                return []
            max_len = max(len(c) for c in curves)
            padded = []
            for c in curves:
                if len(c) < max_len:
                    c = c + [c[-1]] * (max_len - len(c))
                padded.append(c)
            return [_safe_float(sum(vals) / len(vals)) for vals in zip(*padded)]

        combined_strategy = average_curves(all_strategy_curves)
        combined_benchmark = average_curves(all_benchmark_curves)

        # Compute pooled win rate
        total_wins = sum(r.profitable_trades for r in results["per_stock"].values())
        pooled_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        # Compute overall max drawdown from the combined strategy curve
        if combined_strategy:
            peak = float('-inf')
            max_dd = 0
            for val in combined_strategy:
                if val > peak:
                    peak = val
                dd = peak - val
                if dd > max_dd:
                    max_dd = dd
            overall_max_drawdown = -max_dd
        else:
            overall_max_drawdown = 0

        # Sharpe ratio of the combined curve with its 95% interval, so both
        # describe the same equal-weighted strategy as the curve, return and
        # drawdown shown beside them.
        sharpe = _combined_sharpe(combined_strategy)

        # Enhanced aggregate with additional fields
        aggregate["total_trades"] = total_trades
        aggregate["pooled_win_rate_pct"] = _safe_float(pooled_win_rate)
        aggregate["max_drawdown_pct"] = _safe_float(overall_max_drawdown)
        if sharpe is not None:
            aggregate["sharpe_ratio"] = _safe_float(sharpe.estimate)
        aggregate["sharpe_ci_lower"] = _safe_float(sharpe.lower) if sharpe else None
        aggregate["sharpe_ci_upper"] = _safe_float(sharpe.upper) if sharpe else None
        aggregate["excess_return_pct"] = _safe_float(
            aggregate.get("total_return_pct", 0) - aggregate.get("benchmark_return_pct", 0)
        )

        response = {
            "aggregate": aggregate,
            "summary": results["summary_table"].to_dict(orient="records"),
            "combined_strategy_curve": combined_strategy,
            "combined_benchmark_curve": combined_benchmark,
            "per_stock": {},
        }

        for ticker, res in results["per_stock"].items():
            response["per_stock"][ticker] = {
                "total_return_pct": _safe_float(res.total_return_pct),
                "benchmark_return_pct": _safe_float(res.benchmark_return_pct),
                "sharpe_ratio": _safe_float(res.sharpe_ratio),
                "max_drawdown_pct": _safe_float(res.max_drawdown_pct),
                "win_rate_pct": _safe_float(res.win_rate_pct),
                "signal_accuracy_pct": _safe_float(res.signal_accuracy_pct),
                "total_trades": int(res.total_trades),
                "equity_curve": [_safe_float(x) for x in res.equity_curve],
                "benchmark_curve": [_safe_float(x) for x in res.benchmark_curve],
                "report": format_backtest_report(res),
            }

        return jsonify(response)

    except ImportError:
        return jsonify({"error": "yfinance not installed."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
