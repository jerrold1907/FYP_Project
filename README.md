# AI-Powered Stock Recommendation System

A university final-year prototype: a web-based stock advisor that assesses an investor's
risk tolerance, analyses a 20-stock universe with a trained machine-learning model, and
returns personalised, explained recommendations — backed by walk-forward backtesting and
statistical evidence.

Three components work together:

| Component | What it does |
|---|---|
| **Risk Questionnaire** | A 5-item scenario-based instrument scores the investor 5–20 and maps them to one of five risk categories |
| **Stock Recommender** | A trained scikit-learn classifier predicts **Buy / Hold / Avoid** from technical indicators |
| **Suitability Engine** | Combines the two into a final rating — **Suitable / Use Caution / Not Suitable** — with a written explanation |

---

## Quick Start

### 1. Prerequisites

- **Python 3.10+** (developed on 3.13)

### 2. Install dependencies

```bash
pip install flask pandas numpy yfinance scikit-learn
```

### 3. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

> The trained model ships in `models/stock_model.pkl`, so there is nothing to train before
> first run. An internet connection is required — live prices are pulled from Yahoo Finance
> via `yfinance` on each request.

### Optional extras

| Feature | Install |
|---|---|
| Run the test suite | `pip install pytest hypothesis` |
| Run the experiments (`experiments/`) | `pip install scipy matplotlib statsmodels` |
| LSTM baseline (`src/sequence_model.py`) | `pip install torch` |
| Open the training notebooks | `pip install jupyter seaborn joblib matplotlib` |
| Regenerate the Word reports (`docs/`) | `pip install python-docx` |
| Conversational chatbot replies | [Ollama](https://ollama.com) running `llama3.2:1b` locally |

Every optional dependency degrades gracefully. Without Ollama the chatbot still answers,
using deterministic template responses instead of LLM-phrased ones.

---

## Web Interface

| Page | Route | Purpose |
|---|---|---|
| **Advisor** | `/` | Answer the 5-question risk assessment, then see every stock in the universe ranked by suitability, with price charts and indicator readouts |
| **Compare** | `/compare` | Put 2–3 stocks side by side — signal, sector, RSI, MACD, volatility, 90-day price history |
| **Backtest** | `/backtest` | Run a walk-forward backtest over a chosen date range and see strategy vs buy-and-hold equity curves |

A chat widget is available on every page for follow-up questions
("Why is NVDA suitable?", "Compare AAPL and MSFT", "Is TSLA risky?"). Every page also
opens with a disclaimer that the system is an educational prototype, not financial advice.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/recommend` | POST | Questionnaire answers in → risk profile plus per-stock signal, suitability, indicators and price history out |
| `/api/compare` | POST | Side-by-side analysis of up to 3 tickers |
| `/api/backtest` | POST | Walk-forward backtest; returns aggregate metrics, per-stock results and combined equity curves |
| `/api/chat` | POST | Chatbot: deterministic answer, plus an optional LLM summary that is discarded if it contradicts the data |

---

## Project Structure

```
FYP_Project/
├── app.py                       # Flask application — routes and API endpoints
├── retrain_model.py             # Retrain and export the stock model (leakage-free split)
├── pyproject.toml               # pytest, coverage and project metadata
│
├── src/
│   ├── stock_universe.py        # Single source of truth: 20 equities (+ S&P 100 list for exp07)
│   ├── market_data.py           # Dated price snapshot shared by every experiment
│   ├── risk_questionnaire.py    # 5-item scenario instrument → 5 risk categories
│   ├── risk_classifier.py       # Rule-based classifier for raw profile attributes
│   ├── features.py              # Technical indicators + forward-return target labels
│   ├── stock_recommender.py     # Loads the model/scaler and predicts Buy/Hold/Avoid
│   ├── suitability.py           # Risk category × signal → rating + explanation
│   ├── validation.py            # Shared input validation and schemas
│   ├── backtester.py            # Walk-forward backtesting vs buy-and-hold
│   ├── chatbot.py               # Intent classification, data-grounded answers, LLM layer
│   ├── evaluation.py            # Purged chronological splitting (anti-leakage)
│   ├── statistics_tests.py      # Wilson intervals, McNemar, bootstrap, significance tests
│   ├── stationarity.py          # ADF / KPSS diagnostics on the feature set
│   ├── sequence_model.py        # LSTM baseline, to justify the model choice
│   ├── instrument_validation.py # Cronbach's alpha and band sensitivity for the questionnaire
│   └── llm_evaluation.py        # Measures the hallucination filter's error rate
│
├── templates/                   # index.html, compare.html, backtest.html, _chat_widget.html, _disclaimer.html
├── static/                      # app.js, chat_widget.js, style.css
│
├── models/
│   ├── stock_model.pkl          # Trained stock signal classifier (used by the app)
│   ├── scaler.pkl               # Fitted StandardScaler for inference
│   ├── risk_model.pkl           # ML risk classifier from the notebook (see note below)
│   ├── risk_scaler.pkl
│   └── risk_label_encoder.pkl
│
├── notebooks/
│   ├── training.ipynb                   # Full ML pipeline with explanation and plots
│   └── risk_classification_model.ipynb  # Trains the ML risk classifier
│
├── experiments/                 # exp01–exp08: scripts, results and CSV outputs
├── tests/                       # unit/, properties/ (Hypothesis), integration/
├── docs/                        # Reports, PROJECT_EXPLANATION.md, report generators
└── data/                        # raw/ (price snapshot, not committed) and processed/ (investor_profiles.csv)
```

> **Note on the two risk models.** The running app classifies risk from the questionnaire
> using the deterministic scoring in `src/risk_questionnaire.py` — this keeps the advice
> auditable and reproducible. The ML risk classifier in `models/risk_model.pkl`, trained by
> `notebooks/risk_classification_model.ipynb`, is retained as a comparison baseline and is
> not on the live request path.

---

## How It Works

### 1. Risk assessment

Five scenario questions — loss reaction, investment goal, time horizon, emotional comfort
and risk/return preference — each score 1–4 points. The 5–20 total maps to a category:

| Score | Category |
|---|---|
| 5–8 | Conservative |
| 9–11 | Moderately Conservative |
| 12–14 | Balanced |
| 15–17 | Growth |
| 18–20 | Aggressive |

The design follows Grable & Lytton (1999). `src/instrument_validation.py` supplies the
evidence — Cronbach's alpha for internal consistency and a sensitivity analysis of the
score boundaries.

### 2. Feature engineering

`src/features.py` computes, from raw OHLCV data:

- 5-, 20- and 50-day simple moving averages
- Daily return and rolling volatility
- RSI (14-day, Wilder's smoothing)
- MACD (EMA12 − EMA26)

Stocks with fewer than 80 trading days are excluded, because the 50-day average and the
volatility window need history to be meaningful. Target labels come from 30-day
forward-looking returns.

### 3. Signal prediction

`StockRecommender` loads the pickled classifier plus its scaler and returns **Buy**, **Hold**
or **Avoid** for the most recent trading day.

### 4. Suitability

A deterministic lookup table maps every (risk category, signal) pair to a rating and a
plain-English reason. Conservative investors are cautioned even on Buy signals; Aggressive
investors get Suitable on both Buy and Hold. Because the rules are a table rather than a
model, every recommendation can be traced back to a stated policy.

### 5. Backtesting

`src/backtester.py` runs walk-forward validation: train on a rolling window, generate
signals on the next unseen block, simulate the portfolio, then compare against buy-and-hold.
A Buy signal takes a full position until the signal changes and a Hold signal a half
position; transaction costs (0.1% each way) are charged only when a position opens or
closes. Reported metrics include total return, benchmark return, Sharpe ratio (with a
Lo-2002 confidence interval), maximum drawdown, win rate and signal accuracy. Each trade
also records the chance that a random entry of the same length in the same window would
have been profitable, so the win rate can be tested against a fair baseline rather than 50%.

### 6. Chatbot

Answers are built in two stages. First a deterministic stage classifies intent, retrieves
the relevant model output and composes a factual answer. Then, if Ollama is running, a local
model rephrases it conversationally. The rephrased version is shown only if `verify_summary`
confirms it does not contradict the underlying figures — so no number reaches the user
unchecked.

---

## Methodology: the leakage correction

An earlier version of this project evaluated the model with a random stratified
train/test split, reporting a weighted F1 of **0.83**. That split is invalid for financial
time series, for two reasons:

1. **Adjacent-row leakage** — rolling features make consecutive days near-duplicates, so a
   random split lets the model match almost-identical rows instead of forecasting.
2. **Label-horizon overlap** — labels are 30-day forward returns, so training labels
   encoded outcomes from the test period.

Under a purged chronological split — test on the most recent 20% of dates, discard training
rows within 30 trading days of the boundary — the same model scores **0.36**. The corrected
procedure lives in `src/evaluation.py` and is used by `retrain_model.py`; all models are
reported against majority-class and random baselines, since weighted F1 on an imbalanced
three-class target is not interpretable on its own.

## Experiments

Every script reads prices through the dated snapshot in `src/market_data.py`. The snapshot
is written on the first run and not committed, so reruns on one machine agree exactly, while a
fresh download may shift some figures slightly. The last column gives the report section each
script supports.

| Script | Question it answers | Report |
|---|---|---|
| `exp01_leakage_diagnostic.py` | How much does a random split inflate the score? (random vs chronological vs purged) | 5.1 |
| `exp03_statistical_analysis.py` | Are the model and strategy results distinguishable from chance? Intervals, McNemar, per-ticker Sharpe intervals, chance-matched win rate, market regimes | 5.4, 5.5, App. B |
| `exp04_model_justification.py` | Which model should be deployed? Stationarity tests, LSTM baseline, per-class recall, seed stability | 5.2, 5.3 |
| `exp05_instrument_and_llm.py` | Is the questionnaire reliable on simulated responses, and how accurate is the hallucination filter? | 5.7, App. E |
| `exp06_real_respondent_reliability.py` | Is the questionnaire reliable on the usability participants' answers? | 5.7, App. D |
| `exp07_universe_scaling.py` | Do the core results hold on the S&P 100? | 5.6 |
| `exp08_label_sensitivity.py` | Do the results depend on the +10% Buy threshold? | App. H |
| `retrain_model.py` (repo root) | Exports the production Logistic Regression model | 4.3 |

Run any of them directly, e.g.:

```bash
python experiments/exp01_leakage_diagnostic.py
```

Results (`*_results.txt`) and data (`*.csv`) are git-ignored and regenerate on each run.

---

## Testing

353 tests across three suites:

```bash
pytest tests/ -v
```

| Suite | Tests | Focus |
|---|---|---|
| `tests/unit/` | 293 | Specific examples and edge cases, module by module |
| `tests/integration/` | 46 | End-to-end flows, model export, component wiring |
| `tests/properties/` | 14 | Hypothesis property-based tests — invariants that must hold for all valid inputs |

Run one suite, or select by marker (`unit`, `property`, `integration`):

```bash
pytest tests/properties/ -v
```

```bash
pytest -m unit -v
```

Coverage is configured over `src/` with a 70% floor:

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Retraining

To retrain the stock signal model from scratch:

```bash
python retrain_model.py
```

This downloads fresh data for the universe, engineers features, evaluates Logistic
Regression, Decision Tree and Random Forest under the purged chronological split against
baselines, and exports the winner to `models/stock_model.pkl` with its scaler.

The notebooks cover the same ground with commentary and plots:

```bash
jupyter notebook notebooks/training.ipynb
```

---

## Stock Universe

20 equities across 9 sectors, defined once in `src/stock_universe.py`. The set was
expanded from an original 10 large-cap technology names so that claims about generalisation
could actually be tested across market regimes. The same module also lists the S&P 100,
which `experiments/exp07_universe_scaling.py` uses to check whether the results hold at
five times the scale; the application itself still analyses the 20 stocks below.

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, META, NVDA, AMD, MU |
| Consumer Cyclical | AMZN, TSLA, HD |
| Consumer Defensive | KO, PEP, WMT |
| Healthcare | JNJ, UNH |
| Financial Services | JPM |
| Energy | XOM |
| Industrials | CAT |
| Communication Services | DIS |
| Utilities | NEE |

---

## Technology Stack

| Layer | Tools |
|---|---|
| Web | Flask, vanilla JavaScript, Chart.js |
| Data | pandas, numpy, yfinance |
| ML | scikit-learn (optional: PyTorch for the LSTM baseline) |
| Statistics | scipy, statsmodels |
| Testing | pytest, Hypothesis |
| Notebooks | Jupyter, matplotlib, seaborn |
| LLM layer | Ollama (`llama3.2:1b`), optional |

---

## Further Reading

- [`docs/PROJECT_EXPLANATION.md`](docs/PROJECT_EXPLANATION.md) — component-by-component walkthrough of the whole system
- [`docs/specs/`](docs/specs/) — requirements, design and task breakdown

---

## Disclaimer

This is an academic prototype built for a final-year project. It is **not** financial advice.
The models are trained on limited historical data, the backtests cover a small number of
trades, and past performance does not predict future results. Do not make investment
decisions with it.
