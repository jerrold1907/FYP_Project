# CM3070 PROJECT — PRELIMINARY PROJECT REPORT

**Project Idea 1.2: AI-Powered Stock Recommendation System**

**Author:** Jarell Liow Wen Yi  
**Student Number:** 220628507  
**Date of Submission:** [DD/MM/2025]  
**Supervisor:** [Supervisor Name]  
**Template Used:** Data Science Project (Machine Learning Application)

---

## Contents

1. Introduction
2. Literature Review
3. Project Design
4. Feature Prototype
5. Appendices
6. References

---

## CHAPTER 1: INTRODUCTION

### 1.1 Project Concept

This project develops an AI-Powered Stock Recommendation System that combines machine learning-driven stock signal prediction with investor risk profiling to produce personalised suitability recommendations. The system addresses a gap in existing retail investment tools: most stock screeners provide generic Buy/Hold/Sell signals without considering the individual investor's risk tolerance, investment horizon, or experience level.

The core idea is to build a three-component pipeline:

1. **Risk Classifier** — A rule-based classifier that categorises investors as Conservative, Moderate, or Aggressive based on five profile attributes (age, income, investment horizon, experience, and self-assessed risk tolerance).
2. **Stock Recommender** — A machine learning model trained on historical stock price data that generates Buy, Hold, or Avoid signals from technical indicators (RSI, MACD, moving averages, volatility).
3. **Suitability Engine** — A deterministic mapping that combines the investor's risk category with the stock signal to produce a final recommendation: Suitable, Use Caution, or Not Suitable.

### 1.2 Motivation

Retail investors increasingly make self-directed trading decisions using platforms like Robinhood, Trading 212, and eToro. However, research shows that individual investors frequently make suboptimal decisions due to behavioral biases and a mismatch between their risk tolerance and chosen investments (Barber and Odean, 2013). While professional wealth managers provide personalised advice, their services are typically inaccessible to retail investors with smaller portfolios. This project aims to democratise basic investment suitability assessment by providing an automated, transparent system.

### 1.3 Aims and Objectives

**Aim:** To develop and evaluate an AI system that provides personalised stock suitability recommendations by integrating investor risk profiling with machine learning-based stock signal prediction.

**Objectives:**

| # | Objective | Deliverable |
|---|-----------|-------------|
| O1 | Design and implement a rule-based investor risk classification system | `src/risk_classifier.py` module with validation |
| O2 | Engineer technical indicator features from historical stock data | `src/features.py` module and processed datasets |
| O3 | Train and compare ML models (Logistic Regression, Decision Tree, Random Forest) for stock signal prediction | Training notebook with model evaluation |
| O4 | Implement a suitability engine that maps risk profiles to stock signals | `src/suitability.py` module |
| O5 | Build an interactive web interface for end-user interaction | Streamlit web application (`app.py`) |
| O6 | Evaluate system accuracy, usability, and recommendation quality | Evaluation chapter in final report |

### 1.4 Justification

These objectives collectively satisfy the aim: O1 establishes the personalisation dimension, O2–O3 provide the predictive intelligence, O4 bridges both into actionable advice, O5 makes it accessible to non-technical users, and O6 validates the system against its goals. The project follows the Data Science project template with a machine learning application focus.

---

## CHAPTER 2: LITERATURE REVIEW

### 2.1 Stock Prediction with Machine Learning

**Patel et al. (2015)** conducted a comparative study of four ML models (ANN, SVM, Random Forest, and Naive Bayes) for stock market prediction, concluding that Random Forest achieved the highest accuracy (86.69%) on Indian stock indices when using technical indicators as features. Their work demonstrates that technical indicators can serve as effective input features for classification models, which directly motivates the feature engineering approach in this project. However, their study focused on a binary Up/Down prediction rather than a three-class system, which this project extends to Buy/Hold/Avoid to provide more nuanced signals.

**Basak et al. (2019)** reviewed machine learning approaches for stock market prediction, identifying that ensemble methods consistently outperform individual classifiers on financial data. They noted a critical gap: most systems predict price direction without connecting predictions to investor suitability. This gap is precisely what the Suitability Engine component of this project addresses — bridging stock signals with risk profiles.

**Nti et al. (2020)** provided a systematic review of ML techniques in stock market prediction, analysing 122 studies. They found that (i) technical indicators are the most commonly used feature set, (ii) Random Forest and SVM are the most popular algorithms, and (iii) prediction horizon significantly impacts accuracy. Their finding that models perform better on shorter horizons (< 30 days) informed the design choice to use a 30-day forward return for target labeling in this project.

### 2.2 Risk Profiling and Investor Classification

**Grable and Lytton (1999)** developed the foundational framework for financial risk tolerance assessment, identifying age, income, investment horizon, and investment knowledge as key determinants. Their 13-item risk tolerance questionnaire has been widely adopted by financial advisors. This project simplifies their approach into five quantifiable attributes while preserving the core dimensions they identified as most predictive of risk tolerance.

**Kannadhasan (2015)** empirically validated that demographic factors (age, income, experience) significantly correlate with investment risk tolerance in emerging markets. Their regression analysis showed age as the strongest negative predictor (older investors prefer lower risk) and experience as a positive predictor (experienced investors tolerate more risk). These findings directly inform the adjustment factors in this project's Risk Classifier, where age > 60, experience < 2 years, and short horizon each shift the classification toward Conservative.

### 2.3 Technical Indicators and Feature Selection

**Achelis (2001)** provides the canonical reference for technical indicator computation, including RSI, MACD, and moving averages. His mathematical definitions are used directly in the feature engineering module of this project. The choice of RSI (momentum), MACD (trend), and SMA crossovers (support/resistance) covers three distinct signal dimensions, reducing feature redundancy while capturing complementary market dynamics.

### 2.4 Suitability Assessment in Financial Services

**FINRA Rule 2111 (Suitability)** establishes the regulatory framework requiring financial professionals to have a reasonable basis for believing an investment recommendation is suitable for the customer. The rule identifies three types of suitability obligations: reasonable-basis, customer-specific, and quantitative. This project operationalises the customer-specific obligation through its deterministic mapping matrix, making the concept assessable by algorithm rather than exclusively by human judgment.

### 2.5 Critical Evaluation and Gaps

The literature reveals several gaps this project addresses:

1. **Disconnection between prediction and personalisation:** Patel et al. (2015) and Nti et al. (2020) focus purely on prediction accuracy without considering individual investor characteristics. This project bridges that gap through the Suitability Engine.

2. **Lack of transparency:** Many ML-based recommendation systems operate as black boxes. By using a rule-based risk classifier (deterministic, explainable) combined with a lookup table for suitability mapping, this project prioritises interpretability — investors can understand WHY a recommendation was made.

3. **Academic vs. practical applicability:** While Basak et al. (2019) note the dominance of ensemble methods, they acknowledge that few academic systems are packaged as usable tools. This project includes a Streamlit web interface to bridge the gap between research prototype and user-facing application.

4. **Simplistic binary prediction:** Most stock prediction literature uses binary Up/Down labels. The three-class labeling (Buy/Hold/Avoid) used here provides a more actionable taxonomy aligned with actual investment decisions.

The project draws on the validated risk tolerance dimensions of Grable and Lytton (1999) and Kannadhasan (2015), applies the ML techniques confirmed effective by Patel et al. (2015) and Nti et al. (2020), uses canonical indicator definitions from Achelis (2001), and operationalises the regulatory suitability concept from FINRA.

---

## CHAPTER 3: PROJECT DESIGN

### 3.1 Domain and Users

**Domain:** Personal finance / retail investment decision support.

**Target Users:** Self-directed retail investors who:
- Use online trading platforms (e.g., Trading 212, eToro, Robinhood)
- Have basic understanding of stocks but limited technical analysis expertise
- Want data-driven guidance that considers their personal risk tolerance
- Prefer transparency in recommendations (understanding WHY, not just WHAT)

**User Needs:**
| Need | How Addressed |
|------|---------------|
| Understand my risk profile | Risk Classifier with explanation of classification factors |
| Know if a stock suits me | Suitability Engine with human-readable explanations |
| Easy to use, no coding required | Streamlit web UI with form inputs |
| Trust the recommendation | Transparent rule-based logic + ML confidence metrics |

**User Stories:**
- As a new investor (age 25, low experience), I want to know whether a stock showing Buy signals is appropriate for my conservative profile, so I don't take on excessive risk.
- As an experienced investor (age 45, high risk tolerance), I want to quickly see if a stock with neutral indicators is still worth holding, so I can make efficient portfolio decisions.

### 3.2 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB APP (app.py)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐   ┌──────────────────┐               │
│  │  Investor Form  │   │  Stock Indicators │               │
│  │  (Sidebar)      │   │  (Main Panel)     │               │
│  └────────┬────────┘   └────────┬──────────┘               │
│           │                      │                          │
│           ▼                      ▼                          │
│  ┌─────────────────┐   ┌──────────────────┐               │
│  │ Risk Classifier │   │ Stock Recommender │               │
│  │ (rule-based)    │   │ (ML model)        │               │
│  └────────┬────────┘   └────────┬──────────┘               │
│           │                      │                          │
│           │  Risk Category       │  Stock Signal            │
│           └──────────┬───────────┘                          │
│                      ▼                                      │
│           ┌──────────────────┐                              │
│           │ Suitability      │                              │
│           │ Engine (lookup)  │                              │
│           └────────┬─────────┘                              │
│                    ▼                                        │
│           ┌──────────────────┐                              │
│           │ Recommendation   │                              │
│           │ + Explanation    │                              │
│           └──────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

**Training Pipeline (Offline):**
```
yfinance API → Raw OHLCV Data → Feature Engineering → Train/Test Split
    → Model Training (LR, DT, RF) → Evaluation → Best Model Export
        → models/stock_model.pkl + models/scaler.pkl
```

### 3.3 Technology Choices

| Technology | Purpose | Justification |
|------------|---------|---------------|
| Python 3.10+ | Core language | Industry standard for data science, rich ML ecosystem |
| scikit-learn | ML models | Well-documented, supports LR/DT/RF, good for academic use |
| pandas/numpy | Data processing | De facto standard for tabular data manipulation |
| yfinance | Stock data | Free, no API key needed, suitable for academic research |
| Streamlit | Web interface | Rapid prototyping for data apps, no front-end expertise needed |
| Hypothesis | Property testing | Ensures correctness guarantees for rule-based components |
| matplotlib/seaborn | Visualisation | Confusion matrices, model comparison charts |
| joblib/pickle | Model serialization | Native Python, simple persistence for sklearn models |

### 3.4 Design Choices Justified by User Needs

1. **Rule-based Risk Classifier (not ML):** Users need to understand WHY they were classified a certain way. A deterministic algorithm with named adjustment factors (age, horizon, experience) is transparent and auditable, unlike a neural network.

2. **3×3 Suitability Matrix (not continuous scoring):** Clear, categorical recommendations (Suitable / Use Caution / Not Suitable) are more actionable for retail investors than nuanced probability scores.

3. **Three-class stock prediction (not binary):** Buy/Hold/Avoid provides a richer signal than Up/Down, mapping more naturally to investment decisions.

4. **Streamlit UI (not Flask/React):** Prioritises rapid iteration and data-centric UI components. Appropriate for a data science project where the focus is on the ML pipeline rather than front-end engineering.

### 3.5 Workplan

| Week | Period | Tasks | Milestone |
|------|--------|-------|-----------|
| 1-2 | Jun 16 – Jun 29 | Project setup, directory structure, validation module, risk classifier | ✅ Core modules implemented |
| 3-4 | Jun 30 – Jul 13 | Suitability engine, feature engineering module, property tests | ✅ All rule-based components tested |
| 5-6 | Jul 14 – Jul 27 | Training notebook (data acquisition, feature engineering, model training) | ✅ Models trained and evaluated |
| 7-8 | Jul 28 – Aug 10 | Model evaluation, comparison, export. Stock recommender integration | ✅ ML pipeline complete |
| 9-10 | Aug 11 – Aug 24 | Streamlit web app, end-to-end integration testing | ✅ Working prototype |
| 11-12 | Aug 25 – Sep 7 | User evaluation, usability testing, model tuning | ✅ Evaluation complete |
| 13-14 | Sep 8 – Sep 21 | Final report writing (Introduction, Lit Review, Methodology) | ✅ Report drafts |
| 15-16 | Sep 22 – Oct 5 | Final report (Results, Evaluation, Conclusion), video, submission prep | ✅ Submission ready |

**Key Milestones:**
- **Week 4:** All deterministic components (risk classifier, suitability engine) fully tested
- **Week 8:** ML pipeline produces exportable models with evaluation metrics
- **Week 10:** End-to-end system functional (web app + all modules)
- **Week 12:** Evaluation data collected
- **Week 16:** Final submission

### 3.6 Testing and Evaluation Plan

#### Technical Evaluation (Automated)

| Technique | Target | Success Criteria |
|-----------|--------|------------------|
| Property-based testing (Hypothesis) | Risk Classifier, Suitability Engine, Feature Engineering | All 11 correctness properties hold across 100+ random examples |
| Unit testing (pytest) | All modules | 90%+ code coverage, all edge cases pass |
| Integration testing | End-to-end pipeline | Model loads, predicts, and produces valid suitability ratings |
| Model accuracy metrics | Stock Recommender | Weighted F1-score > 0.5 (better than random for 3 classes) |
| Confusion matrix analysis | Stock Recommender | Identify systematic biases (e.g., overpredict Hold) |

#### Model Evaluation

- **Metric:** Weighted F1-score (handles class imbalance in Buy/Hold/Avoid labels)
- **Comparison:** Three models (Logistic Regression, Decision Tree, Random Forest) evaluated on same 80/20 stratified test split
- **Baseline:** Random prediction would give ~33% accuracy; the model should significantly exceed this

#### User Evaluation (Planned for Final Report)

| Method | Purpose | Participants |
|--------|---------|--------------|
| Usability walkthrough | Can users complete the recommendation flow? | 5-8 peers (CS students) |
| Recommendation face validity | Do suitability ratings "make sense" to users? | Same participants |
| System Usability Scale (SUS) questionnaire | Standardised usability measurement | Same participants |
| Expert review | Does the risk classification align with financial principles? | 1-2 finance-literate reviewers |

**Evaluation against Aims:**
- **O1 (Risk Classification):** Validated by property tests (11 correctness properties)
- **O2 (Feature Engineering):** Validated by NaN-free output property + mathematical correctness property
- **O3 (ML Models):** Evaluated by F1-score, accuracy, confusion matrix comparison
- **O4 (Suitability Engine):** 100% deterministic; validated by exhaustive 9-pair property test
- **O5 (Web Interface):** SUS questionnaire score target > 68 (above average)
- **O6 (Overall Evaluation):** Comprehensive evaluation chapter combining all methods above

---

## CHAPTER 4: FEATURE PROTOTYPE

### 4.1 Prototype Description

The feature prototype implements the complete three-component pipeline: Risk Classifier → Stock Recommender → Suitability Engine, delivered as an interactive Streamlit web application. The prototype demonstrates the most technically challenging aspect of the project — integrating a trained machine learning model with a rule-based risk classification system to produce personalised investment suitability recommendations.

The prototype is functional and deployed locally. Users interact through a web browser, filling in their investor profile (age, income, horizon, experience, risk tolerance) in a sidebar form, and optionally entering stock technical indicators. The system then:

1. Classifies the investor's risk category using the rule-based algorithm
2. Generates a stock signal (Buy/Hold/Avoid) using the trained Random Forest model
3. Combines both through the Suitability Engine to produce a final recommendation with explanation

### 4.2 Technical Implementation

#### Risk Classifier
The risk classifier implements a deterministic algorithm:
- **Base category** from risk_score: 1–3 → Conservative, 4–6 → Moderate, 7–10 → Aggressive
- **Adjustment factors** that each shift one level toward Conservative: age > 60, horizon < 3 years, experience < 2 years
- **Floor** at Conservative (cannot shift below)

This approach is deliberately rule-based rather than ML-based because transparency and explainability are critical for investment advice — users need to understand WHY they received a classification.

#### Stock Recommender (ML Model)
The machine learning component trains three classifiers on 5 years of daily stock data (2020–2024) for 10 major stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, KO, PEP):

- **Features:** 9 technical indicators — close price, daily return, 5/20/50-day SMAs, 20-day volatility, volume, 14-day RSI (Wilder's smoothing), MACD (EMA12 − EMA26)
- **Target:** 30-day forward return classified as Buy (>10%), Hold (0–10%), Avoid (<0%)
- **Models compared:** Logistic Regression, Decision Tree, Random Forest
- **Best model selected by:** Highest weighted F1-score (accuracy as tiebreaker)
- **Train/test split:** 80/20 stratified, random_state=42

#### Suitability Engine
A deterministic 3×3 lookup table that maps (Risk_Category, Stock_Signal) → Suitability_Rating:

| | Buy | Hold | Avoid |
|--|-----|------|-------|
| **Conservative** | Use Caution | Not Suitable | Not Suitable |
| **Moderate** | Suitable | Use Caution | Not Suitable |
| **Aggressive** | Suitable | Suitable | Use Caution |

Each combination includes a human-readable explanation of the reasoning.

### 4.3 Evaluation of Prototype

**What works well:**
- The three-component architecture is clean and modular — each piece can be tested independently
- The risk classifier produces intuitive results: a 65-year-old with 1 year experience and risk_score=7 correctly shifts from Aggressive to Conservative
- The Streamlit interface is functional and provides immediate feedback
- Property-based testing gives high confidence in the deterministic components (11 correctness properties verified across 100+ random inputs each)
- The suitability explanations help users understand recommendations

**Current limitations:**
1. **Model accuracy:** The stock prediction model's accuracy is limited by the inherent difficulty of stock prediction. A 3-class classification on noisy financial data is challenging, and weighted F1-scores may be modest.
2. **Feature simplicity:** Only 9 technical indicators are used. Sentiment analysis, fundamental indicators (P/E ratio, earnings), and macroeconomic factors are not included.
3. **Static model:** The model is trained once and serialized. It does not update as new market data becomes available (no online learning).
4. **Limited stock universe:** Only 10 stocks are used for training. The model may not generalise well to stocks with different trading characteristics.
5. **No backtesting:** The system is not evaluated on out-of-sample temporal data to simulate real trading performance.

### 4.4 Planned Improvements

1. **Expand feature set:** Add sentiment features from financial news APIs, fundamental indicators, and sector-relative metrics to improve prediction accuracy.
2. **Ensemble with confidence scores:** Rather than outputting a single class, output class probabilities to provide confidence-weighted recommendations.
3. **Temporal validation:** Implement walk-forward validation rather than random train/test split to better simulate real-world deployment where future data is unavailable during training.
4. **Larger stock universe:** Expand from 10 to 50+ stocks across different sectors to improve model generalisability.
5. **User evaluation integration:** Conduct usability testing and incorporate feedback into UI design.

---

## CHAPTER 5: APPENDICES

*[Appendix A: Screenshots of the Streamlit web application — to be included in final submission]*

*[Appendix B: Model evaluation metrics table from training notebook — to be included]*

*[Appendix C: Property test results showing all 11 correctness properties passing — to be included]*

---

## CHAPTER 6: REFERENCES

Achelis, S.B. (2001) *Technical Analysis from A to Z*. 2nd edn. New York: McGraw-Hill.

Barber, B.M. and Odean, T. (2013) 'The Behavior of Individual Investors', in Constantinides, G.M., Harris, M. and Stulz, R.M. (eds.) *Handbook of the Economics of Finance*. Vol. 2. Elsevier, pp. 1533–1570.

Basak, S., Kar, S., Saha, S., Khaidem, L. and Dey, S.R. (2019) 'Predicting the direction of stock market prices using tree-based ensemble learning', *North American Journal of Economics and Finance*, 47, pp. 552–567.

FINRA (2014) *Regulatory Notice 12-25: Know Your Customer and Suitability*. Financial Industry Regulatory Authority. Available at: https://www.finra.org/rules-guidance/rulebooks/finra-rules/2111 (Accessed: [Date]).

Grable, J.E. and Lytton, R.H. (1999) 'Financial risk tolerance revisited: the development of a risk assessment instrument', *Financial Services Review*, 8(3), pp. 163–181.

Kannadhasan, M. (2015) 'Retail investors' financial risk tolerance and their risk-taking behaviour: The role of demographics as differentiating and classifying factors', *IIMB Management Review*, 27(3), pp. 175–184.

Nti, I.K., Adekoya, A.F. and Weyori, B.A. (2020) 'A systematic review of fundamental and technical analysis of stock market predictions', *Artificial Intelligence Review*, 53, pp. 3007–3057.

Patel, J., Shah, S., Thakkar, P. and Kotecha, K. (2015) 'Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques', *Expert Systems with Applications*, 42(1), pp. 259–268.

---

*Word count estimates: Introduction ~800, Literature Review ~1800, Project Design ~1900, Feature Prototype ~1100. Total ~5600/7000.*
