# AI-Powered Stock Recommendation System

A university prototype that combines machine learning models to provide personalised stock recommendations based on investor risk profiles and technical market analysis.

The system uses two ML models working together:
- **Investor Risk Classifier** — Categorises investors as Conservative, Moderate, or Aggressive based on their profile attributes
- **Stock Recommender** — Predicts Buy, Hold, or Avoid signals from technical indicators computed on historical price data

These outputs are combined by a **Suitability Engine** that produces a final recommendation: Suitable, Use Caution, or Not Suitable.

## Directory Structure

```
FYP_Project/
├── data/
│   ├── raw/                 # Downloaded stock price data
│   └── processed/           # Feature-engineered datasets
├── notebooks/
│   └── training.ipynb       # Complete training pipeline notebook
├── models/
│   ├── stock_model.pkl      # Best performing trained model
│   └── scaler.pkl           # Fitted StandardScaler for inference
├── src/
│   ├── __init__.py          # Package initialisation
│   ├── risk_classifier.py   # Investor risk classification logic
│   ├── stock_recommender.py # Model loading and prediction
│   ├── suitability.py       # Suitability engine (combines outputs)
│   ├── features.py          # Feature engineering functions
│   └── validation.py        # Input validation utilities
├── docs/                    # Documentation and project reports
├── tests/
│   ├── properties/          # Property-based tests (Hypothesis)
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
└── README.md
```

## Running the Training Notebook

### Prerequisites

1. **Python 3.10+** installed on your system
2. Install the required dependencies:

```bash
pip install pandas numpy yfinance scikit-learn matplotlib seaborn jupyter hypothesis pytest
```

### Running the Notebook

1. Navigate to the project root directory:

```bash
cd FYP_Project
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/training.ipynb
```

3. Run all cells in order from top to bottom. The notebook will:
   - Download historical stock data from yfinance (requires internet connection)
   - Compute technical indicators (feature engineering)
   - Train and evaluate three ML models (Logistic Regression, Decision Tree, Random Forest)
   - Compare model performance and select the best model by weighted F1-score
   - Export the best model to `models/stock_model.pkl`

### Running Tests

```bash
pytest tests/ -v
```

## Stock Universe

The system is trained on the following stocks: AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, KO, PEP.

## Technology Stack

- **pandas** — Data manipulation
- **numpy** — Numerical computation
- **yfinance** — Stock price data download
- **scikit-learn** — ML models and preprocessing
- **matplotlib / seaborn** — Visualisation
- **Hypothesis** — Property-based testing
- **pytest** — Test runner
