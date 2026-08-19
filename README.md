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
│   ├── training.ipynb       # Complete training pipeline notebook
│   └── risk_classification_model.ipynb  # Investor risk classification model training
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

## Notebooks

The `notebooks/` folder contains Jupyter notebooks that document the model training process with explanations and visualisations:

| Notebook | Purpose | How to run |
|----------|---------|-----------|
| `training.ipynb` | Full ML pipeline — data download, feature engineering, model training, evaluation and comparison | `jupyter notebook notebooks/training.ipynb` |
| `risk_classification_model.ipynb` | Trains the investor risk classification model that maps profile attributes to risk categories | `jupyter notebook notebooks/risk_classification_model.ipynb` |

> **Note:** The pre-trained model is included in `models/stock_model.pkl`, so you do NOT need to run these notebooks to use the app. They are provided for reproducibility and to demonstrate the training process. If you want to retrain from scratch, you can also run `python retrain_model.py` from the command line.

### Prerequisites

1. **Python 3.10+** installed on your system
2. Install the required dependencies:

```bash
pip install pandas numpy yfinance scikit-learn matplotlib seaborn jupyter hypothesis pytest
```

### Running Notebooks

1. Navigate to the project root directory:

```bash
cd FYP_Project
```

2. Launch Jupyter Notebook:

```bash
pip install jupyter
jupyter notebook notebooks/
```

3. Open the desired notebook and run all cells in order from top to bottom.
   - `training.ipynb` will: download historical stock data from yfinance (requires internet connection), compute technical indicators (feature engineering), train and evaluate three ML models (Logistic Regression, Decision Tree, Random Forest), compare model performance and select the best model by weighted F1-score, and export the best model to `models/stock_model.pkl`.
   - `risk_classification_model.ipynb` will: build the investor profile dataset, train and evaluate the risk classification model, and export it for use by the Suitability Engine.

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
