# Implementation Plan: AI-Powered Stock Recommendation System

## Overview

This plan implements a Python-based AI stock recommendation system consisting of three core components: an investor Risk Classifier, a Stock Recommender trained on technical indicators, and a Suitability Engine that merges both outputs. The implementation follows a bottom-up approach — setting up project structure first, then building validation and classification modules, feature engineering, model training notebook, and finally wiring everything together.

## Tasks

- [x] 1. Set up project structure and core modules
  - [x] 1.1 Create directory structure and empty module files
    - Create `data/raw/`, `data/processed/`, `notebooks/`, `models/`, `src/`, `docs/`, `tests/properties/`, `tests/unit/`, `tests/integration/` directories
    - Create `src/__init__.py`, `src/risk_classifier.py`, `src/stock_recommender.py`, `src/suitability.py`, `src/features.py`, `src/validation.py`
    - Create `README.md` with project description, directory structure listing, and instructions for running the Training_Notebook
    - _Requirements: 8.1, 8.2_

  - [x] 1.2 Set up testing framework configuration
    - Create `pytest.ini` or `pyproject.toml` with pytest configuration
    - Configure test discovery for `tests/` directory
    - Add hypothesis settings profile with min 100 examples
    - _Requirements: Design Testing Strategy_

- [x] 2. Implement input validation module
  - [x] 2.1 Implement `src/validation.py` with investor profile validation
    - Implement `validate_investor_profile(data: dict) -> list[str]` that checks all fields against the schema (age: int 18–120, income: float 0.00–999999999.99, horizon: int 1–50, experience: int 0–50, risk_score: int 1–10)
    - Return a list of all error messages (one per invalid/missing field) identifying the field name and accepted range
    - Implement `validate_risk_category(value: str) -> str | None` for case-sensitive validation against {Conservative, Moderate, Aggressive}
    - Implement `validate_stock_signal(value: str) -> str | None` for case-sensitive validation against {Buy, Hold, Avoid}
    - _Requirements: 1.2, 1.4, 1.5, 7.2, 7.3_

  - [x] 2.2 Write property test for valid profile acceptance
    - **Property 2: Valid investor profiles are accepted without errors**
    - **Validates: Requirements 1.2**
    - Use Hypothesis `@given` with strategies generating profiles within all valid ranges
    - Assert `validate_investor_profile()` returns empty list for all valid profiles

  - [x] 2.3 Write property test for invalid profile error reporting
    - **Property 3: Invalid investor profiles report all errors with field names and ranges**
    - **Validates: Requirements 1.4, 1.5**
    - Use Hypothesis to generate profiles with N (1–5) invalid/missing fields
    - Assert exactly N error messages returned, each naming the invalid field

- [x] 3. Implement Risk Classifier
  - [x] 3.1 Implement `src/risk_classifier.py` with InvestorProfile and RiskClassifier
    - Define `InvestorProfile` dataclass with age, income, horizon, experience, risk_score fields
    - Implement `RiskClassifier.validate()` that delegates to `validate_investor_profile()`
    - Implement `RiskClassifier.classify()` with base category from risk_score (1–3 → Conservative, 4–6 → Moderate, 7–10 → Aggressive) and adjustment logic (age > 60, horizon < 3, experience < 2 each shift one level toward Conservative, floor at Conservative)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 3.2 Write property test for risk classification rules
    - **Property 1: Risk classification follows documented rules**
    - **Validates: Requirements 1.1, 1.3**
    - Use Hypothesis to generate random valid InvestorProfiles
    - Independently compute expected classification using the same algorithm and assert match

  - [x] 3.3 Write unit tests for Risk Classifier edge cases
    - Test boundary values: risk_score=3 (Conservative boundary), risk_score=4 (Moderate boundary), risk_score=7 (Aggressive boundary)
    - Test maximum adjustments: Aggressive profile with all three adjustment factors active → should become Conservative
    - Test Conservative floor: Conservative profile with adjustment factors should remain Conservative
    - _Requirements: 1.1, 1.3_

- [x] 4. Implement Suitability Engine
  - [x] 4.1 Implement `src/suitability.py` with SuitabilityEngine class
    - Define the MAPPING dictionary for all 9 (Risk_Category, Stock_Signal) → Suitability_Rating pairs
    - Define an EXPLANATIONS dictionary mapping each (Risk_Category, Stock_Signal) pair to a human-readable explanation string describing WHY the recommendation was made (e.g., "Conservative investor + Buy signal → Suitable with Caution: your risk profile suggests limiting exposure to high-growth stocks.")
    - Implement `generate_explanation(risk_category, stock_signal) -> str` that returns the explanation string for the given combination
    - Implement `recommend(risk_category, stock_signal) -> dict` that validates inputs using `validate_risk_category()` and `validate_stock_signal()`, returns error message for None/empty/invalid inputs identifying which input was invalid and the received value, and for valid inputs returns a dict with keys `rating` (the Suitability_Rating) and `explanation` (the human-readable explanation string from `generate_explanation()`)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 4.2 Write property test for suitability mapping and explanation correctness
    - **Property 10: Suitability mapping matches defined table and includes explanation**
    - **Validates: Requirements 7.1**
    - Enumerate all 9 valid (Risk_Category, Stock_Signal) pairs and verify the returned `rating` matches the hardcoded expected table
    - Verify the returned `explanation` is a non-empty string that contains both the Risk_Category and Stock_Signal values
    - Verify the explanation references the returned Suitability_Rating

  - [x] 4.3 Write property test for invalid suitability inputs
    - **Property 11: Invalid suitability inputs produce descriptive errors**
    - **Validates: Requirements 7.2, 7.3**
    - Use Hypothesis to generate random strings not in valid sets, verify error messages identify which input is invalid and state the received value
    - Verify that no explanation is returned for invalid inputs

- [x] 5. Checkpoint - Ensure all core module tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Feature Engineering module
  - [x] 6.1 Implement `src/features.py` with compute_features and compute_target_labels
    - Implement `compute_features(df)` that requires at least 80 trading days and computes: close_price, daily_return, ma_5 (5-day SMA), ma_20 (20-day SMA), ma_50 (50-day SMA), volatility (20-day rolling std of daily_return), volume, RSI (14-day using Wilder's smoothing), MACD (EMA12 − EMA26)
    - Drop all rows with NaN values after feature computation
    - Implement `compute_target_labels(df)` computing future_30_day_return and labeling: Buy (>10%), Hold (0%–10% inclusive), Avoid (<0%)
    - Exclude stocks with fewer than 80 trading days
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 6.2 Write property test for NaN-free feature output
    - **Property 4: Feature pipeline produces no NaN values**
    - **Validates: Requirements 2.6, 3.2**
    - Use Hypothesis to generate random price series with 80+ days and no NaN in Close/Volume
    - Assert output DataFrame has zero NaN values

  - [x] 6.3 Write property test for feature computation correctness
    - **Property 5: Feature computations match mathematical definitions**
    - **Validates: Requirements 3.1**
    - Generate price series, independently compute ma_5, ma_20, volatility, verify RSI in [0,100], verify MACD = EMA(12) − EMA(26)

  - [x] 6.4 Write property test for target label rules
    - **Property 6: Target labels follow labeling rules**
    - **Validates: Requirements 3.3**
    - Generate pairs of (current_close, future_close), compute future_30_day_return, verify label matches threshold rules

- [x] 7. Implement Stock Recommender module
  - [x] 7.1 Implement `src/stock_recommender.py` with StockRecommender class
    - Implement `__init__(model_path, scaler_path)` that loads model from pickle/joblib and optionally loads scaler if file exists
    - Implement `predict(features: dict) -> str` that applies scaler if available, runs model prediction, and returns "Buy", "Hold", or "Avoid"
    - Handle FileNotFoundError with descriptive error messages
    - _Requirements: 6.1, 6.2_

  - [x] 7.2 Write property test for model serialization round-trip
    - **Property 9: Model serialization round-trip preserves predictions**
    - **Validates: Requirements 6.2**
    - Train a small RandomForest on synthetic data, save and reload, verify identical predictions on test inputs

- [x] 8. Checkpoint - Ensure all module tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Training Notebook - Data Acquisition
  - [x] 9.1 Create `notebooks/training.ipynb` with Introduction and Data Acquisition sections
    - Add title cell with project name, student name placeholder, course name placeholder, and date
    - Add Introduction markdown section (## heading + explanation paragraph)
    - Add Data Acquisition section: download daily OHLCV data from yfinance for Stock_Universe (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, KO, PEP) for Jan 1 2020 to Dec 31 2024
    - Handle download exceptions: log warning with ticker and reason, continue with remaining tickers
    - Log warning for tickers with < 1000 rows
    - Store in DataFrame with Date, Open, High, Low, Close, Adj Close, Volume, Ticker columns
    - Drop rows with NaN in Close or Volume columns
    - Include markdown explanations and inline code comments per cell
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 8.3, 8.4, 9.1, 9.2, 9.3_

- [x] 10. Implement Training Notebook - Feature Engineering and Model Training
  - [x] 10.1 Add Feature Engineering section to the notebook
    - Add markdown cell explaining feature engineering purpose and methodology
    - Call `compute_features()` and `compute_target_labels()` from `src/features.py` (or replicate logic inline with explanation)
    - Drop NaN rows from rolling window calculations
    - Exclude stocks with < 80 trading days
    - Include inline comments explaining each computation
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 8.3, 9.2, 9.3_

  - [x] 10.2 Add Model Training section to the notebook
    - Add markdown cell explaining training methodology
    - Split data 80/20 stratified with random_state=42
    - Apply StandardScaler to features before Logistic Regression, save fitted scaler
    - Train Logistic Regression, Decision Tree, and Random Forest classifiers
    - Include inline comments explaining algorithmic choices
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 8.3, 9.2, 9.3_

  - [x] 10.3 Write property test for stratified split proportions
    - **Property 7: Stratified split preserves class proportions**
    - **Validates: Requirements 4.3**
    - Generate labeled datasets with 3+ classes, apply 80/20 stratified split with seed=42, verify class proportions differ by ≤ 1 percentage point

- [x] 11. Implement Training Notebook - Model Evaluation, Comparison, and Export
  - [x] 11.1 Add Model Evaluation and Comparison sections to the notebook
    - Add markdown cells explaining evaluation methodology and expected outcomes
    - Compute accuracy, weighted precision, weighted recall, weighted F1-score for each model
    - Display comparison table with models as columns and metrics as rows
    - Generate confusion matrix visualization for each model with class labels (Buy, Hold, Avoid) and model name title
    - Add interpretation markdown cells explaining what metrics indicate
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 8.3, 9.2, 9.3, 9.4_

  - [x] 11.2 Add Model Export section to the notebook
    - Add markdown cell explaining export methodology
    - Select best model by highest weighted F1-score (accuracy as tiebreaker)
    - Save best model to `models/stock_model.pkl` using pickle or joblib
    - Save fitted scaler to `models/scaler.pkl` if used
    - Handle filesystem errors with descriptive error messages
    - Add references markdown cell at end of notebook listing external sources
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 8.3, 9.2, 9.3, 9.5_

  - [x] 11.3 Write property test for best model selection logic
    - **Property 8: Best model selection by F1 with tiebreaker**
    - **Validates: Requirements 5.3**
    - Generate random score triples (F1 and accuracy for 3 models), verify correct model is selected by highest F1, then accuracy as tiebreaker

- [x] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Integration and final wiring
  - [x] 13.1 Wire all components together with end-to-end flow
    - Verify `StockRecommender` can load the exported model and produce predictions
    - Verify the full flow: InvestorProfile → RiskClassifier → Risk_Category, features → StockRecommender → Stock_Signal, then SuitabilityEngine combines both into Suitability_Rating
    - Ensure all src modules import correctly and work together
    - _Requirements: 1.1, 6.2, 7.1_

  - [x] 13.2 Write integration tests for end-to-end pipeline
    - Test model load and predict returns valid Stock_Signal
    - Test full pipeline from investor profile + stock features → suitability rating
    - Mock yfinance for notebook pipeline test verifying model file is produced
    - _Requirements: 1.1, 6.2, 7.1_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The training notebook should be runnable end-to-end once yfinance data is available
- All models use scikit-learn; serialization uses pickle or joblib
- Hypothesis library is used for property-based testing with min 100 examples per property

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1", "4.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "4.2", "4.3"] },
    { "id": 4, "tasks": ["6.1"] },
    { "id": 5, "tasks": ["6.2", "6.3", "6.4", "7.1"] },
    { "id": 6, "tasks": ["7.2", "9.1"] },
    { "id": 7, "tasks": ["10.1", "10.2"] },
    { "id": 8, "tasks": ["10.3", "11.1"] },
    { "id": 9, "tasks": ["11.2", "11.3"] },
    { "id": 10, "tasks": ["13.1"] },
    { "id": 11, "tasks": ["13.2"] }
  ]
}
```
