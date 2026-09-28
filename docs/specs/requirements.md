# Requirements Document

## Introduction

This document specifies the requirements for an AI-Powered Stock Recommendation System designed as a 1-week university prototype. The system combines two machine learning models: an Investor Risk Classification Model that categorizes investors based on their profile, and a Stock Recommendation Model that analyzes market indicators to produce buy/hold/avoid signals. The final output merges both model predictions into a suitability recommendation (Suitable, Use Caution, Not Suitable) for a given investor-stock pair.

## Glossary

- **Risk_Classifier**: The machine learning model that classifies investors into risk categories based on their profile attributes
- **Stock_Recommender**: The machine learning model that predicts stock actions (Buy, Hold, Avoid) based on technical market indicators
- **Suitability_Engine**: The component that combines Risk_Classifier and Stock_Recommender outputs to produce a final recommendation
- **Investor_Profile**: A data record containing age, income, investment horizon, investment experience, and risk tolerance score for an investor
- **Technical_Indicators**: Computed financial metrics including daily return, moving averages, volatility, RSI, MACD, and volume derived from stock price data
- **Risk_Category**: One of three investor classifications: Conservative, Moderate, or Aggressive
- **Stock_Signal**: One of three stock action predictions: Buy, Hold, or Avoid
- **Suitability_Rating**: One of three final recommendations: Suitable, Use Caution, or Not Suitable
- **Training_Notebook**: A Jupyter Notebook that contains all data acquisition, feature engineering, model training, evaluation, and model export steps
- **Stock_Universe**: The set of stocks used for training and evaluation: AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, KO, PEP

## Requirements

### Requirement 1: Investor Risk Classification

**User Story:** As a university student building an AI prototype, I want to classify investors into risk categories based on their profile, so that stock recommendations can be tailored to individual risk tolerance.

#### Acceptance Criteria

1. WHEN an Investor_Profile is provided with age, income, investment horizon, investment experience, and risk tolerance score, THE Risk_Classifier SHALL output exactly one Risk_Category: Conservative, Moderate, or Aggressive.
2. THE Risk_Classifier SHALL accept numeric inputs for age (integer, 18–120), income (float, 0.00–999,999,999.99), investment horizon in years (integer, 1–50), investment experience in years (integer, 0–50), and risk tolerance score (integer, 1–10).
3. THE Risk_Classifier SHALL classify an investor as Conservative when the risk tolerance score is between 1 and 3, as Moderate when the risk tolerance score is between 4 and 6, and as Aggressive when the risk tolerance score is between 7 and 10, with age above 60, investment horizon below 3 years, or investment experience below 2 years each shifting the classification one level toward Conservative.
4. IF any Investor_Profile field is missing or outside its valid numeric range, THEN THE Risk_Classifier SHALL return an error message that identifies each invalid field by name and states the accepted range for that field.
5. IF multiple Investor_Profile fields are invalid, THEN THE Risk_Classifier SHALL return a single error response listing all invalid fields rather than reporting only the first.

### Requirement 2: Stock Data Acquisition

**User Story:** As a university student, I want to download historical stock data from yfinance, so that I can compute features for model training.

#### Acceptance Criteria

1. THE Training_Notebook SHALL download daily price data (Open, High, Low, Close, Adjusted Close, and Volume) from yfinance for all stocks in the Stock_Universe (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, KO, PEP).
2. THE Training_Notebook SHALL retrieve data covering the date range from January 1, 2020 to December 31, 2024.
3. IF a stock ticker download raises an exception or returns an empty DataFrame, THEN THE Training_Notebook SHALL log a warning message identifying the failed ticker and the reason for failure, and continue processing remaining tickers.
4. WHEN data is downloaded successfully, THE Training_Notebook SHALL store the data in a pandas DataFrame with columns for Date, Open, High, Low, Close, Adjusted Close, Volume, and ticker symbol.
5. IF a downloaded ticker contains fewer than 1000 trading-day rows, THEN THE Training_Notebook SHALL log a warning indicating insufficient data for that ticker.
6. WHEN all tickers have been processed, THE Training_Notebook SHALL drop any rows containing NaN values in the Close or Volume columns before returning the final DataFrame.

### Requirement 3: Feature Engineering

**User Story:** As a university student, I want to compute technical indicators from raw stock data, so that the model has meaningful input features for prediction.

#### Acceptance Criteria

1. WHEN raw stock data containing at least 80 trading days of history per stock is available, THE Training_Notebook SHALL compute the following features for each stock: close_price, daily_return (percentage change of close price from previous trading day), ma_5 (5-day moving average of close price), ma_20 (20-day moving average of close price), ma_50 (50-day moving average of close price), volatility (20-day rolling standard deviation of daily returns), volume, RSI (14-day Relative Strength Index), and MACD (12-day EMA of close price minus 26-day EMA of close price).
2. WHEN features or target labels contain NaN values due to rolling window calculations or insufficient future data, THE Training_Notebook SHALL drop those rows before model training.
3. THE Training_Notebook SHALL compute a target label for each row based on future_30_day_return, defined as the percentage change from the current day's close price to the close price 30 trading days ahead: Buy when future_30_day_return exceeds 10%, Hold when future_30_day_return is between 0% and 10% (inclusive of both 0% and 10%), and Avoid when future_30_day_return is below 0%.
4. IF a stock has fewer than 80 trading days of historical data, THEN THE Training_Notebook SHALL exclude that stock from feature computation.

### Requirement 4: Stock Recommendation Model Training

**User Story:** As a university student, I want to train multiple ML models on technical indicators, so that I can compare approaches and select the best performer.

#### Acceptance Criteria

1. THE Training_Notebook SHALL train three classification models: Logistic Regression, Decision Tree, and Random Forest.
2. THE Training_Notebook SHALL use the computed Technical_Indicators (close_price, daily_return, ma_5, ma_20, ma_50, volatility, volume, RSI, MACD) as input features and the target label (Buy, Hold, Avoid) as the output.
3. THE Training_Notebook SHALL split the dataset into training and testing sets using a stratified 80/20 split ratio to preserve class distribution across both sets.
4. THE Training_Notebook SHALL use a fixed random seed value of 42 for reproducibility across all model training and data splitting operations.
5. THE Training_Notebook SHALL apply feature scaling (StandardScaler) to the input features before training the Logistic Regression model, and save the fitted scaler for inference.

### Requirement 5: Model Evaluation and Comparison

**User Story:** As a university student, I want to evaluate models using standard metrics, so that I can justify the selection of the best model in my project report.

#### Acceptance Criteria

1. WHEN model training is complete, THE Training_Notebook SHALL evaluate each model on the test set using accuracy, precision (weighted average), recall (weighted average), F1-score (weighted average), and confusion matrix across the three target classes (Buy, Hold, Avoid).
2. THE Training_Notebook SHALL display a comparison table showing accuracy, weighted precision, weighted recall, and weighted F1-score for all three models (Logistic Regression, Decision Tree, Random Forest) side by side, with each row representing one metric and each column representing one model.
3. THE Training_Notebook SHALL identify the best-performing model as the model with the highest weighted F1-score on the test set. IF two or more models share the same highest weighted F1-score, THEN THE Training_Notebook SHALL select the model with the highest accuracy as a tiebreaker.
4. THE Training_Notebook SHALL generate a confusion matrix visualization for each model with axes labeled by the target class names (Buy, Hold, Avoid) and a title identifying the model name.

### Requirement 6: Model Export

**User Story:** As a university student, I want to save the best trained model to disk, so that it can be loaded for inference without retraining.

#### Acceptance Criteria

1. WHEN model evaluation is complete, THE Training_Notebook SHALL save the best-performing model (as determined by highest weighted F1-score) to the `models/` directory in a file named "stock_model.pkl" using Python's pickle or joblib serialization.
2. WHEN "stock_model.pkl" is loaded back into memory and used to predict on the same test set from training, THE Stock_Recommender SHALL produce identical class label predictions to the model before serialization.
3. IF a feature scaler or preprocessor is used during training, THEN THE Training_Notebook SHALL save it to the `models/` directory in a file named "scaler.pkl" using the same serialization method as the model.
4. IF the model or preprocessor file fails to save due to a filesystem error, THEN THE Training_Notebook SHALL raise an error message indicating which file could not be saved.

### Requirement 7: Suitability Recommendation

**User Story:** As a university student, I want to combine investor risk profile and stock signal into a single recommendation, so that the system provides actionable advice.

#### Acceptance Criteria

1. WHEN a Risk_Category and a Stock_Signal are both available, THE Suitability_Engine SHALL produce a Suitability_Rating according to the following mapping:
   - Conservative + Buy = Use Caution
   - Conservative + Hold = Not Suitable
   - Conservative + Avoid = Not Suitable
   - Moderate + Buy = Suitable
   - Moderate + Hold = Use Caution
   - Moderate + Avoid = Not Suitable
   - Aggressive + Buy = Suitable
   - Aggressive + Hold = Suitable
   - Aggressive + Avoid = Use Caution
2. THE Suitability_Engine SHALL accept only valid Risk_Category values (Conservative, Moderate, Aggressive) and valid Stock_Signal values (Buy, Hold, Avoid), matched case-sensitively.
3. IF an invalid Risk_Category or Stock_Signal is provided, THEN THE Suitability_Engine SHALL return an error message that identifies which input was invalid and states the value that was received.
4. IF one or both of Risk_Category or Stock_Signal are missing (None or empty string), THEN THE Suitability_Engine SHALL return an error message indicating which input is missing.

### Requirement 8: Project Structure

**User Story:** As a university student, I want the project organized into clear folders and stages, so that it is easy to navigate and suitable for academic submission.

#### Acceptance Criteria

1. THE project SHALL be organized into the following directory structure:
   - `data/` for raw and processed datasets
   - `notebooks/` for Jupyter Notebooks
   - `models/` for saved model files
   - `src/` for Python source modules
   - `docs/` for documentation and project reports
2. THE project root SHALL contain a README file that describes the project purpose, lists the directory structure, and provides instructions for running the Training_Notebook.
3. THE Training_Notebook SHALL reside in the `notebooks/` directory and include markdown cells before each code section containing a section header (markdown heading level 2) and at least one paragraph explaining the purpose and approach of that section.
4. THE Training_Notebook SHALL be structured into the following sections in order: Introduction, Data Acquisition, Feature Engineering, Model Training, Model Evaluation, Model Comparison, and Model Export.

### Requirement 9: Training Notebook Documentation

**User Story:** As a university student, I want comprehensive markdown documentation within the notebook, so that my instructor can follow the methodology and reasoning.

#### Acceptance Criteria

1. THE Training_Notebook SHALL include a title cell with the project name, student name, course name, and date of last execution.
2. WHEN introducing each section (Introduction, Data Acquisition, Feature Engineering, Model Training, Model Evaluation, Model Comparison, and Model Export), THE Training_Notebook SHALL include a markdown cell explaining the purpose, methodology, and expected outcomes of that section.
3. THE Training_Notebook SHALL include at least one inline comment per code cell explaining the operation performed or algorithmic choice made in that cell.
4. WHEN displaying evaluation results (accuracy, precision, recall, F1-score, confusion matrix, or model comparison table), THE Training_Notebook SHALL include an interpretation markdown cell explaining what the metrics or visualizations indicate about model performance in the context of stock recommendation.
5. WHEN referencing external libraries, datasets, or formulas, THE Training_Notebook SHALL include a references markdown cell at the end of the notebook listing the sources used.
