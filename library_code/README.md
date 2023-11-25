# Stock Analysis Based on LSTM (SAB-LSTM)

This document provides an overview and usage guide for the **Stock Analysis Based on LSTM (SAB-LSTM)** Library, a Python-based toolset for financial data processing, feature engineering, model training, and evaluation.

## Before Started
Before using this library, ensure you have Python installed on your machine. Additionally, you will need the following packages:
- yfinance
- pandas
- scikit-learn
- matplotlib
- keras
- tensorflow

You can install these packages using pip:

```Bash
pip install yfinance pandas scikit-learn matplotlib keras tensorflow
```

## Library Structure

This library is consist of 4 modules, as shown below: 
```
.
└── SAB-LSTM/
    ├── data_processing.py/
    │   ├── get_historical_price_dataset()
    │   ├── reformat_historical_price_dataset()
    │   ├── data_split()
    │   ├── data_normalization()
    │   ├── scale_target_variable()
    │   └── prepare_training_data()
    ├── feature_engineering.py/
    │   ├── add_daily_return()
    │   ├── add_simple_moving_average()
    │   └── add_relative_strength_index()
    ├── model_training.py/
    │   ├── build_lstm_model()
    │   └── train_model()
    └── model_evaluation.py/
        ├── model_testing()
        ├── model_evaluation()
        └── plot_test_results()
```

## Libray Usage

### 1. Data Processing (`data_processing.py`)
- **`get_historical_price_dataset(ticker: str) -> pd.DataFrame`**: Fetches historical stock data for a given ticker symbol and returns a DataFrame. It also saves a `.csv` file with raw data.
- **`reformat_historical_price_dataset(input_file: str, output_file: str) -> pd.DataFrame`**: Reformats a given CSV file containing historical price data and saves it to a new file.
- **`data_split(df: pd.DataFrame, train_pct: float, test_pct: float, val_pct: float) -> Tuple[pd.DataFrame]`**: Splits a DataFrame into training, testing, and validation sets.
- **`data_normalization(df: pd.DataFrame, num_feature: int) -> pd.DataFrame`**: Normalizes data features within a DataFrame.

### 2. Feature Engineering (`feature_engineering.py`)
- **`add_daily_return(df: pd.DataFrame) -> pd.DataFrame`**: Adds a 'Daily Return' column to the DataFrame.
- **`add_simple_moving_average(df: pd.DataFrame) -> pd.DataFrame`**: Adds simple moving averages (SMA5, SMA20, SMA60) to the DataFrame.
- **`add_relative_strength_index(df: pd.DataFrame, periods: int) -> pd.DataFrame`**: Calculates and adds the Relative Strength Index (RSI) to the DataFrame.

### 3. Model Evaluation (`model_evaluation.py`)
- **`model_testing(train_df, test_df, scaler, regressor) -> np.ndarray`**: Prepares test data and makes predictions using the trained model.
- **`plot_test_results(actual_index, predict_index)`**: Plots actual versus predicted Nasdaq Index values.
- **`evaluate_model(actual_index, predict_index) -> float`**: Evaluates the model's performance based on trend accuracy.

### 4. Model Training (`model_training.py`)
- **`build_lstm_model(input_shape, lstm_units, dropout_rate, output_units) -> keras.Model`**: Builds and compiles an LSTM model.
- **`train_model(model, X_train, y_train, num_epochs, batch_size) -> History`**: Trains the LSTM model with the specified parameters.

### 5. Training Data Preparation (`training_data.py`)
- **`prepare_training_data(df: pd.DataFrame, num_features: int, window_size: int) -> Tuple[np.ndarray]`**: Prepares and reshapes data for LSTM model training.



