import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def get_historical_price_dataset(ticker: str) -> pd.DataFrame:
    """
    Return a pandas DataFrame containing historical data for a specified ticker.
    By default, it also stores a .csv file named by ticker in the current directory.

    Warning: Please do not delete the file as it will be used for further reformat process.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    pd.DataFrame: A DataFrame containing historical stock price data.
    
    The DataFrame includes the following columns:
    - Date: The date of the trading data.
    - Open: The opening price of the stock for that day.
    - High: The highest price of the stock during that day.
    - Low: The lowest price of the stock during that day.
    - Close: The closing price of the stock for that day.
    - Volume: The volume of stock traded during that day.
    - Dividends: Any dividends paid out during that day.
    - Stock Splits: Any stock splits that occurred during that day.
    """
    df = yf.Ticker(ticker)
    hist = df.history(period="max")
    hist.to_csv(f'{ticker}_raw.csv')
    return hist

def reformat_historical_price_dataset(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Given a CSV file with historical price data, create/modify a new file with
    the dates formatted differently, and with the 'Dividends' and 'Stock Splits'
    columns removed.

    Parameters:
    - input_file (str): The name of the file in the current directory containing
      the historical price data in the original format, as returned with the
      `get_historical_price_dataset` function.
    - output_file (str): The name of the file in the current directory to which
      to output the reformatted historical price data.

    Returns:
    - pandas.DataFrame: DataFrame with reformatted data.
    """
    df = pd.read_csv(input_file)

    # Reformatting the date column
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Dropping 'Dividends' and 'Stock Splits' columns
    df = df.drop(columns=['Dividends', 'Stock Splits'])

    # Saving the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    return df
  
def data_split(df: pd.DataFrame, train_pct: float = 0.7, 
               test_pct: float = 0.15, val_pct: float = 0.15) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits a DataFrame into training, testing, and validation sets.

    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    train_pct (float): Percentage of data for training set.
    test_pct (float): Percentage of data for testing set.
    val_pct (float): Percentage of data for validation set.

    Returns:
    tuple: Tuple containing training, testing, and validation DataFrames.
    """

    # Ensure the ratios sum to 1
    assert train_pct + test_pct + val_pct == 1, "Ratios must sum to 1"

    train_idx = int(len(df) * train_pct)
    test_idx = train_idx + int(len(df) * test_pct)
    
    # Split the DataFrame
    train_df = df.iloc[:train_idx]
    test_df = df.iloc[train_idx:test_idx]
    val_df = df.iloc[test_idx:]
    
    return train_df, test_df, val_df

def data_normalization(df: pd.DataFrame, num_features: int) -> pd.DataFrame:
    """
    Normalize features in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The processed DataFrame.
    - num_features (int): Number of features to normalize.

    Returns:
    - pd.DataFrame: DataFrame with normalized features.

    The default column format is:
    Open, High, Low, Close, DR, SMA5, SMA20, SMA60, RSI, Trend
    """
    if num_features < 1:
        raise ValueError("Number of features must be at least 1.")

    # Selecting the specified number of features for scaling
    training_set = df.iloc[:, 3:3 + num_features].values

    # Scaling training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training_set = scaler.fit_transform(training_set)

    # Scale the target variable if more than one feature is being used
    if num_features > 1:
        scale_target_variable(df)

    return scaled_training_set

def scale_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the target variable in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the target variable.

    Returns:
    - pd.DataFrame: DataFrame with the scaled target variable.
    """
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    # Assuming the last column is the target variable (the "trend")
    target_feature = df.iloc[:, -1].values.reshape(-1, 1)
    scaled_target_feature = target_scaler.fit_transform(target_feature)

    return scaled_target_feature

def prepare_training_data(df: pd.DataFrame, num_features: int, window_size: int = 60) -> (np.ndarray, np.ndarray):
    """
    Prepares training data from a DataFrame and reshapes it for LSTM model training.

    Example usage:  Assuming df is your DataFrame and num_features is the number 
                    of features in your dataset
                    
                    `X_train, y_train = prepare_training_data(df, num_features)`
    Parameters:
    df (pd.DataFrame): DataFrame containing the feature set.
    num_features (int): Number of features in the dataset.
    window_size (int): The size of the sliding window to create the sequences, default is 60.

    Returns:
    tuple: Tuple containing X_train (training features) and y_train (target variable).
    """

    # Split the data (using the previously defined data_split function)
    train_df, _, _ = data_split(df)

    # Scaling the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training_set = scaler.fit_transform(train_df.iloc[:, :num_features])

    # Preparing training data
    X_train, y_train = [], []
    for i in range(window_size, len(scaled_training_set)):
        X_train.append(scaled_training_set[i - window_size:i])
        y_train.append(scaled_training_set[i, 0])  # Assuming target variable is at index 0
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))

    return X_train, y_train
