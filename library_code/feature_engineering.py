import pandas as pd
import numpy as np

def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'DR' (Daily Return) column to the DataFrame. 
    The formula to calculate it is: DR = Close[t] / Close[t-1] - 1

    Parameters:
    df (pd.DataFrame): DataFrame with a 'Close' column representing closing prices.

    Returns:
    pd.DataFrame: DataFrame with an additional 'DR' column.
    """
    df['DR'] = df['Close'].pct_change()
    return df

def add_simple_moving_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three simple moving averages (SMA) to the DataFrame, which are common
    in stock analysis.

    Adds the following columns:
    - SMA5: 5-day simple moving average of the 'Close' values.
    - SMA20: 20-day simple moving average of the 'Close' values.
    - SMA60: 60-day simple moving average of the 'Close' values.
    
    Check following article for further understanding of SMA:
    - https://www.investopedia.com/terms/s/sma.asp

    Parameters:
    - df (pd.DataFrame): DataFrame with a 'Close' column.

    Returns:
    - pd.DataFrame: Modified DataFrame with additional SMA columns.
    """
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()

    return df

def add_relative_strength_index(df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
    """
    Calculate and add the Relative Strength Index (RSI) to the DataFrame.

    Check following article for further understanding of RSI:
    - https://www.investopedia.com/terms/r/rsi.asp

    Parameters:
    df (pd.DataFrame): DataFrame with a 'Close' column.
    periods (int): Number of periods to use for RSI calculation, default is 14.

    Returns:
    pd.DataFrame: DataFrame with an additional 'RSI' column.
    """
    # Calculate daily price changes
    df['Price_Change'] = df['Close'].diff()

    # Calculate gains (positive price changes) and losses (negative price changes)
    df['Gain'] = df['Price_Change'].clip(lower=0)
    df['Loss'] = -df['Price_Change'].clip(upper=0)

    # Calculate the average gain and average loss over the specified period
    avg_gain = df['Gain'].rolling(window=periods, min_periods=periods).mean()
    avg_loss = df['Loss'].rolling(window=periods, min_periods=periods).mean()

    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    df['RSI'] = 100 - (100 / (1 + rs))

    # Handle initial NaN values
    df['RSI'][:periods] = np.nan

    # Drop intermediate columns used for calculations
    df.drop(columns=['Price_Change', 'Gain', 'Loss'], inplace=True)

    return df
