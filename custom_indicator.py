import numpy as np
import pandas as pd

def SMA(values, period):
    """Calculate Simple Moving Average (SMA)"""
    return values.rolling(window=period).mean()

# Define RSI (Smoothed Using EMA)
def calculate_rsi(df, column="Close", period=14):
    delta = df[column].diff(1)  # Calculate daily price change

    gain = np.where(delta > 0, delta, 0)  # Only positive gains
    loss = np.where(delta < 0, -delta, 0)  # Only negative losses

    # Exponential Moving Average (EMA) instead of simple rolling mean
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df["RSI14"] = rsi  # Assign RSI back to DataFrame
    return df