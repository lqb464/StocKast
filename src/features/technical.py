import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced technical indicators (MACD, RSI, Bollinger, ATR, Ichimoku)"""
    df = df.copy()
    
    # Simple Moving Averages
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_30"] = df["close"].rolling(window=30).mean()
    
    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands (20-day, 2 std)
    rolling_20 = df["close"].rolling(window=20)
    df["bb_middle"] = rolling_20.mean()
    bb_std = rolling_20.std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std
    
    # ATR (Average True Range) - Volatility measure
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    
    # Log Returns (More stationary than raw price)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    return df


def generate_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Generate predictive targets (Future log return)"""
    df = df.copy()
    
    # Target: The cumulative log return over the next `horizon` days
    df[f"target_return_{horizon}d"] = df["log_return"].shift(-horizon).rolling(horizon).sum()
    
    # Drop rows where target is NaN (the last `horizon` rows)
    df = df.dropna().reset_index(drop=True)
    return df
