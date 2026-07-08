import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch raw stock data from Yahoo Finance."""
    logger.info(f"Fetching data for {ticker} over {period}")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}. May be delisted or invalid.")
        
    # Standardize column names
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    # Rename 'date' to 'datetime' if necessary, but keep it standard as 'date'
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None) # Remove timezone for simplicity
        
    return df[["date", "open", "high", "low", "close", "volume"]]
