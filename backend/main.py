"""
main.py — FastAPI REST Microservice for Stock Price Forecasting.

Endpoints:
    GET  /health               -> System status & latest MLflow metrics
    POST /predict              -> Predict next day close price from input features
    GET  /fetch                -> Fetch raw historical stock prices via yfinance
    POST /predict/ticker       -> Real-time yfinance data fetch & price forecast for ticker
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.src.data import FEATURE_NAMES, fetch_stock_data, load_and_prepare_stock_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "training" / "outputs" / "model.joblib"
METRICS_PATH = BASE_DIR / "training" / "docs" / "assets" / "run_summary.json"

_model = None


def get_model():
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            try:
                _model = joblib.load(MODEL_PATH)
            except Exception as e:
                print(f"[API] Error loading model: {e}", flush=True)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(
    title="Stock Price Forecasting API",
    description=(
        "Production REST microservice for stock price forecasting using XGBoost time series regression. "
        "Supports real-time yfinance data fetching, technical indicator feature extraction, and next-day price prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StockFeatures(BaseModel):
    open: float = Field(default=180.0, gt=0, description="Opening price")
    high: float = Field(default=185.0, gt=0, description="Highest price")
    low: float = Field(default=178.0, gt=0, description="Lowest price")
    close: float = Field(default=182.0, gt=0, description="Closing price")
    volume: float = Field(default=50000000.0, ge=0, description="Trading volume")
    SMA_10: float = Field(default=180.0, gt=0, description="10-day Simple Moving Average")
    SMA_30: float = Field(default=175.0, gt=0, description="30-day Simple Moving Average")
    RSI_14: float = Field(default=55.0, ge=0, le=100, description="14-day RSI score")
    daily_return: float = Field(default=0.01, description="Daily return ratio")
    volatility_10: float = Field(default=0.015, ge=0, description="10-day volatility standard deviation")
    lag_1: float = Field(default=181.0, gt=0, description="1-day lag close price")
    lag_2: float = Field(default=180.0, gt=0, description="2-day lag close price")
    lag_5: float = Field(default=178.0, gt=0, description="5-day lag close price")
    lag_10: float = Field(default=175.0, gt=0, description="10-day lag close price")


class ForecastResponse(BaseModel):
    current_close: float
    predicted_next_close: float
    expected_change: float
    expected_change_pct: float
    trend_signal: str


class TickerRequest(BaseModel):
    ticker: str = Field(default="AAPL", description="Stock ticker symbol (e.g. AAPL, MSFT, GOOGL)")
    period: Optional[str] = Field(default="2y", description="Historical period (e.g. 1y, 2y, 5y)")


@app.get("/health", tags=["System"])
def health():
    metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        except Exception:
            pass
    return {
        "status": "healthy",
        "model_ready": MODEL_PATH.exists(),
        "latest_metrics": metrics,
    }


@app.post("/predict", response_model=ForecastResponse, tags=["Forecasting"])
def predict(features: StockFeatures):
    model = get_model()

    feature_dict = features.model_dump()
    X_input = pd.DataFrame([feature_dict])[FEATURE_NAMES]

    if model is not None:
        predicted_close = float(model.predict(X_input)[0])
    else:
        predicted_close = round(features.close * (1.0 + features.daily_return * 0.5), 2)

    current_close = features.close
    change = round(predicted_close - current_close, 2)
    change_pct = round((change / current_close) * 100, 2)

    if change_pct > 0.5:
        signal = "BULLISH 📈"
    elif change_pct < -0.5:
        signal = "BEARISH 📉"
    else:
        signal = "NEUTRAL ➖"

    return ForecastResponse(
        current_close=current_close,
        predicted_next_close=round(predicted_close, 2),
        expected_change=change,
        expected_change_pct=change_pct,
        trend_signal=signal,
    )


@app.get("/fetch", tags=["Data"])
def fetch_data(
    ticker: str = Query(default="AAPL", description="Ticker symbol"),
    period: str = Query(default="1y", description="Time period"),
):
    try:
        df = fetch_stock_data(ticker=ticker, period=period)
        records = df.tail(30).to_dict(orient="records")
        for r in records:
            if "date" in r and isinstance(r["date"], pd.Timestamp):
                r["date"] = r["date"].strftime("%Y-%m-%d")
        return {"ticker": ticker, "count": len(records), "data": records}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/ticker", response_model=ForecastResponse, tags=["Forecasting"])
def predict_by_ticker(req: TickerRequest):
    try:
        df_featured = load_and_prepare_stock_data(ticker=req.ticker, period=req.period)
        if df_featured.empty:
            raise HTTPException(status_code=400, detail="Insufficient historical data for features.")

        latest_row = df_featured.iloc[-1]
        feat = StockFeatures(
            open=float(latest_row["open"]),
            high=float(latest_row["high"]),
            low=float(latest_row["low"]),
            close=float(latest_row["close"]),
            volume=float(latest_row["volume"]),
            SMA_10=float(latest_row["SMA_10"]),
            SMA_30=float(latest_row["SMA_30"]),
            RSI_14=float(latest_row["RSI_14"]),
            daily_return=float(latest_row["daily_return"]),
            volatility_10=float(latest_row["volatility_10"]),
            lag_1=float(latest_row["lag_1"]),
            lag_2=float(latest_row["lag_2"]),
            lag_5=float(latest_row["lag_5"]),
            lag_10=float(latest_row["lag_10"]),
        )
        return predict(feat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error for '{req.ticker}': {str(e)}")
