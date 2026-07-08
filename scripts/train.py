import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.loader import fetch_stock_data
from src.features.technical import add_technical_indicators, generate_targets
from src.models.trainers import build_xgboost_pipeline, train_rnn_model, predict_rnn
from src.models.backtester import TradingBacktester

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "StocKast-XGBoost-RNN"

def train(ticker: str, period: str, smoke_test: bool):
    logger.info("=== Phase 1: Data Preparation ===")
    df = fetch_stock_data(ticker, period)
    df = add_technical_indicators(df)
    df = generate_targets(df, horizon=1)
    
    if smoke_test:
        df = df.tail(200).reset_index(drop=True)
        logger.info(f"[Smoke Test] Truncated to {len(df)} rows.")

    # Time series train/test split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    features = ["sma_10", "sma_30", "rsi_14", "macd", "bb_upper", "bb_lower", "atr_14", "log_return"]
    X_train, y_train = train_df[features], train_df["target_return_1d"]
    X_test, y_test = test_df[features], test_df["target_return_1d"]
    
    logger.info("=== Phase 2: Ensemble Model Training ===")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run():
        # Model 1: XGBoost
        logger.info("Training XGBoost...")
        xgb_model = build_xgboost_pipeline()
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        
        # Model 2: RNN
        rnn_model = train_rnn_model(X_train.values, y_train, epochs=20 if smoke_test else 100)
        rnn_preds = predict_rnn(rnn_model, X_test.values)
        
        # Ensemble predictions (50/50 blend)
        logger.info("Blending Predictions...")
        test_df = test_df.copy()
        test_df["pred_log_return"] = (xgb_preds + rnn_preds) / 2.0
        
        logger.info("=== Phase 3: Backtesting & Evaluation ===")
        backtester = TradingBacktester(initial_capital=10000, transaction_cost=0.001)
        metrics = backtester.run(test_df, pred_col="pred_log_return")
        
        logger.info(f"Backtest Metrics: {metrics}")
        
        # Log to MLflow
        mlflow.log_params({"ticker": ticker, "period": period, "smoke": smoke_test})
        mlflow.log_metrics({
            "roi_pct": metrics["Total ROI (%)"],
            "sharpe": metrics["Sharpe Ratio"],
            "max_drawdown": metrics["Max Drawdown (%)"]
        })
        
        # Save model
        mlflow.sklearn.log_model(xgb_model, "xgboost_model", skops_trusted_types=["xgboost.sklearn.XGBRegressor", "xgboost.core.Booster"])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    train(args.ticker, args.period, args.smoke_test)
