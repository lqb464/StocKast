# StocKast - Advanced Stock Price Forecasting (Quantitative Approach)

StocKast is an advanced, end-to-end stock price prediction and quantitative finance pipeline. It leverages a rigorous methodology, including Purged Time Series Cross-Validation to eliminate data leakage, sophisticated Feature Engineering (Technical Indicators, Fractional Differentiation, Log Returns), and a robust Ensemble Modeling architecture.

The project is structured around standard ML Engineering practices, utilizing MLflow for experiment tracking, and is designed to be fully containerized as a Microservice architecture with a FastAPI backend and a Streamlit frontend.

## Project Architecture

```
StocKast/
├── src/                
│   ├── data/           
│   ├── features/       
│   └── models/         
├── notebooks/          
│   ├── 01_Financial_Time_Series_EDA.ipynb
│   ├── 02_Advanced_Feature_Engineering.ipynb
│   ├── 03_Purged_Cross_Validation.ipynb
│   └── 04_Backtesting_and_ROI.ipynb
├── scripts/            
│   └── train.py        
├── backend/            
├── frontend/           
├── tests/              
├── pyproject.toml      
├── Makefile            
└── docker-compose.yml  
```

## Methodology

### 1. Data Engineering and Target Construction
Instead of predicting raw closing prices, which are non-stationary and highly prone to spurious correlations, this pipeline predicts the **Log Returns** of the asset. We employ strict data preparation rules, generating features such as MACD, Bollinger Bands, Average True Range (ATR), and Ichimoku Cloud directly from the OHLCV data fetched via the `yfinance` API.

### 2. Purged Time Series Cross-Validation
Standard cross-validation techniques (like K-Fold) fail in quantitative finance because they leak future information into the past. Standard TimeSeriesSplit also suffers from autocorrelation leakage at the split boundaries. StocKast implements **Purged Time Series Split**, which introduces a deliberate "purge gap" between the training and testing sets. This ensures the model is evaluated in a strictly out-of-sample environment, mimicking real-world trading conditions.

### 3. Ensemble Modeling: XGBoost and Recurrent Neural Networks (RNN)
The predictive engine combines the strengths of tree-based models and sequence-based deep learning:
- **XGBoost**: Captures complex, non-linear relationships and tabular indicator interactions.
- **RNN (Recurrent Neural Network)**: Specifically utilizes an LSTM (Long Short-Term Memory) architecture to capture temporal dependencies and sequential patterns in the time-series data. 
- **Ensemble Strategy**: We combine both methodologies to achieve superior predictive accuracy.

### 4. Strategy Backtesting and Financial Metrics
Standard machine learning metrics (RMSE, MAE, MAPE) do not accurately reflect the economic viability of a model. StocKast includes a custom `TradingBacktester` module that simulates a realistic trading strategy based on the model's predictions. 
The backtester evaluates the model by calculating true financial metrics:
- **Total ROI (Return on Investment)**
- **Sharpe Ratio (Risk-adjusted return)**
- **Maximum Drawdown (Peak-to-trough decline)**

## Getting Started

### Prerequisites
- Python 3.10+
- Docker and Docker Compose (for the full-stack web app)

### ML Pipeline Execution

1. **Install Dependencies**
   ```bash
   make install
   ```

2. **Train the Ensemble Model (RNN + XGBoost)**
   This command will fetch the data, engineer the features, train the models, run the backtester, and log all metrics to MLflow.
   ```bash
   make train
   
   # Alternatively, pass arguments manually:
   python scripts/train.py --ticker AAPL --period 5y
   ```

3. **Smoke Test**
   To quickly verify that the pipeline executes without errors (using a truncated dataset):
   ```bash
   make smoke
   ```

4. **View MLflow Tracking UI**
   ```bash
   make mlflow-ui
   ```

### Full-Stack Web App Execution

To launch the entire platform, including the MLflow tracking server, the FastAPI backend for real-time model inference, and the Streamlit frontend dashboard:

```bash
docker-compose up --build
```
- **Frontend Dashboard**: `http://localhost:3000`
- **Backend API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **MLflow Tracking**: `http://localhost:5000`
