import numpy as np
import pandas as pd

class TradingBacktester:
    """
    Simulates a trading strategy based on model predictions.
    Calculates quantitative metrics like ROI, Sharpe Ratio, and Max Drawdown.
    """
    
    def __init__(self, initial_capital: float = 10000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def run(self, df: pd.DataFrame, pred_col: str = "pred_log_return") -> dict:
        """
        Runs the backtest.
        Strategy: Go Long (buy) if predicted log_return > 0, else Go Short (sell/hold).
        For simplicity, we'll assume a long-only strategy where we invest 100% of capital 
        if signal is positive, and hold cash if signal is negative.
        """
        # Signal: 1 if we expect positive return, 0 if negative
        df["signal"] = (df[pred_col] > 0).astype(int)
        
        # Calculate daily returns of the strategy
        # Strategy return = Signal * Actual return - Transaction costs when signal changes
        df["position_change"] = df["signal"].diff().abs().fillna(0)
        
        # Calculate the actual return we got (log return converted to simple return)
        df["actual_simple_return"] = np.exp(df["log_return"]) - 1
        
        # Gross return
        df["strategy_gross_return"] = df["signal"].shift(1) * df["actual_simple_return"]
        
        # Net return after transaction costs
        df["strategy_net_return"] = df["strategy_gross_return"] - (df["position_change"] * self.transaction_cost)
        df["strategy_net_return"] = df["strategy_net_return"].fillna(0)
        
        # Calculate Portfolio Value over time
        df["portfolio_value"] = self.initial_capital * (1 + df["strategy_net_return"]).cumprod()
        
        # Calculate Metrics
        final_value = df["portfolio_value"].iloc[-1]
        total_roi_pct = ((final_value / self.initial_capital) - 1) * 100
        
        # Annualized Sharpe Ratio (assuming 252 trading days)
        daily_returns = df["strategy_net_return"]
        if daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
        else:
            sharpe_ratio = 0.0
            
        # Maximum Drawdown
        running_max = df["portfolio_value"].cummax()
        drawdown = (df["portfolio_value"] - running_max) / running_max
        max_drawdown = drawdown.min() * 100 # In percent
        
        metrics = {
            "Total ROI (%)": round(total_roi_pct, 2),
            "Sharpe Ratio": round(sharpe_ratio, 3),
            "Max Drawdown (%)": round(max_drawdown, 2),
            "Final Portfolio Value": round(final_value, 2)
        }
        
        return metrics
