import numpy as np
import pandas as pd
from src.features.technical import add_technical_indicators, generate_targets

def test_add_technical_indicators():
    df = pd.DataFrame({
        "close": [100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100],
        "high": [101, 102, 103, 102, 101, 100, 99, 98, 99, 100, 101],
        "low": [99, 100, 101, 100, 99, 98, 97, 96, 97, 98, 99]
    })
    
    out = add_technical_indicators(df)
    assert "rsi_14" in out.columns
    assert "macd" in out.columns
    assert "bb_upper" in out.columns
    assert "log_return" in out.columns

def test_generate_targets():
    df = pd.DataFrame({
        "log_return": [0.01, 0.02, -0.01, 0.03]
    })
    
    out = generate_targets(df, horizon=1)
    assert len(out) == 3
    assert out["target_return_1d"].iloc[0] == 0.02
