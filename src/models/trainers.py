import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

def build_xgboost_pipeline(random_state: int = 42) -> Pipeline:
    """Build a robust XGBoost regression pipeline for time series."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

def train_rnn_model(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, lr: float = 0.01):
    """
    Train a PyTorch RNN model for sequence prediction.
    Expects 2D array inputs for X, we will reshape it to (samples, seq_len=1, features).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=2):
            super(SimpleRNN, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x shape: (batch, seq, feature)
            out, _ = self.rnn(x)
            # Take the output of the last time step
            out = self.fc(out[:, -1, :])
            return out

    logger.info("Training PyTorch RNN...")
    
    # Prepare data
    if len(X_train.shape) == 2:
        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) # (N, 1, F)
    else:
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    input_size = X_train_t.shape[2]
    model = SimpleRNN(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
    return model

def predict_rnn(model, X_test: np.ndarray) -> np.ndarray:
    """Predict using PyTorch RNN."""
    import torch
    model.eval()
    if len(X_test.shape) == 2:
        X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    else:
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        
    with torch.no_grad():
        preds = model(X_test_t)
    return preds.numpy().flatten()
