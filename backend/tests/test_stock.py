"""
test_stock.py — Pytest unit & integration test suite for Stock Price Prediction system.
"""

import json
from pathlib import Path
import pytest
import sys
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import backend.main as A
import training.pipeline as P
from backend.src.data import load_and_prepare_stock_data


def test_load_stock_data():
    df = load_and_prepare_stock_data(ticker="AAPL", period="1y")
    assert not df.empty
    assert "close" in df.columns
    assert "target_close" in df.columns
    assert "SMA_10" in df.columns
    assert "RSI_14" in df.columns


def test_train_pipeline_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(P, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(P, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(P, "METRICS_PATH", tmp_path / "run_summary.json")

    pipeline, metrics = P.train(ticker="AAPL", period="1y", smoke_test=True)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "run_summary.json").exists()
    assert "test_rmse" in metrics


def test_api_health():
    client = TestClient(A.app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
