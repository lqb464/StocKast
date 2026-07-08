.PHONY: help install notebooks train smoke test mlflow-ui lint format clean

PYTHON := python
NB_DIR := notebooks

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -e ".[dev,notebooks]"

notebooks:  ## Launch Jupyter Lab
	jupyter lab $(NB_DIR)/

train:  ## Train the full ensemble model (LSTM + XGBoost)
	$(PYTHON) scripts/train.py

smoke:  ## Run a quick smoke test on truncated data
	$(PYTHON) scripts/train.py --smoke-test

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing

mlflow-ui:  ## Start MLflow tracking UI
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

lint:  ## Lint with flake8
	flake8 src/ scripts/ tests/ backend/ --max-line-length=88 --extend-ignore=E203,W503

format:  ## Format with black + isort
	isort src/ scripts/ tests/ backend/
	black src/ scripts/ tests/ backend/

clean:  ## Clean cached files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
