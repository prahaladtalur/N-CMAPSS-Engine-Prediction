.PHONY: install install-dev lint format typecheck test check train list-runs list-runs-json clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install all dependencies including dev tools"
	@echo "  make lint         - Check code formatting with black"
	@echo "  make format       - Auto-format code with black"
	@echo "  make typecheck    - Run mypy type checking"
	@echo "  make test         - Run unit tests with pytest"
	@echo "  make check        - Run all checks (lint + typecheck + test)"
	@echo "  make train        - Train default model (LSTM)"
	@echo "  make list-runs    - Show local checkpoints and W&B runs"
	@echo "  make list-runs-json - Emit the local run index as JSON"
	@echo "  make clean        - Remove cache and build artifacts"

# Installation
install:
	uv sync --no-dev

install-dev:
	uv sync

# Code quality
lint:
	black --check src/ tests/ train_model.py scripts/

format:
	black src/ tests/ train_model.py scripts/

typecheck:
	mypy src/ train_model.py

test:
	pytest tests/ -v

check: lint typecheck test

# Training
train:
	python train_model.py --model lstm --epochs 30

list-runs:
	uv run python scripts/list_local_runs.py

list-runs-json:
	uv run python scripts/list_local_runs.py --json

# Cleanup
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__ src/**/__pycache__
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
