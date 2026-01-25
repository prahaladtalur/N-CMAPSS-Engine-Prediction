.PHONY: install install-dev lint format typecheck check train clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install all dependencies including dev tools"
	@echo "  make lint         - Check code formatting with black"
	@echo "  make format       - Auto-format code with black"
	@echo "  make typecheck    - Run mypy type checking"
	@echo "  make check        - Run all checks (lint + typecheck)"
	@echo "  make train        - Train default model (LSTM)"
	@echo "  make clean        - Remove cache and build artifacts"

# Installation
install:
	uv sync --no-dev

install-dev:
	uv sync

# Code quality
lint:
	black --check src/ train_model.py scripts/

format:
	black src/ train_model.py scripts/

typecheck:
	mypy src/ train_model.py

check: lint typecheck

# Training
train:
	python train_model.py --model lstm --epochs 30

# Cleanup
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__ src/**/__pycache__
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
