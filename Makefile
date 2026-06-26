.PHONY: install install-dev lint format typecheck test check train paper clean help

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
	@echo "  make train        - Train a standard MSTCN run"
	@echo "  make paper        - Build the LaTeX paper"
	@echo "  make clean        - Remove cache and build artifacts"

# Installation
install:
	uv sync --no-dev

install-dev:
	uv sync

# Code quality
lint:
	uv run black --check src/ tests/ train_model.py scripts/

format:
	uv run black src/ tests/ train_model.py scripts/

typecheck:
	uv run mypy src/ train_model.py

test:
	WANDB_MODE=offline uv run pytest tests/ -v

check: lint typecheck test

# Training
train:
	WANDB_MODE=offline uv run python train_model.py --model mstcn --fd 1 --epochs 30 --batch-size 32 --max-seq-length 1000

paper:
	$(MAKE) -C paper

# Cleanup
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__ src/**/__pycache__
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
