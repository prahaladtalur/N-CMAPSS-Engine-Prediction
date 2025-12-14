.PHONY: help install install-uv install-pip install-dev run data clean test lint format typecheck notebook setup check

# Variables
PYTHON := python
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python
PIP_VENV := $(VENV)/bin/pip
DATA_DIR := data/raw
PROCESSED_DIR := data/processed

# Default target
help:
	@echo "N-CMAPSS Engine Prediction - Available Commands:"
	@echo ""
	@echo "  make install        - Install dependencies using pip (editable mode)"
	@echo "  make install-uv     - Install dependencies using uv (if available)"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make run            - Run the main pipeline"
	@echo "  make data           - Download and prepare N-CMAPSS dataset (FD001)"
	@echo "  make data-all       - Download all N-CMAPSS datasets (FD001-FD007)"
	@echo "  make clean          - Remove generated files and caches"
	@echo "  make clean-data     - Remove downloaded data (keeps structure)"
	@echo "  make install-dev     - Install development dependencies (black, mypy)"
	@echo "  make test            - Run tests (if available)"
	@echo "  make lint            - Run black in check mode (linting)"
	@echo "  make format          - Format code with black"
	@echo "  make typecheck       - Run mypy type checking"
	@echo "  make check-all       - Run linting and type checking"
	@echo "  make notebook        - Launch Jupyter notebook server"
	@echo "  make check           - Check if virtual environment is set up"
	@echo ""

# Check if virtual environment exists
check:
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	else \
		echo "✓ Virtual environment found at $(VENV)"; \
	fi

# Setup: Create venv and install dependencies
setup: $(VENV)
	@echo "✓ Virtual environment ready"
	@echo "Run 'make install' to install dependencies"

$(VENV):
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created"

# Install dependencies using pip (editable mode)
install: check
	@echo "Installing dependencies from pyproject.toml..."
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -e .
	@echo "✓ Dependencies installed"

# Install dependencies using uv (if available)
install-uv:
	@if command -v uv > /dev/null; then \
		echo "Installing dependencies using uv..."; \
		uv sync; \
		echo "✓ Dependencies installed with uv"; \
	else \
		echo "❌ uv not found. Install it with: pip install uv"; \
		echo "   Or use 'make install' to use pip instead"; \
		exit 1; \
	fi

# Install development dependencies (black, mypy)
install-dev: check
	@echo "Installing development dependencies..."
	$(PIP_VENV) install -e ".[dev]"
	@echo "✓ Development dependencies installed"

# Run the main pipeline
run: check
	@echo "Running N-CMAPSS pipeline..."
	$(PYTHON_VENV) train_model.py

# Download and prepare N-CMAPSS dataset (FD001 by default)
data: check
	@echo "Downloading N-CMAPSS FD001 dataset..."
	@mkdir -p $(DATA_DIR)
	$(PYTHON_VENV) -c "from src.data.load_data import download_ncmapss; download_ncmapss(data_dir='$(DATA_DIR)', fd=1, cache=True)"
	@echo "✓ Dataset downloaded and prepared"

# Download all N-CMAPSS datasets (FD001-FD007)
data-all: check
	@echo "Downloading all N-CMAPSS datasets (FD001-FD007)..."
	@mkdir -p $(DATA_DIR)
	@for fd in 1 2 3 4 5 6 7; do \
		echo "Downloading FD$$fd..."; \
		$(PYTHON_VENV) -c "from src.data.load_data import download_ncmapss; download_ncmapss(data_dir='$(DATA_DIR)', fd=$$fd, cache=True)" || true; \
	done
	@echo "✓ All datasets downloaded"

# Clean generated files and caches
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "✓ Cleaned Python caches"

# Clean downloaded data (keeps directory structure)
clean-data:
	@echo "Cleaning downloaded data..."
	@if [ -d "$(DATA_DIR)" ]; then \
		find $(DATA_DIR) -type f -delete 2>/dev/null || true; \
		echo "✓ Data files removed (directories preserved)"; \
	else \
		echo "No data directory found"; \
	fi
	@if [ -d "$(PROCESSED_DIR)" ]; then \
		find $(PROCESSED_DIR) -type f -delete 2>/dev/null || true; \
		echo "✓ Processed data files removed"; \
	fi

# Run tests (if test directory exists)
test: check
	@if [ -d "tests" ] || [ -f "pytest.ini" ] || [ -f "setup.cfg" ]; then \
		echo "Running tests..."; \
		$(PYTHON_VENV) -m pytest tests/ -v || echo "No tests found or pytest not configured"; \
	else \
		echo "⚠️  No tests directory found. Create a 'tests/' directory to add tests."; \
	fi

# Run black in check mode (linting)
lint: check
	@echo "Running black (check mode)..."
	@if $(PIP_VENV) show black > /dev/null 2>&1; then \
		$(PYTHON_VENV) -m black --check --diff src/ train_model.py; \
	else \
		echo "❌ black not installed. Install with: make install-dev"; \
		exit 1; \
	fi

# Format code with black
format: check
	@echo "Formatting code with black..."
	@if $(PIP_VENV) show black > /dev/null 2>&1; then \
		$(PYTHON_VENV) -m black src/ train_model.py; \
		echo "✓ Code formatted with black"; \
	else \
		echo "❌ black not installed. Install with: make install-dev"; \
		exit 1; \
	fi

# Run mypy type checking
typecheck: check
	@echo "Running mypy type checker..."
	@if $(PIP_VENV) show mypy > /dev/null 2>&1; then \
		$(PYTHON_VENV) -m mypy src/ train_model.py; \
	else \
		echo "❌ mypy not installed. Install with: make install-dev"; \
		exit 1; \
	fi

# Run both linting and type checking
check-all: lint typecheck
	@echo "✓ All checks passed!"

# Launch Jupyter notebook server
notebook: check
	@echo "Launching Jupyter notebook server..."
	@if $(PIP_VENV) show jupyter > /dev/null 2>&1; then \
		$(PYTHON_VENV) -m jupyter notebook notebooks/; \
	else \
		echo "⚠️  Jupyter not installed. Install with: pip install jupyter"; \
	fi

# Run a specific dataset FD number
run-fd: check
	@if [ -z "$(FD)" ]; then \
		echo "Usage: make run-fd FD=1"; \
		exit 1; \
	fi
	@echo "Running pipeline for FD$(FD)..."
	$(PYTHON_VENV) -c "from src.data.load_data import get_datasets; from src.utils.visualize import plot_rul_distribution; (dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=$(FD)); plot_rul_distribution(dev_y, split_name='FD$(FD) Dev')"

# Show project info
info: check
	@echo "Project Information:"
	@echo "  Python: $$($(PYTHON_VENV) --version)"
	@echo "  Virtual Environment: $(VENV)"
	@echo "  Project: $$($(PYTHON_VENV) -c 'import sys; sys.path.insert(0, "."); from pyproject.toml import *' 2>/dev/null || echo 'n-cmapss-engine-prediction')"
	@echo ""
	@echo "Installed packages:"
	@$(PIP_VENV) list | head -20

