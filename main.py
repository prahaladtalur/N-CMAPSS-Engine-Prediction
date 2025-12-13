#!/usr/bin/env python
"""
Main entry point for N-CMAPSS RUL prediction pipeline.

This is a convenience wrapper around train_model.py that adds:
- Environment variable loading (.env file)
- Optional data visualization step before training

For full CLI options, use train_model.py directly.
"""

import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv is optional

# Import and run train_model's main function
# This allows main.py to be used as an alias for train_model.py
if __name__ == "__main__":
    # Import here to execute train_model.py's main
    import train_model
    train_model.main()
