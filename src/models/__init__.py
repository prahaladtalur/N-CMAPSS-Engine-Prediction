"""Models for RUL prediction."""

# Import training functions from train_model.py (functions moved there from train.py)
import sys
from pathlib import Path

# Add project root to path to import train_model
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from train_model import train_model, compare_models, prepare_sequences, train_lstm

from src.models.architectures import (
    ModelRegistry,
    get_model,
    list_available_models,
    get_model_info,
    get_model_recommendations,
)

__all__ = [
    # Main API
    "prepare_sequences",
    "train_model",
    "compare_models",
    "ModelRegistry",
    "get_model",
    "list_available_models",
    "get_model_info",
    "get_model_recommendations",
    # Legacy (deprecated)
    "train_lstm",
]
