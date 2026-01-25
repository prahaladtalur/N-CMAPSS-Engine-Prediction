"""Models for RUL prediction."""

# Import training functions lazily to avoid circular imports during CLI startup.
import sys
from pathlib import Path

# Add project root to path to import train_model when needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def prepare_sequences(*args, **kwargs):
    from train_model import prepare_sequences as _prepare_sequences

    return _prepare_sequences(*args, **kwargs)


def train_model(*args, **kwargs):
    from train_model import train_model as _train_model

    return _train_model(*args, **kwargs)


def compare_models(*args, **kwargs):
    from train_model import compare_models as _compare_models

    return _compare_models(*args, **kwargs)


def train_lstm(*args, **kwargs):
    from train_model import train_lstm as _train_lstm

    return _train_lstm(*args, **kwargs)


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
