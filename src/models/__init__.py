"""Models for RUL prediction."""

from src.models.train import train_model, compare_models
from src.models.architectures import (
    ModelRegistry,
    get_model,
    list_available_models,
    get_model_info,
)

# Legacy API - deprecated, use train_model() with model_name="lstm" instead
from src.models.train import train_lstm

__all__ = [
    # Main API
    "train_model",
    "compare_models",
    "ModelRegistry",
    "get_model",
    "list_available_models",
    "get_model_info",
    # Legacy (deprecated)
    "train_lstm",
]
