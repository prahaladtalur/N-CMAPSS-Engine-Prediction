"""Models for RUL prediction."""

from src.models.trainer import train_lstm, train_model, compare_models, prepare_sequences
from src.models.architectures import (
    ModelRegistry,
    get_model,
    list_available_models,
    get_model_info,
    get_model_recommendations,
)

__all__ = [
    # Training functions
    "train_lstm",
    "train_model",
    "compare_models",
    "prepare_sequences",
    # Model registry
    "ModelRegistry",
    "get_model",
    "list_available_models",
    "get_model_info",
    "get_model_recommendations",
]
