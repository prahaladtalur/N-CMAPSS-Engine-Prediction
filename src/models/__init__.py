"""Models for RUL prediction."""

from src.models.lstm_model import build_lstm_model
from src.models.train import train_lstm, train_model, compare_models, prepare_sequences
from src.models.architectures import (
    ModelRegistry,
    get_model,
    list_available_models,
    get_model_info,
)

__all__ = [
    # Legacy
    "build_lstm_model",
    "train_lstm",
    "prepare_sequences",
    # New API
    "train_model",
    "compare_models",
    "ModelRegistry",
    "get_model",
    "list_available_models",
    "get_model_info",
]
