"""Compatibility shim for legacy imports.

Model implementations now live in focused modules under ``src.models``.
"""

from src.models import (
    BaseModel,
    ModelRegistry,
    asymmetric_mse,
    get_model,
    get_model_info,
    get_model_recommendations,
    list_available_models,
)
from src.models.base import compile_model_for_training, get_loss_function

__all__ = [
    "BaseModel",
    "ModelRegistry",
    "asymmetric_mse",
    "compile_model_for_training",
    "get_loss_function",
    "get_model",
    "get_model_info",
    "get_model_recommendations",
    "list_available_models",
]
