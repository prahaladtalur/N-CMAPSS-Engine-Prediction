"""
Model registry for easy model switching.

The ModelRegistry class uses the decorator pattern to register model classes,
enabling dynamic model selection by name at runtime.
"""

from typing import Dict, Tuple
from tensorflow import keras


class ModelRegistry:
    """Registry for easy model switching."""

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class.

        Usage:
            @ModelRegistry.register("my_model")
            class MyModel(BaseModel):
                ...
        """

        def decorator(model_class):
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a model class by name."""
        if name not in cls._models:
            available = ", ".join(sorted(cls._models.keys()))
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return sorted(list(cls._models.keys()))

    @classmethod
    def build(
        cls,
        name: str,
        input_shape: Tuple[int, int],
        **kwargs,
    ) -> keras.Model:
        """Build a model by name with given configuration."""
        model_class = cls.get(name)
        return model_class.build(input_shape, **kwargs)
