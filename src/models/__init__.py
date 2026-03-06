"""
Models for RUL prediction.

This package provides a ModelRegistry system for easy model switching.
All models are automatically registered on import.
"""

import sys
from pathlib import Path

# Add project root to path for train_model imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import registry and base first
from .registry import ModelRegistry
from .base import BaseModel, asymmetric_mse

# Import all model categories to register them
from . import rnn  # LSTM, BiLSTM, GRU, BiGRU
from . import cnn  # TCN, WaveNet
from . import attention  # AttentionLSTM, Transformer, ResNetLSTM
from . import hybrid  # CNN-LSTM, CNN-GRU, InceptionLSTM
from . import baseline  # MLP
from . import sota  # MDFA, CNN-LSTM-Attention, MSTCN, CATA-TCN, TTSNet, ATCN, Sparse Transformer


# Utility functions for easier model access
def get_model(
    name: str,
    input_shape: tuple,
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    **kwargs,
):
    """
    Build a model by name with given configuration.

    Args:
        name: Model name (see list_available_models())
        input_shape: (timesteps, features)
        units: Number of units in recurrent/temporal layers
        dense_units: Number of units in dense layers
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        **kwargs: Additional model-specific parameters

    Returns:
        Compiled Keras model

    Example:
        >>> model = get_model('mstcn', input_shape=(1000, 32), epochs=30)
    """
    return ModelRegistry.build(
        name=name,
        input_shape=input_shape,
        units=units,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        **kwargs,
    )


def list_available_models():
    """
    List all registered model names.

    Returns:
        Sorted list of model names
    """
    return ModelRegistry.list_models()


def get_model_info():
    """
    Get descriptions of all available models.

    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        # RNN models
        "lstm": "Standard LSTM - baseline recurrent model",
        "bilstm": "Bidirectional LSTM - captures past and future context",
        "gru": "GRU - simpler than LSTM, often faster",
        "bigru": "Bidirectional GRU",
        # Attention models
        "attention_lstm": "LSTM with additive attention mechanism",
        "transformer": "Multi-head self-attention encoder",
        "resnet_lstm": "LSTM with residual connections",
        # CNN models
        "tcn": "Temporal Convolutional Network - dilated causal convolutions",
        "wavenet": "WaveNet-style gated activations",
        # Hybrid models
        "cnn_lstm": "CNN feature extraction + LSTM temporal modeling",
        "cnn_gru": "CNN + GRU (more stable than CNN-LSTM)",
        "inception_lstm": "Multi-scale CNN + LSTM",
        # SOTA models
        "mdfa": "Multi-scale Dilated Fusion Attention (2025 SOTA)",
        "cnn_lstm_attention": "CNN-LSTM with self-attention (2024)",
        "mstcn": "Multi-Scale TCN + Global Fusion Attention (WINNER: RMSE 6.80)",
        "cata_tcn": "Channel & Temporal Attention TCN",
        "ttsnet": "Transformer + TCN + Self-Attention ensemble",
        "atcn": "Attention-based TCN",
        "sparse_transformer_bigrcu": "Sparse Transformer with Bi-GRCU",
        # Baseline
        "mlp": "Multi-Layer Perceptron baseline (no temporal modeling)",
    }


def get_model_recommendations():
    """
    Get model recommendations by use case.

    Returns:
        Dictionary mapping use cases to recommended models
    """
    return {
        "production": ["mstcn", "transformer", "wavenet"],  # Top 3, within 1% of each other
        "best_single": ["mstcn"],  # Winner: RMSE 6.80, R² 0.90
        "fast_training": ["gru", "bigru", "wavenet"],  # Quick to train
        "interpretable": ["attention_lstm", "transformer"],  # Attention weights visible
        "baseline": ["mlp", "lstm"],  # Simple baselines for comparison
        "research": ["mstcn", "atcn", "cata_tcn", "ttsnet"],  # Latest SOTA models
    }


# Lazy imports for training functions (avoid circular dependencies)
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
    """Deprecated: Use train_model() instead."""
    from train_model import train_lstm as _train_lstm
    return _train_lstm(*args, **kwargs)


__all__ = [
    # Core registry
    "ModelRegistry",
    "BaseModel",
    "asymmetric_mse",
    # Utility functions
    "get_model",
    "list_available_models",
    "get_model_info",
    "get_model_recommendations",
    # Training functions
    "prepare_sequences",
    "train_model",
    "compare_models",
    "train_lstm",  # Deprecated
]
