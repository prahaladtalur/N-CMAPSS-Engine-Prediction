"""Models for RUL prediction."""

import sys
from pathlib import Path

from src.models.base import BaseModel, asymmetric_mse
from src.models.registry import ModelRegistry

# Add project root to path to import train_model when needed.
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import all model categories so decorators register them.
from src.models import attention  # noqa: F401
from src.models import baseline  # noqa: F401
from src.models import cnn  # noqa: F401
from src.models import hybrid  # noqa: F401
from src.models import rnn  # noqa: F401
from src.models import sota  # noqa: F401


def get_model(
    model_name: str,
    input_shape: tuple[int, int],
    **kwargs,
):
    """Get a model by name with the given configuration."""
    return ModelRegistry.build(model_name, input_shape=input_shape, **kwargs)


def list_available_models() -> list:
    """List all available model architectures."""
    return ModelRegistry.list_models()


def get_model_info() -> dict[str, str]:
    """Get descriptions of all available models."""
    return {
        "lstm": "Standard LSTM - baseline RNN for sequence modeling",
        "bilstm": "Bidirectional LSTM - captures both past and future context",
        "gru": "GRU - simpler than LSTM, often faster with similar performance",
        "bigru": "Bidirectional GRU - bidirectional version of GRU",
        "attention_lstm": "LSTM with Attention - focuses on important timesteps",
        "resnet_lstm": "ResNet-LSTM - residual connections for better gradient flow",
        "tcn": "Temporal Convolutional Network - SOTA, parallelizable, large receptive field",
        "wavenet": "WaveNet - gated dilated convolutions, excellent for long sequences",
        "cnn_lstm": "CNN-LSTM hybrid - CNN extracts features, LSTM models time",
        "cnn_gru": "CNN-GRU hybrid - CNN + GRU, faster than CNN-LSTM",
        "inception_lstm": "Inception-LSTM - multi-scale feature extraction",
        "transformer": "Transformer encoder - self-attention based, very SOTA",
        "mdfa": "MDFA - Multi-scale dilated fusion attention (paper SOTA, RMSE_norm 0.021-0.032)",
        "cnn_lstm_attention": "CNN-LSTM-Attention - 2024 SOTA (CMAPSS RMSE 13.907-16.637)",
        "cata_tcn": "CATA-TCN - Channel+Temporal Attention over TCN backbone",
        "ttsnet": "TTSNet - Transformer+TCN+Self-Attention late-fusion hybrid",
        "atcn": "ATCN - Attention-based TCN with ISA and squeeze-excitation (2023 SOTA)",
        "sparse_transformer_bigrcu": "Sparse Transformer+Bi-GRCU - LRLS attention, most recent (2025 SOTA)",
        "mstcn": "MSTCN - Multi-scale TCN with Global Fusion Attention (2024 SOTA)",
        "mlp": "Simple MLP - baseline for comparison (no temporal modeling)",
    }


def get_model_recommendations() -> dict[str, list]:
    """Get model recommendations for different use cases."""
    return {
        "quick_baseline": ["mlp", "gru"],
        "best_accuracy": [
            "sparse_transformer_bigrcu",
            "mstcn",
            "ttsnet",
            "atcn",
            "cata_tcn",
            "cnn_lstm_attention",
            "mdfa",
            "transformer",
            "attention_lstm",
        ],
        "fastest_training": ["gru", "cnn_gru", "tcn"],
        "most_interpretable": ["lstm", "attention_lstm"],
        "long_sequences": [
            "sparse_transformer_bigrcu",
            "mstcn",
            "ttsnet",
            "atcn",
            "mdfa",
            "tcn",
            "wavenet",
            "transformer",
        ],
        "limited_data": ["gru", "lstm"],
        "complex_patterns": [
            "sparse_transformer_bigrcu",
            "mstcn",
            "ttsnet",
            "atcn",
            "cata_tcn",
            "cnn_lstm_attention",
            "mdfa",
            "transformer",
            "wavenet",
        ],
    }


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


__all__ = [
    "BaseModel",
    "ModelRegistry",
    "asymmetric_mse",
    "prepare_sequences",
    "train_model",
    "compare_models",
    "get_model",
    "list_available_models",
    "get_model_info",
    "get_model_recommendations",
    "train_lstm",
]
