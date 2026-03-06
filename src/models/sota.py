"""
State-of-the-Art (SOTA) models for RUL prediction.

Wrapper classes that register the SOTA models and delegate to their
respective build functions in separate files.

Includes:
- MDFA: Multi-scale Dilated Fusion Attention
- CNN-LSTM-Attention: 2024 SOTA architecture
- MSTCN: Multi-Scale TCN + Global Fusion Attention (WINNER)
- CATA-TCN: Channel & Temporal Attention TCN
- TTSNet: Transformer + TCN + Self-Attention ensemble
- ATCN: Attention-based TCN
- Sparse Transformer + Bi-GRCU
"""

from typing import Tuple, Dict, Any
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseModel
from .registry import ModelRegistry

# Import build functions from separate files
from .mdfa import MDFAModule
from .cnn_lstm_attention import build_cnn_lstm_attention_model
from .cata_tcn import build_cata_tcn_model
from .ttsnet import build_ttsnet_model
from .atcn import build_atcn_model
from .sparse_transformer_bigrcu import build_sparse_transformer_bigrcu_model
from .mstcn import build_mstcn_model

@ModelRegistry.register("mdfa")
class MDFAModel(BaseModel):
    """
    Multi-Scale Dilated Fusion Attention (MDFA) model.

    State-of-the-art architecture for RUL prediction from:
    "Remaining Useful Life Prediction for Aero-Engines Based on Multi-Scale Dilated Fusion Attention Model"
    https://www.mdpi.com/2076-3417/15/17/9813

    Achieves RMSE_norm 0.021-0.032 on N-CMAPSS (vs our current 0.098).

    Architecture:
        1. MDFA module with multi-scale dilated convolutions [1, 2, 4, 8]
        2. Channel attention (emphasizes important sensors)
        3. Spatial attention (focuses on critical time windows)
        4. BiLSTM for temporal modeling
        5. Dense output layer

    Key innovations:
        - Multi-scale receptive fields: 3, 5, 9, 17 timesteps
        - Global pooling branch captures sequence-level degradation trends
        - Dual attention suppresses noise and highlights informative patterns
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        dilation_rates: list = None,
    ) -> keras.Model:
        """
        Build MDFA model.

        Args:
            input_shape: (timesteps, features)
            units: Number of units for MDFA module and BiLSTM
            dense_units: Units for final dense layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            dilation_rates: Dilation rates for multi-scale branches (default: [1, 2, 4, 8])

        Returns:
            Compiled Keras model
        """
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        inputs = layers.Input(shape=input_shape)

        # MDFA feature extraction with multi-scale dilated convolutions + attention
        x = MDFAModule(
            filters=units, dilation_rates=dilation_rates, kernel_size=3, dropout_rate=dropout_rate
        )(inputs)

        # Batch normalization for stability
        x = layers.BatchNormalization()(x)

        # BiLSTM for temporal modeling (bidirectional to capture both past and future context)
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=False))(x)
        x = layers.Dropout(dropout_rate)(x)

        # Dense layers
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="mdfa")
        return BaseModel.compile_model(model, learning_rate)


@ModelRegistry.register("cnn_lstm_attention")
class CNNLSTMAttentionModel(BaseModel):
    """
    CNN-LSTM-Attention model for RUL prediction.

    State-of-the-art architecture from:
    "Prediction of Remaining Useful Life of Aero-engines Based on CNN-LSTM-Attention" (2024)
    https://link.springer.com/article/10.1007/s44196-024-00639-w

    Achieves RMSE 13.907-16.637 on CMAPSS datasets (FD001-FD004).

    Architecture:
        1. CNN layers (64→128→256 filters) for local feature extraction
        2. Stacked LSTM (2 layers) for temporal modeling
        3. Self-attention mechanism to focus on critical timesteps
        4. Dense layers for RUL prediction

    Key advantages:
        - Simpler than MDFA → faster training
        - Proven results on CMAPSS benchmark
        - Effective for capturing both local patterns (CNN) and temporal dependencies (LSTM)
        - Attention highlights when degradation accelerates
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 128,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        cnn_filters: list = None,
    ) -> keras.Model:
        """
        Build CNN-LSTM-Attention model.

        Args:
            input_shape: (timesteps, features)
            units: Number of LSTM units (default: 128)
            dense_units: Units for final dense layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            cnn_filters: CNN filter counts (default: [64, 128, 256])

        Returns:
            Compiled Keras model
        """
        if cnn_filters is None:
            cnn_filters = [64, 128, 256]

        # Use the specialized builder function
        model = build_cnn_lstm_attention_model(
            input_shape=input_shape,
            cnn_filters=cnn_filters,
            lstm_units=units,
            attention_units=units // 2,  # Attention units = half of LSTM units
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )

        return model


@ModelRegistry.register("cata_tcn")
class CATATCNModel(BaseModel):
    """
    CATA-TCN model (Channel-and-Temporal Attention TCN).

    Architecture pattern from dual-attention TCN literature for RUL:
    - Dilated residual TCN backbone
    - Channel attention to reweight important sensors
    - Temporal attention to highlight critical degradation windows
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        kernel_size: int = 3,
        num_layers: int = 4,
    ) -> keras.Model:
        return build_cata_tcn_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )


@ModelRegistry.register("ttsnet")
class TTSNetModel(BaseModel):
    """
    TTSNet model (Transformer + TCN + Self-Attention fusion).

    Hybrid late-fusion design inspired by recent top-performing RUL papers:
    - Transformer branch for global dependencies
    - TCN branch for multiscale local temporal features
    - Self-attention recurrent branch for salient sequence dynamics
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        kernel_size: int = 3,
    ) -> keras.Model:
        return build_ttsnet_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            kernel_size=kernel_size,
        )


@ModelRegistry.register("atcn")
class ATCNModel(BaseModel):
    """
    ATCN model (Attention-Based Temporal Convolutional Network).

    State-of-the-art architecture from:
    "An attention-based temporal convolutional network method for predicting
     remaining useful life of aero-engine" (2023)

    Architecture:
        1. Improved Self-Attention (ISA) with learnable position embeddings
        2. TCN blocks with exponential dilation [1, 2, 4, 8]
        3. Squeeze-Excitation channel attention
        4. Dense layers for RUL prediction

    Key advantages:
        - Dual attention (temporal + channel) for comprehensive feature weighting
        - TCN backbone for long-term dependencies
        - Simpler than multi-branch architectures
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        kernel_size: int = 3,
        num_tcn_layers: int = 4,
    ) -> keras.Model:
        return build_atcn_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            num_heads=num_heads,
            kernel_size=kernel_size,
            num_tcn_layers=num_tcn_layers,
        )


@ModelRegistry.register("sparse_transformer_bigrcu")
class SparseTransformerBiGRCUModel(BaseModel):
    """
    Sparse Transformer with Bi-GRCU ensemble model.

    State-of-the-art architecture from:
    "A sequence ensemble method based on Sparse Transformer with bidirectional
     gated recurrent convolution unit" (2025)

    Architecture:
        1. Bi-GRCU branch: Gated fusion of Bi-GRU + Conv1D (short-term)
        2. Sparse Transformer branch: LRLS-Attention (long-term, efficient)
        3. Late ensemble fusion via concatenation
        4. Dense layers for RUL prediction

    Key innovations:
        - LRLS-Attention: O(T×(k+g)) complexity vs O(T²) full attention
        - Bi-GRCU: Learnable gating between recurrent and convolutional features
        - Dual-branch ensemble captures both short and long-term patterns
        - Most recent architecture (2025)
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        local_window: int = 32,
        num_global_tokens: int = 8,
    ) -> keras.Model:
        return build_sparse_transformer_bigrcu_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            local_window=local_window,
            num_global_tokens=num_global_tokens,
        )


@ModelRegistry.register("mstcn")
class MSTCNModel(BaseModel):
    """
    MSTCN model (Multi-Scale Temporal Convolutional Network with Global Fusion Attention).

    State-of-the-art architecture from:
    "An attention-based multi-scale temporal convolutional network for remaining
     useful life prediction" (2024)

    Architecture:
        1. Multi-scale TCN with parallel branches (dilations: 1, 2, 4, 8)
        2. Global Fusion Attention (GFA): channel + temporal + cross-scale
        3. Adaptive gating to suppress redundant multi-scale information
        4. Dense layers for RUL prediction

    Key advantages:
        - Multi-scale captures patterns at different temporal resolutions
        - GFA intelligently fuses scales (vs simple concatenation in MDFA)
        - Adaptive gating reduces information redundancy
        - Efficient for long sequences
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        kernel_size: int = 3,
        dilation_rates: list = None,
    ) -> keras.Model:
        return build_mstcn_model(
            input_shape=input_shape,
            units=units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            kernel_size=kernel_size,
            dilation_rates=dilation_rates,
        )


def get_model(
    model_name: str,
    input_shape: Tuple[int, int],
    **kwargs,
) -> keras.Model:
    """
    Get a model by name with given configuration.

    Args:
        model_name: Name of the model (lstm, bilstm, gru, bigru, attention_lstm, tcn, cnn_lstm, transformer)
        input_shape: Shape of input data (timesteps, features)
        **kwargs: Model-specific arguments (units, dense_units, dropout_rate, learning_rate)

    Returns:
        Compiled Keras model
    """
    return ModelRegistry.build(model_name, input_shape, **kwargs)


def list_available_models() -> list:
    """List all available model architectures."""
    return ModelRegistry.list_models()


def get_model_info() -> Dict[str, str]:
    """Get descriptions of all available models."""
    return {
        # RNN-based models
        "lstm": "Standard LSTM - baseline RNN for sequence modeling",
        "bilstm": "Bidirectional LSTM - captures both past and future context",
        "gru": "GRU - simpler than LSTM, often faster with similar performance",
        "bigru": "Bidirectional GRU - bidirectional version of GRU",
        "attention_lstm": "LSTM with Attention - focuses on important timesteps",
        "resnet_lstm": "ResNet-LSTM - residual connections for better gradient flow",
        # Convolutional models
        "tcn": "Temporal Convolutional Network - SOTA, parallelizable, large receptive field",
        "wavenet": "WaveNet - gated dilated convolutions, excellent for long sequences",
        # Hybrid models
        "cnn_lstm": "CNN-LSTM hybrid - CNN extracts features, LSTM models time",
        "cnn_gru": "CNN-GRU hybrid - CNN + GRU, faster than CNN-LSTM",
        "inception_lstm": "Inception-LSTM - multi-scale feature extraction",
        # Attention-based models
        "transformer": "Transformer encoder - self-attention based, very SOTA",
        "mdfa": "MDFA - Multi-scale dilated fusion attention (paper SOTA, RMSE_norm 0.021-0.032)",
        "cnn_lstm_attention": "CNN-LSTM-Attention - 2024 SOTA (CMAPSS RMSE 13.907-16.637)",
        "cata_tcn": "CATA-TCN - Channel+Temporal Attention over TCN backbone",
        "ttsnet": "TTSNet - Transformer+TCN+Self-Attention late-fusion hybrid",
        "atcn": "ATCN - Attention-based TCN with ISA and squeeze-excitation (2023 SOTA)",
        "sparse_transformer_bigrcu": "Sparse Transformer+Bi-GRCU - LRLS attention, most recent (2025 SOTA)",
        "mstcn": "MSTCN - Multi-scale TCN with Global Fusion Attention (2024 SOTA)",
        # Baseline
        "mlp": "Simple MLP - baseline for comparison (no temporal modeling)",
    }


def get_model_recommendations() -> Dict[str, list]:
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
        "long_sequences": ["sparse_transformer_bigrcu", "mstcn", "ttsnet", "atcn", "mdfa", "tcn", "wavenet", "transformer"],
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
