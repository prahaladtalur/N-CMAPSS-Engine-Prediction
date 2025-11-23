"""
LSTM model for RUL prediction.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Build an LSTM model for RUL prediction.

    Args:
        input_shape: Shape of input data (timesteps, features)
        lstm_units: Number of LSTM units
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units // 2, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="linear"),  # Linear activation for regression
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mape"],
    )

    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get model summary as string.

    Args:
        model: Keras model

    Returns:
        Model summary string
    """
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    return f.getvalue()

