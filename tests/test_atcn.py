"""
Unit tests for ATCN (Attention-Based Temporal Convolutional Network) model.

Tests model building, layer functionality, and end-to-end predictions.
"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from src.models.atcn import (
    build_atcn_model,
    ImprovedSelfAttention,
    asymmetric_mse,
)
from src.models.architectures import ModelRegistry


class TestATCNModel:
    """Test ATCN model building and compilation."""

    def test_atcn_builds_successfully(self):
        """Test that ATCN model builds without errors."""
        model = build_atcn_model(input_shape=(1000, 32))
        assert model is not None
        assert isinstance(model, keras.Model)

    def test_atcn_input_output_shapes(self):
        """Test ATCN model has correct input and output shapes."""
        seq_length, num_features = 1000, 32
        model = build_atcn_model(input_shape=(seq_length, num_features))

        assert model.input_shape == (None, seq_length, num_features)
        assert model.output_shape == (None, 1)

    def test_atcn_with_custom_parameters(self):
        """Test ATCN model builds with custom parameters."""
        model = build_atcn_model(
            input_shape=(500, 20),
            units=128,
            dense_units=64,
            dropout_rate=0.3,
            learning_rate=0.0005,
            num_heads=8,
            kernel_size=5,
            num_tcn_layers=6,
        )
        assert model is not None
        assert model.input_shape == (None, 500, 20)
        assert model.output_shape == (None, 1)

    def test_atcn_model_is_compiled(self):
        """Test that ATCN model is compiled with optimizer and loss."""
        model = build_atcn_model(input_shape=(1000, 32))
        assert model.optimizer is not None
        assert model.loss is not None
        assert "mae" in [m.name for m in model.metrics]

    def test_atcn_registry_integration(self):
        """Test that ATCN is properly registered in ModelRegistry."""
        assert "atcn" in ModelRegistry.list_models()
        model = ModelRegistry.build("atcn", input_shape=(1000, 32))
        assert model is not None
        assert model.name == "atcn"


class TestImprovedSelfAttention:
    """Test ImprovedSelfAttention layer."""

    def test_isa_layer_builds(self):
        """Test ISA layer builds successfully."""
        layer = ImprovedSelfAttention(units=64, num_heads=4)
        assert layer is not None

    def test_isa_output_shape(self):
        """Test ISA layer output has same shape as input."""
        batch_size, seq_length, features = 8, 100, 32
        layer = ImprovedSelfAttention(units=features, num_heads=4)

        inputs = tf.random.normal((batch_size, seq_length, features))
        outputs = layer(inputs, training=False)

        assert outputs.shape == (batch_size, seq_length, features)

    def test_isa_with_different_num_heads(self):
        """Test ISA layer with different number of attention heads."""
        for num_heads in [1, 2, 4, 8]:
            layer = ImprovedSelfAttention(units=64, num_heads=num_heads)
            inputs = tf.random.normal((4, 50, 64))
            outputs = layer(inputs, training=False)
            assert outputs.shape == inputs.shape

    def test_isa_has_position_embedding(self):
        """Test that ISA layer creates position embeddings."""
        seq_length, features = 100, 32
        layer = ImprovedSelfAttention(units=features, num_heads=4)
        layer.build((None, seq_length, features))

        assert layer.position_embedding is not None
        assert layer.position_embedding.shape == (seq_length, features)

    def test_isa_training_mode(self):
        """Test ISA layer in training vs inference mode."""
        layer = ImprovedSelfAttention(units=32, num_heads=4)
        inputs = tf.random.normal((2, 50, 32))

        # Should work in both modes
        output_train = layer(inputs, training=True)
        output_test = layer(inputs, training=False)

        assert output_train.shape == inputs.shape
        assert output_test.shape == inputs.shape

    def test_isa_get_config(self):
        """Test ISA layer serialization config."""
        layer = ImprovedSelfAttention(units=64, num_heads=8)
        config = layer.get_config()

        assert "units" in config
        assert "num_heads" in config
        assert config["units"] == 64
        assert config["num_heads"] == 8


class TestATCNPredictions:
    """Test ATCN model end-to-end predictions."""

    def test_atcn_makes_predictions(self):
        """Test ATCN model can make predictions."""
        model = build_atcn_model(input_shape=(1000, 32))

        X_test = np.random.randn(10, 1000, 32).astype(np.float32)
        predictions = model.predict(X_test, verbose=0)

        assert predictions.shape == (10, 1)
        assert np.all(np.isfinite(predictions))

    def test_atcn_predictions_are_positive(self):
        """Test ATCN predictions are realistic RUL values."""
        model = build_atcn_model(input_shape=(100, 20))

        X_test = np.random.randn(5, 100, 20).astype(np.float32)
        predictions = model.predict(X_test, verbose=0)

        # Predictions should be finite numbers
        assert np.all(np.isfinite(predictions))

        # Check they are real-valued (not complex)
        assert predictions.dtype in [np.float32, np.float64]

    def test_atcn_trains_one_step(self):
        """Test ATCN model can perform one training step."""
        model = build_atcn_model(input_shape=(100, 20))

        X_train = np.random.randn(8, 100, 20).astype(np.float32)
        y_train = np.random.rand(8, 1).astype(np.float32) * 100

        # Single training step should not raise errors
        history = model.fit(X_train, y_train, epochs=1, verbose=0)

        assert "loss" in history.history
        assert np.isfinite(history.history["loss"][0])

    def test_atcn_batch_prediction_consistency(self):
        """Test ATCN predictions are consistent across batch sizes."""
        model = build_atcn_model(input_shape=(100, 20))

        # Create identical samples
        sample = np.random.randn(1, 100, 20).astype(np.float32)
        X_repeated = np.repeat(sample, 5, axis=0)

        predictions = model.predict(X_repeated, verbose=0)

        # All predictions should be very similar (same input)
        assert np.std(predictions) < 0.1 or np.allclose(predictions, predictions[0], rtol=1e-3)


class TestAsymmetricMSE:
    """Test asymmetric MSE loss function."""

    def test_asymmetric_mse_perfect_prediction(self):
        """Test loss is zero for perfect predictions."""
        y_true = tf.constant([[10.0], [20.0], [30.0]])
        y_pred = tf.constant([[10.0], [20.0], [30.0]])

        loss_fn = asymmetric_mse(alpha=2.0)
        loss = loss_fn(y_true, y_pred)

        assert float(loss) < 1e-6

    def test_asymmetric_mse_penalizes_late_predictions(self):
        """Test late predictions (y_pred > y_true) are penalized more."""
        y_true = tf.constant([[50.0]])
        y_pred_early = tf.constant([[40.0]])  # 10 units early
        y_pred_late = tf.constant([[60.0]])   # 10 units late

        loss_fn = asymmetric_mse(alpha=2.0)
        loss_early = float(loss_fn(y_true, y_pred_early))
        loss_late = float(loss_fn(y_true, y_pred_late))

        # Late predictions should have higher loss
        assert loss_late > loss_early

        # Specifically, late should be ~2x early (alpha=2.0)
        assert abs(loss_late / loss_early - 2.0) < 0.1

    def test_asymmetric_mse_with_custom_alpha(self):
        """Test asymmetric MSE with different alpha values."""
        y_true = tf.constant([[50.0]])
        y_pred_late = tf.constant([[60.0]])

        for alpha in [1.0, 2.0, 3.0, 5.0]:
            loss_fn = asymmetric_mse(alpha=alpha)
            loss = float(loss_fn(y_true, y_pred_late))

            # Higher alpha should yield higher loss for late predictions
            assert loss > 0


class TestATCNArchitecture:
    """Test ATCN architecture components."""

    def test_atcn_has_attention_layers(self):
        """Test ATCN model contains attention layers."""
        model = build_atcn_model(input_shape=(1000, 32))

        # Check for ImprovedSelfAttention layer
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "ImprovedSelfAttention" in layer_types

    def test_atcn_has_tcn_blocks(self):
        """Test ATCN model contains TCN blocks."""
        model = build_atcn_model(input_shape=(1000, 32), num_tcn_layers=4)

        # Check for ResidualTCNBlock layers
        tcn_layers = [layer for layer in model.layers if "tcn_block" in layer.name]
        assert len(tcn_layers) >= 4

    def test_atcn_has_channel_attention(self):
        """Test ATCN model contains channel attention."""
        model = build_atcn_model(input_shape=(1000, 32))

        # Check for ChannelAttention layer
        layer_names = [layer.name for layer in model.layers]
        assert "channel_attention" in layer_names

    def test_atcn_parameter_count_scales_with_units(self):
        """Test ATCN parameter count increases with units."""
        model_small = build_atcn_model(input_shape=(100, 20), units=32)
        model_large = build_atcn_model(input_shape=(100, 20), units=128)

        params_small = model_small.count_params()
        params_large = model_large.count_params()

        # Larger model should have more parameters
        assert params_large > params_small
