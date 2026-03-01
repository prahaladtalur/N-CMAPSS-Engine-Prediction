"""
Unit tests for Sparse Transformer + Bi-GRCU model.

Tests model building, BiGRCU layer, LRLS attention mechanism, and ensemble fusion.
"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from src.models.sparse_transformer_bigrcu import (
    build_sparse_transformer_bigrcu_model,
    BiGRCU,
    LRLSAttention,
    asymmetric_mse,
)
from src.models.architectures import ModelRegistry


class TestSparseTransformerBiGRCUModel:
    """Test Sparse Transformer + Bi-GRCU model building and compilation."""

    def test_model_builds_successfully(self):
        """Test that model builds without errors."""
        model = build_sparse_transformer_bigrcu_model(input_shape=(1000, 32))
        assert model is not None
        assert isinstance(model, keras.Model)

    def test_model_input_output_shapes(self):
        """Test model has correct input and output shapes."""
        seq_length, num_features = 1000, 32
        model = build_sparse_transformer_bigrcu_model(input_shape=(seq_length, num_features))

        assert model.input_shape == (None, seq_length, num_features)
        assert model.output_shape == (None, 1)

    def test_model_with_custom_parameters(self):
        """Test model builds with custom parameters."""
        model = build_sparse_transformer_bigrcu_model(
            input_shape=(500, 20),
            units=128,
            dense_units=64,
            dropout_rate=0.3,
            learning_rate=0.0005,
            num_heads=8,
            num_transformer_layers=3,
            local_window=64,
            num_global_tokens=16,
        )
        assert model is not None
        assert model.input_shape == (None, 500, 20)

    def test_model_is_compiled(self):
        """Test that model is compiled with optimizer and loss."""
        model = build_sparse_transformer_bigrcu_model(input_shape=(1000, 32))
        assert model.optimizer is not None
        assert model.loss is not None
        assert "mae" in [m.name for m in model.metrics]

    def test_model_registry_integration(self):
        """Test that model is properly registered in ModelRegistry."""
        assert "sparse_transformer_bigrcu" in ModelRegistry.list_models()
        model = ModelRegistry.build("sparse_transformer_bigrcu", input_shape=(1000, 32))
        assert model is not None
        assert model.name == "sparse_transformer_bigrcu"


class TestBiGRCU:
    """Test Bidirectional Gated Recurrent Convolution Unit layer."""

    def test_bigrcu_layer_builds(self):
        """Test BiGRCU layer builds successfully."""
        layer = BiGRCU(units=64, kernel_size=3)
        assert layer is not None

    def test_bigrcu_output_shape(self):
        """Test BiGRCU layer output shape matches Bi-GRU (units*2)."""
        batch_size, seq_length, features = 8, 100, 32
        units = 64
        layer = BiGRCU(units=units, kernel_size=3)

        inputs = tf.random.normal((batch_size, seq_length, features))
        outputs = layer(inputs, training=False)

        # Bi-GRU outputs units*2 features (forward + backward)
        assert outputs.shape == (batch_size, seq_length, units * 2)

    def test_bigrcu_with_different_kernel_sizes(self):
        """Test BiGRCU with different Conv1D kernel sizes."""
        for kernel_size in [3, 5, 7]:
            layer = BiGRCU(units=64, kernel_size=kernel_size)
            inputs = tf.random.normal((4, 50, 32))
            outputs = layer(inputs, training=False)
            assert outputs.shape == (4, 50, 128)  # units*2 = 64*2

    def test_bigrcu_gated_fusion(self):
        """Test BiGRCU applies gated fusion between RNN and CNN."""
        layer = BiGRCU(units=32, kernel_size=3)
        inputs = tf.random.normal((2, 50, 20))

        outputs = layer(inputs, training=False)

        # Output should be combination of RNN and CNN features
        assert outputs.shape == (2, 50, 64)  # 32*2
        assert tf.reduce_all(tf.math.is_finite(outputs))

    def test_bigrcu_training_mode(self):
        """Test BiGRCU layer in training vs inference mode."""
        layer = BiGRCU(units=32, kernel_size=3)
        inputs = tf.random.normal((2, 50, 20))

        output_train = layer(inputs, training=True)
        output_test = layer(inputs, training=False)

        assert output_train.shape == output_test.shape
        assert output_train.shape == (2, 50, 64)

    def test_bigrcu_get_config(self):
        """Test BiGRCU layer serialization config."""
        layer = BiGRCU(units=64, kernel_size=5)
        config = layer.get_config()

        assert "units" in config
        assert "kernel_size" in config
        assert config["units"] == 64
        assert config["kernel_size"] == 5


class TestLRLSAttention:
    """Test Long-Range Locality Sparse Attention mechanism."""

    def test_lrls_attention_builds(self):
        """Test LRLS attention layer builds successfully."""
        layer = LRLSAttention(num_heads=4, local_window=32, num_global_tokens=8)
        assert layer is not None

    def test_lrls_output_shape(self):
        """Test LRLS attention output shape matches input."""
        batch_size, seq_length, features = 8, 100, 64
        layer = LRLSAttention(num_heads=4, local_window=32, num_global_tokens=8)

        inputs = tf.random.normal((batch_size, seq_length, features))
        outputs = layer(inputs, training=False)

        assert outputs.shape == (batch_size, seq_length, features)

    def test_lrls_sparse_mask_creation(self):
        """Test LRLS creates correct sparse attention mask."""
        seq_length = 100
        local_window = 32
        num_global_tokens = 8

        layer = LRLSAttention(
            num_heads=4, local_window=local_window, num_global_tokens=num_global_tokens
        )
        mask = layer.create_sparse_attention_mask(seq_length)

        assert mask.shape == (seq_length, seq_length)
        assert mask.dtype == tf.float32

        # Check that mask has some sparse structure (not all 1.0)
        total_ones = tf.reduce_sum(mask)
        full_attention_size = seq_length * seq_length

        # Sparse mask should have significantly fewer 1s than full attention
        assert total_ones < full_attention_size * 0.5

    def test_lrls_local_window_pattern(self):
        """Test LRLS local window attention pattern."""
        seq_length = 50
        local_window = 10
        layer = LRLSAttention(num_heads=4, local_window=local_window, num_global_tokens=0)

        mask = layer.create_sparse_attention_mask(seq_length)

        # For position i, should be able to attend to [i-5, i+5] (local_window=10)
        mid_pos = seq_length // 2
        local_radius = local_window // 2

        # Check that mid_pos can attend to nearby positions
        for offset in range(-local_radius, local_radius + 1):
            neighbor = mid_pos + offset
            if 0 <= neighbor < seq_length:
                assert mask[mid_pos, neighbor] == 1.0

    def test_lrls_global_tokens_pattern(self):
        """Test LRLS global token attention pattern."""
        seq_length = 50
        num_global_tokens = 5
        layer = LRLSAttention(num_heads=4, local_window=0, num_global_tokens=num_global_tokens)

        mask = layer.create_sparse_attention_mask(seq_length)

        # All positions should be able to attend to first num_global_tokens positions
        for i in range(seq_length):
            for g in range(num_global_tokens):
                assert mask[i, g] == 1.0

    def test_lrls_complexity_reduction(self):
        """Test LRLS attention has reduced complexity vs full attention."""
        seq_length = 200
        local_window = 32
        num_global_tokens = 8

        layer = LRLSAttention(
            num_heads=4, local_window=local_window, num_global_tokens=num_global_tokens
        )
        mask = layer.create_sparse_attention_mask(seq_length)

        # Count attended positions per query
        attended_per_query = tf.reduce_sum(mask, axis=1)
        avg_attended = tf.reduce_mean(attended_per_query)

        # Sparse attention should attend to much fewer positions than full (seq_length)
        # Expected: roughly local_window + num_global_tokens
        expected_attended = local_window + num_global_tokens

        # Allow some tolerance
        assert avg_attended < seq_length * 0.5
        assert avg_attended >= expected_attended * 0.5

    def test_lrls_with_different_num_heads(self):
        """Test LRLS attention with different numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            layer = LRLSAttention(num_heads=num_heads, local_window=32, num_global_tokens=8)
            inputs = tf.random.normal((4, 50, 64))
            outputs = layer(inputs, training=False)
            assert outputs.shape == inputs.shape

    def test_lrls_training_mode(self):
        """Test LRLS attention in training vs inference mode."""
        layer = LRLSAttention(num_heads=4, local_window=32, num_global_tokens=8)
        inputs = tf.random.normal((2, 50, 64))

        output_train = layer(inputs, training=True)
        output_test = layer(inputs, training=False)

        assert output_train.shape == inputs.shape
        assert output_test.shape == inputs.shape

    def test_lrls_get_config(self):
        """Test LRLS attention serialization config."""
        layer = LRLSAttention(
            num_heads=8, local_window=64, num_global_tokens=16, dropout_rate=0.15
        )
        config = layer.get_config()

        assert config["num_heads"] == 8
        assert config["local_window"] == 64
        assert config["num_global_tokens"] == 16
        assert config["dropout_rate"] == 0.15


class TestModelPredictions:
    """Test Sparse Transformer + Bi-GRCU end-to-end predictions."""

    def test_model_makes_predictions(self):
        """Test model can make predictions."""
        model = build_sparse_transformer_bigrcu_model(input_shape=(1000, 32))

        X_test = np.random.randn(10, 1000, 32).astype(np.float32)
        predictions = model.predict(X_test, verbose=0)

        assert predictions.shape == (10, 1)
        assert np.all(np.isfinite(predictions))

    def test_model_trains_one_step(self):
        """Test model can perform one training step."""
        model = build_sparse_transformer_bigrcu_model(input_shape=(100, 20))

        X_train = np.random.randn(8, 100, 20).astype(np.float32)
        y_train = np.random.rand(8, 1).astype(np.float32) * 100

        history = model.fit(X_train, y_train, epochs=1, verbose=0)

        assert "loss" in history.history
        assert np.isfinite(history.history["loss"][0])

    def test_ensemble_fusion(self):
        """Test model properly fuses Bi-GRCU and Sparse Transformer branches."""
        model = build_sparse_transformer_bigrcu_model(input_shape=(100, 20))

        # Check that model has both BiGRCU and LRLS layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "BiGRCU" in layer_types or any("bigrcu" in layer.name for layer in model.layers)
        assert any("lrls_attention" in layer.name for layer in model.layers)

        # Check for fusion/concatenate layer
        layer_names = [layer.name for layer in model.layers]
        assert any("fusion" in name or "concatenate" in name for name in layer_names)


class TestModelArchitecture:
    """Test Sparse Transformer + Bi-GRCU architecture components."""

    def test_model_has_dual_branches(self):
        """Test model has both Bi-GRCU and Sparse Transformer branches."""
        model = build_sparse_transformer_bigrcu_model(input_shape=(100, 20))

        layer_names = [layer.name for layer in model.layers]

        # Check for BiGRCU branch
        assert any("bigrcu" in name for name in layer_names)

        # Check for Sparse Transformer branch
        assert any("lrls_attention" in name for name in layer_names)

    def test_model_has_pooling_layers(self):
        """Test model has global pooling for both branches."""
        model = build_sparse_transformer_bigrcu_model(input_shape=(100, 20))

        layer_names = [layer.name for layer in model.layers]

        # Should have pooling for both branches
        pooling_layers = [name for name in layer_names if "pooling" in name]
        assert len(pooling_layers) >= 2

    def test_model_parameter_count_scales_with_units(self):
        """Test model parameter count increases with units."""
        model_small = build_sparse_transformer_bigrcu_model(input_shape=(100, 20), units=32)
        model_large = build_sparse_transformer_bigrcu_model(input_shape=(100, 20), units=128)

        params_small = model_small.count_params()
        params_large = model_large.count_params()

        assert params_large > params_small

    def test_model_parameter_count_scales_with_transformer_layers(self):
        """Test adding more Sparse Transformer layers increases parameters."""
        model_shallow = build_sparse_transformer_bigrcu_model(
            input_shape=(100, 20), num_transformer_layers=1
        )
        model_deep = build_sparse_transformer_bigrcu_model(
            input_shape=(100, 20), num_transformer_layers=4
        )

        params_shallow = model_shallow.count_params()
        params_deep = model_deep.count_params()

        assert params_deep > params_shallow
