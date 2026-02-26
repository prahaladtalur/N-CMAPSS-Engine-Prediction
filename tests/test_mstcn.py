"""
Unit tests for MSTCN (Multi-Scale Temporal Convolutional Network) model.

Tests model building, Global Fusion Attention layer, multi-scale architecture.
"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from src.models.mstcn import build_mstcn_model, GlobalFusionAttention, asymmetric_mse
from src.models.architectures import ModelRegistry


class TestMSTCNModel:
    """Test MSTCN model building and compilation."""

    def test_mstcn_builds_successfully(self):
        """Test that MSTCN model builds without errors."""
        model = build_mstcn_model(input_shape=(1000, 32))
        assert model is not None
        assert isinstance(model, keras.Model)

    def test_mstcn_input_output_shapes(self):
        """Test MSTCN model has correct input and output shapes."""
        seq_length, num_features = 1000, 32
        model = build_mstcn_model(input_shape=(seq_length, num_features))

        assert model.input_shape == (None, seq_length, num_features)
        assert model.output_shape == (None, 1)

    def test_mstcn_with_custom_parameters(self):
        """Test MSTCN model builds with custom parameters."""
        model = build_mstcn_model(
            input_shape=(500, 20),
            units=128,
            dense_units=64,
            dropout_rate=0.3,
            learning_rate=0.0005,
            kernel_size=5,
            dilation_rates=[1, 3, 5, 7],
        )
        assert model is not None
        assert model.input_shape == (None, 500, 20)

    def test_mstcn_default_dilation_rates(self):
        """Test MSTCN uses default dilation rates [1, 2, 4, 8]."""
        model = build_mstcn_model(input_shape=(100, 20))
        # Should build successfully with defaults
        assert model is not None

    def test_mstcn_custom_dilation_rates(self):
        """Test MSTCN with different numbers of scales."""
        # 3 scales
        model3 = build_mstcn_model(input_shape=(100, 20), dilation_rates=[1, 2, 4])
        assert model3 is not None

        # 5 scales
        model5 = build_mstcn_model(input_shape=(100, 20), dilation_rates=[1, 2, 4, 8, 16])
        assert model5 is not None

    def test_mstcn_model_is_compiled(self):
        """Test that MSTCN model is compiled with optimizer and loss."""
        model = build_mstcn_model(input_shape=(1000, 32))
        assert model.optimizer is not None
        assert model.loss is not None
        assert "mae" in [m.name for m in model.metrics]

    def test_mstcn_registry_integration(self):
        """Test that MSTCN is properly registered in ModelRegistry."""
        assert "mstcn" in ModelRegistry.list_models()
        model = ModelRegistry.build("mstcn", input_shape=(1000, 32))
        assert model is not None
        assert model.name == "mstcn"


class TestGlobalFusionAttention:
    """Test Global Fusion Attention layer."""

    def test_gfa_layer_builds(self):
        """Test GFA layer builds successfully."""
        layer = GlobalFusionAttention(num_scales=4, reduction_ratio=8)
        assert layer is not None

    def test_gfa_fuses_multiple_scales(self):
        """Test GFA fuses features from multiple scales."""
        batch_size, seq_length, features = 8, 100, 64
        num_scales = 4
        layer = GlobalFusionAttention(num_scales=num_scales, reduction_ratio=8)

        # Create multi-scale features (4 scales, all same shape)
        multi_scale_features = [
            tf.random.normal((batch_size, seq_length, features)) for _ in range(num_scales)
        ]

        outputs = layer(multi_scale_features)

        # Output should have concatenated features from all scales
        expected_features = features * num_scales
        assert outputs.shape == (batch_size, seq_length, expected_features)

    def test_gfa_with_different_num_scales(self):
        """Test GFA with different numbers of scales."""
        batch_size, seq_length, features = 4, 50, 32

        for num_scales in [2, 3, 4, 5]:
            layer = GlobalFusionAttention(num_scales=num_scales, reduction_ratio=8)

            multi_scale_features = [
                tf.random.normal((batch_size, seq_length, features)) for _ in range(num_scales)
            ]

            outputs = layer(multi_scale_features)

            expected_features = features * num_scales
            assert outputs.shape == (batch_size, seq_length, expected_features)

    def test_gfa_scale_weights_are_learnable(self):
        """Test that GFA has learnable scale weights."""
        layer = GlobalFusionAttention(num_scales=4, reduction_ratio=8)

        # Build layer
        input_shapes = [(None, 100, 64) for _ in range(4)]
        layer.build(input_shapes)

        # Check scale weights exist and are trainable
        assert layer.scale_weights is not None
        assert layer.scale_weights.trainable

    def test_gfa_has_fusion_gate(self):
        """Test that GFA has adaptive fusion gate."""
        layer = GlobalFusionAttention(num_scales=4, reduction_ratio=8)

        input_shapes = [(None, 100, 64) for _ in range(4)]
        layer.build(input_shapes)

        # Check fusion gate exists
        assert layer.fusion_gate is not None

    def test_gfa_applies_channel_attention(self):
        """Test that GFA has channel attention for each scale."""
        num_scales = 4
        layer = GlobalFusionAttention(num_scales=num_scales, reduction_ratio=8)

        # Check that we have channel attention layers
        assert len(layer.channel_attentions) == num_scales

    def test_gfa_applies_temporal_attention(self):
        """Test that GFA has temporal attention for each scale."""
        num_scales = 4
        layer = GlobalFusionAttention(num_scales=num_scales, reduction_ratio=8)

        # Check that we have temporal attention layers
        assert len(layer.temporal_attentions) == num_scales

    def test_gfa_get_config(self):
        """Test GFA layer serialization config."""
        layer = GlobalFusionAttention(num_scales=5, reduction_ratio=16)
        config = layer.get_config()

        assert "num_scales" in config
        assert "reduction_ratio" in config
        assert config["num_scales"] == 5
        assert config["reduction_ratio"] == 16


class TestMSTCNPredictions:
    """Test MSTCN model end-to-end predictions."""

    def test_mstcn_makes_predictions(self):
        """Test MSTCN model can make predictions."""
        model = build_mstcn_model(input_shape=(1000, 32))

        X_test = np.random.randn(10, 1000, 32).astype(np.float32)
        predictions = model.predict(X_test, verbose=0)

        assert predictions.shape == (10, 1)
        assert np.all(np.isfinite(predictions))

    def test_mstcn_trains_one_step(self):
        """Test MSTCN model can perform one training step."""
        model = build_mstcn_model(input_shape=(100, 20))

        X_train = np.random.randn(8, 100, 20).astype(np.float32)
        y_train = np.random.rand(8, 1).astype(np.float32) * 100

        history = model.fit(X_train, y_train, epochs=1, verbose=0)

        assert "loss" in history.history
        assert np.isfinite(history.history["loss"][0])

    def test_mstcn_batch_prediction_consistency(self):
        """Test MSTCN predictions are consistent across batch sizes."""
        model = build_mstcn_model(input_shape=(100, 20))

        # Create identical samples
        sample = np.random.randn(1, 100, 20).astype(np.float32)
        X_repeated = np.repeat(sample, 5, axis=0)

        predictions = model.predict(X_repeated, verbose=0)

        # All predictions should be very similar (same input)
        assert np.std(predictions) < 0.1 or np.allclose(predictions, predictions[0], rtol=1e-3)


class TestMSTCNArchitecture:
    """Test MSTCN architecture components."""

    def test_mstcn_has_multi_scale_tcn(self):
        """Test MSTCN model contains multiple TCN branches."""
        model = build_mstcn_model(input_shape=(100, 20), dilation_rates=[1, 2, 4, 8])

        # Check for TCN blocks with different scales
        tcn_layers = [layer for layer in model.layers if "tcn_scale" in layer.name]

        # Should have num_scales * 2 TCN blocks (2 blocks per scale)
        assert len(tcn_layers) >= 8  # 4 scales * 2 blocks

    def test_mstcn_has_global_fusion_attention(self):
        """Test MSTCN model contains Global Fusion Attention layer."""
        model = build_mstcn_model(input_shape=(100, 20))

        # Check for GlobalFusionAttention layer
        layer_types = [type(layer).__name__ for layer in model.layers]
        layer_names = [layer.name for layer in model.layers]

        assert (
            "GlobalFusionAttention" in layer_types or "global_fusion" in layer_names
        ), "MSTCN should have Global Fusion Attention layer"

    def test_mstcn_has_global_pooling(self):
        """Test MSTCN model has global pooling layer."""
        model = build_mstcn_model(input_shape=(100, 20))

        layer_names = [layer.name for layer in model.layers]
        assert any("pooling" in name for name in layer_names)

    def test_mstcn_parameter_count_scales_with_units(self):
        """Test MSTCN parameter count increases with units."""
        model_small = build_mstcn_model(input_shape=(100, 20), units=32)
        model_large = build_mstcn_model(input_shape=(100, 20), units=128)

        params_small = model_small.count_params()
        params_large = model_large.count_params()

        assert params_large > params_small

    def test_mstcn_parameter_count_scales_with_scales(self):
        """Test adding more scales increases parameters."""
        model_few_scales = build_mstcn_model(input_shape=(100, 20), dilation_rates=[1, 2])
        model_many_scales = build_mstcn_model(input_shape=(100, 20), dilation_rates=[1, 2, 4, 8, 16])

        params_few = model_few_scales.count_params()
        params_many = model_many_scales.count_params()

        assert params_many > params_few


class TestMSTCNVsMDFA:
    """Compare MSTCN with MDFA (both are multi-scale)."""

    def test_mstcn_and_mdfa_both_registered(self):
        """Test both MSTCN and MDFA are available."""
        models = ModelRegistry.list_models()
        assert "mstcn" in models
        assert "mdfa" in models

    def test_mstcn_uses_global_fusion_attention(self):
        """Test MSTCN uses GFA (not simple concatenation like MDFA)."""
        model = build_mstcn_model(input_shape=(100, 20))

        # MSTCN should have GlobalFusionAttention
        has_gfa = any(
            "GlobalFusionAttention" in type(layer).__name__ or "global_fusion" in layer.name
            for layer in model.layers
        )
        assert has_gfa, "MSTCN should use Global Fusion Attention"
