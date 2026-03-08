"""
Smoke tests for all registered model architectures.

Tests that each model:
1. Can be built without errors
2. Produces output of correct shape
3. Forward pass completes successfully
4. Model has trainable parameters
"""

import pytest
import numpy as np
from src.models.architectures import ModelRegistry, get_model


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_list_models(self):
        """Test that list_models returns non-empty list."""
        models = ModelRegistry.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_model_registration(self, all_model_names):
        """Test that all expected models are registered."""
        registered = set(ModelRegistry.list_models())
        expected = set(all_model_names)
        # All expected models should be registered
        assert expected.issubset(registered), f"Missing models: {expected - registered}"

    def test_get_invalid_model(self):
        """Test that getting an invalid model raises ValueError."""
        with pytest.raises(ValueError):
            ModelRegistry.get("nonexistent_model")


class TestModelBuilding:
    """Test that all models can be built successfully."""

    @pytest.mark.parametrize("model_name", [
        "lstm", "bilstm", "gru", "bigru",
        "attention_lstm", "transformer", "resnet_lstm",
        "tcn", "wavenet",
        "cnn_lstm", "cnn_gru", "inception_lstm",
        "mlp",
    ])
    def test_model_builds(self, model_name, small_input_shape, default_model_params):
        """Test that each model builds without errors."""
        model = get_model(
            model_name,
            input_shape=small_input_shape,
            **default_model_params
        )
        assert model is not None
        assert hasattr(model, 'predict')

    @pytest.mark.parametrize("model_name", [
        "lstm", "bilstm", "gru", "bigru",
        "attention_lstm", "transformer", "resnet_lstm",
        "tcn", "wavenet",
        "cnn_lstm", "cnn_gru", "inception_lstm",
        "mlp",
    ])
    def test_model_output_shape(self, model_name, small_input_shape, default_model_params):
        """Test that model produces scalar output (RUL prediction)."""
        model = get_model(
            model_name,
            input_shape=small_input_shape,
            **default_model_params
        )

        # Create single sample
        batch_size = 1
        sample = np.random.randn(batch_size, *small_input_shape).astype(np.float32)

        # Forward pass
        output = model.predict(sample, verbose=0)

        # Should output single value per sample
        assert output.shape == (batch_size, 1)

    @pytest.mark.parametrize("model_name", [
        "lstm", "bilstm", "gru", "bigru",
        "attention_lstm", "transformer", "resnet_lstm",
        "tcn", "wavenet",
        "cnn_lstm", "cnn_gru", "inception_lstm",
        "mlp",
    ])
    def test_model_has_trainable_params(self, model_name, small_input_shape, default_model_params):
        """Test that model has trainable parameters."""
        model = get_model(
            model_name,
            input_shape=small_input_shape,
            **default_model_params
        )

        trainable_params = sum([np.prod(v.shape) for v in model.trainable_variables])
        assert trainable_params > 0, f"{model_name} has no trainable parameters"


class TestModelBatchPrediction:
    """Test model predictions with different batch sizes."""

    @pytest.mark.parametrize("model_name,batch_size", [
        ("gru", 1),
        ("gru", 5),
        ("transformer", 1),
        ("transformer", 10),
    ])
    def test_batch_prediction(self, model_name, batch_size, small_input_shape, default_model_params):
        """Test that models handle different batch sizes correctly."""
        model = get_model(
            model_name,
            input_shape=small_input_shape,
            **default_model_params
        )

        # Create batch
        batch = np.random.randn(batch_size, *small_input_shape).astype(np.float32)

        # Forward pass
        output = model.predict(batch, verbose=0)

        # Should output one value per sample
        assert output.shape == (batch_size, 1)


class TestModelCompilation:
    """Test that models are properly compiled."""

    def test_model_has_optimizer(self, small_input_shape, default_model_params):
        """Test that model has optimizer configured."""
        model = get_model("gru", input_shape=small_input_shape, **default_model_params)
        assert model.optimizer is not None

    def test_model_has_loss(self, small_input_shape, default_model_params):
        """Test that model has loss function configured."""
        model = get_model("gru", input_shape=small_input_shape, **default_model_params)
        # Model should be compiled with loss
        assert model.loss is not None

    def test_model_has_metrics(self, small_input_shape, default_model_params):
        """Test that model has metrics configured."""
        model = get_model("gru", input_shape=small_input_shape, **default_model_params)
        # Should have metrics defined (at minimum: loss)
        assert len(model.metrics) > 0
        # Model should have compiled_metrics attribute
        assert hasattr(model, 'compiled_metrics')
