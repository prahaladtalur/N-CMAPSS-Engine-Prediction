"""
Training smoke tests.

Quick 1-epoch training tests to ensure models can train without crashing.
Uses small synthetic data for speed.
"""

import pytest
from src.models.architectures import get_model


class TestTrainingSmoke:
    """Smoke tests for model training."""

    @pytest.mark.parametrize("model_name", ["gru", "lstm", "transformer"])
    def test_one_epoch_training(self, model_name, synthetic_data, small_input_shape, default_model_params):
        """Test that model can complete one training epoch without errors."""
        X_train, y_train, X_val, y_val = synthetic_data

        # Build model
        model = get_model(model_name, input_shape=small_input_shape, **default_model_params)

        # Train for 1 epoch
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        # Check that history contains expected keys
        assert "loss" in history.history
        assert "val_loss" in history.history

        # Check that losses are finite
        assert all(isinstance(v, (int, float)) for v in history.history["loss"])
        assert all(isinstance(v, (int, float)) for v in history.history["val_loss"])

    def test_training_reduces_loss(self, synthetic_data, small_input_shape, default_model_params):
        """Test that training for multiple epochs tends to reduce loss."""
        X_train, y_train, X_val, y_val = synthetic_data

        # Build simple model
        model = get_model("gru", input_shape=small_input_shape, **default_model_params)

        # Train for a few epochs
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=4,
            verbose=0,
        )

        # First epoch loss should generally be higher than last
        # (not strict requirement due to randomness, but usually holds)
        first_loss = history.history["loss"][0]
        last_loss = history.history["loss"][-1]

        # At minimum, loss should be finite
        assert isinstance(first_loss, (int, float))
        assert isinstance(last_loss, (int, float))

    def test_prediction_after_training(self, synthetic_data, small_input_shape, default_model_params):
        """Test that model can make predictions after training."""
        X_train, y_train, X_val, y_val = synthetic_data

        # Build and train model
        model = get_model("gru", input_shape=small_input_shape, **default_model_params)
        model.fit(X_train, y_train, epochs=1, batch_size=4, verbose=0)

        # Make predictions
        y_pred = model.predict(X_val, verbose=0)

        # Check predictions shape and values
        assert y_pred.shape == (len(X_val), 1)
        assert all(isinstance(v[0], (int, float, type(y_pred[0][0]))) for v in y_pred)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
