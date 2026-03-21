import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from predict import RULPredictor


class FakeModel:
    def __init__(self, prediction: float):
        self.prediction = prediction
        self.seen_shapes = []

    def predict(self, X, verbose=0):
        self.seen_shapes.append(X.shape)
        return np.array([[self.prediction]], dtype=np.float32)


def _write_artifacts(
    model_dir: Path,
    *,
    max_sequence_length: int,
    target_scaling: str = "none",
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    clip_value: Optional[float] = None,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.keras").write_text("stub", encoding="utf-8")

    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"max_sequence_length": max_sequence_length}, f)

    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(None, f)

    with open(model_dir / "rul_scaler.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "y_min": scale_min,
                "y_max": scale_max,
                "clip_value": clip_value,
                "target_scaling": target_scaling,
                "scale_min": scale_min,
                "scale_max": scale_max,
            },
            f,
        )


def test_predictor_inverse_transforms_minmax_predictions(monkeypatch, tmp_path):
    model = FakeModel(0.75)
    model_path = tmp_path / "single" / "model.keras"
    _write_artifacts(
        model_path.parent,
        max_sequence_length=4,
        target_scaling="minmax",
        scale_min=25.0,
        scale_max=125.0,
        clip_value=125.0,
    )

    monkeypatch.setattr("predict.tf.keras.models.load_model", lambda *args, **kwargs: model)

    predictor = RULPredictor(model_path=str(model_path))
    result = predictor.predict_single(np.ones((6, 3), dtype=np.float32))

    assert result["prediction"] == pytest.approx(100.0)
    assert result["individual_predictions"]["model"] == pytest.approx(100.0)
    assert model.seen_shapes == [(1, 4, 3)]


def test_ensemble_predictor_uses_per_model_artifacts(monkeypatch, tmp_path):
    model_a = FakeModel(0.5)
    model_b = FakeModel(0.5)

    model_a_path = tmp_path / "a" / "model.keras"
    model_b_path = tmp_path / "b" / "model.keras"
    _write_artifacts(
        model_a_path.parent,
        max_sequence_length=4,
        target_scaling="minmax",
        scale_min=0.0,
        scale_max=100.0,
    )
    _write_artifacts(
        model_b_path.parent,
        max_sequence_length=2,
        target_scaling="minmax",
        scale_min=0.0,
        scale_max=200.0,
    )

    fake_models = {
        str(model_a_path): model_a,
        str(model_b_path): model_b,
    }
    monkeypatch.setattr(
        "predict.tf.keras.models.load_model",
        lambda path, **kwargs: fake_models[str(path)],
    )

    predictor = RULPredictor(ensemble=True, model_paths=[str(model_a_path), str(model_b_path)])
    result = predictor.predict_single(np.ones((8, 3), dtype=np.float32))

    assert result["individual_predictions"]["model"] == pytest.approx(100.0)
    assert result["prediction"] == pytest.approx(75.0)
    assert model_a.seen_shapes == [(1, 4, 3)]
    assert model_b.seen_shapes == [(1, 2, 3)]


def test_evaluate_test_set_clips_ground_truth(monkeypatch, tmp_path):
    model = FakeModel(1.0)
    model_path = tmp_path / "single" / "model.keras"
    _write_artifacts(
        model_path.parent,
        max_sequence_length=5,
        target_scaling="minmax",
        scale_min=0.0,
        scale_max=125.0,
        clip_value=125.0,
    )

    monkeypatch.setattr("predict.tf.keras.models.load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        "predict.get_datasets",
        lambda fd: (None, None, ([np.ones((1, 5, 3), dtype=np.float32)], [np.array([150.0])])),  # type: ignore[arg-type]
    )

    captured = {}

    def fake_compute_all_metrics(y_true, y_pred, y_min=None, y_max=None):
        captured["y_true"] = y_true
        captured["y_pred"] = y_pred
        captured["y_min"] = y_min
        captured["y_max"] = y_max
        return {"rmse": 0.0}

    monkeypatch.setattr("predict.compute_all_metrics", fake_compute_all_metrics)
    monkeypatch.setattr("predict.format_metrics", lambda metrics: "stub")

    predictor = RULPredictor(model_path=str(model_path))
    metrics = predictor.evaluate_test_set(visualize=False)

    assert metrics == {"rmse": 0.0}
    assert np.array_equal(captured["y_true"], np.array([125.0]))
    assert np.array_equal(captured["y_pred"], np.array([125.0]))
    assert captured["y_min"] == 0.0
    assert captured["y_max"] == 125.0
