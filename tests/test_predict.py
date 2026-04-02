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
    resolution_seconds: int = 1,
    target_scaling: str = "none",
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    clip_value: Optional[float] = None,
    prediction_aggregation: str = "none",
    aggregation_window: Optional[int] = None,
    aggregation_decay: float = 0.9,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.keras").write_text("stub", encoding="utf-8")

    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "max_sequence_length": max_sequence_length,
                "resolution_seconds": resolution_seconds,
                "prediction_aggregation": prediction_aggregation,
                "aggregation_window": aggregation_window,
                "aggregation_decay": aggregation_decay,
            },
            f,
        )

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
    captured_load = {}

    def fake_get_datasets(fd, feature_select=None, resolution_seconds=1):  # type: ignore[no-untyped-def]
        captured_load["fd"] = fd
        captured_load["feature_select"] = feature_select
        captured_load["resolution_seconds"] = resolution_seconds
        return None, None, ([np.ones((1, 5, 3), dtype=np.float32)], [np.array([150.0])])

    monkeypatch.setattr("predict.get_datasets", fake_get_datasets)

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
    assert captured_load["resolution_seconds"] == 1


def test_predict_unit_applies_causal_ema(monkeypatch, tmp_path):
    model = FakeModel(10.0)
    model_path = tmp_path / "single" / "model.keras"
    _write_artifacts(
        model_path.parent,
        max_sequence_length=5,
        prediction_aggregation="ema",
        aggregation_window=3,
        aggregation_decay=0.5,
    )

    monkeypatch.setattr("predict.tf.keras.models.load_model", lambda *args, **kwargs: model)

    predictor = RULPredictor(model_path=str(model_path))

    predictions = iter([10.0, 20.0, 40.0, 80.0])

    def fake_predict_single(_sequence):
        return {"prediction": next(predictions)}

    monkeypatch.setattr(predictor, "predict_single", fake_predict_single)
    y_true, y_pred, _ = predictor.predict_unit(
        np.ones((4, 5, 3), dtype=np.float32),
        np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
    )

    assert np.array_equal(y_true, np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32))
    assert np.allclose(y_pred, np.array([10.0, 16.666666, 30.0, 60.0], dtype=np.float32))


def test_evaluate_test_set_uses_saved_resolution_seconds(monkeypatch, tmp_path):
    model = FakeModel(5.0)
    model_path = tmp_path / "single" / "model.keras"
    _write_artifacts(model_path.parent, max_sequence_length=5, resolution_seconds=10)

    monkeypatch.setattr("predict.tf.keras.models.load_model", lambda *args, **kwargs: model)

    captured = {}

    def fake_get_datasets(fd, feature_select=None, resolution_seconds=1):  # type: ignore[no-untyped-def]
        captured["resolution_seconds"] = resolution_seconds
        return None, None, ([np.ones((1, 5, 3), dtype=np.float32)], [np.array([5.0])])

    monkeypatch.setattr("predict.get_datasets", fake_get_datasets)
    monkeypatch.setattr("predict.compute_all_metrics", lambda *args, **kwargs: {"rmse": 0.0})
    monkeypatch.setattr("predict.format_metrics", lambda metrics: "stub")

    predictor = RULPredictor(model_path=str(model_path))
    predictor.evaluate_test_set(visualize=False)

    assert captured["resolution_seconds"] == 10
