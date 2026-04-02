"""Shared model-test assertions that stay stable across Keras versions."""


def assert_model_tracks_metric(model, metric_name: str) -> None:
    """Assert that a compiled Keras model was configured with a named metric."""
    compile_metrics = getattr(model, "_compile_metrics", None)
    user_metrics = getattr(compile_metrics, "_user_metrics", None)

    if user_metrics is None:
        configured_metrics = [metric.name for metric in getattr(model, "metrics", [])]
    else:
        configured_metrics = [
            metric if isinstance(metric, str) else getattr(metric, "name", str(metric))
            for metric in user_metrics
        ]

    assert (
        metric_name in configured_metrics
    ), f"{model.name}: expected metric '{metric_name}', got {configured_metrics}"
