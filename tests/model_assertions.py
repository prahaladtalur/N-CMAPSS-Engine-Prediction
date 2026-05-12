"""Shared model assertions for architecture smoke tests."""


def assert_model_tracks_metric(model, metric_name: str) -> None:
    """Assert a compiled Keras model tracks a metric by name."""
    metric_names = set(getattr(model, "metrics_names", []) or [])
    for metric in getattr(model, "metrics", []) or []:
        name = getattr(metric, "name", None)
        if name:
            metric_names.add(name)

    compile_config = getattr(model, "get_compile_config", lambda: {})() or {}
    for metric in compile_config.get("metrics") or []:
        if isinstance(metric, str):
            metric_names.add(metric)
        elif isinstance(metric, dict):
            name = metric.get("config", {}).get("name") or metric.get("class_name")
            if name:
                metric_names.add(name)

    assert metric_name in metric_names or any(
        metric_name in name for name in metric_names
    ), f"Expected model to track metric {metric_name!r}; got {sorted(metric_names)}"
