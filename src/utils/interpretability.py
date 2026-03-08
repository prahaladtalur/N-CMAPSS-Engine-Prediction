"""
Attention visualization and interpretability toolkit for RUL prediction models.

Provides gradient-based saliency maps (universal, works for all models) and
direct attention weight extraction for attention-based architectures.

Usage
-----
Quick report for any trained model::

    from src.utils.interpretability import generate_interpretability_report
    generate_interpretability_report(model, X_test, y_test,
                                     model_name="mstcn",
                                     save_dir="results/attention")

Individual functions::

    from src.utils.interpretability import (
        compute_saliency_map,
        plot_saliency_heatmap,
        plot_sensor_importance,
        plot_temporal_profile,
        extract_attention_weights,
    )
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# N-CMAPSS sensor names (use as defaults when no names are provided)
# ---------------------------------------------------------------------------

NCMAPSS_SENSOR_NAMES: List[str] = [
    "alt",
    "Mach",
    "TRA",
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P20",
    "P30",
    "Nf",
    "Nc",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "htBleed",
    "Nf_dmd",
    "PCNfR_dmd",
    "W31",
    "W32",
]

# ---------------------------------------------------------------------------
# Gradient-based saliency
# ---------------------------------------------------------------------------


def compute_saliency_map(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute gradient-based input saliency for any Keras model.

    Uses the absolute value of d(prediction)/d(input), which measures how
    much each input timestep-sensor pair influences the RUL prediction.

    Args:
        model: Trained Keras model with shape (None, T, F) → (None, 1).
        X: Input array of shape (N, T, F).
        batch_size: Process this many samples at once to avoid OOM.

    Returns:
        Saliency array of shape (N, T, F) with non-negative values.
        Higher = more influential for the prediction.
    """
    saliency_batches = []

    for start in range(0, len(X), batch_size):
        X_batch = tf.constant(X[start : start + batch_size], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_batch)
            preds = model(X_batch, training=False)
            # Squeeze prediction to scalar sum for gradient computation
            loss = tf.reduce_sum(preds)

        grads = tape.gradient(loss, X_batch)  # (B, T, F)
        saliency_batches.append(tf.abs(grads).numpy())

    return np.concatenate(saliency_batches, axis=0)  # (N, T, F)


def aggregate_saliency(
    saliency: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Aggregate a (N, T, F) saliency array into interpretable summaries.

    Returns:
        dict with keys:
          ``heatmap``    – (T, F) mean saliency across all samples
          ``sensor``     – (F,) per-sensor importance (mean over time and samples)
          ``temporal``   – (T,) per-timestep importance (mean over features and samples)
    """
    return {
        "heatmap": saliency.mean(axis=0),  # (T, F)
        "sensor": saliency.mean(axis=(0, 1)),  # (F,)
        "temporal": saliency.mean(axis=(0, 2)),  # (T,)
    }


# ---------------------------------------------------------------------------
# Direct attention weight extraction (for attention-based models)
# ---------------------------------------------------------------------------

# Maps model names to layer class names that produce channel/temporal attention.
# These are the custom layer types defined in src/models/.
_CHANNEL_ATTENTION_CLASSES = (
    "ChannelAttention1D",  # CATA-TCN, ATCN
    "ChannelAttention",  # MDFA
)
_TEMPORAL_ATTENTION_CLASSES = (
    "TemporalAttention1D",  # CATA-TCN
    "SpatialAttention",  # MDFA
)


def _find_layers_by_class(
    model: tf.keras.Model, class_names: Tuple[str, ...]
) -> List[tf.keras.layers.Layer]:
    """Return all layers whose class name is in *class_names*."""
    return [layer for layer in model.layers if type(layer).__name__ in class_names]


def extract_attention_weights(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int = 32,
) -> Dict[str, Optional[np.ndarray]]:
    """Extract channel and temporal attention weights from a trained model.

    Supports models that contain ``ChannelAttention1D`` / ``ChannelAttention``
    or ``TemporalAttention1D`` / ``SpatialAttention`` layers.  For all other
    models, returns *None* for the corresponding key (use saliency instead).

    The extraction works by building a temporary Keras sub-model whose outputs
    include both the final prediction and each attention layer's output, then
    recovering the normalised weight vectors from those outputs.

    Args:
        model: Trained Keras model.
        X: Input array of shape (N, T, F).
        batch_size: Batch size for inference.

    Returns:
        dict with keys:
          ``channel``  – (N, F) channel attention weights, or *None*
          ``temporal`` – (N, T) temporal attention weights, or *None*
    """
    channel_layers = _find_layers_by_class(model, _CHANNEL_ATTENTION_CLASSES)
    temporal_layers = _find_layers_by_class(model, _TEMPORAL_ATTENTION_CLASSES)

    result: Dict[str, Optional[np.ndarray]] = {"channel": None, "temporal": None}

    # ---- channel attention ------------------------------------------------
    if channel_layers:
        layer = channel_layers[0]
        try:
            extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            all_outputs = []
            for start in range(0, len(X), batch_size):
                out = extractor(X[start : start + batch_size], training=False).numpy()
                all_outputs.append(out)
            attended = np.concatenate(all_outputs, axis=0)  # (N, T, F)

            # Recover weights: channel_out = input * w(t), so w ≈ mean_t(out)/mean_t(in)
            # Use a fresh forward pass through the extractor to get the layer's input
            in_extractor = _build_layer_input_extractor(model, layer)
            if in_extractor is not None:
                all_inputs = []
                for start in range(0, len(X), batch_size):
                    inp = in_extractor(X[start : start + batch_size], training=False).numpy()
                    all_inputs.append(inp)
                layer_input = np.concatenate(all_inputs, axis=0)  # (N, T, F)
                eps = 1e-8
                # Average over time to get channel importance: (N, F)
                weights = np.mean(attended, axis=1) / (np.mean(layer_input, axis=1) + eps)
                weights = np.abs(weights)
                # Normalise each sample to sum to 1
                weights = weights / (weights.sum(axis=1, keepdims=True) + eps)
                result["channel"] = weights
            else:
                # Fallback: use mean pooled output as proxy for importance
                result["channel"] = np.mean(np.abs(attended), axis=1)

        except Exception:
            pass  # Extraction failed — caller falls back to saliency

    # ---- temporal / spatial attention ------------------------------------
    if temporal_layers:
        layer = temporal_layers[0]
        try:
            extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            all_outputs = []
            for start in range(0, len(X), batch_size):
                out = extractor(X[start : start + batch_size], training=False).numpy()
                all_outputs.append(out)
            attended = np.concatenate(all_outputs, axis=0)  # (N, T, F)

            in_extractor = _build_layer_input_extractor(model, layer)
            if in_extractor is not None:
                all_inputs = []
                for start in range(0, len(X), batch_size):
                    inp = in_extractor(X[start : start + batch_size], training=False).numpy()
                    all_inputs.append(inp)
                layer_input = np.concatenate(all_inputs, axis=0)  # (N, T, F)
                eps = 1e-8
                # Average over features to get temporal importance: (N, T)
                weights = np.mean(attended, axis=2) / (np.mean(layer_input, axis=2) + eps)
                weights = np.abs(weights)
                weights = weights / (weights.sum(axis=1, keepdims=True) + eps)
                result["temporal"] = weights
            else:
                result["temporal"] = np.mean(np.abs(attended), axis=2)

        except Exception:
            pass

    return result


def _build_layer_input_extractor(
    model: tf.keras.Model,
    target_layer: tf.keras.layers.Layer,
) -> Optional[tf.keras.Model]:
    """Build a sub-model that outputs the tensor fed into *target_layer*.

    Returns *None* if the layer's inbound nodes cannot be determined (e.g.
    the layer is nested inside a custom layer that uses the subclassing API).
    """
    try:
        inbound_nodes = target_layer._inbound_nodes
        if not inbound_nodes:
            return None
        layer_input_tensor = inbound_nodes[0].input_tensors
        if isinstance(layer_input_tensor, list):
            layer_input_tensor = layer_input_tensor[0]
        return tf.keras.Model(inputs=model.inputs, outputs=layer_input_tensor)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_saliency_heatmap(
    heatmap: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    model_name: str = "Model",
    title: Optional[str] = None,
    max_sensors: int = 32,
    max_timesteps: int = 200,
    save_path: Optional[str] = None,
) -> None:
    """Plot a (T, F) saliency heatmap — sensors on Y-axis, time on X-axis.

    Args:
        heatmap: Array of shape (T, F) — averaged saliency across samples.
        sensor_names: Feature labels for the Y-axis (default: F0, F1, …).
        model_name: Used in the title.
        title: Override automatic title.
        max_sensors: Clip to at most this many sensors for readability.
        max_timesteps: Clip to at most this many timesteps for readability.
        save_path: If given, save the figure here (directory is created if needed).
    """
    T, F = heatmap.shape

    # Down-sample for readability if needed
    if T > max_timesteps:
        indices = np.linspace(0, T - 1, max_timesteps, dtype=int)
        heatmap = heatmap[indices, :]
        T = max_timesteps

    if F > max_sensors:
        heatmap = heatmap[:, :max_sensors]
        F = max_sensors
        if sensor_names is not None:
            sensor_names = sensor_names[:max_sensors]

    labels = sensor_names if sensor_names is not None else [f"F{i}" for i in range(F)]
    # Trim or pad labels to match F
    labels = labels[:F]

    fig_h = max(4, min(F * 0.35, 14))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    im = ax.imshow(
        heatmap.T,  # (F, T) — sensors on Y-axis
        aspect="auto",
        origin="lower",
        cmap="hot",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Saliency (|∂RUL/∂input|)", fontsize=11)

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Sensor / Feature", fontsize=12)
    ax.set_yticks(range(F))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title or f"{model_name} — Input Saliency Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_sensor_importance(
    sensor_weights: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    model_name: str = "Model",
    title: Optional[str] = None,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of per-sensor importance.

    Args:
        sensor_weights: Array of shape (F,) — one importance score per feature.
        sensor_names: Feature labels (default: F0, F1, …).
        model_name: Used in the title.
        title: Override automatic title.
        top_n: Show only the top *top_n* most important sensors.
        save_path: Save path for the figure.
    """
    F = len(sensor_weights)
    # Generate generic labels that always cover all F dimensions
    default_labels = [f"F{i}" for i in range(F)]
    if sensor_names is not None:
        # Extend with generic names if sensor_names is shorter than F
        labels = list(sensor_names) + default_labels[len(sensor_names) :]
        labels = labels[:F]
    else:
        labels = default_labels

    # Sort by importance descending, keep min(top_n, F)
    actual_top_n = min(top_n, F)
    order = np.argsort(sensor_weights)[::-1][:actual_top_n]
    weights = sensor_weights[order]
    names = [labels[i] for i in order]

    # Reverse so most important is at the top
    weights = weights[::-1]
    names = names[::-1]

    fig_h = max(4, top_n * 0.4)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    colors = plt.cm.RdYlGn(weights / (weights.max() + 1e-8))
    bars = ax.barh(range(len(weights)), weights, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(
        title or f"{model_name} — Top {top_n} Sensor Importance",
        fontsize=14,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bars with values
    for bar, val in zip(bars, weights):
        ax.text(
            bar.get_width() + weights.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_temporal_profile(
    temporal_weights: np.ndarray,
    rul_values: Optional[np.ndarray] = None,
    model_name: str = "Model",
    title: Optional[str] = None,
    n_bins: int = 40,
    save_path: Optional[str] = None,
) -> None:
    """Line/scatter plot of temporal attention vs time-to-failure.

    If *rul_values* is provided (shape (N,)), the plot shows how attention
    shifts as the engine approaches failure.  Otherwise it shows the raw
    temporal attention profile over timesteps.

    Args:
        temporal_weights: If *rul_values* is given: shape (N,) — per-sample
            scalar attention (e.g. mean over timesteps).  Otherwise: shape (T,)
            — attention weight per timestep.
        rul_values: RUL labels in cycles, shape (N,). Optional.
        model_name: Used in title.
        title: Override automatic title.
        n_bins: Number of RUL bins for the binned-mean curve.
        save_path: Save path for the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    if rul_values is not None and len(temporal_weights) == len(rul_values):
        # Scatter: attention vs RUL
        sc = ax.scatter(
            rul_values,
            temporal_weights,
            alpha=0.3,
            s=8,
            c=temporal_weights,
            cmap="plasma",
            label="Samples",
        )
        fig.colorbar(sc, ax=ax, label="Attention score", fraction=0.03)

        # Binned mean curve
        bins = np.linspace(rul_values.min(), rul_values.max(), n_bins + 1)
        bin_means = []
        bin_centers = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (rul_values >= lo) & (rul_values < hi)
            if mask.sum() > 0:
                bin_means.append(temporal_weights[mask].mean())
                bin_centers.append((lo + hi) / 2)
        ax.plot(bin_centers, bin_means, "w-", linewidth=2, label="Binned mean")
        ax.plot(bin_centers, bin_means, "k--", linewidth=1.5)

        ax.invert_xaxis()  # Failure on the right
        ax.set_xlabel("RUL (cycles, decreasing → failure)", fontsize=12)
        ax.set_ylabel("Attention / Saliency Score", fontsize=12)
        ax.legend(fontsize=10)
    else:
        # Raw temporal profile
        t = np.arange(len(temporal_weights))
        ax.fill_between(t, temporal_weights, alpha=0.35, color="#1f77b4")
        ax.plot(t, temporal_weights, color="#1f77b4", linewidth=1.5)
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Attention / Saliency Score", fontsize=12)

    ax.set_title(
        title or f"{model_name} — Temporal Attention Profile",
        fontsize=14,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_attention_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    sensor_names: Optional[List[str]] = None,
    top_n: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side bar chart comparing sensor importance across multiple models.

    Args:
        results: Mapping of model_name → aggregated saliency dict (from
            ``aggregate_saliency``).  Each dict must have a ``"sensor"`` key.
        sensor_names: Feature labels.
        top_n: Number of top sensors to compare.
        save_path: Save path for the figure.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    if n_models == 0:
        return

    # Get top_n sensors by average importance across models
    all_sensor_weights = np.stack(
        [results[m]["sensor"] for m in model_names if "sensor" in results[m]], axis=0
    )
    mean_importance = all_sensor_weights.mean(axis=0)
    F = all_sensor_weights.shape[1]
    actual_top_n = min(top_n, F)
    top_indices = np.argsort(mean_importance)[::-1][:actual_top_n]

    default_labels = [f"F{i}" for i in range(F)]
    if sensor_names is not None:
        labels = list(sensor_names) + default_labels[len(sensor_names) :]
        labels = labels[:F]
    else:
        labels = default_labels
    top_labels = [labels[i] for i in top_indices]

    x = np.arange(actual_top_n)
    width = 0.8 / max(n_models, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.8), 5))

    for k, (model_name, color) in enumerate(zip(model_names, colors)):
        if "sensor" not in results[model_name]:
            continue
        weights = results[model_name]["sensor"][top_indices]
        weights = weights / (weights.max() + 1e-8)  # normalise to [0,1] per model
        ax.bar(x + k * width, weights, width, label=model_name, color=color, alpha=0.85)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(top_labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Normalised Importance", fontsize=12)
    ax.set_title(
        f"Sensor Importance Comparison — Top {actual_top_n} Features",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Full report pipeline
# ---------------------------------------------------------------------------


def generate_interpretability_report(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    sensor_names: Optional[List[str]] = None,
    save_dir: str = "results/attention",
    max_samples: int = 500,
    batch_size: int = 32,
) -> Dict[str, str]:
    """Generate a complete interpretability report for a trained model.

    Computes gradient-based saliency maps and (where available) direct
    attention weight extraction, then saves publication-quality figures.

    Args:
        model: Trained Keras model with input shape (None, T, F).
        X_test: Test inputs, shape (N, T, F).
        y_test: True RUL labels, shape (N,).
        model_name: Used in titles and filenames.
        sensor_names: Feature names for axis labels.  If *None* and the model
            has the right number of features, ``NCMAPSS_SENSOR_NAMES`` is used.
        save_dir: Directory where figures are saved.
        max_samples: Cap samples to keep computation fast.
        batch_size: Batch size for inference.

    Returns:
        Dict mapping figure name → saved file path.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Auto-detect sensor names from N-CMAPSS if shapes match
    n_features = X_test.shape[-1]
    if sensor_names is None and n_features <= len(NCMAPSS_SENSOR_NAMES):
        sensor_names = NCMAPSS_SENSOR_NAMES[:n_features]

    # Sub-sample for speed
    if len(X_test) > max_samples:
        idx = np.random.default_rng(0).choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx]
        y_sample = y_test[idx]
    else:
        X_sample = X_test
        y_sample = y_test

    saved: Dict[str, str] = {}

    print(f"\n[Interpretability] Generating report for '{model_name}'")
    print(f"  Samples: {len(X_sample)}  |  Shape: {X_sample.shape}  |  Dir: {save_dir}")

    # ------------------------------------------------------------------
    # 1. Gradient-based saliency (universal)
    # ------------------------------------------------------------------
    print("  Computing gradient saliency…", end=" ", flush=True)
    saliency = compute_saliency_map(model, X_sample, batch_size=batch_size)
    aggregated = aggregate_saliency(saliency)
    print("done")

    # 1a. Saliency heatmap
    path = os.path.join(save_dir, f"{model_name}_saliency_heatmap.png")
    plot_saliency_heatmap(
        aggregated["heatmap"],
        sensor_names=sensor_names,
        model_name=model_name,
        save_path=path,
    )
    saved["saliency_heatmap"] = path
    print(f"  Saved: {path}")

    # 1b. Sensor importance (gradient)
    path = os.path.join(save_dir, f"{model_name}_sensor_importance.png")
    plot_sensor_importance(
        aggregated["sensor"],
        sensor_names=sensor_names,
        model_name=model_name,
        save_path=path,
    )
    saved["sensor_importance"] = path
    print(f"  Saved: {path}")

    # 1c. Temporal attention profile (gradient)
    # Per-sample temporal saliency = mean over features → (N, T)
    per_sample_temporal = saliency.mean(axis=2)  # (N, T)
    # Mean attention over time → scalar per sample for scatter vs RUL
    per_sample_mean = per_sample_temporal.mean(axis=1)  # (N,)
    path = os.path.join(save_dir, f"{model_name}_temporal_profile.png")
    plot_temporal_profile(
        per_sample_mean,
        rul_values=y_sample,
        model_name=model_name,
        title=f"{model_name} — Gradient Saliency vs RUL",
        save_path=path,
    )
    saved["temporal_profile"] = path
    print(f"  Saved: {path}")

    # ------------------------------------------------------------------
    # 2. Direct attention weight extraction (attention-based models)
    # ------------------------------------------------------------------
    has_channel = bool(_find_layers_by_class(model, _CHANNEL_ATTENTION_CLASSES))
    has_temporal = bool(_find_layers_by_class(model, _TEMPORAL_ATTENTION_CLASSES))

    if has_channel or has_temporal:
        print("  Extracting attention weights…", end=" ", flush=True)
        attn = extract_attention_weights(model, X_sample, batch_size=batch_size)
        print("done")

        if attn["channel"] is not None:
            # (N, F) → mean over samples → (F,)
            ch_weights = attn["channel"].mean(axis=0)
            path = os.path.join(save_dir, f"{model_name}_channel_attention.png")
            plot_sensor_importance(
                ch_weights,
                sensor_names=sensor_names,
                model_name=model_name,
                title=f"{model_name} — Channel Attention Weights",
                save_path=path,
            )
            saved["channel_attention"] = path
            print(f"  Saved: {path}")

        if attn["temporal"] is not None:
            # (N, T) → mean over time → (N,) scalar per sample
            temp_mean = attn["temporal"].mean(axis=1)
            path = os.path.join(save_dir, f"{model_name}_temporal_attention.png")
            plot_temporal_profile(
                temp_mean,
                rul_values=y_sample,
                model_name=model_name,
                title=f"{model_name} — Temporal Attention vs RUL",
                save_path=path,
            )
            saved["temporal_attention"] = path
            print(f"  Saved: {path}")

    print(f"  Report complete — {len(saved)} figures saved to {save_dir}/")
    return saved


def compare_models_interpretability(
    models: Dict[str, tf.keras.Model],
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    save_dir: str = "results/attention",
    max_samples: int = 500,
    batch_size: int = 32,
) -> str:
    """Generate per-model reports and a cross-model sensor comparison chart.

    Args:
        models: Mapping of model_name → trained Keras model.
        X_test: Shared test inputs, shape (N, T, F).
        y_test: True RUL labels, shape (N,).
        sensor_names: Feature names.
        save_dir: Root directory for all figures.
        max_samples: Cap samples per model.
        batch_size: Batch size for inference.

    Returns:
        Path to the cross-model comparison figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_features = X_test.shape[-1]
    if sensor_names is None and n_features <= len(NCMAPSS_SENSOR_NAMES):
        sensor_names = NCMAPSS_SENSOR_NAMES[:n_features]

    if len(X_test) > max_samples:
        idx = np.random.default_rng(0).choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx]
        y_sample = y_test[idx]
    else:
        X_sample, y_sample = X_test, y_test

    all_aggregated: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, model in models.items():
        model_dir = os.path.join(save_dir, model_name)
        generate_interpretability_report(
            model,
            X_sample,
            y_sample,
            model_name=model_name,
            sensor_names=sensor_names,
            save_dir=model_dir,
            max_samples=max_samples,
            batch_size=batch_size,
        )
        # Collect saliency for comparison
        saliency = compute_saliency_map(model, X_sample, batch_size=batch_size)
        all_aggregated[model_name] = aggregate_saliency(saliency)

    comparison_path = os.path.join(save_dir, "sensor_importance_comparison.png")
    plot_attention_comparison(
        all_aggregated,
        sensor_names=sensor_names,
        save_path=comparison_path,
    )
    print(f"\nCross-model comparison saved to: {comparison_path}")
    return comparison_path
