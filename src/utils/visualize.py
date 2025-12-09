"""
Visualization utilities for N-CMAPSS dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from scipy import stats


def plot_rul_distribution(
    labels: List[np.ndarray], split_name: str = "Dataset", max_bins: int = 50
) -> None:
    """
    Plot RUL distribution with histogram and box plot.

    Args:
        labels: List of numpy arrays containing RUL values
        split_name: Name of the dataset split
        max_bins: Maximum number of histogram bins
    """
    # Filter out empty arrays and check if we have any data
    non_empty_labels = [label for label in labels if len(label) > 0]
    if not non_empty_labels:
        print(f"Error: No valid labels found for {split_name}")
        return

    all_labels = np.concatenate(non_empty_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(
        all_labels,
        bins=min(max_bins, len(np.unique(all_labels))),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    axes[0].set_xlabel("RUL (cycles)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(f"{split_name} RUL Distribution", fontsize=14, fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot(
        all_labels,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1].set_ylabel("RUL (cycles)", fontsize=12)
    axes[1].set_title(f"{split_name} RUL Statistics", fontsize=14, fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")

    # Statistics
    stats_text = (
        f"Mean: {all_labels.mean():.2f}\n"
        f"Median: {np.median(all_labels):.2f}\n"
        f"Std: {all_labels.std():.2f}"
    )
    axes[1].text(
        1.15,
        np.median(all_labels),
        stats_text,
        fontsize=10,
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\n{split_name} RUL Statistics:")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Mean RUL: {all_labels.mean():.2f} cycles")
    print(f"  Median RUL: {np.median(all_labels):.2f} cycles")
    print(f"  Std Dev: {all_labels.std():.2f} cycles")
    print(f"  Min RUL: {all_labels.min():.2f} cycles")
    print(f"  Max RUL: {all_labels.max():.2f} cycles")


def plot_sensor_time_series(
    features: List[np.ndarray],
    labels: List[np.ndarray],
    unit_idx: int = 0,
    sensor_indices: Optional[List[int]] = None,
    num_sensors: int = 4,
    max_timesteps: int = 500,
) -> None:
    """
    Plot sensor time series for a specific unit.

    Args:
        features: List of arrays (num_cycles, timesteps, num_sensors)
        labels: List of arrays with RUL values
        unit_idx: Index of unit to visualize
        sensor_indices: Specific sensors to plot
        num_sensors: Number of sensors if sensor_indices is None
        max_timesteps: Maximum timesteps to display
    """
    if unit_idx >= len(features):
        print(f"Error: unit_idx {unit_idx} out of range. Only {len(features)} units.")
        return

    unit_data = features[unit_idx]
    unit_labels = labels[unit_idx]

    if sensor_indices is None:
        sensor_indices = list(range(min(num_sensors, unit_data.shape[2])))

    num_sensors = len(sensor_indices)
    num_cycles = unit_data.shape[0]

    # Select representative cycles
    cycle_indices = (
        [0, num_cycles // 2, num_cycles - 1] if num_cycles >= 3 else list(range(num_cycles))
    )

    fig, axes = plt.subplots(num_sensors, 1, figsize=(14, 3 * num_sensors))
    if num_sensors == 1:
        axes = [axes]

    for i, sensor_idx in enumerate(sensor_indices):
        for cycle_idx in cycle_indices:
            cycle_data = unit_data[cycle_idx, :max_timesteps, sensor_idx]
            rul_value = unit_labels[cycle_idx]
            axes[i].plot(
                cycle_data,
                label=f"Cycle {cycle_idx} (RUL={rul_value:.0f})",
                alpha=0.7,
                linewidth=1.5,
            )

        axes[i].set_xlabel("Time Step", fontsize=11)
        axes[i].set_ylabel(f"Sensor {sensor_idx} Value", fontsize=11)
        axes[i].set_title(
            f"Unit {unit_idx} - Sensor {sensor_idx} Time Series", fontsize=12, fontweight="bold"
        )
        axes[i].legend(loc="best", fontsize=9)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nUnit {unit_idx} Information:")
    print(f"  Number of cycles: {num_cycles}")
    print(f"  Timesteps per cycle: {unit_data.shape[1]}")
    print(f"  Number of sensors: {unit_data.shape[2]}")
    print(f"  RUL range: {unit_labels.min():.0f} to {unit_labels.max():.0f} cycles")


def plot_sensor_degradation(
    features: List[np.ndarray],
    labels: List[np.ndarray],
    unit_idx: int = 0,
    sensor_indices: Optional[List[int]] = None,
    num_sensors: int = 14,
) -> None:
    """
    Visualize how sensor values change as the engine degrades (RUL decreases).

    This shows sensor trends over the engine lifecycle, helping identify which
    sensors show clear degradation patterns.

    Args:
        features: List of arrays (num_cycles, timesteps, num_sensors)
        labels: List of arrays with RUL values
        unit_idx: Index of unit to visualize
        sensor_indices: Specific sensors to plot (default: first 6)
        num_sensors: Total number of sensors
    """
    if unit_idx >= len(features):
        print(f"Error: unit_idx {unit_idx} out of range. Only {len(features)} units.")
        return

    unit_data = features[unit_idx]
    unit_labels = labels[unit_idx]

    if sensor_indices is None:
        sensor_indices = list(range(min(6, unit_data.shape[2])))

    # Calculate mean sensor value for each cycle
    sensor_means = np.mean(unit_data, axis=1)  # (num_cycles, num_sensors)

    # Sort by RUL (descending) to show degradation over time
    sort_idx = np.argsort(unit_labels)[::-1]
    rul_sorted = unit_labels[sort_idx]
    sensor_sorted = sensor_means[sort_idx]

    n_sensors = len(sensor_indices)
    n_cols = 3
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_sensors > 1 else [axes]

    for i, sensor_idx in enumerate(sensor_indices):
        ax = axes[i]
        sensor_values = sensor_sorted[:, sensor_idx]

        # Plot sensor value vs RUL
        scatter = ax.scatter(rul_sorted, sensor_values, c=np.arange(len(rul_sorted)),
                           cmap='viridis', alpha=0.6, s=20)

        # Fit trend line
        z = np.polyfit(rul_sorted, sensor_values, 2)
        p = np.poly1d(z)
        ax.plot(rul_sorted, p(rul_sorted), "r--", linewidth=2, alpha=0.8, label='Trend')

        ax.set_xlabel("RUL (cycles)", fontsize=11)
        ax.set_ylabel(f"Sensor {sensor_idx} Mean Value", fontsize=11)
        ax.set_title(f"Sensor {sensor_idx} Degradation Pattern", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time Progression', fontsize=9)

    # Hide unused subplots
    for i in range(n_sensors, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"Unit {unit_idx} - Sensor Degradation Analysis",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.show()

    print(f"\nSensor Degradation Analysis for Unit {unit_idx}:")
    print(f"  Number of cycles analyzed: {len(rul_sorted)}")
    print(f"  RUL range: {rul_sorted.min():.0f} to {rul_sorted.max():.0f} cycles")


def plot_sensor_correlation_heatmap(
    features: List[np.ndarray],
    labels: List[np.ndarray],
    max_sensors: int = 14,
    sample_size: int = 1000,
) -> None:
    """
    Plot correlation heatmap between sensors and RUL.

    This helps identify which sensors are most predictive of engine health.

    Args:
        features: List of arrays (num_cycles, timesteps, num_sensors)
        labels: List of arrays with RUL values
        max_sensors: Maximum number of sensors to include
        sample_size: Number of samples to use for correlation analysis
    """
    # Aggregate data from all units
    all_sensor_means = []
    all_rul = []

    for unit_features, unit_labels in zip(features, labels):
        sensor_means = np.mean(unit_features, axis=1)  # Average over timesteps
        all_sensor_means.append(sensor_means)
        all_rul.extend(unit_labels)

    all_sensor_means = np.vstack(all_sensor_means)
    all_rul = np.array(all_rul)

    # Sample if too large
    if len(all_rul) > sample_size:
        indices = np.random.choice(len(all_rul), sample_size, replace=False)
        all_sensor_means = all_sensor_means[indices]
        all_rul = all_rul[indices]

    # Limit sensors
    n_sensors = min(max_sensors, all_sensor_means.shape[1])
    all_sensor_means = all_sensor_means[:, :n_sensors]

    # Create data matrix with RUL
    data = np.column_stack([all_sensor_means, all_rul])

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data.T)

    # Create labels
    sensor_labels = [f"S{i}" for i in range(n_sensors)]
    labels_all = sensor_labels + ["RUL"]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=labels_all, yticklabels=labels_all,
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax,
                square=True, linewidths=0.5)

    ax.set_title("Sensor-RUL Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()

    # Print top RUL correlations
    rul_correlations = corr_matrix[-1, :-1]
    top_indices = np.argsort(np.abs(rul_correlations))[::-1][:5]

    print("\nTop 5 Sensors Correlated with RUL:")
    for idx in top_indices:
        print(f"  Sensor {idx}: {rul_correlations[idx]:.3f}")


def plot_rul_trajectory(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_length: Optional[List[int]] = None,
    unit_idx: int = 0,
) -> None:
    """
    Plot the RUL trajectory for a specific engine unit over its lifecycle.

    Shows how predicted RUL compares to actual RUL as the engine progresses
    through its operational life.

    Args:
        y_true: True RUL values (flattened or per-unit)
        y_pred: Predicted RUL values
        unit_length: List of cycle counts per unit (if flattened data)
        unit_idx: Which unit to visualize
    """
    if unit_length is not None:
        # Extract specific unit from flattened data
        start_idx = sum(unit_length[:unit_idx])
        end_idx = start_idx + unit_length[unit_idx]
        y_true_unit = y_true[start_idx:end_idx]
        y_pred_unit = y_pred[start_idx:end_idx]
    else:
        y_true_unit = y_true
        y_pred_unit = y_pred

    cycles = np.arange(len(y_true_unit))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top plot: RUL over time
    axes[0].plot(cycles, y_true_unit, 'b-', linewidth=2.5, label='Actual RUL', alpha=0.8)
    axes[0].plot(cycles, y_pred_unit, 'r--', linewidth=2, label='Predicted RUL', alpha=0.8)
    axes[0].fill_between(cycles, y_true_unit, y_pred_unit, alpha=0.2, color='gray', label='Error')

    # Add critical zones
    axes[0].axhspan(0, 30, alpha=0.1, color='red', label='Critical Zone (<30 cycles)')
    axes[0].axhspan(30, 75, alpha=0.1, color='orange', label='Warning Zone (30-75 cycles)')

    axes[0].set_xlabel("Cycle Number", fontsize=12)
    axes[0].set_ylabel("RUL (cycles)", fontsize=12)
    axes[0].set_title(f"Unit {unit_idx} - RUL Trajectory Over Lifecycle",
                     fontsize=14, fontweight="bold")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(alpha=0.3)

    # Bottom plot: Prediction error over time
    error = y_pred_unit - y_true_unit
    axes[1].plot(cycles, error, 'purple', linewidth=2, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    axes[1].fill_between(cycles, 0, error, where=(error >= 0), alpha=0.3, color='red',
                         label='Over-prediction', interpolate=True)
    axes[1].fill_between(cycles, 0, error, where=(error < 0), alpha=0.3, color='blue',
                         label='Under-prediction', interpolate=True)

    axes[1].set_xlabel("Cycle Number", fontsize=12)
    axes[1].set_ylabel("Prediction Error (cycles)", fontsize=12)
    axes[1].set_title(f"Unit {unit_idx} - Prediction Error Over Time",
                     fontsize=14, fontweight="bold")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistics
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    print(f"\nUnit {unit_idx} Trajectory Statistics:")
    print(f"  Total cycles: {len(cycles)}")
    print(f"  Mean Absolute Error: {mae:.2f} cycles")
    print(f"  RMSE: {rmse:.2f} cycles")
    print(f"  Max over-prediction: {error.max():.2f} cycles")
    print(f"  Max under-prediction: {error.min():.2f} cycles")


def plot_critical_zone_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    critical_threshold: int = 30,
    warning_threshold: int = 75,
) -> None:
    """
    Analyze model performance in critical RUL zones.

    Shows how well the model performs when engines are close to failure,
    which is the most important regime for maintenance decisions.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        critical_threshold: RUL threshold for critical zone
        warning_threshold: RUL threshold for warning zone
    """
    # Define zones
    critical_mask = y_true < critical_threshold
    warning_mask = (y_true >= critical_threshold) & (y_true < warning_threshold)
    safe_mask = y_true >= warning_threshold

    zones = {
        'Critical (RUL < 30)': critical_mask,
        'Warning (30 ≤ RUL < 75)': warning_mask,
        'Safe (RUL ≥ 75)': safe_mask
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Error distribution by zone
    ax = axes[0, 0]
    zone_errors = []
    zone_labels = []
    colors = ['red', 'orange', 'green']

    for (zone_name, mask), color in zip(zones.items(), colors):
        if mask.sum() > 0:
            errors = y_pred[mask] - y_true[mask]
            zone_errors.append(errors)
            zone_labels.append(zone_name)

    bp = ax.boxplot(zone_errors, labels=zone_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp['boxes'], colors[:len(zone_errors)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Prediction Error (cycles)", fontsize=11)
    ax.set_title("Error Distribution by RUL Zone", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis='y')

    # 2. Accuracy in each zone
    ax = axes[0, 1]
    zone_names = []
    accuracies_10 = []
    accuracies_20 = []
    accuracies_30 = []

    for zone_name, mask in zones.items():
        if mask.sum() > 0:
            errors = np.abs(y_pred[mask] - y_true[mask])
            zone_names.append(zone_name.split('(')[0].strip())
            accuracies_10.append(100 * np.mean(errors <= 10))
            accuracies_20.append(100 * np.mean(errors <= 20))
            accuracies_30.append(100 * np.mean(errors <= 30))

    x = np.arange(len(zone_names))
    width = 0.25

    ax.bar(x - width, accuracies_10, width, label='±10 cycles', color='darkgreen', alpha=0.8)
    ax.bar(x, accuracies_20, width, label='±20 cycles', color='lightgreen', alpha=0.8)
    ax.bar(x + width, accuracies_30, width, label='±30 cycles', color='palegreen', alpha=0.8)

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Prediction Accuracy by Zone", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(zone_names)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # 3. Sample distribution across zones
    ax = axes[1, 0]
    zone_counts = [mask.sum() for mask in zones.values()]
    zone_names_full = list(zones.keys())
    colors_full = ['red', 'orange', 'green']

    wedges, texts, autotexts = ax.pie(zone_counts, labels=zone_names_full, autopct='%1.1f%%',
                                       colors=colors_full, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax.set_title("Sample Distribution Across Zones", fontsize=12, fontweight="bold")

    # 4. Scatter plot colored by zone
    ax = axes[1, 1]
    for (zone_name, mask), color in zip(zones.items(), colors):
        if mask.sum() > 0:
            ax.scatter(y_true[mask], y_pred[mask], alpha=0.5, s=20,
                      c=color, label=zone_name)

    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')

    ax.set_xlabel("True RUL", fontsize=11)
    ax.set_ylabel("Predicted RUL", fontsize=11)
    ax.set_title("Predictions by Zone", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nCritical Zone Analysis:")
    for zone_name, mask in zones.items():
        if mask.sum() > 0:
            errors = np.abs(y_pred[mask] - y_true[mask])
            print(f"\n{zone_name}:")
            print(f"  Samples: {mask.sum()}")
            print(f"  MAE: {errors.mean():.2f} cycles")
            print(f"  RMSE: {np.sqrt(np.mean((y_pred[mask] - y_true[mask])**2)):.2f} cycles")
            print(f"  Accuracy (±10 cycles): {100 * np.mean(errors <= 10):.1f}%")
            print(f"  Accuracy (±20 cycles): {100 * np.mean(errors <= 20):.1f}%")


def plot_multi_sensor_lifecycle(
    features: List[np.ndarray],
    labels: List[np.ndarray],
    unit_idx: int = 0,
    max_sensors: int = 8,
) -> None:
    """
    Compare multiple sensors side-by-side over the engine lifecycle.

    Shows normalized sensor values over the full operational life to identify
    which sensors show the clearest degradation patterns.

    Args:
        features: List of arrays (num_cycles, timesteps, num_sensors)
        labels: List of arrays with RUL values
        unit_idx: Index of unit to visualize
        max_sensors: Maximum number of sensors to display
    """
    if unit_idx >= len(features):
        print(f"Error: unit_idx {unit_idx} out of range. Only {len(features)} units.")
        return

    unit_data = features[unit_idx]
    unit_labels = labels[unit_idx]

    # Calculate mean sensor values per cycle
    sensor_means = np.mean(unit_data, axis=1)  # (num_cycles, num_sensors)

    # Sort by RUL descending
    sort_idx = np.argsort(unit_labels)[::-1]
    rul_sorted = unit_labels[sort_idx]
    sensor_sorted = sensor_means[sort_idx]

    # Normalize sensors to [0, 1] for comparison
    n_sensors = min(max_sensors, sensor_sorted.shape[1])
    sensor_normalized = np.zeros((sensor_sorted.shape[0], n_sensors))

    for i in range(n_sensors):
        min_val = sensor_sorted[:, i].min()
        max_val = sensor_sorted[:, i].max()
        if max_val > min_val:
            sensor_normalized[:, i] = (sensor_sorted[:, i] - min_val) / (max_val - min_val)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))

    cycles = np.arange(len(rul_sorted))
    colors = plt.cm.tab10(np.linspace(0, 1, n_sensors))

    for i in range(n_sensors):
        ax.plot(cycles, sensor_normalized[:, i], linewidth=2, alpha=0.7,
               label=f'Sensor {i}', color=colors[i])

    # Add RUL overlay (normalized)
    rul_normalized = (rul_sorted - rul_sorted.min()) / (rul_sorted.max() - rul_sorted.min())
    ax.plot(cycles, rul_normalized, 'k--', linewidth=3, alpha=0.5, label='RUL (normalized)')

    ax.set_xlabel("Cycle (sorted by descending RUL)", fontsize=12)
    ax.set_ylabel("Normalized Sensor Value", fontsize=12)
    ax.set_title(f"Unit {unit_idx} - Multi-Sensor Lifecycle Comparison (Normalized)",
                fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, ncol=3)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nMulti-Sensor Lifecycle Analysis for Unit {unit_idx}:")
    print(f"  Cycles displayed: {len(cycles)}")
    print(f"  Sensors compared: {n_sensors}")
    print(f"  All sensors normalized to [0, 1] for comparison")


def plot_prediction_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_predictions: Optional[List[np.ndarray]] = None,
    confidence_percentile: float = 90,
) -> None:
    """
    Visualize prediction confidence and uncertainty.

    If multiple predictions are available (e.g., from ensemble or dropout),
    shows prediction intervals. Otherwise, shows error-based confidence.

    Args:
        y_true: True RUL values
        y_pred: Mean predicted RUL values
        model_predictions: List of prediction arrays from ensemble/dropout (optional)
        confidence_percentile: Percentile for confidence interval
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Prediction with uncertainty
    ax = axes[0, 0]
    errors = np.abs(y_pred - y_true)

    # Sample for visualization
    n_samples = min(500, len(y_true))
    indices = np.random.choice(len(y_true), n_samples, replace=False)
    indices = np.sort(indices)

    if model_predictions is not None and len(model_predictions) > 1:
        # Calculate prediction intervals from ensemble
        all_preds = np.array(model_predictions)
        pred_mean = all_preds.mean(axis=0)
        pred_std = all_preds.std(axis=0)

        lower = pred_mean - 2 * pred_std
        upper = pred_mean + 2 * pred_std

        ax.fill_between(indices, lower[indices], upper[indices], alpha=0.3, color='lightblue',
                       label='95% Confidence Interval')
        ax.plot(indices, pred_mean[indices], 'b-', linewidth=2, label='Mean Prediction')
    else:
        # Use error-based confidence
        window_size = max(10, n_samples // 50)
        rolling_std = np.array([errors[max(0, i-window_size):i+window_size].std()
                               for i in range(len(errors))])

        lower = y_pred - 2 * rolling_std
        upper = y_pred + 2 * rolling_std

        ax.fill_between(indices, lower[indices], upper[indices], alpha=0.3, color='lightblue',
                       label='Estimated Confidence Interval')
        ax.plot(indices, y_pred[indices], 'b-', linewidth=2, label='Prediction')

    ax.plot(indices, y_true[indices], 'r-', linewidth=2, alpha=0.7, label='Actual RUL')
    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("RUL (cycles)", fontsize=11)
    ax.set_title("Predictions with Confidence Intervals", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # 2. Error vs. Prediction confidence
    ax = axes[0, 1]
    if model_predictions is not None and len(model_predictions) > 1:
        all_preds = np.array(model_predictions)
        pred_std = all_preds.std(axis=0)

        scatter = ax.scatter(pred_std, errors, alpha=0.5, s=20, c=y_true, cmap='viridis')
        ax.set_xlabel("Prediction Std Dev", fontsize=11)
        plt.colorbar(scatter, ax=ax, label='True RUL')
    else:
        scatter = ax.scatter(y_true, errors, alpha=0.5, s=20, c=y_pred, cmap='viridis')
        ax.set_xlabel("True RUL", fontsize=11)
        plt.colorbar(scatter, ax=ax, label='Predicted RUL')

    ax.set_ylabel("Absolute Error", fontsize=11)
    ax.set_title("Prediction Uncertainty Analysis", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # 3. Error distribution in different confidence levels
    ax = axes[1, 0]

    # Bin by predicted value
    bins = [0, 25, 50, 100, 200, y_true.max()]
    bin_labels = ['0-25', '25-50', '50-100', '100-200', '200+']

    bin_errors = []
    used_labels = []
    for i in range(len(bins)-1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() > 0:
            bin_errors.append(errors[mask])
            used_labels.append(bin_labels[i])

    bp = ax.boxplot(bin_errors, labels=used_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_xlabel("True RUL Range", fontsize=11)
    ax.set_ylabel("Absolute Error", fontsize=11)
    ax.set_title("Error Distribution by RUL Range", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis='y')

    # 4. Calibration: predicted vs actual error
    ax = axes[1, 1]

    # Sort by predicted value
    sorted_idx = np.argsort(y_pred)
    window = max(20, len(y_pred) // 50)

    smoothed_pred = []
    smoothed_error = []
    smoothed_x = []

    for i in range(0, len(sorted_idx) - window, window):
        idx_window = sorted_idx[i:i+window]
        smoothed_x.append(y_pred[idx_window].mean())
        smoothed_pred.append(y_pred[idx_window].mean())
        smoothed_error.append(errors[idx_window].mean())

    ax.plot(smoothed_x, smoothed_error, 'b-', linewidth=2.5, label='Actual MAE')
    ax.set_xlabel("Predicted RUL", fontsize=11)
    ax.set_ylabel("Mean Absolute Error", fontsize=11)
    ax.set_title("Error vs. Predicted RUL", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nPrediction Confidence Analysis:")
    print(f"  Overall MAE: {errors.mean():.2f} cycles")
    print(f"  Overall RMSE: {np.sqrt(np.mean((y_pred - y_true)**2)):.2f} cycles")

    if model_predictions is not None and len(model_predictions) > 1:
        all_preds = np.array(model_predictions)
        pred_std = all_preds.std(axis=0)
        print(f"  Mean prediction std: {pred_std.mean():.2f} cycles")
        print(f"  Correlation (std vs error): {np.corrcoef(pred_std, errors)[0,1]:.3f}")


def visualize_dataset(
    dev_X: List[np.ndarray],
    dev_y: List[np.ndarray],
    test_X: List[np.ndarray],
    test_y: List[np.ndarray],
    unit_idx: int = 0,
    sensor_indices: Optional[List[int]] = None,
    max_timesteps: int = 500,
) -> None:
    """
    Run all visualization functions for the dataset.

    Args:
        dev_X: Development set features
        dev_y: Development set labels
        test_X: Test set features
        test_y: Test set labels
        unit_idx: Index of unit to visualize for time series
        sensor_indices: Specific sensors to plot
        max_timesteps: Maximum timesteps to display
    """
    # Visualize development set RUL distribution
    print("\n" + "=" * 60)
    print("[Step 1] Visualizing RUL Distribution for Development Set")
    print("=" * 60)
    plot_rul_distribution(dev_y, split_name="Development Set")

    # Visualize test set RUL distribution
    print("\n" + "=" * 60)
    print("[Step 2] Visualizing RUL Distribution for Test Set")
    print("=" * 60)
    plot_rul_distribution(test_y, split_name="Test Set")

    # Visualize sensor time series
    print("\n" + "=" * 60)
    print("[Step 3] Visualizing Sensor Time Series (Unit 0)")
    print("=" * 60)
    plot_sensor_time_series(
        dev_X,
        dev_y,
        unit_idx=unit_idx,
        sensor_indices=sensor_indices,
        max_timesteps=max_timesteps,
    )

    print("\n" + "=" * 60)
    print("✅ All visualizations completed successfully!")
    print("=" * 60)
