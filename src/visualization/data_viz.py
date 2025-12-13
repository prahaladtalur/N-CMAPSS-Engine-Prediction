"""
Data exploration and analysis visualizations for N-CMAPSS dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
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
