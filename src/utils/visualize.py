"""
Visualization utilities for N-CMAPSS dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


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
