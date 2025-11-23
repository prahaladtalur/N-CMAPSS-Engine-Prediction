"""
Visualization orchestration for N-CMAPSS RUL prediction pipeline.
"""
from src.utils.visualize import plot_rul_distribution, plot_sensor_time_series


def visualize_rul_distributions(dev_y, test_y):
    """
    Visualize RUL distributions for both development and test sets.

    Args:
        dev_y: Development set RUL labels
        test_y: Test set RUL labels
    """
    # Visualize development set RUL distribution
    print("\n" + "=" * 60)
    print("[Step 2] Visualizing RUL Distribution for Development Set")
    print("=" * 60)
    plot_rul_distribution(dev_y, split_name="Development Set")

    # Visualize test set RUL distribution
    print("\n" + "=" * 60)
    print("[Step 3] Visualizing RUL Distribution for Test Set")
    print("=" * 60)
    plot_rul_distribution(test_y, split_name="Test Set")


def visualize_sensor_data(dev_X, dev_y, unit_idx=0, sensor_indices=None, max_timesteps=500):
    """
    Visualize sensor time series data for a specific unit.

    Args:
        dev_X: Development set features
        dev_y: Development set RUL labels
        unit_idx: Index of the unit to visualize (default: 0)
        sensor_indices: List of sensor indices to plot (default: [0, 1, 2, 3])
        max_timesteps: Maximum number of timesteps to display (default: 500)
    """
    if sensor_indices is None:
        sensor_indices = [0, 1, 2, 3]

    print("\n" + "=" * 60)
    print(f"[Step 4] Visualizing Sensor Time Series (Unit {unit_idx})")
    print("=" * 60)
    plot_sensor_time_series(
        dev_X, dev_y,
        unit_idx=unit_idx,
        sensor_indices=sensor_indices,
        max_timesteps=max_timesteps
    )


def run_all_visualizations(dev_X, dev_y, test_X, test_y):
    """
    Run all visualization steps for the N-CMAPSS dataset.

    Args:
        dev_X: Development set features
        dev_y: Development set RUL labels
        test_X: Test set features
        test_y: Test set RUL labels
    """
    # Visualize RUL distributions
    visualize_rul_distributions(dev_y, test_y)

    # Visualize sensor time series
    visualize_sensor_data(dev_X, dev_y, unit_idx=0, sensor_indices=[0, 1, 2, 3])

    print("\n" + "=" * 60)
    print("✅ All visualizations completed successfully!")
    print("=" * 60)
