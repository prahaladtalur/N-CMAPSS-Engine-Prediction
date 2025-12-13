#!/usr/bin/env python
"""
Visualize N-CMAPSS dataset and model predictions.

Usage examples:
    # Visualize dataset (data exploration)
    python visualize.py --data --fd 1

    # Visualize specific sensors
    python visualize.py --data --fd 1 --sensors 0 1 2 3

    # Visualize model predictions (requires trained model)
    python visualize.py --model outputs/models/lstm-run/model.h5 --fd 1

    # Generate all visualizations
    python visualize.py --data --all --fd 1
"""

import argparse
import sys
from src.data.loader import get_datasets
from src.visualization.data_viz import (
    plot_rul_distribution,
    plot_sensor_time_series,
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap,
    plot_multi_sensor_lifecycle,
)


def visualize_data(
    dev_X, dev_y, test_X, test_y, unit_idx=0, sensor_indices=None, show_all=False
):
    """
    Run data exploration visualizations.

    Args:
        dev_X: Development set features
        dev_y: Development set labels
        test_X: Test set features
        test_y: Test set labels
        unit_idx: Index of unit to visualize
        sensor_indices: List of sensor indices to visualize
        show_all: Whether to show all visualizations
    """
    print("\n" + "=" * 80)
    print("DATA EXPLORATION VISUALIZATIONS")
    print("=" * 80)

    # RUL Distribution
    print("\n[1/5] Visualizing RUL Distribution - Development Set")
    plot_rul_distribution(dev_y, split_name="Development Set")

    print("\n[2/5] Visualizing RUL Distribution - Test Set")
    plot_rul_distribution(test_y, split_name="Test Set")

    # Sensor Time Series
    print(f"\n[3/5] Visualizing Sensor Time Series (Unit {unit_idx})")
    plot_sensor_time_series(
        dev_X,
        dev_y,
        unit_idx=unit_idx,
        sensor_indices=sensor_indices,
        num_sensors=4 if sensor_indices is None else len(sensor_indices),
    )

    if show_all:
        # Sensor Degradation
        print(f"\n[4/5] Visualizing Sensor Degradation Patterns (Unit {unit_idx})")
        plot_sensor_degradation(
            dev_X, dev_y, unit_idx=unit_idx, sensor_indices=sensor_indices
        )

        # Correlation Heatmap
        print("\n[5/5] Visualizing Sensor-RUL Correlations")
        plot_sensor_correlation_heatmap(dev_X, dev_y)

        # Multi-sensor Lifecycle
        print(f"\nBonus: Multi-Sensor Lifecycle Comparison (Unit {unit_idx})")
        plot_multi_sensor_lifecycle(dev_X, dev_y, unit_idx=unit_idx)

    print("\n" + "=" * 80)
    print("✅ Data visualization complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize N-CMAPSS dataset and model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize dataset
  python visualize.py --data --fd 1

  # Visualize specific sensors
  python visualize.py --data --fd 1 --sensors 0 1 2 3

  # Show all visualizations
  python visualize.py --data --all --fd 1

  # Visualize specific unit
  python visualize.py --data --fd 1 --unit 5
        """,
    )

    # Mode selection
    parser.add_argument(
        "--data",
        action="store_true",
        help="Visualize dataset (data exploration)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model for prediction visualization",
    )

    # Data options
    parser.add_argument(
        "--fd",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="N-CMAPSS sub-dataset index (default: 1)",
    )
    parser.add_argument(
        "--unit",
        type=int,
        default=0,
        help="Unit index to visualize (default: 0)",
    )
    parser.add_argument(
        "--sensors",
        type=int,
        nargs="+",
        default=None,
        help="Sensor indices to visualize (default: first 4)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all available visualizations",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.data and not args.model:
        parser.print_help()
        print("\n\nError: Must specify --data or --model")
        sys.exit(1)

    # Load data
    print(f"\nLoading N-CMAPSS FD{args.fd} dataset...")
    (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=args.fd)

    # Data visualization mode
    if args.data:
        visualize_data(
            dev_X=dev_X,
            dev_y=dev_y,
            test_X=test_X,
            test_y=test_y,
            unit_idx=args.unit,
            sensor_indices=args.sensors,
            show_all=args.all,
        )

    # Model visualization mode
    if args.model:
        print("\n" + "=" * 80)
        print("MODEL PREDICTION VISUALIZATIONS")
        print("=" * 80)
        print("\nModel visualization coming soon!")
        print("For now, use the training script with --model flag to generate")
        print("model evaluation visualizations automatically.")
        print("=" * 80)


if __name__ == "__main__":
    main()
