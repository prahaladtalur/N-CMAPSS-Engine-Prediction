"""
Example script demonstrating visualization capabilities for RUL prediction.

This script shows how to use all visualization functions from src/utils/visualize.py
for analyzing engine degradation and model performance.

Can be run interactively or via command line.
"""

import argparse
import numpy as np
from src.data.load_data import get_datasets
from src.utils import (
    # Data Analysis Visualizations
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap,
    plot_multi_sensor_lifecycle,
    # Model Evaluation Visualizations
    plot_rul_trajectory,
    plot_critical_zone_analysis,
    plot_prediction_confidence,
    # Basic Visualizations
    plot_rul_distribution,
    plot_sensor_time_series,
    visualize_dataset,
)


def example_data_visualizations():
    """
    Demonstrate data-focused visualizations (before model training).
    """
    print("="*80)
    print("DATA ANALYSIS VISUALIZATIONS")
    print("="*80)

    # Load data
    print("\nLoading N-CMAPSS dataset...")
    (dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)

    # 1. Sensor Degradation Analysis
    print("\n" + "="*60)
    print("1. Sensor Degradation Analysis")
    print("="*60)
    print("Shows how sensor values change as engines degrade (RUL decreases)")
    plot_sensor_degradation(dev_X, dev_y, unit_idx=0, sensor_indices=[0, 1, 2, 3, 4, 5])

    # 2. Sensor Correlation Heatmap
    print("\n" + "="*60)
    print("2. Sensor-RUL Correlation Heatmap")
    print("="*60)
    print("Identifies which sensors are most predictive of engine health")
    plot_sensor_correlation_heatmap(dev_X, dev_y, max_sensors=14)

    # 3. Multi-Sensor Lifecycle Comparison
    print("\n" + "="*60)
    print("3. Multi-Sensor Lifecycle Comparison")
    print("="*60)
    print("Compares multiple sensors side-by-side over engine lifecycle")
    plot_multi_sensor_lifecycle(dev_X, dev_y, unit_idx=0, max_sensors=8)


def example_model_visualizations():
    """
    Demonstrate model evaluation visualizations (after model training).

    NOTE: This requires trained model predictions. For demonstration,
    we'll create synthetic predictions. Replace with actual model predictions.
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION VISUALIZATIONS")
    print("="*80)

    # Load data
    print("\nLoading test data...")
    (dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)

    # Flatten test labels for demonstration
    y_true = np.concatenate(test_y)

    # Create synthetic predictions for demonstration
    # In practice, replace this with: y_pred = model.predict(test_X)
    print("\nGenerating synthetic predictions for demonstration...")
    print("(Replace with actual model predictions in practice)")
    np.random.seed(42)
    noise = np.random.normal(0, 15, len(y_true))
    y_pred = y_true + noise
    y_pred = np.clip(y_pred, 0, None)  # RUL can't be negative

    # Get unit lengths for trajectory visualization
    unit_lengths = [len(y) for y in test_y]

    # 4. RUL Trajectory for Individual Engine
    print("\n" + "="*60)
    print("4. RUL Trajectory Analysis")
    print("="*60)
    print("Shows predicted vs actual RUL over an engine's lifecycle")
    plot_rul_trajectory(y_true, y_pred, unit_length=unit_lengths, unit_idx=0)

    # 5. Critical Zone Analysis
    print("\n" + "="*60)
    print("5. Critical Zone Analysis")
    print("="*60)
    print("Analyzes model performance in critical RUL zones")
    plot_critical_zone_analysis(y_true, y_pred, critical_threshold=30, warning_threshold=75)

    # 6. Prediction Confidence Analysis
    print("\n" + "="*60)
    print("6. Prediction Confidence Analysis")
    print("="*60)
    print("Visualizes prediction confidence and uncertainty")

    # Option A: Without ensemble (error-based confidence)
    plot_prediction_confidence(y_true, y_pred)

    # Option B: With ensemble predictions (if available)
    # Create synthetic ensemble for demonstration
    print("\nDemonstrating with synthetic ensemble predictions...")
    ensemble_predictions = [y_true + np.random.normal(0, 15, len(y_true)) for _ in range(5)]
    ensemble_predictions = [np.clip(pred, 0, None) for pred in ensemble_predictions]
    y_pred_ensemble = np.mean(ensemble_predictions, axis=0)

    plot_prediction_confidence(y_true, y_pred_ensemble, model_predictions=ensemble_predictions)


def example_combined_analysis():
    """
    Demonstrate a complete analysis workflow combining multiple visualizations.
    """
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS WORKFLOW")
    print("="*80)

    # Load data
    (dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)

    # Step 1: Understand data characteristics
    print("\nStep 1: Analyzing data characteristics...")
    plot_sensor_correlation_heatmap(dev_X, dev_y, max_sensors=10)

    # Step 2: Identify degradation patterns
    print("\nStep 2: Identifying sensor degradation patterns...")
    plot_sensor_degradation(dev_X, dev_y, unit_idx=0)

    # Step 3: Train model (synthetic for demo)
    print("\nStep 3: Training model (simulated)...")
    y_true = np.concatenate(test_y)
    y_pred = y_true + np.random.normal(0, 12, len(y_true))
    y_pred = np.clip(y_pred, 0, None)

    # Step 4: Evaluate overall performance
    print("\nStep 4: Evaluating model performance...")
    plot_critical_zone_analysis(y_true, y_pred)

    # Step 5: Analyze specific units
    print("\nStep 5: Analyzing specific engine units...")
    unit_lengths = [len(y) for y in test_y]
    plot_rul_trajectory(y_true, y_pred, unit_length=unit_lengths, unit_idx=0)
    plot_rul_trajectory(y_true, y_pred, unit_length=unit_lengths, unit_idx=1)


def run_basic_visualizations(fd=1, unit_idx=0, sensor_indices=None, max_timesteps=500):
    """
    Run basic dataset visualizations (RUL distribution and sensor time series).
    
    Args:
        fd: FD subset (1-7)
        unit_idx: Unit index to visualize
        sensor_indices: List of sensor indices to plot
        max_timesteps: Maximum timesteps to display
    """
    print(f"\nLoading FD00{fd} dataset...")
    (dev_X, dev_y), _, (test_X, test_y) = get_datasets(fd=fd)
    
    print(f"\nVisualizing unit {unit_idx}...")
    visualize_dataset(
        dev_X=dev_X,
        dev_y=dev_y,
        test_X=test_X,
        test_y=test_y,
        unit_idx=unit_idx,
        sensor_indices=sensor_indices,
        max_timesteps=max_timesteps,
    )


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description="Visualization examples for N-CMAPSS RUL prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python example_visualizations.py

  # Run basic visualizations via CLI
  python example_visualizations.py --basic --fd 1 --unit 0

  # Run data analysis visualizations
  python example_visualizations.py --data --fd 1

  # Run model evaluation visualizations
  python example_visualizations.py --model --fd 1
        """,
    )
    
    parser.add_argument("--fd", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                       help="FD subset (default: 1)")
    parser.add_argument("--unit", type=int, default=0,
                       help="Unit index to visualize (default: 0)")
    parser.add_argument("--sensors", nargs="+", type=int, default=None,
                       help="Sensor indices to plot (default: [0, 1, 2, 3])")
    parser.add_argument("--max-timesteps", type=int, default=500,
                       help="Maximum timesteps to display (default: 500)")
    
    # Mode selection
    parser.add_argument("--basic", action="store_true",
                       help="Run basic dataset visualizations")
    parser.add_argument("--data", action="store_true",
                       help="Run data analysis visualizations")
    parser.add_argument("--model", action="store_true",
                       help="Run model evaluation visualizations")
    parser.add_argument("--combined", action="store_true",
                       help="Run complete analysis workflow")
    parser.add_argument("--all", action="store_true",
                       help="Run all visualizations")
    
    args = parser.parse_args()
    
    # If no mode specified, run interactive mode
    if not any([args.basic, args.data, args.model, args.combined, args.all]):
        print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║          Advanced Visualization Examples for RUL Prediction             ║
    ║                   N-CMAPSS Turbofan Engine Dataset                      ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    This script demonstrates all visualization capabilities:

    DATA VISUALIZATIONS (before training):
    • Sensor Degradation Analysis - How sensors change as engines degrade
    • Sensor Correlation Heatmap - Which sensors predict failure best
    • Multi-Sensor Lifecycle Comparison - Side-by-side sensor comparison

    MODEL VISUALIZATIONS (after training):
    • RUL Trajectory - Predicted vs actual RUL over engine lifecycle
    • Critical Zone Analysis - Performance when engines are near failure
    • Prediction Confidence - Uncertainty and confidence intervals

    Choose an option:
    """)

        print("1. Run data analysis visualizations")
        print("2. Run model evaluation visualizations (with synthetic predictions)")
        print("3. Run complete analysis workflow")
        print("4. Run all visualizations\n")

        choice = input("Enter choice (1-4, or 'q' to quit): ").strip()

        if choice == '1':
            example_data_visualizations()
        elif choice == '2':
            example_model_visualizations()
        elif choice == '3':
            example_combined_analysis()
        elif choice == '4':
            example_data_visualizations()
            example_model_visualizations()
            example_combined_analysis()
        elif choice.lower() == 'q':
            print("Exiting...")
            return
        else:
            print("Invalid choice. Running data visualizations by default...")
            example_data_visualizations()
    else:
        # CLI mode
        if args.basic:
            run_basic_visualizations(
                fd=args.fd,
                unit_idx=args.unit,
                sensor_indices=args.sensors,
                max_timesteps=args.max_timesteps
            )
        elif args.data:
            example_data_visualizations()
        elif args.model:
            example_model_visualizations()
        elif args.combined:
            example_combined_analysis()
        elif args.all:
            run_basic_visualizations(
                fd=args.fd,
                unit_idx=args.unit,
                sensor_indices=args.sensors,
                max_timesteps=args.max_timesteps
            )
            example_data_visualizations()
            example_model_visualizations()
            example_combined_analysis()

    print("\n" + "="*80)
    print("✅ Visualization examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
