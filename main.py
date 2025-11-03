"""
Main entry point for N-CMAPSS RUL prediction pipeline.
"""
import sys
from src.data.load_data import get_datasets
from src.utils.visualize import plot_rul_distribution, plot_sensor_time_series


def main():
    """Run the complete data pipeline."""
    print("=" * 60)
    print("N-CMAPSS RUL Prediction Data Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading N-CMAPSS FD001 dataset...")
    try:
        (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=1, data_dir="data/raw")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        sys.exit(1)
    
    # Step 2: Visualize development set RUL distribution
    print("\n" + "=" * 60)
    print("[Step 2] Visualizing RUL Distribution for Development Set")
    print("=" * 60)
    plot_rul_distribution(dev_y, split_name="Development Set")
    
    # Step 3: Visualize test set RUL distribution
    print("\n" + "=" * 60)
    print("[Step 3] Visualizing RUL Distribution for Test Set")
    print("=" * 60)
    plot_rul_distribution(test_y, split_name="Test Set")
    
    # Step 4: Visualize sensor time series
    print("\n" + "=" * 60)
    print("[Step 4] Visualizing Sensor Time Series (Unit 0)")
    print("=" * 60)
    plot_sensor_time_series(
        dev_X, dev_y,
        unit_idx=0,
        sensor_indices=[0, 1, 2, 3],
        max_timesteps=500
    )
    
    print("\n" + "=" * 60)
    print("✅ All visualizations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
