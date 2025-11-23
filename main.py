"""
Main entry point for N-CMAPSS RUL prediction pipeline.
"""
import sys
from src.data.load_data import get_datasets
from visualize import run_all_visualizations


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

    # Step 2-4: Run all visualizations
    run_all_visualizations(dev_X, dev_y, test_X, test_y)


if __name__ == "__main__":
    main()
