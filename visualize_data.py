#!/usr/bin/env python3
"""
Standalone data visualization script for N-CMAPSS dataset.
"""

import argparse
from src.data.load_data import get_datasets
from src.utils.visualize import visualize_dataset


def main():
    parser = argparse.ArgumentParser(description="Visualize N-CMAPSS dataset")
    parser.add_argument("--fd", type=int, default=1, help="FD subset (1-7)")
    parser.add_argument("--unit", type=int, default=0, help="Unit index to visualize")
    parser.add_argument("--sensors", nargs="+", type=int, default=[0, 1, 2, 3], 
                       help="Sensor indices to plot")
    args = parser.parse_args()

    print(f"Loading FD00{args.fd} dataset...")
    (dev_X, dev_y), _, (test_X, test_y) = get_datasets(fd=args.fd, data_dir="data/raw")
    
    print(f"Visualizing unit {args.unit} with sensors {args.sensors}...")
    visualize_dataset(
        dev_X=dev_X,
        dev_y=dev_y,
        test_X=test_X,
        test_y=test_y,
        unit_idx=args.unit,
        sensor_indices=args.sensors,
        max_timesteps=500,
    )


if __name__ == "__main__":
    main()