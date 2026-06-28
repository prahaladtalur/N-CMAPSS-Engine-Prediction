"""
Data loading utilities for N-CMAPSS dataset.
"""

import os
from typing import Tuple, Optional, List
import numpy as np
from rul_datasets.reader.ncmapss import NCmapssReader


FEATURE_SETS = {
    "all": list(range(32)),
    # Reader order: 4 operating conditions, 14 physical sensors, 14 virtual sensors.
    "physical": list(range(18)),
}


def resolve_feature_select(feature_set: str) -> List[int]:
    """Return N-CMAPSS feature indices for a named feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unsupported feature_set '{feature_set}'. Available: {sorted(FEATURE_SETS)}"
        )
    return FEATURE_SETS[feature_set]


def download_ncmapss(
    data_dir: str = "data/raw",
    fd: int = 1,
    cache: bool = True,
    max_rul: int = 65,
    feature_set: str = "all",
    resolution_seconds: int = 1,
) -> NCmapssReader:
    """
    Download and prepare N-CMAPSS dataset.

    Args:
        data_dir: Directory to cache data
        fd: Dataset sub-index (1-7)
        cache: Whether to cache prepared arrays
        max_rul: Maximum RUL cap applied by the reader
        feature_set: "all" for 32 features or "physical" for operating conditions
            plus physical sensors only
        resolution_seconds: Reader downsampling interval in seconds

    Returns:
        NCmapssReader instance
    """
    os.makedirs(data_dir, exist_ok=True)
    os.environ["RUL_DATASETS_DATA_ROOT"] = os.path.abspath(data_dir)

    reader = NCmapssReader(
        fd=fd,
        max_rul=max_rul,
        feature_select=resolve_feature_select(feature_set),
        resolution_seconds=resolution_seconds,
    )
    reader.prepare_data(cache=cache)

    print(
        f"✓ N-CMAPSS FD{fd} prepared and cached in: {data_dir} "
        f"(max_rul={max_rul}, feature_set={feature_set}, resolution_seconds={resolution_seconds})"
    )
    return reader


def get_datasets(
    fd: int = 1,
    data_dir: str = "data/raw",
    cache: bool = True,
    max_rul: int = 65,
    feature_set: str = "all",
    resolution_seconds: int = 1,
) -> Tuple[Tuple[List, List], Optional[Tuple[List, List]], Tuple[List, List]]:
    """
    Load train/dev, validation, and test splits.

    Args:
        fd: N-CMAPSS sub-dataset index (1-7)
        data_dir: Data cache directory
        cache: Whether to cache processed data
        max_rul: Maximum RUL cap applied by the reader
        feature_set: "all" for 32 features or "physical" for operating conditions
            plus physical sensors only
        resolution_seconds: Reader downsampling interval in seconds

    Returns:
        ((dev_X, dev_y), (val_X, val_y), (test_X, test_y))
        Each X is a list of arrays with shape (num_cycles, timesteps, num_sensors)
        Each y is a list of arrays with RUL values
    """
    reader = download_ncmapss(
        data_dir=data_dir,
        fd=fd,
        cache=cache,
        max_rul=max_rul,
        feature_set=feature_set,
        resolution_seconds=resolution_seconds,
    )

    # Load splits
    dev_X, dev_y = reader.load_split("dev")
    test_X, test_y = reader.load_split("test")

    # Validation split may not exist for all datasets
    try:
        val_X, val_y = reader.load_split("val")
        val_pair = (val_X, val_y)
    except Exception:
        val_pair = None
        print("⚠️  No validation split available")

    print(f"✓ Data loaded:")
    print(f"  - Dev units: {len(dev_X)}")
    if val_pair:
        print(f"  - Val units: {len(val_pair[0])}")
    print(f"  - Test units: {len(test_X)}")

    return (dev_X, dev_y), val_pair, (test_X, test_y)


if __name__ == "__main__":
    # Test data loading
    (dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)
    print(f"\nExample shapes:")
    print(f"  Dev unit 0: {dev_X[0].shape}")
    print(f"  Test unit 0: {test_X[0].shape}")
