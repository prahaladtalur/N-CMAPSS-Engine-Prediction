"""
Data loading utilities for N-CMAPSS dataset.

N-CMAPSS channel layout (32 total):
  Channels  0– 3  W    — operating conditions (alt, Mach, TRA, T2)
  Channels  4–17  X_s  — 14 physical sensor measurements
  Channels 18–31  X_v  — 14 virtual (model-computed) sensors

Literature consistently finds that X_v channels are redundant and can hurt
data-driven models.  Use ``feature_select=list(range(18))`` (W + X_s only)
to drop the virtual sensors — this is the recommended setting.
"""

import os
from typing import Tuple, Optional, List
import numpy as np
from rul_datasets.reader.ncmapss import NCmapssReader

# Channel index ranges for the three channel groups
W_CHANNELS = list(range(0, 4))       # operating conditions
XS_CHANNELS = list(range(4, 18))     # physical sensors
XV_CHANNELS = list(range(18, 32))    # virtual sensors (often dropped)
ALL_CHANNELS = list(range(32))
PHYSICAL_CHANNELS = W_CHANNELS + XS_CHANNELS   # 18 channels, recommended default


def download_ncmapss(
    data_dir: str = "data/raw",
    fd: int = 1,
    cache: bool = True,
    feature_select: Optional[List[int]] = None,
) -> NCmapssReader:
    """
    Download and prepare N-CMAPSS dataset.

    Args:
        data_dir: Directory to cache data
        fd: Dataset sub-index (1-7)
        cache: Whether to cache prepared arrays
        feature_select: Channel indices to retain (default: all 32).
            Pass ``PHYSICAL_CHANNELS`` (channels 0-17) to drop virtual sensors.

    Returns:
        NCmapssReader instance
    """
    os.makedirs(data_dir, exist_ok=True)
    os.environ["RUL_DATASETS_DATA_ROOT"] = os.path.abspath(data_dir)

    reader_kwargs: dict = {"fd": fd}
    if feature_select is not None:
        reader_kwargs["feature_select"] = feature_select

    reader = NCmapssReader(**reader_kwargs)
    reader.prepare_data(cache=cache)

    n_features = len(feature_select) if feature_select is not None else 32
    print(f"✓ N-CMAPSS FD{fd} prepared — {n_features} features, cached in: {data_dir}")
    return reader


def get_datasets(
    fd: int = 1,
    data_dir: str = "data/raw",
    cache: bool = True,
    feature_select: Optional[List[int]] = None,
) -> Tuple[Tuple[List, List], Optional[Tuple[List, List]], Tuple[List, List]]:
    """
    Load train/dev, validation, and test splits.

    Args:
        fd: N-CMAPSS sub-dataset index (1-7)
        data_dir: Data cache directory
        cache: Whether to cache processed data
        feature_select: Channel indices to retain (default: all 32).
            Pass ``PHYSICAL_CHANNELS`` to drop virtual sensors (recommended).

    Returns:
        ((dev_X, dev_y), (val_X, val_y), (test_X, test_y))
        Each X is a list of arrays with shape (num_cycles, timesteps, num_sensors)
        Each y is a list of arrays with RUL values
    """
    reader = download_ncmapss(data_dir=data_dir, fd=fd, cache=cache, feature_select=feature_select)

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
