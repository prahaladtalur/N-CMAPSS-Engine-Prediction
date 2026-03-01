# src/data/ — Data Loading

## Purpose

Download, cache, and load the NASA N-CMAPSS turbofan engine degradation dataset using the `rul_datasets` library. Single file module — all logic in `load_data.py`.

## Key Functions

### get_datasets(fd=1, data_dir="data/raw", cache=True)

Main entry point. Returns:
```
((dev_X, dev_y), val_pair_or_None, (test_X, test_y))
```

- `dev` = training split, `test` = held-out evaluation
- `val` may be `None` for some FD sub-datasets
- Each X is `List[np.ndarray]` — one array per engine unit
- Each unit array shape: `(num_cycles, timesteps, num_sensors)`
- Each y is `List[np.ndarray]` — RUL values per cycle

### download_ncmapss(data_dir, fd, cache)

Downloads and prepares data via `NCmapssReader`. Called internally by `get_datasets`. Sets `RUL_DATASETS_DATA_ROOT` env var. First run downloads ~1GB.

## Data Flow in Training

1. `get_datasets()` returns `List[ndarray]` per split
2. `prepare_sequences()` in `train_model.py:68` flattens into single arrays: `(N, timesteps, features)` and `(N,)`
3. `normalize_data()` in `train_model.py:100` fits `StandardScaler` on train, transforms all splits
4. Model receives `(batch, timesteps, features)` tensors

See [../../CLAUDE.md](../../CLAUDE.md) for the full training pipeline overview.

## Available Sub-datasets

FD1 through FD7 (`--fd 1..7`). FD1 is the default and most commonly used. Different sub-datasets have different engine operating conditions and failure modes.

## Pitfalls

- First run triggers ~1GB download; ensure disk space and connectivity
- Validation split may not exist for all FD values (returns `None`)
- Full sequences can be ~20k timesteps; `train_model.py` truncates to 1000 via `max_sequence_length` to avoid OOM
- Data is cached in `data/raw/` after first download — delete this directory to force re-download
