# src/search/ — Hyperparameter Search

## Purpose

Grid and random hyperparameter search with parallel execution and live worker throttling. Driven by JSON spec files. Single file module — all logic in `hparam_search.py`.

## CLI Usage

```bash
# Grid search with 4 parallel workers
python train_model.py --search-config spec.json --search-workers 4

# Random search with live throttling
python train_model.py --search-config spec.json --search-throttle-file workers.txt
```

## JSON Spec Format

```json
{
  "name": "my_search",
  "model": "cnn_gru",
  "method": "grid",
  "fd": 1,
  "base_config": {
    "epochs": 50,
    "batch_size": 32
  },
  "parameters": {
    "units": [32, 64, 128],
    "learning_rate": [0.001, 0.0005]
  }
}
```

- `method`: `"grid"` (all combinations) or `"random"` (requires `"num_trials"`)
- `base_config`: Defaults applied to every trial
- `parameters`: Values to sweep over
- Alternative: use an explicit `"trials"` array for hand-picked configs

## Key Functions

| Function | Purpose |
|----------|---------|
| `run_hparam_search(args)` | Main entry point, called from `train_model.py` CLI |
| `load_search_spec(path)` | Load and validate JSON spec |
| `generate_search_jobs(args, spec)` | Expand spec into concrete job dicts |
| `execute_training_job(job)` | Run single training job (for multiprocessing) |
| `read_worker_limit(default, throttle_file)` | Live worker count adjustment |

## Live Throttling

Write an integer to the throttle file to adjust workers mid-search:
```bash
echo 2 > workers.txt    # reduce to 2 workers
echo 8 > workers.txt    # scale up to 8 workers
```

## Results

Written to `results/search/{name}_results.json`.

## Pitfalls

- Each search worker loads the full dataset independently — memory-heavy with many workers
- Lazy import of `train_model` in `execute_training_job` to avoid circular deps
- `sys.path` manipulation to find project root (necessary for subprocess imports)
- Grid search with many parameters creates combinatorial explosion — prefer random search for large spaces
- Results are not aggregated with W&B automatically; each trial runs as a separate W&B run
- See [../models/CLAUDE.md](../models/CLAUDE.md) for available model names and hyperparameter kwargs
