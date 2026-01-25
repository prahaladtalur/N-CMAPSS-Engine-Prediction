"""
Hyperparameter search framework for training experiments.

Supports grid and random search with live-throttled parallel execution.
"""

import itertools
import json
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.data.load_data import get_datasets
# Import train_model lazily to avoid circular imports during CLI startup.
import sys

# Add project root to path to import train_model when needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_search_spec(path: str) -> Dict:
    """Load search specification from JSON."""
    with open(path, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in search spec {path}: {exc}") from exc


def _grid_combinations(parameter_grid: Dict[str, List]) -> Iterable[Dict]:
    """Yield all combinations for a grid search."""
    keys = list(parameter_grid.keys())
    values = [parameter_grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _random_combinations(parameter_grid: Dict[str, List], num_trials: int) -> Iterable[Dict]:
    """Yield random samples from parameter grid."""
    if num_trials <= 0:
        raise ValueError("Random search requires 'num_trials' > 0.")

    keys = list(parameter_grid.keys())
    for _ in range(num_trials):
        yield {key: random.choice(parameter_grid[key]) for key in keys}


SEARCH_METHODS = {
    "grid": lambda params, cfg: _grid_combinations(params),
    "random": lambda params, cfg: _random_combinations(params, cfg.get("num_trials", 0)),
}


def generate_search_jobs(args, spec: Dict) -> List[Dict]:
    """Generate concrete training jobs from a search specification."""
    base_name = spec.get("name", "search")
    base_model = spec.get("model")
    if not base_model:
        raise ValueError("Search spec must include a 'model' field.")

    base_config = spec.get("base_config", {})
    parameter_grid = spec.get("parameters", {})
    explicit_trials = spec.get("trials")
    method = spec.get("method", spec.get("mode", "grid")).lower()
    fd = spec.get("fd", args.fd)

    jobs = []

    if explicit_trials:
        for idx, trial in enumerate(explicit_trials, start=1):
            config = {**base_config, **trial.get("config", {})}
            job_model = trial.get("model", base_model)
            jobs.append(
                {
                    "job_id": f"{base_name}-trial-{idx}",
                    "model": job_model,
                    "config": config,
                    "fd": trial.get("fd", fd),
                }
            )
        return jobs

    if not parameter_grid:
        raise ValueError("Search spec must define 'parameters' or explicit 'trials'.")

    if method not in SEARCH_METHODS:
        available = ", ".join(sorted(SEARCH_METHODS))
        raise ValueError(
            f"Unsupported search method '{method}'. Available methods: {available}"
        )

    method_config = {"num_trials": spec.get("num_trials", 0)}
    combos = SEARCH_METHODS[method](parameter_grid, method_config)

    for idx, combo in enumerate(combos, start=1):
        config = {**base_config, **combo}
        jobs.append(
            {
                "job_id": f"{base_name}-trial-{idx}",
                "model": base_model,
                "config": config,
                "fd": fd,
            }
        )

    return jobs


def read_worker_limit(default_limit: int, throttle_file: Optional[str]) -> int:
    """Read current worker limit, optionally from a throttle file."""
    limit = default_limit
    if throttle_file:
        try:
            with open(throttle_file, "r", encoding="utf-8") as file:
                value = file.read().strip()
                parsed = int(value)
                if parsed > 0:
                    limit = parsed
        except FileNotFoundError:
            pass
        except ValueError:
            print(
                f"Warning: Invalid worker count in {throttle_file!r}. "
                f"Falling back to {default_limit}."
            )
    return max(1, limit)


def execute_training_job(job: Dict) -> Dict:
    """Execute a single training job (designed for multiprocessing)."""
    try:
        print(f"[{job['job_id']}] Starting {job['model']} with config: {job['config']}")
        (dev_X, dev_y), val_pair, (test_X, test_y) = get_datasets(fd=job["fd"])
        val_X, val_y = val_pair if val_pair else (None, None)

        normalize = job.get("normalize", True)
        visualize = job.get("visualize", False)
        project_name = job.get("project_name", "n-cmapss-rul-search")
        run_name = job.get("run_name") or f"{job['model']}-{job['job_id']}"

        from train_model import train_model

        _, _, metrics = train_model(
            dev_X=dev_X,
            dev_y=dev_y,
            model_name=job["model"],
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            config=job["config"],
            project_name=project_name,
            run_name=run_name,
            normalize=normalize,
            visualize=visualize,
        )

        print(f"[{job['job_id']}] Completed.")
        return {"job_id": job["job_id"], "model": job["model"], "metrics": metrics}

    except Exception as exc:  # pylint: disable=broad-except
        print(f"[{job['job_id']}] Failed: {exc}")
        return {"job_id": job["job_id"], "model": job["model"], "error": str(exc)}


def run_hparam_search(args):
    """Coordinate hyperparameter search with optional parallelism and throttling."""
    spec = load_search_spec(args.search_config)
    search_name = spec.get("name", "search")
    jobs = generate_search_jobs(args, spec)

    if not jobs:
        print("No jobs generated from search spec.")
        return

    print(f"\nStarting search '{search_name}' with {len(jobs)} trials.")
    throttle_file = args.search_throttle_file
    initial_limit = max(1, args.search_workers)
    pool_cap = args.search_max_workers or max(initial_limit, os.cpu_count() or 1)

    normalize = not args.no_normalize
    visualize = not args.no_visualize
    project_name = spec.get("project_name", args.project)

    for job in jobs:
        job["normalize"] = spec.get("normalize", normalize)
        job["visualize"] = spec.get("visualize", visualize)
        job["project_name"] = project_name
        job["run_name"] = f"{search_name}-{job['job_id']}"

    pending_jobs = iter(jobs)
    active_futures = {}
    results = []

    with ProcessPoolExecutor(max_workers=pool_cap) as executor:
        next_job = next(pending_jobs, None)
        while active_futures or next_job is not None:
            current_limit = min(read_worker_limit(initial_limit, throttle_file), pool_cap)

            while next_job is not None and len(active_futures) < current_limit:
                future = executor.submit(execute_training_job, next_job)
                active_futures[future] = next_job["job_id"]
                next_job = next(pending_jobs, None)

            if not active_futures:
                time.sleep(0.5)
                continue

            done, _ = wait(active_futures.keys(), timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                job_id = active_futures.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"[{job_id}] Error retrieving result: {exc}")
                    results.append({"job_id": job_id, "error": str(exc)})

    os.makedirs("results/search", exist_ok=True)
    output_path = Path("results/search") / f"{search_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"search": search_name, "results": results}, file, indent=2)

    print(f"\nSearch complete. Results written to {output_path}")
