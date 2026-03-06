"""
Tests for hyperparameter search utilities.

Covers config parsing, trial generation, and search method selection without
running any actual training.
"""

import json
import types

import pytest

from src.search.hparam_search import (
    load_search_spec,
    generate_search_jobs,
    _grid_combinations,
    _random_combinations,
)


# ---------------------------------------------------------------------------
# _grid_combinations
# ---------------------------------------------------------------------------


class TestGridCombinations:
    def test_single_param(self):
        grid = {"lr": [0.001, 0.01]}
        combos = list(_grid_combinations(grid))
        assert len(combos) == 2
        assert {"lr": 0.001} in combos
        assert {"lr": 0.01} in combos

    def test_two_params_cartesian(self):
        grid = {"lr": [0.001, 0.01], "units": [32, 64]}
        combos = list(_grid_combinations(grid))
        assert len(combos) == 4  # 2 × 2
        # All four combinations present
        assert {"lr": 0.001, "units": 32} in combos
        assert {"lr": 0.01, "units": 64} in combos

    def test_single_value_per_param(self):
        grid = {"lr": [0.001], "units": [64]}
        combos = list(_grid_combinations(grid))
        assert len(combos) == 1
        assert combos[0] == {"lr": 0.001, "units": 64}

    def test_empty_grid(self):
        combos = list(_grid_combinations({}))
        assert combos == [{}]

    def test_three_params(self):
        grid = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        combos = list(_grid_combinations(grid))
        assert len(combos) == 8  # 2^3


# ---------------------------------------------------------------------------
# _random_combinations
# ---------------------------------------------------------------------------


class TestRandomCombinations:
    def test_returns_correct_number_of_trials(self):
        grid = {"lr": [0.001, 0.01, 0.1], "units": [32, 64, 128]}
        combos = list(_random_combinations(grid, num_trials=5))
        assert len(combos) == 5

    def test_each_combo_is_subset_of_grid(self):
        grid = {"lr": [0.001, 0.01], "units": [32, 64]}
        for combo in _random_combinations(grid, num_trials=20):
            assert combo["lr"] in grid["lr"]
            assert combo["units"] in grid["units"]

    def test_zero_trials_raises(self):
        with pytest.raises(ValueError):
            list(_random_combinations({"lr": [0.001]}, num_trials=0))

    def test_negative_trials_raises(self):
        with pytest.raises(ValueError):
            list(_random_combinations({"lr": [0.001]}, num_trials=-1))

    def test_single_option_always_chosen(self):
        grid = {"lr": [0.001]}
        combos = list(_random_combinations(grid, num_trials=10))
        assert all(c["lr"] == 0.001 for c in combos)


# ---------------------------------------------------------------------------
# load_search_spec
# ---------------------------------------------------------------------------


class TestLoadSearchSpec:
    def test_loads_valid_json(self, tmp_path):
        spec = {"model": "lstm", "parameters": {"lr": [0.001]}}
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        result = load_search_spec(str(path))
        assert result == spec

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_search_spec(str(tmp_path / "does_not_exist.json"))

    def test_raises_on_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(ValueError):
            load_search_spec(str(path))


# ---------------------------------------------------------------------------
# generate_search_jobs
# ---------------------------------------------------------------------------


def _fake_args(**kwargs):
    """Return a simple namespace mimicking parsed CLI args."""
    defaults = {"fd": 1, "seed": 42}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class TestGenerateSearchJobs:
    def test_grid_search_generates_correct_jobs(self):
        spec = {
            "model": "lstm",
            "method": "grid",
            "base_config": {"epochs": 10},
            "parameters": {"lr": [0.001, 0.01], "units": [32, 64]},
        }
        jobs = generate_search_jobs(_fake_args(), spec)
        # 2 × 2 = 4 combinations
        assert len(jobs) == 4

    def test_random_search_generates_correct_jobs(self):
        spec = {
            "model": "lstm",
            "method": "random",
            "num_trials": 3,
            "parameters": {"lr": [0.001, 0.01], "units": [32, 64]},
        }
        jobs = generate_search_jobs(_fake_args(), spec)
        assert len(jobs) == 3

    def test_each_job_has_required_keys(self):
        spec = {
            "model": "lstm",
            "method": "grid",
            "parameters": {"lr": [0.001]},
        }
        jobs = generate_search_jobs(_fake_args(), spec)
        assert len(jobs) == 1
        job = jobs[0]
        assert "model_name" in job or "model" in job
        assert "config" in job

    def test_base_config_merged_into_jobs(self):
        spec = {
            "model": "lstm",
            "method": "grid",
            "base_config": {"epochs": 5, "batch_size": 32},
            "parameters": {"lr": [0.001]},
        }
        jobs = generate_search_jobs(_fake_args(), spec)
        config = jobs[0]["config"]
        assert config.get("epochs") == 5
        assert config.get("batch_size") == 32

    def test_missing_model_raises(self):
        spec = {"method": "grid", "parameters": {"lr": [0.001]}}
        with pytest.raises(ValueError, match="model"):
            generate_search_jobs(_fake_args(), spec)

    def test_explicit_trials_used_directly(self):
        spec = {
            "model": "lstm",
            "trials": [
                {"config": {"lr": 0.001}},
                {"config": {"lr": 0.01}},
            ],
        }
        jobs = generate_search_jobs(_fake_args(), spec)
        assert len(jobs) == 2

    def test_fd_from_spec_overrides_args(self):
        spec = {
            "model": "lstm",
            "method": "grid",
            "fd": 3,
            "parameters": {"lr": [0.001]},
        }
        jobs = generate_search_jobs(_fake_args(fd=1), spec)
        assert jobs[0]["fd"] == 3
