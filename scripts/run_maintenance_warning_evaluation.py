#!/usr/bin/env python3
"""Evaluate a maintenance-window warning rule on unseen simulated engines.

This experiment is separate from the FD1 RUL-regression benchmark. It uses the
labeled train partition from the public tiny-N-CMAPSS repository. Each outer
fold holds out complete engine trajectories. Within every outer-training fold,
an inner engine-grouped validation pass chooses the lowest alert threshold that
keeps the false-alert rate on pre-window cycles at or below a fixed budget.

The primary model uses physical measurement summaries only. An operating-state
baseline uses only altitude, Mach, throttle resolver angle, and inlet
temperature summaries. Neither model receives engine ID, flight-cycle number,
or RUL as an input.

Example:
  /Library/Developer/CommandLineTools/usr/bin/python3 \
      scripts/run_maintenance_warning_evaluation.py \
      --data-dir /tmp/ncmapss_tiny \
      --output-dir benchmark_results/maintenance_warning/tiny_ncmapss_nested_20260812
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from run_engine_warning_experiment import (
    OPERATING_COLUMNS,
    SOURCE_REPOSITORY,
    ensure_data,
)


MEASURED_SENSOR_CHANNELS = [
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf",
]

# In the N-CMAPSS schema, Fc (flight class) and hs (health state) are auxiliary
# data. They are retained in the public frame for provenance checks, but they
# are never model inputs. In particular, hs is a simulated health-state label
# and would invalidate a sensor-warning experiment.
AUXILIARY_COLUMNS = ["Fc", "hs"]


def build_cycle_table(train_path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Aggregate observable measurements to one row per engine cycle.

    The returned feature sets deliberately separate elapsed cycle, operating
    conditions, and the 14 measured physical channels. Auxiliary N-CMAPSS
    fields are validated but never summarized as features.
    """
    frame = pd.read_pickle(train_path).copy()
    required = {
        "unit",
        "cycle",
        "RUL",
        *MEASURED_SENSOR_CHANNELS,
        *OPERATING_COLUMNS,
        *AUXILIARY_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame["unit"] = frame["unit"].astype("int16")
    frame["cycle"] = frame["cycle"].astype("int16")
    feature_columns = [*MEASURED_SENSOR_CHANNELS, *OPERATING_COLUMNS]
    frame[feature_columns] = frame[feature_columns].astype("float32")
    grouped = frame.groupby(["unit", "cycle"], sort=True)
    labels = grouped["RUL"].agg(["first", "nunique"]).rename(
        columns={"first": "rul", "nunique": "label_count"}
    )
    if int(labels["label_count"].max()) != 1:
        raise ValueError("RUL labels vary within at least one engine cycle.")
    means = grouped[feature_columns].mean().add_suffix("__mean")
    stds = grouped[feature_columns].std(ddof=0).fillna(0.0).add_suffix("__std")
    cycle_table = pd.concat([means, stds, labels[["rul"]]], axis=1).reset_index()
    cycle_table["cycle"] = cycle_table["cycle"].astype("float32")
    sensor_summary_columns = [
        *(f"{column}__mean" for column in MEASURED_SENSOR_CHANNELS),
        *(f"{column}__std" for column in MEASURED_SENSOR_CHANNELS),
    ]
    operating_summary_columns = [
        *(f"{column}__mean" for column in OPERATING_COLUMNS),
        *(f"{column}__std" for column in OPERATING_COLUMNS),
    ]
    feature_sets = {
        "cycle_only": ["cycle"],
        "cycle_and_operating": ["cycle", *operating_summary_columns],
        "measured_sensor_only": sensor_summary_columns,
        "all_observable": ["cycle", *operating_summary_columns, *sensor_summary_columns],
        # Compatibility names for the original two-input runner below. New
        # analyses should use the four explicit feature sets above.
        "physical_sensor_summaries": sensor_summary_columns,
        "operating_condition_summaries": operating_summary_columns,
    }
    for name, columns in feature_sets.items():
        values = cycle_table[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values found in {name}.")
    for auxiliary in AUXILIARY_COLUMNS:
        if auxiliary in cycle_table.columns or any(
            column.startswith(f"{auxiliary}__")
            for columns in feature_sets.values()
            for column in columns
        ):
            raise RuntimeError(f"Auxiliary field leaked into feature table: {auxiliary}")
    return cycle_table, feature_sets


def make_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def select_threshold_at_false_alert_budget(
    target: np.ndarray,
    probability: np.ndarray,
    false_alert_budget: float,
) -> float:
    """Choose the lowest threshold whose negative-class alert rate meets the budget."""
    negative_probability = probability[target == 0]
    if len(negative_probability) == 0:
        raise ValueError("Threshold selection requires pre-window cycles.")
    values, counts = np.unique(negative_probability, return_counts=True)
    allowed = false_alert_budget * len(negative_probability)
    cumulative = 0
    threshold = float(np.nextafter(values[-1], np.inf))
    for value, count in zip(values[::-1], counts[::-1]):
        if cumulative + int(count) <= allowed + 1e-12:
            cumulative += int(count)
            threshold = float(value)
        else:
            break
    return threshold


def metric_record(target: np.ndarray, probability: np.ndarray, alert: np.ndarray) -> dict[str, float]:
    """Compute cycle-level classification metrics for a fixed alert decision."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, alert, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(target, alert, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(target, alert)),
        "balanced_accuracy": float(balanced_accuracy_score(target, alert)),
        "precision": float(precision),
        "final_window_recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "stable_region_false_alert_rate": float(fp / (fp + tn)),
        "late_window_miss_rate": float(fn / (fn + tp)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def first_alerts(cycle_table: pd.DataFrame, alert: np.ndarray, final_window_cycles: float) -> pd.DataFrame:
    """Return the first warning timing for each engine trajectory."""
    frame = cycle_table[["unit", "cycle", "rul"]].copy()
    frame["maintenance_warning"] = alert.astype(int)
    rows: list[dict[str, Any]] = []
    for unit, unit_rows in frame.groupby("unit", sort=True):
        unit_rows = unit_rows.sort_values("cycle")
        window_start = int(
            unit_rows.loc[unit_rows["rul"] <= final_window_cycles, "cycle"].min()
        )
        warnings = unit_rows.loc[unit_rows["maintenance_warning"] == 1, "cycle"]
        first_warning = int(warnings.min()) if not warnings.empty else None
        lead = None if first_warning is None else int(window_start - first_warning)
        rows.append(
            {
                "unit": int(unit),
                "final_window_start_cycle": window_start,
                "first_maintenance_warning_cycle": first_warning,
                "lead_cycles": lead,
                "warning_at_or_before_final_window": bool(lead is not None and lead >= 0),
            }
        )
    return pd.DataFrame(rows)


def clustered_intervals(
    cycle_table: pd.DataFrame,
    probability: np.ndarray,
    alert: np.ndarray,
    alerts: pd.DataFrame,
    final_window_cycles: float,
    repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    """Return percentile intervals after resampling complete engine trajectories."""
    target = (cycle_table["rul"].to_numpy(dtype=float) <= final_window_cycles).astype(int)
    units = cycle_table["unit"].to_numpy(dtype=int)
    unique_units = np.unique(units)
    positions = {unit: np.flatnonzero(units == unit) for unit in unique_units}
    alert_lookup = alerts.set_index("unit")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "precision": [],
        "final_window_recall": [],
        "stable_region_false_alert_rate": [],
        "auroc": [],
        "on_time_engine_warning_rate": [],
    }
    for _ in range(repeats):
        sampled_units = rng.choice(unique_units, size=len(unique_units), replace=True)
        sample_index = np.concatenate([positions[int(unit)] for unit in sampled_units])
        metric = metric_record(target[sample_index], probability[sample_index], alert[sample_index])
        values["precision"].append(metric["precision"])
        values["final_window_recall"].append(metric["final_window_recall"])
        values["stable_region_false_alert_rate"].append(
            metric["stable_region_false_alert_rate"]
        )
        values["auroc"].append(metric["auroc"])
        values["on_time_engine_warning_rate"].append(
            float(alert_lookup.loc[sampled_units, "warning_at_or_before_final_window"].mean())
        )
    return {
        key: [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]
        for key, sample in values.items()
    }


def paired_model_difference_intervals(
    cycle_table: pd.DataFrame,
    physical_probability: np.ndarray,
    physical_alert: np.ndarray,
    operating_probability: np.ndarray,
    operating_alert: np.ndarray,
    final_window_cycles: float,
    repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    """Return paired engine-cluster intervals for physical-minus-operating gains."""
    target = (cycle_table["rul"].to_numpy(dtype=float) <= final_window_cycles).astype(int)
    units = cycle_table["unit"].to_numpy(dtype=int)
    unique_units = np.unique(units)
    positions = {unit: np.flatnonzero(units == unit) for unit in unique_units}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "auroc_difference": [],
        "final_window_recall_difference": [],
        "stable_region_false_alert_rate_difference": [],
    }
    for _ in range(repeats):
        sampled_units = rng.choice(unique_units, size=len(unique_units), replace=True)
        sample_index = np.concatenate([positions[int(unit)] for unit in sampled_units])
        physical = metric_record(
            target[sample_index],
            physical_probability[sample_index],
            physical_alert[sample_index],
        )
        operating = metric_record(
            target[sample_index],
            operating_probability[sample_index],
            operating_alert[sample_index],
        )
        values["auroc_difference"].append(physical["auroc"] - operating["auroc"])
        values["final_window_recall_difference"].append(
            physical["final_window_recall"] - operating["final_window_recall"]
        )
        values["stable_region_false_alert_rate_difference"].append(
            physical["stable_region_false_alert_rate"]
            - operating["stable_region_false_alert_rate"]
        )
    return {
        key: [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]
        for key, sample in values.items()
    }


def nested_group_evaluation(
    cycle_table: pd.DataFrame,
    columns: list[str],
    final_window_cycles: float,
    false_alert_budget: float,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a warning policy with nested complete-engine holdouts."""
    X = cycle_table[columns].to_numpy(dtype=np.float32)
    target = (cycle_table["rul"].to_numpy(dtype=float) <= final_window_cycles).astype(int)
    groups = cycle_table["unit"].to_numpy(dtype=int)
    probability = np.zeros(len(cycle_table), dtype=float)
    threshold_by_row = np.zeros(len(cycle_table), dtype=float)
    fold_by_row = np.zeros(len(cycle_table), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    outer = GroupKFold(n_splits=outer_folds)
    for outer_fold, (outer_train, outer_test) in enumerate(
        outer.split(X, target, groups), start=1
    ):
        inner_X = X[outer_train]
        inner_target = target[outer_train]
        inner_groups = groups[outer_train]
        inner_probability = np.zeros(len(outer_train), dtype=float)
        inner = GroupKFold(n_splits=inner_folds)
        for inner_train, inner_test in inner.split(inner_X, inner_target, inner_groups):
            inner_model = make_model(seed)
            inner_model.fit(inner_X[inner_train], inner_target[inner_train])
            inner_probability[inner_test] = inner_model.predict_proba(
                inner_X[inner_test]
            )[:, 1]
        threshold = select_threshold_at_false_alert_budget(
            inner_target,
            inner_probability,
            false_alert_budget,
        )
        model = make_model(seed)
        model.fit(X[outer_train], target[outer_train])
        outer_probability = model.predict_proba(X[outer_test])[:, 1]
        outer_alert = outer_probability >= threshold
        probability[outer_test] = outer_probability
        threshold_by_row[outer_test] = threshold
        fold_by_row[outer_test] = outer_fold
        metrics = metric_record(target[outer_test], outer_probability, outer_alert)
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_threshold": threshold,
                "inner_oof_false_alert_rate": float(
                    (inner_probability[inner_target == 0] >= threshold).mean()
                ),
                "train_engines": int(len(np.unique(groups[outer_train]))),
                "test_engines": int(len(np.unique(groups[outer_test]))),
                "train_engine_cycles": int(len(outer_train)),
                "test_engine_cycles": int(len(outer_test)),
                **metrics,
            }
        )
    alert = probability >= threshold_by_row
    metrics = metric_record(target, probability, alert)
    prediction_rows = cycle_table[["unit", "cycle", "rul"]].copy()
    prediction_rows["outer_fold"] = fold_by_row
    prediction_rows["is_final_window"] = target
    prediction_rows["risk_probability"] = probability
    prediction_rows["selected_threshold"] = threshold_by_row
    prediction_rows["maintenance_warning"] = alert.astype(int)
    return {
        "pooled_metrics": metrics,
        "fold_metrics": fold_rows,
        "prediction_rows": prediction_rows,
    }


def write_outputs(
    output_dir: Path,
    results: dict[str, Any],
    physical_prediction_rows: pd.DataFrame,
    physical_alerts: pd.DataFrame,
    physical_fold_metrics: list[dict[str, Any]],
    operating_prediction_rows: pd.DataFrame,
    operating_fold_metrics: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    physical_prediction_rows.to_csv(
        output_dir / "physical_sensor_oof_predictions.csv", index=False
    )
    physical_alerts.to_csv(output_dir / "physical_sensor_engine_warnings.csv", index=False)
    pd.DataFrame(physical_fold_metrics).to_csv(
        output_dir / "physical_sensor_fold_metrics.csv", index=False
    )
    operating_prediction_rows.to_csv(
        output_dir / "operating_condition_oof_predictions.csv", index=False
    )
    pd.DataFrame(operating_fold_metrics).to_csv(
        output_dir / "operating_condition_fold_metrics.csv", index=False
    )

    physical = results["physical_sensor_warning"]
    operating = results["operating_condition_baseline"]
    physical_metrics = physical["pooled_metrics"]
    physical_intervals = physical["bootstrap_95pct_ci"]
    operating_intervals = operating["bootstrap_95pct_ci"]
    alert_summary = physical["engine_warning_summary"]
    lines = [
        "# Nested engine-held-out maintenance-window warning evaluation",
        "",
        "This is a separate simulation experiment, not the FD1 RUL-regression benchmark.",
        "Each outer test fold contains engines that are absent from model fitting and threshold selection.",
        "",
        "## Decision protocol",
        "",
        f"- Target: engine cycle with RUL <= {results['protocol']['final_window_cycles']:.0f}.",
        f"- Warning policy: threshold selected from inner engine-grouped out-of-fold predictions to target <= {results['protocol']['false_alert_budget']:.0%} pre-window false alerts.",
        f"- Outer evaluation: {results['protocol']['outer_folds']} folds with {results['protocol']['test_engines_per_outer_fold']} complete unseen engines per fold.",
        "- Primary input: per-cycle mean and standard deviation for 14 measured physical channels.",
        "- Excluded from the primary input: engine ID, cycle number, RUL, operating-condition fields, and the auxiliary Fc and hs fields.",
        "",
        "## Physical-sensor warning result",
        "",
        "| Metric | Value | Cluster bootstrap 95% interval |",
        "| --- | ---: | ---: |",
        f"| Out-of-fold AUROC | {physical_metrics['auroc']:.3f} | [{physical_intervals['auroc'][0]:.3f}, {physical_intervals['auroc'][1]:.3f}] |",
        f"| Final-window recall | {physical_metrics['final_window_recall']:.3f} | [{physical_intervals['final_window_recall'][0]:.3f}, {physical_intervals['final_window_recall'][1]:.3f}] |",
        f"| Stable-cycle false-alert rate | {physical_metrics['stable_region_false_alert_rate']:.3f} | [{physical_intervals['stable_region_false_alert_rate'][0]:.3f}, {physical_intervals['stable_region_false_alert_rate'][1]:.3f}] |",
        f"| Engines warned at or before final window | {alert_summary['on_time_engines']}/{alert_summary['engine_count']} ({alert_summary['on_time_rate']:.3f}) | [{physical_intervals['on_time_engine_warning_rate'][0]:.3f}, {physical_intervals['on_time_engine_warning_rate'][1]:.3f}] |",
        f"| Median first-warning lead among warned engines | {alert_summary['median_lead_cycles']:.1f} cycles | -- |",
        "",
        "## Operating-condition baseline",
        "",
        f"- AUROC: {operating['pooled_metrics']['auroc']:.3f} [{operating_intervals['auroc'][0]:.3f}, {operating_intervals['auroc'][1]:.3f}]",
        f"- Final-window recall: {operating['pooled_metrics']['final_window_recall']:.3f} [{operating_intervals['final_window_recall'][0]:.3f}, {operating_intervals['final_window_recall'][1]:.3f}]",
        f"- Stable-cycle false-alert rate: {operating['pooled_metrics']['stable_region_false_alert_rate']:.3f} [{operating_intervals['stable_region_false_alert_rate'][0]:.3f}, {operating_intervals['stable_region_false_alert_rate'][1]:.3f}]",
        f"- Physical-sensor AUROC gain: {results['physical_minus_operating']['auroc_difference']:.3f} [{results['physical_minus_operating']['bootstrap_95pct_ci']['auroc_difference'][0]:.3f}, {results['physical_minus_operating']['bootstrap_95pct_ci']['auroc_difference'][1]:.3f}]",
        f"- Physical-sensor recall gain: {results['physical_minus_operating']['final_window_recall_difference']:.3f} [{results['physical_minus_operating']['bootstrap_95pct_ci']['final_window_recall_difference'][0]:.3f}, {results['physical_minus_operating']['bootstrap_95pct_ci']['final_window_recall_difference'][1]:.3f}]",
        "",
        "A warning is a screening action for this simulated data only. It does not establish a maintenance schedule, a safe operational horizon, or performance on fielded aircraft engines.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--final-window-cycles", type=float, default=20.0)
    parser.add_argument("--false-alert-budget", type=float, default=0.05)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.final_window_cycles <= 0:
        raise ValueError("--final-window-cycles must be positive.")
    if not 0.0 < args.false_alert_budget < 1.0:
        raise ValueError("--false-alert-budget must be in (0, 1).")

    checksums = ensure_data(args.data_dir, args.download)
    cycle_table, feature_sets = build_cycle_table(args.data_dir / "train_df.pkl")
    engine_count = int(cycle_table["unit"].nunique())
    if args.outer_folds > engine_count:
        raise ValueError("--outer-folds exceeds engine count.")
    if args.inner_folds > engine_count - engine_count // args.outer_folds:
        raise ValueError("--inner-folds exceeds outer-training engine count.")
    print(
        f"Prepared {len(cycle_table)} engine-cycle rows from {engine_count} simulated engines."
    )
    physical = nested_group_evaluation(
        cycle_table,
        feature_sets["physical_sensor_summaries"],
        args.final_window_cycles,
        args.false_alert_budget,
        args.outer_folds,
        args.inner_folds,
        args.seed,
    )
    operating = nested_group_evaluation(
        cycle_table,
        feature_sets["operating_condition_summaries"],
        args.final_window_cycles,
        args.false_alert_budget,
        args.outer_folds,
        args.inner_folds,
        args.seed,
    )
    physical_alerts = first_alerts(
        cycle_table,
        physical["prediction_rows"]["maintenance_warning"].to_numpy(dtype=bool),
        args.final_window_cycles,
    )
    operating_alerts = first_alerts(
        cycle_table,
        operating["prediction_rows"]["maintenance_warning"].to_numpy(dtype=bool),
        args.final_window_cycles,
    )
    warned = physical_alerts.dropna(subset=["lead_cycles"])
    on_time_engines = int(
        physical_alerts["warning_at_or_before_final_window"].sum()
    )
    bootstrap = clustered_intervals(
        cycle_table,
        physical["prediction_rows"]["risk_probability"].to_numpy(dtype=float),
        physical["prediction_rows"]["maintenance_warning"].to_numpy(dtype=bool),
        physical_alerts,
        args.final_window_cycles,
        args.bootstrap_repeats,
        args.seed,
    )
    operating_bootstrap = clustered_intervals(
        cycle_table,
        operating["prediction_rows"]["risk_probability"].to_numpy(dtype=float),
        operating["prediction_rows"]["maintenance_warning"].to_numpy(dtype=bool),
        operating_alerts,
        args.final_window_cycles,
        args.bootstrap_repeats,
        args.seed,
    )
    paired_difference = paired_model_difference_intervals(
        cycle_table,
        physical["prediction_rows"]["risk_probability"].to_numpy(dtype=float),
        physical["prediction_rows"]["maintenance_warning"].to_numpy(dtype=bool),
        operating["prediction_rows"]["risk_probability"].to_numpy(dtype=float),
        operating["prediction_rows"]["maintenance_warning"].to_numpy(dtype=bool),
        args.final_window_cycles,
        args.bootstrap_repeats,
        args.seed,
    )
    physical_metrics = physical["pooled_metrics"]
    operating_metrics = operating["pooled_metrics"]
    results = {
        "source": {
            "dataset": "tiny-N-CMAPSS public challenge subset",
            "repository": SOURCE_REPOSITORY,
            "description": "100x-downsampled N-CMAPSS subset; separate from FD1 regression.",
            "files_sha256": checksums,
        },
        "dataset": {
            "simulated_engines": engine_count,
            "engine_cycles": int(len(cycle_table)),
            "positive_engine_cycles": int(
                (cycle_table["rul"] <= args.final_window_cycles).sum()
            ),
        },
        "protocol": {
            "task": "Warn when a simulated engine cycle is in the final RUL window.",
            "final_window_cycles": float(args.final_window_cycles),
            "false_alert_budget": float(args.false_alert_budget),
            "outer_folds": int(args.outer_folds),
            "inner_folds": int(args.inner_folds),
            "test_engines_per_outer_fold": int(engine_count // args.outer_folds),
            "threshold_selection": "Lowest inner out-of-fold probability threshold with pre-window false-alert rate at or below the fixed budget.",
            "model": {
                "family": "RandomForestClassifier",
                "n_estimators": 250,
                "max_depth": 12,
                "min_samples_leaf": 3,
                "class_weight": "balanced",
                "random_state": args.seed,
            },
            "physical_sensor_channels": len(MEASURED_SENSOR_CHANNELS),
            "operating_condition_channels": len(OPERATING_COLUMNS),
            "physical_input_excluded_columns": ["unit", "cycle", "RUL", *OPERATING_COLUMNS],
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "bootstrap_seed": int(args.seed),
        },
        "physical_sensor_warning": {
            "pooled_metrics": physical_metrics,
            "bootstrap_95pct_ci": bootstrap,
            "engine_warning_summary": {
                "engine_count": int(len(physical_alerts)),
                "on_time_engines": on_time_engines,
                "on_time_rate": float(on_time_engines / len(physical_alerts)),
                "eventually_warned_engines": int(len(warned)),
                "never_warned_engines": int(
                    physical_alerts["first_maintenance_warning_cycle"].isna().sum()
                ),
                "median_lead_cycles": float(warned["lead_cycles"].median()),
                "mean_lead_cycles": float(warned["lead_cycles"].mean()),
                "minimum_lead_cycles": int(warned["lead_cycles"].min()),
                "maximum_lead_cycles": int(warned["lead_cycles"].max()),
            },
            "selected_threshold_by_outer_fold": [
                {
                    "outer_fold": int(row["outer_fold"]),
                    "threshold": float(row["selected_threshold"]),
                    "inner_oof_false_alert_rate": float(row["inner_oof_false_alert_rate"]),
                }
                for row in physical["fold_metrics"]
            ],
        },
        "operating_condition_baseline": {
            "pooled_metrics": operating_metrics,
            "bootstrap_95pct_ci": operating_bootstrap,
            "selected_threshold_by_outer_fold": [
                {
                    "outer_fold": int(row["outer_fold"]),
                    "threshold": float(row["selected_threshold"]),
                    "inner_oof_false_alert_rate": float(row["inner_oof_false_alert_rate"]),
                }
                for row in operating["fold_metrics"]
            ],
        },
        "physical_minus_operating": {
            "auroc_difference": float(physical_metrics["auroc"] - operating_metrics["auroc"]),
            "final_window_recall_difference": float(
                physical_metrics["final_window_recall"]
                - operating_metrics["final_window_recall"]
            ),
            "stable_region_false_alert_rate_difference": float(
                physical_metrics["stable_region_false_alert_rate"]
                - operating_metrics["stable_region_false_alert_rate"]
            ),
            "bootstrap_95pct_ci": paired_difference,
        },
    }
    write_outputs(
        args.output_dir,
        results,
        physical["prediction_rows"],
        physical_alerts,
        physical["fold_metrics"],
        operating["prediction_rows"],
        operating["fold_metrics"],
    )
    print(f"Wrote result bundle to {args.output_dir}")
    print(
        "Physical-sensor warning: "
        f"AUROC={physical_metrics['auroc']:.3f}; "
        f"recall={physical_metrics['final_window_recall']:.3f}; "
        f"false-alert rate={physical_metrics['stable_region_false_alert_rate']:.3f}; "
        f"on-time engines={on_time_engines}/{len(physical_alerts)}."
    )
    print(
        "Operating-condition baseline: "
        f"AUROC={operating_metrics['auroc']:.3f}; "
        f"recall={operating_metrics['final_window_recall']:.3f}; "
        f"false-alert rate={operating_metrics['stable_region_false_alert_rate']:.3f}."
    )


if __name__ == "__main__":
    main()
