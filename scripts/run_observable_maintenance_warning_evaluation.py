#!/usr/bin/env python3
"""Evaluate maintenance-window warnings with correctly scoped N-CMAPSS inputs.

The experiment compares four information sets under the same nested,
engine-grouped protocol:

* elapsed cycle only;
* elapsed cycle plus operating-condition summaries;
* the 14 measured physical channels only; and
* all observable inputs above.

The N-CMAPSS auxiliary fields Fc (flight class) and hs (health state) are
validated for provenance but are never used as model inputs. The main result
therefore answers whether measured telemetry adds warning information beyond
elapsed cycle and recorded operating conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_maintenance_warning_evaluation import (
    AUXILIARY_COLUMNS,
    MEASURED_SENSOR_CHANNELS,
    OPERATING_COLUMNS,
    SOURCE_REPOSITORY,
    build_cycle_table,
    clustered_intervals,
    ensure_data,
    first_alerts,
    metric_record,
    nested_group_evaluation,
    paired_model_difference_intervals,
)

SOURCE_COMMIT = "b915997ff8d571f4e9d091954d6556835f212ded"
PROJECT_REPOSITORY = "https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction"

VARIANTS: dict[str, dict[str, str]] = {
    "cycle_only": {
        "label": "Elapsed cycle only",
        "description": "Elapsed flight-cycle number only.",
    },
    "cycle_and_operating": {
        "label": "Cycle plus operating conditions",
        "description": "Elapsed cycle plus per-cycle altitude, Mach, throttle resolver angle, and inlet-temperature summaries.",
    },
    "measured_sensor_only": {
        "label": "Measured telemetry only",
        "description": "Per-cycle mean and standard deviation of the 14 measured N-CMAPSS physical channels; no elapsed cycle or operating-condition input.",
    },
    "all_observable": {
        "label": "All observable inputs",
        "description": "Elapsed cycle, operating-condition summaries, and summaries of the 14 measured physical channels.",
    },
}


def json_dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def numeric_summary(
    rows: list[dict[str, float]], metric_names: list[str]
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


def engine_summary(alerts: pd.DataFrame) -> dict[str, float | int]:
    warned = alerts.dropna(subset=["lead_cycles"])
    on_time = int(alerts["warning_at_or_before_final_window"].sum())
    lead_values = warned["lead_cycles"].to_numpy(dtype=float)
    return {
        "engine_count": int(len(alerts)),
        "on_time_engines": on_time,
        "on_time_rate": float(on_time / len(alerts)),
        "eventually_warned_engines": int(len(warned)),
        "never_warned_engines": int(alerts["first_maintenance_warning_cycle"].isna().sum()),
        "median_lead_cycles": float(np.median(lead_values)) if len(lead_values) else float("nan"),
        "mean_lead_cycles": float(np.mean(lead_values)) if len(lead_values) else float("nan"),
        "minimum_lead_cycles": int(np.min(lead_values)) if len(lead_values) else 0,
        "maximum_lead_cycles": int(np.max(lead_values)) if len(lead_values) else 0,
    }


def evaluate_variant(
    name: str,
    cycle_table: pd.DataFrame,
    columns: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one nested out-of-fold warning evaluation and its primary CIs."""
    evaluation = nested_group_evaluation(
        cycle_table,
        columns,
        args.final_window_cycles,
        args.false_alert_budget,
        args.outer_folds,
        args.inner_folds,
        args.primary_seed,
    )
    predictions = evaluation["prediction_rows"]
    alert = predictions["maintenance_warning"].to_numpy(dtype=bool)
    probability = predictions["risk_probability"].to_numpy(dtype=float)
    target = predictions["is_final_window"].to_numpy(dtype=int)
    alerts = first_alerts(cycle_table, alert, args.final_window_cycles)
    bootstrap = clustered_intervals(
        cycle_table,
        probability,
        alert,
        alerts,
        args.final_window_cycles,
        args.bootstrap_repeats,
        args.primary_seed,
    )
    fixed_metrics = metric_record(target, probability, probability >= 0.5)
    record = {
        "label": VARIANTS[name]["label"],
        "description": VARIANTS[name]["description"],
        "feature_count": int(len(columns)),
        "feature_columns": columns,
        "pooled_metrics": evaluation["pooled_metrics"],
        "fixed_threshold_0_5_metrics": fixed_metrics,
        "bootstrap_95pct_ci": bootstrap,
        "engine_warning_summary": engine_summary(alerts),
        "selected_threshold_by_outer_fold": [
            {
                "outer_fold": int(row["outer_fold"]),
                "threshold": float(row["selected_threshold"]),
                "inner_oof_false_alert_rate": float(row["inner_oof_false_alert_rate"]),
            }
            for row in evaluation["fold_metrics"]
        ],
    }
    print(
        f"{name}: AUROC={record['pooled_metrics']['auroc']:.3f}; "
        f"precision={record['pooled_metrics']['precision']:.3f}; "
        f"recall={record['pooled_metrics']['final_window_recall']:.3f}; "
        f"FPR={record['pooled_metrics']['stable_region_false_alert_rate']:.3f}."
    )
    return {
        "record": record,
        "predictions": predictions,
        "alerts": alerts,
        "fold_metrics": evaluation["fold_metrics"],
        "probability": probability,
        "alert": alert,
    }


def seed_robustness(
    cycle_table: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    primary_runs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Repeat the full nested procedure across RF seeds on fixed outer folds."""
    metric_names = [
        "auroc",
        "precision",
        "final_window_recall",
        "stable_region_false_alert_rate",
    ]
    output: dict[str, Any] = {}
    for name in VARIANTS:
        rows: list[dict[str, float]] = []
        for seed in args.robustness_seeds:
            if seed == args.primary_seed:
                metrics = primary_runs[name]["record"]["pooled_metrics"]
            else:
                evaluation = nested_group_evaluation(
                    cycle_table,
                    feature_sets[name],
                    args.final_window_cycles,
                    args.false_alert_budget,
                    args.outer_folds,
                    args.inner_folds,
                    seed,
                )
                metrics = evaluation["pooled_metrics"]
            rows.append(
                {"seed": int(seed), **{metric: float(metrics[metric]) for metric in metric_names}}
            )
        output[name] = {
            "metric_by_seed": rows,
            "summary": numeric_summary(rows, metric_names),
        }
    return output


def paired_gain(
    cycle_table: pd.DataFrame,
    higher: dict[str, Any],
    lower: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Compute an out-of-fold paired engine-cluster interval for a comparison."""
    higher_metrics = higher["record"]["pooled_metrics"]
    lower_metrics = lower["record"]["pooled_metrics"]
    intervals = paired_model_difference_intervals(
        cycle_table,
        higher["probability"],
        higher["alert"],
        lower["probability"],
        lower["alert"],
        args.final_window_cycles,
        args.bootstrap_repeats,
        args.primary_seed,
    )
    return {
        "auroc_difference": float(higher_metrics["auroc"] - lower_metrics["auroc"]),
        "precision_difference": float(higher_metrics["precision"] - lower_metrics["precision"]),
        "final_window_recall_difference": float(
            higher_metrics["final_window_recall"] - lower_metrics["final_window_recall"]
        ),
        "stable_region_false_alert_rate_difference": float(
            higher_metrics["stable_region_false_alert_rate"]
            - lower_metrics["stable_region_false_alert_rate"]
        ),
        "bootstrap_95pct_ci": intervals,
    }


def write_outputs(
    output_dir: Path,
    results: dict[str, Any],
    runs: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(output_dir / "results.json", results)
    for name, run in runs.items():
        run["predictions"].to_csv(output_dir / f"{name}_oof_predictions.csv", index=False)
        run["alerts"].to_csv(output_dir / f"{name}_engine_warnings.csv", index=False)
        pd.DataFrame(run["fold_metrics"]).to_csv(
            output_dir / f"{name}_fold_metrics.csv", index=False
        )

    comparison = results["comparisons"]["all_observable_minus_cycle_and_operating"]
    full = results["variants"]["all_observable"]
    baseline = results["variants"]["cycle_and_operating"]
    lines = [
        "# Corrected N-CMAPSS maintenance-window warning evaluation",
        "",
        "The former 16-channel setup was retired because it included Fc and hs, which are auxiliary N-CMAPSS fields. This result bundle uses only the 14 measured physical channels, separately named elapsed-cycle and operating-condition inputs, and no auxiliary health-state field.",
        "",
        "## Primary comparison",
        "",
        "| Input | AUROC | Precision | Recall | Pre-window false-alert rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in VARIANTS:
        metric = results["variants"][name]["pooled_metrics"]
        lines.append(
            f"| {results['variants'][name]['label']} | {metric['auroc']:.3f} | {metric['precision']:.3f} | {metric['final_window_recall']:.3f} | {metric['stable_region_false_alert_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The primary comparison is all observable inputs minus cycle plus operating conditions.",
            f"The AUROC difference is {comparison['auroc_difference']:.3f} [{comparison['bootstrap_95pct_ci']['auroc_difference'][0]:.3f}, {comparison['bootstrap_95pct_ci']['auroc_difference'][1]:.3f}].",
            f"The recall difference is {comparison['final_window_recall_difference']:.3f} [{comparison['bootstrap_95pct_ci']['final_window_recall_difference'][0]:.3f}, {comparison['bootstrap_95pct_ci']['final_window_recall_difference'][1]:.3f}].",
            "",
            "## Fixed 0.5 threshold check",
            "",
            f"All-observable precision={full['fixed_threshold_0_5_metrics']['precision']:.3f}, recall={full['fixed_threshold_0_5_metrics']['final_window_recall']:.3f}, and pre-window false-alert rate={full['fixed_threshold_0_5_metrics']['stable_region_false_alert_rate']:.3f}.",
            f"Cycle-plus-operating precision={baseline['fixed_threshold_0_5_metrics']['precision']:.3f}, recall={baseline['fixed_threshold_0_5_metrics']['final_window_recall']:.3f}, and pre-window false-alert rate={baseline['fixed_threshold_0_5_metrics']['stable_region_false_alert_rate']:.3f}.",
            "",
            "This is a simulation result only. It does not establish a maintenance schedule, a safe operational horizon, or fielded-aircraft performance.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one integer seed is required.")
    return sorted(set(seeds))


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
    parser.add_argument("--primary-seed", type=int, default=2026)
    parser.add_argument(
        "--robustness-seeds", type=parse_seeds, default=parse_seeds("2024,2025,2026,2027,2028")
    )
    args = parser.parse_args()
    if args.final_window_cycles <= 0:
        raise ValueError("--final-window-cycles must be positive.")
    if not 0 < args.false_alert_budget < 1:
        raise ValueError("--false-alert-budget must be in (0, 1).")
    if args.primary_seed not in args.robustness_seeds:
        args.robustness_seeds = sorted([*args.robustness_seeds, args.primary_seed])

    checksums = ensure_data(args.data_dir, args.download)
    cycle_table, feature_sets = build_cycle_table(args.data_dir / "train_df.pkl")
    engine_count = int(cycle_table["unit"].nunique())
    if args.outer_folds > engine_count:
        raise ValueError("--outer-folds exceeds engine count.")
    if args.inner_folds > engine_count - engine_count // args.outer_folds:
        raise ValueError("--inner-folds exceeds outer-training engine count.")
    if set(VARIANTS) - set(feature_sets):
        raise RuntimeError("One or more expected feature sets were not built.")

    print(f"Prepared {len(cycle_table)} engine-cycle rows from {engine_count} simulated engines.")
    runs: dict[str, dict[str, Any]] = {}
    for name in VARIANTS:
        runs[name] = evaluate_variant(name, cycle_table, feature_sets[name], args)

    first_prediction = runs["all_observable"]["predictions"]
    engine_fold_counts = (
        first_prediction.groupby("unit")["outer_fold"].nunique().to_numpy(dtype=int)
    )
    if not np.array_equal(engine_fold_counts, np.ones(engine_count, dtype=int)):
        raise RuntimeError("At least one engine was not held out in exactly one outer fold.")
    total_life = (
        cycle_table.assign(total_life=cycle_table["cycle"] + cycle_table["rul"])
        .groupby("unit", sort=True)["total_life"]
        .first()
    )
    if (
        int(
            cycle_table.assign(total_life=cycle_table["cycle"] + cycle_table["rul"])
            .groupby("unit")["total_life"]
            .nunique()
            .max()
        )
        != 1
    ):
        raise RuntimeError("RUL plus cycle is not constant within every engine.")
    raw_readings = int(len(pd.read_pickle(args.data_dir / "train_df.pkl")))
    target = (cycle_table["rul"].to_numpy(dtype=float) <= args.final_window_cycles).astype(int)
    robustness = seed_robustness(cycle_table, feature_sets, runs, args)
    comparisons = {
        "all_observable_minus_cycle_and_operating": paired_gain(
            cycle_table, runs["all_observable"], runs["cycle_and_operating"], args
        ),
        "measured_sensor_only_minus_cycle_only": paired_gain(
            cycle_table, runs["measured_sensor_only"], runs["cycle_only"], args
        ),
    }
    results = {
        "source": {
            "dataset": "tiny-N-CMAPSS public challenge subset",
            "repository": SOURCE_REPOSITORY,
            "repository_commit": SOURCE_COMMIT,
            "repository_commit_url": f"{SOURCE_REPOSITORY}/tree/{SOURCE_COMMIT}",
            "project_repository": PROJECT_REPOSITORY,
            "description": "100x-downsampled N-CMAPSS subset; separate from the FD1 RUL-regression benchmark.",
            "files_sha256": checksums,
        },
        "dataset": {
            "simulated_engines": engine_count,
            "raw_readings": raw_readings,
            "engine_cycles": int(len(cycle_table)),
            "mean_raw_readings_per_cycle": float(raw_readings / len(cycle_table)),
            "positive_engine_cycles": int(target.sum()),
            "positive_rate": float(target.mean()),
            "total_life_cycles": {
                "minimum": float(total_life.min()),
                "maximum": float(total_life.max()),
                "mean": float(total_life.mean()),
                "standard_deviation": float(total_life.std(ddof=1)),
            },
        },
        "feature_provenance": {
            "measured_sensor_source_group": "N-CMAPSS Xs measured physical properties",
            "measured_sensor_channels": MEASURED_SENSOR_CHANNELS,
            "operating_condition_columns": OPERATING_COLUMNS,
            "auxiliary_columns_excluded_from_all_models": AUXILIARY_COLUMNS,
            "other_excluded_columns": ["unit", "RUL"],
            "correction": "Fc and hs were present in an earlier 16-channel exploratory setup. They are auxiliary fields and are excluded from this corrected result bundle.",
        },
        "protocol": {
            "task": "Warn when a simulated engine cycle is in the final RUL window.",
            "final_window_cycles": float(args.final_window_cycles),
            "false_alert_budget": float(args.false_alert_budget),
            "outer_folds": int(args.outer_folds),
            "inner_folds": int(args.inner_folds),
            "outer_fold_coverage": "Each engine appears in exactly one outer test fold and in no model fit or threshold selection for that fold.",
            "threshold_selection": "Lowest inner out-of-fold probability threshold with pre-window false-alert rate at or below the fixed budget.",
            "fixed_threshold_check": 0.5,
            "model": {
                "family": "RandomForestClassifier",
                "n_estimators": 250,
                "max_depth": 12,
                "min_samples_leaf": 3,
                "class_weight": "balanced",
                "primary_random_state": int(args.primary_seed),
            },
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "bootstrap_seed": int(args.primary_seed),
            "robustness_seeds": [int(seed) for seed in args.robustness_seeds],
        },
        "variants": {name: run["record"] for name, run in runs.items()},
        "comparisons": comparisons,
        "model_seed_robustness": robustness,
    }
    write_outputs(args.output_dir, results, runs)
    print(f"Wrote corrected result bundle to {args.output_dir}")


if __name__ == "__main__":
    main()
