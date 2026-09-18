#!/usr/bin/env python3
"""Select a maintenance-warning model without using held-out engines.

This runner is a follow-up to the observable-input maintenance experiment.  It
keeps the same complete-engine outer holdouts and warning rule, but chooses a
model family inside each outer training split.  The chosen model maximizes
inner out-of-fold final-window recall subject to the same 5% pre-window false
alert budget.  The outer-fold engines are never used to choose a model or an
alert threshold.

The candidate set is deliberately small and consists of a random-forest
reference plus three common non-neural tabular classifiers.  Fc and hs remain
excluded at every stage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_maintenance_warning_evaluation import (
    AUXILIARY_COLUMNS,
    MEASURED_SENSOR_CHANNELS,
    build_cycle_table,
    clustered_intervals,
    ensure_data,
    first_alerts,
    metric_record,
    nested_group_evaluation,
    paired_model_difference_intervals,
    select_threshold_at_false_alert_budget,
)


ModelFactory = Callable[[int], Any]


def candidate_factories() -> dict[str, ModelFactory]:
    """Return a small, predeclared candidate family for nested selection."""
    return {
        "random_forest_reference": lambda seed: RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": lambda seed: ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        ),
        "rbf_svc": lambda seed: make_pipeline(
            StandardScaler(),
            SVC(
                C=2.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=seed,
            ),
        ),
    }


def candidate_description() -> dict[str, dict[str, Any]]:
    """Store configurations with the output so the selection is reproducible."""
    return {
        "random_forest_reference": {
            "family": "RandomForestClassifier",
            "n_estimators": 250,
            "max_depth": 12,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
        },
        "extra_trees": {
            "family": "ExtraTreesClassifier",
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "max_features": 0.8,
            "class_weight": "balanced",
        },
        "hist_gradient_boosting": {
            "family": "HistGradientBoostingClassifier",
            "max_iter": 300,
            "learning_rate": 0.06,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
        },
        "rbf_svc": {
            "family": "SVC",
            "kernel": "rbf",
            "C": 2.0,
            "class_weight": "balanced",
            "standardized": True,
        },
    }


def select_model_from_inner_oof(
    target: np.ndarray,
    candidate_probability: dict[str, np.ndarray],
    false_alert_budget: float,
) -> tuple[str, float, list[dict[str, Any]]]:
    """Choose the inner-OOF model for recall under the alert-rate constraint.

    The selection order is intentionally specified before outer evaluation:
    final-window recall at the budgeted threshold, AUROC, average precision,
    and precision.  These all use only inner out-of-fold predictions.
    """
    records: list[dict[str, Any]] = []
    for name, probability in candidate_probability.items():
        threshold = select_threshold_at_false_alert_budget(
            target, probability, false_alert_budget
        )
        metrics = metric_record(target, probability, probability >= threshold)
        records.append(
            {
                "candidate": name,
                "selected_threshold": float(threshold),
                **{key: float(value) if isinstance(value, np.floating) else value for key, value in metrics.items()},
            }
        )
    winner = max(
        records,
        key=lambda record: (
            record["final_window_recall"],
            record["auroc"],
            record["average_precision"],
            record["precision"],
            -record["stable_region_false_alert_rate"],
        ),
    )
    return str(winner["candidate"]), float(winner["selected_threshold"]), records


def nested_selected_evaluation(
    cycle_table: pd.DataFrame,
    columns: list[str],
    final_window_cycles: float,
    false_alert_budget: float,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the model-selection policy on complete unseen engines."""
    X = cycle_table[columns].to_numpy(dtype=np.float32)
    target = (cycle_table["rul"].to_numpy(dtype=float) <= final_window_cycles).astype(int)
    groups = cycle_table["unit"].to_numpy(dtype=int)
    probability = np.zeros(len(cycle_table), dtype=float)
    threshold_by_row = np.zeros(len(cycle_table), dtype=float)
    fold_by_row = np.zeros(len(cycle_table), dtype=int)
    selected_by_row = np.empty(len(cycle_table), dtype=object)
    fold_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    factories = candidate_factories()
    outer = GroupKFold(n_splits=outer_folds)
    for outer_fold, (outer_train, outer_test) in enumerate(
        outer.split(X, target, groups), start=1
    ):
        inner_X = X[outer_train]
        inner_target = target[outer_train]
        inner_groups = groups[outer_train]
        inner_probability = {
            name: np.zeros(len(outer_train), dtype=float) for name in factories
        }
        inner = GroupKFold(n_splits=inner_folds)
        for inner_fold, (inner_train, inner_test) in enumerate(
            inner.split(inner_X, inner_target, inner_groups), start=1
        ):
            for candidate_index, (name, factory) in enumerate(factories.items(), start=1):
                model = factory(seed + 10_000 * outer_fold + 100 * inner_fold + candidate_index)
                model.fit(inner_X[inner_train], inner_target[inner_train])
                inner_probability[name][inner_test] = model.predict_proba(
                    inner_X[inner_test]
                )[:, 1]
        selected_model, threshold, selection_records = select_model_from_inner_oof(
            inner_target, inner_probability, false_alert_budget
        )
        for row in selection_records:
            inner_rows.append({"outer_fold": outer_fold, **row})
        final_model = factories[selected_model](seed + 1_000_000 + outer_fold)
        final_model.fit(X[outer_train], target[outer_train])
        outer_probability = final_model.predict_proba(X[outer_test])[:, 1]
        outer_alert = outer_probability >= threshold
        probability[outer_test] = outer_probability
        threshold_by_row[outer_test] = threshold
        fold_by_row[outer_test] = outer_fold
        selected_by_row[outer_test] = selected_model
        metrics = metric_record(target[outer_test], outer_probability, outer_alert)
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_model": selected_model,
                "selected_threshold": float(threshold),
                "inner_oof_false_alert_rate": float(
                    (inner_probability[selected_model][inner_target == 0] >= threshold).mean()
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
    prediction_rows["selected_model"] = selected_by_row
    prediction_rows["is_final_window"] = target
    prediction_rows["risk_probability"] = probability
    prediction_rows["selected_threshold"] = threshold_by_row
    prediction_rows["maintenance_warning"] = alert.astype(int)
    return {
        "pooled_metrics": metrics,
        "fold_metrics": fold_rows,
        "inner_selection_metrics": inner_rows,
        "selection_frequency": dict(Counter(str(row["selected_model"]) for row in fold_rows)),
        "prediction_rows": prediction_rows,
    }


def metrics_by_seed(
    cycle_table: pd.DataFrame,
    columns: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Repeat the entire nested selection policy under multiple RNG seeds."""
    rows: list[dict[str, Any]] = []
    for seed in args.robustness_seeds:
        evaluation = nested_selected_evaluation(
            cycle_table,
            columns,
            args.final_window_cycles,
            args.false_alert_budget,
            args.outer_folds,
            args.inner_folds,
            seed,
        )
        metric = evaluation["pooled_metrics"]
        rows.append(
            {
                "seed": int(seed),
                "auroc": float(metric["auroc"]),
                "precision": float(metric["precision"]),
                "final_window_recall": float(metric["final_window_recall"]),
                "stable_region_false_alert_rate": float(
                    metric["stable_region_false_alert_rate"]
                ),
                "selection_frequency": evaluation["selection_frequency"],
            }
        )
    return rows


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
    parser.add_argument(
        "--robustness-seeds",
        type=int,
        nargs="*",
        default=[2024, 2025, 2026, 2027, 2028],
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Run only the primary nested evaluation.",
    )
    args = parser.parse_args()
    if args.final_window_cycles <= 0:
        raise ValueError("--final-window-cycles must be positive.")
    if not 0.0 < args.false_alert_budget < 1.0:
        raise ValueError("--false-alert-budget must be in (0, 1).")

    checksums = ensure_data(args.data_dir, args.download)
    cycle_table, feature_sets = build_cycle_table(args.data_dir / "train_df.pkl")
    columns = feature_sets["all_observable"]
    if any(column.startswith(("Fc__", "hs__")) for column in columns):
        raise RuntimeError("An auxiliary N-CMAPSS field was included as an input.")
    primary = nested_selected_evaluation(
        cycle_table,
        columns,
        args.final_window_cycles,
        args.false_alert_budget,
        args.outer_folds,
        args.inner_folds,
        args.seed,
    )
    reference = nested_group_evaluation(
        cycle_table,
        columns,
        args.final_window_cycles,
        args.false_alert_budget,
        args.outer_folds,
        args.inner_folds,
        args.seed,
    )
    alert = primary["prediction_rows"]["maintenance_warning"].to_numpy(dtype=bool)
    probability = primary["prediction_rows"]["risk_probability"].to_numpy(dtype=float)
    alerts = first_alerts(cycle_table, alert, args.final_window_cycles)
    bootstrap = clustered_intervals(
        cycle_table,
        probability,
        alert,
        alerts,
        args.final_window_cycles,
        args.bootstrap_repeats,
        args.seed,
    )
    comparison = paired_model_difference_intervals(
        cycle_table,
        probability,
        alert,
        reference["prediction_rows"]["risk_probability"].to_numpy(dtype=float),
        reference["prediction_rows"]["maintenance_warning"].to_numpy(dtype=bool),
        args.final_window_cycles,
        args.bootstrap_repeats,
        args.seed,
    )
    robustness = [] if args.skip_robustness else metrics_by_seed(cycle_table, columns, args)
    summary: dict[str, dict[str, float]] = {}
    if robustness:
        for metric_name in [
            "auroc",
            "precision",
            "final_window_recall",
            "stable_region_false_alert_rate",
        ]:
            values = np.asarray([row[metric_name] for row in robustness], dtype=float)
            summary[metric_name] = {
                "mean": float(values.mean()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    results = {
        "source": {
            "dataset": "tiny-N-CMAPSS public challenge subset",
            "files_sha256": checksums,
            "simulated_engines": int(cycle_table["unit"].nunique()),
            "engine_cycles": int(len(cycle_table)),
        },
        "protocol": {
            "target": f"RUL <= {args.final_window_cycles:.0f} cycles",
            "input_set": "all observable: elapsed cycle, operating summaries, and 14 measured physical-channel summaries",
            "excluded_auxiliary_columns": AUXILIARY_COLUMNS,
            "measured_sensor_channels": MEASURED_SENSOR_CHANNELS,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "false_alert_budget": args.false_alert_budget,
            "model_selection": "Within each outer training split, choose the candidate with highest inner out-of-fold final-window recall at the budgeted threshold; break ties by AUROC, average precision, precision, then lower false-alert rate.",
            "candidate_models": candidate_description(),
        },
        "nested_model_selection": {
            "pooled_metrics": primary["pooled_metrics"],
            "bootstrap_95pct_ci": bootstrap,
            "engine_warning_summary": {
                "engine_count": int(len(alerts)),
                "on_time_engines": int(alerts["warning_at_or_before_final_window"].sum()),
                "on_time_rate": float(alerts["warning_at_or_before_final_window"].mean()),
                "eventually_warned_engines": int(alerts["first_maintenance_warning_cycle"].notna().sum()),
                "never_warned_engines": int(alerts["first_maintenance_warning_cycle"].isna().sum()),
                "median_lead_cycles": float(alerts.dropna(subset=["lead_cycles"])["lead_cycles"].median()),
            },
            "selection_frequency": primary["selection_frequency"],
            "fold_metrics": primary["fold_metrics"],
            "inner_selection_metrics": primary["inner_selection_metrics"],
        },
        "fixed_random_forest_reference": {
            "pooled_metrics": reference["pooled_metrics"],
            "fold_metrics": reference["fold_metrics"],
        },
        "selected_policy_minus_fixed_random_forest": {
            "auroc_difference": float(primary["pooled_metrics"]["auroc"] - reference["pooled_metrics"]["auroc"]),
            "precision_difference": float(primary["pooled_metrics"]["precision"] - reference["pooled_metrics"]["precision"]),
            "final_window_recall_difference": float(primary["pooled_metrics"]["final_window_recall"] - reference["pooled_metrics"]["final_window_recall"]),
            "stable_region_false_alert_rate_difference": float(primary["pooled_metrics"]["stable_region_false_alert_rate"] - reference["pooled_metrics"]["stable_region_false_alert_rate"]),
            "bootstrap_95pct_ci": comparison,
        },
        "seed_robustness": {"metric_by_seed": robustness, "summary": summary},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    primary["prediction_rows"].to_csv(
        args.output_dir / "selected_policy_oof_predictions.csv", index=False
    )
    pd.DataFrame(primary["fold_metrics"]).to_csv(
        args.output_dir / "selected_policy_fold_metrics.csv", index=False
    )
    pd.DataFrame(primary["inner_selection_metrics"]).to_csv(
        args.output_dir / "inner_selection_metrics.csv", index=False
    )
    print(
        "Selected policy: "
        f"AUROC={primary['pooled_metrics']['auroc']:.6f}; "
        f"precision={primary['pooled_metrics']['precision']:.6f}; "
        f"recall={primary['pooled_metrics']['final_window_recall']:.6f}; "
        f"FPR={primary['pooled_metrics']['stable_region_false_alert_rate']:.6f}"
    )
    print(f"Selected models by outer fold: {primary['selection_frequency']}")
    print(
        "Difference versus fixed RF: "
        f"AUROC={results['selected_policy_minus_fixed_random_forest']['auroc_difference']:.6f}; "
        f"recall={results['selected_policy_minus_fixed_random_forest']['final_window_recall_difference']:.6f}"
    )


if __name__ == "__main__":
    main()
