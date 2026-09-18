#!/usr/bin/env python3
"""Evaluate an engine-level final-RUL-window screening task on tiny-N-CMAPSS.

This is intentionally separate from the primary FD1 RUL-regression benchmark.
The public tiny-N-CMAPSS copy is a 100x downsampled subset of the 2021 PHM
Data Challenge data. This script builds one sensor-summary row per engine
flight cycle and holds out entire engines with five-fold GroupKFold.

The task is a maintenance-screening proxy, not an operational maintenance
policy: identify cycles with RUL <= 20 using physical sensor summaries only.
It excludes engine ID, flight-cycle number, and RUL from the model features.

Example:
  /Library/Developer/CommandLineTools/usr/bin/python3 \
      scripts/run_engine_warning_experiment.py \
      --data-dir /tmp/ncmapss_tiny \
      --output-dir benchmark_results/engine_warning/tiny_ncmapss_20260812
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_REPOSITORY = "https://github.com/alovberg/tiny-N-CMAPSS"
DATA_URLS = {
    "train_df.pkl": "https://raw.githubusercontent.com/alovberg/tiny-N-CMAPSS/main/data/train_df.pkl",
    "test_df.pkl": "https://raw.githubusercontent.com/alovberg/tiny-N-CMAPSS/main/data/test_df.pkl",
}
EXPECTED_SHA256 = {
    "train_df.pkl": "c091077acba202e7b2bfc30226aa177db32f6a5333f0a41d15a40a70a586b5a0",
    "test_df.pkl": "0bd442ed0ce2d088621f41028c182d43a6165553cd17f0e0a500a39f388ba944",
}
OPERATING_COLUMNS = ["alt", "Mach", "TRA", "T2"]
MEASURED_SENSOR_COLUMNS = [
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
AUXILIARY_COLUMNS = ["Fc", "hs"]
REQUIRED_COLUMNS = {
    "unit",
    "cycle",
    "RUL",
    *AUXILIARY_COLUMNS,
    *MEASURED_SENSOR_COLUMNS,
    *OPERATING_COLUMNS,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_data(data_dir: Path, allow_download: bool) -> dict[str, str]:
    """Check the two public pickle files and return their verified checksums."""
    data_dir.mkdir(parents=True, exist_ok=True)
    verified: dict[str, str] = {}
    for name, url in DATA_URLS.items():
        path = data_dir / name
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(
                    f"Missing {path}. Re-run with --download or place the verified "
                    f"file there. Source: {url}"
                )
            print(f"Downloading {name} from the documented public source...")
            urlretrieve(url, path)
        actual = sha256(path)
        expected = EXPECTED_SHA256[name]
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {path}. Expected {expected}, got {actual}."
            )
        verified[name] = actual
    return verified


def build_cycle_table(train_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate 100x-downsampled readings to a per-engine-cycle feature table."""
    frame = pd.read_pickle(train_path).copy()
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame["unit"] = frame["unit"].astype("int16")
    frame["cycle"] = frame["cycle"].astype("int16")
    sensor_columns = MEASURED_SENSOR_COLUMNS.copy()
    if any(column in sensor_columns for column in AUXILIARY_COLUMNS):
        raise RuntimeError("Auxiliary N-CMAPSS fields must not be model features.")

    # Cast before aggregation. The published tiny data are float16; averaging in
    # float32 prevents an overflow in group statistics for high-range channels.
    frame[sensor_columns] = frame[sensor_columns].astype("float32")
    grouped = frame.groupby(["unit", "cycle"], sort=True)
    labels = grouped["RUL"].agg(["first", "nunique"]).rename(
        columns={"first": "rul", "nunique": "label_count"}
    )
    if int(labels["label_count"].max()) != 1:
        raise ValueError("RUL labels are inconsistent within an engine cycle.")

    means = grouped[sensor_columns].mean().add_suffix("__mean")
    stds = grouped[sensor_columns].std(ddof=0).fillna(0.0).add_suffix("__std")
    cycle_table = pd.concat([means, stds, labels[["rul"]]], axis=1).reset_index()
    feature_columns = [*means.columns.tolist(), *stds.columns.tolist()]
    if not np.isfinite(cycle_table[feature_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite sensor summary values found after aggregation.")
    return cycle_table, feature_columns


def metric_record(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision),
        "final_window_recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "stable_region_false_alert_rate": float(fp / (fp + tn)),
        "late_window_miss_rate": float(fn / (fn + tp)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def engine_alerts(
    cycle_table: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    final_window_cycles: float,
) -> pd.DataFrame:
    """Record the first one-cycle screening flag for each held-out engine."""
    alerts = cycle_table[["unit", "cycle", "rul"]].copy()
    alerts["risk_probability"] = probabilities
    alerts["screen_alert"] = (probabilities >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for unit, unit_rows in alerts.groupby("unit", sort=True):
        unit_rows = unit_rows.sort_values("cycle")
        threshold_cycle = int(
            unit_rows.loc[unit_rows["rul"] <= final_window_cycles, "cycle"].min()
        )
        flags = unit_rows.loc[unit_rows["screen_alert"] == 1, "cycle"]
        first_alert = int(flags.min()) if not flags.empty else None
        lead = None if first_alert is None else int(threshold_cycle - first_alert)
        rows.append(
            {
                "unit": int(unit),
                "final_window_start_cycle": threshold_cycle,
                "first_screen_alert_cycle": first_alert,
                "lead_cycles": lead,
                "alert_at_or_before_final_window": bool(lead is not None and lead >= 0),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_by_engine(
    cycle_table: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    final_window_cycles: float,
    repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    """Return clustered 95% percentile intervals over held-out engine trajectories."""
    target = (
        cycle_table["rul"].to_numpy(dtype=float) <= final_window_cycles
    ).astype(int)
    units = cycle_table["unit"].to_numpy(dtype=int)
    unique_units = np.unique(units)
    positions = {unit: np.flatnonzero(units == unit) for unit in unique_units}
    alerts = engine_alerts(
        cycle_table,
        probabilities,
        threshold,
        final_window_cycles,
    ).set_index("unit")
    rng = np.random.default_rng(seed)
    results: dict[str, list[float]] = {
        "final_window_recall": [],
        "stable_region_false_alert_rate": [],
        "auroc": [],
        "on_time_engine_screen_rate": [],
    }
    for _ in range(repeats):
        sampled_units = rng.choice(unique_units, size=len(unique_units), replace=True)
        indices = np.concatenate([positions[int(unit)] for unit in sampled_units])
        metrics = metric_record(target[indices], probabilities[indices], threshold)
        results["final_window_recall"].append(metrics["final_window_recall"])
        results["stable_region_false_alert_rate"].append(
            metrics["stable_region_false_alert_rate"]
        )
        results["auroc"].append(metrics["auroc"])
        results["on_time_engine_screen_rate"].append(
            float(alerts.loc[sampled_units, "alert_at_or_before_final_window"].mean())
        )
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in results.items()
    }


def write_outputs(
    output_dir: Path,
    results: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    prediction_rows: pd.DataFrame,
    alerts: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    prediction_rows.to_csv(output_dir / "oof_cycle_predictions.csv", index=False)
    alerts.to_csv(output_dir / "engine_alerts.csv", index=False)

    pooled = results["pooled_metrics"]
    interval = results["bootstrap_95pct_ci"]
    alert_summary = results["engine_alert_summary"]
    lines = [
        "# Held-out engine final-window screening experiment",
        "",
        "This exploratory test is separate from the paper's primary FD1 RUL-regression benchmark.",
        "It uses the public, 100x-downsampled tiny-N-CMAPSS challenge subset and one row per engine cycle.",
        "Each GroupKFold test fold contains entirely unseen engines.",
        "",
        "## Protocol",
        "",
        f"- Simulated engine trajectories: {results['dataset']['engines']}",
        f"- Engine-cycle rows: {results['dataset']['engine_cycles']}",
        f"- Positive label: RUL <= {results['protocol']['final_window_cycles']} cycles",
        f"- Model inputs: {results['protocol']['sensor_channels']} physical sensors summarized by per-cycle mean and standard deviation",
        "- Excluded from model inputs: unit ID, flight-cycle number, RUL, and operating-condition columns.",
        f"- Evaluation: {results['protocol']['cv_folds']}-fold GroupKFold by engine; {results['protocol']['test_engines_per_fold']} held-out engines per fold.",
        f"- Fixed screening threshold: probability >= {results['protocol']['screening_threshold']:.1f}.",
        "",
        "## Pooled out-of-fold result",
        "",
        "| Metric | Value | Cluster bootstrap 95% interval |",
        "| --- | ---: | ---: |",
        f"| Final-window recall | {pooled['final_window_recall']:.3f} | [{interval['final_window_recall'][0]:.3f}, {interval['final_window_recall'][1]:.3f}] |",
        f"| Precision | {pooled['precision']:.3f} | -- |",
        f"| AUROC | {pooled['auroc']:.3f} | [{interval['auroc'][0]:.3f}, {interval['auroc'][1]:.3f}] |",
        f"| Stable-region false-alert rate | {pooled['stable_region_false_alert_rate']:.3f} | [{interval['stable_region_false_alert_rate'][0]:.3f}, {interval['stable_region_false_alert_rate'][1]:.3f}] |",
        f"| Engines first flagged at or before final window | {alert_summary['on_time_engines']}/{alert_summary['engine_count']} ({alert_summary['on_time_rate']:.3f}) | [{interval['on_time_engine_screen_rate'][0]:.3f}, {interval['on_time_engine_screen_rate'][1]:.3f}] |",
        f"| Median first-alert lead among flagged engines | {alert_summary['median_lead_cycles']:.1f} cycles | -- |",
        "",
        "A one-cycle flag is a preliminary screen, not a maintenance command. The result does not validate a safety policy or generalize beyond this downsampled simulation subset.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download verified public pickle files if absent.")
    parser.add_argument("--final-window-cycles", type=float, default=20.0)
    parser.add_argument("--screening-threshold", type=float, default=0.5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.final_window_cycles <= 0:
        raise ValueError("--final-window-cycles must be positive.")
    if not 0.0 < args.screening_threshold < 1.0:
        raise ValueError("--screening-threshold must be in (0, 1).")

    checksums = ensure_data(args.data_dir, args.download)
    cycle_table, feature_columns = build_cycle_table(args.data_dir / "train_df.pkl")
    X = cycle_table[feature_columns].to_numpy(dtype=np.float32)
    y = (cycle_table["rul"].to_numpy(dtype=float) <= args.final_window_cycles).astype(int)
    groups = cycle_table["unit"].to_numpy(dtype=int)
    unique_units = np.unique(groups)
    if args.cv_folds > len(unique_units):
        raise ValueError("Number of folds exceeds number of engines.")

    print(
        f"Prepared {len(cycle_table)} cycle summaries from {len(unique_units)} engines "
        f"with {len(feature_columns)} sensor-summary features."
    )
    probabilities = np.zeros(len(cycle_table), dtype=float)
    folds = np.zeros(len(cycle_table), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=args.cv_folds)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        fold_probability = model.predict_proba(X[test_idx])[:, 1]
        probabilities[test_idx] = fold_probability
        folds[test_idx] = fold
        metrics = metric_record(y[test_idx], fold_probability, args.screening_threshold)
        fold_rows.append(
            {
                "fold": fold,
                "train_engines": int(len(np.unique(groups[train_idx]))),
                "test_engines": int(len(np.unique(groups[test_idx]))),
                "train_engine_cycles": int(len(train_idx)),
                "test_engine_cycles": int(len(test_idx)),
                **metrics,
            }
        )
        print(
            f"Fold {fold}: AUROC={metrics['auroc']:.3f}, "
            f"final-window recall={metrics['final_window_recall']:.3f}, "
            f"false-alert rate={metrics['stable_region_false_alert_rate']:.3f}"
        )

    pooled = metric_record(y, probabilities, args.screening_threshold)
    alerts = engine_alerts(
        cycle_table,
        probabilities,
        args.screening_threshold,
        args.final_window_cycles,
    )
    flagged = alerts.dropna(subset=["lead_cycles"])
    on_time = int(alerts["alert_at_or_before_final_window"].sum())
    bootstrap = bootstrap_by_engine(
        cycle_table,
        probabilities,
        args.screening_threshold,
        args.final_window_cycles,
        args.bootstrap_repeats,
        args.seed,
    )
    results = {
        "source": {
            "dataset": "tiny-N-CMAPSS public challenge subset",
            "repository": SOURCE_REPOSITORY,
            "description": "100x-downsampled subset of N-CMAPSS; not equivalent to the primary FD1 benchmark.",
            "files_sha256": checksums,
        },
        "dataset": {
            "engines": int(len(unique_units)),
            "engine_cycles": int(len(cycle_table)),
            "raw_readings": int(len(pd.read_pickle(args.data_dir / "train_df.pkl"))),
            "positive_engine_cycles": int(y.sum()),
            "positive_rate": float(y.mean()),
        },
        "protocol": {
            "task": "Classify whether a simulated engine cycle is in the final RUL window.",
            "final_window_cycles": float(args.final_window_cycles),
            "sensor_channels": len(MEASURED_SENSOR_COLUMNS),
            "feature_summary": "Per-engine-cycle mean and standard deviation for each physical sensor.",
            "excluded_columns": ["unit", "cycle", "RUL", *OPERATING_COLUMNS],
            "model": {
                "family": "RandomForestClassifier",
                "n_estimators": 250,
                "max_depth": 12,
                "min_samples_leaf": 3,
                "class_weight": "balanced",
                "random_state": args.seed,
            },
            "cv_folds": int(args.cv_folds),
            "test_engines_per_fold": int(len(unique_units) // args.cv_folds),
            "screening_threshold": float(args.screening_threshold),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "bootstrap_seed": int(args.seed),
        },
        "pooled_metrics": pooled,
        "bootstrap_95pct_ci": bootstrap,
        "engine_alert_summary": {
            "engine_count": int(len(alerts)),
            "on_time_engines": on_time,
            "on_time_rate": float(on_time / len(alerts)),
            "eventually_flagged_engines": int(len(flagged)),
            "never_flagged_engines": int(alerts["first_screen_alert_cycle"].isna().sum()),
            "median_lead_cycles": float(flagged["lead_cycles"].median()),
            "mean_lead_cycles": float(flagged["lead_cycles"].mean()),
            "minimum_lead_cycles": int(flagged["lead_cycles"].min()),
            "maximum_lead_cycles": int(flagged["lead_cycles"].max()),
            "alert_definition": "First engine-cycle probability at or above the fixed screening threshold; a preliminary screen, not a maintenance command.",
        },
    }
    prediction_rows = cycle_table[["unit", "cycle", "rul"]].copy()
    prediction_rows["fold"] = folds
    prediction_rows["is_final_window"] = y
    prediction_rows["risk_probability"] = probabilities
    prediction_rows["screen_alert"] = (probabilities >= args.screening_threshold).astype(int)
    write_outputs(args.output_dir, results, fold_rows, prediction_rows, alerts)
    print(f"Wrote reproducible result bundle to {args.output_dir}")
    print(
        f"Pooled AUROC={pooled['auroc']:.3f}; final-window recall="
        f"{pooled['final_window_recall']:.3f}; on-time engine screens={on_time}/{len(alerts)}."
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "2026")
    main()
