#!/usr/bin/env python3
"""Render the six IJSCAR manuscript tables from committed result bundles.

This script performs no model fitting. It converts the machine-readable JSON
outputs produced by the maintenance-warning evaluation scripts into Markdown
tables using the same rounding and row order as the accepted manuscript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY_RESULTS = (
    PROJECT_ROOT
    / "benchmark_results"
    / "maintenance_warning"
    / "corrected_observable_nested_20260812"
    / "results.json"
)
DEFAULT_SELECTED_RESULTS = (
    PROJECT_ROOT
    / "benchmark_results"
    / "maintenance_warning"
    / "model_selected_observable_20260814_robust"
    / "results.json"
)

VARIANT_ORDER = [
    "cycle_only",
    "cycle_and_operating",
    "measured_sensor_only",
    "all_observable",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def interval(value: float, bounds: list[float]) -> str:
    return f"{value:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]"


def mean_range(summary: dict[str, float]) -> str:
    return f"{summary['mean']:.3f} " f"[{summary['minimum']:.3f}, {summary['maximum']:.3f}]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render(primary: dict[str, Any], selected: dict[str, Any]) -> str:
    provenance = primary["feature_provenance"]
    table_1 = markdown_table(
        ["Group", "Count", "Fields and use"],
        [
            [
                "Measured physical (`Xs`)",
                str(len(provenance["measured_sensor_channels"])),
                ", ".join(f"`{name}`" for name in provenance["measured_sensor_channels"])
                + "; included in telemetry variants",
            ],
            [
                "Operating conditions (`W`)",
                str(len(provenance["operating_condition_columns"])),
                ", ".join(f"`{name}`" for name in provenance["operating_condition_columns"])
                + "; used in cycle-plus-operating and all-observable inputs",
            ],
            ["Elapsed cycle", "1", "`cycle`; included only in variants that name it"],
            [
                "Auxiliary fields",
                str(len(provenance["auxiliary_columns_excluded_from_all_models"])),
                ", ".join(
                    f"`{name}`" for name in provenance["auxiliary_columns_excluded_from_all_models"]
                )
                + "; excluded from every model",
            ],
            [
                "Identifier and label",
                str(len(provenance["other_excluded_columns"])),
                ", ".join(f"`{name}`" for name in provenance["other_excluded_columns"])
                + "; excluded from every model",
            ],
        ],
    )

    input_rows: list[list[str]] = []
    for name in VARIANT_ORDER:
        variant = primary["variants"][name]
        metrics = variant["pooled_metrics"]
        bounds = variant["bootstrap_95pct_ci"]
        input_rows.append(
            [
                variant["label"],
                interval(metrics["auroc"], bounds["auroc"]),
                interval(metrics["precision"], bounds["precision"]),
                interval(metrics["final_window_recall"], bounds["final_window_recall"]),
                interval(
                    metrics["stable_region_false_alert_rate"],
                    bounds["stable_region_false_alert_rate"],
                ),
            ]
        )
    table_2 = markdown_table(
        ["Input", "AUROC", "Precision", "Final-window recall", "Pre-window false-alert rate"],
        input_rows,
    )

    fixed = primary["variants"]["all_observable"]
    fixed_metrics = fixed["pooled_metrics"]
    fixed_bounds = fixed["bootstrap_95pct_ci"]
    selected_policy = selected["nested_model_selection"]
    selected_metrics = selected_policy["pooled_metrics"]
    selected_bounds = selected_policy["bootstrap_95pct_ci"]
    table_3 = markdown_table(
        ["Policy", "AUROC", "Precision", "Final-window recall", "Pre-window false-alert rate"],
        [
            [
                "Fixed random forest",
                interval(fixed_metrics["auroc"], fixed_bounds["auroc"]),
                interval(fixed_metrics["precision"], fixed_bounds["precision"]),
                interval(
                    fixed_metrics["final_window_recall"],
                    fixed_bounds["final_window_recall"],
                ),
                interval(
                    fixed_metrics["stable_region_false_alert_rate"],
                    fixed_bounds["stable_region_false_alert_rate"],
                ),
            ],
            [
                "Nested selected policy",
                interval(selected_metrics["auroc"], selected_bounds["auroc"]),
                interval(selected_metrics["precision"], selected_bounds["precision"]),
                interval(
                    selected_metrics["final_window_recall"],
                    selected_bounds["final_window_recall"],
                ),
                interval(
                    selected_metrics["stable_region_false_alert_rate"],
                    selected_bounds["stable_region_false_alert_rate"],
                ),
            ],
        ],
    )

    fixed_threshold_rows: list[list[str]] = []
    for name in VARIANT_ORDER:
        variant = primary["variants"][name]
        metrics = variant["fixed_threshold_0_5_metrics"]
        fixed_threshold_rows.append(
            [
                variant["label"],
                f"{metrics['precision']:.3f}",
                f"{metrics['final_window_recall']:.3f}",
                f"{metrics['stable_region_false_alert_rate']:.3f}",
            ]
        )
    table_4 = markdown_table(
        ["Input", "Precision", "Final-window recall", "Pre-window false-alert rate"],
        fixed_threshold_rows,
    )

    def warning_row(label: str, summary: dict[str, Any]) -> list[str]:
        engine_count = int(summary["engine_count"])
        on_time = int(summary["on_time_engines"])
        eventually = int(summary["eventually_warned_engines"])
        never = int(summary["never_warned_engines"])
        median_lead = float(summary["median_lead_cycles"])
        cycle_word = "cycle" if abs(median_lead) == 1 else "cycles"
        return [
            label,
            f"{on_time}/{engine_count} ({100 * summary['on_time_rate']:.1f}%)",
            f"{eventually - on_time}/{engine_count}",
            f"{never}/{engine_count}",
            f"{median_lead:.0f} {cycle_word}",
        ]

    table_5 = markdown_table(
        ["Input", "By window start", "Late warning", "No warning", "Median lead among warned"],
        [
            warning_row(
                "Measured telemetry only",
                primary["variants"]["measured_sensor_only"]["engine_warning_summary"],
            ),
            warning_row(
                "All observable, fixed random forest",
                primary["variants"]["all_observable"]["engine_warning_summary"],
            ),
            warning_row(
                "All observable, nested selected policy",
                selected_policy["engine_warning_summary"],
            ),
        ],
    )

    seed_rows: list[list[str]] = []
    for name in VARIANT_ORDER:
        label = primary["variants"][name]["label"]
        summary = primary["model_seed_robustness"][name]["summary"]
        seed_rows.append(
            [
                label,
                mean_range(summary["auroc"]),
                mean_range(summary["final_window_recall"]),
                mean_range(summary["stable_region_false_alert_rate"]),
            ]
        )
    selected_summary = selected["seed_robustness"]["summary"]
    seed_rows.append(
        [
            "Nested selected policy",
            mean_range(selected_summary["auroc"]),
            mean_range(selected_summary["final_window_recall"]),
            mean_range(selected_summary["stable_region_false_alert_rate"]),
        ]
    )
    table_6 = markdown_table(
        ["Input", "AUROC", "Final-window recall", "Pre-window false-alert rate"],
        seed_rows,
    )

    tables = [table_1, table_2, table_3, table_4, table_5, table_6]
    return "\n\n".join(f"## Table {index}\n\n{table}" for index, table in enumerate(tables, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-results", type=Path, default=DEFAULT_PRIMARY_RESULTS)
    parser.add_argument("--selected-results", type=Path, default=DEFAULT_SELECTED_RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = render(load_json(args.primary_results), load_json(args.selected_results)) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
