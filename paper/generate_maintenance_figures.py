#!/usr/bin/env python3
"""Generate figures from the corrected observable-input warning evaluation."""

from __future__ import annotations

import json
from csv import DictReader
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas


PAPER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PAPER_DIR.parent
RESULT_DIR = (
    PROJECT_DIR
    / "benchmark_results"
    / "maintenance_warning"
    / "corrected_observable_nested_20260812"
)
RESULTS_PATH = RESULT_DIR / "results.json"
SENSOR_WARNINGS_PATH = RESULT_DIR / "measured_sensor_only_engine_warnings.csv"
MODEL_SELECTION_RESULT_DIR = (
    PROJECT_DIR
    / "benchmark_results"
    / "maintenance_warning"
    / "model_selected_observable_20260814_robust"
)
MODEL_SELECTION_RESULTS_PATH = MODEL_SELECTION_RESULT_DIR / "results.json"
FIGURE_DIR = PAPER_DIR / "figures"

WIDTH = 720
HEIGHT = 350
LEFT = 74
RIGHT = 38
BOTTOM = 72
TOP = 72
INK = colors.HexColor("#182535")
MUTED = colors.HexColor("#526273")
GRID = colors.HexColor("#D9E1E8")
PANEL = colors.HexColor("#F6F8FA")
BLUE = colors.HexColor("#2D6A9F")
TEAL = colors.HexColor("#207A6D")
ORANGE = colors.HexColor("#C7692D")
PURPLE = colors.HexColor("#7157A6")
GRAY = colors.HexColor("#697786")


def base_canvas(path: Path, title: str, subtitle: str) -> canvas.Canvas:
    """Create a clean scientific-figure canvas.

    The title and subtitle are retained as PDF metadata only. The manuscript
    caption carries that information, avoiding redundant dashboard-style
    headings inside the figure itself.
    """
    drawing = canvas.Canvas(str(path), pagesize=(WIDTH, HEIGHT))
    drawing.setTitle(title)
    drawing.setSubject(subtitle)
    return drawing


def footer(drawing: canvas.Canvas, value: str) -> None:
    # Values are stated in the manuscript body and captions. Keep the plotting
    # area free of dashboard-style summary text.
    del drawing, value


def y_position(value: float, y_min: float, y_max: float, y_bottom: float, height: float) -> float:
    return y_bottom + (value - y_min) / (y_max - y_min) * height


def draw_axis(
    drawing: canvas.Canvas,
    x: float,
    width: float,
    y_bottom: float,
    height: float,
    y_min: float,
    y_max: float,
    ticks: list[float],
    y_label: str,
) -> None:
    drawing.setFillColor(PANEL)
    drawing.rect(x, y_bottom, width, height, stroke=0, fill=1)
    drawing.setStrokeColor(GRID)
    drawing.setLineWidth(0.7)
    drawing.setFillColor(MUTED)
    drawing.setFont("Helvetica", 7)
    for tick in ticks:
        y = y_position(tick, y_min, y_max, y_bottom, height)
        drawing.line(x, y, x + width, y)
        drawing.drawRightString(x - 6, y - 2.5, f"{tick:.2f}")
    drawing.setStrokeColor(INK)
    drawing.line(x, y_bottom, x, y_bottom + height)
    drawing.line(x, y_bottom, x + width, y_bottom)
    drawing.saveState()
    drawing.translate(x - 38, y_bottom + height / 2)
    drawing.rotate(90)
    drawing.setFillColor(INK)
    drawing.setFont("Helvetica-Bold", 8)
    drawing.drawCentredString(0, 0, y_label)
    drawing.restoreState()


def draw_metric_panel(
    drawing: canvas.Canvas,
    x: float,
    width: float,
    heading: str,
    metric_name: str,
    results: dict,
    y_min: float,
    y_max: float,
    ticks: list[float],
) -> None:
    y_bottom = 75
    height = 240
    draw_axis(drawing, x, width, y_bottom, height, y_min, y_max, ticks, heading)
    variants = [
        ("cycle_only", ["Cycle"]),
        ("cycle_and_operating", ["Cycle +", "operating"]),
        ("measured_sensor_only", ["Measured", "telemetry"]),
        ("all_observable", ["All", "observable"]),
    ]
    palette = [BLUE, ORANGE, TEAL, PURPLE]
    spacing = width / len(variants)
    bar_width = min(32, spacing * 0.48)
    for index, ((name, labels), color) in enumerate(zip(variants, palette)):
        record = results["variants"][name]
        value = record["pooled_metrics"][metric_name]
        interval = record["bootstrap_95pct_ci"][metric_name]
        center = x + spacing * (index + 0.5)
        top = y_position(value, y_min, y_max, y_bottom, height)
        drawing.setFillColor(color)
        drawing.rect(center - bar_width / 2, y_bottom, bar_width, top - y_bottom, stroke=0, fill=1)
        drawing.setStrokeColor(INK)
        drawing.setLineWidth(0.9)
        lower = y_position(interval[0], y_min, y_max, y_bottom, height)
        upper = y_position(interval[1], y_min, y_max, y_bottom, height)
        drawing.line(center, lower, center, upper)
        drawing.line(center - 3, lower, center + 3, lower)
        drawing.line(center - 3, upper, center + 3, upper)
        drawing.setFillColor(INK)
        drawing.setFont("Helvetica-Bold", 7.4)
        drawing.drawCentredString(center, min(top + 7, y_bottom + height - 5), f"{value:.3f}")
        drawing.setFillColor(INK)
        drawing.setFont("Helvetica", 7.1)
        for line_index, label in enumerate(labels):
            drawing.drawCentredString(center, y_bottom - 16 - 9 * line_index, label)


def telemetry_benchmark() -> None:
    with RESULTS_PATH.open() as stream:
        results = json.load(stream)
    output = FIGURE_DIR / "maintenance_telemetry_benchmark.pdf"
    drawing = base_canvas(
        output,
        "Measured telemetry improves final-window maintenance warnings",
        "Nested engine-held-out evaluation. Thresholds target a 5% pre-window false-alert budget.",
    )
    gap = 52
    panel_width = (WIDTH - LEFT - RIGHT - gap) / 2
    draw_metric_panel(
        drawing,
        LEFT,
        panel_width,
        "AUROC",
        "auroc",
        results,
        0.85,
        1.00,
        [0.85, 0.90, 0.95, 1.00],
    )
    draw_metric_panel(
        drawing,
        LEFT + panel_width + gap,
        panel_width,
        "Final-window recall",
        "final_window_recall",
        results,
        0.0,
        1.0,
        [0.0, 0.25, 0.50, 0.75, 1.0],
    )
    comparison = results["comparisons"]["all_observable_minus_cycle_and_operating"]
    footer(
        drawing,
        f"Adding measured telemetry to cycle plus operating conditions: +{comparison['auroc_difference']:.3f} AUROC and +{comparison['final_window_recall_difference']:.3f} recall. Bars show 95% engine-cluster bootstrap intervals.",
    )
    drawing.save()


def draw_policy_metric_panel(
    drawing: canvas.Canvas,
    x: float,
    width: float,
    heading: str,
    metric_name: str,
    values: list[tuple[list[str], float, list[float], colors.Color]],
    y_min: float,
    y_max: float,
    ticks: list[float],
) -> None:
    y_bottom = 75
    height = 240
    draw_axis(drawing, x, width, y_bottom, height, y_min, y_max, ticks, heading)
    spacing = width / len(values)
    bar_width = min(43, spacing * 0.48)
    for index, (labels, value, interval, color) in enumerate(values):
        center = x + spacing * (index + 0.5)
        top = y_position(value, y_min, y_max, y_bottom, height)
        drawing.setFillColor(color)
        drawing.rect(center - bar_width / 2, y_bottom, bar_width, top - y_bottom, stroke=0, fill=1)
        drawing.setStrokeColor(INK)
        drawing.setLineWidth(0.9)
        lower = y_position(interval[0], y_min, y_max, y_bottom, height)
        upper = y_position(interval[1], y_min, y_max, y_bottom, height)
        drawing.line(center, lower, center, upper)
        drawing.line(center - 3, lower, center + 3, lower)
        drawing.line(center - 3, upper, center + 3, upper)
        drawing.setFillColor(INK)
        drawing.setFont("Helvetica-Bold", 7.4)
        drawing.drawCentredString(center, min(top + 7, y_bottom + height - 5), f"{value:.3f}")
        drawing.setFont("Helvetica", 7.1)
        for line_index, label in enumerate(labels):
            drawing.drawCentredString(center, y_bottom - 16 - 9 * line_index, label)


def nested_model_selection_benchmark() -> None:
    with RESULTS_PATH.open() as stream:
        input_results = json.load(stream)
    with MODEL_SELECTION_RESULTS_PATH.open() as stream:
        selection_results = json.load(stream)
    output = FIGURE_DIR / "maintenance_nested_model_selection.pdf"
    drawing = base_canvas(
        output,
        "Nested model selection improves the all-observable warning policy",
        "Model family and threshold are selected inside each outer training split; held-out engines remain unseen.",
    )
    fixed = input_results["variants"]["all_observable"]
    selected = selection_results["nested_model_selection"]
    fixed_metrics = fixed["pooled_metrics"]
    selected_metrics = selected["pooled_metrics"]
    fixed_interval = fixed["bootstrap_95pct_ci"]
    selected_interval = selected["bootstrap_95pct_ci"]
    values = {
        "auroc": [
            (["Fixed", "forest"], fixed_metrics["auroc"], fixed_interval["auroc"], PURPLE),
            (["Nested", "selected"], selected_metrics["auroc"], selected_interval["auroc"], TEAL),
        ],
        "final_window_recall": [
            (["Fixed", "forest"], fixed_metrics["final_window_recall"], fixed_interval["final_window_recall"], PURPLE),
            (["Nested", "selected"], selected_metrics["final_window_recall"], selected_interval["final_window_recall"], TEAL),
        ],
    }
    gap = 52
    panel_width = (WIDTH - LEFT - RIGHT - gap) / 2
    draw_policy_metric_panel(
        drawing,
        LEFT,
        panel_width,
        "AUROC",
        "auroc",
        values["auroc"],
        0.94,
        1.00,
        [0.94, 0.96, 0.98, 1.00],
    )
    draw_policy_metric_panel(
        drawing,
        LEFT + panel_width + gap,
        panel_width,
        "Final-window recall",
        "final_window_recall",
        values["final_window_recall"],
        0.65,
        0.90,
        [0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    difference = selection_results["selected_policy_minus_fixed_random_forest"]
    footer(
        drawing,
        f"Paired complete-engine comparison: +{difference['auroc_difference']:.3f} AUROC and +{difference['final_window_recall_difference']:.3f} recall. Bars show 95% complete-engine bootstrap intervals.",
    )
    drawing.save()


def sensor_warning_timing() -> None:
    with RESULTS_PATH.open() as stream:
        results = json.load(stream)
    with SENSOR_WARNINGS_PATH.open(newline="") as stream:
        rows = list(DictReader(stream))
    output = FIGURE_DIR / "maintenance_sensor_warning_timing.pdf"
    drawing = base_canvas(
        output,
        "Sensor-only warnings on held-out engine trajectories",
        "14 measured channels only. Positive lead means the first warning precedes RUL <= 20.",
    )
    summary = results["variants"]["measured_sensor_only"]["engine_warning_summary"]
    leads = [float(row["lead_cycles"]) for row in rows if row["lead_cycles"] != ""]
    y_min, y_max = -20.0, 60.0
    y_bottom, height = 75.0, 240.0
    plot_width = WIDTH - LEFT - RIGHT
    draw_axis(
        drawing,
        LEFT,
        plot_width,
        y_bottom,
        height,
        y_min,
        y_max,
        list(range(-20, 61, 10)),
        "Lead to final-window start (cycles)",
    )
    zero = y_position(0, y_min, y_max, y_bottom, height)
    drawing.setStrokeColor(INK)
    drawing.setLineWidth(0.8)
    drawing.setDash(3, 2)
    drawing.line(LEFT, zero, LEFT + plot_width, zero)
    drawing.setDash()
    drawing.setFillColor(MUTED)
    drawing.setFont("Helvetica", 7.2)
    drawing.drawRightString(LEFT + plot_width - 5, zero + 5, "0 = warning at final-window start")
    ranked = sorted(leads, reverse=True)
    for index, lead in enumerate(ranked, start=1):
        x = LEFT + plot_width * (index - 0.5) / len(rows)
        drawing.setFillColor(TEAL if lead >= 0 else ORANGE)
        drawing.circle(x, y_position(lead, y_min, y_max, y_bottom, height), 2.55, stroke=0, fill=1)
    no_warning_count = len(rows) - len(ranked)
    for index in range(len(ranked) + 1, len(rows) + 1):
        x = LEFT + plot_width * (index - 0.5) / len(rows)
        y = y_position(y_min + 2, y_min, y_max, y_bottom, height)
        drawing.setStrokeColor(GRAY)
        drawing.setLineWidth(1.25)
        drawing.line(x - 3, y - 3, x + 3, y + 3)
        drawing.line(x - 3, y + 3, x + 3, y - 3)
    drawing.setFillColor(INK)
    drawing.setFont("Helvetica", 7.3)
    for fraction, label in [(0, "1"), (0.5, "45"), (1, "90")]:
        drawing.drawCentredString(LEFT + plot_width * fraction, y_bottom - 16, label)
    drawing.setFont("Helvetica-Bold", 8)
    drawing.drawCentredString(WIDTH / 2, y_bottom - 34, "Engine trajectories ranked by first warning")
    legend_y = y_bottom + height - 17
    drawing.setFillColor(TEAL)
    drawing.circle(LEFT + 13, legend_y + 2, 3.2, stroke=0, fill=1)
    drawing.setFillColor(INK)
    drawing.setFont("Helvetica", 7.7)
    drawing.drawString(LEFT + 21, legend_y, f"At or before window ({summary['on_time_engines']})")
    drawing.setFillColor(ORANGE)
    drawing.circle(LEFT + 172, legend_y + 2, 3.2, stroke=0, fill=1)
    drawing.setFillColor(INK)
    drawing.drawString(LEFT + 180, legend_y, f"Late warning ({summary['eventually_warned_engines'] - summary['on_time_engines']})")
    cross_x = LEFT + 288
    drawing.setStrokeColor(GRAY)
    drawing.setLineWidth(1.2)
    drawing.line(cross_x - 3, legend_y - 1, cross_x + 3, legend_y + 5)
    drawing.line(cross_x - 3, legend_y + 5, cross_x + 3, legend_y - 1)
    drawing.setFillColor(INK)
    drawing.drawString(cross_x + 8, legend_y, f"No warning ({no_warning_count})")
    footer(
        drawing,
        f"Sensor-only policy: {summary['on_time_engines']}/{summary['engine_count']} engines warned by the final-window start; median lead {summary['median_lead_cycles']:.0f} cycles among {summary['eventually_warned_engines']} warned engines.",
    )
    drawing.save()


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    telemetry_benchmark()
    nested_model_selection_benchmark()
    sensor_warning_timing()
    print(FIGURE_DIR / "maintenance_telemetry_benchmark.pdf")
    print(FIGURE_DIR / "maintenance_nested_model_selection.pdf")
    print(FIGURE_DIR / "maintenance_sensor_warning_timing.pdf")


if __name__ == "__main__":
    main()
