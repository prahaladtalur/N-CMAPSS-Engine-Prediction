"""
Visualization utilities for N-CMAPSS RUL prediction.

This module provides two main categories of visualizations:
1. Data exploration (data_viz): Analyze sensor patterns and RUL distributions
2. Model evaluation (model_viz): Assess model predictions and training performance
"""

# Data exploration visualizations
from src.visualization.data_viz import (
    plot_rul_distribution,
    plot_sensor_time_series,
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap,
    plot_multi_sensor_lifecycle,
)

# Model evaluation visualizations
from src.visualization.model_viz import (
    plot_training_history,
    plot_predictions,
    plot_error_distribution,
    plot_rul_trajectory,
    plot_critical_zone_analysis,
    plot_model_comparison,
    plot_sample_predictions,
    create_evaluation_report,
)

__all__ = [
    # Data visualizations
    "plot_rul_distribution",
    "plot_sensor_time_series",
    "plot_sensor_degradation",
    "plot_sensor_correlation_heatmap",
    "plot_multi_sensor_lifecycle",
    # Model visualizations
    "plot_training_history",
    "plot_predictions",
    "plot_error_distribution",
    "plot_rul_trajectory",
    "plot_critical_zone_analysis",
    "plot_model_comparison",
    "plot_sample_predictions",
    "create_evaluation_report",
]
