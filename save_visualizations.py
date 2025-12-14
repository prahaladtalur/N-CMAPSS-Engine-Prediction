#!/usr/bin/env python3
"""Save all data visualizations to files."""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.data.load_data import get_datasets
from src.utils.visualize import (
    plot_rul_distribution,
    plot_sensor_time_series,
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap,
    plot_multi_sensor_lifecycle,
)

# Create output directory
os.makedirs('visualizations', exist_ok=True)

# Load data
print("Loading data...")
(dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)

# 1. RUL Distribution
print("Saving RUL distribution...")
plot_rul_distribution(dev_y, split_name="Development Set")
plt.savefig('visualizations/01_rul_distribution_dev.png', dpi=300, bbox_inches='tight')
plt.close()

plot_rul_distribution(test_y, split_name="Test Set")
plt.savefig('visualizations/02_rul_distribution_test.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Sensor Time Series
print("Saving sensor time series...")
plot_sensor_time_series(dev_X, dev_y, unit_idx=0, num_sensors=4)
plt.savefig('visualizations/03_sensor_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Sensor Degradation
print("Saving sensor degradation...")
plot_sensor_degradation(dev_X, dev_y, unit_idx=0)
plt.savefig('visualizations/04_sensor_degradation.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlation Heatmap
print("Saving correlation heatmap...")
plot_sensor_correlation_heatmap(dev_X, dev_y)
plt.savefig('visualizations/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Multi-Sensor Lifecycle
print("Saving multi-sensor lifecycle...")
plot_multi_sensor_lifecycle(dev_X, dev_y, unit_idx=0)
plt.savefig('visualizations/06_multi_sensor_lifecycle.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ All visualizations saved to 'visualizations/' directory")
