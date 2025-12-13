# Visualization Guide for RUL Prediction

This guide covers all visualization capabilities for the N-CMAPSS engine RUL prediction project.

## Overview

The visualization toolkit is divided into two main categories:

1. **Data Analysis Visualizations** - Understand sensor patterns, degradation, and correlations
2. **Model Evaluation Visualizations** - Assess model performance, especially in critical zones

## Installation

Install all required dependencies:

```bash
uv sync
```

Required dependencies for visualization functions:
- `seaborn>=0.12.0`
- `scipy>=1.9.0`

## Quick Start

```python
from src.data.load_data import get_datasets
from src.utils import (
    plot_sensor_degradation,
    plot_sensor_correlation_heatmap,
    plot_rul_trajectory,
    plot_critical_zone_analysis,
)

# Load data
(dev_X, dev_y), val, (test_X, test_y) = get_datasets(fd=1)

# Analyze sensor degradation
plot_sensor_degradation(dev_X, dev_y, unit_idx=0)

# Identify sensors correlated with RUL
plot_sensor_correlation_heatmap(dev_X, dev_y)
```

Or run the example script:

```bash
python scripts/example_visualizations.py
```

---

## Data Analysis Visualizations

### 1. Sensor Degradation Analysis

**Purpose:** Visualize how sensor values change as engines degrade (RUL decreases).

**Use Case:** Identify sensors that show clear degradation patterns over the engine lifecycle.

```python
from src.utils import plot_sensor_degradation

plot_sensor_degradation(
    features=dev_X,
    labels=dev_y,
    unit_idx=0,                    # Which engine to analyze
    sensor_indices=[0, 1, 2, 3, 4, 5]  # Which sensors to plot
)
```

**What it shows:**
- Scatter plots of sensor values vs. RUL for multiple sensors
- Trend lines (polynomial fit) showing degradation patterns
- Color gradient indicating time progression
- Individual subplots for each sensor

**Key Insights:**
- Sensors with strong trends are good predictors
- Identifies monotonic vs. non-monotonic degradation patterns
- Reveals which sensors are most informative

---

### 2. Sensor Correlation Heatmap

**Purpose:** Identify sensors most correlated with RUL and with each other.

**Use Case:** Feature selection, understanding sensor relationships, and identifying redundant sensors.

```python
from src.utils import plot_sensor_correlation_heatmap

plot_sensor_correlation_heatmap(
    features=dev_X,
    labels=dev_y,
    max_sensors=14,      # Limit number of sensors
    sample_size=1000     # Samples for correlation analysis
)
```

**What it shows:**
- Full correlation matrix between all sensors and RUL
- Annotated correlation coefficients
- Color-coded heatmap (red=positive, blue=negative correlation)
- Top 5 sensors most correlated with RUL

**Key Insights:**
- Which sensors are most predictive of failure
- Sensor redundancy (high inter-sensor correlation)
- Feature selection guidance

---

### 3. Multi-Sensor Lifecycle Comparison

**Purpose:** Compare multiple sensors side-by-side over the full engine lifecycle.

**Use Case:** Visual comparison of degradation patterns across sensors.

```python
from src.utils import plot_multi_sensor_lifecycle

plot_multi_sensor_lifecycle(
    features=dev_X,
    labels=dev_y,
    unit_idx=0,        # Which engine
    max_sensors=8      # Number of sensors to compare
)
```

**What it shows:**
- Normalized sensor values (0-1 scale) for direct comparison
- All sensors plotted on the same axes
- RUL overlay (normalized) for reference
- Full lifecycle from healthy to failure

**Key Insights:**
- Which sensors degrade monotonically vs. erratically
- Relative timing of sensor changes
- Overall degradation patterns

---

## Model Evaluation Visualizations

### 4. RUL Trajectory Analysis

**Purpose:** Visualize predicted vs. actual RUL over a specific engine's lifecycle.

**Use Case:** Understand model performance for individual engines and identify where predictions fail.

```python
from src.utils import plot_rul_trajectory

plot_rul_trajectory(
    y_true=y_true,
    y_pred=y_pred,
    unit_length=[len(y) for y in test_y],  # Cycle counts per unit
    unit_idx=0  # Which engine to visualize
)
```

**What it shows:**
- Top plot: Predicted vs. actual RUL over time
- Critical zone shading (RUL < 30 = red, 30-75 = orange)
- Bottom plot: Prediction error over time
- Over-prediction (red) vs. under-prediction (blue) shading

**Key Insights:**
- When predictions become accurate
- Over-prediction vs. under-prediction tendencies
- Error patterns over the lifecycle
- Performance in critical zones

---

### 5. Critical Zone Analysis

**Purpose:** Analyze model performance when engines are close to failure.

**Use Case:** Most important metric for predictive maintenance—how well does the model predict imminent failure?

```python
from src.utils import plot_critical_zone_analysis

plot_critical_zone_analysis(
    y_true=y_true,
    y_pred=y_pred,
    critical_threshold=30,    # RUL < 30 = critical
    warning_threshold=75      # 30 ≤ RUL < 75 = warning
)
```

**What it shows:**
- **Error distribution by zone** (boxplots)
- **Accuracy by zone** (±10, ±20, ±30 cycle tolerance)
- **Sample distribution** (pie chart)
- **Predictions colored by zone** (scatter plot)

**Key Insights:**
- Model performance where it matters most (critical zone)
- Whether the model is more accurate in certain RUL ranges
- Sample imbalance across zones
- Detailed statistics (MAE, RMSE) per zone

---

### 6. Prediction Confidence Analysis

**Purpose:** Visualize prediction uncertainty and confidence.

**Use Case:** Understand where the model is confident vs. uncertain, and calibrate risk assessment.

```python
from src.utils import plot_prediction_confidence

# Option 1: Error-based confidence (single model)
plot_prediction_confidence(y_true=y_true, y_pred=y_pred)

# Option 2: Ensemble-based confidence (multiple predictions)
plot_prediction_confidence(
    y_true=y_true,
    y_pred=y_pred_mean,
    model_predictions=[pred1, pred2, pred3, ...]  # Ensemble predictions
)
```

**What it shows:**
- **Predictions with confidence intervals**
- **Error vs. uncertainty** (scatter plot)
- **Error distribution by RUL range** (boxplots)
- **Calibration plot** (error vs. predicted RUL)

**Key Insights:**
- Where the model is uncertain
- Correlation between uncertainty and error
- Model calibration quality
- Risk assessment for maintenance decisions

---

## Complete Analysis Workflow

Here's a recommended workflow for comprehensive analysis:

### Phase 1: Data Understanding

```python
# 1. Check sensor-RUL correlations
plot_sensor_correlation_heatmap(dev_X, dev_y)

# 2. Analyze degradation patterns
plot_sensor_degradation(dev_X, dev_y, unit_idx=0)
plot_sensor_degradation(dev_X, dev_y, unit_idx=1)  # Multiple units

# 3. Compare sensor behaviors
plot_multi_sensor_lifecycle(dev_X, dev_y, unit_idx=0)
```

### Phase 2: Model Evaluation (After Training)

```python
# 1. Overall performance
from src.utils.training_viz import plot_predictions, plot_error_distribution
plot_predictions(y_true, y_pred)
plot_error_distribution(y_true, y_pred)

# 2. Critical zone performance (MOST IMPORTANT)
plot_critical_zone_analysis(y_true, y_pred)

# 3. Individual engine analysis
unit_lengths = [len(y) for y in test_y]
for unit_idx in range(min(5, len(test_y))):
    plot_rul_trajectory(y_true, y_pred, unit_length=unit_lengths, unit_idx=unit_idx)

# 4. Confidence analysis
plot_prediction_confidence(y_true, y_pred)
```

---

## Tips for Effective Analysis

### For Data Analysis

1. **Always check multiple units** - Different engines may show different patterns
2. **Look for monotonic degradation** - Sensors with clear trends are most useful
3. **Use correlation heatmap for feature selection** - Remove redundant sensors
4. **Compare sensor behaviors** - Some sensors lead, others lag

### For Model Evaluation

1. **Prioritize critical zone performance** - This is where predictions matter most
2. **Check multiple engines** - Don't rely on aggregate metrics alone
3. **Look for systematic errors** - Over-prediction vs. under-prediction patterns
4. **Validate confidence** - Ensure uncertainty correlates with actual errors

### Common Insights

- **Early lifecycle**: Predictions often less accurate (high RUL, less degradation signal)
- **Late lifecycle**: More accurate (strong degradation signals)
- **Critical zone**: Most important for maintenance scheduling
- **Sensor selection**: 4-6 well-chosen sensors often outperform all sensors

---

## Integration with Training

The visualizations integrate with the training pipeline in `src/utils/training_viz.py`:

```python
from src.utils.training_viz import create_evaluation_report

# Generate complete evaluation report
create_evaluation_report(
    model_name="LSTM-v2",
    metrics=metrics_dict,
    y_true=y_true,
    y_pred=y_pred,
    history=training_history,
    save_dir="results/lstm_v2"
)

# Then add custom visualizations
from src.utils import plot_critical_zone_analysis
plot_critical_zone_analysis(y_true, y_pred)
```

---

## Saving Figures

All visualization functions support matplotlib's standard saving:

```python
import matplotlib.pyplot as plt

# Create visualization
plot_sensor_degradation(dev_X, dev_y, unit_idx=0)

# Save before showing
plt.savefig('results/sensor_degradation.png', dpi=300, bbox_inches='tight')
plt.show()
```

Or use the built-in save functionality in `training_viz.py` functions:

```python
from src.utils.training_viz import plot_predictions

plot_predictions(y_true, y_pred, save_path='results/predictions.png')
```

---

## Troubleshooting

**Q: Visualizations are too cluttered**
- Reduce `max_sensors` parameter
- Use `sensor_indices` to select specific sensors
- Increase `sample_size` parameter for smoother plots

**Q: Memory issues with large datasets**
- Use `sample_size` parameter to limit data points
- Visualize one unit at a time instead of all units
- Save figures incrementally instead of showing all

**Q: Can't see degradation patterns**
- Try different `unit_idx` values—some engines show clearer patterns
- Check if sensors are normalized/scaled properly
- Use `plot_multi_sensor_lifecycle` with normalization

---

## Example Output Interpretation

### Sensor Degradation Plot
- **Strong downward trend** = Sensor degrades predictably (good for RUL prediction)
- **Flat trend** = Sensor doesn't change with degradation (not useful)
- **Erratic pattern** = Noisy sensor or non-monotonic behavior

### Correlation Heatmap
- **|r| > 0.7** = Strong correlation (very useful sensors)
- **|r| < 0.3** = Weak correlation (consider removing)
- **High inter-sensor correlation** = Redundancy (can reduce features)

### Critical Zone Analysis
- **Lower error in critical zone** = Good! Model is accurate when it matters
- **Higher error in critical zone** = Concerning—needs improvement
- **Accuracy ±10 cycles > 80%** = Good performance for maintenance planning

---

## References

- Original N-CMAPSS Dataset: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- PHM Score: Standard metric for RUL prediction competitions
- Critical Zone Concept: Industry standard for maintenance decision-making

---

## Contributing

To add new visualizations:

1. Add function to `src/utils/visualize.py` (for data/model analysis) or `src/utils/training_viz.py` (for training-specific)
2. Export the function in `src/utils/__init__.py`
3. Add example to `example_visualizations.py`
4. Document in this guide

---

**For questions or issues, please refer to the main README.md**
