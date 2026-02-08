"""
Example: Real-world RUL prediction workflow

Demonstrates how to use the trained CNN-GRU model for engine health monitoring.
Requires model trained via: python train_production_model.py
"""

import numpy as np
from predict import RULPredictor
from src.data.load_data import get_datasets

MODEL_PATH = "models/production/cnn_gru_best.keras"


def example_1_single_prediction():
    """Predict RUL for a single engine's most recent reading."""
    print("=" * 70)
    print("Example 1: Single Engine Prediction")
    print("=" * 70)

    predictor = RULPredictor(model_path=MODEL_PATH)
    _, _, (test_X, test_y) = get_datasets(fd=1)

    unit_id = 2
    # Last cycle = closest to failure (most critical reading)
    last_cycle = test_X[unit_id][-1]
    true_rul = test_y[unit_id][-1]
    predicted_rul = predictor.predict_single(last_cycle)

    print(f"\nEngine Unit #{unit_id} — Latest Reading")
    print(f"  Sensor data:     {last_cycle.shape[0]} timesteps x {last_cycle.shape[1]} sensors")
    print(f"  True RUL:        {true_rul:.2f} cycles")
    print(f"  Predicted RUL:   {predicted_rul:.2f} cycles")
    print(f"  Error:           {abs(predicted_rul - true_rul):.2f} cycles")

    if predicted_rul < 20:
        status = "CRITICAL — Schedule maintenance NOW"
    elif predicted_rul < 50:
        status = "WARNING  — Plan maintenance soon"
    else:
        status = "HEALTHY  — Continue monitoring"
    print(f"  Status:          {status}")
    print()


def example_2_fleet_monitoring():
    """Monitor a fleet: predict current RUL for each engine."""
    print("=" * 70)
    print("Example 2: Fleet Monitoring")
    print("=" * 70)

    predictor = RULPredictor(model_path=MODEL_PATH)
    _, _, (test_X, test_y) = get_datasets(fd=1)

    print(f"\nFleet Status ({len(test_X)} engines):")
    print("-" * 65)
    print(f"{'Engine':<10} {'Cycles':<10} {'True RUL':<12} {'Pred RUL':<12} {'Status'}")
    print("-" * 65)

    critical, warning, healthy = 0, 0, 0
    for unit_id in range(len(test_X)):
        # Latest cycle per unit
        pred_rul = predictor.predict_single(test_X[unit_id][-1])
        true_rul = test_y[unit_id][-1]
        num_cycles = test_X[unit_id].shape[0]

        if pred_rul < 20:
            status = "CRITICAL"
            critical += 1
        elif pred_rul < 50:
            status = "WARNING"
            warning += 1
        else:
            status = "HEALTHY"
            healthy += 1

        print(f"{unit_id:<10} {num_cycles:<10} {true_rul:<12.2f} {pred_rul:<12.2f} {status}")

    print("-" * 65)
    print(f"  Critical: {critical} | Warning: {warning} | Healthy: {healthy}")
    print()


def example_3_maintenance_priority():
    """Rank engines by urgency for maintenance scheduling."""
    print("=" * 70)
    print("Example 3: Maintenance Priority Queue")
    print("=" * 70)

    predictor = RULPredictor(model_path=MODEL_PATH)
    _, _, (test_X, test_y) = get_datasets(fd=1)

    # Get latest-cycle predictions for all engines
    predictions = []
    for unit_id in range(len(test_X)):
        pred = predictor.predict_single(test_X[unit_id][-1])
        predictions.append((unit_id, pred))

    # Sort by predicted RUL (most urgent first)
    predictions.sort(key=lambda x: x[1])

    cycles_per_day = 10  # assume 10 flight cycles per day
    print(f"\nMaintenance Priority (assuming {cycles_per_day} cycles/day):")
    print("-" * 55)
    print(f"{'Priority':<10} {'Engine':<10} {'Pred RUL':<12} {'~Days Left'}")
    print("-" * 55)

    for priority, (engine_id, pred_rul) in enumerate(predictions, 1):
        days = pred_rul / cycles_per_day
        print(f"{priority:<10} {engine_id:<10} {pred_rul:<12.2f} ~{days:.1f}")

    print("-" * 55)
    print()


def example_4_confidence_assessment():
    """Assess prediction confidence using model error characteristics."""
    print("=" * 70)
    print("Example 4: Prediction Confidence Assessment")
    print("=" * 70)

    predictor = RULPredictor(model_path=MODEL_PATH)
    _, _, (test_X, test_y) = get_datasets(fd=1)

    unit_id = 1
    pred_rul = predictor.predict_single(test_X[unit_id][-1])
    true_rul = test_y[unit_id][-1]

    # Model characterization from experiments
    rmse = 6.37  # from production metrics
    mae = 4.84

    # ~95% of predictions fall within ±2*RMSE (approximate normal assumption)
    ci_lower = max(0, pred_rul - 2 * rmse)
    ci_upper = pred_rul + 2 * rmse

    print(f"\nEngine Unit #{unit_id} — Confidence Report")
    print(f"  Predicted RUL:       {pred_rul:.2f} cycles")
    print(f"  95% Confidence:      [{ci_lower:.2f}, {ci_upper:.2f}] cycles")
    print(f"  Expected Error:      ±{mae:.2f} cycles (MAE)")
    print(f"  True RUL:            {true_rul:.2f} cycles")
    print(f"  Actual Error:        {abs(pred_rul - true_rul):.2f} cycles")

    safety_margin = 20
    if ci_lower < safety_margin:
        decision = "Schedule maintenance (conservative — CI includes danger zone)"
    elif pred_rul < safety_margin:
        decision = "Schedule maintenance (point estimate in danger zone)"
    else:
        decision = "Continue operation"
    print(f"\n  Decision: {decision}")
    print()


def example_5_custom_sensor_data():
    """Predict from custom sensor data (simulated)."""
    print("=" * 70)
    print("Example 5: Custom Sensor Data Prediction")
    print("=" * 70)

    predictor = RULPredictor(model_path=MODEL_PATH)

    # Simulate 32-feature sensor data (match training data format)
    np.random.seed(42)
    timesteps = 5000
    num_features = 32
    custom_data = np.random.randn(timesteps, num_features)

    pred_rul = predictor.predict_single(custom_data)

    print(f"\nCustom Engine Analysis:")
    print(f"  Input shape:      ({timesteps}, {num_features})")
    print(f"  Predicted RUL:    {pred_rul:.2f} cycles")

    if pred_rul < 20:
        rec = "URGENT: Schedule immediate inspection"
    elif pred_rul < 50:
        rec = "Plan maintenance within next 2 weeks"
    else:
        rec = "Engine healthy, continue normal operation"
    print(f"  Recommendation:   {rec}")
    print()


if __name__ == "__main__":
    print("\nCNN-GRU RUL Predictor — Usage Examples\n")

    try:
        example_1_single_prediction()
        example_2_fleet_monitoring()
        example_3_maintenance_priority()
        example_4_confidence_assessment()
        example_5_custom_sensor_data()

        print("=" * 70)
        print("All examples completed successfully.")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Integrate RULPredictor into your monitoring system")
        print("  - Set up alerts for RUL < 20 cycles")
        print("  - Deploy as REST API (Flask/FastAPI)")
        print()

    except FileNotFoundError:
        print("\nModel not found. Train it first:")
        print("  WANDB_MODE=offline python train_production_model.py")
        print()
