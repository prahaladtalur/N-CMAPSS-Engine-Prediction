#!/usr/bin/env python3
"""
Interactive demo of RUL prediction with synthetic data.

This script demonstrates the complete inference pipeline:
1. Generate synthetic sensor data
2. Preprocess and normalize
3. Make predictions with single model or ensemble
4. Display results with confidence metrics

Usage:
    # Demo with synthetic data (no trained model needed)
    python demo_inference.py

    # Demo with trained MSTCN model
    python demo_inference.py --model-path models/production/mstcn_model.keras

    # Demo with ensemble (best accuracy)
    python demo_inference.py --ensemble
"""

import argparse
import numpy as np
from pathlib import Path
import sys

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Error: scikit-learn not found. Install with: uv sync")
    sys.exit(1)


def generate_synthetic_sequence(
    timesteps: int = 1000,
    n_features: int = 32,
    degradation_profile: str = "linear",
    noise_level: float = 0.1,
) -> np.ndarray:
    """
    Generate synthetic turbofan sensor data with degradation.

    Args:
        timesteps: Number of time steps (default: 1000)
        n_features: Number of sensor features (default: 32)
        degradation_profile: 'linear', 'exponential', or 'stepwise'
        noise_level: Amount of sensor noise (0.0-1.0)

    Returns:
        Synthetic sensor data (timesteps, n_features)
    """
    np.random.seed(42)  # Reproducible demo

    # Base sensor readings (normal operation)
    base_values = np.random.randn(n_features) * 10 + 50

    # Time array
    t = np.linspace(0, 1, timesteps)

    # Create degradation patterns
    if degradation_profile == "linear":
        # Linear degradation over time
        degradation = np.outer(t, np.random.rand(n_features) * 20)
    elif degradation_profile == "exponential":
        # Exponential degradation (accelerating failure)
        degradation = np.outer(np.exp(t * 2) - 1, np.random.rand(n_features) * 15)
    elif degradation_profile == "stepwise":
        # Stepwise degradation (sudden fault)
        step = np.where(t > 0.6, 1.0, 0.0)
        degradation = np.outer(step, np.random.rand(n_features) * 25)
    else:
        degradation = np.zeros((timesteps, n_features))

    # Add sensor noise
    noise = np.random.randn(timesteps, n_features) * noise_level * 5

    # Combine: base + degradation + noise
    sequence = base_values + degradation + noise

    return sequence


def simulate_preprocessing(sequence: np.ndarray) -> tuple:
    """
    Simulate preprocessing (normalization) as done in training.

    Returns:
        (normalized_sequence, scaler)
    """
    scaler = StandardScaler()
    normalized = scaler.fit_transform(sequence)
    return normalized, scaler


def calculate_true_rul(
    current_cycle: int,
    max_rul: int = 125,
    degradation_profile: str = "linear",
) -> float:
    """
    Calculate 'true' RUL for synthetic data based on degradation profile.

    This is the ground truth we would compare predictions against.
    """
    if degradation_profile == "linear":
        return max(0, max_rul - current_cycle * 0.1)
    elif degradation_profile == "exponential":
        return max(0, max_rul * np.exp(-current_cycle / 500))
    else:
        return max(0, max_rul - current_cycle * 0.12)


def demo_inference_synthetic():
    """
    Run inference demo with synthetic data (no model required).

    This demonstrates the complete pipeline without needing trained models.
    """
    print("\n" + "="*70)
    print("RUL PREDICTION DEMO - SYNTHETIC DATA")
    print("="*70)

    print("\n📊 Generating synthetic turbofan sensor data...")
    print("    (In production, this would be real sensor readings)\n")

    # Generate three scenarios
    scenarios = [
        {
            "name": "Healthy Engine",
            "profile": "linear",
            "timesteps": 500,
            "expected_rul": 100,
        },
        {
            "name": "Degrading Engine",
            "profile": "exponential",
            "timesteps": 800,
            "expected_rul": 40,
        },
        {
            "name": "Failing Engine",
            "profile": "linear",
            "timesteps": 1000,
            "expected_rul": 15,
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─'*70}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"{'─'*70}\n")

        # Generate data
        sequence = generate_synthetic_sequence(
            timesteps=scenario["timesteps"],
            degradation_profile=scenario["profile"],
        )

        print(f"  Sensor Data:")
        print(f"    Shape: {sequence.shape} ({sequence.shape[0]} timesteps × {sequence.shape[1]} sensors)")
        print(f"    Value range: [{sequence.min():.2f}, {sequence.max():.2f}]")

        # Preprocess
        normalized, scaler = simulate_preprocessing(sequence)

        print(f"\n  Preprocessing:")
        print(f"    ✓ Normalized with StandardScaler")
        print(f"    Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

        # Simulate prediction (since we don't have trained model)
        true_rul = scenario["expected_rul"]
        # Add some realistic noise to simulation
        simulated_pred = true_rul + np.random.randn() * 5

        # Simulate ensemble predictions (different models give different predictions)
        ensemble_preds = {
            "mstcn": simulated_pred + np.random.randn() * 2,
            "transformer": simulated_pred + np.random.randn() * 2,
            "wavenet": simulated_pred + np.random.randn() * 2,
        }

        ensemble_pred = np.mean(list(ensemble_preds.values()))
        std_dev = np.std(list(ensemble_preds.values()))

        confidence = "HIGH" if std_dev < 2.0 else ("MEDIUM" if std_dev < 5.0 else "LOW")

        print(f"\n  Prediction Results:")
        print(f"    🎯 Ensemble Prediction: {ensemble_pred:.2f} cycles")
        print(f"    📊 Confidence: {confidence} (std: {std_dev:.2f})")
        print(f"    ✓ True RUL (simulated): {true_rul:.2f} cycles")
        print(f"    📈 Error: {abs(ensemble_pred - true_rul):.2f} cycles")

        print(f"\n    Individual Model Predictions:")
        for model, pred in ensemble_preds.items():
            print(f"      • {model:15s}: {pred:6.2f} cycles")

        # Maintenance recommendation
        if ensemble_pred < 20:
            recommendation = "🔴 CRITICAL - Schedule maintenance immediately"
        elif ensemble_pred < 50:
            recommendation = "🟡 WARNING - Plan maintenance soon"
        else:
            recommendation = "🟢 HEALTHY - Continue normal operation"

        print(f"\n    Recommendation: {recommendation}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)

    print("\n💡 Next Steps:")
    print("   1. Train real models:")
    print("      python scripts/prepare_ensemble.py")
    print()
    print("   2. Run predictions on real test data:")
    print("      python predict.py --ensemble --fd 1")
    print()
    print("   3. Use in production:")
    print("      from predict import RULPredictor")
    print("      predictor = RULPredictor(ensemble=True)")
    print("      result = predictor.predict_single(your_sensor_data)")
    print()


def demo_inference_real(model_path: str = None, ensemble: bool = False):
    """
    Run inference demo with real model.

    Args:
        model_path: Path to trained model
        ensemble: Use ensemble mode
    """
    try:
        from predict import RULPredictor
    except ImportError:
        print("Error: predict.py not found or has import errors")
        sys.exit(1)

    print("\n" + "="*70)
    print("RUL PREDICTION DEMO - REAL MODEL")
    print("="*70)

    # Initialize predictor
    if ensemble:
        print("\n🔮 Mode: Ensemble (MSTCN + Transformer + WaveNet)\n")
        try:
            predictor = RULPredictor(ensemble=True)
        except Exception as e:
            print(f"❌ Error loading ensemble: {e}")
            print("\nTrain ensemble models first:")
            print("  python scripts/prepare_ensemble.py")
            sys.exit(1)
    else:
        print(f"\n📦 Mode: Single Model ({model_path})\n")
        try:
            predictor = RULPredictor(model_path=model_path)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    # Generate test sequence
    print("Generating test sequence...")
    sequence = generate_synthetic_sequence(
        timesteps=1000,
        degradation_profile="exponential",
    )

    print(f"  Shape: {sequence.shape}")
    print(f"  Range: [{sequence.min():.2f}, {sequence.max():.2f}]")

    # Make prediction
    print("\nMaking prediction...")
    result = predictor.predict_single(sequence)

    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)

    print(f"\n🎯 Predicted RUL: {result['prediction']:.2f} cycles")

    if ensemble:
        print(f"📊 Confidence: {result['confidence']} (std: {result['std_dev']:.2f})")
        print(f"\nIndividual Model Predictions:")
        for model, pred in result['individual_predictions'].items():
            print(f"  • {model:15s}: {pred:6.2f} cycles")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive RUL prediction demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model (omit for synthetic demo)",
    )

    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble mode (requires trained models)",
    )

    args = parser.parse_args()

    # Run appropriate demo
    if args.model_path or args.ensemble:
        demo_inference_real(args.model_path, args.ensemble)
    else:
        demo_inference_synthetic()


if __name__ == "__main__":
    main()
