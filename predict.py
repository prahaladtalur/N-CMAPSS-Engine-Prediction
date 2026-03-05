"""
Inference script for production RUL prediction using trained models.

Usage:
    # Ensemble prediction (best accuracy - uses MSTCN + Transformer + WaveNet)
    python predict.py --ensemble --fd 1

    # Single model prediction
    python predict.py --model-path models/production/mstcn_model.keras

    # Predict on specific unit
    python predict.py --model-path models/production/mstcn_model.keras --unit-id 2

    # Predict on custom data (.npy, shape: (timesteps, features))
    python predict.py --model-path models/production/mstcn_model.keras --input-file data.npy
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.data.load_data import get_datasets
from src.utils.metrics import compute_all_metrics, format_metrics


class RULPredictor:
    """Production RUL predictor with single model or ensemble support."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        ensemble: bool = False,
        model_paths: Optional[List[str]] = None,
    ):
        """
        Initialize predictor with trained model(s).

        Args:
            model_path: Path to single saved .keras model
            config_path: Path to config.json (default: same dir as model)
            ensemble: If True, use ensemble of top 3 models (MSTCN, Transformer, WaveNet)
            model_paths: Custom list of model paths for ensemble
        """
        self.ensemble = ensemble
        self.models = []
        self.model_names = []
        self.ensemble_weights = [0.5, 0.3, 0.2]  # Based on benchmark rankings

        if ensemble:
            # Use top 3 models from benchmark
            default_ensemble = model_paths or [
                "models/production/mstcn_model.keras",
                "models/production/transformer_model.keras",
                "models/production/wavenet_model.keras",
            ]
            for path in default_ensemble:
                if Path(path).exists():
                    model = tf.keras.models.load_model(
                        path, custom_objects={"loss": self._asymmetric_mse()}
                    )
                    self.models.append(model)
                    self.model_names.append(Path(path).stem.replace("_model", ""))
                    print(f"✓ Loaded: {Path(path).name}")
                else:
                    print(f"⚠️  Model not found: {path}")

            if not self.models:
                raise FileNotFoundError("No ensemble models found. Train models first.")

            self.model_dir = str(Path(default_ensemble[0]).parent)
        else:
            if model_path is None:
                raise ValueError("model_path required when ensemble=False")
            self.model_dir = str(Path(model_path).parent)
            self.models = [
                tf.keras.models.load_model(
                    model_path, custom_objects={"loss": self._asymmetric_mse()}
                )
            ]
            self.model_names = [Path(model_path).stem.replace("_model", "")]
            print(f"✓ Model loaded from: {model_path}")

        # Load config
        if config_path is None:
            config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
            print(f"✓ Config loaded from: {config_path}")
        else:
            self.config = {"max_sequence_length": 1000}

        self.max_seq_length = self.config.get("max_sequence_length", 1000)

        # Load feature scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"✓ Feature scaler loaded from: {scaler_path}")
        else:
            self.scaler = None
            print("⚠️  No feature scaler found — predictions may be inaccurate")

        # Load RUL scaler (for normalized metrics)
        rul_scaler_path = os.path.join(self.model_dir, "rul_scaler.json")
        if os.path.exists(rul_scaler_path):
            with open(rul_scaler_path, "r") as f:
                rul_scaler = json.load(f)
                self.y_min = rul_scaler["y_min"]
                self.y_max = rul_scaler["y_max"]
            print(f"✓ RUL scaler loaded from: {rul_scaler_path}")
        else:
            self.y_min = None
            self.y_max = None
            print("⚠️  No RUL scaler found — normalized metrics unavailable")

    @staticmethod
    def _asymmetric_mse(alpha: float = 2.0):
        """Asymmetric MSE loss for model loading."""

        def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            error = y_pred - y_true
            return tf.reduce_mean(tf.where(error >= 0, alpha * tf.square(error), tf.square(error)))

        return loss

    def _prepare_single(self, sequence: np.ndarray) -> np.ndarray:
        """
        Prepare a single sequence for inference.

        Args:
            sequence: Raw sensor data (timesteps, features)

        Returns:
            Batch-ready array (1, max_seq_length, features)
        """
        # Truncate to last max_seq_length timesteps
        if len(sequence) > self.max_seq_length:
            sequence = sequence[-self.max_seq_length :]

        # Normalize
        if self.scaler is not None:
            sequence = self.scaler.transform(sequence)

        return sequence[np.newaxis, ...]  # add batch dim

    def predict_single(self, sequence: np.ndarray) -> Dict:
        """
        Predict RUL for a single sequence.

        Args:
            sequence: Sensor data (timesteps, features)

        Returns:
            Dictionary with prediction and confidence info
        """
        X = self._prepare_single(sequence)

        predictions = []
        individual_preds = {}

        for model, name in zip(self.models, self.model_names):
            pred = float(model.predict(X, verbose=0)[0, 0])
            predictions.append(pred)
            individual_preds[name] = pred

        # Calculate ensemble prediction (weighted average)
        if self.ensemble and len(predictions) == len(self.ensemble_weights):
            final_pred = sum(p * w for p, w in zip(predictions, self.ensemble_weights))
        else:
            final_pred = np.mean(predictions)

        return {
            "prediction": final_pred,
            "individual_predictions": individual_preds,
            "std_dev": np.std(predictions) if len(predictions) > 1 else 0.0,
            "confidence": self._calculate_confidence(predictions) if len(predictions) > 1 else "N/A",
        }

    def _calculate_confidence(self, predictions: List[float]) -> str:
        """Calculate prediction confidence based on model agreement."""
        std = np.std(predictions)
        if std < 2.0:
            return "HIGH"
        elif std < 5.0:
            return "MEDIUM"
        else:
            return "LOW"

    def predict_unit(
        self, unit_X: np.ndarray, unit_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Predict RUL for all cycles in a unit.

        Args:
            unit_X: Unit data (num_cycles, timesteps, features)
            unit_y: True RUL values (num_cycles,)

        Returns:
            (y_true, y_pred, details) where details contains confidence info
        """
        y_pred = []
        details = []
        for cycle_idx in range(unit_X.shape[0]):
            result = self.predict_single(unit_X[cycle_idx])
            y_pred.append(result["prediction"])
            details.append(result)
        return unit_y, np.array(y_pred), details

    def evaluate_test_set(
        self,
        fd: int = 1,
        visualize: bool = True,
        output_dir: str = "results/predictions",
    ) -> Dict[str, float]:
        """
        Evaluate model on N-CMAPSS test set.

        Args:
            fd: Dataset sub-index (1-7)
            visualize: Whether to generate plots
            output_dir: Directory to save plots

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nLoading N-CMAPSS FD{fd} test set...")
        _, _, (test_X, test_y) = get_datasets(fd=fd)

        all_true, all_pred = [], []
        for unit_id in range(len(test_X)):
            print(f"  Predicting unit {unit_id} ({test_X[unit_id].shape[0]} cycles)...")
            y_true, y_pred, _ = self.predict_unit(test_X[unit_id], test_y[unit_id])
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)

        metrics = compute_all_metrics(y_true, y_pred, y_min=self.y_min, y_max=self.y_max)
        print("\n" + format_metrics(metrics))

        if visualize:
            os.makedirs(output_dir, exist_ok=True)
            self._plot_predictions(y_true, y_pred, output_dir)
            print(f"\n✓ Plots saved to: {output_dir}")

        return metrics

    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
        """Generate prediction visualizations."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
        lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0].plot(lim, lim, "r--", lw=2)
        axes[0].set_xlabel("True RUL")
        axes[0].set_ylabel("Predicted RUL")
        axes[0].set_title("RUL Predictions vs Ground Truth")
        axes[0].grid(True, alpha=0.3)

        error = y_pred - y_true
        axes[1].hist(error, bins=50, alpha=0.7, edgecolor="black")
        axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Prediction Error (cycles)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Error Distribution")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="RUL Prediction using trained models (single or ensemble)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ensemble prediction (best accuracy)
  python predict.py --ensemble --fd 1

  # Single model prediction
  python predict.py --model-path models/production/mstcn_model.keras

  # Predict specific test unit
  python predict.py --model-path models/production/mstcn_model.keras --unit-id 5
        """,
    )
    parser.add_argument("--model-path", type=str, help="Path to single .keras model")
    parser.add_argument("--config-path", type=str, default=None, help="Path to config.json")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble of top 3 models (MSTCN + Transformer + WaveNet)",
    )
    parser.add_argument("--fd", type=int, default=1, help="N-CMAPSS sub-dataset (1-7)")
    parser.add_argument(
        "--unit-id", type=int, default=None, help="Predict specific test unit (default: all)"
    )
    parser.add_argument(
        "--input-file", type=str, default=None, help="Custom .npy input (timesteps, features)"
    )
    parser.add_argument("--output-dir", type=str, default="results/predictions")
    parser.add_argument("--no-viz", action="store_true", help="Disable plots")

    args = parser.parse_args()

    if not args.model_path and not args.ensemble:
        parser.error("Either --model-path or --ensemble required")

    if args.ensemble:
        print("\n🔮 Ensemble Mode: Using top 3 models (MSTCN, Transformer, WaveNet)")
        print("Ensemble weights: 50% MSTCN, 30% Transformer, 20% WaveNet\n")
        predictor = RULPredictor(ensemble=True)
    else:
        predictor = RULPredictor(model_path=args.model_path, config_path=args.config_path)

    # Custom input file
    if args.input_file:
        print(f"\nLoading custom input from: {args.input_file}")
        X = np.load(args.input_file)
        result = predictor.predict_single(X)
        print(f"\n{'='*60}")
        print(f"Predicted RUL: {result['prediction']:.2f} cycles")
        if predictor.ensemble:
            print(f"Confidence: {result['confidence']} (std: {result['std_dev']:.2f})")
            print(f"\nIndividual predictions:")
            for model, pred in result["individual_predictions"].items():
                print(f"  {model:15s}: {pred:6.2f} cycles")
        print(f"{'='*60}\n")

    # Specific test unit
    elif args.unit_id is not None:
        print(f"\nLoading N-CMAPSS FD{args.fd} test set...")
        _, _, (test_X, test_y) = get_datasets(fd=args.fd)

        if args.unit_id >= len(test_X):
            print(f"❌ Unit ID {args.unit_id} out of range (max: {len(test_X)-1})")
            return

        unit_X = test_X[args.unit_id]
        unit_y = test_y[args.unit_id]
        num_cycles = unit_X.shape[0]

        print(f"\nUnit {args.unit_id}: {num_cycles} cycles")
        print("-" * 50)

        # Predict last cycle (most critical — closest to failure)
        last_cycle_result = predictor.predict_single(unit_X[-1])
        last_cycle_pred = last_cycle_result["prediction"]
        last_cycle_true = unit_y[-1]

        print(f"  Last cycle (most critical):")
        print(f"    True RUL:      {last_cycle_true:.2f} cycles")
        print(f"    Predicted RUL: {last_cycle_pred:.2f} cycles")
        print(f"    Error:         {last_cycle_pred - last_cycle_true:.2f} cycles")
        if predictor.ensemble:
            print(f"    Confidence:    {last_cycle_result['confidence']}")
            print(f"    Std Dev:       {last_cycle_result['std_dev']:.2f} cycles")

        # Evaluate all cycles
        y_true, y_pred, details = predictor.predict_unit(unit_X, unit_y)
        metrics = compute_all_metrics(y_true, y_pred, y_min=predictor.y_min, y_max=predictor.y_max)
        print(f"\n  All cycles:")
        print(f"    RMSE:    {metrics['rmse']:.2f}")
        print(f"    MAE:     {metrics['mae']:.2f}")
        print(f"    R²:      {metrics['r2']:.4f}")
        print(f"    Acc@10:  {metrics['accuracy_10']:.1f}%")
        print(f"    Acc@20:  {metrics['accuracy_20']:.1f}%")
        if "rmse_normalized" in metrics:
            print(f"    RMSE(norm): {metrics['rmse_normalized']:.4f}")
            print(f"    MAE(norm):  {metrics['mae_normalized']:.4f}")

    # Full test set evaluation
    else:
        predictor.evaluate_test_set(
            fd=args.fd,
            visualize=not args.no_viz,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
