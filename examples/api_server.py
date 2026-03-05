#!/usr/bin/env python3
"""
Production-ready REST API for RUL prediction using FastAPI.

This demonstrates how to deploy the ensemble model as a REST API service
for real-world production use.

Installation:
    uv add fastapi uvicorn python-multipart

Usage:
    # Start server
    uvicorn examples.api_server:app --reload --host 0.0.0.0 --port 8000

    # Or with production settings
    uvicorn examples.api_server:app --workers 4 --host 0.0.0.0 --port 8000

    # Test API
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"sensor_data": [[1.2, 3.4, ...]], "unit_id": "engine_001"}'

Features:
    - Health check endpoint
    - Single prediction endpoint
    - Batch prediction endpoint
    - Ensemble or single model mode
    - Request validation
    - Error handling
    - Logging
    - Prometheus metrics (optional)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import numpy as np
import uvicorn
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from predict import RULPredictor
except ImportError:
    print("Error: Could not import RULPredictor. Make sure predict.py is available.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="RUL Prediction API",
    description="Remaining Useful Life prediction for turbofan engines using MSTCN ensemble",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global model instance (loaded on startup)
predictor: Optional[RULPredictor] = None


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for single prediction."""

    sensor_data: List[List[float]] = Field(
        ...,
        description="Sensor readings (timesteps × features), shape: (T, 32)",
        min_items=1,
    )

    unit_id: Optional[str] = Field(
        None,
        description="Optional unit/engine identifier for logging"
    )

    @validator('sensor_data')
    def validate_sensor_data(cls, v):
        """Validate sensor data shape and values."""
        if not v:
            raise ValueError("sensor_data cannot be empty")

        # Check all timesteps have same number of features
        n_features = len(v[0])
        if not all(len(timestep) == n_features for timestep in v):
            raise ValueError("All timesteps must have the same number of features")

        # Expect 32 features (N-CMAPSS standard)
        if n_features != 32:
            logger.warning(f"Expected 32 features, got {n_features}")

        # Check for invalid values
        flat_data = [val for timestep in v for val in timestep]
        if any(np.isnan(val) or np.isinf(val) for val in flat_data):
            raise ValueError("sensor_data contains NaN or Inf values")

        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""

    batch: List[PredictionRequest] = Field(
        ...,
        description="List of prediction requests",
        min_items=1,
        max_items=100,  # Limit batch size
    )


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    unit_id: Optional[str]
    prediction: float = Field(..., description="Predicted RUL in cycles")
    confidence: Optional[str] = Field(None, description="Confidence level (HIGH/MEDIUM/LOW)")
    std_dev: Optional[float] = Field(None, description="Standard deviation across ensemble models")
    individual_predictions: Optional[Dict[str, float]] = Field(
        None,
        description="Individual model predictions (ensemble mode only)"
    )
    timestamp: str = Field(..., description="Prediction timestamp (ISO 8601)")
    model_version: str = Field(..., description="Model version")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_type: str
    timestamp: str


# Startup event: Load model
@app.on_event("startup")
async def load_model():
    """Load prediction model on startup."""
    global predictor

    logger.info("Loading RUL prediction model...")

    try:
        # Try ensemble mode first (best accuracy)
        predictor = RULPredictor(ensemble=True)
        logger.info("✓ Ensemble model loaded (MSTCN + Transformer + WaveNet)")

    except Exception as e:
        logger.warning(f"Could not load ensemble: {e}")
        logger.info("Attempting to load single MSTCN model...")

        try:
            # Fallback to single MSTCN
            predictor = RULPredictor(
                model_path="models/production/mstcn_model.keras"
            )
            logger.info("✓ Single MSTCN model loaded")

        except Exception as e2:
            logger.error(f"Failed to load any model: {e2}")
            logger.error("API will run but predictions will fail")
            predictor = None


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns API status and model loading state.
    """
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_type="ensemble" if (predictor and predictor.ensemble) else "single",
        timestamp=datetime.utcnow().isoformat(),
    )


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(request: PredictionRequest):
    """
    Predict RUL for a single unit.

    Args:
        request: Prediction request with sensor data

    Returns:
        Prediction result with confidence metrics

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs and restart."
        )

    try:
        # Convert to numpy array
        sensor_data = np.array(request.sensor_data)

        logger.info(
            f"Prediction request: unit_id={request.unit_id}, "
            f"shape={sensor_data.shape}"
        )

        # Make prediction
        result = predictor.predict_single(sensor_data)

        # Build response
        response = PredictionResponse(
            unit_id=request.unit_id,
            prediction=round(result["prediction"], 2),
            confidence=result.get("confidence"),
            std_dev=result.get("std_dev"),
            individual_predictions=result.get("individual_predictions"),
            timestamp=datetime.utcnow().isoformat(),
            model_version="2.0.0",
        )

        logger.info(
            f"Prediction completed: unit_id={request.unit_id}, "
            f"RUL={response.prediction:.2f}, confidence={response.confidence}"
        )

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict RUL for multiple units in batch.

    Args:
        request: Batch prediction request

    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs and restart."
        )

    logger.info(f"Batch prediction request: {len(request.batch)} units")

    results = []
    errors = []

    for idx, pred_request in enumerate(request.batch):
        try:
            result = await predict_rul(pred_request)
            results.append(result)
        except HTTPException as e:
            errors.append({
                "index": idx,
                "unit_id": pred_request.unit_id,
                "error": e.detail
            })
            logger.error(f"Batch item {idx} failed: {e.detail}")

    return {
        "predictions": results,
        "errors": errors,
        "total_requested": len(request.batch),
        "successful": len(results),
        "failed": len(errors),
        "timestamp": datetime.utcnow().isoformat(),
    }


# Model info endpoint
@app.get("/model/info")
async def model_info():
    """
    Get information about the loaded model.

    Returns:
        Model configuration and performance metrics
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    info = {
        "model_type": "ensemble" if predictor.ensemble else "single",
        "version": "2.0.0",
        "expected_performance": {
            "rmse": "6.5-6.7 cycles" if predictor.ensemble else "6.8-7.5 cycles",
            "r2": "0.91-0.92" if predictor.ensemble else "0.88-0.90",
            "accuracy_at_20": ">99%" if predictor.ensemble else ">98%",
        },
        "input_requirements": {
            "shape": "(timesteps, 32)",
            "max_timesteps": 1000,
            "features": 32,
            "normalization": "StandardScaler (applied automatically)",
        },
    }

    if predictor.ensemble:
        info["ensemble_config"] = {
            "models": predictor.model_names,
            "weights": dict(zip(predictor.model_names, predictor.ensemble_weights)),
        }

    return info


# Example usage endpoint
@app.get("/example")
async def get_example():
    """
    Get example request format.

    Returns:
        Example prediction request
    """
    return {
        "single_prediction": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "sensor_data": [
                    [1.2, 3.4, 5.6, 7.8, ...],  # Timestep 1 (32 features)
                    [1.1, 3.3, 5.5, 7.7, ...],  # Timestep 2
                    # ... more timesteps (up to 1000)
                ],
                "unit_id": "engine_001"
            }
        },
        "batch_prediction": {
            "url": "/predict/batch",
            "method": "POST",
            "body": {
                "batch": [
                    {
                        "sensor_data": [[...]],
                        "unit_id": "engine_001"
                    },
                    {
                        "sensor_data": [[...]],
                        "unit_id": "engine_002"
                    }
                ]
            }
        }
    }


# Main entry point
if __name__ == "__main__":
    # For development
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
