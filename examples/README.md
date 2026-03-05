# Production Deployment Examples

This directory contains production-ready deployment examples for the RUL prediction models.

---

## REST API Server (`api_server.py`)

FastAPI-based REST API for serving predictions in production.

### Features

- ✅ Health check endpoint
- ✅ Single prediction endpoint
- ✅ Batch prediction endpoint
- ✅ Ensemble or single model support
- ✅ Request validation with Pydantic
- ✅ Automatic error handling
- ✅ Structured logging
- ✅ OpenAPI documentation (Swagger UI)
- ✅ Ready for Docker deployment

### Quick Start

#### 1. Install Dependencies

```bash
# Install FastAPI and Uvicorn
uv add fastapi uvicorn python-multipart
```

#### 2. Prepare Models

```bash
# Train ensemble models (one-time setup)
python scripts/prepare_ensemble.py
```

#### 3. Start Server

```bash
# Development mode (auto-reload)
uvicorn examples.api_server:app --reload --host 0.0.0.0 --port 8000

# Production mode (4 workers)
uvicorn examples.api_server:app --workers 4 --host 0.0.0.0 --port 8000
```

#### 4. Test API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Example request format
curl http://localhost:8000/example
```

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "ensemble",
  "timestamp": "2026-03-04T10:30:00.000Z"
}
```

#### `POST /predict`

Single prediction endpoint.

**Request:**
```json
{
  "sensor_data": [
    [1.2, 3.4, 5.6, ...],  // 32 features per timestep
    [1.1, 3.3, 5.5, ...],  // Up to 1000 timesteps
    ...
  ],
  "unit_id": "engine_001"  // Optional identifier
}
```

**Response:**
```json
{
  "unit_id": "engine_001",
  "prediction": 42.35,
  "confidence": "HIGH",
  "std_dev": 1.2,
  "individual_predictions": {
    "mstcn": 42.10,
    "transformer": 42.45,
    "wavenet": 42.50
  },
  "timestamp": "2026-03-04T10:30:00.000Z",
  "model_version": "2.0.0"
}
```

#### `POST /predict/batch`

Batch prediction endpoint (up to 100 units).

**Request:**
```json
{
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
```

**Response:**
```json
{
  "predictions": [...],
  "errors": [],
  "total_requested": 2,
  "successful": 2,
  "failed": 0,
  "timestamp": "2026-03-04T10:30:00.000Z"
}
```

#### `GET /model/info`

Get model configuration and expected performance.

**Response:**
```json
{
  "model_type": "ensemble",
  "version": "2.0.0",
  "expected_performance": {
    "rmse": "6.5-6.7 cycles",
    "r2": "0.91-0.92",
    "accuracy_at_20": ">99%"
  },
  "ensemble_config": {
    "models": ["mstcn", "transformer", "wavenet"],
    "weights": {
      "mstcn": 0.5,
      "transformer": 0.3,
      "wavenet": 0.2
    }
  }
}
```

### Python Client Example

```python
import requests
import numpy as np

# API endpoint
API_URL = "http://localhost:8000"

# Generate test data (replace with real sensor readings)
sensor_data = np.random.randn(1000, 32).tolist()

# Make prediction request
response = requests.post(
    f"{API_URL}/predict",
    json={
        "sensor_data": sensor_data,
        "unit_id": "engine_001"
    }
)

# Parse result
result = response.json()
print(f"Predicted RUL: {result['prediction']:.2f} cycles")
print(f"Confidence: {result['confidence']}")

# Batch prediction
batch_response = requests.post(
    f"{API_URL}/predict/batch",
    json={
        "batch": [
            {"sensor_data": sensor_data, "unit_id": "engine_001"},
            {"sensor_data": sensor_data, "unit_id": "engine_002"},
        ]
    }
)

batch_result = batch_response.json()
print(f"Batch: {batch_result['successful']}/{batch_result['total_requested']} succeeded")
```

### Performance

**Expected Latency** (M1 MacBook Pro):
- Single prediction: ~30-50ms (ensemble), ~10-20ms (single model)
- Batch prediction (10 units): ~200-300ms (ensemble)

**Throughput** (4 workers):
- ~100-200 requests/second (single predictions)
- ~20-40 batches/second (10 units each)

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install uv and dependencies
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "examples.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:

```bash
# Build image
docker build -t rul-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models rul-api

# Or with docker-compose
docker-compose up
```

### Production Checklist

Before deploying to production:

- [ ] Train and validate ensemble models
- [ ] Set up proper logging (integrate with your log aggregation)
- [ ] Add authentication (API keys, OAuth, etc.)
- [ ] Configure CORS if needed
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Load testing (with `locust` or `k6`)
- [ ] Set resource limits (memory, CPU)
- [ ] Configure autoscaling (Kubernetes HPA)
- [ ] Set up health checks in load balancer
- [ ] Enable HTTPS (TLS certificates)
- [ ] Add rate limiting
- [ ] Set up CI/CD pipeline

### Monitoring

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

# Add to api_server.py
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict_rul(request: PredictionRequest):
    prediction_counter.inc()
    with prediction_latency.time():
        # ... existing code
```

### Scaling

**Horizontal scaling** (multiple instances):
```bash
# Kubernetes deployment
kubectl scale deployment rul-api --replicas=10
```

**Vertical scaling** (more workers):
```bash
# Increase workers
uvicorn examples.api_server:app --workers 8
```

### Troubleshooting

**Issue**: `Model not loaded` error

**Solution**: Ensure models exist in `models/production/`:
```bash
python scripts/prepare_ensemble.py
```

**Issue**: High latency

**Solutions**:
1. Use single model instead of ensemble
2. Reduce input sequence length
3. Add caching for repeated predictions
4. Use model quantization

**Issue**: Out of memory

**Solutions**:
1. Reduce number of workers
2. Use smaller batch sizes
3. Increase container memory limits
4. Use single model instead of ensemble

---

## Future Examples

Coming soon:
- **Streamlit Dashboard** - Interactive web UI for predictions
- **AWS Lambda Deployment** - Serverless deployment example
- **Kubernetes Manifests** - Complete K8s deployment configuration
- **Grafana Dashboards** - Monitoring and visualization
- **Load Testing Scripts** - Performance benchmarking with Locust

---

## Support

For issues or questions:
- **Documentation**: [README.md](../README.md)
- **Ensemble Guide**: [ENSEMBLE_GUIDE.md](../ENSEMBLE_GUIDE.md)
- **Issues**: https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction/issues

---

**Last Updated**: March 4, 2026
