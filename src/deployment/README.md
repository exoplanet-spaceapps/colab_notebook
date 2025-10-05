# Kepler Exoplanet Detection - Deployment Package

Complete deployment solution for serving Keras CNN, XGBoost, and Random Forest models via FastAPI.

## Contents

- **model_server_template.py** - FastAPI inference server with multi-model support
- **model_registry_manager.py** - CLI tool for model version management
- **requirements.txt** - Python dependencies
- **Dockerfile** - Container image definition
- **docker-compose.yml** - Multi-container deployment configuration
- **kubernetes-deployment.yaml** - Kubernetes manifests

## Quick Start

### Option 1: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python model_server_template.py

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Option 2: Docker

```bash
# Build image
docker build -t kepler-model-server:latest -f Dockerfile ../..

# Run container
docker run -d -p 8000:8000 -v $(pwd)/../../models:/app/models:ro kepler-model-server:latest

# Check health
curl http://localhost:8000/api/v1/health
```

### Option 3: Docker Compose

```bash
# Start all services (server, Redis, Prometheus, Grafana)
docker-compose up -d

# View logs
docker-compose logs -f model-server

# Stop all services
docker-compose down
```

### Option 4: Kubernetes

```bash
# Apply manifests
kubectl apply -f kubernetes-deployment.yaml

# Check status
kubectl get pods -l app=kepler-model-server

# Port forward
kubectl port-forward service/kepler-model-service 8000:80
```

## API Usage

### Single Prediction

```python
import requests
import numpy as np

# Generate sample features (784 values)
features = np.random.randn(784).tolist()

# Make prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "features": features,
        "model": "auto",
        "return_probabilities": True
    }
)

result = response.json()
print(f"Prediction: {result['prediction']['class_name']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

### Multi-Model Comparison

```python
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "features": features,
        "return_all_models": True
    }
)

result = response.json()
print(f"Ensemble: {result['ensemble_prediction']['class_name']}")
print(f"Agreement: {result['agreement_score']:.2%}")

for model, pred in result['individual_predictions'].items():
    print(f"{model}: {pred['class_name']} ({pred['confidence']:.2%})")
```

### Batch Prediction

```python
# Multiple samples
samples = [np.random.randn(784).tolist() for _ in range(10)]

response = requests.post(
    "http://localhost:8000/api/v1/predict/batch",
    json={
        "samples": samples,
        "model": "keras_cnn"
    }
)

results = response.json()
for i, pred in enumerate(results['predictions']):
    print(f"Sample {i}: {pred['class_name']} ({pred['confidence']:.2%})")
```

## Model Registry Management

### Register New Model

```bash
# Create metadata.json
cat > metadata.json <<EOF
{
  "framework": "tensorflow",
  "framework_version": "2.15.0",
  "training": {
    "dataset_version": "kepler_q1_q17_dr25",
    "training_samples": 1400,
    "epochs": 50
  },
  "performance": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90
  },
  "input_schema": {
    "features": 784,
    "dtype": "float32"
  },
  "output_schema": {
    "classes": 3,
    "labels": ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
  }
}
EOF

# Register model
python model_registry_manager.py register \
  --name keras_cnn \
  --version 1.0.0 \
  --model-file /path/to/model.keras \
  --metadata metadata.json \
  --preprocessing /path/to/preprocessing.pkl \
  --registry-path ../../models
```

### List Versions

```bash
python model_registry_manager.py list --name keras_cnn
```

### Get Model Info

```bash
python model_registry_manager.py info --name keras_cnn --version latest
```

### Compare Versions

```bash
python model_registry_manager.py compare \
  --name keras_cnn \
  --version-a 1.0.0 \
  --version-b 1.1.0
```

### Validate Model

```bash
python model_registry_manager.py validate --name keras_cnn --version latest
```

## Configuration

Environment variables (set in `.env` or Docker/K8s config):

```bash
# Model registry path
MODEL_REGISTRY_PATH=./models

# Cache size (number of models to keep in memory)
CACHE_SIZE=3

# Enable A/B testing
ENABLE_AB_TESTING=false

# Default model when "auto" is specified
DEFAULT_MODEL=auto
```

## Monitoring

### Prometheus Metrics

Available at `http://localhost:8000/metrics`:

- `predictions_total` - Total predictions counter
- `prediction_latency_seconds` - Prediction latency histogram
- `model_load_time_seconds` - Model load time gauge
- `active_models_count` - Number of active models
- `cache_hits_total` / `cache_misses_total` - Cache performance

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### View Models

```bash
curl http://localhost:8000/api/v1/models
```

## Performance Tuning

### Enable GPU

For Keras CNN and XGBoost models:

```bash
# Docker with GPU support
docker run --gpus all -p 8000:8000 kepler-model-server:latest
```

### Increase Workers

For higher throughput:

```bash
# Multiple workers (CPU-bound workloads)
uvicorn model_server_template:app --workers 4 --host 0.0.0.0 --port 8000
```

### Request Batching

Modify `model_server_template.py` to enable batching:

```python
# Uncomment BatchPredictor in the code
# Set batch_size and max_wait_ms
```

## Troubleshooting

### Issue: High Latency

**Solution**: Increase cache size, enable GPU, or use request batching

```bash
export CACHE_SIZE=5
```

### Issue: Out of Memory

**Solution**: Reduce cache size or increase container memory

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

### Issue: Model Not Found

**Solution**: Verify model registry structure

```bash
ls -la ../../models/keras_cnn/
# Should have: latest -> v1.0.0/, v1.0.0/model.keras, v1.0.0/metadata.json
```

## Architecture

```
Client Request
      ↓
API Gateway (FastAPI)
      ↓
Model Loader (with cache)
      ↓
Prediction Engine
      ↓
Model (Keras/XGBoost/RF)
      ↓
Response (JSON)
```

## Security

1. **API Authentication**: Add JWT/API key middleware
2. **Rate Limiting**: Use slowapi or NGINX
3. **Input Validation**: Pydantic schemas validate all inputs
4. **HTTPS**: Configure SSL certificates in NGINX/Ingress

## Documentation

- Full deployment guide: `../../docs/deployment_guide.md`
- API docs (interactive): `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Support

For issues or questions, refer to:
- Deployment Guide: `../../docs/deployment_guide.md`
- Project README: `../../README.md`
- API Documentation: `/docs` endpoint

---

**Version**: 1.0.0
**Last Updated**: 2025-10-05
**Maintained By**: System Architecture Team
