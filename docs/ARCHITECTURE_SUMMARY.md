# Kepler Exoplanet Detection - System Architecture Summary

**Document Version**: 1.0.0
**Created**: 2025-10-05
**Architect**: System Architecture Team

---

## Executive Summary

This document provides a comprehensive overview of the deployment architecture for the Kepler Exoplanet Detection system. The architecture supports three machine learning models (Keras CNN, XGBoost, Random Forest) through a unified FastAPI-based inference service with production-grade features including:

- Unified prediction API with multi-model support
- Model version management and registry
- A/B testing framework
- Automated rollback mechanisms
- Comprehensive monitoring and observability
- Docker and Kubernetes deployment options

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Environment                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐        ┌────────────────┐                │
│  │  API Gateway  │───────▶│  Load Balancer │                │
│  │   (NGINX)     │        │   (K8s/ALB)    │                │
│  └───────────────┘        └───────┬────────┘                │
│                                    │                          │
│                   ┌────────────────┼────────────────┐        │
│                   │                │                │         │
│           ┌───────▼──────┐ ┌──────▼─────┐ ┌───────▼──────┐ │
│           │   FastAPI    │ │  FastAPI   │ │   FastAPI    │ │
│           │   Server 1   │ │  Server 2  │ │   Server 3   │ │
│           └───────┬──────┘ └──────┬─────┘ └───────┬──────┘ │
│                   │                │                │         │
│                   └────────────────┼────────────────┘        │
│                                    │                          │
│                   ┌────────────────▼────────────────┐        │
│                   │      Model Registry (S3)        │        │
│                   │  - Keras CNN (v1.0.0, v1.1.0)   │        │
│                   │  - XGBoost (v1.0.0)             │        │
│                   │  - Random Forest (v1.0.0)       │        │
│                   └─────────────────────────────────┘        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Prometheus  │  │    Redis     │  │   Grafana    │      │
│  │  (Metrics)   │  │   (Cache)    │  │   (Viz)      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Model Server (`model_server_template.py`)

**Purpose**: FastAPI-based inference server supporting all three model types

**Key Features**:
- Multi-model support (Keras CNN, XGBoost, Random Forest)
- Automatic model selection based on performance
- Request validation using Pydantic
- Model caching with LRU eviction
- Ensemble predictions with soft voting
- Prometheus metrics integration
- Health check endpoints
- Circuit breaker pattern

**API Endpoints**:
```
POST /api/v1/predict              - Single prediction
POST /api/v1/predict/batch        - Batch prediction
GET  /api/v1/models               - List models
GET  /api/v1/models/{name}/info   - Model details
GET  /api/v1/health               - Health check
GET  /metrics                     - Prometheus metrics
```

**Performance**:
- Latency: 10-200ms depending on model
- Throughput: 10-30 RPS per instance
- Memory: 2-4GB per instance

---

### 2. Model Registry Manager (`model_registry_manager.py`)

**Purpose**: CLI tool for managing model versions and lifecycle

**Features**:
- Model registration with metadata
- Version comparison
- Checksum validation
- Model archiving
- Symlink management for latest/stable versions

**Usage**:
```bash
# Register new model
python model_registry_manager.py register \
  --name keras_cnn --version 1.0.0 \
  --model-file model.keras --metadata metadata.json

# List versions
python model_registry_manager.py list --name keras_cnn

# Compare versions
python model_registry_manager.py compare \
  --name keras_cnn --version-a 1.0.0 --version-b 1.1.0

# Validate model
python model_registry_manager.py validate --name keras_cnn
```

---

### 3. Model Deployment Requirements

#### Keras CNN Model
- **Format**: `.keras` (SavedModel)
- **Loading**: `tf.keras.models.load_model()`
- **Dependencies**: TensorFlow 2.15.0, Keras 2.15.0
- **Memory**: 500MB - 1GB
- **Latency**: 50-200ms (CPU), 10-50ms (GPU)

#### XGBoost Model
- **Format**: `.json` (preferred) or `.ubj`
- **Loading**: `xgb.Booster(); booster.load_model()`
- **Dependencies**: XGBoost 2.0.3
- **Memory**: 100MB - 500MB
- **Latency**: 10-50ms

#### Random Forest Model
- **Format**: `.pkl` (joblib)
- **Loading**: `joblib.load()`
- **Dependencies**: scikit-learn 1.4.0, joblib 1.3.2
- **Memory**: 200MB - 1GB
- **Latency**: 20-100ms

---

## Model Version Management

### Directory Structure

```
models/
├── keras_cnn/
│   ├── v1.0.0/
│   │   ├── model.keras
│   │   ├── metadata.json
│   │   ├── preprocessing.pkl
│   │   └── checksum.sha256
│   ├── v1.1.0/
│   │   └── ...
│   ├── latest -> v1.1.0
│   └── stable -> v1.0.0
├── xgboost/
│   └── v1.0.0/
│       ├── model.json
│       ├── metadata.json
│       └── checksum.sha256
└── random_forest/
    └── v1.0.0/
        ├── model.pkl
        ├── metadata.json
        └── checksum.sha256
```

### Metadata Schema

```json
{
  "model_name": "keras_cnn",
  "version": "1.0.0",
  "created_at": "2025-10-05T12:00:00Z",
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
    "f1_score": 0.90,
    "roc_auc": 0.94
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
```

---

## A/B Testing Framework

### Traffic Splitting

The system supports A/B testing for comparing model versions:

```python
# Route 10% traffic to new version
router = ABTestRouter(
    model_a="keras_cnn_v1.0.0",
    model_b="keras_cnn_v1.1.0",
    split_ratio=0.1
)
```

### Metrics Tracking

Track performance across versions:
- Request count per version
- Average latency per version
- Error rate per version
- Prediction distribution per version

### Rollback Triggers

Automatic rollback on:
- Error rate > 5% for 5 minutes
- Latency > 2x baseline
- Prediction drift > 0.3 KL divergence
- Health check failures (3 consecutive)

---

## Deployment Options

### Option 1: Local Development

```bash
pip install -r requirements.txt
python src/deployment/model_server_template.py
```

**Use Case**: Development, testing, debugging

### Option 2: Docker

```bash
docker build -t kepler-model-server:latest .
docker run -p 8000:8000 -v ./models:/app/models:ro kepler-model-server:latest
```

**Use Case**: Consistent environment, easy distribution

### Option 3: Docker Compose

```bash
docker-compose up -d
```

**Includes**:
- Model server (3 replicas)
- Redis cache
- Prometheus monitoring
- Grafana dashboards
- NGINX reverse proxy

**Use Case**: Local production-like environment, integration testing

### Option 4: Kubernetes

```bash
kubectl apply -f src/deployment/kubernetes-deployment.yaml
```

**Features**:
- Horizontal Pod Autoscaler (3-10 replicas)
- Rolling updates with zero downtime
- Health checks and self-healing
- Persistent volume for models
- Ingress with TLS
- Service mesh integration

**Use Case**: Production at scale, high availability

---

## Monitoring & Observability

### Prometheus Metrics

**System Metrics**:
- `predictions_total` - Counter by model, status, class
- `prediction_latency_seconds` - Histogram by model
- `model_load_time_seconds` - Gauge by model and version
- `active_models_count` - Active models in registry

**Cache Metrics**:
- `cache_hits_total` - Cache hits by model
- `cache_misses_total` - Cache misses by model

### Health Checks

```bash
# Liveness probe
GET /api/v1/health

# Response
{
  "status": "healthy",
  "timestamp": "2025-10-05T12:00:00Z",
  "models_loaded": 3,
  "cache_size": 2
}
```

### Logging

Structured JSON logging with:
- Request ID tracking
- Model version tracking
- Error stack traces
- Performance metrics

---

## Security Considerations

### Input Validation
- Pydantic schemas validate all inputs
- Feature count validation (must be 784)
- NaN/Inf detection
- Type checking

### API Security (To Implement)
- JWT authentication
- API key management
- Rate limiting (e.g., 100 req/min per client)
- Request signing

### Network Security
- HTTPS/TLS encryption
- Network policies in Kubernetes
- Private model registry access
- Secrets management (environment variables, K8s secrets)

---

## Performance Optimization

### Caching Strategy
- LRU cache for loaded models (default: 3 models)
- Redis for prediction results (optional)
- Preprocessor caching

### Request Batching
- Automatic batching for high throughput
- Configurable batch size and timeout
- Reduces inference overhead

### GPU Acceleration
- TensorFlow GPU support for Keras CNN
- XGBoost GPU support (tree_method='gpu_hist')
- CUDA 11.8+ and cuDNN 8.6+ required

### Horizontal Scaling
- Stateless design enables easy scaling
- Kubernetes HPA for auto-scaling
- Load balancing across replicas

---

## Deployment Checklist

### Pre-Deployment
- [ ] Model files verified with checksums
- [ ] Dependencies installed and tested
- [ ] Model loading test passed
- [ ] Performance benchmark completed
- [ ] Docker image built and tested
- [ ] Kubernetes manifests validated

### Deployment
- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Canary deployment (10% traffic)
- [ ] Monitor for 30 minutes
- [ ] Gradual rollout to 100%

### Post-Deployment
- [ ] Verify health checks passing
- [ ] Confirm metrics collection
- [ ] Test rollback procedure
- [ ] Update documentation
- [ ] Notify stakeholders

---

## Disaster Recovery

### Backup Strategy
- Model registry backed up to S3/Cloud Storage
- Database backups (if using model registry DB)
- Configuration files in version control

### Rollback Procedure
1. Stop traffic to failing version
2. Warm up stable version
3. Gradual traffic migration (10% → 100%)
4. Monitor health at each step
5. Update registry to stable version

### Recovery Time Objectives
- **RTO (Recovery Time Objective)**: < 5 minutes
- **RPO (Recovery Point Objective)**: < 1 hour
- **Failover**: Automatic via health checks

---

## Cost Optimization

### Resource Planning
- **Development**: 1 instance (2 CPU, 4GB RAM)
- **Staging**: 2 instances (2 CPU, 4GB RAM)
- **Production**: 3-10 instances (2-4 CPU, 4-8GB RAM)

### Autoscaling Rules
- Scale up: CPU > 70% or Memory > 80%
- Scale down: CPU < 30% for 5 minutes
- Min replicas: 3 (for high availability)
- Max replicas: 10 (cost limit)

### Cost Estimation (AWS)
- **Small** (100 req/min): ~$50/month (t3.medium × 3)
- **Medium** (1000 req/min): ~$200/month (t3.large × 5)
- **Large** (10000 req/min): ~$800/month (c5.xlarge × 10)

---

## Future Enhancements

### Phase 2 (Q2 2025)
- Model serving via TensorFlow Serving
- ONNX export for cross-platform deployment
- Model compression (quantization, pruning)
- Edge deployment (TensorFlow Lite)

### Phase 3 (Q3 2025)
- Online learning and continuous retraining
- Feature store integration
- MLOps pipeline automation
- Advanced A/B testing with multi-armed bandits

### Phase 4 (Q4 2025)
- Federated learning support
- Model explanation API (SHAP, LIME)
- Real-time monitoring dashboards
- Automated model governance

---

## References

### Documentation
- **Deployment Guide**: `docs/deployment_guide.md` (30KB)
- **API Documentation**: `/docs` endpoint (interactive Swagger UI)
- **Model Registry**: `src/deployment/README.md`

### Code Artifacts
- **Model Server**: `src/deployment/model_server_template.py` (25KB)
- **Registry Manager**: `src/deployment/model_registry_manager.py` (16KB)
- **Example Client**: `src/deployment/example_client.py` (13KB)
- **Requirements**: `src/deployment/requirements.txt`
- **Dockerfile**: `src/deployment/Dockerfile`
- **Docker Compose**: `src/deployment/docker-compose.yml`
- **Kubernetes**: `src/deployment/kubernetes-deployment.yaml`

### External Resources
- TensorFlow Serving: https://www.tensorflow.org/tfx/guide/serving
- FastAPI: https://fastapi.tiangolo.com/
- Prometheus: https://prometheus.io/docs/
- Kubernetes: https://kubernetes.io/docs/

---

## Appendix A: API Request Examples

### Single Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, ..., 0.784],
    "model": "auto",
    "return_probabilities": true
  }'
```

### Multi-Model Comparison
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, ..., 0.784],
    "return_all_models": true
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [[0.1, ...], [0.2, ...], [0.3, ...]],
    "model": "keras_cnn"
  }'
```

---

## Appendix B: Performance Baselines

| Model | p50 Latency | p95 Latency | Throughput | Memory |
|-------|-------------|-------------|------------|--------|
| **Keras CNN** | 80ms | 180ms | 12 RPS | 800MB |
| **XGBoost** | 15ms | 40ms | 35 RPS | 300MB |
| **Random Forest** | 25ms | 70ms | 25 RPS | 500MB |

**Test Conditions**:
- Hardware: 4 CPU cores, 8GB RAM
- Batch size: 1
- Features: 784
- Python 3.11, Linux x86_64

---

## Contact & Support

**Architecture Team**: System Architecture Team
**Documentation**: `docs/deployment_guide.md`
**Issues**: GitHub Issues
**Last Updated**: 2025-10-05

---

**End of Architecture Summary**
