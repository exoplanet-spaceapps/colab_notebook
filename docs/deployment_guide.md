# Kepler Exoplanet Detection - Model Deployment Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Model Deployment Requirements](#model-deployment-requirements)
3. [Unified Prediction API](#unified-prediction-api)
4. [Model Version Management](#model-version-management)
5. [Deployment Checklist](#deployment-checklist)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Production Best Practices](#production-best-practices)

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│         (Web UI / Mobile App / API Consumers)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ HTTPS/REST
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway / Load Balancer                 │
│              (NGINX / AWS ALB / GCP Load Balancer)           │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│  FastAPI   │ │  FastAPI   │ │  FastAPI   │
│  Instance  │ │  Instance  │ │  Instance  │
│   (Pod 1)  │ │   (Pod 2)  │ │   (Pod 3)  │
└──────┬─────┘ └──────┬─────┘ └──────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  Model Registry │       │  Redis Cache    │
│   (MinIO/S3)    │       │  (Model Cache)  │
└─────────────────┘       └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│          Model Storage Structure             │
├─────────────────────────────────────────────┤
│  models/                                     │
│  ├── keras_cnn/                             │
│  │   ├── v1.0.0/                            │
│  │   │   ├── model.keras                    │
│  │   │   ├── metadata.json                  │
│  │   │   └── preprocessing.pkl              │
│  │   ├── v1.1.0/                            │
│  │   └── latest -> v1.1.0                   │
│  ├── xgboost/                               │
│  │   ├── v1.0.0/                            │
│  │   │   ├── model.json                     │
│  │   │   ├── metadata.json                  │
│  │   │   └── preprocessing.pkl              │
│  │   └── latest -> v1.0.0                   │
│  └── random_forest/                         │
│      ├── v1.0.0/                            │
│      │   ├── model.pkl                      │
│      │   ├── metadata.json                  │
│      │   └── preprocessing.pkl              │
│      └── latest -> v1.0.0                   │
└─────────────────────────────────────────────┘
```

---

## Model Deployment Requirements

### 1. Keras CNN Model

#### File Format & Structure
- **Primary Format**: `.keras` (TensorFlow SavedModel format)
- **Alternative**: `.h5` (HDF5 format, legacy)
- **Recommended**: Use `.keras` for TensorFlow 2.13+

#### Loading Mechanism
```python
import tensorflow as tf
from tensorflow import keras

# Load model
model = keras.models.load_model('path/to/model.keras')

# Verify model structure
model.summary()

# Test inference
predictions = model.predict(input_data)
```

#### Serving Options

**Option A: TensorFlow Serving (Production-grade)**
```bash
# Docker deployment
docker run -p 8501:8501 \
  --mount type=bind,source=/models/keras_cnn,target=/models/keras_cnn \
  -e MODEL_NAME=keras_cnn \
  tensorflow/serving

# REST API endpoint
curl -X POST http://localhost:8501/v1/models/keras_cnn:predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[feature_vector]]}'
```

**Option B: FastAPI + Keras (Flexible)**
```python
from fastapi import FastAPI
import tensorflow as tf
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model('model.keras')

@app.post("/predict")
async def predict(features: List[float]):
    input_data = np.array([features]).reshape(1, -1, 1)
    predictions = model.predict(input_data)
    return {"predictions": predictions.tolist()}
```

#### Resource Requirements
- **Memory**: 500MB - 1GB per instance
- **CPU**: 2-4 cores recommended
- **GPU**: Optional (CUDA 11.8+, cuDNN 8.6+)
- **Latency**: 50-200ms per request (CPU), 10-50ms (GPU)

#### Dependencies
```txt
tensorflow==2.15.0
keras==2.15.0
numpy==1.26.0
h5py==3.10.0  # for .h5 format support
```

---

### 2. XGBoost Model

#### File Format & Structure
- **Primary Format**: `.json` (human-readable, portable)
- **Alternative**: `.ubj` (Universal Binary JSON, faster loading)
- **Legacy**: `.model` (deprecated)

#### Loading Mechanism
```python
import xgboost as xgb
import numpy as np

# Load model
booster = xgb.Booster()
booster.load_model('model.json')

# Alternative: load .ubj format
# booster.load_model('model.ubj')

# Create DMatrix for prediction
dmatrix = xgb.DMatrix(input_data)
predictions = booster.predict(dmatrix)
```

#### Serving Options

**FastAPI + XGBoost**
```python
from fastapi import FastAPI
import xgboost as xgb
import numpy as np

app = FastAPI()
model = xgb.Booster()
model.load_model('model.json')

@app.post("/predict")
async def predict(features: List[float]):
    dmatrix = xgb.DMatrix(np.array([features]))
    predictions = model.predict(dmatrix)
    return {
        "predictions": predictions.tolist(),
        "class": int(np.argmax(predictions)),
        "confidence": float(np.max(predictions))
    }
```

#### Resource Requirements
- **Memory**: 100MB - 500MB per instance
- **CPU**: 2-4 cores recommended
- **GPU**: Optional (CUDA support via tree_method='gpu_hist')
- **Latency**: 10-50ms per request

#### Dependencies
```txt
xgboost==2.0.3
numpy==1.26.0
scipy==1.11.0
```

---

### 3. Random Forest Model

#### File Format & Structure
- **Primary Format**: `.pkl` (joblib/pickle)
- **Alternative**: `.joblib` (preferred for large arrays)
- **ONNX Export**: `.onnx` (for cross-platform deployment)

#### Loading Mechanism
```python
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load model
model = joblib.load('model.pkl')

# Verify model properties
print(f"Number of estimators: {model.n_estimators}")
print(f"Feature count: {model.n_features_in_}")

# Prediction
predictions = model.predict(input_data)
probabilities = model.predict_proba(input_data)
```

#### Serving Options

**FastAPI + scikit-learn**
```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model.pkl')

@app.post("/predict")
async def predict(features: List[float]):
    input_array = np.array([features])
    predictions = model.predict(input_array)
    probabilities = model.predict_proba(input_array)

    return {
        "prediction": int(predictions[0]),
        "probabilities": probabilities[0].tolist(),
        "confidence": float(np.max(probabilities))
    }
```

#### Resource Requirements
- **Memory**: 200MB - 1GB per instance (depends on n_estimators)
- **CPU**: 4-8 cores recommended (parallel prediction)
- **GPU**: Not supported
- **Latency**: 20-100ms per request

#### Dependencies
```txt
scikit-learn==1.4.0
joblib==1.3.2
numpy==1.26.0
```

---

## Unified Prediction API

### API Design

#### Endpoint Structure
```
POST /api/v1/predict
POST /api/v1/predict/batch
POST /api/v1/models/{model_name}/predict
GET  /api/v1/models
GET  /api/v1/models/{model_name}/info
GET  /api/v1/health
```

#### Request Schema
```json
{
  "features": [0.123, 0.456, ...],  // 784 float values
  "model": "auto",                   // auto | keras_cnn | xgboost | random_forest
  "return_probabilities": true,
  "return_all_models": false         // compare all models
}
```

#### Response Schema
```json
{
  "prediction": {
    "class": "CONFIRMED",
    "class_id": 0,
    "confidence": 0.92,
    "probabilities": {
      "CONFIRMED": 0.92,
      "CANDIDATE": 0.05,
      "FALSE_POSITIVE": 0.03
    }
  },
  "model": {
    "name": "keras_cnn",
    "version": "1.0.0",
    "latency_ms": 45
  },
  "metadata": {
    "timestamp": "2025-10-05T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

#### Multi-Model Comparison Response
```json
{
  "ensemble_prediction": {
    "class": "CONFIRMED",
    "confidence": 0.89,
    "voting": "soft"  // soft | hard
  },
  "individual_predictions": {
    "keras_cnn": {
      "class": "CONFIRMED",
      "confidence": 0.92,
      "latency_ms": 45
    },
    "xgboost": {
      "class": "CONFIRMED",
      "confidence": 0.87,
      "latency_ms": 12
    },
    "random_forest": {
      "class": "CANDIDATE",
      "confidence": 0.78,
      "latency_ms": 28
    }
  },
  "agreement_score": 0.67  // 2/3 models agree
}
```

---

## Model Version Management

### Version Control Strategy

#### Semantic Versioning
- **Format**: `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking changes in input/output schema
- **MINOR**: New features, improved accuracy
- **PATCH**: Bug fixes, performance improvements

#### Directory Structure
```
models/
├── keras_cnn/
│   ├── v1.0.0/
│   │   ├── model.keras
│   │   ├── metadata.json
│   │   ├── preprocessing.pkl
│   │   ├── performance_metrics.json
│   │   └── checksum.sha256
│   ├── v1.1.0/
│   │   └── ...
│   ├── latest -> v1.1.0
│   └── stable -> v1.0.0
├── xgboost/
│   └── ...
└── random_forest/
    └── ...
```

#### Metadata Schema
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
    "validation_samples": 466,
    "epochs": 50,
    "batch_size": 32
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
    "dtype": "float32",
    "shape": [1, 784, 1]
  },
  "output_schema": {
    "classes": 3,
    "labels": ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
  },
  "dependencies": {
    "tensorflow": "2.15.0",
    "numpy": "1.26.0"
  },
  "checksum": {
    "algorithm": "sha256",
    "value": "abc123def456..."
  }
}
```

---

### Model Registry

#### Registry Database Schema
```sql
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    metadata JSONB NOT NULL,
    status VARCHAR(20) NOT NULL,  -- active, deprecated, archived
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_name, version)
);

CREATE TABLE deployment_history (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id),
    environment VARCHAR(50) NOT NULL,  -- dev, staging, production
    deployed_at TIMESTAMP DEFAULT NOW(),
    deployed_by VARCHAR(100) NOT NULL,
    rollback_at TIMESTAMP NULL,
    notes TEXT
);

CREATE TABLE prediction_logs (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id),
    request_id VARCHAR(100) NOT NULL,
    input_features JSONB NOT NULL,
    prediction JSONB NOT NULL,
    latency_ms INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

---

### A/B Testing Framework

#### Traffic Splitting
```python
class ABTestRouter:
    def __init__(self, model_a: str, model_b: str, split_ratio: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio

    def route_request(self, request_id: str) -> str:
        """Route request to model A or B based on hash"""
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return self.model_a if (hash_value % 100) < (self.split_ratio * 100) else self.model_b

# Usage
router = ABTestRouter("keras_cnn_v1.0.0", "keras_cnn_v1.1.0", split_ratio=0.1)
model_version = router.route_request(request_id)
```

#### Performance Tracking
```python
@dataclass
class ABTestMetrics:
    model_a_requests: int
    model_b_requests: int
    model_a_avg_latency: float
    model_b_avg_latency: float
    model_a_accuracy: float  # requires ground truth
    model_b_accuracy: float

    @property
    def performance_delta(self) -> Dict[str, float]:
        return {
            "latency_improvement": (self.model_a_avg_latency - self.model_b_avg_latency) / self.model_a_avg_latency,
            "accuracy_improvement": self.model_b_accuracy - self.model_a_accuracy
        }
```

---

### Rollback Mechanism

#### Automatic Rollback Triggers
1. **Error Rate Threshold**: > 5% error rate in 5 minutes
2. **Latency Degradation**: > 2x baseline latency
3. **Prediction Drift**: Distribution shift > 0.3 KL divergence
4. **Health Check Failure**: 3 consecutive failures

#### Rollback Procedure
```python
class ModelDeploymentManager:
    def rollback(self, model_name: str, target_version: str = "stable"):
        """Rollback to stable version"""
        # 1. Stop routing to current version
        self.traffic_manager.set_weight(model_name, "current", 0.0)

        # 2. Warm up target version
        self.model_loader.preload(model_name, target_version)

        # 3. Gradual traffic migration
        for weight in [0.1, 0.3, 0.5, 0.7, 1.0]:
            self.traffic_manager.set_weight(model_name, target_version, weight)
            time.sleep(30)  # Monitor for 30s
            if self.health_check(model_name, target_version):
                continue
            else:
                raise RollbackFailed("Health check failed during rollback")

        # 4. Update registry
        self.registry.set_active_version(model_name, target_version)

        # 5. Log event
        self.audit_log.record_rollback(model_name, target_version)
```

---

## Deployment Checklist

### Pre-Deployment

#### 1. Model File Integrity
```bash
#!/bin/bash
# check_model_integrity.sh

MODEL_PATH=$1
CHECKSUM_FILE="${MODEL_PATH}/checksum.sha256"

# Verify checksum
if [ -f "$CHECKSUM_FILE" ]; then
    cd "$MODEL_PATH"
    sha256sum -c checksum.sha256
    if [ $? -eq 0 ]; then
        echo "✓ Checksum verification passed"
    else
        echo "✗ Checksum verification failed"
        exit 1
    fi
else
    echo "⚠ No checksum file found"
fi

# Verify file existence
for file in model.* metadata.json preprocessing.pkl; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "✓ Found $file"
    else
        echo "✗ Missing $file"
        exit 1
    fi
done
```

#### 2. Dependency Compatibility
```python
# check_dependencies.py
import sys
import importlib
import json

def check_dependencies(metadata_path: str) -> bool:
    with open(metadata_path) as f:
        metadata = json.load(f)

    dependencies = metadata.get('dependencies', {})
    issues = []

    for package, required_version in dependencies.items():
        try:
            module = importlib.import_module(package)
            installed_version = getattr(module, '__version__', 'unknown')

            if installed_version != required_version:
                issues.append(f"{package}: required {required_version}, got {installed_version}")
        except ImportError:
            issues.append(f"{package}: not installed")

    if issues:
        print("✗ Dependency issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All dependencies satisfied")
        return True
```

#### 3. Model Loading Test
```python
# test_model_loading.py
import time
import json

def test_model_loading(model_path: str, model_type: str) -> Dict:
    """Test model can be loaded and perform inference"""

    start_time = time.time()

    if model_type == 'keras':
        import tensorflow as tf
        model = tf.keras.models.load_model(f"{model_path}/model.keras")
        test_input = tf.random.normal([1, 784, 1])
        predictions = model.predict(test_input)

    elif model_type == 'xgboost':
        import xgboost as xgb
        import numpy as np
        model = xgb.Booster()
        model.load_model(f"{model_path}/model.json")
        test_input = xgb.DMatrix(np.random.randn(1, 784))
        predictions = model.predict(test_input)

    elif model_type == 'random_forest':
        import joblib
        import numpy as np
        model = joblib.load(f"{model_path}/model.pkl")
        test_input = np.random.randn(1, 784)
        predictions = model.predict(test_input)

    load_time = time.time() - start_time

    return {
        "status": "success",
        "load_time_seconds": load_time,
        "output_shape": predictions.shape
    }
```

#### 4. Performance Benchmark
```python
# benchmark_inference.py
import time
import numpy as np
from statistics import mean, stdev

def benchmark_model(model, model_type: str, num_samples: int = 100):
    """Benchmark inference performance"""

    latencies = []

    for _ in range(num_samples):
        # Generate random input
        if model_type == 'keras':
            test_input = np.random.randn(1, 784, 1).astype(np.float32)
        else:
            test_input = np.random.randn(1, 784)

        # Measure inference time
        start = time.perf_counter()

        if model_type == 'keras':
            _ = model.predict(test_input, verbose=0)
        elif model_type == 'xgboost':
            import xgboost as xgb
            dmatrix = xgb.DMatrix(test_input)
            _ = model.predict(dmatrix)
        else:  # random_forest
            _ = model.predict(test_input)

        latency = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(latency)

    return {
        "samples": num_samples,
        "mean_latency_ms": mean(latencies),
        "std_latency_ms": stdev(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "throughput_rps": 1000 / mean(latencies)
    }
```

---

### Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Model Preparation
- [ ] Model files verified with checksums
- [ ] Metadata.json created with all required fields
- [ ] Preprocessing artifacts included
- [ ] Model version incremented correctly
- [ ] Performance metrics documented

### Environment Setup
- [ ] Dependencies installed and verified
- [ ] Python version matches training environment
- [ ] GPU drivers updated (if using GPU)
- [ ] Sufficient disk space (>10GB recommended)
- [ ] Sufficient memory (>4GB recommended)

### Testing
- [ ] Model loads successfully
- [ ] Inference test passes
- [ ] Performance benchmark completed
- [ ] Latency within acceptable range (<200ms)
- [ ] Memory usage acceptable (<2GB)

### Infrastructure
- [ ] Docker image built and tested
- [ ] Kubernetes manifests validated
- [ ] Health check endpoints configured
- [ ] Monitoring dashboards created
- [ ] Alerting rules configured

### Security
- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] Input validation implemented
- [ ] Secrets management configured
- [ ] Network policies applied

### Documentation
- [ ] API documentation updated
- [ ] Deployment runbook created
- [ ] Rollback procedure documented
- [ ] Monitoring guide completed
- [ ] Troubleshooting guide updated

## Deployment Steps
1. [ ] Deploy to staging environment
2. [ ] Run integration tests
3. [ ] Perform load testing
4. [ ] Deploy to production (canary 10%)
5. [ ] Monitor metrics for 30 minutes
6. [ ] Increase to 50% traffic
7. [ ] Monitor metrics for 30 minutes
8. [ ] Complete rollout to 100%
9. [ ] Monitor for 24 hours

## Post-Deployment
- [ ] Verify all health checks passing
- [ ] Confirm metrics are being collected
- [ ] Test rollback procedure
- [ ] Update model registry
- [ ] Notify stakeholders
```

---

## Performance Benchmarks

### Baseline Performance Targets

| Metric | Keras CNN | XGBoost | Random Forest |
|--------|-----------|---------|---------------|
| **Latency (p50)** | <100ms | <30ms | <50ms |
| **Latency (p95)** | <200ms | <60ms | <100ms |
| **Throughput** | >10 RPS | >30 RPS | >20 RPS |
| **Memory** | <1GB | <500MB | <800MB |
| **CPU Usage** | <80% | <60% | <70% |

### Load Testing Scenarios

```python
# locustfile.py
from locust import HttpUser, task, between
import numpy as np

class ModelUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_exoplanet(self):
        features = np.random.randn(784).tolist()
        self.client.post("/api/v1/predict", json={
            "features": features,
            "model": "auto"
        })

# Run load test
# locust -f locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10
```

---

## Production Best Practices

### 1. Model Caching
```python
from functools import lru_cache
import tensorflow as tf

class ModelCache:
    def __init__(self, cache_size: int = 3):
        self._cache = {}
        self._cache_size = cache_size

    def get_or_load(self, model_path: str, model_type: str):
        if model_path in self._cache:
            return self._cache[model_path]

        # Load model
        if model_type == 'keras':
            model = tf.keras.models.load_model(model_path)
        # ... other types

        # Add to cache
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[model_path] = model
        return model
```

### 2. Request Batching
```python
import asyncio
from typing import List

class BatchPredictor:
    def __init__(self, model, batch_size: int = 32, max_wait_ms: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()

    async def predict(self, features: np.ndarray):
        """Add request to batch queue"""
        future = asyncio.Future()
        await self.queue.put((features, future))
        return await future

    async def batch_processor(self):
        """Process requests in batches"""
        while True:
            batch = []
            futures = []

            # Collect batch
            start_time = asyncio.get_event_loop().time()
            while len(batch) < self.batch_size:
                timeout = (self.max_wait_ms / 1000) - (asyncio.get_event_loop().time() - start_time)
                if timeout <= 0:
                    break

                try:
                    features, future = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(features)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break

            if batch:
                # Batch prediction
                batch_input = np.array(batch)
                predictions = self.model.predict(batch_input)

                # Return results
                for future, pred in zip(futures, predictions):
                    future.set_result(pred)
```

### 3. Monitoring & Observability
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model', 'status'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency', ['model'])
model_load_time = Gauge('model_load_time_seconds', 'Model load time', ['model'])

def monitored_predict(model, features, model_name: str):
    """Wrapped prediction with monitoring"""
    start_time = time.time()

    try:
        result = model.predict(features)
        prediction_counter.labels(model=model_name, status='success').inc()
        return result
    except Exception as e:
        prediction_counter.labels(model=model_name, status='error').inc()
        raise
    finally:
        latency = time.time() - start_time
        prediction_latency.labels(model=model_name).observe(latency)
```

### 4. Circuit Breaker
```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

---

## Troubleshooting Guide

### Common Issues

#### Issue 1: High Latency
**Symptoms**: p95 latency > 500ms

**Diagnosis**:
```bash
# Check CPU usage
top -p $(pgrep -f "python.*model_server")

# Check memory usage
free -h

# Profile inference
python -m cProfile -o profile.stats inference_test.py
```

**Solutions**:
- Enable model caching
- Implement request batching
- Use GPU acceleration
- Optimize preprocessing pipeline

#### Issue 2: Memory Leaks
**Symptoms**: Memory usage increases over time

**Diagnosis**:
```python
import tracemalloc
tracemalloc.start()

# ... run predictions ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

**Solutions**:
- Clear TensorFlow session after predictions
- Use context managers for model loading
- Implement proper garbage collection

#### Issue 3: Model Version Conflicts
**Symptoms**: Prediction errors after deployment

**Diagnosis**:
```bash
# Check model metadata
cat models/keras_cnn/v1.0.0/metadata.json

# Verify dependencies
pip list | grep -E "(tensorflow|xgboost|scikit-learn)"
```

**Solutions**:
- Use virtual environments per model version
- Pin exact dependency versions
- Implement schema validation

---

## Appendix

### A. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.deployment.model_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### B. Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kepler-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kepler-model-server
  template:
    metadata:
      labels:
        app: kepler-model-server
    spec:
      containers:
      - name: model-server
        image: kepler-model-server:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: kepler-model-service
spec:
  selector:
    app: kepler-model-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-05
**Maintained By**: System Architecture Team
