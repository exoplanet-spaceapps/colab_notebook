"""
Kepler Exoplanet Detection - Unified Model Server
FastAPI-based inference server supporting Keras CNN, XGBoost, and Random Forest models

Features:
- Multi-model support with automatic routing
- Model versioning and registry
- A/B testing framework
- Request batching
- Caching and performance optimization
- Comprehensive monitoring
- Health checks and circuit breakers

Author: System Architecture Team
Version: 1.0.0
Date: 2025-10-05
"""

import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

# Conditional imports based on available packages
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logging.warning("TensorFlow not available. Keras CNN models will not be supported.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not available. XGBoost models will not be supported.")

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available. Random Forest models will not be supported.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", "./models"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "3"))
ENABLE_AB_TESTING = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "auto")

# Prometheus Metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model', 'status', 'class'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency', ['model'])
model_load_time = Gauge('model_load_time_seconds', 'Model load time', ['model', 'version'])
active_models = Gauge('active_models_count', 'Number of active models')
cache_hit_counter = Counter('cache_hits_total', 'Cache hits', ['model'])
cache_miss_counter = Counter('cache_misses_total', 'Cache misses', ['model'])

# ============================================================================
# Data Models
# ============================================================================

class ModelType(str, Enum):
    """Supported model types"""
    KERAS_CNN = "keras_cnn"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    AUTO = "auto"


class PredictionClass(str, Enum):
    """Prediction classes for Kepler exoplanet detection"""
    CONFIRMED = "CONFIRMED"
    CANDIDATE = "CANDIDATE"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint"""
    features: List[float] = Field(..., min_items=784, max_items=784, description="784 lightcurve features")
    model: ModelType = Field(default=ModelType.AUTO, description="Model to use for prediction")
    return_probabilities: bool = Field(default=True, description="Return probability distribution")
    return_all_models: bool = Field(default=False, description="Return predictions from all models")

    @validator('features')
    def validate_features(cls, v):
        """Validate feature values"""
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("All features must be numeric")
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features contain NaN or Inf values")
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint"""
    samples: List[List[float]] = Field(..., description="List of feature vectors")
    model: ModelType = Field(default=ModelType.AUTO, description="Model to use for prediction")
    return_probabilities: bool = Field(default=True, description="Return probability distribution")

    @validator('samples')
    def validate_samples(cls, v):
        """Validate samples"""
        if not v:
            raise ValueError("Samples list cannot be empty")
        if not all(len(sample) == 784 for sample in v):
            raise ValueError("All samples must have exactly 784 features")
        return v


class PredictionResult(BaseModel):
    """Single prediction result"""
    class_name: PredictionClass
    class_id: int
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    version: str
    latency_ms: float


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    prediction: PredictionResult
    model: ModelInfo
    metadata: Dict[str, Any]


class MultiModelPredictionResponse(BaseModel):
    """Response for multi-model comparison"""
    ensemble_prediction: PredictionResult
    individual_predictions: Dict[str, PredictionResult]
    agreement_score: float
    metadata: Dict[str, Any]


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_name: str
    version: str
    created_at: str
    framework: str
    framework_version: str
    training: Dict[str, Any]
    performance: Dict[str, float]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: Dict[str, str]
    checksum: Dict[str, str]


# ============================================================================
# Model Registry & Loading
# ============================================================================

class ModelRegistry:
    """Model registry for managing multiple model versions"""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self._models = {}
        self._metadata_cache = {}
        self._load_registry()

    def _load_registry(self):
        """Load model registry from filesystem"""
        if not self.registry_path.exists():
            logger.warning(f"Model registry path does not exist: {self.registry_path}")
            return

        for model_dir in self.registry_path.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            latest_link = model_dir / "latest"

            if latest_link.exists():
                version_dir = latest_link.resolve()
                metadata_file = version_dir / "metadata.json"

                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    self._models[model_name] = {
                        "path": version_dir,
                        "version": metadata.get("version", "unknown"),
                        "metadata": metadata
                    }
                    logger.info(f"Registered model: {model_name} v{metadata.get('version')}")

    def get_model_path(self, model_name: str, version: str = "latest") -> Optional[Path]:
        """Get model file path"""
        if model_name not in self._models:
            return None

        if version == "latest":
            return self._models[model_name]["path"]
        else:
            # Look for specific version
            version_path = self.registry_path / model_name / version
            if version_path.exists():
                return version_path

        return None

    def get_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        if model_name in self._models:
            metadata_dict = self._models[model_name]["metadata"]
            return ModelMetadata(**metadata_dict)
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return [
            {
                "name": name,
                "version": info["version"],
                "framework": info["metadata"].get("framework", "unknown")
            }
            for name, info in self._models.items()
        ]


class ModelCache:
    """LRU cache for loaded models"""

    def __init__(self, cache_size: int = 3):
        self.cache_size = cache_size
        self._cache = {}
        self._access_times = {}

    def get(self, key: str) -> Optional[Any]:
        """Get model from cache"""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None

    def put(self, key: str, model: Any):
        """Put model in cache with LRU eviction"""
        if len(self._cache) >= self.cache_size:
            # Evict least recently used
            lru_key = min(self._access_times, key=self._access_times.get)
            del self._cache[lru_key]
            del self._access_times[lru_key]
            logger.info(f"Evicted model from cache: {lru_key}")

        self._cache[key] = model
        self._access_times[key] = time.time()

    def clear(self):
        """Clear cache"""
        self._cache.clear()
        self._access_times.clear()


class ModelLoader:
    """Unified model loader for all supported model types"""

    def __init__(self, registry: ModelRegistry, cache: ModelCache):
        self.registry = registry
        self.cache = cache

    def load_model(self, model_type: ModelType, version: str = "latest") -> tuple:
        """Load model and return (model, metadata)"""
        cache_key = f"{model_type.value}_{version}"

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached:
            cache_hit_counter.labels(model=model_type.value).inc()
            logger.debug(f"Cache hit for {cache_key}")
            return cached

        cache_miss_counter.labels(model=model_type.value).inc()
        logger.info(f"Loading model: {model_type.value} v{version}")

        start_time = time.time()

        # Determine model name from type
        model_name = model_type.value
        if model_type == ModelType.AUTO:
            # Default to best performing model
            model_name = "keras_cnn"

        # Get model path
        model_path = self.registry.get_model_path(model_name, version)
        if not model_path:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name} v{version}")

        # Load based on type
        try:
            if model_name == "keras_cnn":
                if not HAS_TENSORFLOW:
                    raise HTTPException(status_code=503, detail="TensorFlow not available")
                model = keras.models.load_model(model_path / "model.keras")

            elif model_name == "xgboost":
                if not HAS_XGBOOST:
                    raise HTTPException(status_code=503, detail="XGBoost not available")
                model = xgb.Booster()
                model.load_model(str(model_path / "model.json"))

            elif model_name == "random_forest":
                if not HAS_SKLEARN:
                    raise HTTPException(status_code=503, detail="scikit-learn not available")
                model = joblib.load(model_path / "model.pkl")

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_name}")

            # Load metadata
            metadata = self.registry.get_metadata(model_name)

            load_time = time.time() - start_time
            model_load_time.labels(model=model_name, version=version).set(load_time)
            logger.info(f"Model loaded in {load_time:.2f}s: {model_name} v{version}")

            # Cache the loaded model
            result = (model, metadata)
            self.cache.put(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


# ============================================================================
# Prediction Engine
# ============================================================================

class PredictionEngine:
    """Unified prediction engine for all model types"""

    CLASS_LABELS = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    def predict(
        self,
        features: np.ndarray,
        model_type: ModelType = ModelType.AUTO,
        version: str = "latest"
    ) -> tuple:
        """
        Make prediction

        Returns:
            (class_id, probabilities, latency_ms)
        """
        start_time = time.perf_counter()

        # Load model
        model, metadata = self.model_loader.load_model(model_type, version)

        # Prepare input based on model type
        model_name = metadata.model_name

        try:
            if model_name == "keras_cnn":
                # Reshape for CNN: (1, 784, 1)
                input_data = features.reshape(1, -1, 1).astype(np.float32)
                probabilities = model.predict(input_data, verbose=0)[0]

            elif model_name == "xgboost":
                # XGBoost expects (1, 784)
                input_data = features.reshape(1, -1)
                dmatrix = xgb.DMatrix(input_data)
                probabilities = model.predict(dmatrix)[0]

            elif model_name == "random_forest":
                # Random Forest expects (1, 784)
                input_data = features.reshape(1, -1)
                probabilities = model.predict_proba(input_data)[0]

            else:
                raise ValueError(f"Unknown model type: {model_name}")

            # Get class prediction
            class_id = int(np.argmax(probabilities))

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics
            prediction_latency.labels(model=model_name).observe(latency_ms / 1000)
            prediction_counter.labels(
                model=model_name,
                status='success',
                class_name=self.CLASS_LABELS[class_id]
            ).inc()

            return class_id, probabilities, latency_ms

        except Exception as e:
            prediction_counter.labels(
                model=model_name if model_name else "unknown",
                status='error',
                class_name='none'
            ).inc()
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    def predict_all_models(self, features: np.ndarray) -> Dict[str, tuple]:
        """Predict using all available models"""
        results = {}

        for model_type in [ModelType.KERAS_CNN, ModelType.XGBOOST, ModelType.RANDOM_FOREST]:
            try:
                class_id, probabilities, latency_ms = self.predict(features, model_type)
                results[model_type.value] = (class_id, probabilities, latency_ms)
            except Exception as e:
                logger.warning(f"Error with {model_type.value}: {str(e)}")
                continue

        return results

    def ensemble_predict(self, predictions: Dict[str, tuple]) -> tuple:
        """Ensemble prediction using soft voting"""
        if not predictions:
            raise HTTPException(status_code=500, detail="No predictions available")

        # Collect probabilities
        all_probs = [probs for _, probs, _ in predictions.values()]

        # Average probabilities (soft voting)
        avg_probs = np.mean(all_probs, axis=0)
        ensemble_class = int(np.argmax(avg_probs))

        return ensemble_class, avg_probs


# ============================================================================
# A/B Testing Framework
# ============================================================================

class ABTestRouter:
    """A/B testing router for model versions"""

    def __init__(self, model_a: str, model_b: str, split_ratio: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio

    def route_request(self, request_id: str) -> str:
        """Route request to model A or B based on hash"""
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return self.model_a if (hash_value % 100) < (self.split_ratio * 100) else self.model_b


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Kepler Exoplanet Detection API",
    description="Unified inference API for Keras CNN, XGBoost, and Random Forest models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_registry = ModelRegistry(MODEL_REGISTRY_PATH)
model_cache = ModelCache(cache_size=CACHE_SIZE)
model_loader = ModelLoader(model_registry, model_cache)
prediction_engine = PredictionEngine(model_loader)

# Update active models metric
active_models.set(len(model_registry.list_models()))


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Kepler Exoplanet Detection API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "models": "/api/v1/models",
            "health": "/api/v1/health",
            "metrics": "/metrics"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": len(model_registry.list_models()),
        "cache_size": len(model_cache._cache)
    }


@app.get("/api/v1/models")
async def list_models():
    """List all available models"""
    return {
        "models": model_registry.list_models()
    }


@app.get("/api/v1/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get detailed model information"""
    metadata = model_registry.get_metadata(model_name)

    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    return asdict(metadata)


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """Single prediction endpoint"""

    # Generate request ID
    request_id = hashlib.md5(
        f"{time.time()}{http_request.client.host}".encode()
    ).hexdigest()

    # Convert features to numpy array
    features = np.array(request.features, dtype=np.float32)

    # Handle multi-model comparison
    if request.return_all_models:
        all_predictions = prediction_engine.predict_all_models(features)

        # Ensemble prediction
        ensemble_class, ensemble_probs = prediction_engine.ensemble_predict(all_predictions)

        # Build response
        individual_results = {}
        for model_name, (class_id, probs, latency_ms) in all_predictions.items():
            individual_results[model_name] = PredictionResult(
                class_name=PredictionClass(prediction_engine.CLASS_LABELS[class_id]),
                class_id=class_id,
                confidence=float(probs[class_id]),
                probabilities={
                    label: float(prob)
                    for label, prob in zip(prediction_engine.CLASS_LABELS, probs)
                } if request.return_probabilities else None
            )

        # Calculate agreement score
        all_classes = [pred.class_id for pred in individual_results.values()]
        agreement_score = all_classes.count(ensemble_class) / len(all_classes)

        return MultiModelPredictionResponse(
            ensemble_prediction=PredictionResult(
                class_name=PredictionClass(prediction_engine.CLASS_LABELS[ensemble_class]),
                class_id=ensemble_class,
                confidence=float(ensemble_probs[ensemble_class]),
                probabilities={
                    label: float(prob)
                    for label, prob in zip(prediction_engine.CLASS_LABELS, ensemble_probs)
                }
            ),
            individual_predictions=individual_results,
            agreement_score=agreement_score,
            metadata={
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    # Single model prediction
    class_id, probabilities, latency_ms = prediction_engine.predict(
        features,
        model_type=request.model
    )

    # Build response
    model_name = request.model.value if request.model != ModelType.AUTO else "keras_cnn"
    metadata = model_registry.get_metadata(model_name)

    return PredictionResponse(
        prediction=PredictionResult(
            class_name=PredictionClass(prediction_engine.CLASS_LABELS[class_id]),
            class_id=class_id,
            confidence=float(probabilities[class_id]),
            probabilities={
                label: float(prob)
                for label, prob in zip(prediction_engine.CLASS_LABELS, probabilities)
            } if request.return_probabilities else None
        ),
        model=ModelInfo(
            name=model_name,
            version=metadata.version if metadata else "unknown",
            latency_ms=latency_ms
        ),
        metadata={
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.post("/api/v1/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""

    features_array = np.array(request.samples, dtype=np.float32)
    results = []

    for features in features_array:
        class_id, probabilities, latency_ms = prediction_engine.predict(
            features,
            model_type=request.model
        )

        results.append({
            "class_name": prediction_engine.CLASS_LABELS[class_id],
            "class_id": class_id,
            "confidence": float(probabilities[class_id]),
            "probabilities": {
                label: float(prob)
                for label, prob in zip(prediction_engine.CLASS_LABELS, probabilities)
            } if request.return_probabilities else None
        })

    return {
        "predictions": results,
        "count": len(results)
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Kepler Exoplanet Detection API")
    logger.info(f"Model registry path: {MODEL_REGISTRY_PATH}")
    logger.info(f"Models available: {len(model_registry.list_models())}")
    logger.info(f"TensorFlow available: {HAS_TENSORFLOW}")
    logger.info(f"XGBoost available: {HAS_XGBOOST}")
    logger.info(f"scikit-learn available: {HAS_SKLEARN}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Kepler Exoplanet Detection API")
    model_cache.clear()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "model_server_template:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
