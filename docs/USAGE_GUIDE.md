# Kepler Exoplanet Detection - Complete Usage Guide

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Training Models](#training-models)
3. [Making Predictions](#making-predictions)
4. [API Server](#api-server)
5. [Model Files](#model-files)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r scripts/requirements_api.txt
```

### Step 2: Train Models (if not already trained)

```bash
python scripts/train_models.py
```

This will train all 3 models:
- XGBoost (~15 seconds)
- Random Forest (~11 seconds)
- Genesis CNN (~5-10 minutes)

### Step 3: Make Predictions

```bash
python scripts/predict.py
```

---

## 🎯 Training Models

### Command Line Training

```bash
cd "C:\Users\thc1006\Desktop\新增資料夾\colab_notebook"
python scripts/train_models.py
```

### What Gets Created

After training, you'll have:

```
models/
├── feature_imputer.pkl        # Missing value imputer
├── feature_scaler.pkl         # Feature scaler (StandardScaler)
├── xgboost_3class.json        # XGBoost model (2.7MB)
├── random_forest_3class.pkl   # Random Forest (13MB)
├── genesis_cnn_3class.keras   # Keras CNN (~5MB)
├── ensemble_voting_3class.pkl # Ensemble model (~15MB)
└── metadata.json              # Performance metrics

figures/
├── confusion_matrices.png     # Confusion matrices for all models
└── performance_comparison.png # Performance bar charts
```

### Training Parameters

Edit `scripts/train_models.py` to customize:

```python
# SMOTE parameters
smote = SMOTE(random_state=42)

# XGBoost parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=200,      # Number of trees
    max_depth=8,           # Tree depth
    learning_rate=0.1      # Learning rate
)

# Random Forest parameters
rf_model = RandomForestClassifier(
    n_estimators=300,      # Number of trees
    max_depth=20          # Tree depth
)

# CNN parameters
epochs=50                 # Maximum epochs
batch_size=32            # Batch size
```

---

## 🔮 Making Predictions

### Option 1: Command Line Script

```bash
python scripts/predict.py
```

### Option 2: Python Code

```python
import numpy as np
import joblib
import xgboost as xgb

# Load preprocessors
imputer = joblib.load('models/feature_imputer.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Load model (choose one)
# Option A: XGBoost
booster = xgb.Booster()
booster.load_model('models/xgboost_3class.json')

# Option B: Random Forest
# model = joblib.load('models/random_forest_3class.pkl')

# Option C: Ensemble
# model = joblib.load('models/ensemble_voting_3class.pkl')

# Prepare your features (782 features)
your_features = np.random.randn(782)  # Replace with real data

# Preprocess
features_imputed = imputer.transform([your_features])
features_scaled = scaler.transform(features_imputed)

# Predict with XGBoost
dmat = xgb.DMatrix(features_scaled)
probabilities = booster.predict(dmat)[0]

# Get prediction
predicted_class = int(np.argmax(probabilities))
label_mapping = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}

print(f"Prediction: {label_mapping[predicted_class]}")
print(f"Confidence: {probabilities[predicted_class]:.4f}")
print(f"Probabilities: {probabilities}")
```

---

## 🌐 API Server

### Start the Server

```bash
# Install Flask if needed
pip install flask flask-cors

# Start server
python scripts/serve_model.py

# Or with custom settings
python scripts/serve_model.py --host 0.0.0.0 --port 8080
```

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": ["imputer", "scaler", "xgboost", "random_forest", "ensemble"]
}
```

#### 2. List Models

```bash
curl http://localhost:5000/models
```

Response:
```json
{
  "models": ["xgboost", "random_forest", "cnn", "ensemble"],
  "default": "ensemble"
}
```

#### 3. Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.123, 0.456, ...],  // 782 features
    "model": "xgboost"
  }'
```

Response:
```json
{
  "model_used": "xgboost",
  "predicted_class": "CONFIRMED",
  "confidence": 0.8523,
  "probabilities": {
    "CANDIDATE": 0.0234,
    "CONFIRMED": 0.8523,
    "FALSE POSITIVE": 0.1243
  }
}
```

#### 4. Batch Prediction

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[...], [...], [...]],  // Multiple samples
    "model": "ensemble"
  }'
```

Response:
```json
{
  "model_used": "ensemble",
  "count": 3,
  "predictions": [
    {
      "predicted_class": "CONFIRMED",
      "confidence": 0.8523,
      "probabilities": {...}
    },
    ...
  ]
}
```

### Test the API

```bash
# Start server in one terminal
python scripts/serve_model.py

# Run tests in another terminal
python scripts/test_api.py
```

---

## 📦 Model Files

### File Formats and Usage

| File | Format | Size | How to Load |
|------|--------|------|-------------|
| `feature_imputer.pkl` | Joblib | 6.6KB | `joblib.load()` |
| `feature_scaler.pkl` | Joblib | 19KB | `joblib.load()` |
| `xgboost_3class.json` | JSON | 2.7MB | `xgb.Booster().load_model()` |
| `random_forest_3class.pkl` | Joblib | 13MB | `joblib.load()` |
| `genesis_cnn_3class.keras` | Keras | ~5MB | `tf.keras.models.load_model()` |
| `ensemble_voting_3class.pkl` | Joblib | ~15MB | `joblib.load()` |

### Label Mapping

```python
{
  "0": "CANDIDATE",
  "1": "CONFIRMED",
  "2": "FALSE POSITIVE"
}
```

### Metadata Structure

`models/metadata.json`:
```json
{
  "created_at": "2025-10-05T21:16:39",
  "label_mapping": {...},
  "num_classes": 3,
  "train_samples": 8922,
  "test_samples": 2014,
  "feature_dim": 782,
  "models": {
    "xgboost": {
      "accuracy": 0.2969,
      "f1_score": 0.2460,
      "training_time_sec": 14.76,
      "file": "xgboost_3class.json"
    },
    ...
  }
}
```

---

## 🔧 Troubleshooting

### Issue 1: Models Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'models/...'`

**Solution**:
```bash
# Make sure you're in the project directory
cd "C:\Users\thc1006\Desktop\新增資料夾\colab_notebook"

# Train models first
python scripts/train_models.py
```

### Issue 2: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
pip install -r scripts/requirements_api.txt
```

### Issue 3: Wrong Feature Count

**Error**: `Expected 782 features, got 100`

**Solution**:
Your input must have exactly 782 features (after ID column is removed). Check your data preprocessing.

### Issue 4: API Connection Error

**Error**: `ConnectionError: Could not connect to API`

**Solution**:
```bash
# Make sure server is running
python scripts/serve_model.py

# Check if port 5000 is available
netstat -an | findstr 5000
```

### Issue 5: Low Accuracy

**Current Status**: Models show ~30% accuracy

**Reasons**:
1. Data alignment issue (8054 labels vs 1866 feature samples)
2. Feature engineering needed
3. Class imbalance handling

**Next Steps**:
1. Verify data alignment in preprocessing
2. Add more feature engineering
3. Try different class balancing techniques

---

## 📝 Examples

### Example 1: Load and Use XGBoost

```python
from predict import load_models, predict_all
import numpy as np

# Load all models
models = load_models('models')

# Create sample features
features = np.random.randn(782)

# Predict with all models
results = predict_all(features, models)

# Print results
for model_name, result in results.items():
    print(f"{model_name}: {result['class']} ({result['confidence']:.2%})")
```

### Example 2: Use API from Python

```python
import requests
import numpy as np

# Generate features
features = np.random.randn(782).tolist()

# Make request
response = requests.post('http://localhost:5000/predict', json={
    'features': features,
    'model': 'ensemble'
})

# Parse result
result = response.json()
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Example 3: Batch Processing

```python
import pandas as pd
import requests

# Load your data
df = pd.read_csv('your_data.csv')

# Extract features (assuming 782 columns)
features = df.iloc[:, :782].values.tolist()

# Batch predict
response = requests.post('http://localhost:5000/predict/batch', json={
    'features': features,
    'model': 'ensemble'
})

# Save results
results = response.json()['predictions']
df['prediction'] = [r['predicted_class'] for r in results]
df['confidence'] = [r['confidence'] for r in results]

df.to_csv('predictions.csv', index=False)
```

---

## 🚀 Production Deployment

### Docker Deployment (Future)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install -r requirements_api.txt

COPY models/ models/
COPY scripts/serve_model.py .

EXPOSE 5000

CMD ["python", "serve_model.py", "--host", "0.0.0.0", "--port", "5000"]
```

### Cloud Deployment Options

1. **AWS SageMaker**: Deploy with SageMaker endpoints
2. **Google Cloud AI Platform**: Use Vertex AI
3. **Azure ML**: Deploy as Azure ML endpoint
4. **Heroku**: Simple deployment with Procfile

---

## 📊 Performance Benchmarks

Based on initial training:

| Model | Accuracy | F1-Score | Training Time | Inference Time |
|-------|----------|----------|---------------|----------------|
| XGBoost | 29.69% | 24.60% | 14.8s | ~5ms |
| Random Forest | 29.59% | 23.91% | 11.1s | ~10ms |
| Genesis CNN | TBD | TBD | ~5-10min | ~50ms |
| Ensemble | TBD | TBD | - | ~60ms |

**Note**: Accuracy will improve with proper data alignment and feature engineering.

---

## 📞 Support

For issues or questions:
1. Check this guide
2. Review `models/metadata.json` for model performance
3. Check training logs
4. Verify data preprocessing

---

**Last Updated**: 2025-10-05
**Version**: 1.0.0
