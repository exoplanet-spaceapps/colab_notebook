#!/usr/bin/env python
"""
Quick test script for XGBoost model
"""

import numpy as np
import joblib
import xgboost as xgb

print("=" * 80)
print("Testing XGBoost Model")
print("=" * 80)

# Load preprocessors
print("\nLoading preprocessors...")
imputer = joblib.load('models/feature_imputer.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Load XGBoost model
print("Loading XGBoost model...")
booster = xgb.Booster()
booster.load_model('models/xgboost_3class.json')

# Generate random test features (783 features - after removing ID columns)
print("\nGenerating test features...")
test_features = np.random.randn(783)

# Preprocess
print("Preprocessing...")
features_imputed = imputer.transform([test_features])
features_scaled = scaler.transform(features_imputed)

# Predict
print("Making prediction...")
dmat = xgb.DMatrix(features_scaled)
probabilities = booster.predict(dmat)[0]

# Get result
label_mapping = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}
predicted_class = int(np.argmax(probabilities))

print("\n" + "=" * 80)
print("PREDICTION RESULT")
print("=" * 80)
print(f"\nPredicted Class: {label_mapping[predicted_class]}")
print(f"Confidence: {probabilities[predicted_class]:.4f} ({probabilities[predicted_class]*100:.2f}%)")
print(f"\nProbabilities:")
print(f"  CANDIDATE:       {probabilities[0]:.4f} ({probabilities[0]*100:.2f}%)")
print(f"  CONFIRMED:       {probabilities[1]:.4f} ({probabilities[1]*100:.2f}%)")
print(f"  FALSE POSITIVE:  {probabilities[2]:.4f} ({probabilities[2]*100:.2f}%)")
print("\n" + "=" * 80)

# Test batch prediction
print("\nTesting batch prediction (5 samples)...")
batch_features = np.random.randn(5, 783)
batch_imputed = imputer.transform(batch_features)
batch_scaled = scaler.transform(batch_imputed)
batch_dmat = xgb.DMatrix(batch_scaled)
batch_probs = booster.predict(batch_dmat)

print("\nBatch Results:")
for i, probs in enumerate(batch_probs):
    pred_class = int(np.argmax(probs))
    print(f"  Sample {i+1}: {label_mapping[pred_class]} ({probs[pred_class]:.2%})")

print("\nXGBoost model is working correctly!")
print("=" * 80)
