#!/usr/bin/env python
"""
Kepler Exoplanet Prediction Script
Load trained models and make predictions on new data
"""

import os
import sys
import json
import numpy as np
import joblib
import xgboost as xgb

def load_models(models_dir='models'):
    """Load all trained models and preprocessors"""
    models = {}

    # Load preprocessors
    print("Loading preprocessors...")
    models['imputer'] = joblib.load(os.path.join(models_dir, 'feature_imputer.pkl'))
    models['scaler'] = joblib.load(os.path.join(models_dir, 'feature_scaler.pkl'))

    # Load XGBoost
    print("Loading XGBoost model...")
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(os.path.join(models_dir, 'xgboost_3class.json'))
    models['xgboost'] = xgb_booster

    # Load Random Forest
    print("Loading Random Forest model...")
    models['random_forest'] = joblib.load(os.path.join(models_dir, 'random_forest_3class.pkl'))

    # Load Ensemble if available
    ensemble_path = os.path.join(models_dir, 'ensemble_voting_3class.pkl')
    if os.path.exists(ensemble_path):
        print("Loading Ensemble model...")
        models['ensemble'] = joblib.load(ensemble_path)

    # Load CNN if available
    cnn_path = os.path.join(models_dir, 'genesis_cnn_3class.keras')
    if os.path.exists(cnn_path):
        print("Loading Genesis CNN model...")
        try:
            import tensorflow as tf
            models['cnn'] = tf.keras.models.load_model(cnn_path)
        except Exception as e:
            print(f"  Warning: Could not load CNN model: {e}")

    # Load metadata
    metadata_path = os.path.join(models_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            models['metadata'] = json.load(f)

    print(f"\n✓ Loaded {len(models)} components")
    return models

def preprocess_features(features, imputer, scaler):
    """Preprocess raw features"""
    # Handle missing values
    features_imputed = imputer.transform([features])
    # Scale features
    features_scaled = scaler.transform(features_imputed)
    return features_scaled

def predict_xgboost(features_scaled, model):
    """Predict using XGBoost"""
    dmat = xgb.DMatrix(features_scaled)
    probabilities = model.predict(dmat)[0]
    predicted_class = int(np.argmax(probabilities))
    return predicted_class, probabilities

def predict_random_forest(features_scaled, model):
    """Predict using Random Forest"""
    probabilities = model.predict_proba(features_scaled)[0]
    predicted_class = int(np.argmax(probabilities))
    return predicted_class, probabilities

def predict_cnn(features_scaled, model):
    """Predict using CNN"""
    probabilities = model.predict(features_scaled, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    return predicted_class, probabilities

def predict_ensemble(features_scaled, model):
    """Predict using Ensemble"""
    probabilities = model.predict_proba(features_scaled)[0]
    predicted_class = int(np.argmax(probabilities))
    return predicted_class, probabilities

def predict_all(features, models):
    """Make predictions with all available models"""
    # Preprocess
    features_scaled = preprocess_features(features, models['imputer'], models['scaler'])

    results = {}
    label_mapping = models.get('metadata', {}).get('label_mapping', {
        '0': 'CANDIDATE', '1': 'CONFIRMED', '2': 'FALSE POSITIVE'
    })

    # XGBoost
    if 'xgboost' in models:
        pred_class, probs = predict_xgboost(features_scaled, models['xgboost'])
        results['xgboost'] = {
            'class': label_mapping[str(pred_class)],
            'probabilities': {
                label_mapping['0']: float(probs[0]),
                label_mapping['1']: float(probs[1]),
                label_mapping['2']: float(probs[2])
            },
            'confidence': float(probs[pred_class])
        }

    # Random Forest
    if 'random_forest' in models:
        pred_class, probs = predict_random_forest(features_scaled, models['random_forest'])
        results['random_forest'] = {
            'class': label_mapping[str(pred_class)],
            'probabilities': {
                label_mapping['0']: float(probs[0]),
                label_mapping['1']: float(probs[1]),
                label_mapping['2']: float(probs[2])
            },
            'confidence': float(probs[pred_class])
        }

    # CNN
    if 'cnn' in models:
        pred_class, probs = predict_cnn(features_scaled, models['cnn'])
        results['cnn'] = {
            'class': label_mapping[str(pred_class)],
            'probabilities': {
                label_mapping['0']: float(probs[0]),
                label_mapping['1']: float(probs[1]),
                label_mapping['2']: float(probs[2])
            },
            'confidence': float(probs[pred_class])
        }

    # Ensemble
    if 'ensemble' in models:
        pred_class, probs = predict_ensemble(features_scaled, models['ensemble'])
        results['ensemble'] = {
            'class': label_mapping[str(pred_class)],
            'probabilities': {
                label_mapping['0']: float(probs[0]),
                label_mapping['1']: float(probs[1]),
                label_mapping['2']: float(probs[2])
            },
            'confidence': float(probs[pred_class])
        }

    return results

def main():
    """Main prediction function"""
    print("=" * 80)
    print("Kepler Exoplanet Prediction")
    print("=" * 80)

    # Load models
    models = load_models()

    print("\n" + "=" * 80)
    print("Example Prediction (using random features)")
    print("=" * 80)

    # Generate random features for demonstration (783 features after removing ID column)
    random_features = np.random.randn(782)

    print(f"\nInput features shape: {random_features.shape}")
    print("\nMaking predictions with all models...")

    # Predict
    results = predict_all(random_features, models)

    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)

    for model_name, result in results.items():
        print(f"\n[{model_name.upper()}]")
        print(f"  Predicted Class: {result['class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"    {class_name}: {prob:.4f}")

    # Voting result
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("ENSEMBLE VOTING (Simple Majority)")
        print("=" * 80)

        votes = {}
        for model_name, result in results.items():
            predicted_class = result['class']
            votes[predicted_class] = votes.get(predicted_class, 0) + 1

        final_prediction = max(votes, key=votes.get)
        print(f"\nFinal Prediction: {final_prediction}")
        print(f"Votes: {votes}")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
