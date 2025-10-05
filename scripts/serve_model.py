#!/usr/bin/env python
"""
Simple Flask API Server for Kepler Exoplanet Detection
Serves trained models via REST API
"""

import os
import sys
import json
import numpy as np
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global models storage
MODELS = {}

def load_all_models(models_dir='models'):
    """Load all trained models"""
    global MODELS

    print("Loading models...")

    # Load preprocessors
    MODELS['imputer'] = joblib.load(os.path.join(models_dir, 'feature_imputer.pkl'))
    MODELS['scaler'] = joblib.load(os.path.join(models_dir, 'feature_scaler.pkl'))

    # Load XGBoost
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(os.path.join(models_dir, 'xgboost_3class.json'))
    MODELS['xgboost'] = xgb_booster

    # Load Random Forest
    MODELS['random_forest'] = joblib.load(os.path.join(models_dir, 'random_forest_3class.pkl'))

    # Load Ensemble if available
    ensemble_path = os.path.join(models_dir, 'ensemble_voting_3class.pkl')
    if os.path.exists(ensemble_path):
        MODELS['ensemble'] = joblib.load(ensemble_path)

    # Load CNN if available
    cnn_path = os.path.join(models_dir, 'genesis_cnn_3class.keras')
    if os.path.exists(cnn_path):
        try:
            import tensorflow as tf
            MODELS['cnn'] = tf.keras.models.load_model(cnn_path)
        except Exception as e:
            print(f"Warning: Could not load CNN: {e}")

    # Load metadata
    metadata_path = os.path.join(models_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            MODELS['metadata'] = json.load(f)

    print(f"Loaded {len(MODELS)} components")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(MODELS.keys())
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    available_models = []

    if 'xgboost' in MODELS:
        available_models.append('xgboost')
    if 'random_forest' in MODELS:
        available_models.append('random_forest')
    if 'cnn' in MODELS:
        available_models.append('cnn')
    if 'ensemble' in MODELS:
        available_models.append('ensemble')

    return jsonify({
        'models': available_models,
        'default': 'ensemble' if 'ensemble' in MODELS else 'xgboost'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction endpoint"""
    try:
        data = request.get_json()

        # Validate input
        if 'features' not in data:
            return jsonify({'error': 'Missing features'}), 400

        features = np.array(data['features'])

        if features.shape[0] != 782:
            return jsonify({
                'error': f'Expected 782 features, got {features.shape[0]}'
            }), 400

        # Get model preference
        model_name = data.get('model', 'ensemble' if 'ensemble' in MODELS else 'xgboost')

        # Preprocess
        features_imputed = MODELS['imputer'].transform([features])
        features_scaled = MODELS['scaler'].transform(features_imputed)

        # Get label mapping
        label_mapping = MODELS.get('metadata', {}).get('label_mapping', {
            '0': 'CANDIDATE', '1': 'CONFIRMED', '2': 'FALSE POSITIVE'
        })

        # Make prediction
        if model_name == 'xgboost' and 'xgboost' in MODELS:
            dmat = xgb.DMatrix(features_scaled)
            probabilities = MODELS['xgboost'].predict(dmat)[0]
        elif model_name == 'random_forest' and 'random_forest' in MODELS:
            probabilities = MODELS['random_forest'].predict_proba(features_scaled)[0]
        elif model_name == 'cnn' and 'cnn' in MODELS:
            probabilities = MODELS['cnn'].predict(features_scaled, verbose=0)[0]
        elif model_name == 'ensemble' and 'ensemble' in MODELS:
            probabilities = MODELS['ensemble'].predict_proba(features_scaled)[0]
        else:
            return jsonify({'error': f'Model {model_name} not available'}), 404

        predicted_class = int(np.argmax(probabilities))

        # Format response
        response = {
            'model_used': model_name,
            'predicted_class': label_mapping[str(predicted_class)],
            'confidence': float(probabilities[predicted_class]),
            'probabilities': {
                label_mapping['0']: float(probabilities[0]),
                label_mapping['1']: float(probabilities[1]),
                label_mapping['2']: float(probabilities[2])
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()

        if 'features' not in data:
            return jsonify({'error': 'Missing features'}), 400

        features_list = np.array(data['features'])
        model_name = data.get('model', 'ensemble' if 'ensemble' in MODELS else 'xgboost')

        # Preprocess all
        features_imputed = MODELS['imputer'].transform(features_list)
        features_scaled = MODELS['scaler'].transform(features_imputed)

        label_mapping = MODELS.get('metadata', {}).get('label_mapping', {
            '0': 'CANDIDATE', '1': 'CONFIRMED', '2': 'FALSE POSITIVE'
        })

        # Batch prediction
        if model_name == 'xgboost' and 'xgboost' in MODELS:
            dmat = xgb.DMatrix(features_scaled)
            all_probabilities = MODELS['xgboost'].predict(dmat)
        elif model_name == 'random_forest' and 'random_forest' in MODELS:
            all_probabilities = MODELS['random_forest'].predict_proba(features_scaled)
        elif model_name == 'cnn' and 'cnn' in MODELS:
            all_probabilities = MODELS['cnn'].predict(features_scaled, verbose=0)
        elif model_name == 'ensemble' and 'ensemble' in MODELS:
            all_probabilities = MODELS['ensemble'].predict_proba(features_scaled)
        else:
            return jsonify({'error': f'Model {model_name} not available'}), 404

        # Format results
        results = []
        for probabilities in all_probabilities:
            predicted_class = int(np.argmax(probabilities))
            results.append({
                'predicted_class': label_mapping[str(predicted_class)],
                'confidence': float(probabilities[predicted_class]),
                'probabilities': {
                    label_mapping['0']: float(probabilities[0]),
                    label_mapping['1']: float(probabilities[1]),
                    label_mapping['2']: float(probabilities[2])
                }
            })

        return jsonify({
            'model_used': model_name,
            'count': len(results),
            'predictions': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Kepler Exoplanet Model Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--models-dir', default='models', help='Directory containing models')

    args = parser.parse_args()

    # Load models
    load_all_models(args.models_dir)

    # Start server
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"API Endpoints:")
    print(f"  GET  /health - Health check")
    print(f"  GET  /models - List available models")
    print(f"  POST /predict - Single prediction")
    print(f"  POST /predict/batch - Batch prediction")
    print(f"\nPress Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)
