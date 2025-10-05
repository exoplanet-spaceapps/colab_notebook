#!/usr/bin/env python
"""
Test script for Kepler Exoplanet Detection API
"""

import requests
import numpy as np
import json

BASE_URL = 'http://localhost:5000'

def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 80)
    print("TEST 1: Health Check")
    print("=" * 80)

    response = requests.get(f'{BASE_URL}/health')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_list_models():
    """Test models listing"""
    print("\n" + "=" * 80)
    print("TEST 2: List Available Models")
    print("=" * 80)

    response = requests.get(f'{BASE_URL}/models')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_single_prediction():
    """Test single prediction"""
    print("\n" + "=" * 80)
    print("TEST 3: Single Prediction")
    print("=" * 80)

    # Generate random features (782 features)
    features = np.random.randn(782).tolist()

    payload = {
        'features': features,
        'model': 'xgboost'
    }

    response = requests.post(f'{BASE_URL}/predict', json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "=" * 80)
    print("TEST 4: Batch Prediction (5 samples)")
    print("=" * 80)

    # Generate random features for 5 samples
    features = np.random.randn(5, 782).tolist()

    payload = {
        'features': features,
        'model': 'random_forest'
    }

    response = requests.post(f'{BASE_URL}/predict/batch', json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Model Used: {result['model_used']}")
        print(f"Count: {result['count']}")
        print(f"\nFirst prediction:")
        print(json.dumps(result['predictions'][0], indent=2))
    else:
        print(f"Error: {response.text}")

def test_error_handling():
    """Test error handling"""
    print("\n" + "=" * 80)
    print("TEST 5: Error Handling (Invalid Input)")
    print("=" * 80)

    # Test with wrong number of features
    payload = {
        'features': [1, 2, 3]  # Only 3 features instead of 782
    }

    response = requests.post(f'{BASE_URL}/predict', json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def main():
    """Run all tests"""
    print("=" * 80)
    print("Kepler Exoplanet Detection API - Test Suite")
    print("=" * 80)
    print(f"\nTesting API at: {BASE_URL}")

    try:
        test_health()
        test_list_models()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to API at {BASE_URL}")
        print("Make sure the server is running: python scripts/serve_model.py")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == '__main__':
    main()
