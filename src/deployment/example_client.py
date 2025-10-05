"""
Example Client for Kepler Exoplanet Detection API

Demonstrates various API usage patterns:
- Single predictions
- Batch predictions
- Multi-model comparison
- Model selection
- Error handling

Author: System Architecture Team
Version: 1.0.0
Date: 2025-10-05
"""

import requests
import numpy as np
import time
from typing import List, Dict, Any


class KeplerAPIClient:
    """Client for Kepler Exoplanet Detection API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()

    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        response = self.session.get(f"{self.base_url}/api/v1/models")
        response.raise_for_status()
        return response.json()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/api/v1/models/{model_name}/info")
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        features: List[float],
        model: str = "auto",
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Make single prediction"""
        payload = {
            "features": features,
            "model": model,
            "return_probabilities": return_probabilities
        }

        response = self.session.post(
            f"{self.base_url}/api/v1/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def predict_all_models(
        self,
        features: List[float],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Predict using all models (ensemble)"""
        payload = {
            "features": features,
            "return_probabilities": return_probabilities,
            "return_all_models": True
        }

        response = self.session.post(
            f"{self.base_url}/api/v1/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def batch_predict(
        self,
        samples: List[List[float]],
        model: str = "auto",
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Make batch predictions"""
        payload = {
            "samples": samples,
            "model": model,
            "return_probabilities": return_probabilities
        }

        response = self.session.post(
            f"{self.base_url}/api/v1/predict/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def example_1_single_prediction():
    """Example 1: Single prediction with automatic model selection"""
    print("\n=== Example 1: Single Prediction (Auto Model) ===")

    client = KeplerAPIClient()

    # Check API health
    health = client.health_check()
    print(f"API Status: {health['status']}")

    # Generate random features (784 values)
    features = np.random.randn(784).tolist()

    # Make prediction
    result = client.predict(features)

    print(f"\nPrediction: {result['prediction']['class_name']}")
    print(f"Confidence: {result['prediction']['confidence']:.2%}")
    print(f"Model: {result['model']['name']} v{result['model']['version']}")
    print(f"Latency: {result['model']['latency_ms']:.1f}ms")

    if result['prediction']['probabilities']:
        print("\nProbabilities:")
        for class_name, prob in result['prediction']['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")


def example_2_model_selection():
    """Example 2: Prediction with specific model"""
    print("\n=== Example 2: Specific Model Selection ===")

    client = KeplerAPIClient()

    features = np.random.randn(784).tolist()

    # Try different models
    models = ["keras_cnn", "xgboost", "random_forest"]

    for model_name in models:
        try:
            result = client.predict(features, model=model_name)
            print(f"\n{model_name.upper()}:")
            print(f"  Prediction: {result['prediction']['class_name']}")
            print(f"  Confidence: {result['prediction']['confidence']:.2%}")
            print(f"  Latency: {result['model']['latency_ms']:.1f}ms")
        except requests.exceptions.HTTPError as e:
            print(f"\n{model_name.upper()}: Not available ({e})")


def example_3_multi_model_comparison():
    """Example 3: Compare all models (ensemble prediction)"""
    print("\n=== Example 3: Multi-Model Comparison ===")

    client = KeplerAPIClient()

    features = np.random.randn(784).tolist()

    result = client.predict_all_models(features)

    print(f"\nEnsemble Prediction: {result['ensemble_prediction']['class_name']}")
    print(f"Ensemble Confidence: {result['ensemble_prediction']['confidence']:.2%}")
    print(f"Agreement Score: {result['agreement_score']:.2%}")

    print("\nIndividual Model Predictions:")
    for model_name, prediction in result['individual_predictions'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Class: {prediction['class_name']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")


def example_4_batch_prediction():
    """Example 4: Batch prediction"""
    print("\n=== Example 4: Batch Prediction ===")

    client = KeplerAPIClient()

    # Generate 10 random samples
    num_samples = 10
    samples = [np.random.randn(784).tolist() for _ in range(num_samples)]

    start_time = time.time()
    result = client.batch_predict(samples, model="keras_cnn")
    elapsed_time = time.time() - start_time

    print(f"\nProcessed {num_samples} samples in {elapsed_time:.2f}s")
    print(f"Average latency: {elapsed_time / num_samples * 1000:.1f}ms per sample")

    print("\nPredictions:")
    for i, pred in enumerate(result['predictions']):
        print(f"  Sample {i+1}: {pred['class_name']} ({pred['confidence']:.2%})")


def example_5_load_real_data():
    """Example 5: Load and predict on real Kepler data"""
    print("\n=== Example 5: Real Data Prediction ===")

    # This example assumes you have processed data files
    # Adjust paths as needed

    try:
        import pandas as pd

        # Try to load test data
        X_test = pd.read_csv("X_test.csv")

        if len(X_test) == 0:
            print("No test data found. Skipping example.")
            return

        client = KeplerAPIClient()

        # Get first sample
        sample = X_test.iloc[0].values.tolist()

        print(f"Predicting on real Kepler lightcurve data...")

        # Single prediction
        result = client.predict(sample)

        print(f"\nPrediction: {result['prediction']['class_name']}")
        print(f"Confidence: {result['prediction']['confidence']:.2%}")

        # Multi-model comparison
        ensemble_result = client.predict_all_models(sample)

        print(f"\nEnsemble: {ensemble_result['ensemble_prediction']['class_name']}")
        print(f"Agreement: {ensemble_result['agreement_score']:.2%}")

        print("\nModel Breakdown:")
        for model_name, pred in ensemble_result['individual_predictions'].items():
            print(f"  {model_name}: {pred['class_name']} ({pred['confidence']:.2%})")

    except FileNotFoundError:
        print("Test data files not found. Skipping example.")
    except ImportError:
        print("pandas not installed. Skipping example.")


def example_6_model_info():
    """Example 6: Get model information"""
    print("\n=== Example 6: Model Information ===")

    client = KeplerAPIClient()

    # List all models
    models = client.list_models()

    print("\nAvailable Models:")
    for model in models['models']:
        print(f"  - {model['name']} v{model['version']} ({model['framework']})")

    # Get detailed info for each model
    for model in models['models']:
        try:
            info = client.get_model_info(model['name'])

            print(f"\n{model['name'].upper()} Details:")
            print(f"  Version: {info['version']}")
            print(f"  Framework: {info['framework']} {info['framework_version']}")
            print(f"  Created: {info['created_at']}")

            if 'performance' in info:
                print("  Performance:")
                for metric, value in info['performance'].items():
                    print(f"    {metric}: {value:.4f}")

        except requests.exceptions.HTTPError:
            print(f"  (Details not available)")


def example_7_error_handling():
    """Example 7: Error handling"""
    print("\n=== Example 7: Error Handling ===")

    client = KeplerAPIClient()

    # Test 1: Invalid feature count
    print("\nTest 1: Invalid feature count")
    try:
        result = client.predict([1.0, 2.0, 3.0])  # Only 3 features instead of 784
        print("ERROR: Should have raised validation error")
    except requests.exceptions.HTTPError as e:
        print(f"✓ Caught expected error: {e}")

    # Test 2: Invalid model name
    print("\nTest 2: Invalid model name")
    try:
        features = np.random.randn(784).tolist()
        result = client.predict(features, model="nonexistent_model")
        print("ERROR: Should have raised validation error")
    except requests.exceptions.HTTPError as e:
        print(f"✓ Caught expected error: {e}")

    # Test 3: Invalid features (NaN)
    print("\nTest 3: Invalid features (NaN)")
    try:
        features = [float('nan')] * 784
        result = client.predict(features)
        print("ERROR: Should have raised validation error")
    except requests.exceptions.HTTPError as e:
        print(f"✓ Caught expected error: {e}")


def example_8_performance_benchmark():
    """Example 8: Performance benchmarking"""
    print("\n=== Example 8: Performance Benchmark ===")

    client = KeplerAPIClient()

    num_requests = 50
    features = np.random.randn(784).tolist()

    print(f"\nBenchmarking with {num_requests} requests...")

    latencies = []

    for i in range(num_requests):
        start = time.perf_counter()
        result = client.predict(features, model="keras_cnn", return_probabilities=False)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_requests} requests")

    latencies = np.array(latencies)

    print(f"\nPerformance Statistics:")
    print(f"  Mean latency: {latencies.mean():.1f}ms")
    print(f"  Median (p50): {np.percentile(latencies, 50):.1f}ms")
    print(f"  p95 latency: {np.percentile(latencies, 95):.1f}ms")
    print(f"  p99 latency: {np.percentile(latencies, 99):.1f}ms")
    print(f"  Min latency: {latencies.min():.1f}ms")
    print(f"  Max latency: {latencies.max():.1f}ms")
    print(f"  Throughput: {1000 / latencies.mean():.1f} requests/sec")


def main():
    """Run all examples"""
    print("=" * 60)
    print("Kepler Exoplanet Detection API - Example Client")
    print("=" * 60)

    examples = [
        ("Single Prediction", example_1_single_prediction),
        ("Model Selection", example_2_model_selection),
        ("Multi-Model Comparison", example_3_multi_model_comparison),
        ("Batch Prediction", example_4_batch_prediction),
        ("Real Data Prediction", example_5_load_real_data),
        ("Model Information", example_6_model_info),
        ("Error Handling", example_7_error_handling),
        ("Performance Benchmark", example_8_performance_benchmark),
    ]

    # Check if server is running
    try:
        client = KeplerAPIClient()
        health = client.health_check()
        if health['status'] != 'healthy':
            print("\n⚠ Warning: API health check failed!")
            print("Make sure the server is running: python model_server_template.py")
            return
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server at http://localhost:8000")
        print("Please start the server first: python model_server_template.py")
        return

    # Run examples
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in example {i} ({name}): {e}")

        # Pause between examples
        if i < len(examples):
            time.sleep(0.5)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
