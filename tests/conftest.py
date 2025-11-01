"""
Pytest configuration and shared fixtures for test suite
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory fixture"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def sample_data():
    """Generate sample Kepler data for testing"""
    n_samples = 1000
    n_features = 3197

    # Generate synthetic data similar to Kepler dataset
    data = {
        'LABEL': np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.3, 0.2])
    }

    # Generate flux features
    for i in range(1, n_features + 1):
        data[f'FLUX.{i}'] = np.random.randn(n_samples)

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_train_test_split(sample_data):
    """Generate train/test split"""
    from sklearn.model_selection import train_test_split

    X = sample_data.drop('LABEL', axis=1).values
    y = sample_data['LABEL'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def encoded_labels(sample_train_test_split):
    """Generate one-hot encoded labels"""
    from tensorflow.keras.utils import to_categorical

    _, _, y_train, y_test = sample_train_test_split

    # Adjust labels to 0-indexed
    y_train_encoded = to_categorical(y_train - 1, num_classes=3)
    y_test_encoded = to_categorical(y_test - 1, num_classes=3)

    return y_train_encoded, y_test_encoded


@pytest.fixture(scope="function")
def simple_model():
    """Create a simple test model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3197,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


@pytest.fixture(scope="function")
def trained_model(simple_model, sample_train_test_split, encoded_labels):
    """Create a trained model for testing"""
    X_train, _, _, _ = sample_train_test_split
    y_train_encoded, _ = encoded_labels

    # Quick training for testing purposes
    simple_model.fit(
        X_train, y_train_encoded,
        epochs=2,
        batch_size=32,
        verbose=0
    )

    return simple_model


@pytest.fixture(scope="function")
def temp_model_path(tmp_path):
    """Temporary directory for model saving/loading tests"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir / "test_model.h5")


@pytest.fixture(autouse=True)
def reset_tensorflow():
    """Reset TensorFlow session after each test"""
    yield
    tf.keras.backend.clear_session()


# Performance testing configuration
@pytest.fixture(scope="session")
def performance_config():
    """Configuration for performance tests"""
    return {
        'max_inference_latency_ms': 100,
        'min_throughput_samples_per_sec': 1000,
        'batch_sizes': [1, 16, 32, 64, 128],
        'num_iterations': 100
    }


# Compatibility testing configuration
@pytest.fixture(scope="session")
def compatibility_config():
    """Configuration for compatibility tests"""
    return {
        'supported_tf_versions': ['2.10', '2.11', '2.12', '2.13', '2.14', '2.15'],
        'python_version': '3.9+',
        'required_packages': ['numpy', 'pandas', 'scikit-learn', 'tensorflow']
    }
