"""
Test data generation utilities

This module provides utilities for generating synthetic test data
that mimics the Kepler exoplanet dataset structure.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_kepler_sample(n_samples=1000, n_features=3197, random_seed=42):
    """
    Generate synthetic Kepler-like data for testing

    Args:
        n_samples: Number of samples to generate
        n_features: Number of flux features (default: 3197)
        random_seed: Random seed for reproducibility

    Returns:
        pandas.DataFrame with LABEL and FLUX columns
    """
    np.random.seed(random_seed)

    # Generate class labels with imbalanced distribution
    # Class 1: 50%, Class 2: 30%, Class 3: 20%
    labels = np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.3, 0.2])

    # Initialize DataFrame
    data = {'LABEL': labels}

    # Generate flux features with class-dependent patterns
    for i in range(1, n_features + 1):
        # Add some class-dependent signal
        base_signal = np.random.randn(n_samples)

        # Class 1: No significant pattern
        # Class 2: Slight positive bias
        # Class 3: Periodic pattern
        class_signal = np.where(
            labels == 1,
            base_signal,
            np.where(
                labels == 2,
                base_signal + 0.3,
                base_signal + 0.5 * np.sin(i / 100)
            )
        )

        data[f'FLUX.{i}'] = class_signal

    return pd.DataFrame(data)


def generate_edge_case_data():
    """
    Generate edge case test data

    Returns:
        dict of edge case datasets
    """
    edge_cases = {}

    # All zeros
    edge_cases['all_zeros'] = pd.DataFrame({
        'LABEL': [1, 2, 3],
        **{f'FLUX.{i}': [0.0, 0.0, 0.0] for i in range(1, 3198)}
    })

    # All ones
    edge_cases['all_ones'] = pd.DataFrame({
        'LABEL': [1, 2, 3],
        **{f'FLUX.{i}': [1.0, 1.0, 1.0] for i in range(1, 3198)}
    })

    # Extreme values
    edge_cases['extreme_positive'] = pd.DataFrame({
        'LABEL': [1, 2, 3],
        **{f'FLUX.{i}': [1000.0, 1000.0, 1000.0] for i in range(1, 3198)}
    })

    edge_cases['extreme_negative'] = pd.DataFrame({
        'LABEL': [1, 2, 3],
        **{f'FLUX.{i}': [-1000.0, -1000.0, -1000.0] for i in range(1, 3198)}
    })

    # Mixed values
    np.random.seed(42)
    edge_cases['mixed_extreme'] = pd.DataFrame({
        'LABEL': [1, 2, 3],
        **{f'FLUX.{i}': np.random.randn(3) * 1000 for i in range(1, 3198)}
    })

    # Highly correlated features
    base = np.random.randn(3)
    edge_cases['correlated'] = pd.DataFrame({
        'LABEL': [1, 2, 3],
        **{f'FLUX.{i}': base + np.random.randn(3) * 0.01 for i in range(1, 3198)}
    })

    return edge_cases


def generate_imbalanced_data(n_samples=1000):
    """
    Generate highly imbalanced dataset

    Args:
        n_samples: Total number of samples

    Returns:
        pandas.DataFrame with severe class imbalance
    """
    np.random.seed(42)

    # Severe imbalance: 90% class 1, 8% class 2, 2% class 3
    labels = np.random.choice([1, 2, 3], size=n_samples, p=[0.90, 0.08, 0.02])

    data = {'LABEL': labels}

    for i in range(1, 3198):
        data[f'FLUX.{i}'] = np.random.randn(n_samples)

    return pd.DataFrame(data)


def generate_minimal_data():
    """
    Generate minimal dataset for quick tests

    Returns:
        pandas.DataFrame with 10 samples
    """
    return generate_kepler_sample(n_samples=10, random_seed=42)


def generate_large_data(n_samples=10000):
    """
    Generate large dataset for performance testing

    Args:
        n_samples: Number of samples

    Returns:
        pandas.DataFrame with many samples
    """
    return generate_kepler_sample(n_samples=n_samples, random_seed=42)


def save_test_data(output_dir=None):
    """
    Save all test datasets to files

    Args:
        output_dir: Directory to save files (default: current directory)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main sample data
    sample_data = generate_kepler_sample()
    sample_data.to_csv(output_dir / 'sample_data.csv', index=False)
    print(f"Saved sample_data.csv ({len(sample_data)} samples)")

    # Save edge cases
    edge_dir = output_dir / 'edge_cases'
    edge_dir.mkdir(exist_ok=True)

    edge_cases = generate_edge_case_data()
    for name, data in edge_cases.items():
        filepath = edge_dir / f'{name}.csv'
        data.to_csv(filepath, index=False)
        print(f"Saved {name}.csv ({len(data)} samples)")

    # Save imbalanced data
    imbalanced = generate_imbalanced_data()
    imbalanced.to_csv(output_dir / 'imbalanced_data.csv', index=False)
    print(f"Saved imbalanced_data.csv ({len(imbalanced)} samples)")

    # Save minimal data
    minimal = generate_minimal_data()
    minimal.to_csv(output_dir / 'minimal_data.csv', index=False)
    print(f"Saved minimal_data.csv ({len(minimal)} samples)")

    print(f"\nAll test data saved to {output_dir}")


if __name__ == '__main__':
    save_test_data()
