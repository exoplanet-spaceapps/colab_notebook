"""
Kepler Exoplanet Data Preprocessing Module

This module provides data preprocessing utilities for three-class
exoplanet classification using Kepler mission data.

Main Functions:
    preprocess_kepler_data: Complete preprocessing pipeline
    validate_preprocessed_data: Validation utility

Classes:
    DataPreprocessor: Preprocessing pipeline class
"""

from .data_preprocessing import (
    DataPreprocessor,
    preprocess_kepler_data,
    validate_preprocessed_data
)

__all__ = [
    'DataPreprocessor',
    'preprocess_kepler_data',
    'validate_preprocessed_data'
]

__version__ = '1.0.0'
