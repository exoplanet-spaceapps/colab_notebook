"""
Unified Model I/O System

Provides consistent save/load interfaces for different model types
including Keras, XGBoost, RandomForest, and ensemble models.
"""

import json
import pickle
from pathlib import Path
from typing import Tuple, Any, Dict, Optional
import sys

# Model-specific imports
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .metadata import ModelMetadata, MetadataManager


class ModelIOError(Exception):
    """Custom exception for model I/O operations."""
    pass


class ModelSaver:
    """Handles saving models with metadata."""

    @staticmethod
    def save_model(
        model: Any,
        model_type: str,
        save_dir: Path,
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> Tuple[Path, Path]:
        """
        Save model and metadata with unified interface.

        Args:
            model: Model object to save
            model_type: Type of model ('keras_cnn', 'xgboost', etc.)
            save_dir: Directory to save files
            metadata: ModelMetadata object
            overwrite: Whether to overwrite existing files

        Returns:
            Tuple of (model_path, metadata_path)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Validate metadata
        MetadataManager.validate_metadata(metadata)

        # Generate filename
        base_filename = MetadataManager.get_model_filename(
            model_type, metadata.version
        )

        # Determine file extension and save method
        if model_type.startswith('keras'):
            if not KERAS_AVAILABLE:
                raise ModelIOError("TensorFlow/Keras not available")
            model_path = save_dir / f"{base_filename}.keras"
            ModelSaver._save_keras(model, model_path, overwrite)

        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ModelIOError("XGBoost not available")
            model_path = save_dir / f"{base_filename}.json"
            ModelSaver._save_xgboost(model, model_path, overwrite)

        elif model_type in ['randomforest', 'gradientboosting']:
            if not SKLEARN_AVAILABLE:
                raise ModelIOError("scikit-learn not available")
            model_path = save_dir / f"{base_filename}.pkl"
            ModelSaver._save_sklearn(model, model_path, overwrite)

        elif model_type.startswith('ensemble'):
            if not SKLEARN_AVAILABLE:
                raise ModelIOError("scikit-learn not available")
            model_path = save_dir / f"{base_filename}.pkl"
            ModelSaver._save_ensemble(model, model_path, overwrite)

        else:
            raise ModelIOError(f"Unsupported model type: {model_type}")

        # Compute model hash
        model_hash = MetadataManager.compute_model_hash(model_path)
        metadata.model_hash = model_hash

        # Add framework versions
        if model_type.startswith('keras') and KERAS_AVAILABLE:
            metadata.framework_version = tf.__version__
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            metadata.framework_version = xgb.__version__
        elif SKLEARN_AVAILABLE:
            import sklearn
            metadata.framework_version = sklearn.__version__

        metadata.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Save metadata
        metadata_path = MetadataManager.get_metadata_path(model_path)
        metadata.to_json(metadata_path)

        return model_path, metadata_path

    @staticmethod
    def _save_keras(model: Any, path: Path, overwrite: bool) -> None:
        """Save Keras model."""
        if path.exists() and not overwrite:
            raise ModelIOError(f"File exists: {path}. Use overwrite=True to replace.")
        model.save(path)

    @staticmethod
    def _save_xgboost(model: Any, path: Path, overwrite: bool) -> None:
        """Save XGBoost model."""
        if path.exists() and not overwrite:
            raise ModelIOError(f"File exists: {path}. Use overwrite=True to replace.")
        model.save_model(path)

    @staticmethod
    def _save_sklearn(model: Any, path: Path, overwrite: bool) -> None:
        """Save scikit-learn model."""
        if path.exists() and not overwrite:
            raise ModelIOError(f"File exists: {path}. Use overwrite=True to replace.")
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def _save_ensemble(model: Any, path: Path, overwrite: bool) -> None:
        """Save ensemble model."""
        ModelSaver._save_sklearn(model, path, overwrite)


class ModelLoader:
    """Handles loading models with metadata."""

    @staticmethod
    def load_model(model_path: Path) -> Tuple[Any, ModelMetadata]:
        """
        Load model and metadata with automatic type detection.

        Args:
            model_path: Path to model file

        Returns:
            Tuple of (model, metadata)
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise ModelIOError(f"Model file not found: {model_path}")

        # Load metadata
        metadata_path = MetadataManager.get_metadata_path(model_path)
        if not metadata_path.exists():
            raise ModelIOError(f"Metadata file not found: {metadata_path}")

        metadata = ModelMetadata.from_json(metadata_path)

        # Verify model hash
        current_hash = MetadataManager.compute_model_hash(model_path)
        if metadata.model_hash and current_hash != metadata.model_hash:
            raise ModelIOError(
                f"Model hash mismatch. File may be corrupted or modified. "
                f"Expected: {metadata.model_hash}, Got: {current_hash}"
            )

        # Load model based on type
        model_type = metadata.model_type

        if model_type.startswith('keras'):
            model = ModelLoader._load_keras(model_path)
        elif model_type == 'xgboost':
            model = ModelLoader._load_xgboost(model_path)
        elif model_type in ['randomforest', 'gradientboosting']:
            model = ModelLoader._load_sklearn(model_path)
        elif model_type.startswith('ensemble'):
            model = ModelLoader._load_ensemble(model_path)
        else:
            raise ModelIOError(f"Unsupported model type in metadata: {model_type}")

        return model, metadata

    @staticmethod
    def _load_keras(path: Path) -> Any:
        """Load Keras model."""
        if not KERAS_AVAILABLE:
            raise ModelIOError("TensorFlow/Keras not available")
        return keras.models.load_model(path)

    @staticmethod
    def _load_xgboost(path: Path) -> Any:
        """Load XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ModelIOError("XGBoost not available")
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model

    @staticmethod
    def _load_sklearn(path: Path) -> Any:
        """Load scikit-learn model."""
        if not SKLEARN_AVAILABLE:
            raise ModelIOError("scikit-learn not available")
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _load_ensemble(path: Path) -> Any:
        """Load ensemble model."""
        return ModelLoader._load_sklearn(path)

    @staticmethod
    def load_latest_version(
        model_dir: Path,
        model_type: str
    ) -> Tuple[Any, ModelMetadata]:
        """
        Load the latest version of a model type.

        Args:
            model_dir: Directory containing models
            model_type: Type of model to load

        Returns:
            Tuple of (model, metadata)
        """
        model_dir = Path(model_dir)

        # Find all models of this type
        pattern_map = {
            'keras_cnn': 'genesis_cnn_v*.keras',
            'keras_lstm': 'genesis_lstm_v*.keras',
            'keras_mlp': 'genesis_mlp_v*.keras',
            'xgboost': 'xgboost_v*.json',
            'randomforest': 'randomforest_v*.pkl',
            'gradientboosting': 'gradientboosting_v*.pkl',
            'ensemble_voting': 'ensemble_voting_v*.pkl',
            'ensemble_stacking': 'ensemble_stacking_v*.pkl'
        }

        if model_type not in pattern_map:
            raise ModelIOError(f"Unknown model type: {model_type}")

        pattern = pattern_map[model_type]
        model_files = list(model_dir.glob(pattern))

        if not model_files:
            raise ModelIOError(f"No models found for type: {model_type}")

        # Extract versions and find latest
        versions = []
        for model_file in model_files:
            version = MetadataManager.extract_version_from_filename(model_file.stem)
            if version:
                versions.append((version, model_file))

        if not versions:
            raise ModelIOError(f"No valid versions found for type: {model_type}")

        # Sort by version (assuming semantic versioning)
        versions.sort(key=lambda x: [int(v) for v in x[0].split('.')], reverse=True)
        latest_path = versions[0][1]

        return ModelLoader.load_model(latest_path)


# Convenience functions
def save_model(
    model: Any,
    model_type: str,
    save_dir: Path,
    metadata: ModelMetadata,
    overwrite: bool = False
) -> Tuple[Path, Path]:
    """
    Convenience function to save model.

    See ModelSaver.save_model for documentation.
    """
    return ModelSaver.save_model(model, model_type, save_dir, metadata, overwrite)


def load_model(model_path: Path) -> Tuple[Any, ModelMetadata]:
    """
    Convenience function to load model.

    See ModelLoader.load_model for documentation.
    """
    return ModelLoader.load_model(model_path)


def load_latest_version(
    model_dir: Path,
    model_type: str
) -> Tuple[Any, ModelMetadata]:
    """
    Convenience function to load latest model version.

    See ModelLoader.load_latest_version for documentation.
    """
    return ModelLoader.load_latest_version(model_dir, model_type)
