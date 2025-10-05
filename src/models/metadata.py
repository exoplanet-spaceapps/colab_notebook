"""
Model Metadata Management System

Handles creation, validation, and persistence of model metadata
for tracking model versions, configurations, and performance metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class ModelMetadata:
    """Structured model metadata."""

    model_type: str
    version: str
    created_at: str
    label_mapping: Dict[str, str]
    input_shape: List[int]
    output_classes: int
    metrics: Dict[str, Any]
    training_config: Dict[str, Any]
    model_hash: Optional[str] = None
    framework_version: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Path) -> None:
        """Save metadata to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: Path) -> 'ModelMetadata':
        """Load metadata from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class MetadataManager:
    """Manages model metadata lifecycle."""

    SUPPORTED_TYPES = [
        'keras_cnn', 'keras_lstm', 'keras_mlp',
        'xgboost', 'randomforest', 'gradientboosting',
        'ensemble_voting', 'ensemble_stacking'
    ]

    @staticmethod
    def create_metadata(
        model_type: str,
        version: str,
        label_mapping: Dict[str, str],
        input_shape: List[int],
        output_classes: int,
        metrics: Dict[str, Any],
        training_config: Dict[str, Any],
        model_hash: Optional[str] = None,
        framework_version: Optional[str] = None,
        python_version: Optional[str] = None,
        dependencies: Optional[Dict[str, str]] = None
    ) -> ModelMetadata:
        """
        Create structured model metadata.

        Args:
            model_type: Type of model (e.g., 'keras_cnn', 'xgboost')
            version: Model version string
            label_mapping: Dictionary mapping class indices to labels
            input_shape: Input tensor/feature shape
            output_classes: Number of output classes
            metrics: Performance metrics dictionary
            training_config: Training configuration parameters
            model_hash: SHA256 hash of model file
            framework_version: Framework version (e.g., TensorFlow version)
            python_version: Python version used
            dependencies: Package dependencies

        Returns:
            ModelMetadata object
        """
        if model_type not in MetadataManager.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {MetadataManager.SUPPORTED_TYPES}"
            )

        created_at = datetime.utcnow().isoformat() + 'Z'

        return ModelMetadata(
            model_type=model_type,
            version=version,
            created_at=created_at,
            label_mapping=label_mapping,
            input_shape=input_shape,
            output_classes=output_classes,
            metrics=metrics,
            training_config=training_config,
            model_hash=model_hash,
            framework_version=framework_version,
            python_version=python_version,
            dependencies=dependencies
        )

    @staticmethod
    def validate_metadata(metadata: ModelMetadata) -> bool:
        """
        Validate metadata completeness and consistency.

        Args:
            metadata: ModelMetadata object to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check required fields
        if not metadata.model_type:
            raise ValueError("model_type is required")

        if not metadata.version:
            raise ValueError("version is required")

        if not metadata.label_mapping:
            raise ValueError("label_mapping is required")

        if not metadata.input_shape:
            raise ValueError("input_shape is required")

        # Validate label mapping consistency
        if len(metadata.label_mapping) != metadata.output_classes:
            raise ValueError(
                f"Label mapping has {len(metadata.label_mapping)} classes "
                f"but output_classes is {metadata.output_classes}"
            )

        # Validate label indices
        expected_indices = set(str(i) for i in range(metadata.output_classes))
        actual_indices = set(metadata.label_mapping.keys())
        if expected_indices != actual_indices:
            raise ValueError(
                f"Label mapping indices {actual_indices} do not match "
                f"expected indices {expected_indices}"
            )

        return True

    @staticmethod
    def compute_model_hash(filepath: Path) -> str:
        """
        Compute SHA256 hash of model file.

        Args:
            filepath: Path to model file

        Returns:
            Hex string of SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in chunks for large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def get_metadata_path(model_path: Path) -> Path:
        """
        Get metadata file path from model path.

        Args:
            model_path: Path to model file

        Returns:
            Path to metadata JSON file
        """
        return model_path.parent / f"{model_path.stem}_metadata.json"

    @staticmethod
    def get_model_filename(model_type: str, version: str) -> str:
        """
        Generate standardized model filename.

        Args:
            model_type: Type of model
            version: Version string

        Returns:
            Filename string (without extension)
        """
        # Normalize model type for filename
        type_map = {
            'keras_cnn': 'genesis_cnn',
            'keras_lstm': 'genesis_lstm',
            'keras_mlp': 'genesis_mlp',
            'xgboost': 'xgboost',
            'randomforest': 'randomforest',
            'gradientboosting': 'gradientboosting',
            'ensemble_voting': 'ensemble_voting',
            'ensemble_stacking': 'ensemble_stacking'
        }

        base_name = type_map.get(model_type, model_type)
        return f"{base_name}_v{version}"

    @staticmethod
    def extract_version_from_filename(filename: str) -> Optional[str]:
        """
        Extract version from model filename.

        Args:
            filename: Model filename

        Returns:
            Version string or None
        """
        if '_v' in filename:
            parts = filename.split('_v')
            if len(parts) >= 2:
                version = parts[-1].split('.')[0]
                return version
        return None

    @staticmethod
    def update_metrics(
        metadata_path: Path,
        new_metrics: Dict[str, Any]
    ) -> ModelMetadata:
        """
        Update metrics in existing metadata.

        Args:
            metadata_path: Path to metadata JSON
            new_metrics: New metrics to merge

        Returns:
            Updated ModelMetadata object
        """
        metadata = ModelMetadata.from_json(metadata_path)
        metadata.metrics.update(new_metrics)
        metadata.to_json(metadata_path)
        return metadata
