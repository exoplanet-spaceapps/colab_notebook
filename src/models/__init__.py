"""
Model I/O and Metadata Management

Unified interfaces for saving/loading models and managing metadata.
"""

from .metadata import (
    ModelMetadata,
    MetadataManager
)

from .model_io import (
    ModelSaver,
    ModelLoader,
    ModelIOError,
    save_model,
    load_model,
    load_latest_version
)

__all__ = [
    'ModelMetadata',
    'MetadataManager',
    'ModelSaver',
    'ModelLoader',
    'ModelIOError',
    'save_model',
    'load_model',
    'load_latest_version'
]
