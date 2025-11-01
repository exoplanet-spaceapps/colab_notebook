# Model I/O and Metadata Management - Usage Guide

## Overview

This system provides unified interfaces for saving and loading ML models with comprehensive metadata tracking.

## Supported Model Types

- **Keras Models**: `keras_cnn`, `keras_lstm`, `keras_mlp`
- **Tree Models**: `xgboost`, `randomforest`, `gradientboosting`
- **Ensemble Models**: `ensemble_voting`, `ensemble_stacking`

## File Naming Convention

Models and metadata follow standardized naming:

```
genesis_cnn_v1.0.0.keras
genesis_cnn_v1.0.0_metadata.json

xgboost_v2.1.json
xgboost_v2.1_metadata.json

ensemble_voting_v1.0.pkl
ensemble_voting_v1.0_metadata.json
```

## Usage Examples

### 1. Saving a Keras Model

```python
from models import save_model, MetadataManager
import tensorflow as tf

# Train your model
model = tf.keras.Sequential([...])
model.fit(X_train, y_train, epochs=100)

# Create metadata
metadata = MetadataManager.create_metadata(
    model_type='keras_cnn',
    version='1.0.0',
    label_mapping={
        '0': 'CONFIRMED',
        '1': 'CANDIDATE',
        '2': 'FALSE POSITIVE'
    },
    input_shape=[784],
    output_classes=3,
    metrics={
        'accuracy': 0.9523,
        'precision': 0.9458,
        'recall': 0.9501,
        'f1_score': 0.9479
    },
    training_config={
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy'
    }
)

# Save model and metadata
model_path, metadata_path = save_model(
    model=model,
    model_type='keras_cnn',
    save_dir='./models',
    metadata=metadata
)

print(f"Model saved: {model_path}")
print(f"Metadata saved: {metadata_path}")
```

### 2. Saving an XGBoost Model

```python
from models import save_model, MetadataManager
import xgboost as xgb

# Train model
model = xgb.XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Create metadata
metadata = MetadataManager.create_metadata(
    model_type='xgboost',
    version='1.0.0',
    label_mapping={
        '0': 'CONFIRMED',
        '1': 'CANDIDATE',
        '2': 'FALSE POSITIVE'
    },
    input_shape=[20],  # Number of features
    output_classes=3,
    metrics={
        'accuracy': 0.9301,
        'roc_auc': 0.9654
    },
    training_config={
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1
    }
)

# Save
model_path, metadata_path = save_model(
    model=model,
    model_type='xgboost',
    save_dir='./models',
    metadata=metadata
)
```

### 3. Saving an Ensemble Model

```python
from models import save_model, MetadataManager
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Create ensemble
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('lr', LogisticRegression())
]
ensemble = VotingClassifier(estimators=estimators, voting='soft')
ensemble.fit(X_train, y_train)

# Create metadata
metadata = MetadataManager.create_metadata(
    model_type='ensemble_voting',
    version='1.0.0',
    label_mapping={
        '0': 'CONFIRMED',
        '1': 'CANDIDATE',
        '2': 'FALSE POSITIVE'
    },
    input_shape=[20],
    output_classes=3,
    metrics={
        'accuracy': 0.9450,
        'ensemble_diversity': 0.23
    },
    training_config={
        'voting': 'soft',
        'estimators': ['DecisionTree', 'LogisticRegression']
    }
)

# Save
model_path, metadata_path = save_model(
    model=model,
    model_type='ensemble_voting',
    save_dir='./models',
    metadata=metadata
)
```

### 4. Loading a Model

```python
from models import load_model

# Load model and metadata
model, metadata = load_model('./models/genesis_cnn_v1.0.0.keras')

# Access metadata
print(f"Model type: {metadata.model_type}")
print(f"Version: {metadata.version}")
print(f"Accuracy: {metadata.metrics['accuracy']}")
print(f"Label mapping: {metadata.label_mapping}")

# Use model for prediction
predictions = model.predict(X_test)
```

### 5. Loading Latest Version

```python
from models import load_latest_version

# Load latest CNN model
model, metadata = load_latest_version(
    model_dir='./models',
    model_type='keras_cnn'
)

print(f"Loaded version: {metadata.version}")
```

### 6. Updating Metrics

```python
from models import MetadataManager

# Update metrics for existing model
updated_metadata = MetadataManager.update_metrics(
    metadata_path='./models/genesis_cnn_v1.0.0_metadata.json',
    new_metrics={
        'test_accuracy': 0.9512,
        'test_f1': 0.9489,
        'inference_time_ms': 2.3
    }
)

print(f"Updated metrics: {updated_metadata.metrics}")
```

## Metadata Structure

Complete metadata JSON structure:

```json
{
  "model_type": "keras_cnn",
  "version": "1.0.0",
  "created_at": "2025-10-05T12:54:22Z",
  "label_mapping": {
    "0": "CONFIRMED",
    "1": "CANDIDATE",
    "2": "FALSE POSITIVE"
  },
  "input_shape": [784],
  "output_classes": 3,
  "metrics": {
    "accuracy": 0.9523,
    "precision": 0.9458,
    "recall": 0.9501,
    "f1_score": 0.9479
  },
  "training_config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy"
  },
  "model_hash": "a3f5d8e9b2c1...",
  "framework_version": "2.15.0",
  "python_version": "3.10.0",
  "dependencies": {
    "tensorflow": "2.15.0",
    "numpy": "1.24.0"
  }
}
```

## Error Handling

### Common Errors

1. **Model file not found**
```python
try:
    model, metadata = load_model('./nonexistent.keras')
except ModelIOError as e:
    print(f"Error: {e}")
```

2. **Hash mismatch (file corruption)**
```python
try:
    model, metadata = load_model('./corrupted_model.keras')
except ModelIOError as e:
    print(f"File corrupted: {e}")
```

3. **Invalid metadata**
```python
try:
    metadata = MetadataManager.create_metadata(
        model_type='invalid_type',
        ...
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Best Practices

1. **Version Management**
   - Use semantic versioning (MAJOR.MINOR.PATCH)
   - Increment MAJOR for breaking changes
   - Increment MINOR for new features
   - Increment PATCH for bug fixes

2. **Metadata Completeness**
   - Always include comprehensive metrics
   - Document training configuration
   - Track framework versions
   - Store label mappings

3. **File Organization**
   ```
   models/
     ├── genesis_cnn_v1.0.0.keras
     ├── genesis_cnn_v1.0.0_metadata.json
     ├── genesis_cnn_v1.1.0.keras
     ├── genesis_cnn_v1.1.0_metadata.json
     ├── xgboost_v1.0.json
     └── xgboost_v1.0_metadata.json
   ```

4. **Hash Verification**
   - Always verify model hash on load
   - Detect file corruption early
   - Prevent using modified models

## Advanced Features

### Custom Dependencies Tracking

```python
import tensorflow as tf
import numpy as np
import xgboost as xgb

dependencies = {
    'tensorflow': tf.__version__,
    'numpy': np.__version__,
    'xgboost': xgb.__version__
}

metadata = MetadataManager.create_metadata(
    ...,
    dependencies=dependencies
)
```

### Batch Metadata Updates

```python
for model_file in Path('./models').glob('*.keras'):
    metadata_path = MetadataManager.get_metadata_path(model_file)
    MetadataManager.update_metrics(
        metadata_path,
        {'validated': True, 'validation_date': '2025-10-05'}
    )
```

### Model Registry

```python
from pathlib import Path
from models import load_model

def list_models(model_dir):
    """List all available models with metadata."""
    models = []
    for metadata_file in Path(model_dir).glob('*_metadata.json'):
        metadata = ModelMetadata.from_json(metadata_file)
        models.append({
            'type': metadata.model_type,
            'version': metadata.version,
            'accuracy': metadata.metrics.get('accuracy'),
            'created': metadata.created_at
        })
    return sorted(models, key=lambda x: x['created'], reverse=True)

# Usage
available_models = list_models('./models')
for m in available_models:
    print(f"{m['type']} v{m['version']}: {m['accuracy']:.4f}")
```

## Integration with Training Pipeline

```python
from models import save_model, MetadataManager

def train_and_save(model, X_train, y_train, X_val, y_val,
                   model_type, version, save_dir):
    """Complete training and saving workflow."""

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val)

    # Create metadata
    metadata = MetadataManager.create_metadata(
        model_type=model_type,
        version=version,
        label_mapping={
            '0': 'CONFIRMED',
            '1': 'CANDIDATE',
            '2': 'FALSE POSITIVE'
        },
        input_shape=list(X_train.shape[1:]),
        output_classes=len(np.unique(y_train)),
        metrics={
            'val_accuracy': float(val_acc),
            'val_loss': float(val_loss),
            'train_accuracy': float(history.history['accuracy'][-1])
        },
        training_config={
            'epochs': 100,
            'batch_size': 32,
            'optimizer': model.optimizer.get_config()['name'],
            'learning_rate': float(model.optimizer.learning_rate.numpy())
        }
    )

    # Save
    model_path, metadata_path = save_model(
        model=model,
        model_type=model_type,
        save_dir=save_dir,
        metadata=metadata
    )

    return model_path, metadata_path
```

## File Locations

- **Source Code**: `C:/Users/thc1006/Desktop/新增資料夾/colab_notebook/src/models/`
  - `metadata.py` - Metadata management
  - `model_io.py` - Save/load interfaces
  - `__init__.py` - Package exports

- **Tests**: `C:/Users/thc1006/Desktop/新增資料夾/colab_notebook/tests/test_model_io.py`

- **Documentation**: `C:/Users/thc1006/Desktop/新增資料夾/colab_notebook/docs/model_io_usage.md`
