# Kepler Exoplanet Data Preprocessing Usage Guide

## Overview

This preprocessing module handles three-class classification of Kepler exoplanet data:
- **CANDIDATE**: Objects of interest that may be exoplanets
- **CONFIRMED**: Verified exoplanets
- **FALSE POSITIVE**: Objects that are not exoplanets

## Quick Start

```python
from src.preprocessing import preprocess_kepler_data, validate_preprocessed_data

# Run complete preprocessing pipeline
X_train, X_test, y_train, y_test, stats = preprocess_kepler_data(
    features_path='koi_lightcurve_features_no_label.csv',
    labels_path='q1_q17_dr25_koi.csv',
    test_size=0.25,           # 25% test set (3:1 split)
    random_state=42,          # For reproducibility
    stratify=True             # Maintain class distribution
)

# Validate the preprocessed data
validate_preprocessed_data(X_train, X_test, y_train, y_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {list(y_train.columns)}")
```

## Pipeline Steps

### 1. Data Loading
- Loads features from `koi_lightcurve_features_no_label.csv`
- Loads labels from `q1_q17_dr25_koi.csv`
- Validates required columns exist
- Extracts KOI IDs for alignment

### 2. Label Encoding
- Converts `koi_disposition` to one-hot encoding
- Creates three binary columns: CANDIDATE, CONFIRMED, FALSE POSITIVE
- Validates encoding (exactly one 1 per row)
- Logs class distribution

### 3. Data Merging
- Aligns features and labels by KOI ID (`kepoi_name`)
- Handles mismatched row counts (8054 labels → 1866 features)
- Sorts by ID to ensure proper alignment
- Removes ID columns after merging

### 4. Shuffling & Splitting
- Random shuffle with `random_state=42`
- Stratified split (maintains class proportions)
- 75% training (1399 samples)
- 25% testing (467 samples)

## Output Data

### Training Set
- **X_train**: (1399, 783) - Features
- **y_train**: (1399, 3) - One-hot encoded labels

### Test Set
- **X_test**: (467, 783) - Features
- **y_test**: (467, 3) - One-hot encoded labels

### Labels Format
```python
# y_train and y_test columns
['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

# Example row (one-hot encoded):
# [0, 1, 0]  # CONFIRMED exoplanet
# [1, 0, 0]  # CANDIDATE
# [0, 0, 1]  # FALSE POSITIVE
```

## Class Distribution

### Full Dataset (1866 samples)
- **CANDIDATE**: 153 (8.2%)
- **CONFIRMED**: 1216 (65.2%)
- **FALSE POSITIVE**: 497 (26.6%)

### Training Set (1399 samples)
- **CANDIDATE**: 115 (8.2%)
- **CONFIRMED**: 912 (65.2%)
- **FALSE POSITIVE**: 372 (26.6%)

### Test Set (467 samples)
- **CANDIDATE**: 38 (8.1%)
- **CONFIRMED**: 304 (65.1%)
- **FALSE POSITIVE**: 125 (26.8%)

Note: Stratified sampling maintains nearly identical class proportions.

## Statistics Dictionary

```python
{
  "original_features_shape": [1866, 784],
  "original_labels_shape": [8054, 2],
  "feature_id_column": "Unnamed: 0",
  "label_id_column": "kepoi_name",
  "class_distribution": {
    "FALSE POSITIVE": 3966,
    "CONFIRMED": 2726,
    "CANDIDATE": 1362
  },
  "merged_size": 1866,
  "matched_ids": 1866,
  "unmatched_features": 0,
  "unmatched_labels": 6188,
  "train_size": 1399,
  "test_size": 467,
  "train_class_dist": {
    "CANDIDATE": 115,
    "CONFIRMED": 912,
    "FALSE POSITIVE": 372
  },
  "test_class_dist": {
    "CANDIDATE": 38,
    "CONFIRMED": 304,
    "FALSE POSITIVE": 125
  }
}
```

## Advanced Usage

### Using the DataPreprocessor Class

```python
from src.preprocessing import DataPreprocessor

# Initialize with custom random state
preprocessor = DataPreprocessor(random_state=123)

# Step-by-step processing
features_df, labels_df = preprocessor.load_data(
    'koi_lightcurve_features_no_label.csv',
    'q1_q17_dr25_koi.csv'
)

encoded_labels = preprocessor.encode_labels(labels_df)

features_aligned, labels_aligned = preprocessor.merge_data(
    features_df,
    encoded_labels
)

X_train, X_test, y_train, y_test = preprocessor.shuffle_and_split(
    features_aligned,
    labels_aligned,
    test_size=0.3,  # Custom split ratio
    stratify=True
)

# Get preprocessing statistics
stats = preprocessor.get_stats()
```

### Custom ID Columns

```python
# If your files have different ID column names
X_train, X_test, y_train, y_test, stats = preprocess_kepler_data(
    features_path='custom_features.csv',
    labels_path='custom_labels.csv',
    feature_id_col='koi_id',      # Custom feature ID column
    label_id_col='object_id'      # Custom label ID column
)
```

### Without Stratification

```python
# Use simple random split (not recommended for imbalanced data)
X_train, X_test, y_train, y_test, stats = preprocess_kepler_data(
    'koi_lightcurve_features_no_label.csv',
    'q1_q17_dr25_koi.csv',
    stratify=False
)
```

## Data Validation

The `validate_preprocessed_data` function checks:
- ✅ No overlapping indices between train/test
- ✅ Matching lengths between X and y
- ✅ No missing values in labels
- ✅ Valid one-hot encoding (exactly one 1 per row)
- ✅ Matching columns in train/test sets
- ✅ Correct three-class labels

```python
from src.preprocessing import validate_preprocessed_data

try:
    validate_preprocessed_data(X_train, X_test, y_train, y_test)
    print("✅ Data validation passed!")
except AssertionError as e:
    print(f"❌ Validation failed: {e}")
```

## Feature Information

The features are extracted from Kepler light curves and include:
- Statistical measures (variance, mean, std, etc.)
- Time series patterns (duplicates, trends, peaks)
- Frequency domain features
- **Total**: 783 features (after removing ID column)

## Error Handling

### Common Issues

1. **Missing Columns**
   ```python
   ValueError: Features file must contain 'Unnamed: 0' column
   ```
   Solution: Verify your features file has the ID column

2. **No Common IDs**
   ```python
   ValueError: No common IDs found between features and labels
   ```
   Solution: Check that KOI IDs match between files

3. **Invalid Disposition Values**
   ```python
   ValueError: Unexpected disposition values: {'UNKNOWN'}
   ```
   Solution: Data contains unexpected classes beyond CANDIDATE, CONFIRMED, FALSE POSITIVE

## Performance

- **Load Time**: ~0.3s for features, ~0.01s for labels
- **Encoding Time**: ~0.02s for one-hot encoding
- **Merge Time**: ~0.03s for ID-based alignment
- **Split Time**: ~0.01s for stratified split
- **Total Pipeline**: ~0.4s

## Next Steps

After preprocessing, the data is ready for:
1. **Model Training**: Use X_train, y_train with your classifier
2. **Evaluation**: Test on X_test, y_test
3. **Feature Engineering**: Additional feature selection or transformation
4. **Cross-Validation**: Further validation using k-fold CV

## Example: Complete Workflow

```python
from src.preprocessing import preprocess_kepler_data, validate_preprocessed_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Preprocess data
X_train, X_test, y_train, y_test, stats = preprocess_kepler_data(
    'koi_lightcurve_features_no_label.csv',
    'q1_q17_dr25_koi.csv'
)

# 2. Validate
validate_preprocessed_data(X_train, X_test, y_train, y_test)

# 3. Convert one-hot to class indices for sklearn
y_train_labels = y_train.values.argmax(axis=1)
y_test_labels = y_test.values.argmax(axis=1)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_labels)

# 5. Evaluate
y_pred = model.predict(X_test)
print(classification_report(
    y_test_labels,
    y_pred,
    target_names=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
))

# 6. Print statistics
print(f"\nPreprocessing Statistics:")
print(f"Total samples: {stats['merged_size']}")
print(f"Train/Test split: {stats['train_size']}/{stats['test_size']}")
print(f"Class balance maintained: {stats['train_class_dist']}")
```

## File Locations

- **Source Code**: `C:\Users\thc1006\Desktop\新增資料夾\colab_notebook\src\preprocessing\data_preprocessing.py`
- **Module Init**: `C:\Users\thc1006\Desktop\新增資料夾\colab_notebook\src\preprocessing\__init__.py`
- **This Guide**: `C:\Users\thc1006\Desktop\新增資料夾\colab_notebook\docs\preprocessing_usage.md`
