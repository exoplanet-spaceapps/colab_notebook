"""
Kepler Exoplanet Detection - Data Preprocessing Script for Google Colab (2025)

This script provides a complete, production-ready data preprocessing pipeline for
the Kepler exoplanet detection project. It handles data loading, label encoding,
merging, shuffling, and train/test splitting with comprehensive validation.

Compatible with:
- NumPy 2.x
- Pandas 2.x
- Scikit-learn 1.x
- Google Colab environment

Author: Claude Code
Date: 2025
"""

import os
import sys
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def detect_colab_environment() -> bool:
    """
    Detect if the script is running in Google Colab environment.

    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def upload_files_colab() -> dict:
    """
    Handle file upload in Google Colab environment.

    Returns:
        dict: Dictionary mapping filenames to their uploaded paths
    """
    from google.colab import files

    print("=" * 80)
    print("📁 FILE UPLOAD - Please upload your CSV files")
    print("=" * 80)
    print("\nRequired files:")
    print("  1. koi_lightcurve_features_no_label.csv (features)")
    print("  2. q1_q17_dr25_koi.csv (labels with 'koi_disposition' column)")
    print("\nClick 'Choose Files' button to upload...\n")

    uploaded = files.upload()

    if len(uploaded) != 2:
        raise ValueError(f"Expected 2 files, but {len(uploaded)} were uploaded")

    return uploaded


def load_csv_file(filepath: str, file_description: str) -> pd.DataFrame:
    """
    Load a CSV file with validation and error handling.

    Args:
        filepath: Path to the CSV file
        file_description: Description of the file for error messages

    Returns:
        pd.DataFrame: Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    print(f"\n📂 Loading {file_description}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
        print(f"   ✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"File is empty: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading {file_description}: {str(e)}")


def validate_dataframe(df: pd.DataFrame, required_columns: list, df_name: str) -> None:
    """
    Validate DataFrame structure and content.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name of the DataFrame for error messages

    Raises:
        ValueError: If validation fails
    """
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"{df_name} is empty")

    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{df_name} is missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"   ✓ Validation passed for {df_name}")


def extract_and_encode_labels(
    labels_df: pd.DataFrame,
    label_column: str = 'koi_disposition'
) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Extract and one-hot encode the disposition labels.

    Args:
        labels_df: DataFrame containing the labels
        label_column: Name of the label column (default: 'koi_disposition')

    Returns:
        Tuple containing:
            - DataFrame with one-hot encoded labels
            - Original label array
            - List of class names
    """
    print(f"\n🏷️  Extracting and encoding labels from '{label_column}'...")

    # Extract labels
    labels = labels_df[label_column].values

    # Get unique classes and their counts
    unique_classes, counts = np.unique(labels, return_counts=True)
    print(f"   ✓ Found {len(unique_classes)} classes:")
    for cls, count in zip(unique_classes, counts):
        print(f"      - {cls}: {count} samples ({count/len(labels)*100:.1f}%)")

    # One-hot encode labels
    lb = LabelBinarizer()
    labels_encoded = lb.fit_transform(labels)

    # Handle binary classification case (LabelBinarizer returns (n, 1) shape)
    if labels_encoded.shape[1] == 1:
        # Convert to (n, 2) format for consistency
        labels_encoded = np.hstack([1 - labels_encoded, labels_encoded])
        class_names = [f"NOT_{lb.classes_[0]}", lb.classes_[0]]
    else:
        class_names = list(lb.classes_)

    # Create DataFrame with encoded labels
    encoded_df = pd.DataFrame(
        labels_encoded,
        columns=class_names,
        index=labels_df.index
    )

    print(f"   ✓ One-hot encoding complete: {encoded_df.shape[1]} classes")
    print(f"   ✓ Class names: {class_names}")

    return encoded_df, labels, class_names


def merge_features_and_labels(
    features_df: pd.DataFrame,
    labels_encoded_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge features and one-hot encoded labels.

    Args:
        features_df: DataFrame with features
        labels_encoded_df: DataFrame with one-hot encoded labels

    Returns:
        pd.DataFrame: Merged DataFrame

    Raises:
        ValueError: If DataFrames have incompatible shapes
    """
    print(f"\n🔗 Merging features and labels...")

    if len(features_df) != len(labels_encoded_df):
        raise ValueError(
            f"Shape mismatch: features ({len(features_df)} rows) vs "
            f"labels ({len(labels_encoded_df)} rows)"
        )

    # Reset indices to ensure proper alignment
    features_df = features_df.reset_index(drop=True)
    labels_encoded_df = labels_encoded_df.reset_index(drop=True)

    # Merge DataFrames
    merged_df = pd.concat([features_df, labels_encoded_df], axis=1)

    print(f"   ✓ Merged successfully: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
    print(f"   ✓ Feature columns: {len(features_df.columns)}")
    print(f"   ✓ Label columns: {len(labels_encoded_df.columns)}")

    return merged_df


def shuffle_data(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Randomly shuffle DataFrame rows.

    Args:
        df: DataFrame to shuffle
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        pd.DataFrame: Shuffled DataFrame
    """
    print(f"\n🔀 Shuffling data (random_state={random_state})...")

    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"   ✓ Data shuffled: {shuffled_df.shape[0]} rows")

    return shuffled_df


def split_train_test(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets with stratification.

    Args:
        features: Feature array
        labels: Label array (one-hot encoded)
        test_size: Proportion of test set (default: 0.25 for 3:1 split)
        random_state: Random seed for reproducibility (default: 42)
        stratify: Array for stratified splitting (default: None, will use labels)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"\n✂️  Splitting data (train:test = {1-test_size:.0%}:{test_size:.0%})...")

    # For stratification, use argmax of one-hot labels to get class indices
    if stratify is None and labels.ndim > 1:
        stratify = np.argmax(labels, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    print(f"   ✓ Training set: {X_train.shape[0]} samples")
    print(f"   ✓ Testing set: {X_test.shape[0]} samples")

    # Verify stratification
    if labels.ndim > 1:
        train_dist = np.mean(y_train, axis=0)
        test_dist = np.mean(y_test, axis=0)
        print(f"   ✓ Class distribution preserved:")
        print(f"      Train: {train_dist}")
        print(f"      Test:  {test_dist}")

    return X_train, X_test, y_train, y_test


def print_summary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list
) -> None:
    """
    Print comprehensive summary of preprocessed data.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        class_names: List of class names
    """
    print("\n" + "=" * 80)
    print("📊 DATA PREPROCESSING SUMMARY")
    print("=" * 80)

    print(f"\n🎯 Dataset Shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_test:  {y_test.shape}")

    print(f"\n📈 Feature Statistics:")
    print(f"   Number of features: {X_train.shape[1]}")
    print(f"   Feature range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"   Feature mean: {X_train.mean():.4f}")
    print(f"   Feature std: {X_train.std():.4f}")

    print(f"\n🏷️  Label Information:")
    print(f"   Number of classes: {len(class_names)}")
    print(f"   Class names: {class_names}")
    print(f"   Label encoding: One-hot")

    print(f"\n🔢 Class Distribution:")
    train_counts = y_train.sum(axis=0)
    test_counts = y_test.sum(axis=0)
    for i, cls in enumerate(class_names):
        train_pct = train_counts[i] / len(y_train) * 100
        test_pct = test_counts[i] / len(y_test) * 100
        print(f"   {cls}:")
        print(f"      Train: {int(train_counts[i])} ({train_pct:.1f}%)")
        print(f"      Test:  {int(test_counts[i])} ({test_pct:.1f}%)")

    print(f"\n✅ Data is ready for training!")
    print(f"   Variables available in memory:")
    print(f"      - X_train: training features")
    print(f"      - y_train: training labels (one-hot)")
    print(f"      - X_test: testing features")
    print(f"      - y_test: testing labels (one-hot)")
    print("=" * 80 + "\n")


def preprocess_kepler_data(
    features_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    label_column: str = 'koi_disposition',
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Main preprocessing pipeline for Kepler exoplanet detection data.

    Args:
        features_path: Path to features CSV (or None for Colab upload)
        labels_path: Path to labels CSV (or None for Colab upload)
        label_column: Name of the label column (default: 'koi_disposition')
        test_size: Proportion of test set (default: 0.25)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, class_names)
    """
    print("\n" + "=" * 80)
    print("🚀 KEPLER EXOPLANET DETECTION - DATA PREPROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Detect environment
    is_colab = detect_colab_environment()
    print(f"\n🔍 Environment: {'Google Colab' if is_colab else 'Local'}")

    # Step 2: Handle file input
    if is_colab and (features_path is None or labels_path is None):
        uploaded = upload_files_colab()

        # Identify files by name
        for filename in uploaded.keys():
            if 'lightcurve_features' in filename or 'features' in filename:
                features_path = filename
            elif 'koi' in filename or 'label' in filename:
                labels_path = filename

        if features_path is None or labels_path is None:
            raise ValueError("Could not identify features and labels files from uploads")

    # Step 3: Load data
    features_df = load_csv_file(features_path, "features file")
    labels_df = load_csv_file(labels_path, "labels file")

    # Step 4: Validate data
    validate_dataframe(features_df, [], "features_df")
    validate_dataframe(labels_df, [label_column], "labels_df")

    # Step 5: Extract and encode labels
    labels_encoded_df, original_labels, class_names = extract_and_encode_labels(
        labels_df, label_column
    )

    # Step 6: Merge features and labels
    merged_df = merge_features_and_labels(features_df, labels_encoded_df)

    # Step 7: Shuffle data
    shuffled_df = shuffle_data(merged_df, random_state=random_state)

    # Step 8: Separate features and labels
    print(f"\n📦 Separating features and labels...")
    feature_cols = list(features_df.columns)
    label_cols = list(labels_encoded_df.columns)

    X = shuffled_df[feature_cols].values
    y = shuffled_df[label_cols].values

    print(f"   ✓ Features shape: {X.shape}")
    print(f"   ✓ Labels shape: {y.shape}")

    # Step 9: Train/test split with stratification
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 10: Print summary
    print_summary(X_train, y_train, X_test, y_test, class_names)

    return X_train, X_test, y_train, y_test, class_names


# Main execution block
if __name__ == "__main__":
    """
    Main execution: Run the preprocessing pipeline.

    Usage in Google Colab:
        # Simply run the cell - it will prompt for file upload
        X_train, X_test, y_train, y_test, class_names = preprocess_kepler_data()

    Usage with local files:
        X_train, X_test, y_train, y_test, class_names = preprocess_kepler_data(
            features_path='path/to/koi_lightcurve_features_no_label.csv',
            labels_path='path/to/q1_q17_dr25_koi.csv'
        )
    """
    try:
        # Run preprocessing pipeline
        X_train, X_test, y_train, y_test, class_names = preprocess_kepler_data()

        # Variables are now available in the global scope for use in subsequent cells
        print("✨ Preprocessing complete! You can now use the following variables:")
        print("   - X_train, y_train: Training data")
        print("   - X_test, y_test: Testing data")
        print("   - class_names: List of class names")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease check:")
        print("  1. Files are correctly named and formatted")
        print("  2. CSV files contain the expected columns")
        print("  3. Files are not corrupted")
        raise
