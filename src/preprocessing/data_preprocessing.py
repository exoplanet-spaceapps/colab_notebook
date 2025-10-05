"""
Kepler Exoplanet Three-Class Data Preprocessing Module

This module provides functions for preprocessing Kepler exoplanet data:
- Load features and labels from CSV files
- One-hot encode three-class disposition labels
- Merge features with labels
- Random shuffle with stratified train/test split
"""

import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing pipeline for Kepler exoplanet three-class classification.

    Handles:
    - Loading features and labels
    - One-hot encoding of disposition labels
    - Data merging and shuffling
    - Stratified train/test split
    """

    # Valid disposition classes
    VALID_CLASSES = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']

    def __init__(self, random_state: int = 42):
        """
        Initialize preprocessor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.label_encoder = LabelBinarizer()
        self.stats: Dict[str, Any] = {}

    def load_data(
        self,
        features_path: str,
        labels_path: str,
        feature_id_col: str = 'Unnamed: 0',
        label_id_col: str = 'kepoi_name'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load features and labels from CSV files.

        Args:
            features_path: Path to features CSV file
            labels_path: Path to labels CSV file (contains disposition column)
            feature_id_col: Column name for KOI ID in features (default: 'Unnamed: 0')
            label_id_col: Column name for KOI ID in labels (default: 'kepoi_name')

        Returns:
            Tuple of (features_df, labels_df) with matching IDs

        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If required columns are missing
        """
        logger.info(f"Loading features from: {features_path}")
        features_df = pd.read_csv(features_path)
        logger.info(f"Features shape: {features_df.shape}")

        # Validate feature ID column exists
        if feature_id_col not in features_df.columns:
            raise ValueError(f"Features file must contain '{feature_id_col}' column")

        logger.info(f"Loading labels from: {labels_path}")
        labels_df = pd.read_csv(labels_path)

        # Validate disposition column exists
        if 'koi_disposition' not in labels_df.columns:
            raise ValueError("Labels file must contain 'koi_disposition' column")

        # Validate label ID column exists
        if label_id_col not in labels_df.columns:
            raise ValueError(f"Labels file must contain '{label_id_col}' column")

        # Extract required columns from labels
        labels_df = labels_df[[label_id_col, 'koi_disposition']].copy()
        logger.info(f"Labels shape: {labels_df.shape}")

        # Store ID column names for merging
        self.feature_id_col = feature_id_col
        self.label_id_col = label_id_col

        # Store stats
        self.stats['original_features_shape'] = features_df.shape
        self.stats['original_labels_shape'] = labels_df.shape
        self.stats['feature_id_column'] = feature_id_col
        self.stats['label_id_column'] = label_id_col

        return features_df, labels_df

    def encode_labels(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert disposition labels to one-hot encoding.

        Args:
            labels_df: DataFrame with 'koi_disposition' column (and ID column)

        Returns:
            DataFrame with ID column and three binary columns for each class

        Raises:
            ValueError: If unexpected disposition values found
        """
        logger.info("Encoding labels to one-hot format...")

        # Get unique classes
        unique_classes = labels_df['koi_disposition'].unique()
        logger.info(f"Found classes: {unique_classes}")

        # Check for unexpected classes
        unexpected = set(unique_classes) - set(self.VALID_CLASSES)
        if unexpected:
            raise ValueError(
                f"Unexpected disposition values: {unexpected}. "
                f"Expected: {self.VALID_CLASSES}"
            )

        # Count class distribution
        class_counts = labels_df['koi_disposition'].value_counts()
        logger.info(f"Class distribution:\n{class_counts}")
        self.stats['class_distribution'] = class_counts.to_dict()

        # Fit and transform to one-hot
        # LabelBinarizer will create columns in alphabetical order
        encoded = self.label_encoder.fit_transform(labels_df['koi_disposition'])

        # Create DataFrame with descriptive column names
        # Sort classes alphabetically to match LabelBinarizer output
        sorted_classes = sorted(self.VALID_CLASSES)
        encoded_df = pd.DataFrame(
            encoded,
            columns=sorted_classes,
            index=labels_df.index
        )

        # Add ID column from labels_df
        encoded_df.insert(0, self.label_id_col, labels_df[self.label_id_col])

        logger.info(f"One-hot encoded shape: {encoded_df.shape}")
        logger.info(f"Column order: {list(encoded_df.columns)}")

        # Validate one-hot encoding (exactly one 1 per row in label columns)
        row_sums = encoded_df[sorted_classes].sum(axis=1)
        assert (row_sums == 1).all(), "Invalid one-hot encoding detected"

        return encoded_df

    def merge_data(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge features with one-hot encoded labels by KOI ID.

        Args:
            features_df: Features DataFrame (with ID column)
            labels_df: One-hot encoded labels DataFrame (with ID column)

        Returns:
            Tuple of (features, labels) with aligned rows by ID
        """
        logger.info("Merging features and labels by KOI ID...")

        # Get ID column names
        feature_id = self.feature_id_col
        label_id = self.label_id_col

        # Find common IDs
        feature_ids = set(features_df[feature_id])
        label_ids = set(labels_df[label_id])
        common_ids = feature_ids & label_ids

        logger.info(f"Features with unique IDs: {len(feature_ids)}")
        logger.info(f"Labels with unique IDs: {len(label_ids)}")
        logger.info(f"Common IDs: {len(common_ids)}")

        if len(common_ids) == 0:
            raise ValueError("No common IDs found between features and labels")

        # Filter to common IDs
        features_matched = features_df[features_df[feature_id].isin(common_ids)].copy()
        labels_matched = labels_df[labels_df[label_id].isin(common_ids)].copy()

        # Sort by ID to ensure alignment
        features_matched = features_matched.sort_values(feature_id).reset_index(drop=True)
        labels_matched = labels_matched.sort_values(label_id).reset_index(drop=True)

        # Verify IDs match
        assert (features_matched[feature_id].values == labels_matched[label_id].values).all(), \
            "ID alignment failed after sorting"

        # Drop ID columns (no longer needed)
        features_matched = features_matched.drop(columns=[feature_id])
        labels_matched = labels_matched.drop(columns=[label_id])

        logger.info(f"Merged data size: {len(features_matched)} samples")
        logger.info(f"Features shape after merge: {features_matched.shape}")
        logger.info(f"Labels shape after merge: {labels_matched.shape}")

        # Store stats
        self.stats['merged_size'] = len(features_matched)
        self.stats['matched_ids'] = len(common_ids)
        self.stats['unmatched_features'] = len(feature_ids - common_ids)
        self.stats['unmatched_labels'] = len(label_ids - common_ids)

        return features_matched, labels_matched

    def shuffle_and_split(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        test_size: float = 0.25,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Shuffle data and split into train/test sets with stratification.

        Args:
            features: Features DataFrame
            labels: One-hot encoded labels DataFrame
            test_size: Proportion of data for testing (default: 0.25 for 3:1 split)
            stratify: Whether to stratify split by labels (default: True)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Shuffling and splitting data...")

        # For stratification, convert one-hot back to single label
        # This is needed because train_test_split stratify requires 1D array
        if stratify:
            # Convert one-hot to class indices
            stratify_labels = labels.values.argmax(axis=1)
            logger.info(f"Using stratification with class distribution: "
                       f"{np.bincount(stratify_labels)}")
        else:
            stratify_labels = None

        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_labels,
            shuffle=True  # Explicit shuffle
        )

        # Log split statistics
        logger.info(f"Train set: {len(X_train)} samples ({len(X_train)/len(features)*100:.1f}%)")
        logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(features)*100:.1f}%)")

        # Log class distribution in splits
        train_dist = y_train.sum(axis=0)
        test_dist = y_test.sum(axis=0)

        logger.info("Train set class distribution:")
        for col, count in train_dist.items():
            logger.info(f"  {col}: {count} ({count/len(y_train)*100:.1f}%)")

        logger.info("Test set class distribution:")
        for col, count in test_dist.items():
            logger.info(f"  {col}: {count} ({count/len(y_test)*100:.1f}%)")

        # Store split stats
        self.stats['train_size'] = len(X_train)
        self.stats['test_size'] = len(X_test)
        self.stats['train_class_dist'] = train_dist.to_dict()
        self.stats['test_class_dist'] = test_dist.to_dict()

        return X_train, X_test, y_train, y_test

    def get_stats(self) -> Dict[str, Any]:
        """
        Get preprocessing statistics.

        Returns:
            Dictionary of preprocessing statistics
        """
        return self.stats.copy()


def preprocess_kepler_data(
    features_path: str,
    labels_path: str,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Complete preprocessing pipeline for Kepler exoplanet data.

    This function:
    1. Loads features and labels from CSV files
    2. One-hot encodes three-class disposition labels
    3. Merges features with labels by index
    4. Shuffles data randomly (with fixed seed)
    5. Splits into stratified train/test sets (3:1 ratio)

    Args:
        features_path: Path to features CSV (koi_lightcurve_features_no_label.csv)
        labels_path: Path to labels CSV (q1_q17_dr25_koi.csv)
        test_size: Proportion for test set (default: 0.25 for 3:1 split)
        random_state: Random seed for reproducibility (default: 42)
        stratify: Use stratified sampling (default: True)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, stats)
        - X_train: Training features
        - X_test: Test features
        - y_train: Training labels (one-hot encoded)
        - y_test: Test labels (one-hot encoded)
        - stats: Dictionary of preprocessing statistics

    Example:
        >>> X_train, X_test, y_train, y_test, stats = preprocess_kepler_data(
        ...     'koi_lightcurve_features_no_label.csv',
        ...     'q1_q17_dr25_koi.csv'
        ... )
        >>> print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        >>> print(f"Classes: {y_train.columns.tolist()}")
    """
    logger.info("=" * 80)
    logger.info("Starting Kepler Exoplanet Data Preprocessing Pipeline")
    logger.info("=" * 80)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_state=random_state)

    # Step 1: Load data
    features_df, labels_df = preprocessor.load_data(features_path, labels_path)

    # Step 2: One-hot encode labels
    encoded_labels = preprocessor.encode_labels(labels_df)

    # Step 3: Merge features and labels
    features_aligned, labels_aligned = preprocessor.merge_data(
        features_df,
        encoded_labels
    )

    # Step 4: Shuffle and split
    X_train, X_test, y_train, y_test = preprocessor.shuffle_and_split(
        features_aligned,
        labels_aligned,
        test_size=test_size,
        stratify=stratify
    )

    # Get statistics
    stats = preprocessor.get_stats()

    logger.info("=" * 80)
    logger.info("Preprocessing Pipeline Complete!")
    logger.info("=" * 80)

    return X_train, X_test, y_train, y_test, stats


def validate_preprocessed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> bool:
    """
    Validate preprocessed data for common issues.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        True if validation passes

    Raises:
        AssertionError: If validation fails
    """
    logger.info("Validating preprocessed data...")

    # Check no overlap between train and test indices
    assert len(set(X_train.index) & set(X_test.index)) == 0, \
        "Train and test sets have overlapping indices"

    # Check matching lengths
    assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
    assert len(X_test) == len(y_test), "X_test and y_test length mismatch"

    # Check no missing values in labels
    assert not y_train.isnull().any().any(), "Missing values in y_train"
    assert not y_test.isnull().any().any(), "Missing values in y_test"

    # Check one-hot encoding validity
    assert (y_train.sum(axis=1) == 1).all(), "Invalid one-hot encoding in y_train"
    assert (y_test.sum(axis=1) == 1).all(), "Invalid one-hot encoding in y_test"

    # Check same columns in train and test
    assert list(X_train.columns) == list(X_test.columns), \
        "Train and test feature columns don't match"
    assert list(y_train.columns) == list(y_test.columns), \
        "Train and test label columns don't match"

    # Check label columns are the expected three classes
    expected_classes = sorted(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
    assert list(y_train.columns) == expected_classes, \
        f"Label columns {list(y_train.columns)} don't match expected {expected_classes}"

    logger.info("✅ Validation passed!")
    return True


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) >= 3:
        features_file = sys.argv[1]
        labels_file = sys.argv[2]
    else:
        features_file = "koi_lightcurve_features_no_label.csv"
        labels_file = "q1_q17_dr25_koi.csv"

    print(f"Processing files:")
    print(f"  Features: {features_file}")
    print(f"  Labels: {labels_file}")

    try:
        X_train, X_test, y_train, y_test, stats = preprocess_kepler_data(
            features_file,
            labels_file
        )

        # Validate results
        validate_preprocessed_data(X_train, X_test, y_train, y_test)

        # Print summary
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY")
        print("=" * 80)
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"\nLabel classes: {list(y_train.columns)}")
        print(f"\nStatistics: {stats}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
