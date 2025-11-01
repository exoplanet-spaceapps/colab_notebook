"""
Unit tests for data preprocessing functionality

Tests cover:
- Data loading and validation
- Train/test splitting
- One-hot encoding
- Data normalization
- Feature extraction
- Class distribution analysis
"""

import pytest
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


@pytest.mark.unit
class TestDataLoading:
    """Test data loading and validation"""

    def test_sample_data_shape(self, sample_data):
        """Verify sample data has correct shape"""
        assert sample_data.shape[0] == 1000, "Should have 1000 samples"
        assert sample_data.shape[1] == 3198, "Should have 3198 columns (1 label + 3197 features)"

    def test_sample_data_columns(self, sample_data):
        """Verify sample data has required columns"""
        assert 'LABEL' in sample_data.columns, "Should have LABEL column"
        flux_columns = [col for col in sample_data.columns if col.startswith('FLUX.')]
        assert len(flux_columns) == 3197, "Should have 3197 FLUX features"

    def test_label_values(self, sample_data):
        """Verify labels are in valid range [1, 2, 3]"""
        labels = sample_data['LABEL'].unique()
        assert set(labels).issubset({1, 2, 3}), "Labels should only be 1, 2, or 3"

    def test_no_missing_values(self, sample_data):
        """Verify no missing values in dataset"""
        assert not sample_data.isnull().any().any(), "Should have no missing values"

    def test_feature_dtypes(self, sample_data):
        """Verify features are numeric"""
        flux_cols = [col for col in sample_data.columns if col.startswith('FLUX.')]
        for col in flux_cols:
            assert np.issubdtype(sample_data[col].dtype, np.number), f"{col} should be numeric"


@pytest.mark.unit
class TestTrainTestSplit:
    """Test train/test splitting functionality"""

    def test_split_ratio(self, sample_train_test_split):
        """Verify 80/20 train/test split"""
        X_train, X_test, y_train, y_test = sample_train_test_split

        total_samples = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total_samples
        test_ratio = len(X_test) / total_samples

        assert abs(train_ratio - 0.8) < 0.01, "Train set should be ~80%"
        assert abs(test_ratio - 0.2) < 0.01, "Test set should be ~20%"

    def test_split_shapes_match(self, sample_train_test_split):
        """Verify X and y shapes match"""
        X_train, X_test, y_train, y_test = sample_train_test_split

        assert len(X_train) == len(y_train), "X_train and y_train should have same length"
        assert len(X_test) == len(y_test), "X_test and y_test should have same length"

    def test_feature_dimensions(self, sample_train_test_split):
        """Verify feature dimensions are preserved"""
        X_train, X_test, _, _ = sample_train_test_split

        assert X_train.shape[1] == 3197, "Should have 3197 features"
        assert X_test.shape[1] == 3197, "Should have 3197 features"

    def test_stratification(self, sample_train_test_split):
        """Verify stratified splitting maintains class distribution"""
        _, _, y_train, y_test = sample_train_test_split

        # Calculate class distributions
        train_dist = np.bincount(y_train.astype(int))[1:] / len(y_train)
        test_dist = np.bincount(y_test.astype(int))[1:] / len(y_test)

        # Distributions should be similar (within 5%)
        for train_pct, test_pct in zip(train_dist, test_dist):
            assert abs(train_pct - test_pct) < 0.05, "Class distributions should be similar"

    def test_no_data_leakage(self, sample_train_test_split):
        """Verify no overlap between train and test sets"""
        X_train, X_test, _, _ = sample_train_test_split

        # Convert to sets of tuples for comparison
        train_samples = set(map(tuple, X_train))
        test_samples = set(map(tuple, X_test))

        assert len(train_samples & test_samples) == 0, "Train and test should not overlap"


@pytest.mark.unit
class TestOneHotEncoding:
    """Test one-hot encoding functionality"""

    def test_encoding_shape(self, encoded_labels):
        """Verify one-hot encoding creates correct shape"""
        y_train_encoded, y_test_encoded = encoded_labels

        assert y_train_encoded.shape[1] == 3, "Should have 3 classes"
        assert y_test_encoded.shape[1] == 3, "Should have 3 classes"

    def test_encoding_values(self, encoded_labels):
        """Verify one-hot encoding produces binary values"""
        y_train_encoded, y_test_encoded = encoded_labels

        # Check all values are 0 or 1
        assert np.all(np.isin(y_train_encoded, [0, 1])), "Should only contain 0 and 1"
        assert np.all(np.isin(y_test_encoded, [0, 1])), "Should only contain 0 and 1"

    def test_encoding_sum(self, encoded_labels):
        """Verify each row sums to 1 (only one class per sample)"""
        y_train_encoded, y_test_encoded = encoded_labels

        assert np.allclose(y_train_encoded.sum(axis=1), 1), "Each row should sum to 1"
        assert np.allclose(y_test_encoded.sum(axis=1), 1), "Each row should sum to 1"

    def test_encoding_reversibility(self, sample_train_test_split, encoded_labels):
        """Verify encoding can be reversed to original labels"""
        _, _, y_train, y_test = sample_train_test_split
        y_train_encoded, y_test_encoded = encoded_labels

        # Reverse encoding
        y_train_decoded = np.argmax(y_train_encoded, axis=1) + 1
        y_test_decoded = np.argmax(y_test_encoded, axis=1) + 1

        assert np.array_equal(y_train, y_train_decoded), "Train labels should match"
        assert np.array_equal(y_test, y_test_decoded), "Test labels should match"

    def test_class_mapping(self):
        """Test specific class mappings"""
        # Label 1 -> [1, 0, 0]
        # Label 2 -> [0, 1, 0]
        # Label 3 -> [0, 0, 1]

        labels = np.array([1, 2, 3])
        encoded = to_categorical(labels - 1, num_classes=3)

        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        assert np.array_equal(encoded, expected), "Class mapping should be correct"


@pytest.mark.unit
class TestDataNormalization:
    """Test data normalization and scaling"""

    def test_standard_scaling(self, sample_data):
        """Test standard scaling (mean=0, std=1)"""
        from sklearn.preprocessing import StandardScaler

        X = sample_data.drop('LABEL', axis=1).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check mean close to 0 and std close to 1
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10), "Mean should be ~0"
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10), "Std should be ~1"

    def test_minmax_scaling(self, sample_data):
        """Test min-max scaling (range [0, 1])"""
        from sklearn.preprocessing import MinMaxScaler

        X = sample_data.drop('LABEL', axis=1).values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.min() >= 0, "Minimum should be >= 0"
        assert X_scaled.max() <= 1, "Maximum should be <= 1"

    def test_robust_scaling(self, sample_data):
        """Test robust scaling (resistant to outliers)"""
        from sklearn.preprocessing import RobustScaler

        X = sample_data.drop('LABEL', axis=1).values
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Verify scaling doesn't produce extreme values
        assert not np.any(np.isinf(X_scaled)), "Should not produce infinite values"
        assert not np.any(np.isnan(X_scaled)), "Should not produce NaN values"


@pytest.mark.unit
class TestClassDistribution:
    """Test class distribution analysis"""

    def test_class_counts(self, sample_data):
        """Verify class distribution"""
        class_counts = sample_data['LABEL'].value_counts().sort_index()

        assert len(class_counts) == 3, "Should have 3 classes"
        assert all(class_counts > 0), "All classes should have samples"

    def test_class_imbalance_ratio(self, sample_data):
        """Calculate class imbalance ratio"""
        class_counts = sample_data['LABEL'].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()

        imbalance_ratio = max_count / min_count

        # Just verify it's calculable, don't enforce specific ratio
        assert imbalance_ratio >= 1.0, "Imbalance ratio should be >= 1"

    def test_stratification_maintains_distribution(self, sample_data):
        """Verify stratified split maintains class distribution"""
        X = sample_data.drop('LABEL', axis=1).values
        y = sample_data['LABEL'].values

        original_dist = np.bincount(y.astype(int))[1:] / len(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_dist = np.bincount(y_train.astype(int))[1:] / len(y_train)
        test_dist = np.bincount(y_test.astype(int))[1:] / len(y_test)

        # All distributions should be similar
        for orig, train, test in zip(original_dist, train_dist, test_dist):
            assert abs(orig - train) < 0.05, "Train distribution should match original"
            assert abs(orig - test) < 0.05, "Test distribution should match original"


@pytest.mark.unit
class TestDataValidation:
    """Test data validation and sanity checks"""

    def test_feature_variance(self, sample_data):
        """Verify features have non-zero variance"""
        X = sample_data.drop('LABEL', axis=1).values
        variances = X.var(axis=0)

        # Most features should have non-zero variance
        non_zero_variance = (variances > 0).sum()
        assert non_zero_variance / len(variances) > 0.95, "Most features should have variance"

    def test_no_constant_features(self, sample_data):
        """Verify no features are constant"""
        X = sample_data.drop('LABEL', axis=1)

        for col in X.columns:
            assert X[col].nunique() > 1, f"{col} should not be constant"

    def test_feature_correlations(self, sample_data):
        """Test feature correlation analysis"""
        X = sample_data.drop('LABEL', axis=1)

        # Calculate correlation matrix
        corr_matrix = X.corr()

        # Verify correlation matrix is valid
        assert corr_matrix.shape[0] == corr_matrix.shape[1], "Should be square matrix"
        assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1"

    def test_outlier_detection(self, sample_data):
        """Test outlier detection using IQR method"""
        X = sample_data.drop('LABEL', axis=1).values

        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Calculate outlier percentage
        outliers = ((X < lower_bound) | (X > upper_bound)).sum()
        outlier_pct = outliers / X.size

        # Should have some outliers but not too many
        assert outlier_pct < 0.1, "Outliers should be less than 10%"


@pytest.mark.unit
def test_preprocessing_pipeline(sample_data):
    """Test complete preprocessing pipeline"""
    from sklearn.preprocessing import StandardScaler

    # Step 1: Separate features and labels
    X = sample_data.drop('LABEL', axis=1).values
    y = sample_data['LABEL'].values

    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 3: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Encode labels
    y_train_encoded = to_categorical(y_train - 1, num_classes=3)
    y_test_encoded = to_categorical(y_test - 1, num_classes=3)

    # Verify final shapes
    assert X_train_scaled.shape == X_train.shape, "Scaling should preserve shape"
    assert X_test_scaled.shape == X_test.shape, "Scaling should preserve shape"
    assert y_train_encoded.shape == (len(y_train), 3), "Encoding should create 3 classes"
    assert y_test_encoded.shape == (len(y_test), 3), "Encoding should create 3 classes"
