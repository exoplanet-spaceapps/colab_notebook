"""
Unit tests for model inference functionality

Tests cover:
- Single sample prediction
- Batch prediction
- Prediction probability outputs
- Class prediction accuracy
- Prediction consistency
- Error handling
- Edge cases
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


@pytest.mark.unit
class TestSinglePrediction:
    """Test single sample prediction"""

    def test_single_sample_prediction_shape(self, trained_model, sample_train_test_split):
        """Verify single prediction output shape"""
        X_test = sample_train_test_split[1]
        single_sample = X_test[0:1]

        prediction = trained_model.predict(single_sample, verbose=0)

        assert prediction.shape == (1, 3), "Should output (1, 3) for single sample"

    def test_single_sample_probability_sum(self, trained_model, sample_train_test_split):
        """Verify prediction probabilities sum to 1"""
        X_test = sample_train_test_split[1]
        single_sample = X_test[0:1]

        prediction = trained_model.predict(single_sample, verbose=0)

        assert np.isclose(prediction.sum(), 1.0, atol=1e-6), "Probabilities should sum to 1"

    def test_single_sample_probability_range(self, trained_model, sample_train_test_split):
        """Verify probabilities are in [0, 1] range"""
        X_test = sample_train_test_split[1]
        single_sample = X_test[0:1]

        prediction = trained_model.predict(single_sample, verbose=0)

        assert np.all(prediction >= 0), "Probabilities should be >= 0"
        assert np.all(prediction <= 1), "Probabilities should be <= 1"

    def test_single_sample_class_prediction(self, trained_model, sample_train_test_split):
        """Test class prediction from probabilities"""
        X_test = sample_train_test_split[1]
        single_sample = X_test[0:1]

        prediction = trained_model.predict(single_sample, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0] + 1  # +1 for 1-indexed labels

        assert predicted_class in [1, 2, 3], "Predicted class should be 1, 2, or 3"

    def test_prediction_deterministic(self, trained_model, sample_train_test_split):
        """Verify predictions are deterministic (same input = same output)"""
        X_test = sample_train_test_split[1]
        single_sample = X_test[0:1]

        pred1 = trained_model.predict(single_sample, verbose=0)
        pred2 = trained_model.predict(single_sample, verbose=0)

        assert np.allclose(pred1, pred2), "Predictions should be deterministic"


@pytest.mark.unit
class TestBatchPrediction:
    """Test batch prediction functionality"""

    def test_batch_prediction_shape(self, trained_model, sample_train_test_split):
        """Verify batch prediction output shape"""
        X_test = sample_train_test_split[1]
        batch_size = 32
        batch = X_test[:batch_size]

        predictions = trained_model.predict(batch, verbose=0)

        assert predictions.shape == (batch_size, 3), f"Should output ({batch_size}, 3)"

    def test_variable_batch_sizes(self, trained_model, sample_train_test_split):
        """Test predictions with different batch sizes"""
        X_test = sample_train_test_split[1]

        for batch_size in [1, 8, 16, 32, 64]:
            batch = X_test[:batch_size]
            predictions = trained_model.predict(batch, verbose=0)

            assert predictions.shape[0] == batch_size, f"Batch size {batch_size} should work"
            assert predictions.shape[1] == 3, "Should have 3 class probabilities"

    def test_all_probabilities_sum_to_one(self, trained_model, sample_train_test_split):
        """Verify all predictions have probabilities summing to 1"""
        X_test = sample_train_test_split[1]
        batch = X_test[:50]

        predictions = trained_model.predict(batch, verbose=0)
        sums = predictions.sum(axis=1)

        assert np.allclose(sums, 1.0, atol=1e-6), "All probability sums should be 1"

    def test_batch_vs_sequential_consistency(self, trained_model, sample_train_test_split):
        """Verify batch prediction matches sequential predictions"""
        X_test = sample_train_test_split[1]
        batch_size = 10
        batch = X_test[:batch_size]

        # Batch prediction
        batch_predictions = trained_model.predict(batch, verbose=0)

        # Sequential predictions
        sequential_predictions = []
        for i in range(batch_size):
            pred = trained_model.predict(batch[i:i+1], verbose=0)
            sequential_predictions.append(pred[0])

        sequential_predictions = np.array(sequential_predictions)

        assert np.allclose(batch_predictions, sequential_predictions, rtol=1e-5), \
            "Batch and sequential predictions should match"

    def test_large_batch_prediction(self, trained_model, sample_train_test_split):
        """Test prediction on large batch"""
        X_test = sample_train_test_split[1]

        predictions = trained_model.predict(X_test, verbose=0)

        assert predictions.shape == (len(X_test), 3), "Should handle full test set"
        assert not np.any(np.isnan(predictions)), "Should not contain NaN"
        assert not np.any(np.isinf(predictions)), "Should not contain inf"


@pytest.mark.unit
class TestPredictionAccuracy:
    """Test prediction accuracy and correctness"""

    def test_prediction_accuracy(self, trained_model, sample_train_test_split, encoded_labels):
        """Calculate and verify prediction accuracy"""
        X_test, y_test = sample_train_test_split[1], sample_train_test_split[3]
        y_test_encoded = encoded_labels[1]

        predictions = trained_model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test_encoded, axis=1)

        accuracy = np.mean(predicted_classes == true_classes)

        # Model should have at least some accuracy (better than random)
        assert accuracy > 0.33, "Accuracy should be better than random (>33%)"

    def test_class_prediction_distribution(self, trained_model, sample_train_test_split):
        """Verify predicted class distribution is reasonable"""
        X_test = sample_train_test_split[1]

        predictions = trained_model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1) + 1

        # Count predictions for each class
        class_counts = np.bincount(predicted_classes.astype(int), minlength=4)[1:]  # Skip index 0

        # All classes should have at least some predictions
        assert all(count > 0 for count in class_counts), "All classes should be predicted at least once"

    def test_confidence_scores(self, trained_model, sample_train_test_split):
        """Test prediction confidence scores"""
        X_test = sample_train_test_split[1]

        predictions = trained_model.predict(X_test[:100], verbose=0)
        max_confidences = predictions.max(axis=1)

        # Check confidence distribution
        assert np.all(max_confidences >= 0.33), "Max confidence should be >= 1/3"
        assert np.all(max_confidences <= 1.0), "Max confidence should be <= 1"

        # At least some predictions should be confident (>0.8)
        confident_predictions = (max_confidences > 0.8).sum()
        assert confident_predictions > 0, "Should have some confident predictions"

    def test_confusion_matrix_calculation(self, trained_model, sample_train_test_split, encoded_labels):
        """Test confusion matrix calculation"""
        from sklearn.metrics import confusion_matrix

        X_test = sample_train_test_split[1]
        y_test_encoded = encoded_labels[1]

        predictions = trained_model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test_encoded, axis=1)

        cm = confusion_matrix(true_classes, predicted_classes)

        assert cm.shape == (3, 3), "Confusion matrix should be 3x3"
        assert cm.sum() == len(X_test), "Total should equal number of samples"


@pytest.mark.unit
class TestPredictionConsistency:
    """Test prediction consistency across different scenarios"""

    def test_multiple_predictions_consistent(self, trained_model, sample_train_test_split):
        """Verify multiple predictions are consistent"""
        X_test = sample_train_test_split[1]
        sample = X_test[0:1]

        predictions = [trained_model.predict(sample, verbose=0) for _ in range(5)]

        # All predictions should be identical
        for pred in predictions[1:]:
            assert np.allclose(predictions[0], pred), "Predictions should be consistent"

    def test_prediction_order_independence(self, trained_model, sample_train_test_split):
        """Verify prediction order doesn't affect results"""
        X_test = sample_train_test_split[1]
        samples = X_test[:10]

        # Forward order
        forward_preds = trained_model.predict(samples, verbose=0)

        # Reverse order
        reverse_preds = trained_model.predict(samples[::-1], verbose=0)[::-1]

        assert np.allclose(forward_preds, reverse_preds, rtol=1e-5), \
            "Prediction order should not matter"

    def test_prediction_after_multiple_calls(self, trained_model, sample_train_test_split):
        """Verify predictions remain consistent after multiple calls"""
        X_test = sample_train_test_split[1]
        sample = X_test[0:1]

        # Get initial prediction
        initial_pred = trained_model.predict(sample, verbose=0)

        # Make many predictions on different data
        for i in range(1, 50):
            _ = trained_model.predict(X_test[i:i+1], verbose=0)

        # Check original sample again
        final_pred = trained_model.predict(sample, verbose=0)

        assert np.allclose(initial_pred, final_pred), \
            "Predictions should remain consistent"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_prediction_with_zeros(self, trained_model):
        """Test prediction with all-zero input"""
        zero_input = np.zeros((1, 3197))

        prediction = trained_model.predict(zero_input, verbose=0)

        assert prediction.shape == (1, 3), "Should handle zero input"
        assert np.isclose(prediction.sum(), 1.0), "Probabilities should sum to 1"

    def test_prediction_with_extreme_values(self, trained_model):
        """Test prediction with extreme input values"""
        # Very large values
        large_input = np.ones((1, 3197)) * 1000

        prediction = trained_model.predict(large_input, verbose=0)

        assert not np.any(np.isnan(prediction)), "Should not produce NaN with large values"
        assert not np.any(np.isinf(prediction)), "Should not produce inf with large values"

    def test_prediction_with_negative_values(self, trained_model):
        """Test prediction with negative input values"""
        negative_input = np.ones((1, 3197)) * -100

        prediction = trained_model.predict(negative_input, verbose=0)

        assert prediction.shape == (1, 3), "Should handle negative input"
        assert np.all(prediction >= 0), "Probabilities should be non-negative"

    def test_prediction_with_mixed_values(self, trained_model):
        """Test prediction with mixed positive/negative values"""
        mixed_input = np.random.randn(10, 3197) * 1000

        predictions = trained_model.predict(mixed_input, verbose=0)

        assert predictions.shape == (10, 3), "Should handle mixed values"
        assert np.allclose(predictions.sum(axis=1), 1.0), "All should sum to 1"

    def test_minimum_input_length(self, trained_model):
        """Test with minimum valid input (single sample)"""
        single_sample = np.random.randn(1, 3197)

        prediction = trained_model.predict(single_sample, verbose=0)

        assert prediction.shape == (1, 3), "Should handle single sample"


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in inference"""

    def test_wrong_input_shape(self, trained_model):
        """Test error handling for wrong input shape"""
        wrong_shape = np.random.randn(10, 100)  # Wrong feature count

        with pytest.raises((ValueError, Exception)):
            trained_model.predict(wrong_shape, verbose=0)

    def test_wrong_input_dimensions(self, trained_model):
        """Test error handling for wrong number of dimensions"""
        wrong_dims = np.random.randn(3197)  # 1D instead of 2D

        with pytest.raises((ValueError, Exception)):
            trained_model.predict(wrong_dims, verbose=0)

    def test_empty_input(self, trained_model):
        """Test error handling for empty input"""
        empty_input = np.array([]).reshape(0, 3197)

        # Some frameworks might handle this differently
        try:
            prediction = trained_model.predict(empty_input, verbose=0)
            assert prediction.shape == (0, 3), "Should return empty prediction"
        except (ValueError, Exception):
            pass  # Also acceptable to raise an error

    def test_non_numeric_input(self, trained_model):
        """Test error handling for non-numeric input"""
        with pytest.raises((TypeError, ValueError)):
            trained_model.predict([["a", "b", "c"]], verbose=0)


@pytest.mark.unit
class TestPredictionPipeline:
    """Test complete prediction pipeline"""

    def test_end_to_end_prediction(self, trained_model, sample_train_test_split):
        """Test complete prediction pipeline"""
        X_test, y_test = sample_train_test_split[1], sample_train_test_split[3]

        # Step 1: Get predictions
        predictions = trained_model.predict(X_test, verbose=0)

        # Step 2: Extract class labels
        predicted_classes = np.argmax(predictions, axis=1) + 1

        # Step 3: Get confidence scores
        confidence_scores = predictions.max(axis=1)

        # Verify pipeline outputs
        assert len(predicted_classes) == len(X_test), "Should predict all samples"
        assert all(c in [1, 2, 3] for c in predicted_classes), "Classes should be valid"
        assert len(confidence_scores) == len(X_test), "Should have confidence for all"

    def test_prediction_with_preprocessing(self, trained_model, sample_data):
        """Test prediction with data preprocessing"""
        from sklearn.preprocessing import StandardScaler

        # Extract features
        X = sample_data.drop('LABEL', axis=1).values

        # Preprocess
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict
        predictions = trained_model.predict(X_scaled[:10], verbose=0)

        assert predictions.shape == (10, 3), "Preprocessed prediction should work"

    def test_prediction_metrics_calculation(self, trained_model, sample_train_test_split, encoded_labels):
        """Test calculation of prediction metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        X_test = sample_train_test_split[1]
        y_test_encoded = encoded_labels[1]

        predictions = trained_model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test_encoded, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=0)
        recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=0)
        f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=0)

        # Verify metrics are valid
        assert 0 <= accuracy <= 1, "Accuracy should be in [0, 1]"
        assert 0 <= precision <= 1, "Precision should be in [0, 1]"
        assert 0 <= recall <= 1, "Recall should be in [0, 1]"
        assert 0 <= f1 <= 1, "F1 should be in [0, 1]"


@pytest.mark.unit
def test_inference_reproducibility(trained_model, sample_train_test_split):
    """Test that inference is reproducible across sessions"""
    X_test = sample_train_test_split[1]

    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # First prediction
    pred1 = trained_model.predict(X_test[:10], verbose=0)

    # Reset seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Second prediction
    pred2 = trained_model.predict(X_test[:10], verbose=0)

    assert np.allclose(pred1, pred2), "Predictions should be reproducible"
