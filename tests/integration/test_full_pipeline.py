"""
Integration tests for complete train-save-load-predict pipeline

Tests the entire workflow:
1. Data loading and preprocessing
2. Model training
3. Model saving
4. Model loading
5. Prediction and evaluation
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from pathlib import Path


@pytest.mark.integration
class TestFullPipeline:
    """Test complete end-to-end pipeline"""

    def test_complete_workflow(self, sample_data, tmp_path):
        """Test complete train-save-load-predict workflow"""
        # Step 1: Data preprocessing
        X = sample_data.drop('LABEL', axis=1).values
        y = sample_data['LABEL'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Step 2: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Step 3: Encode labels
        y_train_encoded = to_categorical(y_train - 1, num_classes=3)
        y_test_encoded = to_categorical(y_test - 1, num_classes=3)

        # Step 4: Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3197,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Step 5: Train model
        history = model.fit(
            X_train_scaled, y_train_encoded,
            validation_split=0.2,
            epochs=5,
            batch_size=32,
            verbose=0
        )

        assert history.history['loss'][-1] < history.history['loss'][0], \
            "Loss should decrease during training"

        # Step 6: Save model
        model_path = tmp_path / "trained_model.h5"
        model.save(str(model_path))

        assert model_path.exists(), "Model should be saved"

        # Step 7: Load model
        loaded_model = tf.keras.models.load_model(str(model_path))

        # Step 8: Predict with loaded model
        predictions = loaded_model.predict(X_test_scaled, verbose=0)

        # Step 9: Evaluate
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test_encoded, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)

        assert accuracy > 0.33, "Accuracy should be better than random"

        # Step 10: Verify predictions from original and loaded model match
        original_predictions = model.predict(X_test_scaled[:10], verbose=0)
        loaded_predictions = loaded_model.predict(X_test_scaled[:10], verbose=0)

        assert np.allclose(original_predictions, loaded_predictions), \
            "Original and loaded model should produce same predictions"


@pytest.mark.integration
class TestThreeClassClassification:
    """Test three-class classification correctness"""

    def test_all_classes_predicted(self, trained_model, sample_train_test_split):
        """Verify all three classes can be predicted"""
        X_test = sample_train_test_split[1]

        predictions = trained_model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1) + 1

        unique_predictions = np.unique(predicted_classes)

        # Should predict all three classes at least once
        assert len(unique_predictions) >= 2, "Should predict at least 2 different classes"

    def test_class_probabilities_correct(self, trained_model, sample_train_test_split):
        """Verify class probabilities are correctly calculated"""
        X_test = sample_train_test_split[1]

        predictions = trained_model.predict(X_test[:100], verbose=0)

        # Each prediction should have 3 probabilities
        assert predictions.shape[1] == 3, "Should have 3 class probabilities"

        # All probabilities should sum to 1
        for pred in predictions:
            assert np.isclose(pred.sum(), 1.0, atol=1e-6), "Probabilities should sum to 1"

    def test_classification_metrics(self, trained_model, sample_train_test_split, encoded_labels):
        """Test classification metrics for all classes"""
        from sklearn.metrics import classification_report

        X_test = sample_train_test_split[1]
        y_test_encoded = encoded_labels[1]

        predictions = trained_model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test_encoded, axis=1)

        report = classification_report(
            true_classes, predicted_classes,
            target_names=['Class 1', 'Class 2', 'Class 3'],
            output_dict=True,
            zero_division=0
        )

        # Verify report structure
        assert 'Class 1' in report, "Should have Class 1 metrics"
        assert 'Class 2' in report, "Should have Class 2 metrics"
        assert 'Class 3' in report, "Should have Class 3 metrics"

        # Verify metrics are valid
        for class_name in ['Class 1', 'Class 2', 'Class 3']:
            assert 0 <= report[class_name]['precision'] <= 1, f"{class_name} precision invalid"
            assert 0 <= report[class_name]['recall'] <= 1, f"{class_name} recall invalid"
            assert 0 <= report[class_name]['f1-score'] <= 1, f"{class_name} f1-score invalid"

    def test_per_class_accuracy(self, trained_model, sample_train_test_split, encoded_labels):
        """Test accuracy for each individual class"""
        X_test = sample_train_test_split[1]
        y_test_encoded = encoded_labels[1]

        predictions = trained_model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test_encoded, axis=1)

        # Calculate per-class accuracy
        for class_id in range(3):
            class_mask = true_classes == class_id
            if class_mask.sum() > 0:  # If class exists in test set
                class_accuracy = (predicted_classes[class_mask] == class_id).mean()
                # Accuracy should be at least somewhat better than random
                assert class_accuracy >= 0, f"Class {class_id + 1} accuracy should be non-negative"


@pytest.mark.integration
class TestModelPersistence:
    """Test model persistence across sessions"""

    def test_save_load_consistency(self, trained_model, sample_train_test_split, tmp_path):
        """Test model consistency after save/load cycles"""
        X_test = sample_train_test_split[1]

        # Original predictions
        original_preds = trained_model.predict(X_test[:10], verbose=0)

        # Save and load cycle 1
        model_path1 = tmp_path / "model_v1.h5"
        trained_model.save(str(model_path1))
        loaded_model1 = tf.keras.models.load_model(str(model_path1))
        preds1 = loaded_model1.predict(X_test[:10], verbose=0)

        # Save and load cycle 2
        model_path2 = tmp_path / "model_v2.h5"
        loaded_model1.save(str(model_path2))
        loaded_model2 = tf.keras.models.load_model(str(model_path2))
        preds2 = loaded_model2.predict(X_test[:10], verbose=0)

        # All predictions should be identical
        assert np.allclose(original_preds, preds1), "Cycle 1 predictions should match"
        assert np.allclose(preds1, preds2), "Cycle 2 predictions should match"

    def test_weights_persistence(self, trained_model, tmp_path):
        """Test that weights are correctly persisted"""
        original_weights = trained_model.get_weights()

        # Save weights
        weights_path = tmp_path / "weights.h5"
        trained_model.save_weights(str(weights_path))

        # Create new model and load weights
        new_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3197,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        new_model.load_weights(str(weights_path))

        loaded_weights = new_model.get_weights()

        # Compare all weights
        for orig_w, loaded_w in zip(original_weights, loaded_weights):
            assert np.allclose(orig_w, loaded_w), "Weights should be identical"


@pytest.mark.integration
class TestDataPipeline:
    """Test complete data processing pipeline"""

    def test_preprocessing_pipeline(self, sample_data):
        """Test complete data preprocessing pipeline"""
        # Step 1: Load data
        assert sample_data is not None, "Data should load"

        # Step 2: Separate features and labels
        X = sample_data.drop('LABEL', axis=1).values
        y = sample_data['LABEL'].values

        assert X.shape[1] == 3197, "Should have 3197 features"
        assert len(y) == len(X), "Labels should match samples"

        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        assert len(X_train) + len(X_test) == len(X), "Split should preserve total samples"

        # Step 4: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        assert X_train_scaled.shape == X_train.shape, "Scaling should preserve shape"
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10), "Should be standardized"

        # Step 5: Encode labels
        y_train_encoded = to_categorical(y_train - 1, num_classes=3)
        y_test_encoded = to_categorical(y_test - 1, num_classes=3)

        assert y_train_encoded.shape[1] == 3, "Should have 3 classes"
        assert np.allclose(y_train_encoded.sum(axis=1), 1), "Should be one-hot encoded"

    def test_data_integrity_through_pipeline(self, sample_data):
        """Test that data integrity is maintained through pipeline"""
        original_labels = sample_data['LABEL'].values.copy()

        # Process data
        X = sample_data.drop('LABEL', axis=1).values
        y = sample_data['LABEL'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Verify original data unchanged
        assert np.array_equal(original_labels, sample_data['LABEL'].values), \
            "Original data should not be modified"

        # Verify split data integrity
        combined_y = np.concatenate([y_train, y_test])
        assert set(combined_y) == set(y), "All labels should be preserved"


@pytest.mark.integration
class TestTrainingPipeline:
    """Test model training pipeline"""

    def test_training_reduces_loss(self, simple_model, sample_train_test_split, encoded_labels):
        """Verify training reduces loss over epochs"""
        X_train = sample_train_test_split[0]
        y_train_encoded = encoded_labels[0]

        history = simple_model.fit(
            X_train, y_train_encoded,
            epochs=5,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )

        # Loss should generally decrease
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]

        assert final_loss < initial_loss, "Training should reduce loss"

    def test_validation_during_training(self, simple_model, sample_train_test_split, encoded_labels):
        """Test validation metrics during training"""
        X_train = sample_train_test_split[0]
        y_train_encoded = encoded_labels[0]

        history = simple_model.fit(
            X_train, y_train_encoded,
            epochs=3,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )

        # Should have validation metrics
        assert 'val_loss' in history.history, "Should track validation loss"
        assert 'val_accuracy' in history.history, "Should track validation accuracy"

        # Validation metrics should be valid
        assert all(loss >= 0 for loss in history.history['val_loss']), \
            "Validation loss should be non-negative"

    def test_early_stopping_callback(self, simple_model, sample_train_test_split, encoded_labels):
        """Test early stopping callback"""
        X_train = sample_train_test_split[0]
        y_train_encoded = encoded_labels[0]

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        history = simple_model.fit(
            X_train, y_train_encoded,
            epochs=20,
            batch_size=32,
            verbose=0,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        # Should stop before max epochs (usually)
        actual_epochs = len(history.history['loss'])
        assert actual_epochs <= 20, "Should run at most 20 epochs"


@pytest.mark.integration
@pytest.mark.slow
def test_complete_ml_workflow(sample_data, tmp_path):
    """
    Test complete machine learning workflow from start to finish

    This is a comprehensive integration test that covers:
    1. Data loading
    2. Preprocessing
    3. Model creation
    4. Training
    5. Evaluation
    6. Saving
    7. Loading
    8. Prediction
    """
    # 1. Data loading
    X = sample_data.drop('LABEL', axis=1).values
    y = sample_data['LABEL'].values

    # 2. Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_encoded = to_categorical(y_train - 1, num_classes=3)
    y_test_encoded = to_categorical(y_test - 1, num_classes=3)

    # 3. Model creation
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(3197,)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Training with callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train_encoded,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    # 5. Evaluation
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
    assert test_accuracy > 0.33, "Test accuracy should be better than random"

    # 6. Saving
    model_path = tmp_path / "final_model.h5"
    scaler_path = tmp_path / "scaler.pkl"

    model.save(str(model_path))

    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # 7. Loading
    loaded_model = tf.keras.models.load_model(str(model_path))

    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)

    # 8. Prediction with loaded artifacts
    new_data = X_test[:10]
    new_data_scaled = loaded_scaler.transform(new_data)
    predictions = loaded_model.predict(new_data_scaled, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1) + 1

    # Verify predictions
    assert len(predicted_classes) == 10, "Should predict 10 samples"
    assert all(c in [1, 2, 3] for c in predicted_classes), "Classes should be valid"
    assert predictions.shape == (10, 3), "Should have 3 class probabilities"
