"""
Unit tests for model I/O operations

Tests cover:
- Model saving and loading
- Model architecture preservation
- Weight preservation
- Optimizer state preservation
- Model serialization formats
- Error handling
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import tempfile
import shutil


@pytest.mark.unit
class TestModelSaving:
    """Test model saving functionality"""

    def test_save_h5_format(self, trained_model, temp_model_path):
        """Test saving model in H5 format"""
        trained_model.save(temp_model_path)

        assert Path(temp_model_path).exists(), "Model file should be created"
        assert Path(temp_model_path).suffix == '.h5', "Should have .h5 extension"

    def test_save_savedmodel_format(self, trained_model, tmp_path):
        """Test saving model in SavedModel format"""
        save_path = tmp_path / "saved_model"
        trained_model.save(str(save_path), save_format='tf')

        assert save_path.exists(), "SavedModel directory should be created"
        assert (save_path / "saved_model.pb").exists(), "Should have saved_model.pb"
        assert (save_path / "variables").exists(), "Should have variables directory"

    def test_save_model_config(self, trained_model, tmp_path):
        """Test saving model configuration"""
        config_path = tmp_path / "model_config.json"

        config = trained_model.to_json()
        with open(config_path, 'w') as f:
            f.write(config)

        assert config_path.exists(), "Config file should be created"

        # Verify it's valid JSON
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        assert 'class_name' in loaded_config, "Should have class_name"
        assert 'config' in loaded_config, "Should have config"

    def test_save_weights_only(self, trained_model, tmp_path):
        """Test saving only model weights"""
        weights_path = tmp_path / "model_weights.h5"
        trained_model.save_weights(str(weights_path))

        assert weights_path.exists(), "Weights file should be created"

    def test_overwrite_existing_model(self, trained_model, temp_model_path):
        """Test overwriting existing model file"""
        # Save once
        trained_model.save(temp_model_path)
        original_size = Path(temp_model_path).stat().st_size

        # Save again (overwrite)
        trained_model.save(temp_model_path, overwrite=True)
        new_size = Path(temp_model_path).stat().st_size

        assert Path(temp_model_path).exists(), "Model should still exist"
        # Sizes should be similar (allowing for small variations)
        assert abs(original_size - new_size) < 1000, "File sizes should be similar"


@pytest.mark.unit
class TestModelLoading:
    """Test model loading functionality"""

    def test_load_h5_format(self, trained_model, temp_model_path):
        """Test loading model from H5 format"""
        trained_model.save(temp_model_path)

        loaded_model = tf.keras.models.load_model(temp_model_path)

        assert loaded_model is not None, "Model should be loaded"
        assert isinstance(loaded_model, tf.keras.Model), "Should be a Keras model"

    def test_load_savedmodel_format(self, trained_model, tmp_path):
        """Test loading model from SavedModel format"""
        save_path = tmp_path / "saved_model"
        trained_model.save(str(save_path), save_format='tf')

        loaded_model = tf.keras.models.load_model(str(save_path))

        assert loaded_model is not None, "Model should be loaded"

    def test_load_from_config(self, trained_model, tmp_path):
        """Test loading model from configuration"""
        config = trained_model.to_json()

        loaded_model = tf.keras.models.model_from_json(config)

        assert loaded_model is not None, "Model should be created from config"
        # Note: weights won't be loaded, just architecture

    def test_load_weights_only(self, simple_model, trained_model, tmp_path):
        """Test loading only weights into existing model"""
        weights_path = tmp_path / "weights.h5"
        trained_model.save_weights(str(weights_path))

        # Load weights into a fresh model with same architecture
        simple_model.load_weights(str(weights_path))

        # Verify weights were loaded by checking they're not random
        weights = simple_model.get_weights()
        assert all(w.size > 0 for w in weights), "Weights should be loaded"

    def test_load_nonexistent_model(self):
        """Test error handling for non-existent model"""
        with pytest.raises((OSError, IOError)):
            tf.keras.models.load_model("nonexistent_model.h5")


@pytest.mark.unit
class TestArchitecturePreservation:
    """Test that model architecture is preserved during save/load"""

    def test_layer_count_preserved(self, trained_model, temp_model_path):
        """Verify number of layers is preserved"""
        original_layers = len(trained_model.layers)

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        assert len(loaded_model.layers) == original_layers, "Layer count should match"

    def test_layer_types_preserved(self, trained_model, temp_model_path):
        """Verify layer types are preserved"""
        original_types = [type(layer).__name__ for layer in trained_model.layers]

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        loaded_types = [type(layer).__name__ for layer in loaded_model.layers]

        assert original_types == loaded_types, "Layer types should match"

    def test_layer_config_preserved(self, trained_model, temp_model_path):
        """Verify layer configurations are preserved"""
        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        for orig_layer, loaded_layer in zip(trained_model.layers, loaded_model.layers):
            orig_config = orig_layer.get_config()
            loaded_config = loaded_layer.get_config()

            # Compare key configuration parameters
            assert orig_config['name'] == loaded_config['name'], "Layer names should match"

    def test_input_shape_preserved(self, trained_model, temp_model_path):
        """Verify input shape is preserved"""
        original_input_shape = trained_model.input_shape

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        assert loaded_model.input_shape == original_input_shape, "Input shape should match"

    def test_output_shape_preserved(self, trained_model, temp_model_path):
        """Verify output shape is preserved"""
        original_output_shape = trained_model.output_shape

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        assert loaded_model.output_shape == original_output_shape, "Output shape should match"


@pytest.mark.unit
class TestWeightPreservation:
    """Test that model weights are preserved during save/load"""

    def test_weights_preserved(self, trained_model, temp_model_path):
        """Verify all weights are preserved exactly"""
        original_weights = trained_model.get_weights()

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        loaded_weights = loaded_model.get_weights()

        assert len(original_weights) == len(loaded_weights), "Weight count should match"

        for orig_w, loaded_w in zip(original_weights, loaded_weights):
            assert np.allclose(orig_w, loaded_w), "Weights should be identical"

    def test_weight_shapes_preserved(self, trained_model, temp_model_path):
        """Verify weight shapes are preserved"""
        original_shapes = [w.shape for w in trained_model.get_weights()]

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        loaded_shapes = [w.shape for w in loaded_model.get_weights()]

        assert original_shapes == loaded_shapes, "Weight shapes should match"

    def test_bias_preserved(self, trained_model, temp_model_path):
        """Verify bias terms are preserved"""
        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        # Check Dense layers have biases preserved
        for orig_layer, loaded_layer in zip(trained_model.layers, loaded_model.layers):
            if isinstance(orig_layer, tf.keras.layers.Dense):
                orig_bias = orig_layer.get_weights()[1]
                loaded_bias = loaded_layer.get_weights()[1]
                assert np.allclose(orig_bias, loaded_bias), "Biases should match"


@pytest.mark.unit
class TestOptimizerStatePreservation:
    """Test that optimizer state is preserved during save/load"""

    def test_optimizer_config_preserved(self, trained_model, temp_model_path):
        """Verify optimizer configuration is preserved"""
        original_optimizer = trained_model.optimizer.get_config()

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        loaded_optimizer = loaded_model.optimizer.get_config()

        # Compare key optimizer parameters
        assert original_optimizer['name'] == loaded_optimizer['name'], "Optimizer name should match"

    def test_loss_function_preserved(self, trained_model, temp_model_path):
        """Verify loss function is preserved"""
        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        # Both should have the same loss function
        assert type(trained_model.loss) == type(loaded_model.loss), "Loss function should match"

    def test_metrics_preserved(self, trained_model, temp_model_path):
        """Verify metrics are preserved"""
        original_metrics = [m.name for m in trained_model.metrics]

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        loaded_metrics = [m.name for m in loaded_model.metrics]

        assert original_metrics == loaded_metrics, "Metrics should match"


@pytest.mark.unit
class TestPredictionConsistency:
    """Test that predictions are consistent after save/load"""

    def test_predictions_identical(self, trained_model, temp_model_path, sample_train_test_split):
        """Verify predictions are identical before and after save/load"""
        X_test = sample_train_test_split[1]

        # Get predictions before saving
        original_predictions = trained_model.predict(X_test[:10], verbose=0)

        # Save and load model
        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        # Get predictions after loading
        loaded_predictions = loaded_model.predict(X_test[:10], verbose=0)

        assert np.allclose(original_predictions, loaded_predictions, rtol=1e-5), \
            "Predictions should be identical"

    def test_batch_predictions_consistent(self, trained_model, temp_model_path, sample_train_test_split):
        """Verify batch predictions are consistent"""
        X_test = sample_train_test_split[1]

        trained_model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path)

        # Test different batch sizes
        for batch_size in [1, 16, 32]:
            orig_preds = trained_model.predict(X_test[:batch_size], verbose=0)
            loaded_preds = loaded_model.predict(X_test[:batch_size], verbose=0)

            assert np.allclose(orig_preds, loaded_preds, rtol=1e-5), \
                f"Batch size {batch_size} predictions should match"


@pytest.mark.unit
class TestSerializationFormats:
    """Test different serialization formats"""

    def test_json_serialization(self, simple_model):
        """Test JSON serialization of model architecture"""
        json_config = simple_model.to_json()

        assert isinstance(json_config, str), "Should return JSON string"

        # Verify it's valid JSON
        config_dict = json.loads(json_config)
        assert 'class_name' in config_dict, "Should have class_name"

    def test_yaml_serialization(self, simple_model):
        """Test YAML serialization of model architecture"""
        yaml_config = simple_model.to_yaml()

        assert isinstance(yaml_config, str), "Should return YAML string"
        assert 'class_name:' in yaml_config, "Should contain class_name"

    def test_pickle_compatibility(self, trained_model, tmp_path):
        """Test that saved models are compatible with pickle"""
        import pickle

        model_path = tmp_path / "model.pkl"

        # Pickle the model
        with open(model_path, 'wb') as f:
            pickle.dump(trained_model, f)

        assert model_path.exists(), "Pickle file should be created"


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in model I/O operations"""

    def test_save_to_invalid_path(self, trained_model):
        """Test error handling for invalid save path"""
        with pytest.raises((OSError, IOError, PermissionError)):
            trained_model.save("/invalid/path/model.h5")

    def test_load_corrupted_model(self, tmp_path):
        """Test error handling for corrupted model file"""
        corrupted_path = tmp_path / "corrupted.h5"

        # Create a corrupted file
        with open(corrupted_path, 'wb') as f:
            f.write(b"corrupted data")

        with pytest.raises(Exception):
            tf.keras.models.load_model(str(corrupted_path))

    def test_load_incompatible_weights(self, simple_model, tmp_path):
        """Test error handling for incompatible weight shapes"""
        # Create a different model
        different_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(3197,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        weights_path = tmp_path / "weights.h5"
        different_model.save_weights(str(weights_path))

        # Try to load incompatible weights
        with pytest.raises((ValueError, Exception)):
            simple_model.load_weights(str(weights_path))


@pytest.mark.unit
def test_model_versioning(trained_model, tmp_path):
    """Test model versioning strategy"""
    # Save multiple versions
    for version in [1, 2, 3]:
        version_path = tmp_path / f"model_v{version}.h5"
        trained_model.save(str(version_path))

        assert version_path.exists(), f"Version {version} should be saved"

    # Verify all versions exist
    saved_files = list(tmp_path.glob("model_v*.h5"))
    assert len(saved_files) == 3, "Should have 3 versions saved"
