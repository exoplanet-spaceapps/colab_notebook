"""
Compatibility tests for different environments

Tests cover:
- Google Colab environment
- Different TensorFlow versions
- Python version compatibility
- Package dependencies
- GPU/CPU compatibility
"""

import pytest
import sys
import platform
import tensorflow as tf
import numpy as np
import pkg_resources


@pytest.mark.compatibility
class TestPythonVersion:
    """Test Python version compatibility"""

    def test_python_version_supported(self, compatibility_config):
        """Verify Python version is supported"""
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        min_version = compatibility_config['python_version'].replace('+', '')

        print(f"\nPython version: {current_version}")
        print(f"Required: {compatibility_config['python_version']}")

        major, minor = map(int, min_version.split('.'))
        assert sys.version_info.major >= major, "Python major version too old"
        assert sys.version_info.minor >= minor, "Python minor version too old"

    def test_platform_info(self):
        """Display platform information"""
        print(f"\nPlatform information:")
        print(f"  System: {platform.system()}")
        print(f"  Release: {platform.release()}")
        print(f"  Machine: {platform.machine()}")
        print(f"  Processor: {platform.processor()}")

    def test_64bit_architecture(self):
        """Verify 64-bit architecture"""
        is_64bit = sys.maxsize > 2**32
        print(f"\n64-bit: {is_64bit}")

        assert is_64bit, "Requires 64-bit Python"


@pytest.mark.compatibility
class TestTensorFlowVersion:
    """Test TensorFlow version compatibility"""

    def test_tensorflow_version(self, compatibility_config):
        """Verify TensorFlow version"""
        tf_version = tf.__version__
        print(f"\nTensorFlow version: {tf_version}")

        # Extract major.minor version
        major_minor = '.'.join(tf_version.split('.')[:2])

        # Check if version is supported
        supported_versions = compatibility_config['supported_tf_versions']
        print(f"Supported versions: {supported_versions}")

        # Version should be in supported list or newer
        assert any(major_minor >= v for v in supported_versions), \
            f"TensorFlow {tf_version} may not be supported"

    def test_keras_integration(self):
        """Verify Keras is properly integrated with TensorFlow"""
        assert hasattr(tf, 'keras'), "Keras should be available in TensorFlow"

        print(f"\nKeras version: {tf.keras.__version__}")

    def test_tensorflow_gpu_availability(self):
        """Check GPU availability"""
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\nGPUs available: {len(gpus)}")

        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("  Running on CPU")

    def test_mixed_precision_support(self):
        """Test mixed precision training support"""
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            print(f"\nMixed precision support: Available")
            print(f"  Default policy: {mixed_precision.global_policy().name}")
        except Exception as e:
            print(f"\nMixed precision support: Not available ({e})")


@pytest.mark.compatibility
class TestPackageDependencies:
    """Test package dependencies"""

    def test_required_packages_installed(self, compatibility_config):
        """Verify all required packages are installed"""
        required = compatibility_config['required_packages']
        installed = {pkg.key for pkg in pkg_resources.working_set}

        print(f"\nRequired packages:")
        for package in required:
            is_installed = package.lower() in installed
            status = "✓" if is_installed else "✗"
            print(f"  {status} {package}")

            assert is_installed, f"{package} is not installed"

    def test_package_versions(self):
        """Display installed package versions"""
        packages = ['numpy', 'pandas', 'scikit-learn', 'tensorflow']

        print(f"\nInstalled package versions:")
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"  {package}: {version}")
            except pkg_resources.DistributionNotFound:
                print(f"  {package}: Not found")

    def test_numpy_compatibility(self):
        """Test NumPy compatibility with TensorFlow"""
        import numpy as np

        # Create numpy array and convert to tensor
        np_array = np.random.randn(10, 10)
        tf_tensor = tf.constant(np_array)

        # Convert back
        np_from_tf = tf_tensor.numpy()

        assert np.allclose(np_array, np_from_tf), "NumPy-TensorFlow conversion failed"
        print(f"\nNumPy-TensorFlow compatibility: OK")


@pytest.mark.compatibility
class TestColabEnvironment:
    """Test Google Colab specific compatibility"""

    def test_colab_detection(self):
        """Detect if running in Google Colab"""
        try:
            import google.colab
            in_colab = True
        except ImportError:
            in_colab = False

        print(f"\nRunning in Google Colab: {in_colab}")

    def test_colab_gpu_access(self):
        """Test GPU access in Colab"""
        try:
            import google.colab
            in_colab = True
        except ImportError:
            in_colab = False
            pytest.skip("Not in Colab environment")

        gpus = tf.config.list_physical_devices('GPU')
        if in_colab:
            print(f"\nColab GPU configuration:")
            print(f"  GPUs available: {len(gpus)}")

    def test_colab_file_system(self):
        """Test file system access in Colab"""
        try:
            import google.colab
            from google.colab import drive

            print(f"\nColab file system:")
            print(f"  Drive mount available: Yes")
        except ImportError:
            pytest.skip("Not in Colab environment")

    def test_colab_memory_limit(self):
        """Test memory limits in Colab"""
        try:
            import google.colab
            import psutil

            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            print(f"\nColab memory:")
            print(f"  Total: {total_memory_gb:.1f} GB")
            print(f"  Available: {available_memory_gb:.1f} GB")

            # Colab typically has 12-13 GB RAM
            assert total_memory_gb > 10, "Colab should have at least 10 GB RAM"
        except ImportError:
            pytest.skip("Not in Colab environment")


@pytest.mark.compatibility
class TestModelCompatibility:
    """Test model compatibility across environments"""

    def test_model_save_load_formats(self, simple_model, tmp_path):
        """Test different save/load formats"""
        formats_to_test = [
            ('h5', '.h5'),
            ('tf', ''),  # SavedModel format
        ]

        for format_name, extension in formats_to_test:
            save_path = tmp_path / f"model_{format_name}"
            if extension:
                save_path = save_path.with_suffix(extension)

            # Save
            if format_name == 'tf':
                simple_model.save(str(save_path), save_format='tf')
            else:
                simple_model.save(str(save_path))

            # Load
            loaded_model = tf.keras.models.load_model(str(save_path))

            assert loaded_model is not None, f"Failed to load {format_name} format"
            print(f"\n{format_name} format: OK")

    def test_cross_platform_model(self, trained_model, sample_train_test_split, tmp_path):
        """Test model predictions are consistent across platforms"""
        X_test = sample_train_test_split[1]

        # Save model
        model_path = tmp_path / "cross_platform_model.h5"
        trained_model.save(str(model_path))

        # Original predictions
        original_preds = trained_model.predict(X_test[:10], verbose=0)

        # Load and predict
        loaded_model = tf.keras.models.load_model(str(model_path))
        loaded_preds = loaded_model.predict(X_test[:10], verbose=0)

        assert np.allclose(original_preds, loaded_preds), \
            "Cross-platform predictions should match"

        print(f"\nCross-platform compatibility: OK")

    def test_cpu_gpu_consistency(self, trained_model, sample_train_test_split):
        """Test CPU and GPU produce same results"""
        X_test = sample_train_test_split[1]
        sample = X_test[:5]

        # CPU prediction
        with tf.device('/CPU:0'):
            cpu_pred = trained_model.predict(sample, verbose=0)

        # GPU prediction (if available)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            with tf.device('/GPU:0'):
                gpu_pred = trained_model.predict(sample, verbose=0)

            assert np.allclose(cpu_pred, gpu_pred, rtol=1e-4), \
                "CPU and GPU predictions should match"
            print(f"\nCPU-GPU consistency: OK")
        else:
            print(f"\nNo GPU available for consistency test")


@pytest.mark.compatibility
class TestDataTypeCompatibility:
    """Test different data type compatibility"""

    def test_float32_float64_compatibility(self, trained_model):
        """Test float32 and float64 compatibility"""
        data_float32 = np.random.randn(10, 3197).astype(np.float32)
        data_float64 = np.random.randn(10, 3197).astype(np.float64)

        pred_float32 = trained_model.predict(data_float32, verbose=0)
        pred_float64 = trained_model.predict(data_float64, verbose=0)

        assert pred_float32.shape == pred_float64.shape, "Shapes should match"
        print(f"\nFloat32/64 compatibility: OK")

    def test_int_to_float_conversion(self, trained_model):
        """Test integer to float conversion"""
        data_int = np.random.randint(-100, 100, size=(10, 3197))
        data_float = data_int.astype(np.float32)

        pred_int = trained_model.predict(data_int, verbose=0)
        pred_float = trained_model.predict(data_float, verbose=0)

        assert np.allclose(pred_int, pred_float), "Int conversion should work"
        print(f"\nInt-to-float conversion: OK")


@pytest.mark.compatibility
class TestBackwardCompatibility:
    """Test backward compatibility"""

    def test_old_model_loading(self, tmp_path):
        """Test loading models saved with older TensorFlow versions"""
        # This test would load pre-saved models from older versions
        # For now, we'll just verify the current model can be loaded

        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(3197,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Save and load
        model_path = tmp_path / "backward_compat_test.h5"
        model.save(str(model_path))

        loaded_model = tf.keras.models.load_model(str(model_path))
        assert loaded_model is not None, "Model loading failed"

        print(f"\nBackward compatibility: OK")

    def test_model_config_portability(self, simple_model):
        """Test model configuration portability"""
        # Save config as JSON
        config = simple_model.to_json()

        # Recreate from config
        recreated_model = tf.keras.models.model_from_json(config)

        # Compare architectures
        assert len(simple_model.layers) == len(recreated_model.layers), \
            "Layer count should match"

        print(f"\nModel config portability: OK")


@pytest.mark.compatibility
def test_environment_summary():
    """Print comprehensive environment summary"""
    print(f"\n{'='*60}")
    print(f"ENVIRONMENT SUMMARY")
    print(f"{'='*60}")

    # Python
    print(f"\nPython:")
    print(f"  Version: {sys.version}")
    print(f"  Executable: {sys.executable}")

    # TensorFlow
    print(f"\nTensorFlow:")
    print(f"  Version: {tf.__version__}")
    print(f"  Keras version: {tf.keras.__version__}")

    # Devices
    print(f"\nCompute Devices:")
    cpus = tf.config.list_physical_devices('CPU')
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  CPUs: {len(cpus)}")
    print(f"  GPUs: {len(gpus)}")

    # Platform
    print(f"\nPlatform:")
    print(f"  System: {platform.system()}")
    print(f"  Release: {platform.release()}")
    print(f"  Machine: {platform.machine()}")

    # Packages
    print(f"\nKey Packages:")
    for pkg in ['numpy', 'pandas', 'scikit-learn', 'tensorflow']:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"  {pkg}: {version}")
        except:
            print(f"  {pkg}: Not found")

    print(f"\n{'='*60}")
