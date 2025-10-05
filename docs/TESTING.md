# Testing Guide for Kepler Exoplanet Detection

## Overview

This document provides comprehensive testing guidelines for the Kepler exoplanet detection system. The test suite ensures high code quality, reliability, and performance across different environments.

## Test Suite Statistics

- **Total Test Files**: 11
- **Test Categories**: 8
- **Estimated Tests**: 150+
- **Target Coverage**: 80%+
- **Test Frameworks**: pytest, pytest-cov, pytest-benchmark

## Quick Start

### Installation

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Verify installation
pytest --version
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific category
pytest -m unit
pytest -m integration
pytest -m performance
pytest -m compatibility

# Run with coverage
pytest --cov=. --cov-report=html

# Run in parallel
pytest -n auto
```

## Test Categories

### 1. Unit Tests (⚡ Fast, Isolated)

#### Data Preprocessing Tests (`test_preprocessing.py`)

**Purpose**: Validate data loading, transformation, and validation

**Test Classes**:
- `TestDataLoading`: Data structure and format validation
  - Sample data shape verification
  - Column validation (LABEL + 3197 FLUX features)
  - Label value range [1, 2, 3]
  - Missing value detection
  - Data type validation

- `TestTrainTestSplit`: Data splitting validation
  - 80/20 split ratio verification
  - Shape consistency
  - Feature dimension preservation
  - Stratification validation
  - Data leakage prevention

- `TestOneHotEncoding`: Label encoding validation
  - Shape correctness (samples × 3 classes)
  - Binary value validation
  - Row sum verification (equals 1)
  - Encoding reversibility
  - Class mapping accuracy

- `TestDataNormalization`: Data scaling tests
  - Standard scaling (mean=0, std=1)
  - Min-max scaling [0, 1]
  - Robust scaling (outlier resistant)

- `TestClassDistribution`: Class balance analysis
  - Class count validation
  - Imbalance ratio calculation
  - Stratification effectiveness

- `TestDataValidation`: Data quality checks
  - Feature variance analysis
  - Constant feature detection
  - Correlation matrix validation
  - Outlier detection (IQR method)

**Key Assertions**:
```python
assert sample_data.shape == (1000, 3198)  # 1000 samples, 3198 columns
assert set(labels).issubset({1, 2, 3})    # Valid label range
assert len(X_train) + len(X_test) == len(X)  # No data loss
assert np.allclose(y_encoded.sum(axis=1), 1)  # Valid one-hot
```

#### Model I/O Tests (`test_model_io.py`)

**Purpose**: Ensure model persistence and loading correctness

**Test Classes**:
- `TestModelSaving`: Save functionality
  - H5 format saving
  - SavedModel format saving
  - Model configuration export (JSON/YAML)
  - Weights-only saving
  - File overwrite handling

- `TestModelLoading`: Load functionality
  - H5 format loading
  - SavedModel loading
  - Config-based reconstruction
  - Weights-only loading
  - Error handling (non-existent files)

- `TestArchitecturePreservation`: Architecture integrity
  - Layer count preservation
  - Layer type preservation
  - Layer configuration preservation
  - Input/output shape preservation

- `TestWeightPreservation`: Weight integrity
  - Exact weight preservation
  - Weight shape preservation
  - Bias term preservation

- `TestOptimizerStatePreservation`: Optimizer state
  - Optimizer config preservation
  - Loss function preservation
  - Metrics preservation

- `TestPredictionConsistency`: Prediction accuracy
  - Identical predictions before/after save
  - Batch size consistency
  - Numerical precision validation

- `TestSerializationFormats`: Format compatibility
  - JSON serialization
  - YAML serialization
  - Pickle compatibility

- `TestErrorHandling`: Error scenarios
  - Invalid path handling
  - Corrupted file handling
  - Incompatible weights handling

**Key Assertions**:
```python
assert Path(model_path).exists()  # File created
assert np.allclose(orig_weights, loaded_weights)  # Exact weights
assert orig_preds == loaded_preds  # Consistent predictions
```

#### Inference Tests (`test_inference.py`)

**Purpose**: Validate prediction logic and edge cases

**Test Classes**:
- `TestSinglePrediction`: Single sample inference
  - Output shape (1, 3)
  - Probability sum equals 1
  - Probability range [0, 1]
  - Class prediction [1, 2, 3]
  - Deterministic predictions

- `TestBatchPrediction`: Batch inference
  - Batch shape validation
  - Variable batch sizes [1, 8, 16, 32, 64]
  - Probability validation (all sum to 1)
  - Batch vs sequential consistency
  - Large batch handling

- `TestPredictionAccuracy`: Accuracy validation
  - Accuracy > random (>33%)
  - Class distribution reasonableness
  - Confidence score distribution
  - Confusion matrix calculation

- `TestPredictionConsistency`: Consistency checks
  - Multiple prediction consistency
  - Order independence
  - Temporal consistency

- `TestEdgeCases`: Boundary conditions
  - All-zero input
  - Extreme values (±1000)
  - Negative values
  - Mixed value ranges
  - Minimum input (single sample)

- `TestErrorHandling`: Error scenarios
  - Wrong input shape
  - Wrong dimensions
  - Empty input
  - Non-numeric input

- `TestPredictionPipeline`: End-to-end flow
  - Complete prediction workflow
  - Preprocessing integration
  - Metrics calculation

**Key Assertions**:
```python
assert prediction.shape == (1, 3)  # Correct shape
assert np.isclose(prediction.sum(), 1.0)  # Valid probabilities
assert predicted_class in [1, 2, 3]  # Valid class
assert accuracy > 0.33  # Better than random
```

### 2. Integration Tests (🔗 Pipeline)

#### Full Pipeline Tests (`integration/test_full_pipeline.py`)

**Purpose**: Validate complete ML workflow

**Test Classes**:
- `TestFullPipeline`: End-to-end workflow
  1. Data preprocessing
  2. Feature scaling
  3. Label encoding
  4. Model building
  5. Model training
  6. Model saving
  7. Model loading
  8. Prediction
  9. Evaluation

- `TestThreeClassClassification`: Multi-class validation
  - All classes predicted
  - Class probabilities correct
  - Classification metrics (precision, recall, F1)
  - Per-class accuracy

- `TestModelPersistence`: Persistence validation
  - Save/load cycle consistency
  - Multiple save/load cycles
  - Weight persistence

- `TestDataPipeline`: Data flow integrity
  - Complete preprocessing pipeline
  - Data integrity maintenance

- `TestTrainingPipeline`: Training validation
  - Loss reduction over epochs
  - Validation metrics tracking
  - Early stopping callback

**Key Workflow**:
```python
Data → Preprocess → Train → Save → Load → Predict → Evaluate
```

**Assertions**:
```python
assert final_loss < initial_loss  # Training works
assert np.allclose(orig_preds, loaded_preds)  # Persistence works
assert test_accuracy > 0.33  # Model learns
```

### 3. Performance Tests (⚡ Benchmarks)

#### Benchmark Tests (`performance/test_benchmarks.py`)

**Purpose**: Ensure performance meets requirements

**Test Classes**:
- `TestInferenceLatency`: Latency measurement
  - Single inference latency (<100ms target)
  - Batch inference latency
  - Latency consistency (low variance)
  - P95 latency tracking

- `TestThroughput`: Throughput measurement
  - Single-thread throughput (>1000 samples/sec)
  - Batch size scaling
  - Sustained throughput (5s test)

- `TestMemoryUsage`: Memory tracking
  - Single inference memory
  - Batch inference memory
  - Memory leak detection
  - Memory growth monitoring

- `TestScalability`: Scalability analysis
  - Input size scaling
  - Concurrent prediction handling
  - Multi-threading performance

- `TestModelSize`: Model efficiency
  - Model file size (<100MB)
  - Model loading time (<5s)
  - Model memory footprint (<500MB)

**Performance Targets**:
| Metric | Target | Test Method |
|--------|--------|-------------|
| Single inference | <100ms | `test_single_inference_latency` |
| Throughput | >1000 samples/sec | `test_throughput_single_thread` |
| Model load time | <5s | `test_model_loading_time` |
| Memory footprint | <500MB | `test_model_memory_footprint` |

**Measurement Example**:
```python
start = time.perf_counter()
model.predict(sample)
latency_ms = (time.perf_counter() - start) * 1000
assert latency_ms < 100  # Under 100ms
```

### 4. Compatibility Tests (🔧 Environment)

#### Environment Tests (`compatibility/test_environments.py`)

**Purpose**: Ensure cross-platform and version compatibility

**Test Classes**:
- `TestPythonVersion`: Python compatibility
  - Version 3.9+ requirement
  - 64-bit architecture
  - Platform information

- `TestTensorFlowVersion`: TensorFlow compatibility
  - Supported versions (2.10-2.15)
  - Keras integration
  - GPU availability detection
  - Mixed precision support

- `TestPackageDependencies`: Package validation
  - Required packages installed
  - Package version compatibility
  - NumPy-TensorFlow compatibility

- `TestColabEnvironment`: Google Colab specific
  - Colab detection
  - GPU access validation
  - File system access
  - Memory limits (12-13GB)

- `TestModelCompatibility`: Cross-platform models
  - Save/load format compatibility (H5, SavedModel)
  - Cross-platform predictions
  - CPU/GPU consistency

- `TestDataTypeCompatibility`: Data type handling
  - Float32/Float64 compatibility
  - Integer conversion
  - Type casting

- `TestBackwardCompatibility`: Version compatibility
  - Old model loading
  - Config portability

**Environment Requirements**:
```python
Python: 3.9+
TensorFlow: 2.10-2.15
NumPy: 1.24+
Pandas: 2.0+
Scikit-learn: 1.3+
```

## Test Fixtures

### Session-Scoped Fixtures (Created Once)

```python
@pytest.fixture(scope="session")
def sample_data():
    """Generate 1000 sample Kepler data"""
    # Returns DataFrame with LABEL + 3197 FLUX columns

@pytest.fixture(scope="session")
def sample_train_test_split(sample_data):
    """Pre-split train/test data (80/20)"""
    # Returns X_train, X_test, y_train, y_test

@pytest.fixture(scope="session")
def encoded_labels(sample_train_test_split):
    """One-hot encoded labels"""
    # Returns y_train_encoded, y_test_encoded
```

### Function-Scoped Fixtures (Created Per Test)

```python
@pytest.fixture(scope="function")
def simple_model():
    """Fresh untrained model"""
    # Returns compiled model

@pytest.fixture(scope="function")
def trained_model(simple_model, data):
    """Pre-trained model (2 epochs)"""
    # Returns trained model

@pytest.fixture(scope="function")
def temp_model_path(tmp_path):
    """Temporary file path"""
    # Returns path for model saving
```

## Running Tests

### Basic Commands

```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_preprocessing.py

# Specific test
pytest tests/test_preprocessing.py::TestDataLoading::test_sample_data_shape

# Specific marker
pytest -m unit
pytest -m integration
pytest -m performance
pytest -m "not slow"
```

### With Coverage

```bash
# Terminal coverage
pytest --cov=. --cov-report=term-missing

# HTML coverage
pytest --cov=. --cov-report=html
open htmlcov/index.html

# XML coverage (for CI)
pytest --cov=. --cov-report=xml
```

### Test Runner Script

```bash
# Basic usage
python tests/run_tests.py

# Specific categories
python tests/run_tests.py --unit
python tests/run_tests.py --integration
python tests/run_tests.py --performance

# With options
python tests/run_tests.py --coverage
python tests/run_tests.py --html
python tests/run_tests.py --parallel

# Combine options
python tests/run_tests.py --unit --coverage --html
```

### Parallel Execution

```bash
# Auto-detect CPU cores
pytest -n auto

# Specific number of workers
pytest -n 4

# Using test runner
python tests/run_tests.py --parallel
```

### Verbose Output

```bash
# Verbose
pytest -v

# Very verbose
pytest -vv

# Show print statements
pytest -s

# Show locals on failure
pytest -l
```

## Test Data Management

### Generating Test Data

```bash
# Generate all test data files
python tests/test_data/test_fixtures.py

# Files created:
# - sample_data.csv (1000 samples)
# - imbalanced_data.csv (90/8/2 split)
# - minimal_data.csv (10 samples)
# - edge_cases/*.csv (various edge cases)
```

### Test Data Types

1. **Normal Data**: Standard Kepler-like features
2. **Imbalanced Data**: Severe class imbalance (90/8/2)
3. **Edge Cases**: Zeros, extremes, correlations
4. **Minimal Data**: Quick tests (10 samples)

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, '3.10', 3.11]
        tensorflow-version: ['2.13', '2.14', '2.15']

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install tensorflow==${{ matrix.tensorflow-version }}
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Run tests
        run: |
          pytest --cov=. --cov-report=xml --cov-fail-under=80

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## Best Practices

### Writing Tests

1. **Follow AAA Pattern**: Arrange, Act, Assert
```python
def test_example():
    # Arrange
    data = create_test_data()

    # Act
    result = process_data(data)

    # Assert
    assert result == expected
```

2. **Use Descriptive Names**
```python
# Good
def test_model_saves_weights_correctly():
    ...

# Bad
def test_model():
    ...
```

3. **Test One Thing**
```python
# Good - single assertion
def test_prediction_shape():
    assert predictions.shape == (10, 3)

# Better - related assertions
def test_prediction_probabilities():
    assert predictions.shape == (10, 3)
    assert np.allclose(predictions.sum(axis=1), 1.0)
```

4. **Use Fixtures for Setup**
```python
@pytest.fixture
def trained_model():
    model = create_model()
    model.fit(X_train, y_train)
    return model

def test_inference(trained_model):
    predictions = trained_model.predict(X_test)
    assert predictions.shape[0] == len(X_test)
```

5. **Test Edge Cases**
```python
@pytest.mark.parametrize("input_data", [
    np.zeros((1, 3197)),           # All zeros
    np.ones((1, 3197)) * 1000,     # Large values
    np.ones((1, 3197)) * -1000,    # Negative values
    np.random.randn(1, 3197) * 100 # Random extreme
])
def test_edge_cases(trained_model, input_data):
    prediction = trained_model.predict(input_data)
    assert not np.any(np.isnan(prediction))
```

### Test Organization

```
tests/
├── conftest.py          # Shared fixtures
├── test_unit.py         # Unit tests
├── integration/         # Integration tests
│   └── test_pipeline.py
├── performance/         # Performance tests
│   └── test_benchmarks.py
└── compatibility/       # Compatibility tests
    └── test_env.py
```

### Debugging Failed Tests

```bash
# Run failed tests only
pytest --lf

# Show failed test details
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Increase verbosity
pytest -vv
```

## Coverage Analysis

### Viewing Coverage

```bash
# Generate HTML report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=. --cov-report=term-missing
```

### Coverage Requirements

- **Minimum**: 80% overall
- **Target**: 90%+ for critical code
- **Statements**: >80%
- **Branches**: >75%
- **Functions**: >80%

### Excluding Code from Coverage

```python
# Exclude from coverage
def debug_function():  # pragma: no cover
    print("Debug info")

# Exclude if statement
if __name__ == "__main__":  # pragma: no cover
    main()
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use editable install
pip install -e .
```

**2. TensorFlow Warnings**
```python
# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

**3. Memory Issues**
```bash
# Run sequentially
pytest -n 0

# Increase timeout
pytest --timeout=600
```

**4. Fixture Not Found**
```bash
# Check conftest.py is loaded
pytest --fixtures

# Verify fixture scope
pytest -v --setup-show
```

**5. Slow Tests**
```bash
# Skip slow tests
pytest -m "not slow"

# Show slowest tests
pytest --durations=10
```

## Metrics and Reporting

### Test Execution Time

```bash
# Show test durations
pytest --durations=0

# Show slowest 10 tests
pytest --durations=10
```

### HTML Reports

```bash
# Generate HTML report
pytest --html=tests/reports/test_report.html --self-contained-html
```

### Benchmark Results

```bash
# Run benchmarks
pytest --benchmark-only

# Save benchmark results
pytest --benchmark-save=baseline
```

## Contributing Tests

1. **Choose test category** (unit/integration/performance/compatibility)
2. **Write test with clear docstring**
3. **Add appropriate markers**
4. **Use existing fixtures**
5. **Ensure deterministic results**
6. **Run tests locally**
7. **Update documentation**

## Summary

This comprehensive test suite ensures:
- ✅ Code correctness (unit tests)
- ✅ System integration (integration tests)
- ✅ Performance requirements (performance tests)
- ✅ Cross-platform compatibility (compatibility tests)
- ✅ High code coverage (>80%)
- ✅ Continuous quality assurance

**Test Stats**:
- 11 test files
- 8 test categories
- 150+ individual tests
- 80%+ code coverage
- <5 min total execution time

---

**Last Updated**: 2025-10-05
**Test Framework**: pytest 7.4+
**Coverage Tool**: pytest-cov 4.1+
