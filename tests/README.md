# Kepler Exoplanet Detection - Test Suite

Comprehensive test suite for the Kepler exoplanet detection machine learning system.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── pytest.ini                     # Pytest settings
├── requirements-test.txt          # Test dependencies
├── run_tests.py                   # Test runner script
│
├── test_preprocessing.py          # Data preprocessing tests
├── test_model_io.py              # Model save/load tests
├── test_inference.py             # Inference logic tests
│
├── integration/                   # Integration tests
│   └── test_full_pipeline.py     # End-to-end pipeline tests
│
├── performance/                   # Performance tests
│   └── test_benchmarks.py        # Latency and throughput tests
│
├── compatibility/                 # Compatibility tests
│   └── test_environments.py      # Colab and TF version tests
│
└── test_data/                     # Test data
    ├── README.md
    ├── test_fixtures.py          # Data generation utilities
    └── edge_cases/               # Edge case datasets
```

## Test Categories

### 1. Unit Tests (⚡ Fast)

**Data Preprocessing** (`test_preprocessing.py`)
- Data loading and validation
- Train/test splitting (80/20)
- One-hot encoding correctness
- Data normalization
- Class distribution analysis

**Model I/O** (`test_model_io.py`)
- Model saving (H5, SavedModel formats)
- Model loading
- Architecture preservation
- Weight preservation
- Optimizer state preservation

**Inference** (`test_inference.py`)
- Single sample prediction
- Batch prediction
- Prediction probability validation
- Class prediction accuracy
- Error handling

### 2. Integration Tests (🔗 Pipeline)

**Full Pipeline** (`test_full_pipeline.py`)
- Complete train-save-load-predict workflow
- Three-class classification validation
- Model persistence across sessions
- Data pipeline integrity

### 3. Performance Tests (⚡ Benchmarks)

**Benchmarks** (`test_benchmarks.py`)
- Inference latency (<100ms target)
- Batch prediction throughput (>1000 samples/sec)
- Memory usage tracking
- Scalability testing
- Model loading time

### 4. Compatibility Tests (🔧 Environment)

**Environments** (`test_environments.py`)
- Google Colab environment
- TensorFlow versions (2.10-2.15)
- Python version compatibility (3.9+)
- GPU/CPU compatibility
- Package dependencies

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
pytest tests/

# Or use the test runner
python tests/run_tests.py
```

### Specific Test Types

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Performance tests only
pytest -m performance

# Compatibility tests only
pytest -m compatibility
```

### Using Test Runner

```bash
# Run with coverage report
python tests/run_tests.py --coverage

# Run with HTML report
python tests/run_tests.py --html

# Run in parallel
python tests/run_tests.py --parallel

# Run specific file
python tests/run_tests.py --path tests/test_preprocessing.py

# Run unit tests with coverage
python tests/run_tests.py --unit --coverage
```

### Advanced Options

```bash
# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run in parallel (requires pytest-xdist)
pytest -n auto

# Run only failed tests
pytest --lf

# Run specific test
pytest tests/test_preprocessing.py::TestDataLoading::test_sample_data_shape

# Generate HTML report
pytest --html=tests/reports/report.html --self-contained-html
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.compatibility` - Compatibility tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.benchmark` - Benchmark tests

## Coverage Requirements

- **Minimum coverage**: 80%
- **Target coverage**: 90%+

Coverage reports are generated in:
- Terminal: `--cov-report=term-missing`
- HTML: `htmlcov/index.html`

## Test Data

Test data is automatically generated using pytest fixtures in `conftest.py`:

- **sample_data**: 1000 samples with 3197 features
- **sample_train_test_split**: Pre-split train/test data
- **encoded_labels**: One-hot encoded labels
- **simple_model**: Untrained model for testing
- **trained_model**: Pre-trained model for inference tests

### Generating Custom Test Data

```bash
# Generate test data files
python tests/test_data/test_fixtures.py
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install -r tests/requirements-test.txt
      - run: pytest --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Performance Benchmarks

Target performance metrics:

| Metric | Target | Test |
|--------|--------|------|
| Single inference latency | <100ms | `test_single_inference_latency` |
| Batch throughput | >1000 samples/sec | `test_throughput_single_thread` |
| Model loading time | <5s | `test_model_loading_time` |
| Memory footprint | <500MB | `test_model_memory_footprint` |

## Fixtures

### Session-scoped Fixtures

- `test_data_dir`: Test data directory path
- `sample_data`: Generated Kepler-like data
- `sample_train_test_split`: Train/test split
- `encoded_labels`: One-hot encoded labels
- `performance_config`: Performance test configuration
- `compatibility_config`: Compatibility test configuration

### Function-scoped Fixtures

- `simple_model`: Fresh untrained model
- `trained_model`: Trained model
- `temp_model_path`: Temporary path for model saving
- `reset_tensorflow`: Cleans up TensorFlow session

## Troubleshooting

### Common Issues

**1. TensorFlow GPU not detected**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**2. Memory errors during tests**
```bash
# Run tests sequentially
pytest -n 0

# Increase timeout
pytest --timeout=600
```

**3. Coverage not working**
```bash
# Install coverage
pip install pytest-cov coverage

# Run with coverage
pytest --cov=. --cov-report=html
```

**4. Import errors**
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Best Practices

1. **Write tests first** (TDD approach)
2. **Keep tests isolated** (no dependencies between tests)
3. **Use descriptive test names** (explain what is being tested)
4. **Test edge cases** (zeros, extremes, empty inputs)
5. **Mock external dependencies** (APIs, databases)
6. **Maintain high coverage** (>80%)
7. **Run tests before commits**
8. **Document complex tests**

## Contributing

When adding new tests:

1. Choose appropriate test category (unit/integration/performance/compatibility)
2. Add relevant markers (`@pytest.mark.unit`, etc.)
3. Use existing fixtures when possible
4. Write clear docstrings
5. Ensure tests are deterministic
6. Update this README if needed

## Reports

Test reports are generated in `tests/reports/`:

- `test_report.html` - HTML test report
- `coverage/` - Coverage HTML report
- `benchmark_results.json` - Performance benchmark results

## Contact

For test-related questions or issues, please contact the testing team or open an issue on GitHub.

---

**Test Coverage**: ![Coverage](https://img.shields.io/badge/coverage-80%25-yellow)
**Tests**: ![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
**Python**: ![Python](https://img.shields.io/badge/python-3.9+-blue)
**TensorFlow**: ![TensorFlow](https://img.shields.io/badge/tensorflow-2.10+-orange)
