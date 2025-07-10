# Bathymetric CAE Test Suite Documentation

## Overview

The Bathymetric CAE test suite provides comprehensive testing coverage for all package components including unit tests, integration tests, and performance validation. The test suite is built using pytest and includes specialized fixtures, mocks, and utilities for testing bathymetric data processing workflows.

## Table of Contents

1. [Test Structure](#test-structure)
2. [Test Configuration](#test-configuration)
3. [Test Fixtures](#test-fixtures)
4. [Test Modules](#test-modules)
5. [Running Tests](#running-tests)
6. [Test Categories](#test-categories)
7. [Test Utilities](#test-utilities)
8. [Continuous Integration](#continuous-integration)

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_config.py           # Configuration module tests
├── test_model.py            # Model architecture tests
├── test_pipeline.py         # Pipeline integration tests
└── test_processor.py        # Data processor tests
```

## Test Configuration

### conftest.py

The `conftest.py` file provides the foundation for all tests including:

#### Global Test Configuration
```python
TEST_CONFIG = {
    "test_data_dir": Path(__file__).parent / "data",
    "temp_dir_prefix": "bathymetric_cae_test_",
    "default_grid_size": 64,        # Small for fast tests
    "default_batch_size": 2,
    "default_epochs": 2,
    "timeout_seconds": 30
}
```

#### Environment Setup
- Automatic TensorFlow configuration for testing
- GPU memory growth configuration
- Test data directory creation
- Logging level management

#### Conditional Testing
Tests are automatically skipped based on available dependencies:
- `@pytest.mark.tensorflow`: Requires TensorFlow
- `@pytest.mark.gdal`: Requires GDAL
- `@pytest.mark.gpu`: Requires GPU hardware
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests

## Test Fixtures

### Core Fixtures

#### `temp_dir`
Creates isolated temporary directories for each test:
```python
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test isolation."""
```

#### `sample_config`
Provides test-optimized configuration:
```python
@pytest.fixture
def sample_config(temp_dir) -> Config:
    """Create sample configuration for testing."""
```

### Data Fixtures

#### `sample_bathymetric_data`
Generates realistic synthetic bathymetric data:
```python
@pytest.fixture
def sample_bathymetric_data() -> np.ndarray:
    """Generate sample bathymetric data for testing."""
```

#### `sample_uncertainty_data`
Creates uncertainty data for multi-channel testing:
```python
@pytest.fixture
def sample_uncertainty_data() -> np.ndarray:
    """Generate sample uncertainty data for testing."""
```

#### `sample_multi_channel_data`
Combines depth and uncertainty data:
```python
@pytest.fixture
def sample_multi_channel_data(sample_bathymetric_data, sample_uncertainty_data) -> np.ndarray:
    """Generate multi-channel data (depth + uncertainty)."""
```

### Mock Fixtures

#### `mock_gdal_dataset`
Provides GDAL dataset mock for testing without real files:
```python
@pytest.fixture
def mock_gdal_dataset():
    """Mock GDAL dataset for testing without real files."""
```

#### `mock_tensorflow_model`
Creates minimal TensorFlow model for testing:
```python
@pytest.fixture
def mock_tensorflow_model():
    """Mock TensorFlow model for testing without actual training."""
```

#### `create_test_files`
Factory fixture for creating test files in various formats:
```python
@pytest.fixture
def create_test_files(temp_dir, sample_bathymetric_data):
    """Create test files in various formats for testing."""
```

## Test Modules

### test_config.py

Tests the configuration management system including validation, serialization, and parameter handling.

#### Test Classes:
- **TestConfig**: Core configuration functionality
- **TestConfigUtilityFunctions**: Helper functions
- **TestConfigSerialization**: JSON save/load operations
- **TestConfigEdgeCases**: Boundary conditions

#### Key Test Areas:
```python
def test_config_validation_valid()          # Valid parameter validation
def test_config_validation_invalid_epochs() # Invalid parameter detection
def test_config_save_and_load()            # Serialization roundtrip
def test_config_update()                   # Configuration updates
```

#### Example Test:
```python
def test_config_validation_invalid_epochs(self):
    """Test validation with invalid epochs."""
    with pytest.raises(ValueError, match="epochs must be positive"):
        Config(epochs=-10)
```

### test_model.py

Comprehensive testing of the Advanced CAE model architecture.

#### Test Classes:
- **TestAdvancedCAE**: Core model functionality
- **TestModelUtilityFunctions**: Helper functions
- **TestModelTraining**: Training processes
- **TestModelArchitectureVariations**: Different configurations
- **TestModelPerformance**: Performance characteristics
- **TestModelErrorHandling**: Error conditions
- **TestModelIntegration**: Component integration

#### Key Test Areas:
```python
def test_create_model_single_channel()     # Single-channel model creation
def test_create_model_multi_channel()      # Multi-channel model creation
def test_model_prediction_shape()          # Output shape validation
def test_combined_loss_function()          # Loss function testing
def test_model_training_step()             # Training functionality
```

#### Architecture Testing:
```python
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_different_depths(self, sample_config, depth):
    """Test model creation with different depths."""
    sample_config.depth = depth
    model_builder = AdvancedCAE(sample_config)
    model = model_builder.create_model(channels=1)
    assert model is not None
```

### test_pipeline.py

Tests the main processing pipeline including workflow orchestration and integration.

#### Test Classes:
- **TestBathymetricCAEPipeline**: Core pipeline functionality
- **TestPipelineIntegration**: End-to-end integration
- **TestPipelineUtilityFunctions**: Helper functions
- **TestPipelinePerformance**: Performance characteristics
- **TestPipelineEdgeCases**: Error conditions

#### Key Test Areas:
```python
def test_pipeline_initialization()         # Pipeline setup
def test_validate_paths_valid()           # Path validation
def test_analyze_sample_file()            # File analysis
def test_process_files()                  # Batch processing
def test_generate_final_report()          # Report generation
```

#### Integration Testing:
```python
@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_run(self, sample_config, create_test_files, temp_dir):
    """Test complete pipeline execution."""
    # Comprehensive end-to-end test with mocked heavy operations
```

### test_processor.py

Tests bathymetric data processing including file loading, validation, and preprocessing.

#### Test Classes:
- **TestBathymetricProcessor**: Core processor functionality
- **TestProcessorUtilityFunctions**: Helper functions
- **TestProcessorIntegration**: Integration testing
- **TestProcessorEdgeCases**: Boundary conditions
- **TestProcessorPerformance**: Performance testing

#### Key Test Areas:
```python
def test_validate_file_format_valid()      # Format validation
def test_process_bag_file()               # BAG file processing
def test_validate_and_clean_data()        # Data cleaning
def test_robust_normalize()               # Data normalization
def test_batch_preprocess_files()         # Batch processing
```

#### Data Processing Testing:
```python
def test_validate_and_clean_data_with_nans(self, sample_config):
    """Test data validation and cleaning with NaN values."""
    processor = BathymetricProcessor(sample_config)
    test_data = TestDataGenerator.create_bathymetric_grid()
    test_data[10:15, 10:15] = np.nan  # Add NaN values
    
    result = processor._validate_and_clean_data(test_data, Path("test.tif"))
    
    assert np.all(np.isfinite(result))
    assert not np.any(np.isnan(result[10:15, 10:15]))
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest tests/test_config.py

# Run specific test class
pytest tests/test_model.py::TestAdvancedCAE

# Run specific test method
pytest tests/test_pipeline.py::TestBathymetricCAEPipeline::test_pipeline_initialization
```

### Test Categories

```bash
# Run only fast tests (exclude slow and integration)
pytest -m "not slow and not integration"

# Run only TensorFlow tests
pytest -m tensorflow

# Run only GDAL tests
pytest -m gdal

# Run integration tests
pytest -m integration

# Run GPU tests (requires GPU)
pytest -m gpu
```

### Coverage Reports

```bash
# Run tests with coverage
pytest --cov=bathymetric_cae

# Generate HTML coverage report
pytest --cov=bathymetric_cae --cov-report=html

# Generate coverage report with missing lines
pytest --cov=bathymetric_cae --cov-report=term-missing
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n 4  # Use 4 processes

# Run tests in parallel with coverage
pytest -n 4 --cov=bathymetric_cae
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (< 1 second per test)
- **Scope**: Single functions or methods
- **Dependencies**: Minimal, heavily mocked

### Integration Tests
- **Purpose**: Test component interactions
- **Speed**: Medium (1-10 seconds per test)
- **Scope**: Multiple components working together
- **Dependencies**: Real or realistic mocks

### Performance Tests
- **Purpose**: Validate performance characteristics
- **Speed**: Variable
- **Scope**: Memory usage, execution time, throughput
- **Dependencies**: May require specific hardware

### End-to-End Tests
- **Purpose**: Test complete workflows
- **Speed**: Slow (10+ seconds per test)
- **Scope**: Full pipeline execution
- **Dependencies**: All components, may use real data

## Test Utilities

### Helper Functions

#### `assert_array_properties`
Validates numpy array properties:
```python
def assert_array_properties(array: np.ndarray, 
                          expected_shape: Tuple = None,
                          expected_dtype: np.dtype = None,
                          expected_range: Tuple = None):
    """Assert properties of numpy arrays in tests."""
```

#### `assert_file_exists_and_valid`
Validates file existence and properties:
```python
def assert_file_exists_and_valid(filepath: Path, min_size: int = 0):
    """Assert that a file exists and has valid properties."""
```

### Data Generators

#### `TestDataGenerator`
Provides synthetic data generation:
```python
class TestDataGenerator:
    @staticmethod
    def create_bathymetric_grid(size: int = 64, 
                               depth_range: Tuple[float, float] = (-100, 0),
                               add_noise: bool = True) -> np.ndarray:
        """Create synthetic bathymetric grid."""
    
    @staticmethod
    def create_uncertainty_grid(size: int = 64,
                              uncertainty_range: Tuple[float, float] = (0.1, 2.0)) -> np.ndarray:
        """Create synthetic uncertainty grid."""
```

### Mock Management

#### Conditional Mocking
```python
# Mock TensorFlow components when not available
if not TENSORFLOW_AVAILABLE:
    mock_model = Mock()
    mock_model.predict.return_value = np.random.rand(1, 64, 64, 1)
```

#### GDAL Dataset Mocking
```python
# Comprehensive GDAL dataset mock
mock_dataset = Mock()
mock_dataset.RasterXSize = 64
mock_dataset.RasterYSize = 64
mock_dataset.RasterCount = 2
mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
```

## Test Data Management

### Synthetic Data Generation
All tests use synthetic data generated by `TestDataGenerator` to ensure:
- **Reproducibility**: Consistent test results
- **Independence**: No external data dependencies
- **Coverage**: Various data characteristics
- **Performance**: Fast test execution

### Temporary File Management
Tests use isolated temporary directories:
- **Isolation**: Each test gets clean environment
- **Cleanup**: Automatic cleanup after test completion
- **Safety**: No interference with real files
- **Portability**: Works across platforms

## Error Testing Strategies

### Exception Testing
```python
def test_config_validation_invalid_epochs(self):
    """Test validation with invalid epochs."""
    with pytest.raises(ValueError, match="epochs must be positive"):
        Config(epochs=-10)
```

### Logging Validation
```python
def test_error_logging(self, caplog):
    """Test error logging functionality."""
    # Trigger error condition
    # Check log output
    assert "Expected error message" in caplog.text
```

### Graceful Degradation
```python
def test_fallback_behavior(self):
    """Test fallback when primary method fails."""
    with patch('primary_method', side_effect=Exception("Primary failed")):
        result = component.process_with_fallback()
        assert result is not None  # Should use fallback
```

## Performance Testing

### Memory Testing
```python
def test_memory_efficiency(self):
    """Test memory usage during processing."""
    large_data = create_large_test_data()
    initial_memory = get_memory_usage()
    
    process_data(large_data)
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    assert memory_increase < ACCEPTABLE_MEMORY_INCREASE
```

### Timing Tests
```python
def test_processing_speed(self):
    """Test processing performance."""
    test_data = create_test_data()
    
    start_time = time.time()
    result = process_data(test_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    assert processing_time < MAX_ACCEPTABLE_TIME
```

## Continuous Integration

### GitHub Actions Configuration
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests
      run: |
        pytest --cov=bathymetric_cae --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Environment Matrix
Tests run across multiple environments:
- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **Operating systems**: Ubuntu, macOS, Windows
- **Dependencies**: Various TensorFlow/GDAL versions
- **Hardware**: CPU-only, GPU-enabled

## Best Practices

### Test Design
1. **Isolation**: Each test is independent
2. **Repeatability**: Tests produce consistent results
3. **Speed**: Fast execution for rapid feedback
4. **Coverage**: Comprehensive functionality coverage
5. **Clarity**: Clear test intent and assertions

### Mock Strategy
1. **External Dependencies**: Mock file systems, networks
2. **Heavy Operations**: Mock training, large computations
3. **Hardware Dependencies**: Mock GPU, GDAL operations
4. **Non-deterministic Operations**: Mock random, time-based

### Data Strategy
1. **Synthetic Data**: Use generated test data
2. **Minimal Size**: Small data for fast tests
3. **Representative**: Cover realistic data characteristics
4. **Edge Cases**: Include boundary conditions

### Error Testing
1. **Expected Errors**: Test proper error handling
2. **Edge Cases**: Test boundary conditions
3. **Resource Limits**: Test memory, disk constraints
4. **Partial Failures**: Test graceful degradation

## Debugging Tests

### Running Single Tests
```bash
# Run with debugging output
pytest -v -s tests/test_config.py::TestConfig::test_config_validation

# Run with pdb on failure
pytest --pdb tests/test_model.py

# Run with detailed output
pytest -vvv --tb=long tests/test_pipeline.py
```

### Log Analysis
```bash
# Capture all logs during testing
pytest --log-cli-level=DEBUG

# Save logs to file
pytest --log-file=test_logs.txt
```

### Memory Debugging
```bash
# Run with memory profiling
pytest --profile tests/test_processor.py

# Monitor memory usage
pytest --memprof tests/test_pipeline.py
```

## Contributing Tests

### Adding New Tests
1. **Follow naming conventions**: `test_<functionality>.py`
2. **Use appropriate fixtures**: Leverage existing fixtures
3. **Include docstrings**: Document test purpose
4. **Add markers**: Use appropriate pytest markers
5. **Test edge cases**: Include boundary conditions

### Test Review Checklist
- [ ] Tests are isolated and independent
- [ ] Appropriate fixtures and mocks used
- [ ] Edge cases and error conditions covered
- [ ] Performance implications considered
- [ ] Documentation updated
- [ ] CI integration verified

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### GDAL/TensorFlow Not Available
```bash
# Install optional dependencies
pip install bathymetric-cae[dev]

# Check availability
python -c "import tensorflow; print('TF OK')"
python -c "from osgeo import gdal; print('GDAL OK')"
```

#### Test Failures
```bash
# Run with maximum verbosity
pytest -vvv --tb=long --capture=no

# Isolate failing test
pytest tests/test_module.py::TestClass::test_method

# Skip problematic tests temporarily
pytest -k "not test_problematic_function"
```

### Getting Help

1. **Check test logs**: Look for detailed error messages
2. **Review fixtures**: Ensure proper test setup
3. **Validate environment**: Check dependencies
4. **Run subset**: Isolate failing tests
5. **Compare platforms**: Test on different systems

The test suite provides comprehensive coverage while maintaining fast execution and clear documentation. Regular test execution ensures code quality and prevents regressions during development.