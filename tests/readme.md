# Test Suite for Enhanced Bathymetric CAE Processing v2.0

## Overview
This comprehensive test suite validates all components of the Enhanced Bathymetric CAE Processing system, including unit tests, integration tests, and performance tests.

## Test Structure

### Unit Tests
- `test_config.py` - Configuration management tests
- `test_core_*.py` - Core functionality tests (enums, adaptive processing, quality metrics, constraints)
- `test_models_*.py` - Model architecture and ensemble tests
- `test_processing_*.py` - Data processing and pipeline tests
- `test_cli_interface.py` - Command line interface tests
- `test_review_expert_system.py` - Expert review system tests
- `test_utils_*.py` - Utility function tests

### Integration Tests
- `test_integration.py` - End-to-end pipeline tests
- Tests complete workflow from input to output
- Validates feature integration and data flow

### Performance Tests
- `test_performance.py` - Performance and stress tests
- Memory usage validation
- Inference speed benchmarks

### Test Fixtures
- `conftest.py` - Shared pytest fixtures
- `test_fixtures/sample_data_generator.py` - Generate test data

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov pytest-html pytest-mock
```

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/test_*.py -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests
pytest tests/test_performance.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Run Individual Test Files
```bash
pytest tests/test_config.py -v
pytest tests/test_models_architectures.py -v
```

## Test Data Generation

### Create Sample Test Data
```python
from tests.test_fixtures.sample_data_generator import TestDataGenerator

# Create test dataset
output_dir = Path("test_data")
files, metadata = TestDataGenerator.create_test_dataset(output_dir, num_files=10)
```

### Manual Test Data Setup
```bash
mkdir test_input
python -c "
from tests.test_fixtures.sample_data_generator import TestDataGenerator
from pathlib import Path
TestDataGenerator.create_test_dataset(Path('test_input'), 5)
"
```

## Test Configuration

### Environment Variables
```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1  # Disable auto plugin loading
export TF_CPP_MIN_LOG_LEVEL=2            # Reduce TensorFlow logging
```

### Custom Test Configuration
Create `pytest.ini`:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    performance: marks tests as performance tests
```

## Test Coverage

### Coverage Reports
```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html:htmlcov

# Generate terminal coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Coverage with XML output for CI/CD
pytest tests/ --cov=. --cov-report=xml
```

### Expected Coverage Targets
- Core modules: >90%
- Models: >85%
- Processing: >90%
- Utils: >80%
- Overall: >85%

## Continuous Integration

### GitHub Actions Example
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-html
    - name: Run tests
      run: python tests/run_all_tests.py
```

## Mock and Fixture Guidelines

### When to Use Mocks
- External dependencies (GDAL, file I/O)
- Time-consuming operations (model training)
- Hardware-specific functionality (GPU operations)

### Test Data Patterns
- Use small grid sizes (32x32, 64x64) for speed
- Create realistic bathymetry patterns
- Include edge cases (NaN values, extreme depths)

### Memory Management in Tests
- Clean up large test data
- Use pytest fixtures for setup/teardown
- Monitor memory usage in performance tests

## Troubleshooting Tests

### Common Issues

#### GDAL Not Available
```bash
# Install GDAL
conda install gdal
# or
apt-get install gdal-bin python3-gdal
```

#### TensorFlow GPU Issues
```python
# Disable GPU in tests
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

#### Memory Issues
```bash
# Run tests with memory monitoring
pytest tests/ --verbose --tb=short -s
```

### Debug Mode
```bash
# Run with debug output
pytest tests/ --log-cli-level=DEBUG -s

# Run single test with debug
pytest tests/test_specific.py::TestClass::test_method -s -vv
```

## Test Best Practices

### Test Organization
- One test file per module
- Group related tests in classes
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

### Test Data
- Keep test data small and focused
- Use parametrized tests for multiple scenarios
- Clean up test artifacts

### Assertions
- Use specific assertions
- Test both positive and negative cases
- Validate error conditions

### Performance
- Set reasonable timeouts
- Use appropriate test data sizes
- Monitor resource usage

## Contributing Tests

### Adding New Tests
1. Follow existing naming conventions
2. Add tests for new features
3. Update coverage requirements
4. Document complex test scenarios

### Test Review Checklist
- [ ] Tests cover happy path and edge cases
- [ ] Error conditions are tested
- [ ] Mocks are used appropriately
- [ ] Tests are deterministic
- [ ] Performance implications considered
- [ ] Documentation is updated

### Test Maintenance
- Review tests with code changes
- Update test data as needed
- Monitor test execution time
- Keep dependencies up to date
