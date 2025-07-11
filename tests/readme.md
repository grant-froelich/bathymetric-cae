# tests/README.md
"""
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
"""


# Makefile for Test Management
"""
# Makefile

.PHONY: test test-unit test-integration test-performance test-coverage clean-test

# Python and pytest configuration
PYTHON := python3
PYTEST := pytest
PYTEST_ARGS := -v --tb=short --strict-markers

# Test directories
TEST_DIR := tests
UNIT_TESTS := $(TEST_DIR)/test_*.py
INTEGRATION_TESTS := $(TEST_DIR)/test_integration.py
PERFORMANCE_TESTS := $(TEST_DIR)/test_performance.py

# Coverage configuration
COV_ARGS := --cov=config --cov=core --cov=models --cov=processing --cov=review --cov=utils
COV_REPORT := --cov-report=html:htmlcov --cov-report=term-missing --cov-report=xml

# Default test target
test: test-unit test-integration

# Run all unit tests
test-unit:
	@echo "Running unit tests..."
	$(PYTEST) $(UNIT_TESTS) $(PYTEST_ARGS)

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	$(PYTEST) $(INTEGRATION_TESTS) $(PYTEST_ARGS)

# Run performance tests
test-performance:
	@echo "Running performance tests..."
	$(PYTEST) $(PERFORMANCE_TESTS) $(PYTEST_ARGS) -m "not slow"

# Run all tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(PYTEST) $(TEST_DIR) $(PYTEST_ARGS) $(COV_ARGS) $(COV_REPORT)

# Run tests and generate reports
test-reports:
	@echo "Running tests with reports..."
	$(PYTEST) $(TEST_DIR) $(PYTEST_ARGS) $(COV_ARGS) $(COV_REPORT) \
		--junitxml=test_results.xml \
		--html=test_report.html --self-contained-html

# Run specific test file
test-file:
	@echo "Usage: make test-file FILE=test_filename.py"
	$(PYTEST) $(TEST_DIR)/$(FILE) $(PYTEST_ARGS) -v

# Generate test data
test-data:
	@echo "Generating test data..."
	$(PYTHON) -c "from tests.test_fixtures.sample_data_generator import TestDataGenerator; \
		from pathlib import Path; \
		TestDataGenerator.create_test_dataset(Path('test_data'), 10)"

# Clean test artifacts
clean-test:
	@echo "Cleaning test artifacts..."
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf test_results.xml
	rm -rf test_report.html
	rm -rf test_data/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Setup test environment
setup-test:
	@echo "Setting up test environment..."
	pip install pytest pytest-cov pytest-html pytest-mock
	mkdir -p logs plots expert_reviews test_data

# Run tests in parallel
test-parallel:
	@echo "Running tests in parallel..."
	$(PYTEST) $(TEST_DIR) $(PYTEST_ARGS) -n auto

# Run tests with memory profiling
test-memory:
	@echo "Running tests with memory profiling..."
	$(PYTEST) $(TEST_DIR) $(PYTEST_ARGS) --memray

# Lint test code
lint-tests:
	@echo "Linting test code..."
	flake8 $(TEST_DIR)
	black --check $(TEST_DIR)

# Format test code
format-tests:
	@echo "Formatting test code..."
	black $(TEST_DIR)
	isort $(TEST_DIR)

# Help target
help:
	@echo "Available test targets:"
	@echo "  test              - Run unit and integration tests"
	@echo "  test-unit         - Run only unit tests"
	@echo "  test-integration  - Run only integration tests"
	@echo "  test-performance  - Run performance tests"
	@echo "  test-coverage     - Run tests with coverage report"
	@echo "  test-reports      - Run tests and generate all reports"
	@echo "  test-file         - Run specific test file (use FILE=filename)"
	@echo "  test-data         - Generate test data"
	@echo "  test-parallel     - Run tests in parallel"
	@echo "  test-memory       - Run tests with memory profiling"
	@echo "  clean-test        - Clean test artifacts"
	@echo "  setup-test        - Setup test environment"
	@echo "  lint-tests        - Lint test code"
	@echo "  format-tests      - Format test code"
"""


# Docker configuration for testing
"""
# docker/test.Dockerfile

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY tests/requirements-test.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

# Create test directories
RUN mkdir -p logs plots expert_reviews test_data

# Run tests by default
CMD ["python", "tests/run_all_tests.py"]
"""


# GitHub Actions workflow for testing
"""
# .github/workflows/test.yml

name: Enhanced Bathymetric CAE Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y gdal-bin libgdal-dev python3-gdal

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-html pytest-mock

    - name: Create test directories
      run: |
        mkdir -p logs plots expert_reviews test_data

    - name: Generate test data
      run: |
        python -c "
        from tests.test_fixtures.sample_data_generator import TestDataGenerator
        from pathlib import Path
        TestDataGenerator.create_test_dataset(Path('test_data'), 5)
        "

    - name: Run unit tests
      run: |
        pytest tests/test_*.py -v --tb=short --strict-markers \
          --junitxml=test-results-unit.xml

    - name: Run integration tests
      run: |
        pytest tests/test_integration.py -v --tb=short \
          --junitxml=test-results-integration.xml

    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=config --cov=core --cov=models \
          --cov=processing --cov=review --cov=utils \
          --cov-report=xml --cov-report=html \
          --junitxml=test-results-coverage.xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

    - name: Archive test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: |
          test-results-*.xml
          htmlcov/
          test_report.html

    - name: Archive test logs
      uses: actions/upload-artifact@v3
      if: failure()
      with:
        name: test-logs-${{ matrix.python-version }}
        path: logs/

  performance-test:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y gdal-bin libgdal-dev python3-gdal
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-benchmark memory-profiler

    - name: Run performance tests
      run: |
        pytest tests/test_performance.py -v --benchmark-only \
          --benchmark-json=benchmark-results.json

    - name: Archive performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: benchmark-results.json

  security-test:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety

    - name: Run security scan
      run: |
        bandit -r . -f json -o bandit-report.json
        safety check --json --output safety-report.json || true

    - name: Archive security results
      uses: actions/upload-artifact@v3
      with:
        name: security-results
        path: |
          bandit-report.json
          safety-report.json
"""


# Test requirements file
"""
# tests/requirements-test.txt

# Core testing framework
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-html>=3.1.0
pytest-mock>=3.10.0
pytest-xdist>=3.0.0  # For parallel testing

# Performance testing
pytest-benchmark>=4.0.0
memory-profiler>=0.60.0
pytest-memray>=1.0.0

# Code quality
flake8>=5.0.0
black>=22.0.0
isort>=5.0.0

# Additional testing utilities
hypothesis>=6.0.0  # Property-based testing
factory-boy>=3.2.0  # Test data factories
responses>=0.20.0  # Mock HTTP requests
freezegun>=1.2.0  # Time mocking

# Coverage and reporting
coverage[toml]>=6.0.0
codecov>=2.1.0

# Optional development tools
jupyter>=1.0.0
ipython>=8.0.0
"""


# Advanced test utilities
"""
# tests/utils/test_helpers.py

import numpy as np
import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import Mock
import tensorflow as tf


class TestHelpers:
    '''Advanced test helper utilities.'''
    
    @staticmethod
    def create_mock_model(input_shape=(64, 64, 1), output_shape=None):
        '''Create a mock TensorFlow model for testing.'''
        if output_shape is None:
            output_shape = input_shape
            
        model = Mock()
        model.input_shape = (None,) + input_shape
        model.predict.return_value = np.random.random((1,) + output_shape)
        model.fit.return_value = Mock()
        model.evaluate.return_value = [0.1, 0.05]  # loss, mae
        model.count_params.return_value = 10000
        
        return model
    
    @staticmethod
    def assert_array_properties(array, expected_shape=None, expected_dtype=None, 
                               expected_range=None, no_nan=True, no_inf=True):
        '''Assert multiple array properties at once.'''
        if expected_shape:
            assert array.shape == expected_shape
        if expected_dtype:
            assert array.dtype == expected_dtype
        if expected_range:
            assert expected_range[0] <= array.min() <= array.max() <= expected_range[1]
        if no_nan:
            assert not np.isnan(array).any()
        if no_inf:
            assert not np.isinf(array).any()
    
    @staticmethod
    def create_gradient_data(shape, direction='horizontal'):
        '''Create test data with known gradients.'''
        if direction == 'horizontal':
            return np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1))
        elif direction == 'vertical':
            return np.tile(np.linspace(0, 1, shape[0])[:, None], (1, shape[1]))
        elif direction == 'radial':
            center = (shape[0]//2, shape[1]//2)
            y, x = np.ogrid[:shape[0], :shape[1]]
            return np.sqrt((x - center[1])**2 + (y - center[0])**2) / max(shape)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    @staticmethod
    @contextmanager
    def temporary_config_override(config, **overrides):
        '''Temporarily override configuration values.'''
        original_values = {}
        for key, value in overrides.items():
            if hasattr(config, key):
                original_values[key] = getattr(config, key)
                setattr(config, key, value)
        
        try:
            yield config
        finally:
            for key, value in original_values.items():
                setattr(config, key, value)
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        '''Measure function execution time.'''
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def compare_arrays_with_tolerance(arr1, arr2, rtol=1e-5, atol=1e-8):
        '''Compare arrays with relative and absolute tolerance.'''
        return np.allclose(arr1, arr2, rtol=rtol, atol=atol)


# tests/utils/mock_gdal.py

class MockGDALDataset:
    '''Mock GDAL dataset for testing without file dependencies.'''
    
    def __init__(self, data, geotransform=None, projection=None, metadata=None):
        self.data = data if isinstance(data, list) else [data]
        self.geotransform = geotransform or (0, 1, 0, 0, 0, -1)
        self.projection = projection or 'EPSG:4326'
        self.metadata = metadata or {}
        self.RasterCount = len(self.data)
    
    def GetRasterBand(self, band_num):
        band_data = self.data[band_num - 1]  # 1-indexed
        
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = band_data
        mock_band.SetDescription = Mock()
        mock_band.WriteArray = Mock()
        mock_band.SetNoDataValue = Mock()
        
        return mock_band
    
    def GetGeoTransform(self):
        return self.geotransform
    
    def GetProjection(self):
        return self.projection
    
    def GetMetadata(self):
        return self.metadata
    
    def SetGeoTransform(self, transform):
        self.geotransform = transform
    
    def SetProjection(self, projection):
        self.projection = projection
    
    def SetMetadata(self, metadata, domain=''):
        if domain:
            self.metadata[domain] = metadata
        else:
            self.metadata.update(metadata)
    
    def FlushCache(self):
        pass


# tests/utils/performance_monitor.py

import time
import psutil
import threading
from contextlib import contextmanager


class PerformanceMonitor:
    '''Monitor performance metrics during test execution.'''
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.max_memory = 0
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    @contextmanager
    def monitor(self, sample_interval=0.1):
        '''Context manager for performance monitoring.'''
        self.start_monitoring(sample_interval)
        try:
            yield self
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self, sample_interval=0.1):
        '''Start performance monitoring.'''
        self.reset()
        self.start_time = time.time()
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor,
            args=(sample_interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        '''Stop performance monitoring.'''
        self.end_time = time.time()
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _memory_monitor(self, interval):
        '''Monitor memory usage in separate thread.'''
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                self.max_memory = max(self.max_memory, memory_mb)
                time.sleep(interval)
            except Exception:
                break
    
    def get_results(self):
        '''Get monitoring results.'''
        execution_time = None
        if self.start_time and self.end_time:
            execution_time = self.end_time - self.start_time
        
        return {
            'execution_time': execution_time,
            'max_memory_mb': self.max_memory,
            'avg_memory_mb': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            'memory_samples': len(self.memory_samples)
        }
"""