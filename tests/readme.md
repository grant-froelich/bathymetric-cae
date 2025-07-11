# Test Suite for Enhanced Bathymetric CAE Processing v2.0

## Overview
This comprehensive test suite validates all components of the Enhanced Bathymetric CAE Processing system, providing professional-grade quality assurance with unit tests, integration tests, performance tests, and automated quality gates.

## 🏗️ Test Package Structure

### Complete Directory Structure
```
tests/
├── __init__.py                     # Main test package initialization
├── conftest.py                     # Shared pytest fixtures and configuration
├── pytest.ini                     # Pytest configuration and markers
├── test_config.json               # Advanced test runner configuration
├── requirements-test.txt           # Testing framework dependencies
├── Makefile                        # Test automation commands (in project root)
│
├── 📝 Individual Test Files
├── test_config.py                  # Configuration management tests
├── test_core_enums.py             # Core enumerations tests
├── test_core_adaptive_processor.py # Adaptive processing tests
├── test_core_quality_metrics.py   # Quality metrics tests
├── test_core_constraints.py       # Constitutional constraints tests
├── test_models_architectures.py   # Model architecture tests
├── test_models_ensemble.py        # Ensemble model tests
├── test_processing_data_processor.py # Data processing tests
├── test_processing_pipeline.py    # Pipeline tests (referenced)
├── test_cli_interface.py          # CLI interface tests
├── test_review_expert_system.py   # Expert review system tests
├── test_utils_memory_utils.py     # Memory utilities tests
├── test_utils_visualization.py    # Visualization tests
├── test_utils_logging_utils.py    # Logging utilities tests
├── test_integration.py            # End-to-end integration tests
├── test_performance.py            # Performance and benchmark tests
│
├── 🏗️ fixtures/                   # Advanced test fixtures
│   ├── __init__.py                # Fixtures package initialization
│   ├── advanced_fixtures.py      # Comprehensive test fixtures
│   └── mock_fixtures.py           # Mock environment fixtures
│
├── 🏭 factories/                  # Test data factories
│   ├── __init__.py                # Factories package initialization
│   └── data_factory.py            # Realistic test data generation
│
├── 🛠️ utils/                      # Test utilities
│   ├── __init__.py                # Utils package initialization
│   ├── test_helpers.py            # Advanced testing utility functions
│   ├── mock_gdal.py               # GDAL mocking for file I/O testing
│   └── performance_monitor.py     # Performance and memory monitoring
│
├── 📊 test_fixtures/              # Sample data generators
│   ├── __init__.py                # Test fixtures package initialization
│   └── sample_data_generator.py   # Realistic bathymetric data generation
│
├── 🚀 Test Runners & Automation
├── run_tests_advanced.py          # Advanced test runner with analysis
└── test_automation.sh             # Shell script for automated testing
```

### Package Imports
The test suite is organized as a proper Python package with clear imports:

```python
# Import from main test package
from tests import (
    BathymetricDataFactory, 
    PerformanceMonitor,
    TestHelpers,
    get_test_suite_info
)

# Import from specific subpackages
from tests.factories import ConfigurationFactory, TestFileFactory
from tests.fixtures import ModelTestFixture, PipelineTestFixture
from tests.utils import MockGDALDataset, BenchmarkTimer
from tests.test_fixtures import TestDataGenerator
```

## 🧪 Test Categories and Coverage

### Test Categories with Detailed Breakdown

| Category | Files | Coverage Target | Description | Execution Time |
|----------|-------|----------------|-------------|----------------|
| **Unit Tests** | `test_*.py` (excluding integration/performance) | 90%+ | Fast, isolated component tests | < 2 minutes |
| **Integration Tests** | `test_integration.py` | 85%+ | End-to-end pipeline validation | < 5 minutes |
| **Performance Tests** | `test_performance.py` | 80%+ | Speed and memory benchmarks | < 3 minutes |
| **Mock Tests** | All files using fixtures | 85%+ | Isolated testing without dependencies | Throughout suite |

### Coverage Standards by Module

| Module | Minimum Coverage | Current Target | Priority |
|--------|-----------------|----------------|----------|
| **core/** | 95% | 98% | Critical |
| **models/** | 90% | 95% | High |
| **processing/** | 95% | 97% | Critical |
| **review/** | 85% | 90% | Medium |
| **utils/** | 80% | 85% | Medium |
| **cli/** | 75% | 80% | Low |
| **Overall** | 85% | 90% | Required |

### Test Markers and Selection

Use pytest markers to run specific test categories:

```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run performance tests
pytest -m performance

# Exclude slow tests
pytest -m "not slow"

# Run GPU-specific tests
pytest -m gpu

# Run mock-heavy tests
pytest -m mock
```

Available markers:
- `unit`: Fast, isolated unit tests
- `integration`: End-to-end integration tests
- `performance`: Performance and benchmark tests
- `slow`: Tests that take significant time (>30s)
- `gpu`: Tests requiring GPU hardware
- `mock`: Tests using extensive mocking

## 🚀 Running Tests

### Quick Start Commands

```bash
# Quick test run (unit tests only)
make test-unit

# Full test suite
make test

# With coverage report
make test-coverage

# Performance benchmarks
make test-performance

# Parallel execution
make test-parallel

# Clean up test artifacts
make clean-test
```

### Advanced Test Runner

The advanced test runner provides comprehensive analysis and reporting:

```bash
# Complete test suite with analysis
python tests/run_tests_advanced.py --category all --verbose

# Quick unit test run
python tests/run_tests_advanced.py --quick

# Integration tests only
python tests/run_tests_advanced.py --category integration

# Performance benchmarks with detailed analysis
python tests/run_tests_advanced.py --category performance --verbose

# Generate comprehensive reports
python tests/run_tests_advanced.py --category all --reports
```

### Automated Test Script

The shell script provides cross-platform automation:

```bash
# Make script executable (Unix/Linux/macOS)
chmod +x test_automation.sh

# Run all tests with comprehensive reporting
./test_automation.sh all --verbose

# Quick unit test run
./test_automation.sh unit --quick

# Run with cleanup after completion
./test_automation.sh all --cleanup

# Setup test environment only
./test_automation.sh setup

# Run security and quality checks
./test_automation.sh lint security
```

### Docker-based Testing

```bash
# Build test container
docker build -f docker/test.Dockerfile -t bathymetric-cae-tests .

# Run tests in container
docker run --rm bathymetric-cae-tests

# Run with volume mounts for artifacts
docker run --rm -v $(pwd)/test_reports:/app/test_reports bathymetric-cae-tests
```

## 📊 Test Data Management

### Realistic Test Data Generation

The test suite includes sophisticated data factories for generating realistic bathymetric data:

```python
# Generate data for specific seafloor types
from tests.factories import BathymetricDataFactory
from core.enums import SeafloorType

# Create seamount data
factory = BathymetricDataFactory(
    seafloor_type=SeafloorType.SEAMOUNT,
    width=128, height=128,
    noise_level=0.1
)
test_data = factory.build()

# Access depth and uncertainty data
depth = test_data['depth_data']
uncertainty = test_data['uncertainty_data']
```

### Test File Creation

```python
# Create realistic test files
from tests.test_fixtures import TestDataGenerator

# Generate complete test dataset
files, metadata = TestDataGenerator.create_test_dataset(
    Path("test_data"), num_files=10
)

# Create specific file types
TestDataGenerator.create_sample_bag_file(
    Path("test_seamount.bag"), 
    seafloor_type=SeafloorType.SEAMOUNT
)
```

### Test Data Characteristics

| Seafloor Type | Depth Range | Key Features | Use Cases |
|---------------|-------------|--------------|-----------|
| **Shallow Coastal** | 0-200m | High variability, channels, sandbars | Edge case testing |
| **Continental Shelf** | 200-2000m | Gradual slopes, shelf break | Standard processing |
| **Deep Ocean** | 2000-6000m | Steep slopes, canyons | Performance testing |
| **Seamount** | Variable | High relief, volcanic features | Feature preservation |
| **Abyssal Plain** | 6000-11000m | Low relief, minimal features | Noise handling |

## 🔧 Test Fixtures and Mocking

### Advanced Test Fixtures

The test suite provides comprehensive fixtures for different testing scenarios:

```python
# Model testing fixtures
def test_model_training(model_test_fixture):
    """Test using model fixture."""
    model = model_test_fixture.create_test_model('advanced')
    test_data = model_test_fixture.create_test_data('training')
    
    # Benchmark performance
    results = model_test_fixture.benchmark_model_performance(model, test_data)
    assert results['execution_time'] < 30.0

# Data processing fixtures
def test_file_processing(data_processing_fixture):
    """Test using data processing fixture."""
    workspace = data_processing_fixture.create_temp_workspace()
    files = data_processing_fixture.create_mock_bathymetric_files(workspace, 3)
    
    with data_processing_fixture.mock_gdal_environment():
        # Test processing logic
        pass

# Pipeline testing fixtures
def test_complete_pipeline(pipeline_test_fixture):
    """Test using pipeline fixture."""
    with pipeline_test_fixture.pipeline_test_context("test_run") as env:
        # Run complete pipeline test
        assert env['workspace'].exists()
        assert len(env['test_files']) > 0
```

### Mock Environments

#### GDAL Mocking
```python
# Mock GDAL for file I/O testing
from tests.fixtures import MockGDALFixture

def test_with_mock_gdal():
    fixture = MockGDALFixture()
    
    # Register mock files
    fixture.create_bathymetric_file_mock("test.bag", "seamount")
    
    with fixture.mock_gdal_environment():
        # Test code that uses GDAL
        pass
```

#### TensorFlow Mocking
```python
# Mock TensorFlow models
from tests.fixtures import MockTensorFlowFixture

def test_with_mock_tensorflow():
    fixture = MockTensorFlowFixture()
    
    # Create mock ensemble
    models = fixture.create_mock_ensemble(ensemble_size=3)
    
    with fixture.mock_tensorflow_environment():
        # Test model-related code
        pass
```

#### Database Mocking
```python
# Mock database operations
from tests.fixtures import MockDatabaseFixture

def test_expert_review_system():
    fixture = MockDatabaseFixture()
    
    with fixture.mock_database_environment() as (mock_db, db_path):
        # Test database operations
        mock_db.flag_for_review("test.bag", (0, 0, 100, 100), "low_quality", 0.8)
        reviews = mock_db.get_pending_reviews()
        assert len(reviews) == 1
```

## 📋 Quality Assurance and Reporting

### Test Reports Generated

| Report Type | File Location | Description |
|-------------|---------------|-------------|
| **Coverage Report** | `htmlcov/index.html` | Interactive HTML coverage report |
| **Test Results** | `test-results.xml` | JUnit XML for CI/CD integration |
| **Performance Report** | `test-performance.json` | Detailed performance metrics |
| **Failure Analysis** | `test-failures.json` | Categorized failure analysis |
| **Quality Summary** | `test-summary.json` | Overall quality metrics |

### Quality Gates

Tests must pass the following quality gates:

#### Coverage Gates
- **Overall Coverage**: ≥ 85%
- **Critical Modules**: ≥ 95% (core, processing)
- **New Code**: ≥ 90%

#### Performance Gates
- **Execution Time**: ≤ 10 minutes total
- **Memory Usage**: ≤ 2GB peak
- **Individual Tests**: ≤ 30 seconds each

#### Quality Gates
- **Success Rate**: ≥ 95%
- **Performance Warnings**: ≤ 5 total
- **Security Issues**: 0 high/critical

### Continuous Integration

The test suite integrates with GitHub Actions for automated testing:

```yaml
# .github/workflows/test.yml (excerpt)
- name: Run comprehensive test suite
  run: python tests/run_tests_advanced.py --category all --verbose

- name: Upload test artifacts
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: |
      test-results.xml
      htmlcov/
      test-performance.json
      test-summary.json
```

## 🔍 Test Development Guidelines

### Writing Effective Tests

#### AAA Pattern (Arrange, Act, Assert)
```python
def test_depth_consistency_calculation():
    # Arrange
    test_data = np.ones((10, 10))
    metrics = BathymetricQualityMetrics()
    expected_consistency = 1.0
    
    # Act
    consistency = metrics.calculate_depth_consistency(test_data)
    
    # Assert
    assert consistency == pytest.approx(expected_consistency, abs=1e-6)
```

#### Parameterized Tests
```python
@pytest.mark.parametrize("seafloor_type,expected_smoothing", [
    (SeafloorType.SHALLOW_COASTAL, 0.3),
    (SeafloorType.DEEP_OCEAN, 0.7),
    (SeafloorType.SEAMOUNT, 0.2),
])
def test_adaptive_smoothing_parameters(seafloor_type, expected_smoothing):
    processor = AdaptiveProcessor()
    depth_data = create_test_data_for_type(seafloor_type)
    
    params = processor.get_processing_parameters(depth_data)
    assert params['smoothing_factor'] == expected_smoothing
```

#### Error Condition Testing
```python
def test_config_validation_with_invalid_epochs():
    """Test that invalid epochs raise appropriate error."""
    with pytest.raises(ValueError, match="epochs must be positive"):
        Config(epochs=0)
```

### Test Data Patterns

#### Use Factories for Complex Data
```python
def test_seafloor_classification():
    # Use factory for realistic test data
    factory = BathymetricDataFactory(
        seafloor_type=SeafloorType.SEAMOUNT,
        width=64, height=64
    )
    test_data = factory.build()
    
    classifier = SeafloorClassifier()
    result = classifier.classify(test_data['depth_data'])
    
    assert result == SeafloorType.SEAMOUNT
```

#### Mock External Dependencies
```python
@patch('processing.data_processor.gdal.Open')
def test_bag_file_processing(mock_gdal_open, sample_config):
    # Setup mock
    mock_dataset = create_mock_bag_dataset(depth_data, uncertainty_data)
    mock_gdal_open.return_value = mock_dataset
    
    processor = BathymetricProcessor(sample_config)
    result = processor.preprocess_bathymetric_grid("test.bag")
    
    assert result[0].shape[-1] == 2  # depth + uncertainty
```

### Code Quality in Tests

#### Test Naming Conventions
- Descriptive names: `test_<what_is_being_tested>`
- Include conditions: `test_config_validation_with_invalid_epochs`
- Use underscores for readability

#### Test Organization
```python
class TestBathymetricProcessor:
    """Group related tests in classes."""
    
    def test_bag_file_processing(self):
        """Test BAG file processing."""
        pass
    
    def test_geotiff_processing(self):
        """Test GeoTIFF processing."""
        pass
    
    def test_error_handling(self):
        """Test error handling."""
        pass
```

## 🐛 Debugging and Troubleshooting

### Debug Test Execution

```bash
# Run specific test with debug output
pytest tests/test_specific.py::TestClass::test_method -s -vv

# Run with logging enabled
pytest tests/ --log-cli-level=DEBUG

# Run with Python debugger
pytest tests/ --pdb

# Run with coverage and debug
pytest tests/ --cov=. --cov-report=term-missing -s
```

### Common Test Issues and Solutions

#### Memory Issues
```bash
# Symptoms: Out of memory during tests
# Solutions:
pytest tests/ --maxfail=1  # Stop on first failure
pytest tests/test_unit.py  # Run smaller test subset
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Limit TensorFlow memory
```

#### GDAL Issues
```bash
# Symptoms: GDAL import errors, file format issues
# Solutions:
pip install gdal  # Reinstall GDAL
export GDAL_DATA=/path/to/gdal/data  # Set GDAL data path
pytest tests/ --ignore=tests/test_integration.py  # Skip integration tests
```

#### TensorFlow Issues
```bash
# Symptoms: TensorFlow errors, GPU issues
# Solutions:
export CUDA_VISIBLE_DEVICES=""  # Use CPU only
pytest tests/ -m "not gpu"  # Skip GPU tests
pip install tensorflow-cpu  # Use CPU-only TensorFlow
```

### Test Data Debugging

```python
def test_with_data_inspection():
    """Test with intermediate data inspection."""
    factory = BathymetricDataFactory(seafloor_type=SeafloorType.SEAMOUNT)
    test_data = factory.build()
    
    # Save data for inspection
    np.save("debug_depth.npy", test_data['depth_data'])
    
    # Enable debug logging
    logging.getLogger('core').setLevel(logging.DEBUG)
    
    # Continue with test
    assert test_data['depth_data'].shape == (64, 64)
```

## 🔄 Test Maintenance

### Adding New Tests

#### For New Features
```python
# 1. Add unit tests
def test_new_feature_basic_functionality():
    """Test basic functionality of new feature."""
    pass

# 2. Add integration tests
def test_new_feature_integration():
    """Test new feature integration with existing code."""
    pass

# 3. Add performance tests if applicable
def test_new_feature_performance():
    """Test performance of new feature."""
    pass
```

#### For Bug Fixes
```python
# 1. Add regression test
def test_bug_fix_regression():
    """Test that bug doesn't reoccur."""
    # Reproduce the bug conditions
    # Assert the fix works
    pass
```

### Test Suite Maintenance Tasks

#### Regular Maintenance (Weekly)
- [ ] Review test execution times
- [ ] Check coverage reports
- [ ] Update test data if needed
- [ ] Review failed tests in CI

#### Monthly Maintenance
- [ ] Update test dependencies
- [ ] Review and optimize slow tests
- [ ] Clean up obsolete tests
- [ ] Update test documentation

#### Release Maintenance
- [ ] Full test suite execution
- [ ] Performance regression testing
- [ ] Update test baselines
- [ ] Generate release test report

## 📚 Test Resources and Tools

### Required Testing Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| **pytest** | Core testing framework | ≥7.0.0 |
| **pytest-cov** | Coverage reporting | ≥4.0.0 |
| **pytest-html** | HTML test reports | ≥3.1.0 |
| **pytest-mock** | Enhanced mocking | ≥3.10.0 |
| **pytest-xdist** | Parallel testing | ≥3.0.0 |
| **pytest-benchmark** | Performance benchmarking | ≥4.0.0 |

### Optional Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **black** | Code formatting | `black tests/` |
| **flake8** | Code linting | `flake8 tests/` |
| **isort** | Import sorting | `isort tests/` |
| **bandit** | Security scanning | `bandit -r tests/` |
| **mypy** | Type checking | `mypy tests/` |

### IDE Integration

#### VS Code Settings
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests",
        "-v",
        "--tb=short"
    ],
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    "python.defaultInterpreterPath": "./venv/bin/python"
}
```

#### PyCharm Configuration
- Set test runner to pytest
- Configure coverage measurement
- Enable automatic test discovery
- Set up run configurations for different test categories

## 📞 Support and Contributing

### Getting Help with Tests

1. **Check existing test examples** in the test suite
2. **Review test documentation** in this README
3. **Run diagnostic commands**:
   ```bash
   python tests/run_tests_advanced.py --quick
   ./test_automation.sh setup
   ```
4. **Check GitHub Issues** for test-related problems
5. **Contact maintainers** for complex testing issues

### Contributing Tests

#### Requirements for Test Contributions
- [ ] Tests follow naming conventions
- [ ] AAA pattern used consistently
- [ ] Appropriate test markers applied
- [ ] Documentation updated
- [ ] Coverage maintained/improved
- [ ] Performance impact considered

#### Test Review Process
1. **Automated checks** run on PR
2. **Coverage analysis** performed
3. **Performance impact** assessed
4. **Code quality** reviewed
5. **Documentation** validated

### Best Practices Summary

1. **Write tests first** (TDD approach recommended)
2. **Use descriptive names** for tests and fixtures
3. **Keep tests independent** and isolated
4. **Mock external dependencies** appropriately
5. **Test edge cases** and error conditions
6. **Maintain good coverage** without over-testing
7. **Write fast tests** that provide value
8. **Use factories** for complex test data
9. **Document test purpose** and expectations
10. **Clean up resources** properly
