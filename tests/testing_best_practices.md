# Testing Best Practices for Enhanced Bathymetric CAE Processing

## Overview

This document outlines testing best practices, patterns, and guidelines for maintaining high-quality, reliable tests in the Enhanced Bathymetric CAE Processing project.

## Test Structure and Organization

### Directory Structure
```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                    # Pytest configuration
├── test_config.json              # Test runner configuration
├── requirements-test.txt          # Test dependencies
│
├── test_*.py                      # Unit test files
├── test_integration.py            # Integration tests
├── test_performance.py            # Performance tests
│
├── fixtures/                     # Advanced test fixtures
│   ├── __init__.py
│   ├── advanced_fixtures.py
│   └── mock_fixtures.py
│
├── factories/                    # Test data factories
│   ├── __init__.py
│   └── data_factory.py
│
├── utils/                        # Test utilities
│   ├── __init__.py
│   ├── test_helpers.py
│   ├── mock_gdal.py
│   └── performance_monitor.py
│
└── test_fixtures/                # Sample data generators
    ├── __init__.py
    └── sample_data_generator.py
```

### Naming Conventions

#### Test Files
- Unit tests: `test_<module_name>.py`
- Integration tests: `test_integration.py`
- Performance tests: `test_performance.py`

#### Test Functions
- Descriptive names: `test_<what_is_being_tested>`
- Use underscores for readability
- Include expected behavior: `test_model_prediction_returns_correct_shape`
- Include conditions: `test_config_validation_with_invalid_epochs`

#### Test Classes
- Group related tests: `class TestBathymetricProcessor:`
- Use descriptive class names
- Follow PascalCase convention

## Test Categories and Markers

### Test Markers
Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_quality_metrics_calculation():
    """Unit test for quality metrics."""
    pass

@pytest.mark.integration
def test_complete_pipeline_execution():
    """Integration test for full pipeline."""
    pass

@pytest.mark.performance
def test_model_inference_speed():
    """Performance test for model inference."""
    pass

@pytest.mark.slow
def test_large_dataset_processing():
    """Test that takes significant time."""
    pass

@pytest.mark.gpu
def test_gpu_memory_optimization():
    """Test requiring GPU."""
    pass
```

### Running Specific Test Categories
```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run GPU tests only
pytest -m gpu
```

## Writing Effective Tests

### AAA Pattern (Arrange, Act, Assert)

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

### Test Data Management

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

#### Parameterized Tests for Multiple Scenarios
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

### Error Condition Testing

```python
def test_config_validation_with_invalid_epochs():
    """Test that invalid epochs raise appropriate error."""
    with pytest.raises(ValueError, match="epochs must be positive"):
        Config(epochs=0)

def test_data_processor_handles_corrupt_files():
    """Test graceful handling of corrupt files."""
    processor = BathymetricProcessor(config)
    
    with pytest.raises(ValueError, match="Cannot open file"):
        processor.preprocess_bathymetric_grid("nonexistent_file.bag")
```

## Mocking and Fixtures

### GDAL Mocking Pattern
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

### TensorFlow Model Mocking
```python
def test_ensemble_prediction_with_mock_models():
    ensemble = BathymetricEnsemble(config)
    
    # Create mock models
    mock_models = [Mock() for _ in range(3)]
    for i, model in enumerate(mock_models):
        model.predict.return_value = np.ones((1, 64, 64, 1)) * (i + 1)
    
    ensemble.models = mock_models
    ensemble.weights = [1/3, 1/3, 1/3]
    
    input_data = np.random.random((1, 64, 64, 1))
    prediction, metrics = ensemble.predict_ensemble(input_data)
    
    # Should return average of mock predictions
    assert np.allclose(prediction, 2.0)  # (1+2+3)/3 = 2
```

## Performance Testing

### Benchmark Tests
```python
def test_model_inference_performance(benchmark):
    """Benchmark model inference speed."""
    model = create_test_model()
    test_data = np.random.random((10, 64, 64, 1))
    
    # Benchmark the inference
    result = benchmark(model.predict, test_data, verbose=0)
    
    assert result.shape == test_data.shape
```

### Memory Usage Testing
```python
def test_memory_usage_within_limits():
    """Test that processing stays within memory limits."""
    with PerformanceMonitor().monitor() as monitor:
        # Perform memory-intensive operation
        large_dataset = create_large_test_dataset(1000)
        process_dataset(large_dataset)
    
    results = monitor.get_results()
    assert results['max_memory_mb'] < 2000  # 2GB limit
```

## Integration Testing Patterns

### End-to-End Pipeline Testing
```python
def test_complete_pipeline_with_mock_data():
    """Test complete pipeline execution."""
    with IntegratedMockFixture().complete_mock_environment() as env:
        pipeline = EnhancedBathymetricCAEPipeline(config)
        
        # Run pipeline
        pipeline.run(
            input_folder="test_input",
            output_folder="test_output", 
            model_path="test_model.h5"
        )
        
        # Verify outputs
        output_files = list(Path("test_output").glob("enhanced_*.bag"))
        assert len(output_files) > 0
        
        # Verify processing summary
        assert Path("enhanced_processing_summary.json").exists()
```

### Database Integration Testing
```python
def test_expert_review_workflow():
    """Test expert review database workflow."""
    with MockDatabaseFixture().mock_database_environment() as (mock_db, db_path):
        # Flag region for review
        mock_db.flag_for_review("test.bag", (0, 0, 100, 100), "low_quality", 0.8)
        
        # Get pending reviews
        pending = mock_db.get_pending_reviews()
        assert len(pending) == 1
        assert pending[0]['filename'] == "test.bag"
```

## Test Data Management

### Realistic Test Data
```python
def create_realistic_seamount_data():
    """Create realistic seamount bathymetry for testing."""
    x = np.linspace(-5, 5, 64)
    y = np.linspace(-5, 5, 64)
    X, Y = np.meshgrid(x, y)
    
    # Base deep ocean floor
    depth = np.full((64, 64), -3000.0)
    
    # Add seamount peak
    distance = np.sqrt(X**2 + Y**2)
    seamount_height = 2500 * np.exp(-distance**2 / 4)
    depth += seamount_height
    
    return depth.astype(np.float32)
```

### Test Data Cleanup
```python
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    
    # Cleanup test output files
    for pattern in ["enhanced_*.bag", "test_*.tif", "*.log"]:
        for file_path in Path(".").glob(pattern):
            file_path.unlink(missing_ok=True)
```

## Quality Assurance

### Coverage Requirements
- **Minimum Coverage**: 85% overall
- **Critical Modules**: 95% (core processing, quality metrics)
- **Integration Points**: 90% (pipeline, data processing)

### Performance Standards
- **Unit Test Speed**: < 30 seconds per test
- **Integration Test Speed**: < 5 minutes per test
- **Memory Usage**: < 2GB during testing
- **Total Test Suite**: < 10 minutes

### Code Quality Checks
```bash
# Run all quality checks
make lint-tests
make format-tests

# Check test coverage
pytest --cov=. --cov-report=html --cov-fail-under=85
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
- name: Run comprehensive tests
  run: |
    python tests/run_tests_advanced.py --category all --verbose
    
- name: Check quality gates
  run: |
    python tests/check_quality_gates.py
    
- name: Upload test artifacts
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: |
      test-results.xml
      htmlcov/
      test-performance.json
```

## Debugging Tests

### Debug Mode
```bash
# Run specific test with debug output
pytest tests/test_specific.py::TestClass::test_method -s -vv

# Run with logging
pytest tests/ --log-cli-level=DEBUG

# Run with Python debugger
pytest tests/ --pdb
```

### Test Data Inspection
```python
def test_with_data_inspection():
    """Test with intermediate data inspection."""
    processor = BathymetricProcessor(config)
    
    # Enable debug logging
    logging.getLogger('processing').setLevel(logging.DEBUG)
    
    result = processor.preprocess_bathymetric_grid("test.bag")
    
    # Save intermediate results for inspection
    np.save("debug_output.npy", result[0])
    
    assert result[0].shape == (64, 64, 1)
```

## Common Testing Pitfalls

### Avoid These Anti-patterns

#### 1. Tests That Depend on External Resources
```python
# ❌ Bad: depends on external file
def test_real_file_processing():
    result = process_file("/path/to/real/file.bag")
    assert result is not None

# ✅ Good: uses mock data
def test_file_processing_with_mock_data():
    with mock_gdal_environment():
        result = process_file("mock_file.bag")
        assert result is not None
```

#### 2. Non-Deterministic Tests
```python
# ❌ Bad: random behavior
def test_random_processing():
    data = np.random.random((64, 64))
    result = process_data(data)
    assert result.mean() > 0.4  # May fail randomly

# ✅ Good: controlled randomness
def test_deterministic_processing():
    np.random.seed(42)
    data = np.random.random((64, 64))
    result = process_data(data)
    assert result.mean() == pytest.approx(0.5, abs=0.1)
```

#### 3. Tests That Test Implementation Details
```python
# ❌ Bad: tests internal implementation
def test_internal_method_calls():
    processor = BathymetricProcessor(config)
    with patch.object(processor, '_validate_data') as mock_validate:
        processor.process_file("test.bag")
        mock_validate.assert_called_once()

# ✅ Good: tests behavior
def test_file_processing_behavior():
    processor = BathymetricProcessor(config)
    result = processor.process_file("test.bag")
    assert result.shape == (64, 64, 1)
    assert not np.isnan(result).any()
```

## Test Maintenance

### Regular Maintenance Tasks
1. **Review Test Performance**: Identify and optimize slow tests
2. **Update Test Data**: Refresh test datasets with realistic scenarios
3. **Check Coverage**: Ensure coverage remains above thresholds
4. **Review Test Quality**: Remove or improve low-quality tests
5. **Update Mocks**: Keep mocks in sync with real implementations

### Test Refactoring Guidelines
- Extract common setup into fixtures
- Use parameterized tests for similar test cases
- Break large tests into smaller, focused tests
- Remove obsolete tests when features change
- Keep test code clean and readable

## Resources and Tools

### Useful Testing Libraries
- **pytest**: Core testing framework
- **pytest-cov**: Coverage reporting
- **pytest-html**: HTML test reports
- **pytest-xdist**: Parallel test execution
- **pytest-benchmark**: Performance benchmarking
- **factory-boy**: Test data factories
- **responses**: HTTP request mocking
- **freezegun**: Time mocking

### VS Code Testing Setup
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests",
        "-v",
        "--tb=short"
    ],
    "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest tests/test_*.py -x
        language: system
        pass_filenames: false
        always_run: true
```

## Conclusion

Following these testing best practices ensures:
- **Reliability**: Tests consistently pass and catch regressions
- **Maintainability**: Tests are easy to understand and modify
- **Performance**: Test suite runs efficiently
- **Coverage**: Critical functionality is thoroughly tested
- **Quality**: High-quality code through comprehensive testing

Remember: **Good tests are an investment in code quality and developer productivity.**