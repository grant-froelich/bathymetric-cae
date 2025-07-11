# Bathymetric CAE Processing v2.0

Advanced bathymetric grid processing using ensemble Convolutional Autoencoders with domain-specific enhancements, constitutional AI constraints, expert review systems, and comprehensive testing infrastructure.

## 🌊 Features

### Core Capabilities
- **🤖 Ensemble Model Architecture**: Multiple CAE variants for improved robustness
- **🧠 Adaptive Processing**: Seafloor type-specific processing strategies
- **⚖️ Constitutional AI Constraints**: Ensures bathymetric data integrity
- **👥 Expert Review System**: Human-in-the-loop validation workflow
- **📊 Comprehensive Quality Metrics**: IHO S-44 compliant assessment
- **🎯 Multi-objective Optimization**: Customizable quality metric weights
- **🧪 Comprehensive Test Suite**: Professional-grade testing infrastructure

### Advanced Model Architectures
- **AdvancedCAE**: Full-featured model with residual and skip connections
- **UncertaintyCAE**: Dual-output model estimating depth and uncertainty
- **LightweightCAE**: Resource-efficient variant for constrained environments
- **Custom Loss Functions**: Domain-specific edge preservation and consistency losses

### Seafloor Intelligence
- **Automatic Classification**: Shallow coastal, continental shelf, deep ocean, abyssal plain, seamount
- **Adaptive Parameters**: Processing strategies tailored to seafloor characteristics
- **Feature Preservation**: Maintains critical bathymetric features during processing

### Quality Assurance & Testing
- **Unit Testing**: 85%+ code coverage with comprehensive test suite
- **Integration Testing**: End-to-end pipeline validation
- **Performance Testing**: Memory usage and speed benchmarks
- **Mock Testing**: Isolated testing without external dependencies
- **CI/CD Integration**: Automated testing with GitHub Actions

## 🚀 Installation

### Prerequisites
- Python 3.8+
- GDAL system libraries
- CUDA-capable GPU (optional, recommended for training)

### Quick Install
```bash
git clone https://github.com/your-repo/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae
pip install -r requirements.txt
```

### Development Install with Testing
```bash
pip install -e .
pip install -r tests/requirements-test.txt
```

### GPU Support (Recommended)
```bash
pip install tensorflow[and-cuda]
```

### Optional Hydrographic Tools
```bash
pip install pyproj rasterio fiona geopandas
```

## 🏃‍♂️ Quick Start

### Basic Processing
```bash
python main.py
```

### Production Configuration
```bash
python main.py \
    --input /path/to/bathymetric/files \
    --output /path/to/processed/output \
    --enable-adaptive \
    --enable-expert-review \
    --enable-constitutional \
    --ensemble-size 5 \
    --quality-threshold 0.8
```

### Custom Training
```bash
python main.py \
    --epochs 200 \
    --batch-size 16 \
    --learning-rate 0.0005 \
    --validation-split 0.25 \
    --grid-size 1024
```

### Configuration Management
```bash
# Save configuration
python main.py --save-config production_config.json

# Load configuration
python main.py --config production_config.json
```

## 🧪 Testing

The project includes a comprehensive test suite with multiple testing approaches and automation tools.

### Quick Testing
```bash
# Run all tests
make test

# Run specific test categories
make test-unit           # Unit tests only
make test-integration    # Integration tests
make test-performance    # Performance benchmarks

# Generate coverage report
make test-coverage
```

### Advanced Test Runner
```bash
# Comprehensive test suite with analysis
python tests/run_tests_advanced.py --category all --verbose

# Quick unit test run
python tests/run_tests_advanced.py --quick

# Performance benchmarks only
python tests/run_tests_advanced.py --category performance
```

### Automated Test Script
```bash
# Make script executable
chmod +x test_automation.sh

# Run all tests with reports
./test_automation.sh all --verbose

# Quick test run
./test_automation.sh unit --quick

# Run with cleanup
./test_automation.sh all --cleanup
```

### Test Categories

| Category | Description | Files | Execution Time |
|----------|-------------|-------|----------------|
| **Unit** | Fast, isolated component tests | `test_*.py` | < 2 minutes |
| **Integration** | End-to-end pipeline tests | `test_integration.py` | < 5 minutes |
| **Performance** | Speed and memory benchmarks | `test_performance.py` | < 3 minutes |

### Test Coverage Standards
- **Overall Coverage**: 85%+ required
- **Core Modules**: 95%+ (processing, models, quality metrics)
- **Integration Points**: 90%+ (pipeline, data processing)

### CI/CD Integration
Tests automatically run on:
- Pull requests to main/develop branches
- Pushes to main branch
- Nightly builds for performance regression testing

View test results and coverage reports in GitHub Actions artifacts.

## 📁 Module Architecture

```
enhanced_bathymetric_cae/
├── 📋 README.md                    # This file
├── 📦 requirements.txt             # Dependencies
├── ⚙️ setup.py                     # Package installation
├── 🔧 Makefile                     # Test automation commands
│
├── 🔧 config/                      # Configuration management
│   ├── __init__.py
│   └── config.py                   # Config dataclass with validation
│
├── 🧠 core/                        # Domain-specific functionality
│   ├── __init__.py
│   ├── enums.py                    # SeafloorType and other enums
│   ├── constraints.py              # Constitutional AI rules
│   ├── quality_metrics.py          # IHO-compliant quality assessment
│   └── adaptive_processor.py       # Seafloor classification & adaptation
│
├── 🤖 models/                      # AI model architectures
│   ├── __init__.py
│   ├── ensemble.py                 # Ensemble management & prediction
│   └── architectures.py            # CAE variants & custom losses
│
├── ⚡ processing/                   # Data processing pipeline
│   ├── __init__.py
│   ├── data_processor.py           # File I/O & preprocessing
│   └── pipeline.py                 # Main processing orchestration
│
├── 👥 review/                      # Expert review system
│   ├── __init__.py
│   └── expert_system.py            # Human-in-the-loop validation
│
├── 🛠️ utils/                       # Utilities
│   ├── __init__.py
│   ├── logging_utils.py            # Colored logging & setup
│   ├── memory_utils.py             # Memory monitoring & GPU optimization
│   └── visualization.py            # Plotting & report generation
│
├── 💻 cli/                         # Command-line interface
│   ├── __init__.py
│   └── interface.py                # Argument parsing & config updates
│
├── 🧪 tests/                       # Comprehensive test suite
│   ├── conftest.py                 # Shared test fixtures
│   ├── pytest.ini                 # Pytest configuration
│   ├── test_config.json           # Test runner settings
│   ├── requirements-test.txt      # Testing dependencies
│   │
│   ├── test_*.py                   # Unit test files
│   ├── test_integration.py         # Integration tests
│   ├── test_performance.py         # Performance tests
│   │
│   ├── fixtures/                   # Advanced test fixtures
│   ├── factories/                  # Test data factories
│   ├── utils/                      # Test utilities & mocks
│   ├── test_fixtures/              # Sample data generators
│   │
│   ├── run_tests_advanced.py       # Advanced test runner
│   └── test_automation.sh          # Test automation script
│
└── 🎯 main.py                      # Application entry point
```

## 🗃️ Supported File Formats

| Format | Extension | Description | Uncertainty Support |
|--------|-----------|-------------|-------------------|
| BAG | `.bag` | Bathymetric Attributed Grid | ✅ |
| GeoTIFF | `.tif`, `.tiff` | Tagged Image File Format | ❌ |
| ASCII Grid | `.asc` | ESRI ASCII Grid | ❌ |
| XYZ | `.xyz` | Point cloud data | ❌ |

## 🌊 Seafloor Classification

The system automatically detects and adapts processing for different seafloor environments:

| Seafloor Type | Depth Range | Key Characteristics | Processing Strategy |
|---------------|-------------|-------------------|-------------------|
| **Shallow Coastal** | 0-200m | High variability, complex features | Conservative smoothing, high feature preservation |
| **Continental Shelf** | 200-2000m | Moderate slopes, sedimentary features | Balanced approach |
| **Deep Ocean** | 2000-6000m | Steep slopes, volcanic features | Edge-preserving, moderate smoothing |
| **Abyssal Plain** | 6000-11000m | Low relief, sedimentary | Aggressive smoothing, noise reduction |
| **Seamount** | Variable | High relief, steep gradients | Maximum feature preservation |

## 📊 Quality Metrics

### Primary Metrics
- **🎯 SSIM (Structural Similarity)**: Measures structural preservation (0-1, higher better)
- **🔧 Feature Preservation**: Bathymetric feature retention score (0-1, higher better)
- **📏 Depth Consistency**: Local measurement consistency (0-1, higher better)
- **🌊 Roughness**: Seafloor surface roughness (lower better)
- **⚓ IHO S-44 Compliance**: Hydrographic standards compliance (0-1, higher better)

### Composite Quality Score
Weighted combination: `0.3×SSIM + 0.2×(1-Roughness) + 0.3×FeaturePreservation + 0.2×Consistency`

### Quality Thresholds
- **🟢 High Quality** (>0.8): Ready for hydrographic use
- **🟡 Medium Quality** (0.6-0.8): Suitable with review
- **🔴 Low Quality** (<0.6): Requires expert attention

## 👥 Expert Review System

### Automatic Flagging
Files are flagged for expert review based on:
- Composite quality below threshold (default: 0.7)
- Feature preservation < 0.5
- IHO S-44 compliance < 0.7
- SSIM score < 0.6

### Review Database
- SQLite database for tracking reviews
- Persistent flagging and comments
- Review status and history
- Quality improvement tracking

### Review Workflow
1. **Automatic Processing**: All files processed with quality assessment
2. **Quality Flagging**: Low-quality results flagged for review
3. **Expert Evaluation**: Human reviewers assess flagged files
4. **Database Tracking**: Reviews stored with ratings and comments
5. **Report Generation**: Summary reports for quality control

## ⚙️ Configuration

### Example Configuration File
```json
{
  "input_folder": "/data/bathymetry/input",
  "output_folder": "/data/bathymetry/processed",
  "model_path": "models/cae_ensemble.h5",
  
  "epochs": 150,
  "batch_size": 12,
  "learning_rate": 0.0008,
  "validation_split": 0.2,
  
  "grid_size": 512,
  "ensemble_size": 5,
  "base_filters": 32,
  "depth": 4,
  "dropout_rate": 0.2,
  
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "enable_constitutional_constraints": true,
  "quality_threshold": 0.75,
  
  "ssim_weight": 0.3,
  "roughness_weight": 0.2,
  "feature_preservation_weight": 0.3,
  "consistency_weight": 0.2
}
```

### Performance Tuning

#### Memory Optimization
```bash
# Reduce memory usage
python main.py --ensemble-size 1 --grid-size 256 --batch-size 4

# For large datasets
python main.py --max-workers 1 --batch-size 2
```

#### GPU Optimization
```bash
# Enable mixed precision (default)
python main.py --use-mixed-precision

# Disable GPU if needed
python main.py --no-gpu
```

#### Quality vs Speed Trade-offs
```bash
# Fast processing (lower quality)
python main.py --ensemble-size 1 --epochs 50 --quality-threshold 0.5

# High quality (slower)
python main.py --ensemble-size 7 --epochs 300 --quality-threshold 0.85
```

## 🔧 Development

### Setting Up Development Environment
```bash
git clone https://github.com/your-repo/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae
pip install -e ".[dev]"

# Install testing dependencies
pip install -r tests/requirements-test.txt

# Setup pre-commit hooks (optional)
pre-commit install
```

### Running Tests During Development
```bash
# Quick unit tests during development
make test-unit

# Test specific module
pytest tests/test_models_architectures.py -v

# Test with coverage
make test-coverage

# Performance tests
make test-performance

# Run tests in parallel
pytest tests/ -n auto
```

### Code Quality Checks
```bash
# Run all quality checks
make lint-tests

# Format code
make format-tests

# Check imports
isort tests/ --check-only

# Security scan
bandit -r . -x tests/
```

### Adding New Features

#### New Quality Metric
```python
# In core/quality_metrics.py
@staticmethod
def calculate_new_metric(data: np.ndarray) -> float:
    """Calculate new quality metric."""
    # Implementation here
    return metric_value

# Add corresponding test in tests/test_core_quality_metrics.py
def test_new_metric_calculation():
    """Test new metric calculation."""
    # Test implementation
    pass
```

#### New Seafloor Type
```python
# In core/enums.py
class SeafloorType(Enum):
    NEW_TYPE = "new_type"

# In core/adaptive_processor.py
def _new_type_strategy(self, depth_data: np.ndarray) -> Dict:
    """Processing strategy for new seafloor type."""
    return {'smoothing_factor': 0.4, ...}

# Add tests in tests/test_core_adaptive_processor.py
def test_new_seafloor_type_classification():
    """Test new seafloor type classification."""
    # Test implementation
    pass
```

#### New Model Architecture
```python
# In models/architectures.py
class NewCAE(AdvancedCAE):
    """New CAE variant."""
    
    def create_model(self, input_shape: tuple, variant_config: Dict = None):
        # Implementation here
        return model

# Add tests in tests/test_models_architectures.py
def test_new_cae_creation():
    """Test new CAE model creation."""
    # Test implementation
    pass
```

### Testing Guidelines

#### Writing Good Tests
```python
# Follow AAA pattern: Arrange, Act, Assert
def test_quality_metric_calculation():
    # Arrange
    test_data = np.ones((10, 10))
    expected_result = 1.0
    
    # Act
    result = calculate_quality_metric(test_data)
    
    # Assert
    assert result == pytest.approx(expected_result, abs=1e-6)
```

#### Using Test Fixtures
```python
# Use provided fixtures for common test scenarios
def test_with_bathymetric_data(sample_depth_data):
    """Test using fixture-provided data."""
    processor = BathymetricProcessor(config)
    result = processor.process(sample_depth_data)
    assert result.shape == sample_depth_data.shape
```

#### Performance Testing
```python
def test_processing_performance():
    """Test processing performance benchmarks."""
    with PerformanceMonitor().monitor() as monitor:
        # Perform operation
        process_large_dataset()
    
    results = monitor.get_results()
    assert results['execution_time'] < 30.0  # 30 second limit
    assert results['max_memory_mb'] < 2000   # 2GB limit
```

### Code Style Guidelines
- **PEP 8 compliance**: Use `black` for formatting
- **Type hints**: Add type annotations for all functions
- **Docstrings**: Use Google-style docstrings
- **Error handling**: Comprehensive try/except blocks
- **Logging**: Use module-level loggers
- **Testing**: Write tests for all new functionality

### Debugging and Profiling
```bash
# Debug mode with verbose logging
python main.py --log-level DEBUG --input test_data/ --epochs 5

# Profile memory usage
python -m memory_profiler main.py --input test_data/ --epochs 1

# Profile execution time
python -m cProfile -o profile_results.prof main.py --input test_data/
```

## 📈 Output Files

### Processed Bathymetry
- **Format**: Same as input (BAG, GeoTIFF, etc.)
- **Naming**: `enhanced_{original_filename}`
- **Metadata**: Comprehensive processing information

### Quality Reports
- **Summary Report**: `enhanced_processing_summary.json`
- **Expert Reviews**: `expert_reviews/pending_reviews.json`
- **Training Logs**: `logs/training_history_*.csv`

### Test Reports
- **Coverage Report**: `htmlcov/index.html`
- **Test Results**: `test-results.xml`
- **Performance Report**: `test-performance.json`

### Visualizations
- **Comparison Plots**: `plots/enhanced_comparison_*.png`
- **Training History**: `plots/training_history_ensemble_*.png`
- **Quality Metrics**: Embedded in comparison plots

### Metadata Structure
```json
{
  "PROCESSING": {
    "PROCESSING_DATE": "2024-01-15T10:30:45",
    "PROCESSING_SOFTWARE": "Enhanced Bathymetric CAE v2.0",
    "MODEL_TYPE": "Ensemble Convolutional Autoencoder",
    "ENSEMBLE_SIZE": "5",
    "SEAFLOOR_TYPE": "continental_shelf"
  },
  "QUALITY": {
    "QUALITY_COMPOSITE_QUALITY": "0.8234",
    "QUALITY_SSIM": "0.8567",
    "QUALITY_FEATURE_PRESERVATION": "0.7891",
    "QUALITY_HYDROGRAPHIC_COMPLIANCE": "0.8456"
  },
  "ADAPTIVE": {
    "ADAPTIVE_SMOOTHING_FACTOR": "0.5",
    "ADAPTIVE_EDGE_PRESERVATION": "0.6",
    "ADAPTIVE_NOISE_THRESHOLD": "0.15"
  },
  "TESTING": {
    "TEST_COVERAGE": "87.5%",
    "QUALITY_GATES_PASSED": true,
    "LAST_TEST_RUN": "2024-01-15T09:15:30"
  }
}
```

## 🚨 Troubleshooting

### Common Issues

#### Memory Problems
```bash
# Symptoms: Out of memory errors, system slowdown
# Solutions:
python main.py --ensemble-size 1 --grid-size 256 --batch-size 2
python main.py --max-workers 1
```

#### GPU Issues
```bash
# Symptoms: CUDA errors, GPU not detected
# Solutions:
python main.py --no-gpu  # Use CPU only
nvidia-smi  # Check GPU status
pip install tensorflow[and-cuda]  # Reinstall GPU support
```

#### File Format Problems
```bash
# Symptoms: Cannot open file, format not supported
# Solutions:
gdalinfo your_file.bag  # Check file validity
pip install gdal  # Ensure GDAL is installed
```

#### Test Failures
```bash
# Symptoms: Tests failing, coverage below threshold
# Solutions:
make test-unit  # Run unit tests only
pytest tests/test_specific.py -v  # Debug specific test
python tests/run_tests_advanced.py --quick  # Quick test run
```

#### Quality Issues
```bash
# Symptoms: All files flagged for review, poor quality scores
# Solutions:
python main.py --quality-threshold 0.5  # Lower threshold
python main.py --epochs 200  # More training
python main.py --ensemble-size 7  # Larger ensemble
```

#### Performance Issues
```bash
# Symptoms: Very slow processing
# Solutions:
python main.py --ensemble-size 1  # Smaller ensemble
python main.py --grid-size 256  # Smaller grid
python main.py --epochs 50  # Fewer epochs
```

### Debug Mode
```bash
python main.py --log-level DEBUG --input small_dataset/ --epochs 5
```

### Test Debugging
```bash
# Run specific test with debug output
pytest tests/test_specific.py::TestClass::test_method -s -vv

# Run tests with logging
pytest --log-cli-level=DEBUG tests/

# Debug test failures
pytest --pdb tests/
```

### Getting Help
1. **Check logs**: `logs/bathymetric_processing.log`
2. **Review test results**: `htmlcov/index.html`
3. **Run diagnostics**: `./test_automation.sh setup`
4. **Check system resources**: Ensure adequate RAM and disk space
5. **Validate environment**: `python tests/run_tests_advanced.py --quick`

## 🤝 Contributing

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-new-metric`
3. **Follow** the code style guidelines
4. **Add** tests for new functionality (required)
5. **Update** documentation
6. **Ensure** tests pass: `make test`
7. **Submit** a pull request

### Development Workflow
```bash
# Setup development environment
git clone your-fork-url
cd enhanced-bathymetric-cae
pip install -e ".[dev]"
pip install -r tests/requirements-test.txt

# Create feature branch
git checkout -b feature-description

# Make changes and test
make test-unit
pytest tests/test_new_feature.py -v

# Run full test suite
make test

# Commit and push
git add .
git commit -m "Add new feature: description"
git push origin feature-description
```

### Code Review Checklist
- [ ] Code follows PEP 8 guidelines
- [ ] All functions have type hints and docstrings
- [ ] Error handling is comprehensive
- [ ] Tests pass with 85%+ coverage
- [ ] Performance tests within thresholds
- [ ] Documentation is updated
- [ ] No breaking changes to existing API

### Testing Requirements
- **Unit tests** required for all new functions
- **Integration tests** for new features
- **Performance tests** for processing changes
- **85%+ code coverage** maintained
- **Quality gates** must pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this software in your research, please cite:

```bibtex
@software{enhanced_bathymetric_cae_2025,
  title={Bathymetric CAE Processing: Advanced Deep Learning for Seafloor Data},
  author={Grant Froelich},
  year={2025},
  version={2.0.0},
  url={https://github.com/noaa-ocs-hydrography/bathymetric-cae},
  doi={10.xxxx/xxxxx}
}
```

## 🌐 Related Projects

- **IHO S-44 Standards**: [International Hydrographic Organization](https://iho.int/)
- **GDAL**: [Geospatial Data Abstraction Library](https://gdal.org/)
- **TensorFlow**: [Machine Learning Platform](https://tensorflow.org/)
- **BAG Format**: [Bathymetric Attributed Grid](https://www.opennavsurf.org/bag)
- **Pytest**: [Python Testing Framework](https://pytest.org/)

## 📞 Support

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/noaa-ocs-hydrography/bathymetric-cae/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/noaa-ocs=hydrography/bathymetric-cae/discussions)
- **📧 Email**: grant.froelich@noaa.gov
- **📚 Documentation**: [Wiki](https://github.com/noaa-ocs-hydrography/bathymetric-cae/docs)
- **🧪 Testing Guide**: [Testing Best Practices](tests/README.md)

## 🔄 Changelog

### v2.0.0 (Current)
- ✨ **New**: Complete modular architecture
- ✨ **New**: Ensemble model system with multiple CAE variants
- ✨ **New**: Adaptive processing based on seafloor classification
- ✨ **New**: Constitutional AI constraints for data integrity
- ✨ **New**: Expert review system with database tracking
- ✨ **New**: Advanced quality metrics (IHO S-44 compliant)
- ✨ **New**: Custom loss functions for bathymetric data
- ✨ **New**: Comprehensive visualization and reporting
- ✨ **New**: Professional test suite with 85%+ coverage
- ✨ **New**: CI/CD integration with GitHub Actions
- ✨ **New**: Performance monitoring and benchmarking
- ✨ **New**: Automated test execution and reporting
- 🔧 **Improved**: Memory management and GPU optimization
- 🔧 **Improved**: Error handling and logging
- 🔧 **Improved**: Configuration management
- 📚 **Updated**: Complete documentation rewrite

### v1.0.0 (Legacy)
- 🎯 Initial monolithic implementation
- 🤖 Basic CAE architecture
- 📁 Simple file processing
- 📊 Basic quality metrics
