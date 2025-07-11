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
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8-3.11
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 10 GB free space
- **Conda**: Miniconda or Anaconda (strongly recommended for GDAL)

### Recommended Installation (Conda)

The **conda installation is strongly recommended** for reliable GDAL support and dependency management.

#### Step 1: Install Conda
If you don't have conda installed:
- **Miniconda** (minimal): https://docs.conda.io/en/latest/miniconda.html
- **Anaconda** (full): https://www.anaconda.com/products/distribution

#### Step 2: Create Environment and Install
```bash
# Clone the repository
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd enhanced-bathymetric-cae

# Create conda environment (includes GDAL and dependencies)
conda env create -f environment.yml

# Activate environment
conda activate bathymetric-cae

# Install the package
pip install -e .

# Verify installation
bathymetric-cae --version
python -c "from osgeo import gdal; print(f'GDAL {gdal.__version__} ready!')"
```

#### Step 3: Quick Test
```bash
# Generate test data and run basic processing
python -c "
from tests.test_fixtures.sample_data_generator import TestDataGenerator
from pathlib import Path
TestDataGenerator.create_test_dataset(Path('test_data'), 2)
print('✅ Test data generated successfully')
"

# Run quick test
bathymetric-cae --input test_data --output test_output --epochs 2 --ensemble-size 1
```

### Alternative: pip Installation (Limited Support)

⚠️ **Warning**: pip installation may encounter GDAL dependency issues. Use conda method above for best results.

#### For Advanced Users Only
```bash
# Install system GDAL first (varies by platform)
# Ubuntu/Debian: sudo apt install gdal-bin libgdal-dev
# macOS: brew install gdal  
# Windows: Not recommended - use conda instead

# Install package
pip install enhanced-bathymetric-cae

# Verify (may fail with pip installation)
python -c "from osgeo import gdal; print('GDAL OK')"
```

### Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd enhanced-bathymetric-cae

# Create development environment
conda env create -f environment.yml
conda activate bathymetric-cae

# Install in development mode
pip install -e .

# Install additional test dependencies (if not in environment.yml)
pip install -r tests/requirements-test.txt

# Verify development setup
python tests/run_tests_advanced.py --quick
```

### Docker Installation

For containerized deployment:

```bash
# Using conda-based Dockerfile
docker build -f Dockerfile.conda -t bathymetric-cae .

# Run with sample data
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    bathymetric-cae \
    bathymetric-cae --input /app/data --output /app/output
```

### Platform-Specific Notes

#### Windows
- **Strongly recommend conda** - pip GDAL installation is very difficult on Windows
- Use Anaconda Prompt or PowerShell with conda in PATH
- Ensure Visual Studio Build Tools are installed if compiling from source

#### macOS
```bash
# Install conda via Homebrew (optional)
brew install miniconda

# Or download directly from conda website
# Then follow conda installation steps above
```

#### Linux (Ubuntu/Debian)
```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Follow conda installation steps above
```

### GPU Support

For CUDA GPU acceleration:

```bash
# During conda environment creation, or after:
conda activate bathymetric-cae

# Install GPU-enabled TensorFlow
pip uninstall tensorflow
pip install tensorflow[and-cuda]>=2.13.0

# Verify GPU detection
python -c "
import tensorflow as tf
print('GPUs available:', len(tf.config.list_physical_devices('GPU')))
"
```

### Troubleshooting Installation

#### Common GDAL Issues

**Problem**: `ImportError: No module named 'osgeo'`
```bash
# Solution: Use conda installation
conda activate bathymetric-cae
conda install -c conda-forge gdal --force-reinstall
```

**Problem**: `GDAL version mismatch`
```bash
# Solution: Ensure consistent conda environment
conda activate bathymetric-cae
conda update --all
```

**Problem**: `Memory errors during installation`
```bash
# Solution: Close other applications and try:
conda env create -f environment.yml --force
```

#### Verification Script

Test your installation:

```bash
# Save as test_installation.py
python << 'EOF'
#!/usr/bin/env python3
"""Test installation completeness."""

def test_imports():
    """Test critical imports."""
    try:
        # Test GDAL
        from osgeo import gdal, ogr, osr
        print(f"✅ GDAL {gdal.__version__}")
        
        # Test core packages  
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        # Test package import
        import enhanced_bathymetric_cae
        print("✅ Enhanced Bathymetric CAE package")
        
        # Test GPU (optional)
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPUs available: {len(gpus)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 Installation verification successful!")
    else:
        print("\n❌ Installation verification failed!")
        print("Try: conda env create -f environment.yml --force")
EOF
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
├── 📦 requirements.txt             # Pip dependencies (conda preferred)
├── 📦 environment.yml              # Conda environment (recommended)
├── ⚙️ setup.py                     # Package installation
├── 🔧 Makefile                     # Build and test automation
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
├── 🐳 Dockerfile.conda             # Conda-based Docker image
├── 🐳 docker-compose.yml           # Container orchestration
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
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd enhanced-bathymetric-cae

# Use conda for reliable GDAL support
conda env create -f environment.yml
conda activate bathymetric-cae
pip install -e .

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
make lint

# Format code
make format

# Check imports
isort tests/ --check-only

# Security scan
bandit -r . -x tests/
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
conda install -c conda-forge gdal  # Ensure GDAL is installed via conda
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

# Use conda for reliable GDAL support
conda env create -f environment.yml
conda activate bathymetric-cae
pip install -e .

# Install test dependencies
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
- **Conda-Forge**: [Community-led conda packages](https://conda-forge.org/)
- **TensorFlow**: [Machine Learning Platform](https://tensorflow.org/)
- **BAG Format**: [Bathymetric Attributed Grid](https://www.opennavsurf.org/bag)
- **Pytest**: [Python Testing Framework](https://pytest.org/)

## 📞 Support

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/noaa-ocs-hydrography/bathymetric-cae/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/noaa-ocs-hydrography/bathymetric-cae/discussions)
- **📧 Email**: grant.froelich@noaa.gov
- **📚 Documentation**: [Wiki](https://github.com/noaa-ocs-hydrography/bathymetric-cae/docs)
- **🧪 Testing Guide**: [Testing Best Practices](tests/README.md)

## 🔄 Changelog

### v2.0.0 (Current)
- ✨ **New**: Complete modular architecture
- ✨ **New**: Conda-forge integration for reliable GDAL support
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
- ✨ **New**: Docker containerization with conda
- 🔧 **Improved**: Memory management and GPU optimization
- 🔧 **Improved**: Error handling and logging
- 🔧 **Improved**: Configuration management
- 🔧 **Improved**: Cross-platform installation reliability
- 📚 **Updated**: Complete documentation rewrite with conda-first approach

### v1.0.0 (Legacy)
- 🎯 Initial monolithic implementation
- 🤖 Basic CAE architecture
- 📁 Simple file processing
- 📊 Basic quality metrics

---

**Next Steps**: After successful installation, continue with the [Quick Start Guide](docs/getting-started/quick-start.md)
