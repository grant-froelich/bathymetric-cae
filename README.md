# Enhanced Bathymetric CAE Processing v2.0

A modular, high-performance system for bathymetric data cleaning and processing using advanced ensemble convolutional autoencoders with domain-specific enhancements.

## 🌊 Features

### Core Capabilities
- **Ensemble Learning**: Multiple diverse models for improved robustness
- **Adaptive Processing**: Automatic seafloor type classification and parameter optimization
- **Constitutional AI Constraints**: Domain-specific rules ensuring physical plausibility
- **Expert Review System**: Human-in-the-loop validation and quality control
- **Comprehensive Quality Metrics**: Aligned with IHO hydrographic standards

### Advanced Features
- **Multi-format Support**: BAG, GeoTIFF, ASCII Grid, XYZ
- **Uncertainty Estimation**: Prediction confidence and uncertainty quantification
- **GPU Acceleration**: Mixed precision training and inference
- **Memory Optimization**: Efficient processing of large datasets
- **Visualization**: Enhanced plots and quality assessment dashboards

## 📁 Repository Structure

```
bathymetric_cae/
├── README.md
├── requirements.txt
├── setup.py
├── main.py                    # Main entry point
├── config/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   └── default_config.json   # Default settings
├── core/                     # Core domain logic
│   ├── __init__.py
│   ├── enums.py             # Domain enumerations
│   ├── constraints.py        # Constitutional AI constraints
│   ├── seafloor_classifier.py # Adaptive seafloor classification
│   ├── adaptive_processor.py  # Adaptive processing strategies
│   └── quality_metrics.py    # Quality assessment metrics
├── models/                   # Machine learning models
│   ├── __init__.py
│   ├── ensemble.py          # Ensemble architecture
│   └── base_models.py       # Individual model variants
├── processing/              # Data processing pipeline
│   ├── __init__.py
│   ├── data_processor.py    # Data loading and preprocessing
│   ├── pipeline.py          # Main processing pipeline
│   └── memory_utils.py      # Memory management
├── review/                  # Expert review system
│   ├── __init__.py
│   ├── expert_system.py     # Review workflow management
│   └── database.py          # Review database operations
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── logging_utils.py     # Enhanced logging
│   ├── visualization.py     # Plotting and visualization
│   └── cli.py              # Command-line interface
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── examples/                # Usage examples
└── scripts/                 # Setup and utility scripts
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/bathymetric-cae.git
   cd bathymetric-cae
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU support:**
   ```bash
   pip install tensorflow[and-cuda]>=2.13.0
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

### Basic Usage

1. **Default processing:**
   ```bash
   python main.py --input /path/to/input --output /path/to/output
   ```

2. **Enable all enhanced features:**
   ```bash
   python main.py \
     --input /path/to/input \
     --output /path/to/output \
     --enable-adaptive \
     --enable-expert-review \
     --enable-constitutional
   ```

3. **Custom configuration:**
   ```bash
   python main.py --config custom_config.json
   ```

### Configuration

Create a custom configuration file:

```json
{
  "ensemble_size": 3,
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "enable_constitutional_constraints": true,
  "quality_threshold": 0.7,
  "ssim_weight": 0.3,
  "roughness_weight": 0.2,
  "feature_preservation_weight": 0.3,
  "consistency_weight": 0.2
}
```

## 🧩 Modular Architecture

### Core Modules

**`core/`** - Domain-specific logic
- `enums.py`: Seafloor types, quality levels, processing strategies
- `constraints.py`: Physical plausibility rules and corrections
- `seafloor_classifier.py`: Automatic seafloor type classification
- `adaptive_processor.py`: Adaptive parameter selection
- `quality_metrics.py`: Comprehensive quality assessment

**`models/`** - Machine learning components
- `ensemble.py`: Multi-model ensemble architecture
- `base_models.py`: Individual model variants (lightweight, standard, robust)

**`processing/`** - Data processing pipeline
- `pipeline.py`: Main orchestration logic
- `data_processor.py`: File I/O and preprocessing
- `memory_utils.py`: Memory management and optimization

### Key Features by Module

#### Adaptive Processing (`core/adaptive_processor.py`)
```python
from bathymetric_cae.core import AdaptiveProcessor

processor = AdaptiveProcessor()
params = processor.get_processing_parameters(depth_data)
# Automatically adjusts based on seafloor type
```

#### Quality Assessment (`core/quality_metrics.py`)
```python
from bathymetric_cae.core import BathymetricQualityMetrics

metrics = BathymetricQualityMetrics()
quality_report = metrics.generate_quality_report(original, processed)
```

#### Expert Review (`review/expert_system.py`)
```python
from bathymetric_cae.review import ExpertReviewSystem

reviewer = ExpertReviewSystem()
reviewer.flag_for_review(filename, region, "low_quality", 0.8)
```

## 📊 Seafloor Types and Processing

The system automatically classifies seafloor environments and adapts processing:

| Seafloor Type | Depth Range | Processing Strategy |
|---------------|-------------|-------------------|
| Shallow Coastal | 0-200m | High feature preservation |
| Continental Shelf | 200-2000m | Balanced approach |
| Deep Ocean | 2000-6000m | Moderate smoothing |
| Seamount | Variable | Maximum feature preservation |
| Abyssal Plain | 6000-11000m | Aggressive smoothing |

## 🔬 Quality Metrics

### IHO Standards Compliance
- **Special Order**: ±0.25m + 0.0075×depth
- **Order 1a/1b**: ±0.5m + 0.013×depth  
- **Order 2**: ±1.0m + 0.023×depth

### Comprehensive Assessment
- **SSIM**: Structural similarity preservation
- **Feature Preservation**: Bathymetric feature retention
- **Consistency**: Local depth measurement coherence
- **Roughness**: Seafloor texture metrics
- **Composite Quality**: Weighted combination score

## 🛠️ Advanced Usage

### Ensemble Configuration
```bash
python main.py \
  --ensemble-size 5 \
  --quality-threshold 0.8 \
  --enable-uncertainty
```

### Custom Quality Weights
```bash
python main.py \
  --ssim-weight 0.4 \
  --feature-weight 0.4 \
  --consistency-weight 0.2
```

### Expert Review Workflow
```bash
# Process with automatic flagging
python main.py --enable-expert-review --quality-threshold 0.7

# Generate review report
python -m bathymetric_cae.review.generate_report
```

## 📈 Performance Optimization

### GPU Configuration
```python
# Automatic GPU memory growth
optimize_gpu_memory()

# Mixed precision training
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### Memory Management
```python
# Monitor memory usage
with memory_monitor("processing"):
    process_large_dataset()
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=bathymetric_cae
```

## 📚 Documentation

- **API Reference**: `docs/api_reference.md`
- **User Guide**: `docs/usage.md`
- **Installation**: `docs/installation.md`
- **Troubleshooting**: `docs/troubleshooting.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black bathymetric_cae/
flake8 bathymetric_cae/

# Type checking
mypy bathymetric_cae/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- GDAL contributors for geospatial data support
- IHO for hydrographic standards
- Scientific community for bathymetric processing research

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/username/bathymetric-cae/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/bathymetric-cae/discussions)
- **Documentation**: [Read the Docs](https://bathymetric-cae.readthedocs.io)

## 📊 Changelog

### Version 2.0.0
- Complete modular refactor
- Ensemble learning implementation
- Adaptive processing based on seafloor classification
- Constitutional AI constraints
- Expert review system
- Enhanced quality metrics
- Comprehensive logging and monitoring

### Version 1.0.0
- Initial monolithic implementation
- Basic CAE processing
- Simple quality metrics

---

**Enhanced Bathymetric CAE v2.0** - Bringing advanced AI to hydrographic data processing 🌊