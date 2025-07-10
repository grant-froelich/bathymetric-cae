# Enhanced Bathymetric CAE Processing v2.0

Advanced bathymetric grid processing using ensemble Convolutional Autoencoders with domain-specific enhancements, constitutional AI constraints, and expert review systems.

## 🌊 Features

### Core Capabilities
- **🤖 Ensemble Model Architecture**: Multiple CAE variants for improved robustness
- **🧠 Adaptive Processing**: Seafloor type-specific processing strategies
- **⚖️ Constitutional AI Constraints**: Ensures bathymetric data integrity
- **👥 Expert Review System**: Human-in-the-loop validation workflow
- **📊 Comprehensive Quality Metrics**: IHO S-44 compliant assessment
- **🎯 Multi-objective Optimization**: Customizable quality metric weights

### Advanced Model Architectures
- **AdvancedCAE**: Full-featured model with residual and skip connections
- **UncertaintyCAE**: Dual-output model estimating depth and uncertainty
- **LightweightCAE**: Resource-efficient variant for constrained environments
- **Custom Loss Functions**: Domain-specific edge preservation and consistency losses

### Seafloor Intelligence
- **Automatic Classification**: Shallow coastal, continental shelf, deep ocean, abyssal plain, seamount
- **Adaptive Parameters**: Processing strategies tailored to seafloor characteristics
- **Feature Preservation**: Maintains critical bathymetric features during processing

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

### Development Install
```bash
pip install -e .
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

## 📁 Module Architecture

```
enhanced_bathymetric_cae/
├── 📋 README.md                    # This file
├── 📦 requirements.txt             # Dependencies
├── ⚙️ setup.py                     # Package installation
├── 🔧 config/                      # Configuration management
│   ├── __init__.py
│   └── config.py                   # Config dataclass with validation
├── 🧠 core/                        # Domain-specific functionality
│   ├── __init__.py
│   ├── enums.py                    # SeafloorType and other enums
│   ├── constraints.py              # Constitutional AI rules
│   ├── quality_metrics.py          # IHO-compliant quality assessment
│   └── adaptive_processor.py       # Seafloor classification & adaptation
├── 🤖 models/                      # AI model architectures
│   ├── __init__.py
│   ├── ensemble.py                 # Ensemble management & prediction
│   └── architectures.py            # CAE variants & custom losses
├── ⚡ processing/                   # Data processing pipeline
│   ├── __init__.py
│   ├── data_processor.py           # File I/O & preprocessing
│   └── pipeline.py                 # Main processing orchestration
├── 👥 review/                      # Expert review system
│   ├── __init__.py
│   └── expert_system.py            # Human-in-the-loop validation
├── 🛠️ utils/                       # Utilities
│   ├── __init__.py
│   ├── logging_utils.py            # Colored logging & setup
│   ├── memory_utils.py             # Memory monitoring & GPU optimization
│   └── visualization.py            # Plotting & report generation
├── 💻 cli/                         # Command-line interface
│   ├── __init__.py
│   └── interface.py                # Argument parsing & config updates
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
```

### Running Tests
```bash
# Basic functionality test
python main.py --input test_data/ --output test_output/ --epochs 5

# Debug mode
python main.py --log-level DEBUG --input test_data/ --epochs 10

# Configuration validation
python main.py --save-config test_config.json
python main.py --config test_config.json --epochs 5
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
```

#### New Model Architecture
```python
# In models/architectures.py
class NewCAE(AdvancedCAE):
    """New CAE variant."""
    
    def create_model(self, input_shape: tuple, variant_config: Dict = None):
        # Implementation here
        return model
```

### Code Style Guidelines
- **PEP 8 compliance**: Use `black` for formatting
- **Type hints**: Add type annotations for all functions
- **Docstrings**: Use Google-style docstrings
- **Error handling**: Comprehensive try/except blocks
- **Logging**: Use module-level loggers

### Testing Strategy
```bash
# Unit tests (when implemented)
pytest tests/

# Integration tests
python main.py --input test_data/ --output test_output/ --epochs 10

# Performance tests
python main.py --input large_dataset/ --log-level INFO --epochs 50
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

### Getting Help
1. **Check logs**: `logs/bathymetric_processing.log`
2. **Review configuration**: Ensure all paths and parameters are valid
3. **Test with small dataset**: Use minimal configuration first
4. **Check system resources**: Ensure adequate RAM and disk space

## 🤝 Contributing

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-new-metric`
3. **Follow** the code style guidelines
4. **Add** tests for new functionality
5. **Update** documentation
6. **Submit** a pull request

### Development Workflow
```bash
# Setup development environment
git clone your-fork-url
cd enhanced-bathymetric-cae
pip install -e ".[dev]"

# Create feature branch
git checkout -b feature-description

# Make changes and test
python main.py --input test_data/ --epochs 10

# Commit and push
git add .
git commit -m "Add new feature: description"
git push origin feature-description
```

### Code Review Checklist
- [ ] Code follows PEP 8 guidelines
- [ ] All functions have type hints and docstrings
- [ ] Error handling is comprehensive
- [ ] Tests pass with small dataset
- [ ] Documentation is updated
- [ ] No breaking changes to existing API

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this software in your research, please cite:

```bibtex
@software{enhanced_bathymetric_cae_2024,
  title={Enhanced Bathymetric CAE Processing: Advanced Deep Learning for Seafloor Data},
  author={Your Name and Contributors},
  year={2024},
  version={2.0.0},
  url={https://github.com/your-repo/enhanced-bathymetric-cae},
  doi={10.xxxx/xxxxx}
}
```

## 🌐 Related Projects

- **IHO S-44 Standards**: [International Hydrographic Organization](https://iho.int/)
- **GDAL**: [Geospatial Data Abstraction Library](https://gdal.org/)
- **TensorFlow**: [Machine Learning Platform](https://tensorflow.org/)
- **BAG Format**: [Bathymetric Attributed Grid](https://www.opennavsurf.org/bag)

## 📞 Support

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-repo/enhanced-bathymetric-cae/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/your-repo/enhanced-bathymetric-cae/discussions)
- **📧 Email**: your.email@institution.org
- **📚 Documentation**: [Wiki](https://github.com/your-repo/enhanced-bathymetric-cae/wiki)

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
- 🔧 **Improved**: Memory management and GPU optimization
- 🔧 **Improved**: Error handling and logging
- 🔧 **Improved**: Configuration management
- 📚 **Updated**: Complete documentation rewrite

### v1.0.0 (Legacy)
- 🎯 Initial monolithic implementation
- 🤖 Basic CAE architecture
- 📁 Simple file processing
- 📊 Basic quality metrics

---

<div align="center">

**🌊 Advancing Bathymetric Data Processing with AI 🤖**

*Built with ❤️ for the oceanographic and hydrographic communities*

</div>