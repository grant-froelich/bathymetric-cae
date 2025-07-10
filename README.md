# Bathymetric CAE - Advanced Convolutional Autoencoder for Bathymetric Data Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)

A comprehensive pipeline for processing bathymetric data using advanced Convolutional Autoencoders with modern machine learning techniques.

## Features

- **Advanced CNN Architecture**: Residual blocks, attention mechanisms, and skip connections
- **Mixed Precision Training**: Optimized for modern GPUs with automatic memory management
- **Comprehensive Data Support**: BAG, GeoTIFF, ASCII Grid, and XYZ formats
- **Uncertainty Handling**: Native support for uncertainty data in BAG files
- **Enhanced Visualization**: Training history, comparison plots, and statistical analysis
- **Flexible Configuration**: JSON-based configuration with command-line overrides
- **Memory Optimization**: Intelligent memory monitoring and cleanup
- **Parallel Processing**: Multi-threaded file processing capabilities
- **Geospatial Metadata**: Preserves projection and coordinate system information

## Installation

### Prerequisites

- Python 3.8 or higher
- GDAL 3.4 or higher
- CUDA-compatible GPU (optional but recommended)

### Install from PyPI

```bash
pip install bathymetric-cae
```

### Install from Source

```bash
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd bathymetric-cae
pip install -e .
```

### GPU Support

For GPU acceleration:

```bash
pip install bathymetric-cae[gpu]
```

### Development Installation

```bash
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd bathymetric-cae
pip install -e .[dev]
```

## Quick Start

### Command Line Usage

```bash
# Basic processing
bathymetric-cae --input /path/to/input --output /path/to/output

# Custom training parameters
bathymetric-cae --input ./data --output ./results --epochs 200 --batch-size 16

# Process single file
bathymetric-cae --single-file input.bag --model trained_model.h5 --show-plots

# Validate installation
bathymetric-cae --validate-requirements
```

### Python API Usage

```python
from bathymetric_cae import BathymetricCAEPipeline, Config

# Create configuration
config = Config(
    input_folder="./input_data",
    output_folder="./output_data",
    epochs=100,
    batch_size=8,
    grid_size=512
)

# Create and run pipeline
pipeline = BathymetricCAEPipeline(config)
results = pipeline.run(
    input_folder=config.input_folder,
    output_folder=config.output_folder,
    model_path="my_model.h5"
)

print(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
```

### Quick Start Function

```python
from bathymetric_cae import quick_start

results = quick_start(
    input_folder="./input_data",
    output_folder="./output_data",
    epochs=50,
    batch_size=4
)
```

## Configuration

### Configuration File Example

```json
{
  "input_folder": "/path/to/input",
  "output_folder": "/path/to/output",
  "model_path": "bathymetric_model.h5",
  "epochs": 100,
  "batch_size": 8,
  "learning_rate": 0.001,
  "grid_size": 512,
  "base_filters": 32,
  "depth": 4,
  "dropout_rate": 0.2,
  "early_stopping_patience": 15,
  "gpu_memory_growth": true,
  "use_mixed_precision": true
}
```

### Loading Configuration

```bash
bathymetric-cae --config my_config.json
```

```python
from bathymetric_cae import Config

config = Config.load("my_config.json")
```

## Supported Data Formats

| Format | Extension | Description | Uncertainty Support |
|--------|-----------|-------------|-------------------|
| BAG | .bag | Bathymetric Attributed Grid | ✅ |
| GeoTIFF | .tif, .tiff | Tagged Image File Format | ❌ |
| ASCII Grid | .asc | ESRI ASCII Grid | ❌ |
| XYZ | .xyz | Point cloud format | ❌ |

## Model Architecture

The Advanced CAE features:

- **Encoder-Decoder Architecture**: Deep convolutional layers with progressive downsampling
- **Residual Connections**: Skip connections for better gradient flow
- **Attention Mechanisms**: Spatial attention for feature enhancement
- **Batch Normalization**: Stable training with normalized activations
- **Mixed Precision**: FP16/FP32 computation for memory efficiency

## Performance Optimization

### GPU Configuration

```python
from bathymetric_cae import optimize_gpu_performance

# Optimize GPU settings
results = optimize_gpu_performance()
print(f"Mixed precision enabled: {results['mixed_precision']}")
```

### Memory Management

```python
from bathymetric_cae import memory_monitor

with memory_monitor("Training phase"):
    # Your training code here
    pass
```

## Visualization

The package provides comprehensive visualization capabilities:

- Training history plots (loss, metrics, learning rate)
- Data comparison plots (original vs. cleaned)
- Difference maps and statistical analysis
- Processing summary reports
- Uncertainty visualization for BAG files

## Examples

### Batch Processing

```python
from bathymetric_cae import BathymetricCAEPipeline, Config, setup_logging

# Setup logging
setup_logging(log_level='INFO')

# Configure pipeline
config = Config(
    input_folder="./bathymetric_data",
    output_folder="./cleaned_data",
    epochs=150,
    batch_size=16,
    grid_size=1024,
    base_filters=64
)

# Run pipeline
pipeline = BathymetricCAEPipeline(config)
results = pipeline.run(
    input_folder=config.input_folder,
    output_folder=config.output_folder,
    model_path="large_model.h5"
)
```

### Single File Processing

```python
from bathymetric_cae import BathymetricCAEPipeline, Config

config = Config()
pipeline = BathymetricCAEPipeline(config)

results = pipeline.process_single_file_interactive(
    file_path="sample.bag",
    model_path="trained_model.h5",
    show_plots=True
)
```

### Custom Model Configuration

```python
from bathymetric_cae import AdvancedCAE, Config

config = Config(
    grid_size=256,
    base_filters=16,
    depth=3,
    dropout_rate=0.1
)

model_builder = AdvancedCAE(config)
model = model_builder.create_model(channels=2)  # Depth + uncertainty
```

## Command Line Reference

### Basic Commands

```bash
# Process with default settings
bathymetric-cae

# Custom input/output paths
bathymetric-cae --input /data/bathymetry --output /results/cleaned

# Load configuration file
bathymetric-cae --config config.json

# Save current configuration
bathymetric-cae --save-config my_config.json
```

### Training Parameters

```bash
# Custom training settings
bathymetric-cae --epochs 200 --batch-size 32 --learning-rate 0.0005

# Model architecture
bathymetric-cae --grid-size 1024 --base-filters 64 --depth 5

# Force retraining
bathymetric-cae --force-retrain
```

### Processing Options

```bash
# Disable GPU
bathymetric-cae --no-gpu

# Verbose output
bathymetric-cae --verbose

# Silent mode
bathymetric-cae --silent

# Disable plots
bathymetric-cae --no-plots
```

## API Reference

### Core Classes

- `Config`: Configuration management
- `BathymetricCAEPipeline`: Main processing pipeline
- `BathymetricProcessor`: Data preprocessing
- `AdvancedCAE`: Model architecture
- `Visualizer`: Plotting and visualization

### Utility Functions

- `setup_logging()`: Configure logging
- `memory_monitor()`: Monitor memory usage
- `optimize_gpu_performance()`: GPU optimization
- `get_valid_files()`: File discovery
- `validate_pipeline_requirements()`: System validation

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd bathymetric-cae
pip install -e .[dev]
pre-commit install
```

### Running Tests

```bash
pytest tests/
pytest --cov=bathymetric_cae tests/
```

## Troubleshooting

### Common Issues

1. **Out of memory errors**:
   - Reduce `batch_size` and `grid_size`
   - Enable GPU memory growth
   - Use CPU processing for large files

2. **GDAL errors**:
   - Ensure GDAL is properly installed
   - Check file permissions and paths
   - Verify file format support

3. **Training convergence issues**:
   - Adjust learning rate
   - Increase patience for callbacks
   - Check data quality

### Getting Help

- [Documentation](https://github.com/bathymetric-cae/bathymetric-cae/docs)
- [Issue Tracker](https://github.com/noaa-ocs-hydrography/bathymetric-cae)
- [Discussions](https://github.com/noaa-ocs-hydrography/bathymetric-cae)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{bathymetric_cae,
  title={Bathymetric CAE: Advanced Convolutional Autoencoder for Bathymetric Data Processing},
  author={Bathymetric CAE Team},
  url={https://github.com/noaa-ocs-hydrography/bathymetric-cae},
  version={1.0.0},
  year={2025}
}
```

## Acknowledgments

- TensorFlow team for the deep learning framework
- GDAL community for geospatial data processing capabilities
- Scientific Python community for the excellent ecosystem
