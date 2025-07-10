# docs/installation.md

# Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8 GB RAM
- 10 GB free disk space

### Recommended Requirements
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM
- 50 GB free disk space

### Operating Systems
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

## Installation Methods

### Method 1: pip Installation (Recommended)

```bash
# Create virtual environment
python -m venv bathymetric_env
source bathymetric_env/bin/activate  # On Windows: bathymetric_env\Scripts\activate

# Install from PyPI (when published)
pip install bathymetric-cae

# Or install from source
pip install git+https://github.com/username/bathymetric-cae.git
```

### Method 2: Development Installation

```bash
# Clone repository
git clone https://github.com/username/bathymetric-cae.git
cd bathymetric-cae

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Method 3: Conda Installation

```bash
# Create conda environment
conda create -n bathymetric python=3.10
conda activate bathymetric

# Install dependencies
conda install tensorflow gdal matplotlib scikit-image -c conda-forge

# Install package
pip install -e .
```

## Dependency Installation

### Core Dependencies

```bash
# Required packages
pip install tensorflow>=2.13.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install scikit-image>=0.19.0
pip install scikit-learn>=1.1.0
pip install scipy>=1.9.0
pip install opencv-python>=4.6.0
pip install joblib>=1.1.0
pip install psutil>=5.8.0
```

### Geospatial Dependencies

#### GDAL Installation

**Windows:**
```bash
# Using conda (recommended)
conda install gdal -c conda-forge

# Or using OSGeo4W
# Download from https://trac.osgeo.org/osgeo4w/
```

**macOS:**
```bash
# Using Homebrew
brew install gdal

# Using conda
conda install gdal -c conda-forge
```

**Linux (Ubuntu/Debian):**
```bash
# System packages
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev

# Python bindings
pip install gdal
```

**Linux (CentOS/RHEL):**
```bash
# System packages
sudo yum install gdal gdal-devel

# Python bindings
pip install gdal
```

### GPU Support

#### NVIDIA GPU Setup

```bash
# Install CUDA toolkit (version 11.8 recommended)
# Download from https://developer.nvidia.com/cuda-toolkit

# Install cuDNN
# Download from https://developer.nvidia.com/cudnn

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]>=2.13.0
```

#### Verify GPU Installation

```python
import tensorflow as tf
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("TensorFlow version:", tf.__version__)
```

### Optional Dependencies

```bash
# Hydrographic tools
pip install pyproj>=3.3.0
pip install rasterio>=1.3.0
pip install fiona>=1.8.0
pip install geopandas>=0.11.0

# Development tools
pip install pytest>=7.0.0
pip install pytest-cov>=3.0.0
pip install black>=22.0.0
pip install flake8>=4.0.0
pip install mypy>=0.910

# Documentation
pip install sphinx>=4.0.0
pip install sphinx-rtd-theme>=1.0.0
pip install myst-parser>=0.17.0
```

## Verification

### Test Installation

```bash
# Test basic import
python -c "import bathymetric_cae; print('Installation successful!')"

# Test command line interface
bathymetric-cae --help

# Run test suite
pytest tests/
```

### Sample Processing Test

```bash
# Create test configuration
python -c "
from bathymetric_cae import Config
config = Config()
config.save('test_config.json')
print('Test configuration created')
"

# Test with sample data (if available)
python main.py --config test_config.json --help
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'gdal'

**Solution:**
```bash
# On Windows with conda
conda install gdal -c conda-forge

# On Linux
sudo apt-get install python3-gdal
```

#### GPU not detected

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall TensorFlow with GPU support
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

#### Memory issues during processing

**Solution:**
```bash
# Reduce batch size and grid size in configuration
python -c "
from bathymetric_cae import Config
config = Config()
config.batch_size = 4
config.grid_size = 256
config.save('memory_optimized_config.json')
"
```

#### Permission denied errors

**Solution:**
```bash
# Create directories with proper permissions
mkdir -p logs plots expert_reviews
chmod 755 logs plots expert_reviews
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
python -c "
from bathymetric_cae.utils import log_system_info
log_system_info()
"
```

#### GPU Optimization
```bash
# Set GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Limit GPU memory
export TF_GPU_MEMORY_LIMIT=4096
```

### Getting Help

- **Documentation**: [Link to full documentation]
- **Issues**: [GitHub Issues](https://github.com/username/bathymetric-cae/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/bathymetric-cae/discussions)
- **Email**: contact@example.com

---

# docs/usage.md

# Usage Guide

## Quick Start

### Basic Processing

```bash
# Process bathymetric files with default settings
python main.py --input /path/to/input --output /path/to/output

# Enable enhanced features
python main.py \
  --input /path/to/input \
  --output /path/to/output \
  --enable-adaptive \
  --enable-expert-review \
  --enable-constitutional
```

### Using Configuration Files

```bash
# Create default configuration
python -c "
from bathymetric_cae import Config
config = Config()
config.save('my_config.json')
"

# Edit configuration as needed, then run
python main.py --config my_config.json
```

## Configuration

### Basic Configuration

```json
{
  "ensemble_size": 3,
  "grid_size": 512,
  "epochs": 100,
  "batch_size": 8,
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "enable_constitutional_constraints": true,
  "quality_threshold": 0.7
}
```

### Advanced Configuration

```json
{
  "ensemble_size": 5,
  "grid_size": 1024,
  "epochs": 200,
  "batch_size": 4,
  "learning_rate": 0.0005,