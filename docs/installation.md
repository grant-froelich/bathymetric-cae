# Installation Guide

This guide provides comprehensive instructions for installing the Bathymetric CAE package across different platforms and environments.

## Quick Installation

### From PyPI (Recommended)

The simplest way to install Bathymetric CAE is using pip:

```bash
pip install bathymetric-cae
```

For GPU support:

```bash
pip install bathymetric-cae[gpu]
```

For development:

```bash
pip install bathymetric-cae[dev]
```

### From Source

To install the latest development version:

```bash
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd bathymetric-cae
pip install -e .
```

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **Memory**: 4 GB RAM
- **Storage**: 2 GB available disk space
- **GDAL**: 3.4 or higher

### Recommended Requirements

- **Python**: 3.9 or higher
- **Memory**: 16 GB RAM
- **GPU**: NVIDIA GPU with 6+ GB VRAM
- **Storage**: 10 GB available disk space
- **CUDA**: 11.2 or higher (for GPU support)

### Supported Platforms

- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+
- **macOS**: 10.15 (Catalina) or higher
- **Windows**: Windows 10/11

## Detailed Installation Instructions

### Linux Installation

#### Ubuntu/Debian

1. **Update system packages:**
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. **Install system dependencies:**
   ```bash
   sudo apt install python3 python3-pip python3-dev
   sudo apt install gdal-bin libgdal-dev
   sudo apt install build-essential
   ```

3. **Install Python packages:**
   ```bash
   pip3 install bathymetric-cae
   ```

#### CentOS/RHEL

1. **Install EPEL repository:**
   ```bash
   sudo yum install epel-release
   ```

2. **Install system dependencies:**
   ```bash
   sudo yum install python3 python3-pip python3-devel
   sudo yum install gdal gdal-devel
   sudo yum install gcc gcc-c++
   ```

3. **Install Python packages:**
   ```bash
   pip3 install bathymetric-cae
   ```

#### Arch Linux

1. **Install system dependencies:**
   ```bash
   sudo pacman -S python python-pip gdal base-devel
   ```

2. **Install Python packages:**
   ```bash
   pip install bathymetric-cae
   ```

### macOS Installation

#### Using Homebrew (Recommended)

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install system dependencies:**
   ```bash
   brew install python gdal
   ```

3. **Install Python packages:**
   ```bash
   pip3 install bathymetric-cae
   ```

#### Using MacPorts

1. **Install MacPorts dependencies:**
   ```bash
   sudo port install python39 py39-pip gdal
   ```

2. **Install Python packages:**
   ```bash
   pip3 install bathymetric-cae
   ```

### Windows Installation

#### Using Conda (Recommended)

1. **Install Anaconda or Miniconda:**
   - Download from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation wizard

2. **Create conda environment:**
   ```cmd
   conda create -n bathymetric_cae python=3.9
   conda activate bathymetric_cae
   ```

3. **Install dependencies:**
   ```cmd
   conda install -c conda-forge gdal tensorflow
   pip install bathymetric-cae
   ```

#### Using pip with pre-built binaries

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked

2. **Install GDAL:**
   - Download GDAL binaries from [GISInternals](https://www.gisinternals.com/)
   - Or use conda: `conda install -c conda-forge gdal`

3. **Install Python packages:**
   ```cmd
   pip install bathymetric-cae
   ```

## GPU Support

### NVIDIA GPU Setup

1. **Install NVIDIA drivers:**
   - Download from [NVIDIA website](https://www.nvidia.com/drivers/)
   - Or use system package manager

2. **Install CUDA Toolkit:**
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

3. **Install cuDNN:**
   - Download from [NVIDIA Developer](https://developer.nvidia.com/cudnn)
   - Follow installation instructions for your platform

4. **Install TensorFlow with GPU support:**
   ```bash
   pip install tensorflow[and-cuda]
   ```

### Verify GPU Installation

```python
import tensorflow as tf
print("GPU Available: ", len(tf.config.list_physical_devices('GPU')) > 0)
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))
```

## Virtual Environment Setup

### Using venv (Built-in)

```bash
# Create virtual environment
python3 -m venv bathymetric_cae_env

# Activate (Linux/macOS)
source bathymetric_cae_env/bin/activate

# Activate (Windows)
bathymetric_cae_env\Scripts\activate

# Install package
pip install bathymetric-cae

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create environment
conda create -n bathymetric_cae python=3.9

# Activate
conda activate bathymetric_cae

# Install dependencies
conda install -c conda-forge gdal tensorflow

# Install package
pip install bathymetric-cae

# Deactivate when done
conda deactivate
```

### Using virtualenv

```bash
# Install virtualenv
pip install virtualenv

# Create environment
virtualenv bathymetric_cae_env

# Activate (Linux/macOS)
source bathymetric_cae_env/bin/activate

# Activate (Windows)
bathymetric_cae_env\Scripts\activate

# Install package
pip install bathymetric-cae
```

## Docker Installation

### Using Docker

1. **Create Dockerfile:**
   ```dockerfile
   FROM tensorflow/tensorflow:2.13.0-gpu

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gdal-bin libgdal-dev \
       && rm -rf /var/lib/apt/lists/*

   # Install Python packages
   RUN pip install bathymetric-cae

   # Set working directory
   WORKDIR /workspace

   # Default command
   CMD ["python", "-c", "import bathymetric_cae; print('Bathymetric CAE ready!')"]
   ```

2. **Build and run:**
   ```bash
   docker build -t bathymetric-cae .
   docker run --gpus all -it bathymetric-cae
   ```

### Using Docker Compose

```yaml
version: '3.8'
services:
  bathymetric-cae:
    build: .
    volumes:
      - ./data:/workspace/data
      - ./output:/workspace/output
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    runtime: nvidia
```

## Development Installation

### For Contributors

1. **Clone repository:**
   ```bash
   git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
   cd bathymetric-cae
   ```

2. **Create development environment:**
   ```bash
   python -m venv dev_env
   source dev_env/bin/activate  # Linux/macOS
   # or
   dev_env\Scripts\activate     # Windows
   ```

3. **Install in development mode:**
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Run tests:**
   ```bash
   pytest tests/
   ```

## Troubleshooting

### Common Issues

#### GDAL Installation Problems

**Problem**: `ImportError: No module named 'osgeo'`

**Solutions:**
```bash
# Option 1: Use conda
conda install -c conda-forge gdal

# Option 2: Install system GDAL first
sudo apt install gdal-bin libgdal-dev  # Ubuntu
brew install gdal                       # macOS

# Option 3: Use pip with system GDAL
pip install gdal==$(gdal-config --version)
```

#### TensorFlow GPU Issues

**Problem**: GPU not detected by TensorFlow

**Solutions:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Reinstall TensorFlow: `pip install tensorflow[and-cuda]`
4. Set environment variables:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
   ```

#### Memory Issues

**Problem**: Out of memory during training

**Solutions:**
1. Reduce batch size in configuration
2. Reduce grid size
3. Enable GPU memory growth
4. Use CPU processing for large files

#### Permission Issues

**Problem**: Permission denied errors

**Solutions:**
```bash
# Fix pip permissions
pip install --user bathymetric-cae

# Or use virtual environment
python -m venv myenv
source myenv/bin/activate
pip install bathymetric-cae
```

### Getting Help

If you encounter issues not covered here:

1. **Check the troubleshooting section** in the documentation
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Operating system and version
   - Python version
   - Complete error message
   - Steps to reproduce

### Verification

After installation, verify everything works:

```python
# Test basic import
import bathymetric_cae
print(f"Bathymetric CAE version: {bathymetric_cae.__version__}")

# Test dependencies
from bathymetric_cae import validate_installation
validation_results = validate_installation()
print(f"All requirements met: {validation_results['all_requirements_met']}")

# Test GPU (if available)
from bathymetric_cae import check_gpu_availability
gpu_info = check_gpu_availability()
print(f"GPU available: {gpu_info['gpu_available']}")
```

## Next Steps

Once installed successfully:

1. **Read the [Quick Start Guide](quickstart.md)** for basic usage
2. **Explore the [Examples](examples.md)** for hands-on learning
3. **Review the [Configuration Guide](configuration.md)** for customization
4. **Check the [API Reference](api/index.md)** for detailed documentation

## Updating

To update to the latest version:

```bash
pip install --upgrade bathymetric-cae
```

To update from source:

```bash
cd bathymetric-cae
git pull origin main
pip install -e .
```
