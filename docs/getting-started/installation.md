# Installation Guide

This guide provides comprehensive installation instructions for the Enhanced Bathymetric CAE Processing system across different platforms and environments.

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 10 GB free space
- **CPU**: Multi-core processor recommended

### Recommended Requirements
- **RAM**: 32 GB for large datasets
- **GPU**: NVIDIA GPU with CUDA support (for accelerated processing)
- **Storage**: SSD with 50+ GB free space
- **Network**: High-speed internet for downloading models and data

### Dependencies
- **GDAL**: Geospatial Data Abstraction Library (3.4+)
- **TensorFlow**: 2.13+ (CPU or GPU version)
- **NumPy**: 1.21+
- **SciPy**: 1.7+
- **OpenCV**: 4.5+

## 🚀 Quick Installation

### Using pip (Recommended)

```bash
# Install from PyPI
pip install enhanced-bathymetric-cae

# Verify installation
bathymetric-cae --version
```

### Using conda

```bash
# Install from conda-forge
conda install -c conda-forge enhanced-bathymetric-cae

# Verify installation
bathymetric-cae --version
```

## 🔧 Development Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae
```

### 2. Create Virtual Environment

=== "Using venv"

    ```bash
    python -m venv venv
    
    # Activate on Linux/macOS
    source venv/bin/activate
    
    # Activate on Windows
    venv\Scripts\activate
    ```

=== "Using conda"

    ```bash
    conda create -n bathymetric-cae python=3.9
    conda activate bathymetric-cae
    ```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r tests/requirements-test.txt

# Install package in development mode
pip install -e .
```

## 🖥️ Platform-Specific Installation

### Windows Installation

#### Prerequisites

1. **Install Python**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked
   - Verify: `python --version`

2. **Install Visual Studio Build Tools**:
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Required for compiling some dependencies

3. **Install GDAL**:
   ```powershell
   # Using conda (recommended)
   conda install -c conda-forge gdal
   
   # Or download from OSGeo4W
   # https://trac.osgeo.org/osgeo4w/
   ```

#### Installation Steps

```powershell
# Clone repository
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
bathymetric-cae --version
```

### macOS Installation

#### Prerequisites

1. **Install Homebrew**:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python and GDAL**:
   ```bash
   brew install python@3.9 gdal
   ```

3. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

#### Installation Steps

```bash
# Clone repository
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
bathymetric-cae --version
```

### Linux Installation (Ubuntu/Debian)

#### Prerequisites

```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install -y \
    python3 python3-pip python3-venv \
    gdal-bin libgdal-dev \
    build-essential \
    git

# Install Python GDAL bindings
pip3 install gdal==$(gdal-config --version)
```

#### Installation Steps

```bash
# Clone repository
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
bathymetric-cae --version
```

### CentOS/RHEL Installation

#### Prerequisites

```bash
# Install EPEL repository
sudo yum install -y epel-release

# Install system dependencies
sudo yum install -y \
    python3 python3-pip python3-devel \
    gdal gdal-devel \
    gcc gcc-c++ \
    git

# Install Python GDAL bindings
pip3 install gdal==$(gdal-config --version)
```

## 🐳 Docker Installation

### Using Pre-built Image

```bash
# Pull the latest image
docker pull bathymetriccae/enhanced-bathymetric-cae:latest

# Run with sample data
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    bathymetriccae/enhanced-bathymetric-cae:latest \
    bathymetric-cae --input /app/data --output /app/output
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae

# Build Docker image
docker build -t enhanced-bathymetric-cae .

# Run container
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    enhanced-bathymetric-cae \
    bathymetric-cae --input /app/data --output /app/output
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  bathymetric-cae:
    image: bathymetriccae/enhanced-bathymetric-cae:latest
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
      - TF_CPP_MIN_LOG_LEVEL=2
    command: >
      bathymetric-cae 
      --input /app/data 
      --output /app/output 
      --enable-adaptive 
      --enable-expert-review
```

## ☁️ Cloud Installation

### AWS EC2

#### 1. Launch EC2 Instance

```bash
# Use Deep Learning AMI (Ubuntu 18.04)
# Instance type: p3.2xlarge or similar for GPU acceleration
```

#### 2. Install on EC2

```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone and install
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae

# Install with GPU support
pip install -r requirements.txt
pip install tensorflow[and-cuda]
pip install -e .
```

### Google Cloud Platform

#### 1. Create Compute Instance

```bash
# Create instance with GPU
gcloud compute instances create bathymetric-processing \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-k80,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

#### 2. Install on GCP

```bash
# SSH to instance
gcloud compute ssh bathymetric-processing --zone=us-central1-a

# Install CUDA drivers (if needed)
sudo /opt/deeplearning/install-driver.sh

# Clone and install
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae
pip install -r requirements.txt
pip install -e .
```

### Microsoft Azure

#### 1. Create VM with GPU

```bash
# Create resource group
az group create --name bathymetric-rg --location eastus

# Create VM with GPU
az vm create \
    --resource-group bathymetric-rg \
    --name bathymetric-vm \
    --image "microsoft-dsvm:ubuntu-1804:1804:latest" \
    --size Standard_NC6 \
    --admin-username azureuser \
    --generate-ssh-keys
```

#### 2. Install on Azure

```bash
# SSH to VM
ssh azureuser@your-vm-ip

# Clone and install
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
cd enhanced-bathymetric-cae
pip install -r requirements.txt
pip install -e .
```

## ⚙️ GPU Setup

### NVIDIA GPU Setup

#### 1. Install NVIDIA Drivers

=== "Ubuntu/Debian"

    ```bash
    # Add NVIDIA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    
    # Install CUDA toolkit
    sudo apt-get install -y cuda
    
    # Add to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

=== "Windows"

    1. Download CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
    2. Install following the wizard
    3. Verify: `nvcc --version`

=== "macOS"

    ```bash
    # Install via Homebrew
    brew install --cask nvidia-cuda
    
    # Add to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.zshrc
    source ~/.zshrc
    ```

#### 2. Install cuDNN

1. Download cuDNN from [NVIDIA](https://developer.nvidia.com/cudnn)
2. Extract and copy files:

```bash
# Linux
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

#### 3. Install TensorFlow GPU

```bash
pip install tensorflow[and-cuda]

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### AMD GPU Setup (ROCm)

```bash
# Install ROCm (Linux only)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms

# Install TensorFlow ROCm
pip install tensorflow-rocm
```

## 🧪 Testing Installation

### Basic Functionality Test

```bash
# Test basic import
python -c "
import enhanced_bathymetric_cae
print('✓ Package imported successfully')
"

# Test dependencies
python -c "
import tensorflow as tf
import numpy as np
from osgeo import gdal
print('✓ All dependencies available')
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
"
```

### Run Test Suite

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run basic tests
pytest tests/test_config.py -v

# Run full test suite (optional)
python tests/run_tests_advanced.py --quick
```

### Sample Processing Test

```bash
# Generate test data
python -c "
from tests.test_fixtures.sample_data_generator import TestDataGenerator
from pathlib import Path
TestDataGenerator.create_test_dataset(Path('test_data'), 2)
print('✓ Test data generated')
"

# Run sample processing
bathymetric-cae \
    --input test_data \
    --output test_output \
    --epochs 2 \
    --batch-size 1 \
    --ensemble-size 1
```

## 🔧 Configuration

### Initial Configuration

Create a configuration file:

```bash
# Generate default configuration
bathymetric-cae --save-config my_config.json

# Edit configuration
# See configuration guide for details
```

### Environment Variables

```bash
# Optional environment variables
export BATHYMETRIC_CAE_DATA_DIR=/path/to/data
export BATHYMETRIC_CAE_MODEL_DIR=/path/to/models
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

## 🚨 Troubleshooting

### Common Issues

#### GDAL Import Error

```bash
# Error: ImportError: No module named 'osgeo'
# Solution: Install GDAL properly

# Ubuntu/Debian
sudo apt install gdal-bin libgdal-dev
pip install gdal==$(gdal-config --version)

# Windows (using conda)
conda install -c conda-forge gdal

# macOS
brew install gdal
pip install gdal==$(gdal-config --version)
```

#### TensorFlow GPU Issues

```bash
# Error: Could not load dynamic library 'libcudart.so.11.0'
# Solution: Install correct CUDA version

# Check TensorFlow CUDA requirements
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"

# Install compatible CUDA version
# See TensorFlow GPU guide
```

#### Memory Issues

```bash
# Error: OOM when loading model
# Solution: Reduce memory usage

# Use CPU only
export CUDA_VISIBLE_DEVICES=""

# Limit TensorFlow GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Reduce batch size and ensemble size
bathymetric-cae --batch-size 1 --ensemble-size 1
```

#### Permission Issues (Linux/macOS)

```bash
# Error: Permission denied
# Solution: Fix permissions

# Make scripts executable
chmod +x test_automation.sh

# Install with user flag
pip install --user -e .

# Or use sudo (not recommended for development)
sudo pip install -e .
```

### Getting Help

If you encounter issues:

1. **Check logs**: Review `logs/bathymetric_processing.log`
2. **Run diagnostics**: `python tests/run_tests_advanced.py --quick`
3. **Check system requirements**: Verify all dependencies
4. **Search issues**: [GitHub Issues](https://github.com/your-org/enhanced-bathymetric-cae/issues)
5. **Ask for help**: [Community Forum](https://community.bathymetric-cae.org)

## 📚 Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quick-start.md)**
2. **Try the [Basic Usage Tutorial](basic-usage.md)**
3. **Configure your [Settings](configuration.md)**
4. **Explore [User Guide](../user-guide/index.md)**

## 🔄 Updating

### Update from PyPI

```bash
pip install --upgrade enhanced-bathymetric-cae
```

### Update Development Installation

```bash
cd enhanced-bathymetric-cae
git pull origin main
pip install -r requirements.txt
pip install -e .
```

### Migration Between Versions

See [Migration Guides](../changelog/migration-guides/) for version-specific update instructions.

---

<div align="center">

**Need Help?** 

Visit our [Support Page](../getting-started/troubleshooting.md) or join our [Community](https://community.bathymetric-cae.org)

</div>