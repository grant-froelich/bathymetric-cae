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
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
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
git clone https://github.com/your-org/enhanced-bathymetric-cae.git
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

### Getting Help

If installation fails:

1. **Check system requirements** - Ensure you have adequate RAM and disk space
2. **Use conda method** - Strongly recommended over pip for GDAL
3. **Create fresh environment** - `conda env remove -n bathymetric-cae && conda env create -f environment.yml`
4. **Check logs** - Look for specific error messages
5. **Ask community** - [GitHub Issues](https://github.com/your-org/enhanced-bathymetric-cae/issues)

### Performance Optimization

After installation, optimize performance:

```bash
# Set environment variables (add to ~/.bashrc or ~/.zshrc)
export GDAL_CACHEMAX=1024
export GDAL_NUM_THREADS=ALL_CPUS
export TF_CPP_MIN_LOG_LEVEL=2

# For large datasets
export OMP_NUM_THREADS=4
```

---

**Next Steps**: After successful installation, continue with the [Quick Start Guide](docs/getting-started/quick-start.md)