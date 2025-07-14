"""
Setup script for Enhanced Bathymetric CAE Processing
Updated to handle conda/pip installation differences
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
def get_long_description():
    """Get long description from README.md if it exists."""
    this_directory = Path(__file__).parent
    readme_path = this_directory / "README.md"
    
    if readme_path.exists():
        try:
            return readme_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not read README.md: {e}")
            return ""
    else:
        return ""

# Check if we're in a conda environment
def is_conda_environment():
    """Check if we're running in a conda environment."""
    return (
        'CONDA_DEFAULT_ENV' in os.environ or 
        'CONDA_PREFIX' in os.environ or
        os.path.exists(os.path.join(sys.prefix, 'conda-meta')) or
        'conda' in sys.version.lower()
    )

# Read base requirements
def get_requirements():
    """Read requirements from requirements.txt if it exists."""
    requirements = []
    req_file = Path('requirements.txt')
    
    if req_file.exists():
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Skip GDAL-related packages if in conda environment
                        if is_conda_environment() and any(pkg in line.lower() for pkg in ['gdal', 'rasterio', 'opencv']):
                            continue
                        requirements.append(line)
        except Exception as e:
            print(f"Warning: Could not read requirements.txt: {e}")
    
    return requirements

# Get install requirements based on environment
def get_install_requires():
    """Get install requirements based on environment."""
    base_requirements = [
        "tensorflow>=2.13.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.19.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "psutil>=5.8.0",
    ]
    
    # Add pathlib2 only for Python < 3.4 (though we require 3.8+)
    if sys.version_info < (3, 4):
        base_requirements.append("pathlib2>=2.3.0")
    
    # Add geospatial packages only if not in conda environment
    if not is_conda_environment():
        print("Warning: Installing without conda may cause GDAL issues.")
        print("   Recommended: conda env create -f environment.yml")
        geospatial_requirements = [
            "opencv-python>=4.5.0",
            # Note: GDAL and rasterio are in extras_require['gdal']
        ]
        base_requirements.extend(geospatial_requirements)
    
    # Add requirements from requirements.txt
    base_requirements.extend(get_requirements())
    
    return base_requirements

# Get version from __init__.py or version.py
def get_version():
    """Get version from package."""
    try:
        # Try to read from __init__.py
        init_file = Path('bathymetric_cae') / '__init__.py'
        if init_file.exists():
            with open(init_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('__version__'):
                        return line.split('=')[1].strip().strip('"\'')
        
        # Try to read from version.py
        version_file = Path('bathymetric_cae') / 'version.py'
        if version_file.exists():
            with open(version_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('VERSION'):
                        return line.split('=')[1].strip().strip('"\'')
                    
    except Exception as e:
        print(f"Warning: Could not read version: {e}")
    
    return "2.0.0"  # Fallback version

# Enhanced long description with installation guidance
def get_enhanced_long_description():
    """Get enhanced long description with installation guidance."""
    base_description = get_long_description()
    
    installation_guide = """

## Installation

### Recommended: Conda Installation

For the best experience with GDAL and geospatial dependencies:

```bash
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd enhanced-bathymetric-cae
conda env create -f environment.yml
conda activate bathymetric-cae
pip install -e .
```

### Alternative: Pip Installation

```bash
pip install bathymetric-cae
```

**Note:** Pip installation may require separate GDAL system installation:
- **Ubuntu/Debian:** `sudo apt install gdal-bin libgdal-dev`
- **macOS:** `brew install gdal`
- **Windows:** Use conda installation (recommended)

### With GDAL support (pip only)

```bash
pip install bathymetric-cae[gdal]
```

### Complete installation with all dependencies

```bash
pip install bathymetric-cae[complete]
```

### Verification

```python
# Test installation
try:
    from osgeo import gdal
    print("GDAL available!")
except ImportError:
    print("GDAL not available - some features may be limited")

import bathymetric_cae
print("Installation successful!")
```
"""
    
    return base_description + installation_guide

# Custom command to print post-install message
class PostInstallCommand:
    """Custom command to print helpful message after installation."""
    
    @staticmethod
    def print_message():
        """Print helpful message after installation."""
        if is_conda_environment():
            print("\nInstallation completed in conda environment!")
            print("   GDAL and geospatial dependencies should be available.")
        else:
            print("\nInstallation completed with pip.")
            print("   For best GDAL support, consider using conda:")
            print("   conda env create -f environment.yml")
            print("   conda activate bathymetric-cae")
        
        print("\nTest your installation:")
        print("   bathymetric-cae --version")
        print("   python -c \"from osgeo import gdal; print('GDAL OK')\"")
        print("\nDocumentation:")
        print("   https://github.com/noaa-ocs-hydrography/bathymetric-cae/docs")

setup(
    name="bathymetric-cae",
    version=get_version(),
    author="Grant Froelich",
    author_email="grant.froelich@noaa.gov",
    description="Advanced bathymetric grid processing using ensemble Convolutional Autoencoders",
    long_description=get_enhanced_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/noaa-ocs-hydrography/bathymetric-cae",
    packages=find_packages(exclude=['tests*', 'docs*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    python_requires=">=3.8,<3.12",
    install_requires=get_install_requires(),
    extras_require={
        "gpu": [
            "tensorflow[and-cuda]>=2.13.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-html>=3.1.0",
            "pytest-mock>=3.10.0",
            "pytest-xdist>=3.0.0",
            "pytest-benchmark>=4.0.0",
            "hypothesis>=6.0.0",
            "factory-boy>=3.2.0",
            "memory-profiler>=0.60.0",
        ],
        "hydro": [
            "pyproj>=3.3.0",
            "fiona>=1.8.0", 
            "geopandas>=0.12.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
            "mkdocs-git-revision-date-localized-plugin>=1.0.0",
        ],
        "gdal": [
            "GDAL>=3.4.0",
            "rasterio>=1.3.0",
        ],
        "complete": [
            "GDAL>=3.4.0",
            "rasterio>=1.3.0",
            "pyproj>=3.3.0",
            "fiona>=1.8.0",
            "geopandas>=0.12.0",
            "opencv-python>=4.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bathymetric-cae=bathymetric_cae.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bathymetric_cae": ["*.json", "*.yaml", "*.yml", "*.md"],
        "tests": ["test_fixtures/*", "factories/*", "utils/*"],
        "docs": ["**/*"],
    },
    keywords=[
        "bathymetry", 
        "machine learning", 
        "deep learning", 
        "oceanography", 
        "hydrography",
        "gdal",
        "geospatial",
        "autoencoder",
        "tensorflow",
        "conda-forge"
    ],
    project_urls={
        "Bug Reports": "https://github.com/noaa-ocs-hydrography/bathymetric-cae/issues",
        "Source": "https://github.com/noaa-ocs-hydrography/bathymetric-cae",
        "Documentation": "https://github.com/noaa-ocs-hydrography/bathymetric-cae/docs",
        "Conda Environment": "https://github.com/noaa-ocs-hydrography/bathymetric-cae/blob/main/environment.yml",
        "Docker": "https://github.com/noaa-ocs-hydrography/bathymetric-cae/blob/main/Dockerfile.conda",
    },
    zip_safe=False,
)

# Print post-install message
if __name__ == "__main__":
    PostInstallCommand.print_message()