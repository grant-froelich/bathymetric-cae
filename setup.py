"""
Setup script for Enhanced Bathymetric CAE Processing
Updated to handle conda/pip installation differences
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Check if we're in a conda environment
def is_conda_environment():
    """Check if we're running in a conda environment."""
    return ('CONDA_DEFAULT_ENV' in os.environ or 
            'CONDA_PREFIX' in os.environ or
            os.path.exists(os.path.join(sys.prefix, 'conda-meta')))

# Read base requirements
requirements = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # Skip GDAL-related packages if in conda environment
            if is_conda_environment() and any(pkg in line.lower() for pkg in ['gdal', 'rasterio', 'opencv']):
                continue
            requirements.append(line)

# Conditional requirements based on environment
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
        "pathlib2>=2.3.0",
    ]
    
    # Add geospatial packages only if not in conda environment
    if not is_conda_environment():
        print("⚠️  Warning: Installing without conda may cause GDAL issues.")
        print("   Recommended: conda env create -f environment.yml")
        geospatial_requirements = [
            "opencv-python>=4.5.0",
            # Note: GDAL and rasterio commented out due to installation complexity
            # "GDAL>=3.4.0",  # Uncomment if system GDAL is installed
            # "rasterio>=1.3.0",  # Uncomment if system GDAL is installed
        ]
        base_requirements.extend(geospatial_requirements)
    
    return base_requirements

# Enhanced long description with installation guidance
enhanced_long_description = long_description + """

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

### Verification

```python
# Test installation
from osgeo import gdal
import enhanced_bathymetric_cae
print("Installation successful!")
```

"""

setup(
    name="bathymetric-cae",
    version="2.0.0",
    author="Grant Froelich",
    author_email="grant.froelich@noaa.gov",
    description="Advanced bathymetric grid processing using ensemble Convolutional Autoencoders",
    long_description=enhanced_long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noaa-ocs-hydrography/bathymetric-cae",
    packages=find_packages(),
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
        # Fallback option for pip-only installation
        "gdal": [
            "GDAL>=3.4.0",
            "rasterio>=1.3.0",
        ],
        # Complete installation (discouraged without conda)
        "complete": [
            "GDAL>=3.4.0",
            "rasterio>=1.3.0",
            "pyproj>=3.3.0",
            "fiona>=1.8.0",
            "geopandas>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bathymetric-cae=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.md"],
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
    
    # Custom installation message
    cmdclass={},
)

# Post-installation message
def print_post_install_message():
    """Print helpful message after installation."""
    if is_conda_environment():
        print("\n✅ Installation completed in conda environment!")
        print("   GDAL and geospatial dependencies should be available.")
    else:
        print("\n⚠️  Installation completed with pip.")
        print("   For best GDAL support, consider using conda:")
        print("   conda env create -f environment.yml")
        print("   conda activate bathymetric-cae")
    
    print("\n🧪 Test your installation:")
    print("   bathymetric-cae --version")
    print("   python -c \"from osgeo import gdal; print('GDAL OK')\"")
    print("\n📚 Documentation:")
    print("   https://github.com/noaa-ocs-hydrography/bathymetric-cae/docs")

# Print message if this is being run directly
if __name__ == "__main__":
    # Don't print during build/install, only during direct execution
    if len(sys.argv) > 1 and 'install' in sys.argv:
        import atexit
        atexit.register(print_post_install_message)