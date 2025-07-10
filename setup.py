"""
Setup script for bathymetric_cae package.

Author: Bathymetric CAE Team
License: MIT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-'):
            # Handle conditional dependencies
            if ';' in line:
                requirements.append(line)
            else:
                requirements.append(line)

# Package metadata
setup(
    name="bathymetric-cae",
    version="1.0.0",
    author="Bathymetric CAE Team",
    author_email="grant.froelich@noaa.gov",
    description="Advanced Convolutional Autoencoder for Bathymetric Data Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noaa-ocs-hydrography/bathymetric-cae",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0", 
            "black>=21.0.0",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
            "twine>=3.4.0"
        ],
        "gpu": [
            "tensorflow[and-cuda]>=2.13.0"
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "bathymetric-cae=bathymetric_cae.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bathymetric_cae": [
            "config/*.json",
            "docs/*.md",
            "examples/*.py"
        ]
    },
    zip_safe=False,
    keywords=[
        "bathymetry",
        "deep-learning", 
        "autoencoder",
        "geospatial",
        "machine-learning",
        "data-processing",
        "gis",
        "remote-sensing"
    ],
    project_urls={
        "Bug Reports": "https://github.com/noaa-ocs-hydrography/bathymetric-cae/issues",
        "Documentation": "https://github.com/noaa-ocs-hydrography/bathymetric-cae/docs",
        "Source": "https://github.com/noaa-ocs-hydrography/bathymetric-cae",
    },
)
