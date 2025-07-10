"""
Setup script for Enhanced Bathymetric CAE Processing
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="enhanced-bathymetric-cae",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced bathymetric grid processing using ensemble Convolutional Autoencoders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enhanced-bathymetric-cae",
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
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["tensorflow[and-cuda]"],
        "dev": ["pytest", "black", "flake8", "mypy"],
        "hydro": ["pyproj", "fiona", "geopandas"],
    },
    entry_points={
        "console_scripts": [
            "bathymetric-cae=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="bathymetry, machine learning, deep learning, oceanography, hydrography",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/enhanced-bathymetric-cae/issues",
        "Source": "https://github.com/yourusername/enhanced-bathymetric-cae",
        "Documentation": "https://github.com/yourusername/enhanced-bathymetric-cae/wiki",
    },
)
