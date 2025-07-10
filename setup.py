"""
Setup script for Enhanced Bathymetric CAE Processing package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "tensorflow>=2.13.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "opencv-python>=4.6.0",
        "gdal>=3.4.0",
        "joblib>=1.1.0",
        "psutil>=5.8.0",
        "pathlib2>=2.3.0; python_version<'3.4'",
    ]

# Optional dependencies
extras_require = {
    'gpu': ['tensorflow[and-cuda]>=2.13.0'],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.910',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'myst-parser>=0.17.0',
    ],
    'hydro': [
        'pyproj>=3.3.0',
        'rasterio>=1.3.0',
        'fiona>=1.8.0',
        'geopandas>=0.11.0',
    ]
}

# All extra dependencies
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name="bathymetric-cae",
    version="2.0.0",
    author="Enhanced Bathymetric CAE Team",
    author_email="contact@example.com",
    description="Enhanced Bathymetric Grid Processing using Advanced Ensemble Convolutional Autoencoders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/bathymetric-cae",
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
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'bathymetric-cae=main:main',
            'bathymetric-process=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'bathymetric_cae': [
            'config/*.json',
            'docs/*.md',
            'examples/*.py',
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/username/bathymetric-cae/issues",
        "Source": "https://github.com/username/bathymetric-cae",
        "Documentation": "https://bathymetric-cae.readthedocs.io",
    },
    keywords=[
        "bathymetry", 
        "deep-learning", 
        "convolutional-autoencoder", 
        "ensemble-learning",
        "hydrography", 
        "data-cleaning", 
        "tensorflow",
        "adaptive-processing",
        "quality-metrics"
    ],
    zip_safe=False,
)
