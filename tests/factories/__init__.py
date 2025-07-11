# tests/factories/__init__.py

"""
Test data factories package for Enhanced Bathymetric CAE Processing.

This package provides factory classes for generating realistic test data
including bathymetric grids, configuration objects, and quality metrics.
"""

from .data_factory import (
    BathymetricDataFactory,
    ConfigurationFactory,
    QualityMetricsFactory,
    TestFileFactory
)

__all__ = [
    'BathymetricDataFactory',
    'ConfigurationFactory', 
    'QualityMetricsFactory',
    'TestFileFactory'
]

__version__ = '2.0.0'
__author__ = 'Enhanced Bathymetric CAE Team'