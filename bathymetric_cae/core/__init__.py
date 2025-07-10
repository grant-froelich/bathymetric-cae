# core/__init__.py
"""Core modules for bathymetric CAE processing."""

from .processor import BathymetricProcessor, get_supported_formats, validate_gdal_installation
from .model import AdvancedCAE, create_data_augmentation_layer, calculate_model_memory_requirements
from .pipeline import BathymetricCAEPipeline, run_pipeline_from_config, validate_pipeline_requirements

__all__ = [
    'BathymetricProcessor',
    'AdvancedCAE', 
    'BathymetricCAEPipeline',
    'get_supported_formats',
    'validate_gdal_installation',
    'create_data_augmentation_layer',
    'calculate_model_memory_requirements',
    'run_pipeline_from_config',
    'validate_pipeline_requirements'
]

# ================================================================================

