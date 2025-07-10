# bathymetric_cae/__init__.py
"""
Enhanced Bathymetric CAE Processing Package.

A modular system for bathymetric data cleaning using ensemble convolutional autoencoders
with domain-specific enhancements including adaptive processing, constitutional constraints,
and expert review capabilities.
"""

from .config import Config, load_config_from_file
from .core import (
    SeafloorType, QualityLevel, BathymetricConstraints,
    SeafloorClassifier, AdaptiveProcessor, BathymetricQualityMetrics
)

__version__ = "2.0.0"
__author__ = "Enhanced Bathymetric CAE Team"

# Package-level convenience functions
def create_default_pipeline(config=None):
    """Create a default processing pipeline with standard configuration."""
    from .core import create_full_processing_pipeline
    return create_full_processing_pipeline(config)

def quick_process(input_path, output_path, **kwargs):
    """Quick processing function for simple use cases."""
    from .processing import EnhancedBathymetricCAEPipeline
    
    config = Config()
    config.update_from_dict(kwargs)
    
    pipeline = EnhancedBathymetricCAEPipeline(config)
    pipeline.run(input_path, output_path, config.model_path)

__all__ = [
    'Config', 'load_config_from_file',
    'SeafloorType', 'QualityLevel', 'BathymetricConstraints',
    'SeafloorClassifier', 'AdaptiveProcessor', 'BathymetricQualityMetrics',
    'create_default_pipeline', 'quick_process'
]