"""
Bathymetric CAE - Advanced Convolutional Autoencoder for Bathymetric Data Processing

This package provides a comprehensive pipeline for processing bathymetric data
using advanced Convolutional Autoencoders with modern machine learning techniques.

Author: Bathymetric CAE Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Bathymetric CAE Team"
__license__ = "MIT"
__email__ = "grant.froelich@noaa.gov"

# Core imports
from .config.config import Config
from .core.pipeline import BathymetricCAEPipeline
from .core.pipeline import validate_installation
from .core.processor import BathymetricProcessor
from .core.model import AdvancedCAE
from .visualization.visualizer import Visualizer

# Utility imports
from .utils.logging_utils import setup_logging, get_logger
from .utils.memory_utils import memory_monitor, get_memory_info
from .utils.gpu_utils import check_gpu_availability, optimize_gpu_performance
from .utils.file_utils import get_valid_files, FileManager

# Version info
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'license': __license__,
    'description': 'Advanced Convolutional Autoencoder for Bathymetric Data Processing'
}

# Package-level constants
SUPPORTED_FORMATS = ['.bag', '.tif', '.tiff', '.asc', '.xyz']
DEFAULT_GRID_SIZE = 512
DEFAULT_BATCH_SIZE = 8

# Public API
__all__ = [
    # Core classes
    'Config',
    'BathymetricCAEPipeline', 
    'BathymetricProcessor',
    'AdvancedCAE',
    'Visualizer',
    
    # Utility functions
    'setup_logging',
    'get_logger',
    'memory_monitor',
    'get_memory_info',
    'check_gpu_availability',
    'optimize_gpu_performance',
    'get_valid_files',
    'FileManager',
    
    # Constants
    'VERSION_INFO',
    'SUPPORTED_FORMATS',
    'DEFAULT_GRID_SIZE',
    'DEFAULT_BATCH_SIZE',
    
    # Quick start function
    'quick_start'
]

def quick_start(input_folder: str, output_folder: str, epochs: int = 50, batch_size: int = 8, **kwargs):
    """
    Quick start function for basic bathymetric processing.
    
    Args:
        input_folder: Path to input bathymetric files
        output_folder: Path for output processed files
        epochs: Number of training epochs
        batch_size: Training batch size
        **kwargs: Additional configuration parameters
        
    Returns:
        dict: Processing results
    """
    from .config.config import Config
    from .core.pipeline import BathymetricCAEPipeline
    
    # Create configuration
    config = Config(
        input_folder=input_folder,
        output_folder=output_folder,
        epochs=epochs,
        batch_size=batch_size,
        **kwargs
    )
    
    # Create and run pipeline
    pipeline = BathymetricCAEPipeline(config)
    
    return pipeline.run(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=f"{output_folder}/quick_start_model.h5"
    )