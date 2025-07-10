"""
Configuration Management Module

This module provides a flexible configuration system for the bathymetric CAE processing pipeline.
It supports loading from JSON files, validation, and type hints for all parameters.

Author: Bathymetric CAE Team
License: MIT
"""

import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """
    Configuration class with validation and type hints.
    
    This class manages all configuration parameters for the bathymetric processing pipeline,
    including I/O paths, training parameters, model architecture, and performance settings.
    
    Attributes:
        input_folder (str): Path to input bathymetric files
        output_folder (str): Path for output processed files
        model_path (str): Path to save/load model file
        log_dir (str): Directory for logging output
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        validation_split (float): Fraction of data for validation
        learning_rate (float): Learning rate for optimizer
        grid_size (int): Input grid size for model
        base_filters (int): Base number of filters in CNN
        depth (int): Model depth (number of encoder/decoder layers)
        dropout_rate (float): Dropout rate for regularization
        early_stopping_patience (int): Patience for early stopping
        reduce_lr_patience (int): Patience for learning rate reduction
        reduce_lr_factor (float): Factor for learning rate reduction
        min_lr (float): Minimum learning rate
        supported_formats (List[str]): Supported file formats
        min_patch_size (int): Minimum patch size for processing
        max_workers (int): Maximum number of worker processes
        gpu_memory_growth (bool): Enable GPU memory growth
        use_mixed_precision (bool): Use mixed precision training
        prefetch_buffer_size (int): Buffer size for data prefetching
    """
    
    # I/O Paths
    input_folder: str = r"\\network_folder\input_bathymetric_files"
    output_folder: str = r"\\network_folder\output_bathymetric_files"
    model_path: str = "cae_model_with_uncertainty.h5"
    
    # Logging
    log_dir: str = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_level: str = "INFO"
    
    # Training Parameters
    epochs: int = 100
    batch_size: int = 8
    validation_split: float = 0.2
    learning_rate: float = 0.001
    
    # Model Architecture
    grid_size: int = 512
    base_filters: int = 32
    depth: int = 4
    dropout_rate: float = 0.2
    
    # Callbacks
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Processing
    supported_formats: Optional[List[str]] = None
    min_patch_size: int = 32
    max_workers: int = -1
    
    # Performance
    gpu_memory_growth: bool = True
    use_mixed_precision: bool = True
    prefetch_buffer_size: int = None  # Will be set to tf.data.AUTOTUNE
    
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.supported_formats is None:
            self.supported_formats = ['.bag', '.tif', '.tiff', '.asc', '.xyz']
        
        # Set prefetch buffer size if not specified
        if self.prefetch_buffer_size is None:
            try:
                import tensorflow as tf
                self.prefetch_buffer_size = tf.data.AUTOTUNE
            except ImportError:
                self.prefetch_buffer_size = 2
        
        self.validate()
    
    def validate(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        errors = []
        
        # Training parameter validation
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if not 0 < self.validation_split < 1:
            errors.append("validation_split must be between 0 and 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        # Model architecture validation
        if self.grid_size < 32:
            errors.append("grid_size must be at least 32")
        if self.base_filters <= 0:
            errors.append("base_filters must be positive")
        if self.depth <= 0:
            errors.append("depth must be positive")
        if not 0 <= self.dropout_rate < 1:
            errors.append("dropout_rate must be between 0 and 1")
        
        # Callback validation
        if self.early_stopping_patience <= 0:
            errors.append("early_stopping_patience must be positive")
        if self.reduce_lr_patience <= 0:
            errors.append("reduce_lr_patience must be positive")
        if not 0 < self.reduce_lr_factor < 1:
            errors.append("reduce_lr_factor must be between 0 and 1")
        if self.min_lr <= 0:
            errors.append("min_lr must be positive")
        
        # Processing validation
        if self.min_patch_size < 16:
            errors.append("min_patch_size must be at least 16")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    def save(self, path: Union[str, Path]):
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dict and handle special values
        config_dict = asdict(self)
        
        # Handle non-serializable values
        for key, value in config_dict.items():
            if hasattr(value, '__name__'):  # For tf.data.AUTOTUNE and similar
                config_dict[key] = str(value)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Config: Loaded configuration instance
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration file is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            # Handle special string values that need conversion
            if 'prefetch_buffer_size' in config_dict:
                if isinstance(config_dict['prefetch_buffer_size'], str):
                    config_dict['prefetch_buffer_size'] = None  # Will be set in __post_init__
            
            return cls(**config_dict)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid configuration file: {e}")
    
    def update(self, **kwargs) -> 'Config':
        """
        Create a new configuration with updated parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            Config: New configuration instance with updated parameters
        """
        config_dict = asdict(self)
        config_dict.update(kwargs)
        return Config(**config_dict)
    
    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict: Configuration as dictionary
        """
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(epochs={self.epochs}, batch_size={self.batch_size}, " \
               f"grid_size={self.grid_size}, learning_rate={self.learning_rate})"
    
    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"Config({asdict(self)})"


def load_default_config() -> Config:
    """
    Load default configuration.
    
    Returns:
        Config: Default configuration instance
    """
    return Config()


def load_config_from_file(path: Union[str, Path]) -> Config:
    """
    Load configuration from file with fallback to default.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Config: Loaded configuration or default if file doesn't exist
    """
    try:
        return Config.load(path)
    except FileNotFoundError:
        print(f"Configuration file {path} not found, using default configuration")
        return Config()
    except ValueError as e:
        print(f"Error loading configuration: {e}, using default configuration")
        return Config()


def create_config_from_args(args) -> Config:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Config: Configuration instance
    """
    # Start with default config
    config = Config()
    
    # Update with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            # Convert argument names to config attribute names
            config_key = key.replace('-', '_')
            if hasattr(config, config_key):
                setattr(config, config_key, value)
    
    return config
