"""
Configuration module for Enhanced Bathymetric CAE Processing.

This module contains the main configuration class and validation logic.
"""

import json
import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    """Enhanced configuration class with domain-specific parameters."""
    
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
    supported_formats: List[str] = None
    min_patch_size: int = 32
    max_workers: int = -1
    
    # Performance
    gpu_memory_growth: bool = True
    use_mixed_precision: bool = True
    prefetch_buffer_size: int = None  # Will be set to tf.data.AUTOTUNE
    
    # Domain-specific parameters
    enable_adaptive_processing: bool = True
    enable_expert_review: bool = True
    enable_constitutional_constraints: bool = True
    quality_threshold: float = 0.7
    auto_flag_threshold: float = 0.5
    ensemble_size: int = 3
    
    # Multi-objective weights
    ssim_weight: float = 0.3
    roughness_weight: float = 0.2
    feature_preservation_weight: float = 0.3
    consistency_weight: float = 0.2
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.supported_formats is None:
            self.supported_formats = ['.bag', '.tif', '.tiff', '.asc', '.xyz']
        
        # Set TensorFlow AUTOTUNE if not set
        if self.prefetch_buffer_size is None:
            try:
                import tensorflow as tf
                self.prefetch_buffer_size = tf.data.AUTOTUNE
            except ImportError:
                self.prefetch_buffer_size = 2
        
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
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
        if self.base_filters < 1:
            errors.append("base_filters must be positive")
        if self.depth < 1:
            errors.append("depth must be at least 1")
        if not 0 <= self.dropout_rate < 1:
            errors.append("dropout_rate must be between 0 and 1")
        
        # Ensemble validation
        if self.ensemble_size < 1:
            errors.append("ensemble_size must be at least 1")
        
        # Quality thresholds validation
        if not 0 <= self.quality_threshold <= 1:
            errors.append("quality_threshold must be between 0 and 1")
        if not 0 <= self.auto_flag_threshold <= 1:
            errors.append("auto_flag_threshold must be between 0 and 1")
        
        # Weight validation
        weight_sum = (self.ssim_weight + self.roughness_weight + 
                     self.feature_preservation_weight + self.consistency_weight)
        if abs(weight_sum - 1.0) > 1e-6:
            errors.append("Quality metric weights must sum to 1.0")
        
        # Individual weight validation
        weights = [self.ssim_weight, self.roughness_weight, 
                  self.feature_preservation_weight, self.consistency_weight]
        if any(w < 0 for w in weights):
            errors.append("All weights must be non-negative")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and handle special values
        config_dict = asdict(self)
        
        # Handle TensorFlow AUTOTUNE
        if str(config_dict.get('prefetch_buffer_size', '')).endswith('AUTOTUNE'):
            config_dict['prefetch_buffer_size'] = -1  # Use -1 as placeholder
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        # Handle special values
        if config_data.get('prefetch_buffer_size') == -1:
            config_data['prefetch_buffer_size'] = None  # Will be set in __post_init__
        
        return cls(**config_data)
    
    def update_from_dict(self, updates: dict):
        """Update configuration from dictionary."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def get_model_config(self) -> dict:
        """Get model-specific configuration."""
        return {
            'grid_size': self.grid_size,
            'base_filters': self.base_filters,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'ensemble_size': self.ensemble_size
        }
    
    def get_training_config(self) -> dict:
        """Get training-specific configuration."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'learning_rate': self.learning_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'reduce_lr_patience': self.reduce_lr_patience,
            'reduce_lr_factor': self.reduce_lr_factor,
            'min_lr': self.min_lr
        }
    
    def get_quality_weights(self) -> dict:
        """Get quality metric weights."""
        return {
            'ssim_weight': self.ssim_weight,
            'roughness_weight': self.roughness_weight,
            'feature_preservation_weight': self.feature_preservation_weight,
            'consistency_weight': self.consistency_weight
        }


def create_default_config() -> Config:
    """Create a default configuration instance."""
    return Config()


def load_config_from_file(file_path: str) -> Config:
    """Load configuration from file with error handling."""
    try:
        return Config.load(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def merge_configs(base_config: Config, override_config: dict) -> Config:
    """Merge base configuration with overrides."""
    base_dict = base_config.to_dict()
    base_dict.update(override_config)
    return Config(**base_dict)
