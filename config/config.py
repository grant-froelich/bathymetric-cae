"""
Configuration module for Enhanced Bathymetric CAE Processing.
"""

import json
import datetime
from typing import List
from dataclasses import dataclass, asdict


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
    prefetch_buffer_size: int = -1  # Will be set to tf.data.AUTOTUNE
    
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
        if self.supported_formats is None:
            self.supported_formats = ['.bag', '.tif', '.tiff', '.asc', '.xyz']
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        errors = []
        
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if not 0 < self.validation_split < 1:
            errors.append("validation_split must be between 0 and 1")
        if self.grid_size < 32:
            errors.append("grid_size must be at least 32")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.ensemble_size < 1:
            errors.append("ensemble_size must be at least 1")
        
        # Validate weight sum
        weight_sum = (self.ssim_weight + self.roughness_weight + 
                     self.feature_preservation_weight + self.consistency_weight)
        if abs(weight_sum - 1.0) > 1e-6:
            errors.append("Quality metric weights must sum to 1.0")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls(**json.load(f))
