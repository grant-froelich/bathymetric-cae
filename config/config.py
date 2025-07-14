"""
Configuration module for Enhanced Bathymetric CAE Processing.
Updated to support modern Keras format as default.
"""

import json
import datetime
from typing import List
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    """Enhanced configuration class with modern Keras format support."""
    # I/O Paths
    input_folder: str = r"\\network_folder\input_bathymetric_files"
    output_folder: str = r"\\network_folder\output_bathymetric_files"
    model_path: str = "cae_model_with_uncertainty.keras"  # Updated to .keras format
    supported_formats: List[str] = field(default_factory=lambda: ['.bag', '.tif', '.tiff', '.asc', '.xyz'])
    
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
    min_patch_size: int = 32
    max_workers: int = -1
    
    # Performance
    gpu_memory_growth: bool = True
    use_mixed_precision: bool = True
    prefetch_buffer_size: int = -1  # Will be set to tf.data.AUTOTUNE
    
    # Model Format (New)
    model_format: str = "keras"  # "keras" (modern) or "h5" (legacy)
    auto_convert_legacy: bool = True  # Automatically convert H5 to Keras format
    
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
        self.validate()
        self._normalize_model_path()
    
    def _normalize_model_path(self):
        """Ensure model path uses correct format extension."""
        if self.model_format == "keras":
            if self.model_path.endswith('.h5'):
                self.model_path = self.model_path.replace('.h5', '.keras')
            elif not self.model_path.endswith('.keras'):
                if '.' in self.model_path:
                    # Replace existing extension
                    base_path = '.'.join(self.model_path.split('.')[:-1])
                    self.model_path = f"{base_path}.keras"
                else:
                    # Add extension
                    self.model_path += '.keras'
        elif self.model_format == "h5":
            if self.model_path.endswith('.keras'):
                self.model_path = self.model_path.replace('.keras', '.h5')
            elif not self.model_path.endswith('.h5'):
                if '.' in self.model_path:
                    # Replace existing extension
                    base_path = '.'.join(self.model_path.split('.')[:-1])
                    self.model_path = f"{base_path}.h5"
                else:
                    # Add extension
                    self.model_path += '.h5'
    
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
        
        # Validate model format
        if self.model_format not in ["keras", "h5"]:
            errors.append("model_format must be 'keras' or 'h5'")
        
        # Validate weight sum
        weight_sum = (self.ssim_weight + self.roughness_weight + 
                     self.feature_preservation_weight + self.consistency_weight)
        if abs(weight_sum - 1.0) > 1e-6:
            errors.append("Quality metric weights must sum to 1.0")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    def set_model_format(self, format_type: str):
        """Set model format and update model path accordingly."""
        if format_type not in ["keras", "h5"]:
            raise ValueError("format_type must be 'keras' or 'h5'")
        
        self.model_format = format_type
        self._normalize_model_path()
    
    def get_model_extension(self) -> str:
        """Get the appropriate file extension for current model format."""
        return ".keras" if self.model_format == "keras" else ".h5"
    
    def is_modern_format(self) -> bool:
        """Check if using modern Keras format."""
        return self.model_format == "keras"
    
    def get_ensemble_paths(self, base_path: str = None) -> List[str]:
        """Get list of ensemble model paths."""
        if base_path is None:
            base_path = self.model_path
        
        # Remove extension to get base name
        if base_path.endswith('.keras'):
            base_name = base_path[:-6]  # Remove .keras
        elif base_path.endswith('.h5'):
            base_name = base_path[:-3]  # Remove .h5
        else:
            base_name = base_path
        
        extension = self.get_model_extension()
        return [f"{base_name}_ensemble_{i}{extension}" for i in range(self.ensemble_size)]
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        config_data = asdict(self)
        
        # Add metadata
        config_data['_metadata'] = {
            'created_date': datetime.datetime.now().isoformat(),
            'config_version': '2.0.0',
            'model_format_support': 'keras_native',
            'description': 'Enhanced Bathymetric CAE Configuration with Modern Keras Format'
        }
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file with format migration."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Remove metadata if present
        if '_metadata' in data:
            metadata = data.pop('_metadata')
            # Could use metadata for version-specific loading logic
        
        # Handle legacy configurations
        if 'model_format' not in data:
            # Default to modern format for new configs
            data['model_format'] = 'keras'
            
        # Handle legacy model paths
        if 'model_path' in data and data['model_path'].endswith('.h5'):
            if data.get('model_format', 'keras') == 'keras':
                # Auto-migrate to modern format
                data['model_path'] = data['model_path'].replace('.h5', '.keras')
        
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def create_quick_test_config(cls):
        """Create configuration optimized for quick testing."""
        return cls(
            epochs=20,
            batch_size=2,
            learning_rate=0.001,
            grid_size=256,
            ensemble_size=1,
            base_filters=16,
            depth=3,
            enable_adaptive_processing=True,
            enable_expert_review=False,
            enable_constitutional_constraints=False,
            quality_threshold=0.4,
            validation_split=0.1,
            model_format='keras'  # Use modern format
        )
    
    @classmethod
    def create_production_config(cls):
        """Create configuration optimized for production."""
        return cls(
            epochs=200,
            batch_size=8,
            learning_rate=0.0005,
            grid_size=1024,
            ensemble_size=5,
            base_filters=48,
            depth=5,
            enable_adaptive_processing=True,
            enable_expert_review=True,
            enable_constitutional_constraints=True,
            quality_threshold=0.85,
            validation_split=0.2,
            model_format='keras'  # Use modern format
        )
    
    @classmethod
    def migrate_legacy_config(cls, legacy_config_path: str, output_path: str = None):
        """Migrate legacy H5-based configuration to modern Keras format."""
        # Load legacy config
        config = cls.load(legacy_config_path)
        
        # Force modern format
        config.set_model_format('keras')
        
        # Update other settings for better compatibility
        if hasattr(config, 'auto_convert_legacy'):
            config.auto_convert_legacy = True
        
        # Save migrated config
        if output_path is None:
            output_path = legacy_config_path.replace('.json', '_migrated.json')
        
        config.save(output_path)
        return config, output_path