# Configuration Guide

The Enhanced Bathymetric CAE Processing system provides extensive configuration options to adapt processing to your specific needs. This guide covers all configuration aspects from basic settings to advanced customization.

## 🎯 Configuration Overview

The system uses a hierarchical configuration approach:

1. **Default Values**: Built-in sensible defaults
2. **Configuration Files**: JSON files for persistent settings
3. **Environment Variables**: System-level overrides
4. **Command Line Arguments**: Runtime overrides
5. **Programmatic Settings**: Direct API configuration

## ⚙️ Configuration Methods

### 1. Using Configuration Files (Recommended)

Create a JSON configuration file:

```json
{
  "input_folder": "data/bathymetry",
  "output_folder": "data/processed",
  "epochs": 150,
  "batch_size": 8,
  "ensemble_size": 3,
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "quality_threshold": 0.8
}
```

Use the configuration:
```bash
# Save current settings to file
bathymetric-cae --save-config my_config.json

# Load and use configuration
bathymetric-cae --config my_config.json

# Override specific settings
bathymetric-cae --config my_config.json --epochs 200
```

### 2. Command Line Configuration

```bash
# Complete command line configuration
bathymetric-cae \
    --input data/bathymetry \
    --output data/processed \
    --epochs 150 \
    --batch-size 8 \
    --ensemble-size 3 \
    --grid-size 512 \
    --learning-rate 0.001 \
    --enable-adaptive \
    --enable-expert-review \
    --enable-constitutional \
    --quality-threshold 0.8
```

### 3. Python API Configuration

```python
from enhanced_bathymetric_cae import Config

# Create configuration object
config = Config(
    input_folder="data/bathymetry",
    output_folder="data/processed",
    epochs=150,
    batch_size=8,
    ensemble_size=3,
    enable_adaptive_processing=True,
    enable_expert_review=True,
    quality_threshold=0.8
)

# Validate configuration
config.validate()

# Save to file
config.save("my_config.json")

# Load from file
config = Config.load("my_config.json")
```

### 4. Environment Variables

```bash
# Set environment variables
export BATHYMETRIC_CAE_INPUT_FOLDER="data/bathymetry"
export BATHYMETRIC_CAE_OUTPUT_FOLDER="data/processed"
export BATHYMETRIC_CAE_EPOCHS=150
export BATHYMETRIC_CAE_ENABLE_ADAPTIVE=true

# Run with environment variables
bathymetric-cae
```

## 📋 Configuration Categories

### Basic I/O Configuration

```python
config = Config(
    # Input/Output paths
    input_folder="data/bathymetry",           # Input directory path
    output_folder="data/processed",           # Output directory path
    model_path="models/cae_ensemble.h5",      # Model save/load path
    
    # Supported file formats
    supported_formats=['.bag', '.tif', '.tiff', '.asc', '.xyz'],
    
    # Logging
    log_dir="logs/",                          # Log directory
    log_level="INFO"                          # DEBUG, INFO, WARNING, ERROR
)
```

### Training Parameters

```python
config = Config(
    # Core training settings
    epochs=100,                               # Training iterations (10-500)
    batch_size=8,                            # Batch size (1-32)
    learning_rate=0.001,                     # Learning rate (1e-6 to 1e-2)
    validation_split=0.2,                    # Validation data ratio (0.1-0.3)
    
    # Early stopping and learning rate reduction
    early_stopping_patience=15,              # Epochs to wait before stopping
    reduce_lr_patience=8,                    # Epochs before reducing LR
    reduce_lr_factor=0.5,                   # LR reduction factor
    min_lr=1e-7                             # Minimum learning rate
)
```

### Model Architecture

```python
config = Config(
    # Model structure
    grid_size=512,                           # Input grid resolution (128-2048)
    base_filters=32,                         # Base number of filters (16-64)
    depth=4,                                 # Model depth (2-6)
    dropout_rate=0.2,                       # Dropout rate (0.0-0.5)
    
    # Ensemble settings
    ensemble_size=3                          # Number of models (1-10)
)
```

### Enhanced Features

```python
config = Config(
    # AI enhancements
    enable_adaptive_processing=True,         # Seafloor-specific processing
    enable_expert_review=True,              # Human-in-the-loop validation
    enable_constitutional_constraints=True,  # AI safety constraints
    
    # Quality control
    quality_threshold=0.7,                  # Expert review trigger (0.0-1.0)
    auto_flag_threshold=0.5                 # Auto-flagging threshold (0.0-1.0)
)
```

### Quality Metric Weights

```python
config = Config(
    # Quality metric importance (must sum to 1.0)
    ssim_weight=0.3,                        # Structural similarity weight
    roughness_weight=0.2,                  # Surface roughness weight
    feature_preservation_weight=0.3,        # Feature preservation weight
    consistency_weight=0.2                  # Depth consistency weight
)
```

### Performance Settings

```python
config = Config(
    # Processing optimization
    max_workers=-1,                         # CPU cores (-1 = auto)
    min_patch_size=32,                      # Minimum processing patch size
    
    # GPU settings
    gpu_memory_growth=True,                 # Enable GPU memory growth
    use_mixed_precision=True,               # Enable mixed precision training
    
    # Data pipeline
    prefetch_buffer_size=-1                 # Prefetch buffer (auto)
)
```

## 🏭 Predefined Configurations

### Development Configuration

```python
dev_config = Config(
    # Fast processing for development
    epochs=25,
    batch_size=2,
    ensemble_size=1,
    grid_size=256,
    
    # Minimal features
    enable_adaptive_processing=False,
    enable_expert_review=False,
    enable_constitutional_constraints=False,
    
    # Relaxed quality
    quality_threshold=0.5
)
```

### Production Configuration

```python
prod_config = Config(
    # High quality processing
    epochs=200,
    batch_size=8,
    ensemble_size=5,
    grid_size=1024,
    
    # All features enabled
    enable_adaptive_processing=True,
    enable_expert_review=True,
    enable_constitutional_constraints=True,
    
    # Strict quality
    quality_threshold=0.85,
    
    # Optimized weights for production
    ssim_weight=0.25,
    roughness_weight=0.15,
    feature_preservation_weight=0.35,
    consistency_weight=0.25
)
```

### Research Configuration

```python
research_config = Config(
    # Thorough processing for research
    epochs=300,
    batch_size=4,
    ensemble_size=7,
    grid_size=1024,
    
    # Conservative settings
    learning_rate=0.0005,
    dropout_rate=0.1,
    
    # Maximum quality
    enable_adaptive_processing=True,
    enable_expert_review=True,
    enable_constitutional_constraints=True,
    quality_threshold=0.9
)
```

### Testing Configuration

```python
test_config = Config(
    # Minimal resources for testing
    epochs=5,
    batch_size=1,
    ensemble_size=1,
    grid_size=128,
    
    # Fast execution
    enable_adaptive_processing=True,
    enable_expert_review=False,
    enable_constitutional_constraints=False,
    
    # Relaxed validation
    quality_threshold=0.3
)
```

## 📊 Configuration Templates

### Template: High-Performance Processing

```json
{
  "name": "High Performance Configuration",
  "description": "Optimized for speed with acceptable quality",
  
  "training": {
    "epochs": 75,
    "batch_size": 16,
    "learning_rate": 0.002
  },
  
  "model": {
    "ensemble_size": 2,
    "grid_size": 512,
    "base_filters": 24
  },
  
  "features": {
    "enable_adaptive_processing": true,
    "enable_expert_review": false,
    "enable_constitutional_constraints": true
  },
  
  "quality": {
    "quality_threshold": 0.7,
    "ssim_weight": 0.4,
    "feature_preservation_weight": 0.3,
    "consistency_weight": 0.2,
    "roughness_weight": 0.1
  }
}
```

### Template: Maximum Quality Processing

```json
{
  "name": "Maximum Quality Configuration",
  "description": "Best possible quality regardless of processing time",
  
  "training": {
    "epochs": 400,
    "batch_size": 4,
    "learning_rate": 0.0003,
    "validation_split": 0.25
  },
  
  "model": {
    "ensemble_size": 9,
    "grid_size": 1024,
    "base_filters": 48,
    "depth": 5
  },
  
  "features": {
    "enable_adaptive_processing": true,
    "enable_expert_review": true,
    "enable_constitutional_constraints": true
  },
  
  "quality": {
    "quality_threshold": 0.95,
    "auto_flag_threshold": 0.8,
    "ssim_weight": 0.2,
    "feature_preservation_weight": 0.4,
    "consistency_weight": 0.3,
    "roughness_weight": 0.1
  }
}
```

### Template: Memory-Constrained Processing

```json
{
  "name": "Memory Constrained Configuration",
  "description": "Optimized for systems with limited RAM",
  
  "training": {
    "epochs": 100,
    "batch_size": 1,
    "learning_rate": 0.001
  },
  
  "model": {
    "ensemble_size": 1,
    "grid_size": 256,
    "base_filters": 16,
    "depth": 3
  },
  
  "performance": {
    "max_workers": 1,
    "gpu_memory_growth": true,
    "use_mixed_precision": false
  },
  
  "features": {
    "enable_adaptive_processing": true,
    "enable_expert_review": false,
    "enable_constitutional_constraints": true
  }
}
```

## 🔧 Advanced Configuration

### Custom Quality Metrics

```python
config = Config(
    # Custom quality metric weights for specific use cases
    
    # For feature-rich areas (seamounts, canyons)
    ssim_weight=0.2,
    feature_preservation_weight=0.5,
    consistency_weight=0.2,
    roughness_weight=0.1,
    
    # For smooth areas (abyssal plains)
    # ssim_weight=0.4,
    # feature_preservation_weight=0.1,
    # consistency_weight=0.3,
    # roughness_weight=0.2
)
```

### Adaptive Processing Customization

```python
# Custom adaptive processing parameters
adaptive_config = {
    "shallow_coastal": {
        "smoothing_factor": 0.2,      # Very light smoothing
        "edge_preservation": 0.9,      # High edge preservation
        "noise_threshold": 0.05        # Low noise tolerance
    },
    "deep_ocean": {
        "smoothing_factor": 0.8,      # Heavy smoothing
        "edge_preservation": 0.3,      # Low edge preservation
        "noise_threshold": 0.3         # High noise tolerance
    }
}

config = Config(
    enable_adaptive_processing=True,
    adaptive_parameters=adaptive_config
)
```

### Expert Review Customization

```python
config = Config(
    enable_expert_review=True,
    
    # Review triggering conditions
    quality_threshold=0.75,           # Main quality threshold
    auto_flag_threshold=0.5,          # Auto-flagging threshold
    
    # Specific metric thresholds
    min_ssim_threshold=0.6,           # Minimum SSIM before flagging
    min_feature_preservation=0.4,     # Minimum feature preservation
    max_roughness_threshold=0.8,      # Maximum acceptable roughness
    
    # Review database settings
    review_db_path="reviews.db",      # Database file path
    review_retention_days=365         # Keep reviews for 1 year
)
```

## 🌍 Environment-Specific Settings

### Development Environment

```bash
# .env.development
BATHYMETRIC_CAE_EPOCHS=25
BATHYMETRIC_CAE_ENSEMBLE_SIZE=1
BATHYMETRIC_CAE_GRID_SIZE=256
BATHYMETRIC_CAE_ENABLE_EXPERT_REVIEW=false
BATHYMETRIC_CAE_LOG_LEVEL=DEBUG
```

### Production Environment

```bash
# .env.production
BATHYMETRIC_CAE_EPOCHS=200
BATHYMETRIC_CAE_ENSEMBLE_SIZE=5
BATHYMETRIC_CAE_GRID_SIZE=1024
BATHYMETRIC_CAE_ENABLE_EXPERT_REVIEW=true
BATHYMETRIC_CAE_QUALITY_THRESHOLD=0.85
BATHYMETRIC_CAE_LOG_LEVEL=INFO
```

### Testing Environment

```bash
# .env.testing
BATHYMETRIC_CAE_EPOCHS=5
BATHYMETRIC_CAE_ENSEMBLE_SIZE=1
BATHYMETRIC_CAE_GRID_SIZE=128
BATHYMETRIC_CAE_BATCH_SIZE=1
BATHYMETRIC_CAE_ENABLE_ALL_FEATURES=false
```

## 📊 Configuration Validation

### Automatic Validation

The system automatically validates configurations:

```python
config = Config(
    epochs=0,              # Invalid: must be positive
    batch_size=-1,         # Invalid: must be positive
    validation_split=1.5,  # Invalid: must be 0-1
    ensemble_size=0        # Invalid: must be at least 1
)

try:
    config.validate()
except ValueError as e:
    print(f"Configuration error: {e}")
    # Output: Configuration errors: epochs must be positive, batch_size must be positive, ...
```

### Custom Validation

```python
def validate_production_config(config):
    """Custom validation for production configurations."""
    errors = []
    
    # Production-specific checks
    if config.epochs < 100:
        errors.append("Production requires at least 100 epochs")
    
    if config.ensemble_size < 3:
        errors.append("Production requires ensemble_size >= 3")
    
    if not config.enable_expert_review:
        errors.append("Production requires expert review")
    
    if config.quality_threshold < 0.8:
        errors.append("Production requires quality_threshold >= 0.8")
    
    if errors:
        raise ValueError(f"Production validation failed: {'; '.join(errors)}")

# Usage
try:
    validate_production_config(config)
    print("✅ Configuration valid for production")
except ValueError as e:
    print(f"❌ {e}")
```

## 🔄 Configuration Management

### Configuration Inheritance

```python
# Base configuration
base_config = Config(
    epochs=100,
    batch_size=8,
    enable_adaptive_processing=True
)

# Inherit and override for specific use case
high_quality_config = Config.load_from_base(
    base_config,
    epochs=200,              # Override epochs
    ensemble_size=5,         # Override ensemble size
    quality_threshold=0.9    # Override quality threshold
)
```

### Configuration Versioning

```python
# Save configuration with version info
config_with_version = {
    "version": "2.0.0",
    "created_date": "2024-01-15",
    "description": "Production configuration for Survey 2024-Q1",
    "configuration": config.to_dict()
}

with open("production_v2.json", "w") as f:
    json.dump(config_with_version, f, indent=2)
```

### Configuration Profiles

```python
class ConfigurationProfiles:
    """Predefined configuration profiles."""
    
    @staticmethod
    def development():
        return Config(
            epochs=25,
            ensemble_size=1,
            grid_size=256,
            enable_expert_review=False
        )
    
    @staticmethod
    def production():
        return Config(
            epochs=200,
            ensemble_size=5,
            grid_size=1024,
            enable_expert_review=True,
            quality_threshold=0.85
        )
    
    @staticmethod
    def research():
        return Config(
            epochs=300,
            ensemble_size=7,
            grid_size=1024,
            quality_threshold=0.9
        )
    
    @staticmethod
    def testing():
        return Config(
            epochs=5,
            ensemble_size=1,
            grid_size=128,
            quality_threshold=0.3
        )

# Usage
config = ConfigurationProfiles.production()
```

## ⚡ Performance Optimization

### CPU Optimization

```python
cpu_optimized_config = Config(
    # Use all available CPU cores
    max_workers=-1,
    
    # Optimize batch size for CPU
    batch_size=4,
    
    # Disable GPU-specific optimizations
    use_mixed_precision=False,
    
    # CPU-friendly model size
    base_filters=24,
    grid_size=512
)
```

### GPU Optimization

```python
gpu_optimized_config = Config(
    # Larger batch size for GPU
    batch_size=16,
    
    # Enable GPU optimizations
    use_mixed_precision=True,
    gpu_memory_growth=True,
    
    # GPU-friendly model size
    base_filters=48,
    grid_size=1024,
    
    # Larger ensemble for GPU power
    ensemble_size=5
)
```

### Memory Optimization

```python
memory_optimized_config = Config(
    # Minimal memory usage
    batch_size=1,
    grid_size=256,
    ensemble_size=1,
    
    # Single worker to reduce memory overhead
    max_workers=1,
    
    # Disable memory-intensive features
    use_mixed_precision=False,
    
    # Smaller model
    base_filters=16,
    depth=3
)
```

## 🧪 Testing Configurations

### Unit Test Configuration

```python
unit_test_config = Config(
    epochs=2,
    batch_size=1,
    ensemble_size=1,
    grid_size=64,
    enable_adaptive_processing=True,
    enable_expert_review=False,
    enable_constitutional_constraints=True,
    quality_threshold=0.1
)
```

### Integration Test Configuration

```python
integration_test_config = Config(
    epochs=5,
    batch_size=2,
    ensemble_size=2,
    grid_size=128,
    enable_adaptive_processing=True,
    enable_expert_review=True,
    enable_constitutional_constraints=True,
    quality_threshold=0.5
)
```

### Performance Test Configuration

```python
performance_test_config = Config(
    epochs=10,
    batch_size=8,
    ensemble_size=3,
    grid_size=256,
    enable_adaptive_processing=True,
    enable_expert_review=False,
    enable_constitutional_constraints=True
)
```

## 📋 Configuration Best Practices

### 1. Start with Defaults

```python
# Begin with default configuration
config = Config()

# Gradually customize as needed
config.epochs = 150
config.ensemble_size = 3
config.enable_adaptive_processing = True
```

### 2. Use Environment-Specific Configs

```python
import os

# Load configuration based on environment
env = os.getenv("ENVIRONMENT", "development")

if env == "production":
    config = ConfigurationProfiles.production()
elif env == "testing":
    config = ConfigurationProfiles.testing()
else:
    config = ConfigurationProfiles.development()
```

### 3. Document Your Configurations

```json
{
  "_metadata": {
    "name": "Survey Processing Configuration",
    "version": "1.2.0",
    "created_by": "Hydrographic Team",
    "created_date": "2024-01-15",
    "description": "Optimized for shallow water multibeam surveys",
    "use_case": "High-resolution coastal mapping",
    "quality_requirements": "IHO S-44 Special Order"
  },
  
  "epochs": 180,
  "ensemble_size": 4,
  "quality_threshold": 0.88,
  "enable_adaptive_processing": true
}
```

### 4. Validate Before Use

```python
def safe_config_load(config_path):
    """Safely load and validate configuration."""
    try:
        config = Config.load(config_path)
        config.validate()
        return config
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Using default configuration instead")
        return Config()

# Usage
config = safe_config_load("my_config.json")
```

## 🚨 Common Configuration Issues

### Issue: Memory Errors

**Problem**: Out of memory during processing

**Solution**:
```python
# Reduce memory usage
config = Config(
    batch_size=1,          # Reduce batch size
    grid_size=256,         # Reduce grid size
    ensemble_size=1,       # Single model
    max_workers=1          # Single worker
)
```

### Issue: Slow Processing

**Problem**: Processing takes too long

**Solution**:
```python
# Speed up processing
config = Config(
    epochs=50,             # Fewer epochs
    ensemble_size=1,       # Single model
    grid_size=256,         # Lower resolution
    batch_size=16          # Larger batches (if memory allows)
)
```

### Issue: Poor Quality Results

**Problem**: Quality metrics are consistently low

**Solution**:
```python
# Improve quality
config = Config(
    epochs=200,            # More training
    ensemble_size=5,       # More models
    grid_size=1024,        # Higher resolution
    enable_adaptive_processing=True,
    enable_constitutional_constraints=True,
    quality_threshold=0.85
)
```

---

## 📚 Configuration Reference

For complete configuration options, see:
- **[Configuration Reference](../reference/configuration-reference.md)** - All available options
- **[API Documentation](../api/config.md)** - Programmatic configuration
- **[Performance Tuning](../user-guide/performance-tuning.md)** - Optimization guide

**Next Steps**: Try the **[Basic Usage Guide](basic-usage.md)** to see configurations in action.