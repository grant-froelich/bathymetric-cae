# Usage Guide

This comprehensive guide covers all aspects of using the Bathymetric CAE package for processing bathymetric data with advanced Convolutional Autoencoders.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command Line Interface](#command-line-interface)
3. [Python API](#python-api)
4. [Configuration](#configuration)
5. [Data Formats](#data-formats)
6. [Processing Workflows](#processing-workflows)
7. [Visualization](#visualization)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Quick Start

### Basic Processing

The simplest way to process bathymetric data:

```bash
# Basic processing with default settings
bathymetric-cae --input /path/to/input --output /path/to/output

# Custom training parameters
bathymetric-cae \
  --input ./data/bathymetry \
  --output ./results \
  --epochs 100 \
  --batch-size 8 \
  --grid-size 512
```

### Python API

```python
from bathymetric_cae import BathymetricCAEPipeline, Config

# Create configuration
config = Config(
    input_folder="./data/input",
    output_folder="./data/output",
    epochs=50,
    batch_size=8
)

# Run pipeline
pipeline = BathymetricCAEPipeline(config)
results = pipeline.run(
    input_folder=config.input_folder,
    output_folder=config.output_folder,
    model_path="my_model.h5"
)

print(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
```

## Command Line Interface

### Basic Commands

```bash
# Process with default settings
bathymetric-cae

# Custom input/output paths
bathymetric-cae --input /data/bathymetry --output /results

# Load configuration file
bathymetric-cae --config my_config.json

# Save current configuration
bathymetric-cae --save-config my_settings.json
```

### Training Parameters

```bash
# Custom training settings
bathymetric-cae \
  --epochs 200 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --validation-split 0.2

# Model architecture
bathymetric-cae \
  --grid-size 1024 \
  --base-filters 64 \
  --depth 5 \
  --dropout-rate 0.15

# Force model retraining
bathymetric-cae --force-retrain
```

### Processing Options

```bash
# Disable GPU (use CPU only)
bathymetric-cae --no-gpu

# Verbose output for debugging
bathymetric-cae --verbose

# Silent mode (minimal output)
bathymetric-cae --silent

# Disable plot generation
bathymetric-cae --no-plots
```

### Single File Processing

```bash
# Process single file interactively
bathymetric-cae \
  --single-file input.bag \
  --model trained_model.h5 \
  --show-plots

# Process with custom output location
bathymetric-cae \
  --single-file data/sample.tif \
  --model models/my_model.h5 \
  --output results/processed_sample.tif
```

### System Validation

```bash
# Validate installation and requirements
bathymetric-cae --validate-requirements

# Check system capabilities
python -c "from bathymetric_cae import check_gpu_availability; print(check_gpu_availability())"
```

## Python API

### Basic Usage

```python
from bathymetric_cae import BathymetricCAEPipeline, Config, setup_logging

# Setup logging
setup_logging(log_level='INFO')

# Create configuration
config = Config(
    input_folder="./input_data",
    output_folder="./output_data",
    model_path="bathymetric_model.h5",
    epochs=100,
    batch_size=8,
    grid_size=512
)

# Create and run pipeline
pipeline = BathymetricCAEPipeline(config)
results = pipeline.run(
    input_folder=config.input_folder,
    output_folder=config.output_folder,
    model_path=config.model_path
)

# Access results
print(f"Total files processed: {results['pipeline_info']['total_files']}")
print(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
print(f"Mean SSIM: {results['summary_statistics']['mean_ssim']:.4f}")
```

### Advanced Configuration

```python
from bathymetric_cae import Config

# High-performance configuration
high_perf_config = Config(
    # Training parameters
    epochs=200,
    batch_size=32,
    learning_rate=0.0001,
    validation_split=0.15,
    
    # Model architecture
    grid_size=1024,
    base_filters=64,
    depth=6,
    dropout_rate=0.15,
    
    # Performance settings
    gpu_memory_growth=True,
    use_mixed_precision=True,
    max_workers=8,
    
    # Callbacks
    early_stopping_patience=25,
    reduce_lr_patience=12,
    reduce_lr_factor=0.3
)

# Memory-efficient configuration
memory_config = Config(
    grid_size=256,
    batch_size=4,
    base_filters=16,
    depth=3,
    max_workers=2
)
```

### Single File Processing

```python
from bathymetric_cae import BathymetricCAEPipeline, Config

config = Config()
pipeline = BathymetricCAEPipeline(config)

# Interactive processing with visualization
results = pipeline.process_single_file_interactive(
    file_path="sample_bathymetry.bag",
    model_path="trained_model.h5",
    show_plots=True
)

if results['processing_successful']:
    print(f"SSIM improvement: {results['ssim']:.4f}")
    print(f"Has uncertainty data: {results['has_uncertainty']}")
else:
    print(f"Processing failed: {results['error']}")
```

### Custom Model Creation

```python
from bathymetric_cae import AdvancedCAE, Config

# Custom model configuration
config = Config(
    grid_size=256,
    base_filters=32,
    depth=4,
    dropout_rate=0.2
)

# Create model builder
model_builder = AdvancedCAE(config)

# Create model for depth + uncertainty data
model = model_builder.create_model(channels=2)

# Get model summary
print(f"Model parameters: {model.count_params():,}")
summary = model_builder.get_model_summary(model)
print(summary)
```

### Data Processing

```python
from bathymetric_cae import BathymetricProcessor, Config

config = Config(grid_size=512)
processor = BathymetricProcessor(config)

# Process single file
input_data, original_shape, metadata = processor.preprocess_bathymetric_grid(
    "bathymetry_file.bag"
)

print(f"Processed shape: {input_data.shape}")
print(f"Original shape: {original_shape}")
print(f"Has uncertainty: {input_data.shape[-1] > 1}")

# Batch processing
file_paths = ["file1.bag", "file2.tif", "file3.asc"]
results = processor.batch_preprocess_files(file_paths)
print(f"Successfully processed: {len(results)} files")
```

### Visualization

```python
from bathymetric_cae import Visualizer
import numpy as np

visualizer = Visualizer()

# Assuming you have original, cleaned, and uncertainty data
visualizer.plot_comparison(
    original=original_data,
    cleaned=cleaned_data,
    uncertainty=uncertainty_data,
    filename="comparison.png",
    show_plot=True
)

# Plot difference map
visualizer.plot_difference_map(
    original=original_data,
    cleaned=cleaned_data,
    filename="difference.png"
)

# Data distribution analysis
data_dict = {
    "Original": original_data,
    "Cleaned": cleaned_data
}
visualizer.plot_data_distribution(
    data_dict=data_dict,
    filename="distribution.png"
)
```

## Configuration

### Configuration File Format

```json
{
  "input_folder": "./data/input",
  "output_folder": "./data/output",
  "model_path": "bathymetric_model.h5",
  "epochs": 100,
  "batch_size": 8,
  "learning_rate": 0.001,
  "grid_size": 512,
  "base_filters": 32,
  "depth": 4,
  "dropout_rate": 0.2,
  "early_stopping_patience": 15,
  "gpu_memory_growth": true,
  "use_mixed_precision": true,
  "log_level": "INFO"
}
```

### Loading and Saving Configurations

```python
from bathymetric_cae import Config

# Load from file
config = Config.load("my_config.json")

# Modify configuration
updated_config = config.update(
    epochs=150,
    batch_size=16,
    learning_rate=0.0005
)

# Save configuration
updated_config.save("updated_config.json")

# Create from command line args
from argparse import Namespace
args = Namespace(epochs=200, batch_size=32, learning_rate=0.0001)
config_from_args = Config.create_from_args(args)
```

### Configuration Validation

```python
try:
    config = Config(
        epochs=100,
        batch_size=8,
        learning_rate=0.001
    )
    config.validate()  # Explicit validation
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Data Formats

### Supported Formats

| Format | Extension | Description | Uncertainty Support |
|--------|-----------|-------------|-------------------|
| BAG | .bag | Bathymetric Attributed Grid | ✅ Native support |
| GeoTIFF | .tif, .tiff | Tagged Image File Format | ❌ Single band only |
| ASCII Grid | .asc | ESRI ASCII Grid format | ❌ Single band only |
| XYZ | .xyz | Point cloud format | ❌ Single band only |

### BAG Files (Recommended)

BAG files provide the best support with native uncertainty handling:

```python
# BAG files automatically extract both depth and uncertainty
input_data, shape, metadata = processor.preprocess_bathymetric_grid("data.bag")
print(f"Channels: {input_data.shape[-1]}")  # Usually 2 (depth + uncertainty)
```

### File Validation

```python
from bathymetric_cae import BathymetricProcessor, get_valid_files

# Check supported formats
supported_formats = get_valid_files("/path/to/data", ['.bag', '.tif', '.asc'])
print(f"Found {len(supported_formats)} supported files")

# Validate individual file
processor = BathymetricProcessor(config)
is_supported = processor.validate_file_format("bathymetry.bag")

# Get detailed file information
file_info = processor.get_file_info("bathymetry.bag")
print(file_info)
```

## Processing Workflows

### Batch Processing Workflow

```python
from bathymetric_cae import BathymetricCAEPipeline, Config, setup_logging

def batch_processing_workflow():
    # Setup
    setup_logging(log_level='INFO')
    
    config = Config(
        input_folder="./large_dataset",
        output_folder="./processed_results",
        model_path="production_model.h5",
        epochs=150,
        batch_size=16,
        grid_size=512
    )
    
    # Create pipeline
    pipeline = BathymetricCAEPipeline(config)
    
    # Run processing
    results = pipeline.run(
        input_folder=config.input_folder,
        output_folder=config.output_folder,
        model_path=config.model_path
    )
    
    # Generate report
    print("Processing Summary:")
    print(f"Total files: {results['pipeline_info']['total_files']}")
    print(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
    
    if results['summary_statistics']:
        stats = results['summary_statistics']
        print(f"Quality metrics:")
        print(f"  Mean SSIM: {stats['mean_ssim']:.4f}")
        print(f"  SSIM range: {stats['min_ssim']:.4f} - {stats['max_ssim']:.4f}")
    
    return results

# Run workflow
results = batch_processing_workflow()
```

### Progressive Training Workflow

```python
def progressive_training_workflow():
    """Train model with progressively increasing complexity."""
    
    # Phase 1: Quick training with small resolution
    phase1_config = Config(
        grid_size=128,
        base_filters=16,
        depth=2,
        epochs=50,
        batch_size=16
    )
    
    # Phase 2: Medium resolution training
    phase2_config = phase1_config.update(
        grid_size=256,
        base_filters=32,
        depth=3,
        epochs=100
    )
    
    # Phase 3: High resolution training
    phase3_config = phase2_config.update(
        grid_size=512,
        base_filters=64,
        depth=4,
        epochs=150
    )
    
    for i, config in enumerate([phase1_config, phase2_config, phase3_config], 1):
        print(f"Training Phase {i}")
        pipeline = BathymetricCAEPipeline(config)
        
        results = pipeline.run(
            input_folder="./training_data",
            output_folder=f"./phase_{i}_results",
            model_path=f"model_phase_{i}.h5"
        )
        
        print(f"Phase {i} completed - Success rate: {results['pipeline_info']['success_rate']:.1f}%")
```

### Quality Assessment Workflow

```python
def quality_assessment_workflow():
    """Comprehensive quality assessment of processed results."""
    
    from bathymetric_cae import Visualizer
    
    # Load results
    results = pipeline.run(input_folder, output_folder, model_path)
    
    # Quality analysis
    processing_stats = results['processing_stats']
    ssim_scores = [s['ssim'] for s in processing_stats if 'ssim' in s]
    
    if ssim_scores:
        import numpy as np
        print("Quality Assessment:")
        print(f"  Mean SSIM: {np.mean(ssim_scores):.4f}")
        print(f"  Std SSIM: {np.std(ssim_scores):.4f}")
        print(f"  Min SSIM: {np.min(ssim_scores):.4f}")
        print(f"  Max SSIM: {np.max(ssim_scores):.4f}")
        
        # Categorize quality
        excellent = sum(1 for s in ssim_scores if s > 0.9)
        good = sum(1 for s in ssim_scores if 0.8 < s <= 0.9)
        fair = sum(1 for s in ssim_scores if 0.7 < s <= 0.8)
        poor = sum(1 for s in ssim_scores if s <= 0.7)
        
        print(f"\nQuality Distribution:")
        print(f"  Excellent (>0.9): {excellent}")
        print(f"  Good (0.8-0.9): {good}")
        print(f"  Fair (0.7-0.8): {fair}")
        print(f"  Poor (≤0.7): {poor}")
        
        # Create quality visualization
        visualizer = Visualizer()
        visualizer.plot_processing_summary(
            processing_stats,
            filename="quality_assessment.png"
        )
```

## Visualization

### Training Progress Visualization

```python
from bathymetric_cae import Visualizer

# Assuming you have a training history object
visualizer = Visualizer()

# Plot comprehensive training history
visualizer.plot_training_history(
    history=training_history,
    save_path="training_progress.png",
    show_plot=True
)
```

### Data Comparison Visualization

```python
# Compare original and processed data
visualizer.plot_comparison(
    original=original_bathymetry,
    cleaned=processed_bathymetry,
    uncertainty=uncertainty_data,  # Optional
    filename="data_comparison.png",
    title="Bathymetry Processing Results"
)

# Difference analysis
visualizer.plot_difference_map(
    original=original_bathymetry,
    cleaned=processed_bathymetry,
    filename="difference_analysis.png"
)
```

### Statistical Analysis Visualization

```python
# Multi-dataset comparison
data_dict = {
    "Original": original_data,
    "Processed": processed_data,
    "Uncertainty": uncertainty_data
}

visualizer.plot_data_distribution(
    data_dict=data_dict,
    filename="statistical_analysis.png"
)
```

### Custom Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def create_custom_visualization(data_dict, output_path):
    """Create custom visualization for specific needs."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    im1 = axes[0, 0].imshow(data_dict['original'], cmap='viridis')
    axes[0, 0].set_title('Original Bathymetry')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Processed data
    im2 = axes[0, 1].imshow(data_dict['processed'], cmap='viridis')
    axes[0, 1].set_title('Processed Bathymetry')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference
    diff = data_dict['processed'] - data_dict['original']
    im3 = axes[1, 0].imshow(diff, cmap='RdBu_r')
    axes[1, 0].set_title('Difference')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Statistics
    axes[1, 1].hist(diff.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Difference Distribution')
    axes[1, 1].set_xlabel('Difference Value')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
```

## Performance Optimization

### Memory Optimization

```python
from bathymetric_cae import get_memory_info, memory_monitor

# Monitor memory usage
with memory_monitor("Processing operation"):
    # Your processing code here
    results = pipeline.run(input_folder, output_folder, model_path)

# Check memory status
memory_info = get_memory_info()
print(f"Available memory: {memory_info['available_mb']:.0f} MB")

# Optimize configuration for available memory
def optimize_for_memory(available_mb):
    if available_mb > 16000:  # > 16GB
        return Config(batch_size=32, grid_size=1024, base_filters=64)
    elif available_mb > 8000:  # > 8GB
        return Config(batch_size=16, grid_size=512, base_filters=32)
    else:  # < 8GB
        return Config(batch_size=4, grid_size=256, base_filters=16)

optimal_config = optimize_for_memory(memory_info['available_mb'])
```

### GPU Optimization

```python
from bathymetric_cae import check_gpu_availability, optimize_gpu_performance

# Check GPU status
gpu_info = check_gpu_availability()
print(f"GPU available: {gpu_info['gpu_available']}")
print(f"Number of GPUs: {gpu_info['num_physical_gpus']}")

# Optimize GPU performance
if gpu_info['gpu_available']:
    optimization_results = optimize_gpu_performance()
    print(f"Mixed precision enabled: {optimization_results['mixed_precision']}")
    print(f"Memory growth configured: {optimization_results['memory_growth']}")
```

### Parallel Processing

```python
# Configure parallel processing
config = Config(
    max_workers=4,  # Adjust based on CPU cores
    batch_size=8,   # Balance with memory
    use_mixed_precision=True  # GPU optimization
)

# For CPU-only processing
cpu_config = Config(
    max_workers=2,  # Fewer workers for CPU
    batch_size=4,   # Smaller batches
    use_mixed_precision=False  # Not applicable for CPU
)
```

### Batch Size Optimization

```python
def find_optimal_batch_size(config, test_data_shape):
    """Find optimal batch size through testing."""
    
    from bathymetric_cae import AdvancedCAE
    
    batch_sizes = [2, 4, 8, 16, 32]
    optimal_size = 2
    
    for batch_size in batch_sizes:
        try:
            test_config = config.update(batch_size=batch_size)
            model_builder = AdvancedCAE(test_config)
            model = model_builder.create_model(channels=test_data_shape[-1])
            
            # Test with dummy data
            import numpy as np
            test_batch = np.random.rand(batch_size, *test_data_shape)
            model.predict(test_batch, verbose=0)
            
            optimal_size = batch_size
            print(f"Batch size {batch_size}: OK")
            
        except Exception as e:
            print(f"Batch size {batch_size}: Failed ({e})")
            break
    
    return optimal_size

# Find optimal batch size
optimal_batch = find_optimal_batch_size(config, (512, 512, 2))
config = config.update(batch_size=optimal_batch)
```

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory Errors

```python
# Solution 1: Reduce batch size and grid size
memory_efficient_config = Config(
    batch_size=2,
    grid_size=256,
    base_filters=16,
    depth=3
)

# Solution 2: Enable GPU memory growth
config = Config(gpu_memory_growth=True)

# Solution 3: Use CPU processing
# Add --no-gpu flag or:
from bathymetric_cae import disable_gpu
disable_gpu()
```

#### GDAL Import Errors

```bash
# Install GDAL using conda (recommended)
conda install -c conda-forge gdal

# Or check system installation
gdal-config --version

# Verify in Python
python -c "from osgeo import gdal; print('GDAL OK')"
```

#### TensorFlow GPU Issues

```python
# Check GPU availability
import tensorflow as tf
print("GPUs available:", len(tf.config.list_physical_devices('GPU')))

# Check CUDA installation
print("CUDA available:", tf.test.is_built_with_cuda())

# Force CPU if GPU issues persist
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

#### File Format Issues

```python
# Validate file format
from bathymetric_cae import BathymetricProcessor

processor = BathymetricProcessor(config)
file_info = processor.get_file_info("problematic_file.bag")

if 'error' in file_info:
    print(f"File error: {file_info['error']}")
else:
    print(f"File info: {file_info}")
```

#### Model Loading Issues

```python
# Check model file
from pathlib import Path

model_path = Path("my_model.h5")
if not model_path.exists():
    print("Model file not found - train a new model")
elif model_path.stat().st_size == 0:
    print("Model file is empty - retrain required")
else:
    print(f"Model file size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
```

### Debug Mode

```python
# Enable verbose logging
from bathymetric_cae import setup_logging

setup_logging(log_level='DEBUG', console_output=True)

# Or use command line
# bathymetric-cae --verbose
```

### Performance Debugging

```python
import time
from bathymetric_cae import memory_monitor

def debug_performance():
    """Debug performance issues step by step."""
    
    print("=== Performance Debugging ===")
    
    # Check system resources
    memory_info = get_memory_info()
    print(f"Available memory: {memory_info['available_mb']:.0f} MB")
    
    gpu_info = check_gpu_availability()
    print(f"GPU available: {gpu_info['gpu_available']}")
    
    # Time each step
    with memory_monitor("Data loading"):
        start = time.time()
        # Your data loading code
        load_time = time.time() - start
        print(f"Data loading: {load_time:.2f} seconds")
    
    with memory_monitor("Model creation"):
        start = time.time()
        # Your model creation code
        model_time = time.time() - start
        print(f"Model creation: {model_time:.2f} seconds")
    
    # Continue for other steps...

debug_performance()
```

## Best Practices

### Data Preparation

1. **File Organization**
   ```
   project/
   ├── data/
   │   ├── input/           # Original bathymetric files
   │   ├── output/          # Processed results
   │   └── validation/      # Validation dataset
   ├── models/              # Trained models
   ├── configs/             # Configuration files
   ├── logs/                # Training logs
   └── reports/             # Processing reports
   ```

2. **Data Quality Checks**
   ```python
   def validate_input_data(input_folder):
       """Validate input data quality."""
       from bathymetric_cae import get_valid_files
       
       files = get_valid_files(input_folder, ['.bag', '.tif', '.asc'])
       
       print(f"Found {len(files)} files")
       
       for file_path in files:
           info = processor.get_file_info(file_path)
           if 'error' in info:
               print(f"⚠️  {file_path.name}: {info['error']}")
           else:
               print(f"✅ {file_path.name}: {info['width']}x{info['height']}")
   ```

### Model Training

1. **Progressive Training Strategy**
   - Start with small models and low resolution
   - Gradually increase complexity
   - Use transfer learning when possible

2. **Validation Strategy**
   ```python
   # Use temporal or spatial splits, not random
   config = Config(
       validation_split=0.2,
       early_stopping_patience=15,
       reduce_lr_patience=10
   )
   ```

3. **Model Checkpointing**
   ```python
   # Save intermediate models
   config = Config(
       model_path="models/bathymetric_v1.h5",
       # Callbacks will save best model automatically
   )
   ```

### Production Deployment

1. **Configuration Management**
   ```python
   # Use environment-specific configs
   production_config = Config.load("configs/production.json")
   development_config = Config.load("configs/development.json")
   
   # Validate before deployment
   production_config.validate()
   ```

2. **Error Handling**
   ```python
   def robust_processing(input_folder, output_folder):
       """Robust processing with error recovery."""
       try:
           results = pipeline.run(input_folder, output_folder, model_path)
           return results
       except Exception as e:
           logging.error(f"Processing failed: {e}")
           # Implement fallback strategy
           return fallback_processing(input_folder, output_folder)
   ```

3. **Monitoring and Logging**
   ```python
   # Production logging setup
   setup_logging(
       log_level='INFO',
       log_file=f'production_{datetime.now().strftime("%Y%m%d")}.log',
       console_output=False,
       file_output=True
   )
   ```

### Quality Assurance

1. **Validation Metrics**
   ```python
   # Define quality thresholds
   QUALITY_THRESHOLDS = {
       'min_ssim': 0.7,
       'max_processing_time': 300,  # seconds
       'min_success_rate': 0.95
   }
   
   def validate_results(results):
       stats = results['summary_statistics']
       return (
           stats['mean_ssim'] >= QUALITY_THRESHOLDS['min_ssim'] and
           results['pipeline_info']['success_rate'] >= QUALITY_THRESHOLDS['min_success_rate']
       )
   ```

2. **Automated Testing**
   ```python
   def run_integration_test():
       """Run integration test with sample data."""
       test_config = Config(
           input_folder="test_data/",
           output_folder="test_results/",
           epochs=5,  # Minimal for testing
           batch_size=2
       )
       
       pipeline = BathymetricCAEPipeline(test_config)
       results = pipeline.run(
           input_folder=test_config.input_folder,
           output_folder=test_config.output_folder,
           model_path="test_model.h5"
       )
       
       assert results['pipeline_info']['success_rate'] > 0.8
       print("✅ Integration test passed")
   ```

### Performance Optimization

1. **Resource Management**
   ```python
   # Monitor and optimize resource usage
   def optimize_resources():
       memory_info = get_memory_info()
       gpu_info = check_gpu_availability()
       
       # Adjust configuration based on available resources
       if memory_info['available_mb'] < 4000:  # Less than 4GB
           return Config(batch_size=2, grid_size=256)
       elif gpu_info['gpu_available']:
           return Config(batch_size=16, use_mixed_precision=True)
       else:
           return Config(batch_size=8, max_workers=4)
   ```

2. **Batch Processing Optimization**
   ```python
   # Process files in optimal batch sizes
   def process_in_batches(file_list, batch_size=10):
       """Process files in manageable batches."""
       for i in range(0, len(file_list), batch_size):
           batch = file_list[i:i+batch_size]
           batch_num = i//batch_size + 1
           total_batches = (len(file_list) + batch_size - 1) // batch_size
           
           print(f"Processing batch {batch_num}/{total_batches}")
           
           try:
               # Process batch
               for file_path in batch:
                   process_single_file(file_path)
               
               # Memory cleanup between batches
               force_garbage_collection()
               
           except Exception as e:
               print(f"Batch {batch_num} failed: {e}")
               continue
   ```

3. **Caching and Reuse**
   ```python
   # Cache preprocessed data for reuse
   import pickle
   from pathlib import Path
   
   def cache_preprocessed_data(file_path, data, cache_dir="cache/"):
       """Cache preprocessed data to avoid reprocessing."""
       cache_path = Path(cache_dir)
       cache_path.mkdir(exist_ok=True)
       
       cache_file = cache_path / f"{Path(file_path).stem}_preprocessed.pkl"
       
       with open(cache_file, 'wb') as f:
           pickle.dump(data, f)
   
   def load_cached_data(file_path, cache_dir="cache/"):
       """Load cached preprocessed data if available."""
       cache_file = Path(cache_dir) / f"{Path(file_path).stem}_preprocessed.pkl"
       
       if cache_file.exists():
           with open(cache_file, 'rb') as f:
               return pickle.load(f)
       return None
   ```

## Advanced Usage Patterns

### Multi-Model Ensemble

```python
def create_ensemble_predictor(model_paths, weights=None):
    """Create ensemble predictor from multiple models."""
    from bathymetric_cae import AdvancedCAE
    
    models = []
    for path in model_paths:
        model_builder = AdvancedCAE(config)
        model = model_builder.load_model(path)
        models.append(model)
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    def ensemble_predict(input_data):
        predictions = []
        for model in models:
            pred = model.predict(input_data, verbose=0)
            predictions.append(pred)
        
        # Weighted average
        import numpy as np
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    return ensemble_predict

# Usage
ensemble = create_ensemble_predictor([
    "model_v1.h5", 
    "model_v2.h5", 
    "model_v3.h5"
], weights=[0.4, 0.35, 0.25])

result = ensemble(input_bathymetric_data)
```

### Transfer Learning

```python
def fine_tune_model(base_model_path, new_data_path, output_model_path):
    """Fine-tune existing model on new data."""
    
    # Load base model
    model_builder = AdvancedCAE(config)
    base_model = model_builder.load_model(base_model_path)
    
    # Freeze early layers
    for layer in base_model.layers[:-5]:  # Freeze all but last 5 layers
        layer.trainable = False
    
    # Recompile with lower learning rate
    import tensorflow as tf
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=model_builder._combined_loss,
        metrics=['mae', model_builder._ssim_metric]
    )
    
    # Fine-tune on new data
    # (Load your new training data here)
    history = base_model.fit(
        new_X_train, new_y_train,
        epochs=50,  # Fewer epochs for fine-tuning
        validation_split=0.2,
        callbacks=model_builder.create_callbacks(output_model_path)
    )
    
    return base_model, history

# Usage
fine_tuned_model, history = fine_tune_model(
    "base_model.h5",
    "new_training_data/",
    "fine_tuned_model.h5"
)
```

### Real-time Processing

```python
def setup_real_time_processor():
    """Setup processor for real-time data streams."""
    
    class RealTimeProcessor:
        def __init__(self, model_path, config):
            self.model_builder = AdvancedCAE(config)
            self.model = self.model_builder.load_model(model_path)
            self.processor = BathymetricProcessor(config)
            
        def process_stream(self, data_stream):
            """Process streaming bathymetric data."""
            for data_chunk in data_stream:
                try:
                    # Preprocess chunk
                    processed_chunk = self.processor._robust_normalize(data_chunk)
                    processed_chunk = np.expand_dims(processed_chunk, axis=(0, -1))
                    
                    # Run inference
                    result = self.model.predict(processed_chunk, verbose=0)
                    
                    yield result[0, :, :, 0]  # Remove batch and channel dims
                    
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
    
    return RealTimeProcessor(model_path, config)

# Usage
rt_processor = setup_real_time_processor()
for processed_chunk in rt_processor.process_stream(data_stream):
    # Handle processed data in real-time
    handle_result(processed_chunk)
```

### Custom Loss Functions

```python
def create_custom_loss(depth_weight=1.0, uncertainty_weight=0.5):
    """Create custom loss function for specific requirements."""
    
    import tensorflow as tf
    
    def custom_bathymetric_loss(y_true, y_pred):
        # MSE component
        mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        
        # SSIM component
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        
        # Gradient preservation component
        def gradient_loss(true, pred):
            true_grad_x = tf.abs(true[:, 1:, :, :] - true[:, :-1, :, :])
            pred_grad_x = tf.abs(pred[:, 1:, :, :] - pred[:, :-1, :, :])
            
            true_grad_y = tf.abs(true[:, :, 1:, :] - true[:, :, :-1, :])
            pred_grad_y = tf.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            
            grad_diff_x = tf.reduce_mean(tf.abs(true_grad_x - pred_grad_x))
            grad_diff_y = tf.reduce_mean(tf.abs(true_grad_y - pred_grad_y))
            
            return grad_diff_x + grad_diff_y
        
        grad_loss = gradient_loss(y_true, y_pred)
        
        # Combine losses
        total_loss = (depth_weight * mse + 
                     uncertainty_weight * ssim_loss + 
                     0.1 * grad_loss)
        
        return total_loss
    
    return custom_bathymetric_loss

# Usage in model
def create_model_with_custom_loss():
    model_builder = AdvancedCAE(config)
    model = model_builder.create_model(channels=2)
    
    # Use custom loss
    custom_loss = create_custom_loss(depth_weight=1.0, uncertainty_weight=0.3)
    
    model.compile(
        optimizer='adam',
        loss=custom_loss,
        metrics=['mae']
    )
    
    return model
```

### Automated Hyperparameter Tuning

```python
def hyperparameter_search(search_space, max_trials=20):
    """Automated hyperparameter optimization."""
    
    import itertools
    import numpy as np
    
    best_score = 0
    best_params = None
    results = []
    
    # Generate parameter combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))
        
        print(f"Testing parameters: {params}")
        
        try:
            # Create config with current parameters
            test_config = Config(**params)
            
            # Quick training run for evaluation
            pipeline = BathymetricCAEPipeline(test_config)
            results = pipeline.run(
                input_folder="validation_data/",
                output_folder="temp_results/",
                model_path="temp_model.h5"
            )
            
            # Evaluate performance
            score = results['summary_statistics']['mean_ssim']
            
            results.append({
                'params': params,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                
        except Exception as e:
            print(f"Parameter combination failed: {e}")
            continue
    
    return best_params, best_score, results

# Define search space
search_space = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [4, 8, 16],
    'base_filters': [16, 32, 64],
    'depth': [3, 4, 5],
    'dropout_rate': [0.1, 0.2, 0.3]
}

# Run hyperparameter search
best_params, best_score, all_results = hyperparameter_search(search_space)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

## Integration Examples

### GIS Integration

```python
def integrate_with_qgis():
    """Example integration with QGIS workflow."""
    
    import processing
    from qgis.core import QgsVectorLayer, QgsProject
    
    def process_bathymetry_layer(layer_path, output_path):
        # Export QGIS layer to supported format
        processing.run("gdal:translate", {
            'INPUT': layer_path,
            'OUTPUT': 'temp_bathymetry.tif',
            'OPTIONS': '',
            'DATA_TYPE': 5,  # Float32
            'COPY_SUBDATASETS': False,
            'NODATA': -9999
        })
        
        # Process with Bathymetric CAE
        config = Config(
            input_folder="./",
            output_folder="./",
            model_path="production_model.h5"
        )
        
        pipeline = BathymetricCAEPipeline(config)
        results = pipeline.process_single_file_interactive(
            file_path="temp_bathymetry.tif",
            model_path=config.model_path,
            output_path=output_path
        )
        
        # Load result back into QGIS
        if results['processing_successful']:
            result_layer = QgsVectorLayer(output_path, "Processed Bathymetry", "gdal")
            QgsProject.instance().addMapLayer(result_layer)
        
        return results
```

### Database Integration

```python
def integrate_with_database():
    """Example integration with spatial database."""
    
    import psycopg2
    import numpy as np
    from osgeo import gdal
    
    def process_from_database(connection_string, table_name, geom_column, depth_column):
        # Connect to database
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Query bathymetric data
        query = f"""
        SELECT ST_AsBinary({geom_column}), {depth_column}
        FROM {table_name}
        WHERE {depth_column} IS NOT NULL
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Convert to raster format
        # (Implementation depends on your specific database schema)
        
        # Process with Bathymetric CAE
        processed_results = []
        for geom, depth in results:
            # Convert geometry and depth to raster
            raster_data = convert_to_raster(geom, depth)
            
            # Process with model
            processed = process_raster_data(raster_data)
            processed_results.append(processed)
        
        return processed_results
```

### Cloud Processing Integration

```python
def setup_cloud_processing():
    """Example cloud processing setup."""
    
    def process_on_aws_batch(input_s3_path, output_s3_path, model_s3_path):
        """Process bathymetric data using AWS Batch."""
        
        import boto3
        
        # Download data from S3
        s3 = boto3.client('s3')
        s3.download_file('bucket', input_s3_path, 'local_input.bag')
        s3.download_file('model-bucket', model_s3_path, 'model.h5')
        
        # Process locally
        config = Config(
            input_folder="./",
            output_folder="./",
            model_path="model.h5"
        )
        
        pipeline = BathymetricCAEPipeline(config)
        results = pipeline.process_single_file_interactive(
            file_path="local_input.bag",
            model_path="model.h5",
            output_path="processed_output.tif"
        )
        
        # Upload results to S3
        if results['processing_successful']:
            s3.upload_file('processed_output.tif', 'output-bucket', output_s3_path)
        
        return results
    
    def process_on_google_cloud(project_id, input_path, output_path):
        """Process using Google Cloud Platform."""
        
        from google.cloud import storage
        
        # Similar implementation for GCP
        client = storage.Client(project=project_id)
        
        # Download, process, upload workflow
        # Implementation details depend on GCP services used
```

## Monitoring and Maintenance

### Health Checks

```python
def system_health_check():
    """Comprehensive system health check."""
    
    from bathymetric_cae import validate_pipeline_requirements
    
    print("=== System Health Check ===")
    
    # Check requirements
    requirements = validate_pipeline_requirements()
    all_good = requirements['all_requirements_met']
    
    if all_good:
        print("✅ All requirements satisfied")
    else:
        print("❌ Some requirements missing:")
        for req, status in requirements.items():
            if not status and req != 'all_requirements_met':
                print(f"  - {req}")
    
    # Check system resources
    memory_info = get_memory_info()
    print(f"💾 Available Memory: {memory_info['available_mb']:.0f} MB")
    
    if memory_info['percent'] > 85:
        print("⚠️  High memory usage detected")
    
    # Check GPU
    gpu_info = check_gpu_availability()
    if gpu_info['gpu_available']:
        print(f"🎮 GPU Available: {gpu_info['num_physical_gpus']} device(s)")
    else:
        print("⚠️  No GPU available - using CPU processing")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (2**30)
    
    if free_gb < 5:
        print(f"⚠️  Low disk space: {free_gb} GB free")
    else:
        print(f"💿 Disk Space: {free_gb} GB free")
    
    return all_good

# Run health check
health_status = system_health_check()
```

### Performance Monitoring

```python
def setup_performance_monitoring():
    """Setup comprehensive performance monitoring."""
    
    import time
    import psutil
    from collections import defaultdict
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = defaultdict(list)
            self.start_time = None
        
        def start_monitoring(self):
            self.start_time = time.time()
            
        def log_metric(self, name, value):
            self.metrics[name].append({
                'timestamp': time.time() - self.start_time,
                'value': value
            })
        
        def log_system_metrics(self):
            memory_info = get_memory_info()
            
            self.log_metric('memory_usage_mb', memory_info['rss_mb'])
            self.log_metric('memory_percent', memory_info['percent'])
            self.log_metric('cpu_percent', psutil.cpu_percent())
        
        def generate_report(self):
            report = {}
            for metric_name, values in self.metrics.items():
                if values:
                    report[metric_name] = {
                        'count': len(values),
                        'mean': sum(v['value'] for v in values) / len(values),
                        'max': max(v['value'] for v in values),
                        'min': min(v['value'] for v in values)
                    }
            return report
    
    return PerformanceMonitor()

# Usage
monitor = setup_performance_monitoring()
monitor.start_monitoring()

# During processing
monitor.log_system_metrics()
monitor.log_metric('processing_time', processing_duration)
monitor.log_metric('ssim_score', ssim_result)

# Generate performance report
performance_report = monitor.generate_report()
print("Performance Report:", performance_report)
```

### Automated Backup and Recovery

```python
def setup_backup_system():
    """Setup automated backup and recovery system."""
    
    import shutil
    import datetime
    from pathlib import Path
    
    class BackupManager:
        def __init__(self, backup_dir="backups"):
            self.backup_dir = Path(backup_dir)
            self.backup_dir.mkdir(exist_ok=True)
        
        def backup_models(self, model_dir="models"):
            """Backup trained models."""
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"models_{timestamp}"
            
            if Path(model_dir).exists():
                shutil.copytree(model_dir, backup_path)
                print(f"Models backed up to: {backup_path}")
                return backup_path
        
        def backup_configs(self, config_dir="configs"):
            """Backup configuration files."""
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"configs_{timestamp}"
            
            if Path(config_dir).exists():
                shutil.copytree(config_dir, backup_path)
                print(f"Configs backed up to: {backup_path}")
                return backup_path
        
        def cleanup_old_backups(self, keep_days=30):
            """Remove backups older than specified days."""
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=keep_days)
            
            for backup_path in self.backup_dir.iterdir():
                if backup_path.is_dir():
                    creation_time = datetime.datetime.fromtimestamp(
                        backup_path.stat().st_ctime
                    )
                    
                    if creation_time < cutoff_time:
                        shutil.rmtree(backup_path)
                        print(f"Removed old backup: {backup_path}")
        
        def restore_from_backup(self, backup_path, restore_to):
            """Restore from backup."""
            if Path(backup_path).exists():
                shutil.copytree(backup_path, restore_to)
                print(f"Restored from {backup_path} to {restore_to}")
            else:
                print(f"Backup not found: {backup_path}")
    
    return BackupManager()

# Usage
backup_manager = setup_backup_system()
backup_manager.backup_models()
backup_manager.backup_configs()
backup_manager.cleanup_old_backups(keep_days=30)
```

## Conclusion

This comprehensive usage guide covers all aspects of the Bathymetric CAE package, from basic usage to advanced integration patterns. The package provides flexible tools for processing bathymetric data using state-of-the-art machine learning techniques.

### Key Takeaways

1. **Start Simple**: Begin with default configurations and basic workflows
2. **Optimize Gradually**: Tune parameters based on your data and hardware
3. **Monitor Performance**: Use built-in monitoring tools to track processing
4. **Validate Results**: Always verify output quality and processing success
5. **Plan for Scale**: Design workflows that can grow with your data volume

### Getting Help

- **Documentation**: Comprehensive guides and API reference
- **Examples**: Working code examples for common use cases
- **Community**: GitHub discussions and issue tracking
- **Support**: Professional support for production deployments

### Next Steps

1. **Explore Examples**: Try the provided example scripts
2. **Customize Configuration**: Adapt settings for your specific needs
3. **Integrate Workflows**: Connect with your existing data pipelines
4. **Scale Operations**: Optimize for production data volumes
5. **Contribute**: Share improvements and extensions with the community

The Bathymetric CAE package is designed to grow with your needs, from initial exploration to large-scale production deployments. Happy processing! 🌊