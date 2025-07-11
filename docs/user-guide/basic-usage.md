# Basic Usage Guide

This guide covers the essential usage patterns for the Enhanced Bathymetric CAE Processing system. After completing the [Quick Start Tutorial](quick-start.md), use this guide to understand core functionality and common workflows.

## 🎯 Overview

The Enhanced Bathymetric CAE Processing system provides multiple ways to process bathymetric data:

- **Command Line Interface (CLI)**: Simple commands for most use cases
- **Python API**: Programmatic control for integration and automation
- **Configuration Files**: Reproducible processing with saved settings
- **Batch Processing**: Efficient handling of multiple files

## 💻 Command Line Usage

### Basic Processing Command

```bash
# Simplest form - process all files in input folder
bathymetric-cae --input data/input --output data/output
```

### Common Options

```bash
# Full featured processing with all enhancements
bathymetric-cae \
    --input data/bathymetry \
    --output data/processed \
    --enable-adaptive \
    --enable-expert-review \
    --enable-constitutional \
    --quality-threshold 0.8
```

### Customizing Processing Parameters

```bash
# High quality processing (slower)
bathymetric-cae \
    --input data/input \
    --output data/output \
    --epochs 200 \
    --ensemble-size 5 \
    --grid-size 1024 \
    --batch-size 4

# Fast processing (lower quality)
bathymetric-cae \
    --input data/input \
    --output data/output \
    --epochs 25 \
    --ensemble-size 1 \
    --grid-size 256 \
    --batch-size 8
```

### Using Configuration Files

```bash
# Save current settings to a file
bathymetric-cae --save-config my_settings.json

# Use saved configuration
bathymetric-cae --config my_settings.json

# Override specific settings
bathymetric-cae \
    --config my_settings.json \
    --epochs 150 \
    --output different_output
```

### Memory and Performance Options

```bash
# CPU-only processing
bathymetric-cae --no-gpu --input data/input --output data/output

# Reduce memory usage
bathymetric-cae \
    --batch-size 1 \
    --grid-size 256 \
    --ensemble-size 1 \
    --max-workers 1

# Increase parallelization
bathymetric-cae --max-workers 8 --input data/input --output data/output
```

## 🐍 Python API Usage

### Basic Python Script

```python
from enhanced_bathymetric_cae import Config, EnhancedBathymetricCAEPipeline

# Create configuration
config = Config(
    input_folder="data/bathymetry",
    output_folder="data/processed",
    epochs=100,
    enable_adaptive_processing=True
)

# Initialize and run pipeline
pipeline = EnhancedBathymetricCAEPipeline(config)
pipeline.run(
    input_folder=config.input_folder,
    output_folder=config.output_folder,
    model_path="models/my_model.h5"
)
```

### Processing Single Files

```python
from pathlib import Path
from enhanced_bathymetric_cae import BathymetricProcessor, Config

# Setup processor
config = Config(grid_size=512)
processor = BathymetricProcessor(config)

# Process single file
input_data, shape, metadata = processor.preprocess_bathymetric_grid(
    Path("sample.bag")
)

print(f"Processed data shape: {input_data.shape}")
print(f"Original dimensions: {shape}")
print(f"Metadata keys: {list(metadata.keys())}")
```

### Batch Processing with Python

```python
from pathlib import Path
from enhanced_bathymetric_cae import Config, EnhancedBathymetricCAEPipeline

def process_multiple_folders(base_folder):
    """Process multiple subfolders of bathymetric data."""
    base_path = Path(base_folder)
    
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            print(f"Processing {subfolder.name}...")
            
            config = Config(
                input_folder=str(subfolder),
                output_folder=str(subfolder.parent / f"{subfolder.name}_processed"),
                epochs=100,
                enable_adaptive_processing=True
            )
            
            pipeline = EnhancedBathymetricCAEPipeline(config)
            pipeline.run(
                input_folder=config.input_folder,
                output_folder=config.output_folder,
                model_path=f"models/{subfolder.name}_model.h5"
            )

# Usage
process_multiple_folders("data/surveys")
```

### Quality Assessment

```python
import numpy as np
from enhanced_bathymetric_cae.core import BathymetricQualityMetrics

# Load your data (original and processed)
original_data = np.load("original.npy")
processed_data = np.load("processed.npy")

# Calculate quality metrics
metrics = BathymetricQualityMetrics()

ssim = metrics.calculate_ssim_safe(original_data, processed_data)
roughness = metrics.calculate_roughness(processed_data)
feature_preservation = metrics.calculate_feature_preservation(original_data, processed_data)
consistency = metrics.calculate_depth_consistency(processed_data)

print(f"SSIM: {ssim:.4f}")
print(f"Roughness: {roughness:.4f}")
print(f"Feature Preservation: {feature_preservation:.4f}")
print(f"Consistency: {consistency:.4f}")
```

## ⚙️ Configuration Management

### Creating Configurations

```python
from enhanced_bathymetric_cae import Config

# Basic configuration
config = Config()

# Custom configuration
config = Config(
    # Training parameters
    epochs=150,
    batch_size=16,
    learning_rate=0.0008,
    
    # Model architecture
    ensemble_size=5,
    grid_size=1024,
    
    # Enhanced features
    enable_adaptive_processing=True,
    enable_expert_review=True,
    enable_constitutional_constraints=True,
    
    # Quality settings
    quality_threshold=0.8,
    ssim_weight=0.3,
    feature_preservation_weight=0.3
)
```

### Saving and Loading Configurations

```python
# Save configuration
config.save("production_config.json")

# Load configuration
config = Config.load("production_config.json")

# Modify loaded configuration
config.epochs = 200
config.ensemble_size = 7
```

### Environment-Specific Configurations

```python
# Development configuration (fast, lower quality)
dev_config = Config(
    epochs=25,
    ensemble_size=1,
    grid_size=256,
    batch_size=2,
    enable_adaptive_processing=False
)

# Production configuration (high quality)
prod_config = Config(
    epochs=300,
    ensemble_size=7,
    grid_size=1024,
    batch_size=8,
    enable_adaptive_processing=True,
    enable_expert_review=True,
    quality_threshold=0.85
)

# Testing configuration (minimal resources)
test_config = Config(
    epochs=5,
    ensemble_size=1,
    grid_size=128,
    batch_size=1
)
```

## 📁 File and Folder Management

### Input Folder Structure

Organize your input data for best results:

```
input_folder/
├── survey_2024_01/
│   ├── line_001.bag
│   ├── line_002.bag
│   └── line_003.bag
├── survey_2024_02/
│   ├── area_north.tif
│   └── area_south.tif
└── legacy_data/
    ├── old_survey.asc
    └── converted_data.xyz
```

### Output Organization

The system creates organized output:

```
output_folder/
├── enhanced_line_001.bag
├── enhanced_line_002.bag
├── enhanced_line_003.bag
├── enhanced_area_north.tif
├── enhanced_area_south.tif
├── enhanced_processing_summary.json
├── logs/
│   └── bathymetric_processing.log
├── plots/
│   ├── enhanced_comparison_line_001.png
│   ├── enhanced_comparison_line_002.png
│   └── training_history_ensemble_0.png
└── expert_reviews/
    └── pending_reviews.json
```

### Working with Different File Formats

```python
from pathlib import Path
from enhanced_bathymetric_cae import BathymetricProcessor, Config

config = Config()
processor = BathymetricProcessor(config)

# Process BAG file (with uncertainty)
bag_file = Path("data/multibeam.bag")
if bag_file.exists():
    data, shape, metadata = processor.preprocess_bathymetric_grid(bag_file)
    print(f"BAG file channels: {data.shape[-1]}")  # Should be 2 (depth + uncertainty)

# Process GeoTIFF file
tiff_file = Path("data/bathymetry.tif")
if tiff_file.exists():
    data, shape, metadata = processor.preprocess_bathymetric_grid(tiff_file)
    print(f"GeoTIFF channels: {data.shape[-1]}")  # Should be 1 (depth only)

# Process ASCII Grid
asc_file = Path("data/survey.asc")
if asc_file.exists():
    data, shape, metadata = processor.preprocess_bathymetric_grid(asc_file)
    print(f"ASCII Grid processed: {shape}")
```

## 📊 Understanding Output

### Enhanced Data Files

Processed files maintain their original format but include enhanced metadata:

```python
from osgeo import gdal

# Open enhanced file
dataset = gdal.Open("enhanced_sample.bag")

# Check processing metadata
processing_metadata = dataset.GetMetadata("PROCESSING")
quality_metadata = dataset.GetMetadata("QUALITY")
adaptive_metadata = dataset.GetMetadata("ADAPTIVE")

print("Processing Information:")
for key, value in processing_metadata.items():
    print(f"  {key}: {value}")

print("\nQuality Metrics:")
for key, value in quality_metadata.items():
    print(f"  {key}: {value}")
```

### Processing Summary Report

```python
import json

# Read processing summary
with open("enhanced_processing_summary.json", "r") as f:
    summary = json.load(f)

print(f"Files processed: {summary['successful_files']}")
print(f"Success rate: {summary['success_rate']:.1f}%")
print(f"Mean quality: {summary['summary_statistics']['mean_composite_quality']:.4f}")

# Check seafloor distribution
seafloor_dist = summary['seafloor_distribution']
for seafloor_type, count in seafloor_dist.items():
    print(f"{seafloor_type}: {count} files")
```

### Quality Visualizations

The system generates comparison plots automatically:

```python
import matplotlib.pyplot as plt
from pathlib import Path

# View generated plots
plots_dir = Path("plots")
for plot_file in plots_dir.glob("enhanced_comparison_*.png"):
    print(f"Generated plot: {plot_file}")
    # Open with default image viewer
    # plt.imread(plot_file)
    # plt.imshow(img)
    # plt.show()
```

## 🔍 Monitoring and Debugging

### Enabling Detailed Logging

```bash
# Enable debug logging
bathymetric-cae --log-level DEBUG --input data/input --output data/output

# Check log file
tail -f logs/bathymetric_processing.log
```

### Memory Monitoring

```python
from enhanced_bathymetric_cae.utils import memory_monitor, log_memory_usage

# Monitor memory usage
with memory_monitor("Processing large dataset"):
    # Your processing code here
    pipeline.run(input_folder, output_folder, model_path)

# Log current memory
log_memory_usage("After processing")
```

### Performance Profiling

```python
import time
from enhanced_bathymetric_cae.utils import memory_monitor

def profile_processing(input_folder, output_folder):
    """Profile processing performance."""
    start_time = time.time()
    
    with memory_monitor("Complete processing") as monitor:
        pipeline = EnhancedBathymetricCAEPipeline(config)
        pipeline.run(input_folder, output_folder, "models/ensemble.h5")
    
    end_time = time.time()
    results = monitor.get_results()
    
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Peak memory: {results['max_memory_mb']:.1f} MB")
    print(f"Average memory: {results['avg_memory_mb']:.1f} MB")

# Usage
profile_processing("data/input", "data/output")
```

## 🎯 Common Workflows

### Workflow 1: Single Survey Processing

```bash
# 1. Organize data
mkdir -p survey_2024_03/{input,output}
cp *.bag survey_2024_03/input/

# 2. Process with quality focus
bathymetric-cae \
    --input survey_2024_03/input \
    --output survey_2024_03/output \
    --enable-adaptive \
    --enable-expert-review \
    --quality-threshold 0.8

# 3. Review results
ls -la survey_2024_03/output/
cat survey_2024_03/output/enhanced_processing_summary.json
```

### Workflow 2: Batch Survey Processing

```python
#!/usr/bin/env python3
"""
Batch processing script for multiple surveys
"""
import os
import sys
from pathlib import Path
from enhanced_bathymetric_cae import Config, EnhancedBathymetricCAEPipeline

def process_surveys(base_directory):
    """Process all surveys in subdirectories."""
    base_path = Path(base_directory)
    
    # Standard configuration for all surveys
    config = Config(
        epochs=150,
        ensemble_size=3,
        enable_adaptive_processing=True,
        enable_expert_review=True,
        quality_threshold=0.75
    )
    
    for survey_dir in base_path.iterdir():
        if not survey_dir.is_dir():
            continue
            
        input_folder = survey_dir / "input"
        output_folder = survey_dir / "output"
        
        if not input_folder.exists():
            print(f"Skipping {survey_dir.name}: no input folder")
            continue
        
        print(f"Processing survey: {survey_dir.name}")
        
        # Create output directory
        output_folder.mkdir(exist_ok=True)
        
        # Process survey
        try:
            pipeline = EnhancedBathymetricCAEPipeline(config)
            pipeline.run(
                str(input_folder),
                str(output_folder),
                str(survey_dir / "model.h5")
            )
            print(f"✅ Completed: {survey_dir.name}")
            
        except Exception as e:
            print(f"❌ Failed: {survey_dir.name} - {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_surveys.py <base_directory>")
        sys.exit(1)
    
    process_surveys(sys.argv[1])
```

### Workflow 3: Quality Control Pipeline

```python
#!/usr/bin/env python3
"""
Quality control workflow with expert review
"""
from enhanced_bathymetric_cae import Config, EnhancedBathymetricCAEPipeline
from enhanced_bathymetric_cae.review import ExpertReviewSystem

def quality_control_workflow(input_folder, output_folder):
    """Complete quality control workflow."""
    
    # Step 1: Process with strict quality requirements
    config = Config(
        enable_expert_review=True,
        quality_threshold=0.85,  # High threshold
        auto_flag_threshold=0.6,
        ensemble_size=5
    )
    
    pipeline = EnhancedBathymetricCAEPipeline(config)
    pipeline.run(input_folder, output_folder, "models/qc_model.h5")
    
    # Step 2: Review flagged files
    review_system = ExpertReviewSystem()
    pending_reviews = review_system.get_pending_reviews()
    
    print(f"Files flagged for review: {len(pending_reviews)}")
    for review in pending_reviews:
        print(f"  - {review['filename']}: {review['flag_type']}")
    
    # Step 3: Generate quality report
    stats = review_system.get_review_statistics()
    print(f"Review completion rate: {stats['completion_rate']:.1f}%")

# Usage
quality_control_workflow("data/survey_input", "data/survey_output")
```

## ⚡ Performance Tips

### Optimizing for Speed

1. **Reduce ensemble size**: Use `--ensemble-size 1` for fastest processing
2. **Lower resolution**: Use `--grid-size 256` instead of 512
3. **Fewer epochs**: Use `--epochs 25-50` for development
4. **Increase batch size**: Use `--batch-size 16` if you have enough memory
5. **Use GPU**: Ensure CUDA is properly installed

### Optimizing for Quality

1. **Increase ensemble size**: Use `--ensemble-size 5-7`
2. **More training**: Use `--epochs 200-300`
3. **Higher resolution**: Use `--grid-size 1024`
4. **Enable all features**: Use all enhancement flags
5. **Lower quality threshold**: Use `--quality-threshold 0.85`

### Balancing Speed and Quality

```python
# Balanced configuration for production
balanced_config = Config(
    epochs=150,
    ensemble_size=3,
    grid_size=512,
    batch_size=8,
    enable_adaptive_processing=True,
    enable_constitutional_constraints=True,
    quality_threshold=0.8
)
```

---

## 📚 Next Steps

Now that you understand basic usage:

- **[Learn about Advanced Features](../user-guide/adaptive-processing.md)**
- **[Explore Configuration Options](configuration.md)**
- **[Try Batch Processing Tutorial](../tutorials/batch-automation.md)**
- **[Understand Quality Metrics](../user-guide/quality-metrics.md)**

Need help? Check the [Troubleshooting Guide](troubleshooting.md) or visit our [Community Forum](https://community.bathymetric-cae.org).