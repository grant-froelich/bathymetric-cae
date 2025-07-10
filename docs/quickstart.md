# Quick Start Guide

This guide will get you up and running with Bathymetric CAE in just a few minutes. Follow these steps to process your first bathymetric dataset.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher installed
- At least 4 GB of available RAM
- Some bathymetric data files (BAG, GeoTIFF, ASCII Grid, or XYZ format)

## Step 1: Installation

Install Bathymetric CAE using pip:

```bash
pip install bathymetric-cae
```

For GPU support (recommended):

```bash
pip install bathymetric-cae[gpu]
```

Verify the installation:

```bash
bathymetric-cae --validate-requirements
```

## Step 2: Prepare Your Data

Create a directory structure for your project:

```bash
mkdir my_bathymetry_project
cd my_bathymetry_project
mkdir data/input data/output models logs
```

Place your bathymetric files in the `data/input` directory. Supported formats:
- **BAG files** (`.bag`) - with uncertainty support
- **GeoTIFF files** (`.tif`, `.tiff`)
- **ASCII Grid files** (`.asc`)
- **XYZ point clouds** (`.xyz`)

## Step 3: Basic Processing

### Command Line Interface

The simplest way to get started:

```bash
# Basic processing with default settings
bathymetric-cae --input data/input --output data/output

# With custom parameters
bathymetric-cae \
  --input data/input \
  --output data/output \
  --epochs 100 \
  --batch-size 8 \
  --grid-size 512
```

### Python API

For more control, use the Python API:

```python
from bathymetric_cae import BathymetricCAEPipeline, Config

# Create configuration
config = Config(
    input_folder="data/input",
    output_folder="data/output",
    model_path="models/my_model.h5",
    epochs=50,
    batch_size=8,
    grid_size=256  # Start small for faster training
)

# Create and run pipeline
pipeline = BathymetricCAEPipeline(config)
results = pipeline.run(
    input_folder=config.input_folder,
    output_folder=config.output_folder,
    model_path=config.model_path
)

# Print results
print(f"Processing completed!")
print(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
print(f"Mean SSIM: {results['summary_statistics']['mean_ssim']:.4f}")
```

## Step 4: Review Results

After processing, check your output directory:

```
data/output/
├── cleaned_file1.tif
├── cleaned_file2.tif
└── ...

plots/
├── training_history.png
├── comparison_file1.png
└── processing_summary.png

logs/
└── bathymetric_processing.log

processing_report.json
```

### Key Output Files

- **Cleaned data files**: Enhanced bathymetric grids in the output directory
- **Visualizations**: Training progress and comparison plots in `plots/`
- **Processing report**: Detailed statistics in `processing_report.json`
- **Model file**: Trained model saved for future use

## Step 5: Single File Processing

For interactive analysis of individual files:

```bash
# Process a single file with visualization
bathymetric-cae \
  --single-file data/input/sample.bag \
  --model models/my_model.h5 \
  --show-plots
```

Or with Python:

```python
# Interactive single file processing
pipeline = BathymetricCAEPipeline(config)

results = pipeline.process_single_file_interactive(
    file_path="data/input/sample.bag",
    model_path="models/my_model.h5",
    show_plots=True
)

print(f"SSIM improvement: {results['ssim']:.4f}")
```

## Common Configuration Options

### Memory-Efficient Settings

For systems with limited memory:

```python
config = Config(
    grid_size=128,      # Smaller input size
    batch_size=4,       # Smaller batches
    base_filters=16,    # Fewer filters
    depth=2,            # Shallower network
    epochs=50
)
```

### High-Performance Settings

For powerful systems with GPU:

```python
config = Config(
    grid_size=512,      # Higher resolution
    batch_size=16,      # Larger batches
    base_filters=64,    # More filters
    depth=5,            # Deeper network
    epochs=200,
    use_mixed_precision=True,
    gpu_memory_growth=True
)
```

### Quick Prototyping

For fast testing and iteration:

```python
config = Config(
    grid_size=64,       # Very small for speed
    epochs=10,          # Few epochs
    batch_size=2,
    early_stopping_patience=3
)
```

## Understanding the Output

### Quality Metrics

The pipeline provides several quality metrics:

- **SSIM (Structural Similarity Index)**: Measures structural similarity (0-1, higher is better)
- **MSE (Mean Squared Error)**: Pixel-wise error (lower is better)
- **Processing Success Rate**: Percentage of successfully processed files

### Typical Results

Good processing results typically show:
- SSIM improvement of 15-40%
- Reduced noise and artifacts
- Preserved bathymetric features
- Consistent uncertainty estimates (for BAG files)

## Next Steps

### 1. Explore Advanced Features

```python
# Save custom configuration
config.save("my_config.json")

# Load and modify configuration
config = Config.load("my_config.json")
config = config.update(epochs=100, learning_rate=0.0005)

# Use different model architectures
from bathymetric_cae.examples import custom_config
high_perf_config = custom_config.create_high_performance_config()
```

### 2. Batch Processing

For processing many files efficiently:

```python
# Setup batch processing with monitoring
from bathymetric_cae.examples import batch_processing
batch_processing.main()  # Run comprehensive batch example
```

### 3. Custom Model Architectures

Explore advanced model configurations:

```python
from bathymetric_cae.examples import advanced_model_config
advanced_model_config.main()  # Explore custom architectures
```

### 4. Visualization and Analysis

Create detailed visualizations:

```python
from bathymetric_cae import Visualizer

visualizer = Visualizer()
visualizer.plot_comparison(original, cleaned, uncertainty)
visualizer.plot_data_distribution({"Original": original, "Cleaned": cleaned})
```

## Troubleshooting Quick Fixes

### Out of Memory Errors

```python
# Reduce memory usage
config = config.update(
    batch_size=2,
    grid_size=128,
    max_workers=1
)
```

### GPU Not Detected

```bash
# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU usage if needed
bathymetric-cae --no-gpu --input data/input --output data/output
```

### GDAL Import Errors

```bash
# Install GDAL using conda (recommended)
conda install -c conda-forge gdal

# Or check system installation
gdal-config --version
```

### Poor Quality Results

1. **Check input data quality**: Ensure files are valid and not corrupted
2. **Increase training epochs**: Try 100-200 epochs for better convergence
3. **Adjust model size**: Use larger models for complex data
4. **Verify data normalization**: Check that data ranges are reasonable

## Performance Tips

### Optimize for Your Hardware

```python
from bathymetric_cae import get_memory_info, check_gpu_availability

# Check system capabilities
memory_info = get_memory_info()
gpu_info = check_gpu_availability()

# Adapt configuration accordingly
if memory_info['available_mb'] > 16000:  # > 16GB RAM
    config.batch_size = 16
    config.grid_size = 512
elif gpu_info['gpu_available']:
    config.use_mixed_precision = True
    config.gpu_memory_growth = True
```

### Monitor Training Progress

```bash
# View training logs in real-time
tail -f logs/bathymetric_processing.log

# Launch TensorBoard for detailed monitoring
tensorboard --logdir logs/fit
```

## Getting Help

If you encounter issues:

1. **Check the logs**: Look in `logs/bathymetric_processing.log` for detailed error messages
2. **Validate installation**: Run `bathymetric-cae --validate-requirements`
3. **Review examples**: Check the `examples/` folder for working code
4. **Read documentation**: Visit the full documentation for detailed guides
5. **Ask for help**: Open an issue on GitHub with your error details

## What's Next?

Now that you've successfully processed your first bathymetric dataset:

- **Learn about [Configuration](configuration.md)** for customizing the pipeline
- **Explore [Examples](examples.md)** for advanced usage patterns
- **Study [Model Architectures](model_architectures.md)** for understanding the algorithms
- **Read about [Performance Optimization](performance_optimization.md)** for large-scale processing

Happy processing! 🌊
