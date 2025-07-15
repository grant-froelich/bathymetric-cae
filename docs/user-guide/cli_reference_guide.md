# Enhanced Bathymetric CAE Processing - Command Line Reference Guide

Complete guide to command line options, usage patterns, and best practices for the Enhanced Bathymetric CAE Processing system.

## 📋 Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Input/Output Options](#inputoutput-options)
- [Configuration Options](#configuration-options)
- [Training Parameters](#training-parameters)
- [Model Architecture](#model-architecture)
- [Enhanced Features](#enhanced-features)
- [Processing Options](#processing-options)
- [Quality Metric Weights](#quality-metric-weights)
- [Common Usage Patterns](#common-usage-patterns)
- [Troubleshooting Commands](#troubleshooting-commands)

## 📋 Quick Reference Table

| Option | Type | Default | Range | Purpose |
|--------|------|---------|-------|---------|
| `--input` | Path | `\\network_folder\input_bathymetric_files` | - | Input directory |
| `--output` | Path | `\\network_folder\output_bathymetric_files` | - | Output directory |
| `--model` | Path | `cae_model_with_uncertainty.keras` | - | Model file path |
| `--config` | Path | None | - | Load config file |
| `--save-config` | Path | None | - | Save config file |
| `--epochs` | Integer | 100 | 10-500 | Training iterations |
| `--batch-size` | Integer | 8 | 1-32 | Batch size |
| `--learning-rate` | Float | 0.001 | 0.0001-0.01 | Learning rate |
| `--validation-split` | Float | 0.2 | 0.1-0.3 | Validation ratio |
| `--grid-size` | Integer | 512 | 128,256,512,1024,2048 | Processing resolution |
| `--base-filters` | Integer | 32 | 16-64 | Model complexity |
| `--depth` | Integer | 4 | 2-6 | Model depth |
| `--dropout-rate` | Float | 0.2 | 0.0-0.5 | Regularization |
| `--ensemble-size` | Integer | 3 | 1-10 | Number of models |
| `--enable-adaptive` | Flag | False | - | Seafloor adaptation |
| `--enable-expert-review` | Flag | False | - | Expert review system |
| `--enable-constitutional` | Flag | False | - | AI constraints |
| `--quality-threshold` | Float | 0.7 | 0.0-1.0 | Review threshold |
| `--max-workers` | Integer | -1 | 1-32 | Parallel workers |
| `--log-level` | Choice | INFO | DEBUG,INFO,WARNING,ERROR | Logging level |
| `--no-gpu` | Flag | False | - | Disable GPU |
| `--ssim-weight` | Float | 0.3 | 0.0-1.0 | SSIM importance |
| `--roughness-weight` | Float | 0.2 | 0.0-1.0 | Roughness importance |
| `--feature-weight` | Float | 0.3 | 0.0-1.0 | Feature importance |
| `--consistency-weight` | Float | 0.2 | 0.0-1.0 | Consistency importance |

---

## 🚀 Quick Start Examples

### Basic Usage
```bash
# Process all files in current directory with defaults
bathymetric-cae

# Process specific folder
bathymetric-cae --input data/bathymetry --output data/processed

# Load saved configuration
bathymetric-cae --config my_settings.json
```

### Common Workflows
```bash
# Development (fast, lower quality)
bathymetric-cae --epochs 25 --ensemble-size 1 --grid-size 256

# Production (high quality)
bathymetric-cae --epochs 200 --ensemble-size 5 --enable-adaptive --enable-expert-review

# Quick test run
bathymetric-cae --epochs 5 --batch-size 1 --grid-size 128
```

---

## 📁 Input/Output Options

### `--input` / `--input-folder`
**Purpose**: Specify the directory containing bathymetric files to process  
**Type**: String (path)  
**Default**: `\\network_folder\input_bathymetric_files`

**Examples**:
```bash
# Absolute path
bathymetric-cae --input /data/survey_2024/bathymetry

# Relative path
bathymetric-cae --input ./input_data

# Windows network path
bathymetric-cae --input "\\server\surveys\january_2024"

# Current directory
bathymetric-cae --input .
```

**Best Practices**:
- Use absolute paths for production scripts
- Ensure directory contains supported formats (`.bag`, `.tif`, `.asc`, `.xyz`)
- Verify read permissions on input directory

### `--output` / `--output-folder`
**Purpose**: Specify where processed files will be saved  
**Type**: String (path)  
**Default**: `\\network_folder\output_bathymetric_files`

**Examples**:
```bash
# Standard output directory
bathymetric-cae --output /data/processed/survey_2024

# Timestamped output
bathymetric-cae --output "./results_$(date +%Y%m%d_%H%M%S)"

# Network storage
bathymetric-cae --output "\\storage\processed_surveys"
```

**Best Practices**:
- Ensure write permissions on output directory
- Directory will be created if it doesn't exist
- Use descriptive names for organization

### `--model` / `--model-path`
**Purpose**: Path to save/load the trained AI model  
**Type**: String (path)  
**Default**: `cae_model_with_uncertainty.keras`

**Examples**:
```bash
# Custom model path
bathymetric-cae --model models/survey_2024_model.keras

# Load existing model
bathymetric-cae --model trained_models/coastal_model.keras

# Timestamped model
bathymetric-cae --model "models/model_$(date +%Y%m%d).keras"
```

**Best Practices**:
- Use `.keras` extension for modern TensorFlow format
- Organize models by survey type or date
- Keep trained models for reuse on similar data

---

## ⚙️ Configuration Options

### `--config`
**Purpose**: Load configuration from JSON file  
**Type**: String (file path)  
**Default**: None (uses built-in defaults)

**Examples**:
```bash
# Load production configuration
bathymetric-cae --config configs/production.json

# Load survey-specific settings
bathymetric-cae --config surveys/coastal_mapping.json

# Load with overrides
bathymetric-cae --config base_config.json --epochs 300
```

**Sample Configuration File**:
```json
{
  "input_folder": "data/input",
  "output_folder": "data/output",
  "epochs": 150,
  "batch_size": 8,
  "ensemble_size": 3,
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "quality_threshold": 0.8
}
```

### `--save-config`
**Purpose**: Save current configuration to JSON file  
**Type**: String (file path)  
**Default**: None (doesn't save unless specified)

**Examples**:
```bash
# Save current settings
bathymetric-cae --save-config my_settings.json

# Save production configuration
bathymetric-cae --epochs 200 --ensemble-size 5 --save-config production.json

# Save and run
bathymetric-cae --save-config settings.json --input data/test
```

**Best Practices**:
- Save configurations for reproducible processing
- Use descriptive filenames
- Version control configuration files

---

## 🏋️ Training Parameters

### `--epochs`
**Purpose**: Number of training iterations  
**Type**: Integer  
**Default**: 100  
**Range**: 10-500

**Examples**:
```bash
# Quick test (faster, lower quality)
bathymetric-cae --epochs 25

# Standard processing
bathymetric-cae --epochs 100

# High quality (slower, better results)
bathymetric-cae --epochs 300

# Production quality
bathymetric-cae --epochs 200
```

**Usage Guidelines**:
- **25-50**: Development and testing
- **100-150**: Standard processing
- **200-300**: High-quality production
- **300+**: Research-grade quality

### `--batch-size`
**Purpose**: Number of samples processed simultaneously  
**Type**: Integer  
**Default**: 8  
**Range**: 1-32

**Examples**:
```bash
# Memory-constrained systems
bathymetric-cae --batch-size 1

# Standard processing
bathymetric-cae --batch-size 8

# High-memory systems
bathymetric-cae --batch-size 16

# GPU optimization
bathymetric-cae --batch-size 32
```

**Memory Guidelines**:
- **1-2**: < 8GB RAM
- **4-8**: 8-16GB RAM  
- **16-32**: > 16GB RAM + GPU

### `--learning-rate`
**Purpose**: AI model learning speed  
**Type**: Float  
**Default**: 0.001  
**Range**: 0.0001-0.01

**Examples**:
```bash
# Conservative learning (stable)
bathymetric-cae --learning-rate 0.0005

# Standard learning
bathymetric-cae --learning-rate 0.001

# Fast learning (may be unstable)
bathymetric-cae --learning-rate 0.005
```

**Usage Guidelines**:
- Lower values: More stable, slower convergence
- Higher values: Faster training, risk of instability
- Adjust based on data complexity

### `--validation-split`
**Purpose**: Fraction of data used for validation  
**Type**: Float  
**Default**: 0.2  
**Range**: 0.1-0.3

**Examples**:
```bash
# Small validation set
bathymetric-cae --validation-split 0.1

# Standard validation
bathymetric-cae --validation-split 0.2

# Large validation set
bathymetric-cae --validation-split 0.3
```

---

## 🏗️ Model Architecture

### `--grid-size`
**Purpose**: Processing resolution (pixels)  
**Type**: Integer  
**Default**: 512  
**Options**: 128, 256, 512, 1024, 2048

**Examples**:
```bash
# Fast processing (lower quality)
bathymetric-cae --grid-size 256

# Standard processing
bathymetric-cae --grid-size 512

# High resolution (slower, better quality)
bathymetric-cae --grid-size 1024

# Maximum resolution (very slow)
bathymetric-cae --grid-size 2048
```

**Performance vs Quality**:
- **128-256**: Fast development, lower detail
- **512**: Balanced performance/quality
- **1024**: High quality, slower processing
- **2048**: Research quality, very slow

### `--base-filters`
**Purpose**: AI model complexity (number of filters)  
**Type**: Integer  
**Default**: 32  
**Range**: 16-64

**Examples**:
```bash
# Lightweight model
bathymetric-cae --base-filters 16

# Standard model
bathymetric-cae --base-filters 32

# Complex model
bathymetric-cae --base-filters 48
```

### `--depth`
**Purpose**: AI model depth (number of layers)  
**Type**: Integer  
**Default**: 4  
**Range**: 2-6

**Examples**:
```bash
# Shallow model (fast)
bathymetric-cae --depth 3

# Standard model
bathymetric-cae --depth 4

# Deep model (better features)
bathymetric-cae --depth 5
```

### `--dropout-rate`
**Purpose**: AI regularization (prevents overfitting)  
**Type**: Float  
**Default**: 0.2  
**Range**: 0.0-0.5

**Examples**:
```bash
# No dropout
bathymetric-cae --dropout-rate 0.0

# Light regularization
bathymetric-cae --dropout-rate 0.1

# Standard regularization
bathymetric-cae --dropout-rate 0.2

# Heavy regularization
bathymetric-cae --dropout-rate 0.3
```

### `--ensemble-size`
**Purpose**: Number of AI models to combine  
**Type**: Integer  
**Default**: 3  
**Range**: 1-10

**Examples**:
```bash
# Single model (fastest)
bathymetric-cae --ensemble-size 1

# Standard ensemble
bathymetric-cae --ensemble-size 3

# High-accuracy ensemble
bathymetric-cae --ensemble-size 5

# Research-grade ensemble
bathymetric-cae --ensemble-size 7
```

**Performance Trade-offs**:
- **1**: Fastest, lowest accuracy
- **3**: Good balance
- **5**: High accuracy, slower
- **7+**: Maximum accuracy, very slow

---

## ✨ Enhanced Features

### `--enable-adaptive`
**Purpose**: Enable automatic seafloor type detection and adaptive processing  
**Type**: Flag (boolean)  
**Default**: False (disabled)

**Examples**:
```bash
# Enable adaptive processing
bathymetric-cae --enable-adaptive

# Combine with other features
bathymetric-cae --enable-adaptive --enable-expert-review

# Production setup
bathymetric-cae --enable-adaptive --quality-threshold 0.8
```

**What It Does**:
- Automatically detects seafloor type (coastal, deep ocean, seamount, etc.)
- Adjusts processing parameters for optimal results
- Uses different strategies for different underwater environments

### `--enable-expert-review`
**Purpose**: Enable human expert review system for quality control  
**Type**: Flag (boolean)  
**Default**: False (disabled)

**Examples**:
```bash
# Enable expert review
bathymetric-cae --enable-expert-review

# With custom quality threshold
bathymetric-cae --enable-expert-review --quality-threshold 0.85

# Production quality control
bathymetric-cae --enable-expert-review --enable-constitutional
```

**What It Does**:
- Flags low-quality results for human review
- Maintains database of expert assessments
- Generates review reports and statistics

### `--enable-constitutional`
**Purpose**: Enable constitutional AI constraints for data integrity  
**Type**: Flag (boolean)  
**Default**: False (disabled)

**Examples**:
```bash
# Enable safety constraints
bathymetric-cae --enable-constitutional

# Full feature set
bathymetric-cae --enable-adaptive --enable-expert-review --enable-constitutional

# Safety-first processing
bathymetric-cae --enable-constitutional --quality-threshold 0.9
```

**What It Does**:
- Prevents physically unrealistic results
- Preserves important bathymetric features
- Ensures gradient continuity and feature preservation

### `--quality-threshold`
**Purpose**: Set quality threshold for expert review flagging  
**Type**: Float  
**Default**: 0.7  
**Range**: 0.0-1.0

**Examples**:
```bash
# Relaxed threshold (fewer reviews)
bathymetric-cae --quality-threshold 0.6

# Standard threshold
bathymetric-cae --quality-threshold 0.7

# Strict threshold (more reviews)
bathymetric-cae --quality-threshold 0.85

# Research-grade threshold
bathymetric-cae --quality-threshold 0.9
```

**Guidelines**:
- **0.5-0.6**: Development/testing
- **0.7-0.8**: Standard production
- **0.85-0.9**: High-quality surveys
- **0.9+**: Research applications

---

## ⚡ Processing Options

### `--max-workers`
**Purpose**: Maximum number of parallel processing workers  
**Type**: Integer  
**Default**: -1 (auto-detect available CPU cores)  
**Range**: 1-32

**Examples**:
```bash
# Auto-detect cores
bathymetric-cae --max-workers -1

# Single-threaded (memory constrained)
bathymetric-cae --max-workers 1

# Use 4 cores
bathymetric-cae --max-workers 4

# Use all available cores
bathymetric-cae --max-workers 16
```

**Performance Guidelines**:
- **1**: Memory-constrained systems
- **4-8**: Standard workstations
- **16+**: High-performance systems

### `--log-level`
**Purpose**: Control logging verbosity  
**Type**: Choice  
**Options**: DEBUG, INFO, WARNING, ERROR  
**Default**: INFO

**Examples**:
```bash
# Detailed debugging information
bathymetric-cae --log-level DEBUG

# Standard information
bathymetric-cae --log-level INFO

# Only warnings and errors
bathymetric-cae --log-level WARNING

# Errors only
bathymetric-cae --log-level ERROR
```

**When to Use**:
- **DEBUG**: Troubleshooting, development
- **INFO**: Standard operation
- **WARNING**: Production monitoring
- **ERROR**: Minimal logging

### `--no-gpu`
**Purpose**: Disable GPU acceleration, use CPU only  
**Type**: Flag (boolean)  
**Default**: False (GPU enabled if available)

**Examples**:
```bash
# Force CPU processing
bathymetric-cae --no-gpu

# CPU with multiple workers
bathymetric-cae --no-gpu --max-workers 8

# Memory-optimized CPU processing
bathymetric-cae --no-gpu --batch-size 2 --max-workers 1
```

**When to Use**:
- GPU driver issues
- Memory constraints
- CPU-only systems
- Debugging purposes

---

## 📊 Quality Metric Weights

These options control how different quality aspects are weighted in the final score. **All weights must sum to 1.0**.

### `--ssim-weight`
**Purpose**: Weight for Structural Similarity Index  
**Type**: Float  
**Default**: 0.3  
**Range**: 0.0-1.0

### `--roughness-weight`
**Purpose**: Weight for surface roughness metric  
**Type**: Float  
**Default**: 0.2  
**Range**: 0.0-1.0

### `--feature-weight` / `--feature-preservation-weight`
**Purpose**: Weight for bathymetric feature preservation  
**Type**: Float  
**Default**: 0.3  
**Range**: 0.0-1.0

### `--consistency-weight`
**Purpose**: Weight for depth measurement consistency  
**Type**: Float  
**Default**: 0.2  
**Range**: 0.0-1.0

**Examples**:
```bash
# Emphasize feature preservation
bathymetric-cae --feature-weight 0.5 --ssim-weight 0.2 --consistency-weight 0.2 --roughness-weight 0.1

# Emphasize smoothness
bathymetric-cae --roughness-weight 0.4 --ssim-weight 0.3 --feature-weight 0.2 --consistency-weight 0.1

# Balanced weights (default)
bathymetric-cae --ssim-weight 0.3 --feature-weight 0.3 --consistency-weight 0.2 --roughness-weight 0.2
```

**⚠️ Important**: All weights must sum to 1.0!

---

## 🎯 Common Usage Patterns

### Development and Testing
```bash
# Quick development test
bathymetric-cae --epochs 10 --batch-size 1 --grid-size 128 --ensemble-size 1

# Feature testing
bathymetric-cae --epochs 25 --enable-adaptive --log-level DEBUG

# Performance testing
bathymetric-cae --epochs 50 --batch-size 8 --max-workers 4
```

### Production Processing
```bash
# Standard production
bathymetric-cae --epochs 150 --ensemble-size 3 --enable-adaptive --enable-expert-review

# High-quality production
bathymetric-cae --epochs 200 --ensemble-size 5 --grid-size 1024 --quality-threshold 0.85

# Batch production with configuration
bathymetric-cae --config production.json --input batch_data/ --output results/
```

### Specialized Surveys
```bash
# Coastal mapping (high detail)
bathymetric-cae --grid-size 1024 --feature-weight 0.4 --enable-adaptive --quality-threshold 0.8

# Deep ocean survey (noise reduction focus)
bathymetric-cae --roughness-weight 0.4 --ensemble-size 5 --enable-constitutional

# Research quality
bathymetric-cae --epochs 300 --ensemble-size 7 --grid-size 1024 --quality-threshold 0.9
```

### Resource-Constrained Processing
```bash
# Low memory
bathymetric-cae --batch-size 1 --grid-size 256 --ensemble-size 1 --max-workers 1

# CPU-only processing
bathymetric-cae --no-gpu --max-workers 8 --batch-size 4

# Network storage
bathymetric-cae --input "\\server\data" --output "\\server\results" --log-level WARNING
```

---

## 🔧 Troubleshooting Commands

### Memory Issues
```bash
# Minimal memory usage
bathymetric-cae --batch-size 1 --grid-size 256 --ensemble-size 1 --max-workers 1 --no-gpu

# Memory debugging
bathymetric-cae --log-level DEBUG --batch-size 2 --epochs 5
```

### Performance Issues
```bash
# Profile performance
bathymetric-cae --epochs 10 --log-level DEBUG --max-workers 1

# GPU debugging
bathymetric-cae --log-level DEBUG --batch-size 1 --epochs 5
```

### Quality Issues
```bash
# Debug quality problems
bathymetric-cae --epochs 100 --enable-adaptive --enable-constitutional --log-level DEBUG

# Strict quality checking
bathymetric-cae --quality-threshold 0.9 --enable-expert-review --log-level INFO
```

### File/Path Issues
```bash
# Test with minimal data
bathymetric-cae --input . --output ./test_output --epochs 5

# Verbose file processing
bathymetric-cae --log-level DEBUG --input data/ --output results/
```

---

## 📋 Complete Example Commands

### Beginner Examples
```bash
# Most basic usage
bathymetric-cae

# Simple custom paths
bathymetric-cae --input my_data --output my_results

# Quick test
bathymetric-cae --epochs 10 --batch-size 1
```

### Intermediate Examples
```bash
# Balanced production setup
bathymetric-cae --input surveys/2024 --output processed/2024 \
  --epochs 150 --ensemble-size 3 --enable-adaptive --quality-threshold 0.8

# Save and load configuration
bathymetric-cae --epochs 200 --ensemble-size 5 --save-config my_settings.json
bathymetric-cae --config my_settings.json --input new_data/
```

### Advanced Examples
```bash
# High-quality research processing
bathymetric-cae \
  --input research_surveys/ \
  --output high_quality_results/ \
  --config research_config.json \
  --epochs 300 \
  --ensemble-size 7 \
  --grid-size 1024 \
  --batch-size 4 \
  --enable-adaptive \
  --enable-expert-review \
  --enable-constitutional \
  --quality-threshold 0.9 \
  --feature-weight 0.4 \
  --ssim-weight 0.3 \
  --consistency-weight 0.2 \
  --roughness-weight 0.1 \
  --log-level INFO

# Memory-optimized batch processing
bathymetric-cae \
  --input large_dataset/ \
  --output batch_results/ \
  --batch-size 1 \
  --grid-size 512 \
  --ensemble-size 3 \
  --max-workers 1 \
  --epochs 100 \
  --enable-adaptive \
  --log-level WARNING
```

---

## 💡 Pro Tips

### Configuration Management
1. **Save successful configurations**: Use `--save-config` for settings that work well
2. **Version control configs**: Keep configuration files in version control
3. **Environment-specific configs**: Create separate configs for dev/test/prod

### Performance Optimization
1. **Start small**: Test with small datasets and low epochs first
2. **Monitor resources**: Use `--log-level DEBUG` to monitor memory/GPU usage
3. **Scale gradually**: Increase batch size and grid size based on available resources

### Quality Optimization
1. **Enable all features**: Use `--enable-adaptive --enable-expert-review --enable-constitutional` for best results
2. **Adjust thresholds**: Lower `--quality-threshold` for development, raise for production
3. **Customize weights**: Adjust quality metric weights based on survey requirements

### Workflow Integration
1. **Script automation**: Create shell scripts with common parameter combinations
2. **Batch processing**: Process multiple surveys with consistent settings
3. **Result validation**: Always review quality reports and expert review feedback

---

## 🆘 Getting Help

```bash
# Show all available options
bathymetric-cae --help

# Show version information
bathymetric-cae --version

# Test installation
bathymetric-cae --epochs 1 --batch-size 1 --log-level DEBUG
```

For more detailed help:
- 📖 Check the [User Guide](../user-guide/README.md)
- 🐛 [Report Issues](https://github.com/noaa-ocs-hydrography/bathymetric-cae/issues)
- 💬 [Community Discussions](https://github.com/noaa-ocs-hydrography/bathymetric-cae/discussions)

---

**Remember**: Start with simple commands and gradually add complexity as you become familiar with the system!