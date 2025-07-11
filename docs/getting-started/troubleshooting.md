# Troubleshooting Guide

This guide helps you resolve common issues when using the Enhanced Bathymetric CAE Processing system. Issues are organized by category with step-by-step solutions.

## 🚨 Quick Diagnostic

If you're experiencing issues, run this quick diagnostic:

```bash
# Test basic functionality
bathymetric-cae --version

# Test dependencies
python -c "
import enhanced_bathymetric_cae
import tensorflow as tf
from osgeo import gdal
print('✅ All imports successful')
print(f'TensorFlow: {tf.__version__}')
print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
"

# Test with sample data
python -c "
from tests.test_fixtures.sample_data_generator import TestDataGenerator
from pathlib import Path
TestDataGenerator.create_test_dataset(Path('diagnostic_test'), 1)
print('✅ Test data generated')
"
```

## 🔧 Installation Issues

### Import Errors

#### Error: `ModuleNotFoundError: No module named 'enhanced_bathymetric_cae'`

**Cause**: Package not installed or not in Python path.

**Solutions**:
```bash
# Verify installation
pip list | grep enhanced-bathymetric-cae

# Reinstall if missing
pip install enhanced-bathymetric-cae

# For development installation
cd enhanced-bathymetric-cae
pip install -e .
```

#### Error: `ImportError: No module named 'osgeo'`

**Cause**: GDAL not properly installed.

**Solutions**:

=== "Ubuntu/Debian"
    ```bash
    sudo apt update
    sudo apt install gdal-bin libgdal-dev
    pip install gdal==$(gdal-config --version)
    ```

=== "macOS"
    ```bash
    brew install gdal
    pip install gdal==$(gdal-config --version)
    ```

=== "Windows"
    ```bash
    # Use conda (recommended)
    conda install -c conda-forge gdal
    
    # Or download from OSGeo4W
    # https://trac.osgeo.org/osgeo4w/
    ```

#### Error: `ImportError: cannot import name 'tf' from 'tensorflow'`

**Cause**: TensorFlow not installed or version incompatible.

**Solutions**:
```bash
# Check TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# Install/upgrade TensorFlow
pip install --upgrade tensorflow>=2.13.0

# For GPU support
pip install tensorflow[and-cuda]
```

### Version Compatibility Issues

#### Error: `AttributeError: module 'tensorflow' has no attribute 'X'`

**Cause**: TensorFlow version too old.

**Solution**:
```bash
# Check required version
pip show enhanced-bathymetric-cae | grep Requires

# Upgrade TensorFlow
pip install --upgrade tensorflow>=2.13.0
```

#### Error: `CUDA out of memory` during installation

**Cause**: GPU memory insufficient for installation.

**Solutions**:
```bash
# Use CPU-only version
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow-cpu

# Or limit GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## 🗂️ File Processing Issues

### File Format Problems

#### Error: `ValueError: Unsupported file format: .xyz`

**Cause**: File format not supported or incorrect extension.

**Supported formats**: `.bag`, `.tif`, `.tiff`, `.asc`

**Solutions**:
```bash
# Check file format
file your_file.xyz

# Convert if needed (example with GDAL)
gdal_translate input.xyz output.tif

# Verify file is valid
gdalinfo your_file.bag
```

#### Error: `Cannot open file: filename.bag`

**Cause**: File corrupted, access denied, or format invalid.

**Solutions**:
```bash
# Check file permissions
ls -la filename.bag

# Test with GDAL directly
gdalinfo filename.bag

# Check file integrity
md5sum filename.bag  # Compare with known good checksum
```

### Data Validation Errors

#### Error: `DataValidationError: All values are invalid`

**Cause**: File contains only NaN, infinity, or no-data values.

**Solutions**:
```bash
# Inspect data with Python
python -c "
from osgeo import gdal
ds = gdal.Open('filename.bag')
band = ds.GetRasterBand(1)
data = band.ReadAsArray()
print(f'Data range: {data.min()} to {data.max()}')
print(f'Valid values: {(~np.isnan(data)).sum()}')
"

# Check no-data value
gdalinfo -stats filename.bag
```

#### Error: `ValueError: Empty data in filename.bag`

**Cause**: File has zero size or no valid data.

**Solutions**:
```bash
# Check file size
ls -lh filename.bag

# Verify data bands
gdalinfo filename.bag | grep "Band"

# Create test file if needed
python -c "
from tests.test_fixtures.sample_data_generator import TestDataGenerator
TestDataGenerator.create_sample_bag_file('test.bag')
"
```

## 💾 Memory and Performance Issues

### Out of Memory Errors

#### Error: `MemoryError` or `OOM when allocating tensor`

**Cause**: Insufficient RAM or GPU memory.

**Solutions**:

**Reduce Memory Usage**:
```bash
# Use smaller parameters
bathymetric-cae \
    --batch-size 1 \
    --grid-size 256 \
    --ensemble-size 1

# Disable GPU if needed
bathymetric-cae --no-gpu

# Set memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

**Monitor Memory**:
```python
from utils.memory_utils import memory_monitor, log_memory_usage

with memory_monitor("Processing"):
    # Your processing code
    pass

log_memory_usage("Current status")
```

### Slow Processing

#### Issue: Processing takes too long

**Solutions**:

**Enable GPU Acceleration**:
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU support
pip install tensorflow[and-cuda]
```

**Optimize Parameters**:
```bash
# Faster processing (lower quality)
bathymetric-cae \
    --epochs 25 \
    --ensemble-size 1 \
    --grid-size 256

# Use parallel processing
bathymetric-cae --max-workers 4
```

**Profile Performance**:
```python
from tests.utils.performance_monitor import PerformanceMonitor

with PerformanceMonitor().monitor() as monitor:
    # Your code
    pass

results = monitor.get_results()
print(f"Execution time: {results['execution_time']:.2f}s")
print(f"Memory usage: {results['max_memory_mb']:.1f}MB")
```

## 🤖 Model Training Issues

### Training Failures

#### Error: `Loss becomes NaN during training`

**Cause**: Learning rate too high, gradient explosion, or bad data.

**Solutions**:
```python
# Reduce learning rate
config = Config(learning_rate=0.0001)  # Default: 0.001

# Add gradient clipping
# This is handled automatically in the system

# Check data quality
from core.quality_metrics import BathymetricQualityMetrics
metrics = BathymetricQualityMetrics()
consistency = metrics.calculate_depth_consistency(your_data)
print(f"Data consistency: {consistency}")
```

#### Error: `Model fails to converge`

**Cause**: Poor data quality, inappropriate parameters, or insufficient training.

**Solutions**:
```python
# Increase training duration
config = Config(epochs=200)  # Default: 100

# Improve data preprocessing
config = Config(
    enable_adaptive_processing=True,
    enable_constitutional_constraints=True
)

# Use ensemble for better robustness
config = Config(ensemble_size=5)  # Default: 3
```

### GPU Training Issues

#### Error: `CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
bathymetric-cae --batch-size 2  # Default: 8

# Use gradient accumulation
# (handled automatically by the system)

# Monitor GPU memory
nvidia-smi
```

#### Error: `Could not load dynamic library 'libcudart.so.11.0'`

**Cause**: CUDA installation mismatch.

**Solutions**:
```bash
# Check CUDA version
nvcc --version

# Install compatible TensorFlow
pip install tensorflow[and-cuda]==2.13.0

# Use CPU if CUDA issues persist
export CUDA_VISIBLE_DEVICES=""
```

## ⚙️ Configuration Issues

### Invalid Configuration

#### Error: `ValidationError: epochs must be positive`

**Cause**: Invalid configuration parameters.

**Solutions**:
```python
# Check configuration validation
config = Config(epochs=100)  # Must be > 0
config.validate()

# Load from file with validation
try:
    config = Config.load("config.json")
except ValueError as e:
    print(f"Configuration error: {e}")
```

#### Error: `Quality metric weights must sum to 1.0`

**Solutions**:
```python
# Ensure weights sum to 1.0
config = Config(
    ssim_weight=0.25,
    roughness_weight=0.25,
    feature_preservation_weight=0.25,
    consistency_weight=0.25
)
```

### Path and Directory Issues

#### Error: `FileNotFoundError: Input folder does not exist`

**Solutions**:
```bash
# Check path exists
ls -la /path/to/input

# Create directory if needed
mkdir -p /path/to/input

# Use absolute paths
bathymetric-cae --input $(pwd)/data/input
```

#### Error: `PermissionError: Cannot write to output directory`

**Solutions**:
```bash
# Check permissions
ls -ld /path/to/output

# Fix permissions
chmod 755 /path/to/output

# Use writable directory
bathymetric-cae --output ~/bathymetric_output
```

## 🔍 Quality and Processing Issues

### Poor Quality Results

#### Issue: Low quality scores across all files

**Diagnostic**:
```python
# Check input data quality
from core.quality_metrics import BathymetricQualityMetrics
metrics = BathymetricQualityMetrics()

roughness = metrics.calculate_roughness(input_data)
consistency = metrics.calculate_depth_consistency(input_data)

print(f"Input roughness: {roughness}")
print(f"Input consistency: {consistency}")
```

**Solutions**:
```python
# Increase training duration
config = Config(epochs=200)

# Use larger ensemble
config = Config(ensemble_size=5)

# Enable all features
config = Config(
    enable_adaptive_processing=True,
    enable_constitutional_constraints=True
)
```

#### Issue: Features being lost during processing

**Solutions**:
```python
# Increase feature preservation weight
config = Config(feature_preservation_weight=0.4)

# Use seamount strategy for high-feature areas
# (automatically detected if enable_adaptive_processing=True)

# Reduce smoothing
config = Config(enable_constitutional_constraints=True)
```

### Expert Review Issues

#### Error: `sqlite3.OperationalError: database is locked`

**Cause**: Multiple processes accessing review database.

**Solutions**:
```bash
# Check for other processes
ps aux | grep bathymetric-cae

# Remove lock file if safe
rm expert_reviews.db-journal

# Use different database file
bathymetric-cae --review-db reviews_$(date +%s).db
```

#### Issue: No files flagged for expert review

**Solutions**:
```python
# Lower quality threshold
config = Config(quality_threshold=0.5)  # Default: 0.7

# Enable expert review
config = Config(enable_expert_review=True)

# Check auto-flag threshold
config = Config(auto_flag_threshold=0.4)  # Default: 0.5
```

## 🧪 Testing Issues

### Test Failures

#### Error: `ImportError: No module named 'tests'`

**Solutions**:
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run from project root
cd enhanced-bathymetric-cae
python -m pytest tests/
```

#### Error: Tests fail with GDAL errors

**Solutions**:
```bash
# Use mock environment
pytest tests/test_processing_data_processor.py -v

# Skip integration tests if GDAL issues
pytest tests/ -m "not integration"

# Generate test data first
python -c "
from tests.test_fixtures.sample_data_generator import TestDataGenerator
TestDataGenerator.create_test_dataset('test_data', 3)
"
```

### Performance Test Failures

#### Error: `AssertionError: execution_time > threshold`

**Solutions**:
```bash
# Run on faster hardware
# Or adjust thresholds in test_config.json

# Run performance tests separately
pytest tests/test_performance.py -v --tb=short
```

## 🌐 Environment Issues

### Python Environment

#### Issue: Conflicting package versions

**Solutions**:
```bash
# Create clean environment
python -m venv clean_env
source clean_env/bin/activate  # Linux/macOS
# or clean_env\Scripts\activate  # Windows

# Install fresh
pip install enhanced-bathymetric-cae

# Check for conflicts
pip check
```

#### Issue: Permission denied when installing

**Solutions**:
```bash
# Install in user directory
pip install --user enhanced-bathymetric-cae

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install enhanced-bathymetric-cae
```

### System Resources

#### Issue: Disk space errors

**Solutions**:
```bash
# Check disk space
df -h

# Clean up temporary files
rm -rf /tmp/bathymetric_*

# Clean up old models
rm -rf ~/.cache/bathymetric-cae/
```

## 📞 Getting Additional Help

### Diagnostic Information

When reporting issues, include:

```bash
# System information
python --version
pip list | grep -E "(enhanced|tensorflow|gdal)"
uname -a  # Linux/macOS
systeminfo | findstr /B /C:"OS Name" /C:"Total Physical Memory"  # Windows

# Error logs
cat logs/bathymetric_processing.log

# Test run
bathymetric-cae --input test_data --output test_output --epochs 2 --batch-size 1
```

### Support Channels

1. **Documentation**: Check relevant sections first
   - [Installation Guide](installation.md)
   - [Configuration Reference](../reference/configuration-reference.md)
   - [API Documentation](../api/)

2. **Community Support**:
   - [GitHub Issues](https://github.com/your-org/enhanced-bathymetric-cae/issues)
   - [Discord Community](https://discord.gg/bathymetric-cae)
   - [Stack Overflow](https://stackoverflow.com/questions/tagged/bathymetric-cae)

3. **Professional Support**:
   - Email: support@bathymetric-cae.org
   - Enterprise support available

### Issue Reporting Template

When reporting bugs, use this template:

```markdown
**Environment**:
- OS: [Windows 10 / Ubuntu 20.04 / macOS 12.0]
- Python version: [3.9.7]
- Package version: [2.0.0]

**Expected Behavior**:
What you expected to happen.

**Actual Behavior**:
What actually happened.

**Steps to Reproduce**:
1. Step one
2. Step two
3. Step three

**Error Message**:
```
Full error message and stack trace
```

**Configuration**:
```json
{
  "epochs": 100,
  "batch_size": 8
}
```

**Additional Context**:
Any other relevant information.
```

## 🔄 Common Solutions Summary

### Quick Fixes

| Problem | Quick Solution |
|---------|---------------|
| Import errors | `pip install --upgrade enhanced-bathymetric-cae` |
| GDAL issues | `conda install -c conda-forge gdal` |
| GPU errors | `export CUDA_VISIBLE_DEVICES=""` |
| Memory issues | `--batch-size 1 --grid-size 256` |
| Slow processing | `--epochs 25 --ensemble-size 1` |
| Permission errors | `chmod 755 directory` or `--output ~/output` |
| Test failures | `pip install -r tests/requirements-test.txt` |

### Best Practices for Avoiding Issues

1. **Start Simple**: Use default settings first
2. **Test Installation**: Run diagnostic commands
3. **Check Resources**: Ensure adequate RAM and disk space  
4. **Use Virtual Environments**: Avoid package conflicts
5. **Read Logs**: Check log files for detailed error information
6. **Keep Updated**: Use latest stable versions
7. **Document Problems**: Keep notes for reproducing issues

---

**Still having issues?** Check our [FAQ](../reference/faq.md) or [contact support](https://community.bathymetric-cae.org).