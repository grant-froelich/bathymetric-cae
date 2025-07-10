  "learning_rate": 0.0005,
  "early_stopping_patience": 25,
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "enable_constitutional_constraints": true,
  "quality_threshold": 0.8,
  "ssim_weight": 0.4,
  "roughness_weight": 0.15,
  "feature_preservation_weight": 0.3,
  "consistency_weight": 0.15,
  "use_mixed_precision": true,
  "max_workers": 8
}
```

## Processing Workflows

### Workflow 1: Standard Processing

```bash
# Step 1: Prepare input data
mkdir -p input_data output_data

# Step 2: Configure processing
python -c "
from bathymetric_cae import Config
config = Config()
config.ensemble_size = 3
config.quality_threshold = 0.7
config.save('standard_config.json')
"

# Step 3: Process data
python main.py \
  --config standard_config.json \
  --input input_data \
  --output output_data

# Step 4: Review results
ls output_data/
cat enhanced_processing_summary.json
```

### Workflow 2: High-Quality Processing

```bash
# For critical data requiring highest quality
python main.py \
  --input critical_data \
  --output high_quality_results \
  --ensemble-size 5 \
  --quality-threshold 0.9 \
  --enable-adaptive \
  --enable-expert-review \
  --enable-constitutional \
  --epochs 300
```

### Workflow 3: Fast Processing

```bash
# For quick processing of large datasets
python main.py \
  --input large_dataset \
  --output quick_results \
  --ensemble-size 1 \
  --epochs 50 \
  --batch-size 16 \
  --grid-size 256
```

## Expert Review System

### Setting Up Expert Review

```python
from bathymetric_cae.review import ExpertReviewSystem

# Initialize review system
reviewer = ExpertReviewSystem("my_project_reviews.db")

# Check pending reviews
pending = reviewer.get_pending_reviews()
print(f"Pending reviews: {len(pending)}")

# Get reviewer dashboard
dashboard = reviewer.get_reviewer_dashboard("expert_1")
print(dashboard)
```

### Review Workflow

```python
# Flag a file for review
flag_id = reviewer.flag_for_review(
    filename="problematic_file.bag",
    region=(0, 0, 512, 512),
    flag_type="low_quality",
    confidence=0.3,
    description="Poor quality in deep water region"
)

# Submit expert review
review_id = reviewer.submit_review(
    filename="problematic_file.bag",
    region_id=str(flag_id),
    rating=3,
    quality_score=0.6,
    comments="Acceptable after manual inspection",
    reviewer_id="expert_marine_biologist"
)

# Generate review report
report = reviewer.generate_review_report(days=30)
print(report['summary'])
```

## Adaptive Processing

### Seafloor Type Processing

```python
from bathymetric_cae.core import SeafloorClassifier, AdaptiveProcessor

# Classify seafloor type
classifier = SeafloorClassifier()
depth_data = load_your_data()  # Your data loading function
seafloor_type = classifier.classify(depth_data)
print(f"Detected seafloor type: {seafloor_type.value}")

# Get adaptive parameters
processor = AdaptiveProcessor()
params = processor.get_processing_parameters(depth_data)
print(f"Recommended smoothing: {params['smoothing_factor']}")
print(f"Edge preservation: {params['edge_preservation']}")
```

### Custom Processing Strategies

```python
from bathymetric_cae.core.adaptive_processor import AdaptiveStrategy

class CustomStrategy(AdaptiveStrategy):
    def get_parameters(self, depth_data, local_characteristics=None):
        return {
            'smoothing_factor': 0.4,
            'edge_preservation': 0.7,
            'noise_threshold': 0.12,
            'gradient_constraint': 0.08,
            'feature_preservation_weight': 0.8
        }
    
    def get_strategy_name(self):
        return "custom_shallow_water"

# Register custom strategy
processor = AdaptiveProcessor()
processor.add_custom_strategy(SeafloorType.SHALLOW_COASTAL, CustomStrategy())
```

## Quality Assessment

### Calculating Quality Metrics

```python
from bathymetric_cae.core import BathymetricQualityMetrics

# Initialize quality calculator
quality_calc = BathymetricQualityMetrics()

# Calculate metrics
original_data = load_original_data()
processed_data = load_processed_data()

metrics = quality_calc.calculate_all_metrics(original_data, processed_data)

print(f"SSIM: {metrics['ssim']:.3f}")
print(f"Feature Preservation: {metrics['feature_preservation']:.3f}")
print(f"Composite Quality: {metrics['composite_quality']:.3f}")

# Generate full quality report
report = quality_calc.generate_quality_report(original_data, processed_data)
print(report['assessment']['recommendations'])
```

### Custom Quality Weights

```python
# Define custom quality weights
custom_weights = {
    'ssim': 0.4,
    'roughness': 0.1,
    'feature_preservation': 0.4,
    'consistency': 0.1
}

# Use in processing
python main.py \
  --ssim-weight 0.4 \
  --roughness-weight 0.1 \
  --feature-weight 0.4 \
  --consistency-weight 0.1
```

## Programming Interface

### Basic Processing

```python
from bathymetric_cae import Config, quick_process

# Quick processing
quick_process(
    input_path="data/input",
    output_path="data/output",
    ensemble_size=3,
    enable_adaptive_processing=True
)
```

### Advanced Processing

```python
from bathymetric_cae.processing import EnhancedBathymetricCAEPipeline
from bathymetric_cae.config import Config

# Create configuration
config = Config()
config.ensemble_size = 5
config.enable_expert_review = True

# Initialize pipeline
pipeline = EnhancedBathymetricCAEPipeline(config)

# Run processing
pipeline.run(
    input_folder="data/bathymetric_files",
    output_folder="data/processed_results", 
    model_path="models/my_ensemble"
)
```

### Model Training

```python
from bathymetric_cae.models import BathymetricEnsemble
from bathymetric_cae.processing import ModelTrainer

# Create ensemble
ensemble = BathymetricEnsemble(config)
models = ensemble.create_ensemble(channels=1)

# Train models
trainer = ModelTrainer(config)
trained_models = trainer.train_ensemble(models, X_train, y_train, "my_model")

# Save ensemble
ensemble.save_ensemble("trained_ensemble")
```

## File Format Support

### Supported Formats

| Format | Extension | Features | Uncertainty Support |
|--------|-----------|----------|-------------------|
| BAG | .bag | Full metadata, multi-band | ✅ Yes |
| GeoTIFF | .tif, .tiff | Geospatial metadata | ❌ No |
| ASCII Grid | .asc | Header with spatial info | ❌ No |
| XYZ | .xyz | Point cloud format | ❌ No |

### Format-Specific Usage

#### BAG Files
```python
# BAG files automatically include uncertainty data
processor = BathymetricProcessor(config)
input_data, shape, metadata = processor.preprocess_bathymetric_grid("survey.bag")
print(f"Channels: {input_data.shape[-1]}")  # Should be 2 (depth + uncertainty)
```

#### GeoTIFF Files
```python
# Standard raster processing
input_data, shape, metadata = processor.preprocess_bathymetric_grid("bathymetry.tif")
print(f"Projection: {metadata['projection']}")
print(f"Geotransform: {metadata['geotransform']}")
```

## Performance Optimization

### Memory Management

```python
from bathymetric_cae.processing.memory_utils import configure_memory_management

# Configure memory management
configure_memory_management({
    'cleanup_threshold_mb': 1500,
    'gpu_memory_growth': True,
    'mixed_precision': True
})
```

### GPU Optimization

```bash
# Set environment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_MEMORY_LIMIT=6144  # 6GB limit

# Run with GPU optimization
python main.py --mixed-precision --memory-limit 6144
```

### Parallel Processing

```python
# Configure parallel processing
config = Config()
config.max_workers = 8  # Use 8 CPU cores
config.batch_size = 16  # Larger batches for GPU efficiency
```

## Monitoring and Logging

### Setting Up Logging

```python
from bathymetric_cae.utils import setup_logging

# Setup enhanced logging
setup_logging(
    log_level="INFO",
    log_file="bathymetric_processing.log",
    enable_json=True
)
```

### Performance Monitoring

```python
from bathymetric_cae.utils import get_performance_logger

# Monitor performance
perf_logger = get_performance_logger("my_operation")
perf_logger.start_timer("data_loading")
# ... your code ...
perf_logger.end_timer("data_loading")
perf_logger.log_all_counters()
```

### Progress Tracking

```python
from bathymetric_cae.utils import get_progress_logger

# Track processing progress
progress = get_progress_logger(total_files, "file_processing")
for file in files:
    # Process file
    progress.update(1, f"Processing {file.name}")
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Reduce memory usage
python main.py --grid-size 256 --batch-size 2 --ensemble-size 1
```

#### Slow Processing
```bash
# Enable GPU and mixed precision
python main.py --mixed-precision --no-cpu-fallback
```

#### Quality Issues
```bash
# Increase ensemble size and quality threshold
python main.py --ensemble-size 5 --quality-threshold 0.9
```

### Debug Mode

```bash
# Enable debug logging
python main.py --log-level DEBUG

# Save debug plots
python main.py --create-plots --save-metrics
```

---

# docs/api_reference.md

# API Reference

## Core Modules

### bathymetric_cae.config

#### Config Class

```python
class Config:
    """Main configuration class for bathymetric processing."""
    
    def __init__(self, **kwargs):
        """Initialize configuration with default values."""
    
    def validate(self):
        """Validate configuration parameters."""
    
    def save(self, path: str):
        """Save configuration to JSON file."""
    
    @classmethod  
    def load(cls, path: str):
        """Load configuration from JSON file."""
    
    def update_from_dict(self, updates: dict):
        """Update configuration from dictionary."""
```

**Parameters:**
- `grid_size`: int, default 512 - Input grid size for processing
- `ensemble_size`: int, default 3 - Number of models in ensemble
- `epochs`: int, default 100 - Training epochs
- `batch_size`: int, default 8 - Training batch size
- `learning_rate`: float, default 0.001 - Learning rate for training
- `enable_adaptive_processing`: bool, default True - Enable adaptive processing
- `enable_expert_review`: bool, default True - Enable expert review system
- `quality_threshold`: float, default 0.7 - Quality threshold for review flagging

### bathymetric_cae.core

#### SeafloorClassifier

```python
class SeafloorClassifier:
    """Classify seafloor type for adaptive processing."""
    
    def __init__(self, use_ml_classification: bool = False):
        """Initialize classifier."""
    
    def classify(self, depth_data: np.ndarray) -> SeafloorType:
        """Classify seafloor type based on depth data."""
    
    def classify_with_confidence(self, depth_data: np.ndarray) -> Tuple[SeafloorType, float]:
        """Classify with confidence score."""
    
    def get_classification_features(self, depth_data: np.ndarray) -> Dict[str, float]:
        """Get all features used for classification."""
```

#### AdaptiveProcessor

```python
class AdaptiveProcessor:
    """Adaptive processing based on seafloor type."""
    
    def __init__(self, use_local_analysis: bool = True):
        """Initialize adaptive processor."""
    
    def get_processing_parameters(self, depth_data: np.ndarray) -> Dict[str, Any]:
        """Get adaptive processing parameters."""
    
    def get_processing_parameters_for_type(self, seafloor_type: SeafloorType, 
                                         depth_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get parameters for specific seafloor type."""
    
    def add_custom_strategy(self, seafloor_type: SeafloorType, strategy: AdaptiveStrategy):
        """Add custom processing strategy."""
```

#### BathymetricQualityMetrics

```python
class BathymetricQualityMetrics:
    """Calculate quality metrics for bathymetric data."""
    
    def __init__(self, iho_standard: str = "order_1a"):
        """Initialize with IHO standard."""
    
    def calculate_all_metrics(self, original: np.ndarray, processed: np.ndarray,
                            uncertainty: Optional[np.ndarray] = None,
                            weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate all quality metrics."""
    
    def generate_quality_report(self, original: np.ndarray, processed: np.ndarray,
                              uncertainty: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
    
    def assess_quality_level(self, composite_score: float) -> QualityLevel:
        """Assess quality level from composite score."""
```

#### BathymetricConstraints

```python
class BathymetricConstraints:
    """Constitutional AI constraints for bathymetric data."""
    
    def __init__(self, seafloor_type: Optional[SeafloorType] = None):
        """Initialize constraints for seafloor type."""
    
    def validate_all(self, data: np.ndarray, original: np.ndarray = None) -> Dict[str, ConstraintViolation]:
        """Validate data against all constraints."""
    
    def apply_corrections(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Apply all constraint corrections."""
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations."""
```

### bathymetric_cae.models

#### BathymetricEnsemble

```python
class BathymetricEnsemble:
    """Ensemble of models for improved robustness."""
    
    def __init__(self, config: Config):
        """Initialize ensemble with configuration."""
    
    def create_ensemble(self, channels: int) -> List[tf.keras.Model]:
        """Create ensemble of diverse models."""
    
    def predict_ensemble(self, input_data: np.ndarray, 
                        adaptive_params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Make ensemble prediction."""
    
    def save_ensemble(self, base_path: str):
        """Save all models in ensemble."""
    
    def load_ensemble(self, base_path: str):
        """Load ensemble from saved files."""
    
    def update_weights_from_performance(self, performance_scores: List[float]):
        """Update ensemble weights based on performance."""
```

#### Model Creation Functions

```python
def create_model_by_type(model_type: str, config: Config, 
                        input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Create model by type name."""

def get_model_variants(config: Config) -> Dict[str, Dict[str, Any]]:
    """Get available model variants."""

def compile_model(model: tf.keras.Model, config: Config, 
                 uncertainty_aware: bool = False) -> tf.keras.Model:
    """Compile model with appropriate settings."""
```

**Available Model Types:**
- `'lightweight'`: Fast model with reduced parameters
- `'standard'`: Balanced performance and accuracy  
- `'robust'`: Enhanced model with attention mechanisms
- `'deep'`: Deep architecture for complex patterns
- `'wide'`: Wide architecture with parallel processing
- `'uncertainty'`: Dual-output model with uncertainty estimation

### bathymetric_cae.processing

#### EnhancedBathymetricCAEPipeline

```python
class EnhancedBathymetricCAEPipeline:
    """Main processing pipeline with all enhancements."""
    
    def __init__(self, config: Config):
        """Initialize pipeline with configuration."""
    
    def run(self, input_folder: str, output_folder: str, model_path: str):
        """Run the complete processing pipeline."""
```

#### BathymetricProcessor

```python
class BathymetricProcessor:
    """Process bathymetric data files."""
    
    def __init__(self, config: Config):
        """Initialize processor."""
    
    def preprocess_bathymetric_grid(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int], Optional[Dict]]:
        """Load and preprocess bathymetric file."""
    
    def save_enhanced_results(self, data: np.ndarray, output_path: Path, 
                            original_shape: Tuple[int, int], geo_metadata: Dict,
                            quality_metrics: Dict, adaptive_params: Dict):
        """Save results with enhanced metadata."""
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate bathymetric file."""
    
    def get_supported_formats(self) -> Dict[str, bool]:
        """Get supported file formats."""
```

#### ModelTrainer

```python
class ModelTrainer:
    """Training infrastructure for models."""
    
    def __init__(self, config: Config):
        """Initialize trainer."""
    
    def train_ensemble(self, models: List[tf.keras.Model], 
                      X_train: np.ndarray, y_train: np.ndarray,
                      model_base_path: str) -> List[tf.keras.Model]:
        """Train ensemble of models."""
    
    def cross_validate_model(self, model_fn, X_data: np.ndarray, y_data: np.ndarray,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation."""
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
```

### bathymetric_cae.review

#### ExpertReviewSystem

```python
class ExpertReviewSystem:
    """Human-in-the-loop validation system."""
    
    def __init__(self, db_path: str = "expert_reviews.db"):
        """Initialize review system."""
    
    def flag_for_review(self, filename: str, region: Tuple[int, int, int, int], 
                       flag_type: str, confidence: float, description: str = "") -> int:
        """Flag region for expert review."""
    
    def submit_review(self, filename: str, region_id: Optional[str], 
                     rating: int, quality_score: float, comments: str = "",
                     reviewer_id: str = "unknown") -> int:
        """Submit expert review."""
    
    def get_pending_reviews(self, limit: Optional[int] = None) -> List[Dict]:
        """Get pending reviews."""
    
    def auto_review_quality(self, filename: str, quality_metrics: Dict) -> Optional[int]:
        """Perform automatic quality review."""
    
    def generate_review_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate review report."""
```

#### ReviewDatabase

```python
class ReviewDatabase:
    """Database operations for review system."""
    
    def __init__(self, db_path: str = "expert_reviews.db"):
        """Initialize database."""
    
    def flag_region_for_review(self, filename: str, region: Tuple[int, int, int, int], 
                              flag_type: str, confidence: float, description: str = "") -> int:
        """Flag region in database."""
    
    def submit_expert_review(self, filename: str, region_id: Optional[str], 
                           rating: int, quality_score: float, comments: str = "",
                           reviewer_id: str = "unknown") -> int:
        """Submit review to database."""
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get comprehensive review statistics."""
    
    def export_reviews_to_csv(self, output_path: str, include_metrics: bool = False):
        """Export reviews to CSV."""
```

### bathymetric_cae.utils

#### Logging Functions

```python
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None,
                 enable_console: bool = True, enable_json: bool = False) -> Dict[str, logging.Handler]:
    """Setup enhanced logging system."""

def get_logger(name: str, extra_fields: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Get logger with optional extra fields."""

def get_performance_logger(name: str) -> PerformanceLogger:
    """Get performance logger for timing operations."""

def get_progress_logger(total_items: int, name: str = "processing") -> ProcessingProgressLogger:
    """Get progress logger for tracking completion."""
```

#### Visualization Functions

```python
def create_enhanced_visualization(original: np.ndarray, cleaned: np.ndarray,
                                uncertainty: Optional[np.ndarray], metrics: Dict,
                                file_path: Path, adaptive_params: Dict):
    """Create enhanced visualization with quality metrics."""

def create_comparison_plot(original: np.ndarray, processed: np.ndarray, 
                         title: str = "Comparison", save_path: Optional[str] = None) -> str:
    """Create side-by-side comparison plot."""

def create_quality_dashboard(processing_stats: List[Dict], 
                           save_path: str = "plots/quality_dashboard.png"):
    """Create comprehensive quality dashboard."""

def plot_training_history(history, filename: str = "training_history.png"):
    """Plot model training history."""
```

#### CLI Functions

```python
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""

def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Update configuration with command line arguments."""

def validate_arguments(args: argparse.Namespace) -> Dict[str, str]:
    """Validate command line arguments."""
```

## Enumerations

### SeafloorType

```python
class SeafloorType(Enum):
    SHALLOW_COASTAL = "shallow_coastal"      # 0-200m depth
    DEEP_OCEAN = "deep_ocean"                # 2000-6000m depth  
    CONTINENTAL_SHELF = "continental_shelf"   # 200-2000m depth
    SEAMOUNT = "seamount"                    # Underwater mountains
    ABYSSAL_PLAIN = "abyssal_plain"          # 6000-11000m depth
    UNKNOWN = "unknown"                      # Unclassified
```

### QualityLevel

```python
class QualityLevel(Enum):
    EXCELLENT = "excellent"      # Score >= 0.9
    GOOD = "good"               # Score >= 0.8
    ACCEPTABLE = "acceptable"    # Score >= 0.7
    POOR = "poor"               # Score >= 0.5
    UNACCEPTABLE = "unacceptable" # Score < 0.5
```

### ExpertReviewFlag

```python
class ExpertReviewFlag(Enum):
    LOW_QUALITY = "low_quality"
    FEATURE_LOSS = "feature_loss"
    STANDARDS_VIOLATION = "standards_violation"
    POOR_SIMILARITY = "poor_similarity"
    HIGH_UNCERTAINTY = "high_uncertainty"
    PROCESSING_ERROR = "processing_error"
    MANUAL_REQUEST = "manual_request"
```

## Exceptions

### Configuration Errors

```python
class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

class ModelError(Exception):
    """Raised when model operations fail."""

class ProcessingError(Exception):
    """Raised when data processing fails."""

class QualityError(Exception):
    """Raised when quality assessment fails."""
```

## Constants

### File Format Support

```python
SUPPORTED_EXTENSIONS = ['.bag', '.tif', '.tiff', '.asc', '.xyz']
DEFAULT_NODATA_VALUE = -9999.0
MEMORY_CLEANUP_THRESHOLD = 1000  # MB
```

### IHO Standards

```python
IHO_STANDARDS = {
    "special_order": {"horizontal_accuracy": 2.0, "depth_accuracy": "0.25m + 0.0075*depth"},
    "order_1a": {"horizontal_accuracy": 5.0, "depth_accuracy": "0.5m + 0.013*depth"},
    "order_1b": {"horizontal_accuracy": 5.0, "depth_accuracy": "0.5m + 0.013*depth"},
    "order_2": {"horizontal_accuracy": 20.0, "depth_accuracy": "1.0m + 0.023*depth"}
}
```

---

# docs/troubleshooting.md

# Troubleshooting Guide

## Installation Issues

### GDAL Installation Problems

**Problem:** ImportError: No module named 'gdal'

**Solutions:**

1. **Windows (Conda - Recommended):**
```bash
conda install gdal -c conda-forge
```

2. **Windows (OSGeo4W):**
```bash
# Download OSGeo4W installer from https://trac.osgeo.org/osgeo4w/
# Install GDAL, then:
pip install gdal
```

3. **macOS:**
```bash
# Using Homebrew
brew install gdal
pip install gdal

# Using Conda
conda install gdal -c conda-forge
```

4. **Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev python3-gdal
pip install gdal
```

5. **Linux (CentOS/RHEL):**
```bash
sudo yum install gdal gdal-devel
pip install gdal
```

### TensorFlow GPU Issues

**Problem:** GPU not detected or CUDA errors

**Solutions:**

1. **Check GPU availability:**
```python
import tensorflow as tf
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("CUDA Version:", tf.test.is_built_with_cuda())
```

2. **Install CUDA and cuDNN:**
```bash
# Download CUDA 11.8 from NVIDIA
# Download cuDNN 8.6 from NVIDIA
# Add to PATH and LD_LIBRARY_PATH
```

3. **Reinstall TensorFlow with GPU support:**
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]>=2.13.0
```

4. **Check CUDA installation:**
```bash
nvidia-smi
nvcc --version
```

### Memory Issues During Installation

**Problem:** Out of memory during package installation

**Solutions:**

1. **Increase swap space (Linux):**
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

2. **Use pip with no cache:**
```bash
pip install --no-cache-dir bathymetric-cae
```

3. **Install packages individually:**
```bash
pip install tensorflow
pip install numpy
pip install matplotlib
# ... etc
```

## Runtime Issues

### Memory Problems

**Problem:** Out of memory during processing

**Symptoms:**
- `ResourceExhaustedError`
- `MemoryError`
- System freezing

**Solutions:**

1. **Reduce grid size:**
```python
config = Config()
config.grid_size = 256  # Instead of 512
config.save('memory_config.json')
```

2. **Reduce batch size:**
```bash
python main.py --batch-size 2  # Instead of 8
```

3. **Reduce ensemble size:**
```bash
python main.py --ensemble-size 1  # Instead of 3
```

4. **Enable mixed precision:**
```bash
python main.py --mixed-precision
```

5. **Set GPU memory limit:**
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_limit(gpus[0], 4096)  # 4GB
```

6. **Process files individually:**
```bash
# Instead of batch processing
for file in *.bag; do
    python main.py --input "$file" --output "processed_$file"
done
```

### Performance Issues

**Problem:** Very slow processing

**Symptoms:**
- Processing takes hours
- High CPU usage, low GPU usage
- Memory usage constantly increasing

**Solutions:**

1. **Enable GPU acceleration:**
```bash
# Check GPU usage
nvidia-smi

# Enable mixed precision
python main.py --mixed-precision
```

2. **Optimize batch size:**
```bash
# Find optimal batch size for your GPU
python main.py --batch-size 16  # Try different values
```

3. **Use multiple workers:**
```bash
python main.py --max-workers 8
```

4. **Profile your code:**
```python
import cProfile
cProfile.run('your_processing_function()')
```

5. **Check disk I/O:**
```bash
# Use faster storage (SSD)
# Avoid network drives for processing
```

### Quality Issues

**Problem:** Poor quality results

**Symptoms:**
- Low SSIM scores
- Poor feature preservation
- Visual artifacts in output

**Solutions:**

1. **Increase ensemble size:**
```bash
python main.py --ensemble-size 5
```

2. **Adjust quality weights:**
```bash
python main.py \
  --feature-weight 0.5 \
  --ssim-weight 0.3 \
  --consistency-weight 0.2
```

3. **Enable constitutional constraints:**
```bash
python main.py --enable-constitutional
```

4. **Use higher quality threshold:**
```bash
python main.py --quality-threshold 0.9
```

5. **Check input data quality:**
```python
from bathymetric_cae.processing import BathymetricProcessor

processor = BathymetricProcessor(config)
validation = processor.validate_file("your_file.bag")
print(validation)
```

## File Format Issues

### BAG File Problems

**Problem:** Cannot read BAG files

**Solutions:**

1. **Install BAG support:**
```bash
# Linux
sudo apt-get install libhdf5-dev
pip install h5py

# Windows
conda install hdf5 -c conda-forge
```

2. **Check BAG file validity:**
```python
from osgeo import gdal
dataset = gdal.Open("your_file.bag")
if dataset is None:
    print("Cannot open BAG file")
else:
    print(f"Bands: {dataset.RasterCount}")
    print(f"Size: {dataset.RasterXSize}x{dataset.RasterYSize}")
```

3. **Convert BAG to GeoTIFF:**
```bash
gdal_translate input.bag output.tif
```

### GeoTIFF Issues

**Problem:** GeoTIFF files not loading correctly

**Solutions:**

1. **Check TIFF validity:**
```bash
gdalinfo your_file.tif
```

2. **Handle large TIFF files:**
```python
config = Config()
config.grid_size = 1024  # Increase if needed
```

3. **Convert coordinate systems:**
```bash
gdalwarp -t_srs EPSG:4326 input.tif output_wgs84.tif
```

### ASCII Grid Issues

**Problem:** ASCII grid parsing errors

**Solutions:**

1. **Check header format:**
```
ncols         4
nrows         6
xllcorner     0.0
yllcorner     0.0
cellsize      50.0
NODATA_value  -9999
```

2. **Fix encoding issues:**
```bash
# Convert to UTF-8
iconv -f ISO-8859-1 -t UTF-8 input.asc > output.asc
```

## Configuration Issues

### Invalid Configuration

**Problem:** Configuration validation errors

**Solutions:**

1. **Check weight sum:**
```python
config = Config()
# Weights must sum to 1.0
config.ssim_weight = 0.3
config.roughness_weight = 0.2
config.feature_preservation_weight = 0.3
config.consistency_weight = 0.2
config.validate()
```

2. **Fix parameter ranges:**
```python
# Valid ranges
config.epochs = 100  # > 0
config.batch_size = 8  # > 0
config.validation_split = 0.2  # 0 < x < 1
config.quality_threshold = 0.7  # 0 <= x <= 1
```

### Path Issues

**Problem:** File path errors

**Solutions:**

1. **Use absolute paths:**
```bash
python main.py --input /full/path/to/input --output /full/path/to/output
```

2. **Check permissions:**
```bash
ls -la /path/to/data/
chmod 755 /path/to/data/
```

3. **Handle network paths:**
```bash
# Mount network drive first
# Use local paths when possible
```

## Expert Review Issues

### Database Problems

**Problem:** SQLite database errors

**Solutions:**

1. **Check database permissions:**
```bash
chmod 666 expert_reviews.db
```

2. **Backup and recreate database:**
```python
from bathymetric_cae.review import backup_review_database
backup_path = backup_review_database("expert_reviews.db", "backups/")
```

3. **Migrate database schema:**
```python
from bathymetric_cae.review.database import migrate_database
migrate_database("expert_reviews.db")
```

### Review Queue Issues

**Problem:** Reviews not appearing in queue

**Solutions:**

1. **Check flagging criteria:**
```python
system = ExpertReviewSystem()
pending = system.get_pending_reviews()
print(f"Pending: {len(pending)}")
```

2. **Verify quality thresholds:**
```python
config.quality_threshold = 0.5  # Lower threshold
```

## Model Training Issues

### Training Failures

**Problem:** Model training fails or doesn't converge

**Solutions:**

1. **Check training data:**
```python
from bathymetric_cae.processing.training import validate_training_data
valid = validate_training_data(X_train, y_train)
if not valid:
    print("Training data validation failed")
```

2. **Adjust learning rate:**
```bash
python main.py --learning-rate 0.0001  # Lower learning rate
```

3. **Increase patience:**
```bash
python main.py --early-stopping-patience 30
```

4. **Use gradient clipping:**
```python
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    clipnorm=1.0  # Gradient clipping
)
```

### Convergence Issues

**Problem:** Training loss not decreasing

**Solutions:**

1. **Check data normalization:**
```python
processor = BathymetricProcessor(config)
# Data should be normalized to [0, 1]
```

2. **Reduce model complexity:**
```bash
python main.py --depth 3 --base-filters 16
```

3. **Increase training data:**
```bash
# Use data augmentation
python main.py --augment-data
```

## Debugging Tools

### Enable Debug Mode

```bash
# Maximum verbosity
python main.py --log-level DEBUG --create-plots --save-metrics
```

### Memory Profiling

```python
from bathymetric_cae.processing.memory_utils import MemoryTracker

tracker = MemoryTracker("debug_session")
tracker.start_tracking()
# ... your code ...
tracker.checkpoint("after_data_loading")
summary = tracker.get_summary()
print(summary)
```

### Performance Profiling

```python
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()
# ... your code ...
pr.disable()

stats = pstats.Stats(pr)
stats.sort_stats('cumulative').print_stats(20)
```

### GPU Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv --loop=1
```

## Common Error Messages

### "ResourceExhaustedError: OOM when allocating tensor"

**Cause:** Out of GPU memory

**Solutions:**
- Reduce batch size: `--batch-size 2`
- Reduce grid size: `--grid-size 256`
- Enable mixed precision: `--mixed-precision`

### "InvalidArgumentError: Incompatible shapes"

**Cause:** Model input/output shape mismatch

**Solutions:**
- Check input data dimensions
- Verify grid size configuration
- Ensure consistent preprocessing

### "ModuleNotFoundError: No module named 'bathymetric_cae'"

**Cause:** Package not installed or PYTHONPATH issue

**Solutions:**
```bash
pip install -e .
# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/bathymetric-cae"
```

### "Permission denied: expert_reviews.db"

**Cause:** Database file permissions

**Solutions:**
```bash
chmod 666 expert_reviews.db
# Or run with sudo (not recommended)
```

## Getting Help

### Collect Debug Information

```bash
# Create debug report
python -c "
import sys
import tensorflow as tf
from bathymetric_cae.utils import log_system_info

print('Python:', sys.version)
print('TensorFlow:', tf.__version__)
print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)
log_system_info()
"
```

### Report Issues

When reporting issues, include:

1. **System information:**
   - Operating system and version
   - Python version
   - TensorFlow version
   - GPU information

2. **Configuration:**
   - Your config.json file
   - Command line arguments used

3. **Error details:**
   - Complete error message
   - Stack trace
   - Log files

4. **Data information:**
   - File formats being processed
   - Approximate file sizes
   - Number of files

### Community Support

- **GitHub Issues**: [Link to issues]
- **GitHub Discussions**: [Link to discussions]
- **Documentation**: [Link to full docs]
- **Email Support**: contact@example.com

---

## Appendix: Performance Benchmarks

### System Requirements vs Performance

| Configuration | RAM | GPU | Processing Speed | Quality |
|---------------|-----|-----|------------------|---------|
| Minimum | 8GB | None | 5 files/hour | Good |
| Recommended | 16GB | GTX 1080 | 20 files/hour | Very Good |
| High-end | 32GB | RTX 3080 | 50 files/hour | Excellent |

### Optimization Guidelines

1. **For speed**: Use lightweight models, small grid size
2. **For quality**: Use ensemble with 5+ models, constitutional constraints  
3. **For memory**: Reduce batch size, enable mixed precision
4. **For accuracy**: Enable adaptive processing, expert review# docs/installation.md

# Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8 GB RAM
- 10 GB free disk space

### Recommended Requirements
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM
- 50 GB free disk space

### Operating Systems
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

## Installation Methods

### Method 1: pip Installation (Recommended)

```bash
# Create virtual environment
python -m venv bathymetric_env
source bathymetric_env/bin/activate  # On Windows: bathymetric_env\Scripts\activate

# Install from PyPI (when published)
pip install bathymetric-cae

# Or install from source
pip install git+https://github.com/username/bathymetric-cae.git
```

### Method 2: Development Installation

```bash
# Clone repository
git clone https://github.com/username/bathymetric-cae.git
cd bathymetric-cae

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Method 3: Conda Installation

```bash
# Create conda environment
conda create -n bathymetric python=3.10
conda activate bathymetric

# Install dependencies
conda install tensorflow gdal matplotlib scikit-image -c conda-forge

# Install package
pip install -e .
```

## Dependency Installation

### Core Dependencies

```bash
# Required packages
pip install tensorflow>=2.13.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install scikit-image>=0.19.0
pip install scikit-learn>=1.1.0
pip install scipy>=1.9.0
pip install opencv-python>=4.6.0
pip install joblib>=1.1.0
pip install psutil>=5.8.0
```

### Geospatial Dependencies

#### GDAL Installation

**Windows:**
```bash
# Using conda (recommended)
conda install gdal -c conda-forge

# Or using OSGeo4W
# Download from https://trac.osgeo.org/osgeo4w/
```

**macOS:**
```bash
# Using Homebrew
brew install gdal

# Using conda
conda install gdal -c conda-forge
```

**Linux (Ubuntu/Debian):**
```bash
# System packages
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev

# Python bindings
pip install gdal
```

**Linux (CentOS/RHEL):**
```bash
# System packages
sudo yum install gdal gdal-devel

# Python bindings
pip install gdal
```

### GPU Support

#### NVIDIA GPU Setup

```bash
# Install CUDA toolkit (version 11.8 recommended)
# Download from https://developer.nvidia.com/cuda-toolkit

# Install cuDNN
# Download from https://developer.nvidia.com/cudnn

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]>=2.13.0
```

#### Verify GPU Installation

```python
import tensorflow as tf
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("TensorFlow version:", tf.__version__)
```

### Optional Dependencies

```bash
# Hydrographic tools
pip install pyproj>=3.3.0
pip install rasterio>=1.3.0
pip install fiona>=1.8.0
pip install geopandas>=0.11.0

# Development tools
pip install pytest>=7.0.0
pip install pytest-cov>=3.0.0
pip install black>=22.0.0
pip install flake8>=4.0.0
pip install mypy>=0.910

# Documentation
pip install sphinx>=4.0.0
pip install sphinx-rtd-theme>=1.0.0
pip install myst-parser>=0.17.0
```

## Verification

### Test Installation

```bash
# Test basic import
python -c "import bathymetric_cae; print('Installation successful!')"

# Test command line interface
bathymetric-cae --help

# Run test suite
pytest tests/
```

### Sample Processing Test

```bash
# Create test configuration
python -c "
from bathymetric_cae import Config
config = Config()
config.save('test_config.json')
print('Test configuration created')
"

# Test with sample data (if available)
python main.py --config test_config.json --help
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'gdal'

**Solution:**
```bash
# On Windows with conda
conda install gdal -c conda-forge

# On Linux
sudo apt-get install python3-gdal
```

#### GPU not detected

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall TensorFlow with GPU support
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

#### Memory issues during processing

**Solution:**
```bash
# Reduce batch size and grid size in configuration
python -c "
from bathymetric_cae import Config
config = Config()
config.batch_size = 4
config.grid_size = 256
config.save('memory_optimized_config.json')
"
```

#### Permission denied errors

**Solution:**
```bash
# Create directories with proper permissions
mkdir -p logs plots expert_reviews
chmod 755 logs plots expert_reviews
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
python -c "
from bathymetric_cae.utils import log_system_info
log_system_info()
"
```

#### GPU Optimization
```bash
# Set GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Limit GPU memory
export TF_GPU_MEMORY_LIMIT=4096
```

### Getting Help

- **Documentation**: [Link to full documentation]
- **Issues**: [GitHub Issues](https://github.com/username/bathymetric-cae/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/bathymetric-cae/discussions)
- **Email**: contact@example.com

---

# docs/usage.md

# Usage Guide

## Quick Start

### Basic Processing

```bash
# Process bathymetric files with default settings
python main.py --input /path/to/input --output /path/to/output

# Enable enhanced features
python main.py \
  --input /path/to/input \
  --output /path/to/output \
  --enable-adaptive \
  --enable-expert-review \
  --enable-constitutional
```

### Using Configuration Files

```bash
# Create default configuration
python -c "
from bathymetric_cae import Config
config = Config()
config.save('my_config.json')
"

# Edit configuration as needed, then run
python main.py --config my_config.json
```

## Configuration

### Basic Configuration

```json
{
  "ensemble_size": 3,
  "grid_size": 512,
  "epochs": 100,
  "batch_size": 8,
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "enable_constitutional_constraints": true,
  "quality_threshold": 0.7
}
```

### Advanced Configuration

```json
{
  "ensemble_size": 5,
  "grid_size": 1024,
  "epochs": 200,
  "batch_size": 4,
  "learning_rate": 0.0005,