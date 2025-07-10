"""
Pytest Configuration and Fixtures for Bathymetric CAE Tests

This module provides common test fixtures, configuration, and utilities
for testing the bathymetric CAE package.

Author: Bathymetric CAE Team
License: MIT
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, Generator, Tuple
from unittest.mock import Mock, patch

# Add the package to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from bathymetric_cae import Config


# Test configuration
TEST_CONFIG = {
    "test_data_dir": Path(__file__).parent / "data",
    "temp_dir_prefix": "bathymetric_cae_test_",
    "default_grid_size": 64,  # Small for fast tests
    "default_batch_size": 2,
    "default_epochs": 2,
    "timeout_seconds": 30
}


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration dictionary."""
    return TEST_CONFIG


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests."""
    # Configure TensorFlow for testing if available
    if TENSORFLOW_AVAILABLE:
        # Set memory growth for GPU
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # Memory growth must be set before GPUs have been initialized
        
        # Set log level to reduce verbose output
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Configure test data directory
    test_data_dir = TEST_CONFIG["test_data_dir"]
    test_data_dir.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    # Note: Individual test cleanup is handled by temp_dir fixture


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test isolation."""
    temp_dir = Path(tempfile.mkdtemp(prefix=TEST_CONFIG["temp_dir_prefix"]))
    
    try:
        yield temp_dir
    finally:
        # Cleanup temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config(temp_dir) -> Config:
    """Create sample configuration for testing."""
    return Config(
        input_folder=str(temp_dir / "input"),
        output_folder=str(temp_dir / "output"),
        model_path=str(temp_dir / "test_model.h5"),
        epochs=TEST_CONFIG["default_epochs"],
        batch_size=TEST_CONFIG["default_batch_size"],
        grid_size=TEST_CONFIG["default_grid_size"],
        base_filters=8,
        depth=2,
        log_level="ERROR"  # Reduce logging during tests
    )


@pytest.fixture
def sample_bathymetric_data() -> np.ndarray:
    """Generate sample bathymetric data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Create realistic bathymetric data
    size = TEST_CONFIG["default_grid_size"]
    
    # Generate base bathymetry with some structure
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # Create depth data with some patterns
    depth_data = -20 + 10 * np.sin(X * 0.5) * np.cos(Y * 0.3) + 5 * np.random.randn(size, size)
    
    return depth_data.astype(np.float32)


@pytest.fixture
def sample_uncertainty_data() -> np.ndarray:
    """Generate sample uncertainty data for testing."""
    np.random.seed(43)  # Different seed for uncertainty
    
    size = TEST_CONFIG["default_grid_size"]
    
    # Generate uncertainty data (always positive)
    uncertainty_data = 0.5 + 2.0 * np.random.rand(size, size)
    
    return uncertainty_data.astype(np.float32)


@pytest.fixture
def sample_multi_channel_data(sample_bathymetric_data, sample_uncertainty_data) -> np.ndarray:
    """Generate multi-channel data (depth + uncertainty)."""
    return np.stack([sample_bathymetric_data, sample_uncertainty_data], axis=-1)


@pytest.fixture
def mock_gdal_dataset():
    """Mock GDAL dataset for testing without real files."""
    mock_dataset = Mock()
    mock_dataset.RasterXSize = TEST_CONFIG["default_grid_size"]
    mock_dataset.RasterYSize = TEST_CONFIG["default_grid_size"]
    mock_dataset.RasterCount = 2
    mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
    mock_dataset.GetProjection.return_value = "EPSG:4326"
    mock_dataset.GetMetadata.return_value = {"source": "test_data"}
    
    # Mock raster band
    mock_band = Mock()
    mock_band.ReadAsArray.return_value = np.random.rand(
        TEST_CONFIG["default_grid_size"], 
        TEST_CONFIG["default_grid_size"]
    ).astype(np.float32)
    mock_band.DataType = gdal.GDT_Float32 if GDAL_AVAILABLE else 6
    mock_band.GetNoDataValue.return_value = -9999
    
    mock_dataset.GetRasterBand.return_value = mock_band
    
    return mock_dataset


@pytest.fixture
def create_test_files(temp_dir, sample_bathymetric_data):
    """Create test files in various formats for testing."""
    def _create_files(formats=None):
        if formats is None:
            formats = ['.tif', '.asc']  # Exclude .bag for basic tests
        
        created_files = []
        
        for fmt in formats:
            filename = f"test_bathymetry{fmt}"
            filepath = temp_dir / filename
            
            if fmt == '.tif':
                # Create a simple text file simulating GeoTIFF
                # In real tests, this would be a proper GDAL-created file
                filepath.write_text(f"# Mock GeoTIFF file\n# Size: {sample_bathymetric_data.shape}")
            elif fmt == '.asc':
                # Create ASCII grid format
                header = f"""ncols {sample_bathymetric_data.shape[1]}
nrows {sample_bathymetric_data.shape[0]}
xllcorner 0
yllcorner 0
cellsize 1
NODATA_value -9999
"""
                data_str = '\n'.join([' '.join(map(str, row)) for row in sample_bathymetric_data])
                filepath.write_text(header + data_str)
            else:
                # Generic test file
                filepath.write_text(f"# Test bathymetric file {fmt}")
            
            created_files.append(filepath)
        
        return created_files
    
    return _create_files


@pytest.fixture
def mock_tensorflow_model():
    """Mock TensorFlow model for testing without actual training."""
    if not TENSORFLOW_AVAILABLE:
        # Return a simple mock if TensorFlow is not available
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(1, 64, 64, 1)
        mock_model.fit.return_value = Mock()
        mock_model.save.return_value = None
        mock_model.count_params.return_value = 1000
        return mock_model
    
    # Create a minimal real TensorFlow model for testing
    inputs = tf.keras.layers.Input(shape=(64, 64, 1))
    x = tf.keras.layers.Conv2D(4, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer='adam', loss='mse')
    
    return model


@pytest.fixture
def capture_logs():
    """Capture log output for testing logging functionality."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    # Get the bathymetric_cae logger
    logger = logging.getLogger('bathymetric_cae')
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


@pytest.fixture
def mock_memory_info():
    """Mock memory information for testing memory utilities."""
    return {
        'rss_mb': 1024.0,
        'vms_mb': 2048.0,
        'percent': 25.0,
        'available_mb': 8192.0,
        'total_mb': 16384.0
    }


@pytest.fixture
def mock_gpu_info():
    """Mock GPU information for testing GPU utilities."""
    return {
        'tensorflow_version': '2.13.0',
        'gpu_available': True,
        'num_physical_gpus': 1,
        'num_logical_gpus': 1,
        'physical_gpus': ['/physical_device:GPU:0'],
        'logical_gpus': ['/device:GPU:0'],
        'cuda_available': True,
        'gpu_support': True
    }


# Pytest markers for conditional testing
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "tensorflow: mark test as requiring TensorFlow"
    )
    config.addinivalue_line(
        "markers", "gdal: mark test as requiring GDAL"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle conditional tests."""
    skip_tensorflow = pytest.mark.skip(reason="TensorFlow not available")
    skip_gdal = pytest.mark.skip(reason="GDAL not available")
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        # Skip TensorFlow tests if not available
        if "tensorflow" in item.keywords and not TENSORFLOW_AVAILABLE:
            item.add_marker(skip_tensorflow)
        
        # Skip GDAL tests if not available
        if "gdal" in item.keywords and not GDAL_AVAILABLE:
            item.add_marker(skip_gdal)
        
        # Skip GPU tests if no GPU available
        if "gpu" in item.keywords:
            gpu_available = False
            if TENSORFLOW_AVAILABLE:
                gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            if not gpu_available:
                item.add_marker(skip_gpu)


# Helper functions for tests
def assert_array_properties(array: np.ndarray, 
                          expected_shape: Tuple = None,
                          expected_dtype: np.dtype = None,
                          expected_range: Tuple = None):
    """Assert properties of numpy arrays in tests."""
    assert isinstance(array, np.ndarray), f"Expected numpy array, got {type(array)}"
    
    if expected_shape is not None:
        assert array.shape == expected_shape, f"Expected shape {expected_shape}, got {array.shape}"
    
    if expected_dtype is not None:
        assert array.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {array.dtype}"
    
    if expected_range is not None:
        min_val, max_val = expected_range
        assert np.min(array) >= min_val, f"Array minimum {np.min(array)} below expected {min_val}"
        assert np.max(array) <= max_val, f"Array maximum {np.max(array)} above expected {max_val}"


def assert_file_exists_and_valid(filepath: Path, min_size: int = 0):
    """Assert that a file exists and has valid properties."""
    assert filepath.exists(), f"File does not exist: {filepath}"
    assert filepath.is_file(), f"Path is not a file: {filepath}"
    assert filepath.stat().st_size >= min_size, f"File too small: {filepath.stat().st_size} < {min_size}"


def create_minimal_model_for_testing():
    """Create minimal TensorFlow model for testing."""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    inputs = tf.keras.layers.Input(shape=(64, 64, 1))
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# Test data generators
class TestDataGenerator:
    """Generate various types of test data."""
    
    @staticmethod
    def create_bathymetric_grid(size: int = 64, 
                               depth_range: Tuple[float, float] = (-100, 0),
                               add_noise: bool = True) -> np.ndarray:
        """Create synthetic bathymetric grid."""
        np.random.seed(42)
        
        # Create coordinate grids
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Create bathymetric features
        depth_min, depth_max = depth_range
        
        # Base depth field with some structure
        depth = depth_min + (depth_max - depth_min) * (
            0.5 + 0.3 * np.sin(3 * X) * np.cos(2 * Y) + 
            0.2 * np.sin(5 * (X + Y))
        )
        
        # Add noise if requested
        if add_noise:
            noise_level = 0.1 * (depth_max - depth_min)
            depth += noise_level * np.random.randn(size, size)
        
        return depth.astype(np.float32)
    
    @staticmethod
    def create_uncertainty_grid(size: int = 64,
                              uncertainty_range: Tuple[float, float] = (0.1, 2.0)) -> np.ndarray:
        """Create synthetic uncertainty grid."""
        np.random.seed(43)
        
        unc_min, unc_max = uncertainty_range
        
        # Create uncertainty field (higher near edges/complex areas)
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Distance from center (higher uncertainty near edges)
        dist_from_center = np.sqrt(X**2 + Y**2)
        uncertainty = unc_min + (unc_max - unc_min) * (
            0.3 * dist_from_center + 0.7 * np.random.rand(size, size)
        )
        
        return uncertainty.astype(np.float32)


# Export test utilities
__all__ = [
    'TEST_CONFIG',
    'assert_array_properties',
    'assert_file_exists_and_valid',
    'create_minimal_model_for_testing',
    'TestDataGenerator',
    'TENSORFLOW_AVAILABLE',
    'GDAL_AVAILABLE'
]
