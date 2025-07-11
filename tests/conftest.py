# tests/conftest.py
"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import tensorflow as tf

from config.config import Config
from models.ensemble import BathymetricEnsemble
from core.adaptive_processor import AdaptiveProcessor
from processing.data_processor import BathymetricProcessor


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return Config(
        input_folder="test_input",
        output_folder="test_output",
        epochs=5,
        batch_size=2,
        grid_size=64,
        ensemble_size=2,
        enable_adaptive_processing=True,
        enable_expert_review=True,
        enable_constitutional_constraints=True
    )


@pytest.fixture
def sample_depth_data():
    """Create sample depth data for testing."""
    np.random.seed(42)
    return np.random.uniform(-100, -10, (64, 64)).astype(np.float32)


@pytest.fixture
def sample_uncertainty_data():
    """Create sample uncertainty data for testing."""
    np.random.seed(42)
    return np.random.uniform(0.1, 2.0, (64, 64)).astype(np.float32)


@pytest.fixture
def mock_gdal_dataset(sample_depth_data):
    """Mock GDAL dataset for testing."""
    mock_dataset = Mock()
    mock_dataset.RasterCount = 1
    mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
    mock_dataset.GetProjection.return_value = "EPSG:4326"
    mock_dataset.GetMetadata.return_value = {}
    
    mock_band = Mock()
    mock_band.ReadAsArray.return_value = sample_depth_data
    mock_dataset.GetRasterBand.return_value = mock_band
    
    return mock_dataset