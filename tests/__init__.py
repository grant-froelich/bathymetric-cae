# tests/__init__.py
"""Test suite for Enhanced Bathymetric CAE Processing."""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add the parent directory to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test utilities
def create_test_data(shape: tuple = (512, 512), with_uncertainty: bool = False):
    """Create synthetic test data for testing."""
    np.random.seed(42)
    
    # Create realistic bathymetric data
    depth_data = np.random.normal(-100, 50, shape).astype(np.float32)
    
    # Add some features
    x, y = np.meshgrid(np.linspace(0, 10, shape[1]), np.linspace(0, 10, shape[0]))
    depth_data += 20 * np.sin(x) * np.cos(y)  # Add some structure
    
    if with_uncertainty:
        uncertainty_data = np.abs(np.random.normal(1, 0.5, shape)).astype(np.float32)
        return depth_data, uncertainty_data
    
    return depth_data

def create_temp_config():
    """Create temporary configuration for testing."""
    from bathymetric_cae.config import Config
    
    config = Config()
    config.grid_size = 128  # Smaller for faster tests
    config.epochs = 2  # Minimal for testing
    config.batch_size = 2
    config.ensemble_size = 2
    
    return config

def get_test_data_dir():
    """Get test data directory."""
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    return test_dir