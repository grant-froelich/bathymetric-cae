# tests/conftest.py
"""Pytest configuration and fixtures."""

import pytest
import tempfile
import numpy as np
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_config():
    """Create test configuration."""
    from tests import create_temp_config
    return create_temp_config()

@pytest.fixture
def test_data():
    """Create test bathymetric data."""
    from tests import create_test_data
    return create_test_data()

@pytest.fixture
def test_data_with_uncertainty():
    """Create test data with uncertainty."""
    from tests import create_test_data
    return create_test_data(with_uncertainty=True)

@pytest.fixture(scope="session")
def test_database():
    """Create test database for session."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)