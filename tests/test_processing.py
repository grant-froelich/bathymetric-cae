# tests/test_processing.py  
"""Tests for processing pipeline."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from bathymetric_cae.processing import (
    BathymetricProcessor,
    EnhancedBathymetricCAEPipeline
)
from bathymetric_cae.config import Config
from tests import create_test_data, create_temp_config, get_test_data_dir


class TestBathymetricProcessor:
    """Test bathymetric data processor."""
    
    def test_processor_creation(self):
        """Test processor initialization."""
        config = create_temp_config()
        processor = BathymetricProcessor(config)
        
        assert processor.config == config
        assert hasattr(processor, 'logger')
    
    def test_data_validation(self):
        """Test data validation."""
        config = create_temp_config()
        processor = BathymetricProcessor(config)
        
        # Valid data
        valid_data = create_test_data((100, 100))
        cleaned = processor._validate_and_clean_data(valid_data, Path("test.txt"))
        assert cleaned.shape == valid_data.shape
        
        # Data with NaN values
        invalid_data = valid_data.copy()
        invalid_data[0:10, 0:10] = np.nan
        cleaned = processor._validate_and_clean_data(invalid_data, Path("test.txt"))
        assert not np.any(np.isnan(cleaned))
    
    def test_robust_normalization(self):
        """Test robust normalization."""
        config = create_temp_config()
        processor = BathymetricProcessor(config)
        
        data = np.random.normal(0, 100, (50, 50))
        normalized = processor._robust_normalize(data)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert normalized.shape == data.shape
    
    @pytest.mark.skipif(True, reason="Requires test files")
    def test_file_validation(self):
        """Test file validation."""
        config = create_temp_config()
        processor = BathymetricProcessor(config)
        
        # This would require actual test files
        # For now, just test the interface
        assert hasattr(processor, 'validate_file')


class TestEnhancedPipeline:
    """Test enhanced processing pipeline."""
    
    def test_pipeline_creation(self):
        """Test pipeline initialization."""
        config = create_temp_config()
        pipeline = EnhancedBathymetricCAEPipeline(config)
        
        assert pipeline.config == config
        assert hasattr(pipeline, 'ensemble')
        assert hasattr(pipeline, 'quality_metrics')
    
    def test_file_list_processing(self):
        """Test getting valid files."""
        config = create_temp_config()
        pipeline = EnhancedBathymetricCAEPipeline(config)
        
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test files
            (temp_path / "test1.tif").touch()
            (temp_path / "test2.bag").touch()
            (temp_path / "test3.txt").touch()  # Unsupported
            
            valid_files = pipeline._get_valid_files(str(temp_path))
            
            assert len(valid_files) == 2
            assert all(f.suffix in config.supported_formats for f in valid_files)
