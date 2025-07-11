# tests/test_processing_data_processor.py
"""
Test data processing functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from config.config import Config
from processing.data_processor import BathymetricProcessor


class TestBathymetricProcessor:
    """Test bathymetric data processor."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for processor testing."""
        return Config(
            grid_size=64,
            supported_formats=['.bag', '.tif', '.asc']
        )
    
    @patch('processing.data_processor.gdal.Open')
    def test_preprocess_standard_file(self, mock_gdal_open, sample_config, mock_gdal_dataset):
        """Test preprocessing of standard file."""
        mock_gdal_open.return_value = mock_gdal_dataset
        
        processor = BathymetricProcessor(sample_config)
        
        # Test processing
        file_path = Path("test_file.tif")
        input_data, shape, metadata = processor.preprocess_bathymetric_grid(file_path)
        
        assert input_data.shape[-1] == 1  # Single channel
        assert shape == (64, 64)
        assert 'geotransform' in metadata
    
    @patch('processing.data_processor.gdal.Open')
    def test_preprocess_bag_file(self, mock_gdal_open, sample_config, sample_depth_data, sample_uncertainty_data):
        """Test preprocessing of BAG file with uncertainty."""
        # Mock BAG dataset with two bands
        mock_dataset = Mock()
        mock_dataset.RasterCount = 2
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_dataset.GetMetadata.return_value = {}
        
        # Mock bands
        mock_depth_band = Mock()
        mock_depth_band.ReadAsArray.return_value = sample_depth_data
        mock_uncertainty_band = Mock()
        mock_uncertainty_band.ReadAsArray.return_value = sample_uncertainty_data
        
        def get_band(band_num):
            if band_num == 1:
                return mock_depth_band
            else:
                return mock_uncertainty_band
        
        mock_dataset.GetRasterBand.side_effect = get_band
        mock_gdal_open.return_value = mock_dataset
        
        processor = BathymetricProcessor(sample_config)
        
        # Test processing
        file_path = Path("test_file.bag")
        input_data, shape, metadata = processor.preprocess_bathymetric_grid(file_path)
        
        assert input_data.shape[-1] == 2  # Depth + uncertainty channels
        assert shape == (64, 64)
    
    def test_unsupported_format(self, sample_config):
        """Test handling of unsupported file format."""
        processor = BathymetricProcessor(sample_config)
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            processor.preprocess_bathymetric_grid(Path("test_file.xyz"))
    
    def test_data_validation_and_cleaning(self, sample_config):
        """Test data validation and cleaning."""
        processor = BathymetricProcessor(sample_config)
        
        # Test with invalid values
        data_with_nans = np.array([[1, 2, np.nan], [4, np.inf, 6], [7, 8, 9]], dtype=np.float32)
        cleaned_data = processor._validate_and_clean_data(data_with_nans, Path("test.tif"))
        
        assert not np.isnan(cleaned_data).any()
        assert not np.isinf(cleaned_data).any()
    
    def test_robust_normalization(self, sample_config):
        """Test robust normalization."""
        processor = BathymetricProcessor(sample_config)
        
        # Test with outliers
        data = np.array([1, 2, 3, 4, 5, 100], dtype=np.float32)  # 100 is outlier
        normalized = processor._robust_normalize(data)
        
        assert 0 <= normalized.min() <= normalized.max() <= 1
        assert not np.isnan(normalized).any()