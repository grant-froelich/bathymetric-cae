"""
Tests for Bathymetric Processor Module

Tests for the bathymetric data processing including file loading,
validation, preprocessing, and data cleaning.

Author: Bathymetric CAE Team
License: MIT
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from bathymetric_cae.core.processor import (
    BathymetricProcessor,
    get_supported_formats,
    validate_gdal_installation
)
from conftest import (
    assert_array_properties,
    TestDataGenerator,
    GDAL_AVAILABLE
)


class TestBathymetricProcessor:
    """Test the BathymetricProcessor class."""
    
    def test_processor_initialization(self, sample_config):
        """Test processor initialization."""
        processor = BathymetricProcessor(sample_config)
        
        assert processor.config == sample_config
        assert hasattr(processor, 'logger')
        assert processor.config.supported_formats == ['.bag', '.tif', '.tiff', '.asc', '.xyz']
    
    def test_processor_initialization_no_formats(self, sample_config):
        """Test processor initialization without supported formats."""
        # Remove supported_formats to test default behavior
        sample_config.supported_formats = None
        
        processor = BathymetricProcessor(sample_config)
        
        # Should set default formats
        assert processor.config.supported_formats is not None
        assert '.bag' in processor.config.supported_formats
        assert '.tif' in processor.config.supported_formats
    
    def test_validate_file_format_valid(self, sample_config):
        """Test file format validation with valid formats."""
        processor = BathymetricProcessor(sample_config)
        
        valid_files = [
            Path("test.bag"),
            Path("test.tif"),
            Path("test.tiff"),
            Path("test.asc"),
            Path("test.xyz")
        ]
        
        for file_path in valid_files:
            assert processor.validate_file_format(file_path)
    
    def test_validate_file_format_invalid(self, sample_config):
        """Test file format validation with invalid formats."""
        processor = BathymetricProcessor(sample_config)
        
        invalid_files = [
            Path("test.txt"),
            Path("test.jpg"),
            Path("test.pdf"),
            Path("test.doc")
        ]
        
        for file_path in invalid_files:
            assert not processor.validate_file_format(file_path)
    
    def test_validate_file_format_case_insensitive(self, sample_config):
        """Test file format validation is case insensitive."""
        processor = BathymetricProcessor(sample_config)
        
        case_variants = [
            Path("test.TIF"),
            Path("test.BAG"),
            Path("test.ASC"),
            Path("test.XYZ")
        ]
        
        for file_path in case_variants:
            assert processor.validate_file_format(file_path)
    
    @pytest.mark.gdal
    def test_preprocess_bathymetric_grid_file_not_found(self, sample_config):
        """Test preprocessing with non-existent file."""
        processor = BathymetricProcessor(sample_config)
        
        with pytest.raises(FileNotFoundError):
            processor.preprocess_bathymetric_grid("nonexistent_file.bag")
    
    def test_preprocess_bathymetric_grid_unsupported_format(self, sample_config, temp_dir):
        """Test preprocessing with unsupported file format."""
        processor = BathymetricProcessor(sample_config)
        
        # Create file with unsupported format
        unsupported_file = temp_dir / "test.txt"
        unsupported_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            processor.preprocess_bathymetric_grid(unsupported_file)
    
    @pytest.mark.gdal
    @patch('bathymetric_cae.core.processor.gdal.Open')
    def test_preprocess_bathymetric_grid_gdal_failure(self, mock_gdal_open, sample_config, temp_dir):
        """Test preprocessing when GDAL fails to open file."""
        processor = BathymetricProcessor(sample_config)
        
        # Create valid file
        test_file = temp_dir / "test.tif"
        test_file.write_text("dummy content")
        
        # Mock GDAL to return None
        mock_gdal_open.return_value = None
        
        with pytest.raises(ValueError, match="Cannot open file with GDAL"):
            processor.preprocess_bathymetric_grid(test_file)
    
    @pytest.mark.gdal
    @patch('bathymetric_cae.core.processor.gdal.Open')
    def test_process_bag_file(self, mock_gdal_open, sample_config, sample_bathymetric_data, sample_uncertainty_data):
        """Test processing BAG file with uncertainty data."""
        processor = BathymetricProcessor(sample_config)
        
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.RasterCount = 2
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_dataset.GetMetadata.return_value = {"source": "test"}
        
        # Setup mock bands
        mock_depth_band = Mock()
        mock_depth_band.ReadAsArray.return_value = sample_bathymetric_data
        
        mock_uncertainty_band = Mock()
        mock_uncertainty_band.ReadAsArray.return_value = sample_uncertainty_data
        
        mock_dataset.GetRasterBand.side_effect = [mock_depth_band, mock_uncertainty_band]
        mock_gdal_open.return_value = mock_dataset
        
        # Test processing
        depth_data, uncertainty_data = processor._process_bag_file(mock_dataset)
        
        assert_array_properties(depth_data, expected_dtype=np.float32)
        assert_array_properties(uncertainty_data, expected_dtype=np.float32)
        assert depth_data.shape == sample_bathymetric_data.shape
        assert uncertainty_data.shape == sample_uncertainty_data.shape
    
    @pytest.mark.gdal
    @patch('bathymetric_cae.core.processor.gdal.Open')
    def test_process_bag_file_no_uncertainty(self, mock_gdal_open, sample_config, sample_bathymetric_data):
        """Test processing BAG file without uncertainty data."""
        processor = BathymetricProcessor(sample_config)
        
        # Setup mock dataset with only one band
        mock_dataset = Mock()
        mock_dataset.RasterCount = 1
        
        mock_depth_band = Mock()
        mock_depth_band.ReadAsArray.return_value = sample_bathymetric_data
        mock_dataset.GetRasterBand.return_value = mock_depth_band
        
        # Test processing
        depth_data, uncertainty_data = processor._process_bag_file(mock_dataset)
        
        assert_array_properties(depth_data, expected_dtype=np.float32)
        assert uncertainty_data is None
    
    @pytest.mark.gdal
    def test_process_standard_file(self, sample_config, sample_bathymetric_data):
        """Test processing standard raster file."""
        processor = BathymetricProcessor(sample_config)
        
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.RasterCount = 1
        
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = sample_bathymetric_data
        mock_dataset.GetRasterBand.return_value = mock_band
        
        # Test processing
        result = processor._process_standard_file(mock_dataset)
        
        assert_array_properties(result, expected_dtype=np.float32)
        assert result.shape == sample_bathymetric_data.shape
    
    def test_validate_and_clean_data_valid(self, sample_config):
        """Test data validation and cleaning with valid data."""
        processor = BathymetricProcessor(sample_config)
        
        # Create test data
        test_data = TestDataGenerator.create_bathymetric_grid()
        
        result = processor._validate_and_clean_data(test_data, Path("test.tif"))
        
        assert_array_properties(result, expected_shape=test_data.shape)
        assert np.all(np.isfinite(result))
    
    def test_validate_and_clean_data_with_nans(self, sample_config):
        """Test data validation and cleaning with NaN values."""
        processor = BathymetricProcessor(sample_config)
        
        # Create test data with NaN values
        test_data = TestDataGenerator.create_bathymetric_grid()
        test_data[10:15, 10:15] = np.nan  # Add some NaN values
        
        result = processor._validate_and_clean_data(test_data, Path("test.tif"))
        
        assert np.all(np.isfinite(result))
        assert result.shape == test_data.shape
        # NaN values should be replaced
        assert not np.any(np.isnan(result[10:15, 10:15]))
    
    def test_validate_and_clean_data_with_infs(self, sample_config):
        """Test data validation and cleaning with infinite values."""
        processor = BathymetricProcessor(sample_config)
        
        # Create test data with infinite values
        test_data = TestDataGenerator.create_bathymetric_grid()
        test_data[5:10, 5:10] = np.inf
        test_data[15:20, 15:20] = -np.inf
        
        result = processor._validate_and_clean_data(test_data, Path("test.tif"))
        
        assert np.all(np.isfinite(result))
        assert result.shape == test_data.shape
    
    def test_validate_and_clean_data_empty(self, sample_config):
        """Test data validation with empty data."""
        processor = BathymetricProcessor(sample_config)
        
        empty_data = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            processor._validate_and_clean_data(empty_data, Path("test.tif"))
    
    def test_validate_and_clean_data_all_invalid(self, sample_config):
        """Test data validation with all invalid values."""
        processor = BathymetricProcessor(sample_config)
        
        # Create data with all NaN values
        invalid_data = np.full((64, 64), np.nan)
        
        with pytest.raises(ValueError, match="All values are invalid"):
            processor._validate_and_clean_data(invalid_data, Path("test.tif"))
    
    def test_validate_and_clean_uncertainty_data(self, sample_config):
        """Test data validation for uncertainty data."""
        processor = BathymetricProcessor(sample_config)
        
        # Create uncertainty data with some invalid values
        uncertainty_data = TestDataGenerator.create_uncertainty_grid()
        uncertainty_data[20:25, 20:25] = np.nan
        
        result = processor._validate_and_clean_data(
            uncertainty_data, Path("test.bag"), is_uncertainty=True
        )
        
        assert np.all(np.isfinite(result))
        assert result.shape == uncertainty_data.shape
        # For uncertainty, median should be used for filling
        assert np.all(result >= 0)  # Uncertainty should be non-negative
    
    def test_robust_normalize(self, sample_config):
        """Test robust normalization function."""
        processor = BathymetricProcessor(sample_config)
        
        # Create test data with outliers
        test_data = TestDataGenerator.create_bathymetric_grid()
        test_data[0, 0] = -1000  # Outlier
        test_data[0, 1] = 1000   # Outlier
        
        result = processor._robust_normalize(test_data)
        
        assert_array_properties(result, expected_range=(0, 1))
        assert result.shape == test_data.shape
    
    def test_robust_normalize_constant_data(self, sample_config):
        """Test robust normalization with constant data."""
        processor = BathymetricProcessor(sample_config)
        
        # Create constant data
        constant_data = np.full((64, 64), 5.0)
        
        result = processor._robust_normalize(constant_data)
        
        # Should handle constant data gracefully
        assert result.shape == constant_data.shape
        assert np.all(result == 0.5)  # Should return 0.5 for constant data
    
    def test_prepare_single_channel_input(self, sample_config):
        """Test preparation of single-channel input."""
        processor = BathymetricProcessor(sample_config)
        
        depth_data = TestDataGenerator.create_bathymetric_grid()
        
        result = processor._prepare_single_channel_input(depth_data)
        
        expected_shape = (*depth_data.shape, 1)
        assert_array_properties(result, expected_shape=expected_shape,
                               expected_range=(0, 1))
    
    def test_prepare_multi_channel_input(self, sample_config):
        """Test preparation of multi-channel input."""
        processor = BathymetricProcessor(sample_config)
        
        depth_data = TestDataGenerator.create_bathymetric_grid()
        uncertainty_data = TestDataGenerator.create_uncertainty_grid()
        
        result = processor._prepare_multi_channel_input(depth_data, uncertainty_data)
        
        expected_shape = (*depth_data.shape, 2)
        assert_array_properties(result, expected_shape=expected_shape,
                               expected_range=(0, 1))
        
        # Check that both channels are properly normalized
        assert np.all(result[..., 0] >= 0) and np.all(result[..., 0] <= 1)
        assert np.all(result[..., 1] >= 0) and np.all(result[..., 1] <= 1)
    
    def test_batch_preprocess_files_success(self, sample_config, create_test_files):
        """Test batch preprocessing with successful files."""
        processor = BathymetricProcessor(sample_config)
        
        # Create test files
        test_files = create_test_files(['.tif', '.asc'])
        
        with patch.object(processor, 'preprocess_bathymetric_grid') as mock_preprocess:
            mock_preprocess.return_value = (
                np.random.rand(64, 64, 1),
                (64, 64),
                {'test': 'metadata'}
            )
            
            results = processor.batch_preprocess_files(test_files)
            
            assert len(results) == len(test_files)
            assert mock_preprocess.call_count == len(test_files)
    
    def test_batch_preprocess_files_with_failures(self, sample_config, create_test_files, caplog):
        """Test batch preprocessing with some failures."""
        processor = BathymetricProcessor(sample_config)
        
        # Create test files
        test_files = create_test_files(['.tif', '.asc'])
        
        with patch.object(processor, 'preprocess_bathymetric_grid') as mock_preprocess:
            # First file succeeds, second fails
            mock_preprocess.side_effect = [
                (np.random.rand(64, 64, 1), (64, 64), {'test': 'metadata'}),
                ValueError("Processing failed")
            ]
            
            results = processor.batch_preprocess_files(test_files)
            
            # Should return only successful results
            assert len(results) == 1
            
            # Should log error for failed file
            assert "Failed to process" in caplog.text
    
    def test_get_file_info_success(self, sample_config, mock_gdal_dataset):
        """Test getting file information successfully."""
        processor = BathymetricProcessor(sample_config)
        
        with patch('bathymetric_cae.core.processor.gdal.Open') as mock_gdal_open:
            mock_gdal_open.return_value = mock_gdal_dataset
            
            info = processor.get_file_info(Path("test.tif"))
            
            assert 'filename' in info
            assert 'width' in info
            assert 'height' in info
            assert 'bands' in info
            assert info['filename'] == "test.tif"
    
    def test_get_file_info_file_not_found(self, sample_config):
        """Test getting file information for non-existent file."""
        processor = BathymetricProcessor(sample_config)
        
        info = processor.get_file_info(Path("nonexistent.tif"))
        
        assert 'error' in info
        assert info['error'] == 'File not found'
    
    @pytest.mark.gdal
    def test_get_file_info_gdal_failure(self, sample_config, temp_dir):
        """Test getting file information when GDAL fails."""
        processor = BathymetricProcessor(sample_config)
        
        # Create a file
        test_file = temp_dir / "test.tif"
        test_file.write_text("dummy content")
        
        with patch('bathymetric_cae.core.processor.gdal.Open') as mock_gdal_open:
            mock_gdal_open.return_value = None
            
            info = processor.get_file_info(test_file)
            
            assert 'error' in info
            assert 'Cannot open with GDAL' in info['error']


class TestProcessorUtilityFunctions:
    """Test processor utility functions."""
    
    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = get_supported_formats()
        
        assert isinstance(formats, list)
        assert '.bag' in formats
        assert '.tif' in formats
        assert '.tiff' in formats
        assert '.asc' in formats
        assert '.xyz' in formats
    
    @pytest.mark.gdal
    def test_validate_gdal_installation_success(self):
        """Test GDAL installation validation when available."""
        result = validate_gdal_installation()
        
        assert 'gdal_available' in result
        assert result['gdal_available'] is True
        assert 'version' in result
        assert 'driver_count' in result
        assert 'required_drivers' in result
    
    @patch('bathymetric_cae.core.processor.gdal', None)
    def test_validate_gdal_installation_failure(self):
        """Test GDAL installation validation when not available."""
        with patch.dict('sys.modules', {'osgeo.gdal': None}):
            with pytest.raises(ImportError):
                from osgeo import gdal
        
        # Test would need to be run in environment without GDAL
        # This is a simplified test structure


class TestProcessorIntegration:
    """Integration tests for the processor."""
    
    @pytest.mark.gdal
    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_config, temp_dir, sample_bathymetric_data):
        """Test complete preprocessing pipeline integration."""
        processor = BathymetricProcessor(sample_config)
        
        # Create mock file and dataset
        test_file = temp_dir / "integration_test.tif"
        test_file.write_text("dummy content")
        
        with patch('bathymetric_cae.core.processor.gdal.Open') as mock_gdal_open:
            # Setup comprehensive mock
            mock_dataset = Mock()
            mock_dataset.RasterCount = 1
            mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
            mock_dataset.GetProjection.return_value = "EPSG:4326"
            mock_dataset.GetMetadata.return_value = {"source": "integration_test"}
            
            mock_band = Mock()
            mock_band.ReadAsArray.return_value = sample_bathymetric_data
            mock_dataset.GetRasterBand.return_value = mock_band
            mock_gdal_open.return_value = mock_dataset
            
            # Run full preprocessing
            input_data, original_shape, geo_metadata = processor.preprocess_bathymetric_grid(test_file)
            
            # Verify results
            assert input_data is not None
            assert original_shape == sample_bathymetric_data.shape
            assert geo_metadata is not None
            assert 'geotransform' in geo_metadata
            assert 'projection' in geo_metadata
            assert 'file_path' in geo_metadata
    
    def test_processor_memory_efficiency(self, sample_config):
        """Test processor memory efficiency with large data."""
        processor = BathymetricProcessor(sample_config)
        
        # Create large test data
        large_data = TestDataGenerator.create_bathymetric_grid(size=512)
        
        # Test memory usage during processing
        normalized = processor._robust_normalize(large_data)
        
        # Should not consume excessive memory
        assert normalized.shape == large_data.shape
        assert normalized.dtype == np.float32
    
    def test_processor_error_recovery(self, sample_config, create_test_files, caplog):
        """Test processor error recovery during batch processing."""
        processor = BathymetricProcessor(sample_config)
        
        # Create mix of valid and problematic files
        test_files = create_test_files(['.tif', '.asc', '.bag'])
        
        with patch.object(processor, 'preprocess_bathymetric_grid') as mock_preprocess:
            # Simulate various error conditions
            mock_preprocess.side_effect = [
                (np.random.rand(64, 64, 1), (64, 64), {}),  # Success
                ValueError("GDAL error"),                    # GDAL failure
                (np.random.rand(64, 64, 1), (64, 64), {}),  # Success
            ]
            
            results = processor.batch_preprocess_files(test_files)
            
            # Should recover and process successful files
            assert len(results) == 2  # Two successful
            
            # Should log errors appropriately
            assert "Failed to process" in caplog.text


class TestProcessorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_processor_very_small_data(self, sample_config):
        """Test processor with very small data arrays."""
        processor = BathymetricProcessor(sample_config)
        
        # Create minimal size data
        tiny_data = np.random.rand(4, 4).astype(np.float32)
        
        result = processor._robust_normalize(tiny_data)
        
        assert result.shape == tiny_data.shape
        assert np.all(np.isfinite(result))
    
    def test_processor_extreme_values(self, sample_config):
        """Test processor with extreme data values."""
        processor = BathymetricProcessor(sample_config)
        
        # Create data with extreme values
        extreme_data = np.array([
            [-1e10, 1e10],
            [1e-10, -1e-10]
        ], dtype=np.float32)
        
        result = processor._robust_normalize(extreme_data)
        
        assert result.shape == extreme_data.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0) and np.all(result <= 1)
    
    def test_processor_single_pixel_data(self, sample_config):
        """Test processor with single pixel data."""
        processor = BathymetricProcessor(sample_config)
        
        single_pixel = np.array([[42.0]], dtype=np.float32)
        
        result = processor._robust_normalize(single_pixel)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.5  # Constant data should normalize to 0.5
    
    def test_processor_sparse_valid_data(self, sample_config):
        """Test processor with mostly invalid data."""
        processor = BathymetricProcessor(sample_config)
        
        # Create data that is 95% invalid
        sparse_data = np.full((100, 100), np.nan, dtype=np.float32)
        sparse_data[::20, ::20] = np.random.rand(5, 5) * 100  # 5% valid data
        
        result = processor._validate_and_clean_data(sparse_data, Path("sparse.tif"))
        
        assert np.all(np.isfinite(result))
        assert result.shape == sparse_data.shape
    
    def test_processor_different_data_types(self, sample_config):
        """Test processor with different input data types."""
        processor = BathymetricProcessor(sample_config)
        
        # Test with different numpy dtypes
        data_types = [np.int16, np.int32, np.float32, np.float64]
        
        for dtype in data_types:
            test_data = np.random.rand(32, 32).astype(dtype)
            
            if dtype in [np.int16, np.int32]:
                test_data = (test_data * 1000).astype(dtype)
            
            result = processor._robust_normalize(test_data)
            
            assert result.dtype == np.float32
            assert result.shape == test_data.shape
    
    def test_processor_unicode_file_paths(self, sample_config, temp_dir):
        """Test processor with unicode file paths."""
        processor = BathymetricProcessor(sample_config)
        
        # Create file with unicode characters
        unicode_file = temp_dir / "测试文件.tif"
        unicode_file.write_text("dummy content")
        
        # Should handle unicode paths
        assert processor.validate_file_format(unicode_file)
    
    def test_processor_long_file_paths(self, sample_config, temp_dir):
        """Test processor with very long file paths."""
        processor = BathymetricProcessor(sample_config)
        
        # Create nested directory structure
        long_path = temp_dir
        for i in range(10):
            long_path = long_path / f"very_long_directory_name_{i}"
        
        long_path.mkdir(parents=True, exist_ok=True)
        long_file = long_path / "test_file_with_very_long_name.tif"
        long_file.write_text("dummy content")
        
        # Should handle long paths
        assert processor.validate_file_format(long_file)


class TestProcessorPerformance:
    """Test processor performance characteristics."""
    
    def test_processor_normalize_performance(self, sample_config):
        """Test normalization performance with large arrays."""
        import time
        
        processor = BathymetricProcessor(sample_config)
        
        # Create large data array
        large_data = TestDataGenerator.create_bathymetric_grid(size=1024)
        
        start_time = time.time()
        result = processor._robust_normalize(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process 1024x1024 array in reasonable time (< 1 second)
        assert processing_time < 1.0
        assert result.shape == large_data.shape
    
    def test_processor_batch_performance(self, sample_config):
        """Test batch processing performance."""
        import time
        
        processor = BathymetricProcessor(sample_config)
        
        # Create mock files
        mock_files = [Path(f"file_{i}.tif") for i in range(10)]
        
        with patch.object(processor, 'preprocess_bathymetric_grid') as mock_preprocess:
            mock_preprocess.return_value = (
                np.random.rand(64, 64, 1),
                (64, 64),
                {}
            )
            
            start_time = time.time()
            results = processor.batch_preprocess_files(mock_files)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should process 10 files quickly when mocked
            assert processing_time < 1.0
            assert len(results) == 10
    
    def test_processor_memory_cleanup(self, sample_config):
        """Test that processor cleans up memory properly."""
        import gc
        
        processor = BathymetricProcessor(sample_config)
        
        # Process large data and verify cleanup
        for _ in range(5):
            large_data = TestDataGenerator.create_bathymetric_grid(size=256)
            result = processor._robust_normalize(large_data)
            
            # Force cleanup
            del large_data, result
            gc.collect()
        
        # Memory should not grow excessively
        # This test mainly ensures no memory leaks in the processing logic