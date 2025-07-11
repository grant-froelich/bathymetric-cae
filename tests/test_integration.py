# tests/test_integration.py
"""
Integration tests for the complete pipeline.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

from config.config import Config
from processing.pipeline import EnhancedBathymetricCAEPipeline


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return Config(
            epochs=2,
            batch_size=1,
            grid_size=32,
            ensemble_size=1,
            enable_adaptive_processing=True,
            enable_expert_review=True,
            enable_constitutional_constraints=True,
            validation_split=0.5
        )
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        temp_dir = tempfile.mkdtemp()
        workspace = Path(temp_dir)
        
        # Create directory structure
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "logs").mkdir()
        (workspace / "plots").mkdir()
        (workspace / "expert_reviews").mkdir()
        
        yield workspace
        shutil.rmtree(temp_dir)
    
    def create_mock_bathymetric_file(self, file_path: Path, data_shape=(32, 32)):
        """Create a mock bathymetric file for testing."""
        # This would normally create a real GDAL file, but for testing
        # we'll just create a placeholder file
        file_path.touch()
        return np.random.uniform(-100, -10, data_shape).astype(np.float32)
    
    @patch('processing.data_processor.gdal.Open')
    @patch('processing.pipeline.Path.glob')
    def test_complete_pipeline_execution(self, mock_glob, mock_gdal_open, 
                                       integration_config, temp_workspace):
        """Test complete pipeline execution."""
        # Setup mock files
        input_files = [
            temp_workspace / "input" / "test1.bag",
            temp_workspace / "input" / "test2.bag"
        ]
        
        for file_path in input_files:
            self.create_mock_bathymetric_file(file_path)
        
        mock_glob.return_value = input_files
        
        # Mock GDAL dataset
        mock_dataset = Mock()
        mock_dataset.RasterCount = 1
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_dataset.GetMetadata.return_value = {}
        
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = np.random.random((32, 32)).astype(np.float32)
        mock_dataset.GetRasterBand.return_value = mock_band
        mock_gdal_open.return_value = mock_dataset
        
        # Update config paths
        integration_config.input_folder = str(temp_workspace / "input")
        integration_config.output_folder = str(temp_workspace / "output")
        
        # Create and run pipeline
        pipeline = EnhancedBathymetricCAEPipeline(integration_config)
        
        # Mock the training to avoid long execution time
        with patch.object(pipeline, '_train_ensemble') as mock_train:
            mock_train.return_value = [Mock() for _ in range(integration_config.ensemble_size)]
            
            # Run pipeline
            pipeline.run(
                str(temp_workspace / "input"),
                str(temp_workspace / "output"),
                "test_model.h5"
            )
        
        # Verify outputs were created
        assert (temp_workspace / "output").exists()
        assert (Path("enhanced_processing_summary.json")).exists()
    
    def test_error_handling_in_pipeline(self, integration_config, temp_workspace):
        """Test error handling in pipeline."""
        pipeline = EnhancedBathymetricCAEPipeline(integration_config)
        
        # Test with non-existent input folder
        with pytest.raises(FileNotFoundError):
            pipeline.run(
                str(temp_workspace / "nonexistent"),
                str(temp_workspace / "output"),
                "test_model.h5"
            )
    
    @patch('processing.data_processor.gdal.Open')
    def test_adaptive_processing_integration(self, mock_gdal_open, integration_config):
        """Test integration of adaptive processing features."""
        # Mock different seafloor types
        shallow_data = np.full((32, 32), -50.0, dtype=np.float32)  # Shallow coastal
        deep_data = np.full((32, 32), -3000.0, dtype=np.float32)   # Deep ocean
        
        mock_dataset = Mock()
        mock_dataset.RasterCount = 1
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_dataset.GetProjection.return_value = "EPSG:4326"
        mock_dataset.GetMetadata.return_value = {}
        
        pipeline = EnhancedBathymetricCAEPipeline(integration_config)
        
        # Test shallow coastal processing
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = shallow_data
        mock_dataset.GetRasterBand.return_value = mock_band
        mock_gdal_open.return_value = mock_dataset
        
        from processing.data_processor import BathymetricProcessor
        processor = BathymetricProcessor(integration_config)
        
        input_data, _, _ = processor.preprocess_bathymetric_grid(Path("test_shallow.bag"))
        depth_data = input_data[..., 0]
        
        # Get adaptive parameters
        adaptive_params = pipeline.adaptive_processor.get_processing_parameters(depth_data)
        
        # Verify shallow coastal parameters
        assert adaptive_params['smoothing_factor'] == 0.3
        assert adaptive_params['feature_preservation_weight'] == 0.9