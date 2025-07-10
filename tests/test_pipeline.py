"""
Tests for Pipeline Module

Tests for the main processing pipeline including workflow orchestration,
integration testing, error handling, and end-to-end processing.

Author: Bathymetric CAE Team
License: MIT
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from conftest import (
    assert_array_properties,
    assert_file_exists_and_valid,
    TestDataGenerator,
    TENSORFLOW_AVAILABLE
)

from bathymetric_cae.core.pipeline import (
    BathymetricCAEPipeline,
    run_pipeline_from_config,
    validate_pipeline_requirements
)


class TestBathymetricCAEPipeline:
    """Test the main processing pipeline."""
    
    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initialization."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        assert pipeline.config == sample_config
        assert hasattr(pipeline, 'processor')
        assert hasattr(pipeline, 'model_builder')
        assert hasattr(pipeline, 'visualizer')
        assert hasattr(pipeline, 'logger')
    
    def test_pipeline_setup_environment(self, sample_config, temp_dir):
        """Test pipeline environment setup."""
        sample_config.output_folder = str(temp_dir / "output")
        
        pipeline = BathymetricCAEPipeline(sample_config)
        pipeline._setup_environment()
        
        # Check that directories are created
        assert Path(sample_config.output_folder).exists()
        assert Path("logs").exists()
        assert Path("plots").exists()
    
    def test_validate_paths_valid(self, sample_config, temp_dir):
        """Test path validation with valid paths."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Should not raise exception
        pipeline._validate_paths(str(input_dir), str(output_dir))
        
        # Output directory should be created
        assert output_dir.exists()
    
    def test_validate_paths_invalid_input(self, sample_config, temp_dir):
        """Test path validation with invalid input path."""
        input_dir = temp_dir / "nonexistent"
        output_dir = temp_dir / "output"
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with pytest.raises(FileNotFoundError):
            pipeline._validate_paths(str(input_dir), str(output_dir))
    
    def test_validate_paths_permission_error(self, sample_config, temp_dir):
        """Test path validation with permission issues."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "readonly_output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock permission error
        with patch('pathlib.Path.write_text', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                pipeline._validate_paths(str(input_dir), str(output_dir))
    
    def test_get_valid_files(self, sample_config, create_test_files):
        """Test getting valid files from directory."""
        test_files = create_test_files(['.tif', '.bag', '.asc'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        valid_files = pipeline._get_valid_files(str(test_files[0].parent))
        
        assert len(valid_files) >= 3
        assert all(f.suffix.lower() in sample_config.supported_formats for f in valid_files)
    
    def test_analyze_sample_file(self, sample_config, create_test_files):
        """Test sample file analysis."""
        test_files = create_test_files(['.tif'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with patch.object(pipeline.processor, 'preprocess_bathymetric_grid') as mock_preprocess:
            mock_preprocess.return_value = (
                TestDataGenerator.create_bathymetric_grid(64),
                (64, 64),
                {'test': 'metadata'}
            )
            
            model_params = pipeline._analyze_sample_file(test_files[0])
            
            assert 'grid_size' in model_params
            assert 'channels' in model_params
            assert 'original_shape' in model_params
            assert 'has_uncertainty' in model_params
    
    def test_analyze_sample_file_failure(self, sample_config, create_test_files):
        """Test sample file analysis with processing failure."""
        test_files = create_test_files(['.tif'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with patch.object(pipeline.processor, 'preprocess_bathymetric_grid', 
                         side_effect=ValueError("Processing failed")):
            with pytest.raises(ValueError):
                pipeline._analyze_sample_file(test_files[0])
    
    @pytest.mark.tensorflow
    def test_get_or_train_model_existing(self, sample_config, temp_dir, mock_tensorflow_model):
        """Test loading existing model."""
        model_path = temp_dir / "existing_model.h5"
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock successful model loading
        with patch.object(pipeline.model_builder, 'load_model', return_value=mock_tensorflow_model):
            # Create dummy model file
            model_path.write_text("dummy model")
            
            model_params = {'channels': 1}
            result_model = pipeline._get_or_train_model(
                str(model_path), [], model_params, force_retrain=False
            )
            
            assert result_model == mock_tensorflow_model
    
    @pytest.mark.tensorflow
    def test_get_or_train_model_force_retrain(self, sample_config, temp_dir, mock_tensorflow_model):
        """Test forcing model retraining."""
        model_path = temp_dir / "existing_model.h5"
        model_path.write_text("dummy model")
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with patch.object(pipeline, '_train_model', return_value=mock_tensorflow_model) as mock_train:
            model_params = {'channels': 1}
            result_model = pipeline._get_or_train_model(
                str(model_path), [], model_params, force_retrain=True
            )
            
            mock_train.assert_called_once()
            assert result_model == mock_tensorflow_model
    
    @pytest.mark.tensorflow
    def test_train_model(self, sample_config, temp_dir, mock_tensorflow_model):
        """Test model training process."""
        model_path = temp_dir / "new_model.h5"
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock dependencies
        with patch.object(pipeline.model_builder, 'create_model', return_value=mock_tensorflow_model), \
             patch.object(pipeline, '_prepare_training_data', return_value=(None, None)), \
             patch.object(pipeline.model_builder, 'create_callbacks', return_value=[]), \
             patch.object(pipeline.visualizer, 'plot_training_history'):
            
            # Mock model.fit to return history
            mock_history = Mock()
            mock_history.history = {'loss': [1.0, 0.5], 'val_loss': [1.1, 0.6]}
            mock_tensorflow_model.fit.return_value = mock_history
            
            result_model = pipeline._train_model(str(model_path), [], {'channels': 1})
            
            assert result_model == mock_tensorflow_model
            mock_tensorflow_model.fit.assert_called_once()
    
    def test_prepare_training_data(self, sample_config, create_test_files):
        """Test training data preparation."""
        test_files = create_test_files(['.tif', '.asc'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock processor output
        mock_data = TestDataGenerator.create_bathymetric_grid(64)
        mock_input = mock_data[..., None]  # Add channel dimension
        
        with patch.object(pipeline.processor, 'preprocess_bathymetric_grid') as mock_preprocess:
            mock_preprocess.return_value = (mock_input, (64, 64), {})
            
            X_train, y_train = pipeline._prepare_training_data(test_files)
            
            assert X_train is not None
            assert y_train is not None
            assert X_train.shape[0] == len(test_files)  # Batch size matches file count
            assert y_train.shape[0] == len(test_files)
    
    def test_prepare_training_data_failures(self, sample_config, create_test_files, caplog):
        """Test training data preparation with some failures."""
        test_files = create_test_files(['.tif', '.asc', '.bag'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock mixed success/failure
        mock_data = TestDataGenerator.create_bathymetric_grid(64)
        mock_input = mock_data[..., None]
        
        with patch.object(pipeline.processor, 'preprocess_bathymetric_grid') as mock_preprocess:
            mock_preprocess.side_effect = [
                (mock_input, (64, 64), {}),  # Success
                ValueError("Processing failed"),  # Failure
                (mock_input, (64, 64), {}),  # Success
            ]
            
            X_train, y_train = pipeline._prepare_training_data(test_files)
            
            # Should return data from successful files only
            assert X_train.shape[0] == 2  # Two successful files
            assert "Skipping" in caplog.text
    
    def test_prepare_training_data_no_success(self, sample_config, create_test_files):
        """Test training data preparation with no successful files."""
        test_files = create_test_files(['.tif'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with patch.object(pipeline.processor, 'preprocess_bathymetric_grid',
                         side_effect=ValueError("All failed")):
            with pytest.raises(ValueError, match="No valid training data found"):
                pipeline._prepare_training_data(test_files)
    
    def test_process_files(self, sample_config, create_test_files, temp_dir):
        """Test batch file processing."""
        test_files = create_test_files(['.tif', '.asc'])
        output_dir = temp_dir / "output"
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock model and processing
        mock_model = Mock()
        
        with patch.object(pipeline, '_process_single_file') as mock_process:
            mock_process.return_value = {
                'filename': 'test.tif',
                'ssim': 0.85,
                'processing_successful': True
            }
            
            results = pipeline._process_files(mock_model, test_files, str(output_dir))
            
            assert 'successful_files' in results
            assert 'failed_files' in results
            assert 'success_rate' in results
            assert len(results['successful_files']) == len(test_files)
    
    def test_process_files_with_failures(self, sample_config, create_test_files, temp_dir, caplog):
        """Test batch file processing with some failures."""
        test_files = create_test_files(['.tif', '.asc', '.bag'])
        output_dir = temp_dir / "output"
        
        pipeline = BathymetricCAEPipeline(sample_config)
        mock_model = Mock()
        
        with patch.object(pipeline, '_process_single_file') as mock_process:
            # Mix of success and failure
            mock_process.side_effect = [
                {'filename': 'test1.tif', 'ssim': 0.85, 'processing_successful': True},
                Exception("Processing failed"),
                {'filename': 'test3.bag', 'ssim': 0.90, 'processing_successful': True}
            ]
            
            results = pipeline._process_files(mock_model, test_files, str(output_dir))
            
            assert len(results['successful_files']) == 2
            assert len(results['failed_files']) == 1
            assert results['success_rate'] == (2/3) * 100
    
    def test_process_single_file(self, sample_config, create_test_files, temp_dir):
        """Test single file processing."""
        test_files = create_test_files(['.tif'])
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock dependencies
        mock_model = Mock()
        mock_data = TestDataGenerator.create_bathymetric_grid(64)
        mock_input = mock_data[..., None]
        mock_prediction = mock_data[None, ..., None]  # Add batch dimension
        
        mock_model.predict.return_value = mock_prediction
        
        with patch.object(pipeline.processor, 'preprocess_bathymetric_grid') as mock_preprocess, \
             patch.object(pipeline, '_save_processed_data'), \
             patch.object(pipeline.visualizer, 'plot_comparison'):
            
            mock_preprocess.return_value = (mock_input, (64, 64), {'test': 'metadata'})
            
            stats = pipeline._process_single_file(mock_model, test_files[0], output_dir)
            
            assert 'filename' in stats
            assert 'ssim' in stats
            assert 'processing_time' in stats
            assert stats['filename'] == test_files[0].name
    
    def test_calculate_ssim_safe(self, sample_config):
        """Test SSIM calculation with error handling."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Valid data
        original = TestDataGenerator.create_bathymetric_grid(32)
        cleaned = original + 0.1 * TestDataGenerator.create_bathymetric_grid(32)
        
        ssim_score = pipeline._calculate_ssim_safe(original, cleaned)
        
        assert 0 <= ssim_score <= 1
        assert isinstance(ssim_score, float)
    
    def test_calculate_ssim_safe_shape_mismatch(self, sample_config, caplog):
        """Test SSIM calculation with shape mismatch."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        original = TestDataGenerator.create_bathymetric_grid(32)
        cleaned = TestDataGenerator.create_bathymetric_grid(64)  # Different size
        
        ssim_score = pipeline._calculate_ssim_safe(original, cleaned)
        
        assert ssim_score == 0.0
        assert "Shape mismatch" in caplog.text
    
    def test_calculate_ssim_safe_error_handling(self, sample_config, caplog):
        """Test SSIM calculation error handling."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Invalid data that might cause SSIM to fail
        original = TestDataGenerator.create_bathymetric_grid(32)
        cleaned = None
        
        ssim_score = pipeline._calculate_ssim_safe(original, cleaned)
        
        assert ssim_score == 0.0
        assert "Error calculating SSIM" in caplog.text
    
    def test_calculate_uncertainty_metrics(self, sample_config):
        """Test uncertainty metrics calculation."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        uncertainty_data = TestDataGenerator.create_uncertainty_grid(32)
        cleaned_data = TestDataGenerator.create_bathymetric_grid(32)
        
        metrics = pipeline._calculate_uncertainty_metrics(uncertainty_data, cleaned_data)
        
        assert 'mean_uncertainty' in metrics
        assert 'std_uncertainty' in metrics
        assert 'max_uncertainty' in metrics
        assert 'uncertainty_reduction' in metrics
        assert 'uncertainty_correlation' in metrics
    
    def test_calculate_uncertainty_metrics_none(self, sample_config):
        """Test uncertainty metrics with no uncertainty data."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        cleaned_data = TestDataGenerator.create_bathymetric_grid(32)
        
        metrics = pipeline._calculate_uncertainty_metrics(None, cleaned_data)
        
        assert metrics == {}
    
    def test_calculate_uncertainty_metrics_error(self, sample_config, caplog):
        """Test uncertainty metrics calculation error handling."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Create problematic data
        uncertainty_data = TestDataGenerator.create_uncertainty_grid(32)
        cleaned_data = TestDataGenerator.create_bathymetric_grid(64)  # Different size
        
        metrics = pipeline._calculate_uncertainty_metrics(uncertainty_data, cleaned_data)
        
        assert metrics == {}
        assert "Error calculating uncertainty metrics" in caplog.text
    
    def test_save_processed_data(self, sample_config, temp_dir):
        """Test saving processed data."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        data = TestDataGenerator.create_bathymetric_grid(64)
        output_path = temp_dir / "output.tif"
        original_shape = (64, 64)
        geo_metadata = {
            'geotransform': (0, 1, 0, 0, 0, -1),
            'projection': 'EPSG:4326',
            'metadata': {'source': 'test'}
        }
        
        # Mock GDAL operations
        with patch('bathymetric_cae.core.pipeline.gdal') as mock_gdal:
            mock_driver = Mock()
            mock_dataset = Mock()
            mock_band = Mock()
            
            mock_gdal.GetDriverByName.return_value = mock_driver
            mock_driver.Create.return_value = mock_dataset
            mock_dataset.GetRasterBand.return_value = mock_band
            
            # Should not raise exception
            pipeline._save_processed_data(data, output_path, original_shape, geo_metadata)
            
            # Verify GDAL calls
            mock_gdal.GetDriverByName.assert_called_once()
            mock_driver.Create.assert_called_once()
    
    def test_save_processed_data_error(self, sample_config, temp_dir, caplog):
        """Test saving processed data with errors."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        data = TestDataGenerator.create_bathymetric_grid(64)
        output_path = temp_dir / "output.tif"
        original_shape = (64, 64)
        geo_metadata = {}
        
        # Mock GDAL to return None (failure)
        with patch('bathymetric_cae.core.pipeline.gdal') as mock_gdal:
            mock_gdal.GetDriverByName.return_value = None
            
            with pytest.raises(RuntimeError):
                pipeline._save_processed_data(data, output_path, original_shape, geo_metadata)
    
    def test_generate_final_report(self, sample_config, temp_dir):
        """Test final report generation."""
        pipeline = BathymetricCAEPipeline(sample_config)
        
        processing_results = {
            'successful_files': ['file1.tif', 'file2.bag'],
            'failed_files': [],
            'processing_stats': [
                {'filename': 'file1.tif', 'ssim': 0.85},
                {'filename': 'file2.bag', 'ssim': 0.90}
            ],
            'success_rate': 100.0
        }
        
        model_params = {'channels': 2, 'grid_size': 256}
        file_list = [Path('file1.tif'), Path('file2.bag')]
        
        with patch.object(pipeline.visualizer, 'plot_processing_summary'):
            report = pipeline._generate_final_report(processing_results, model_params, file_list)
            
            assert 'pipeline_info' in report
            assert 'model_parameters' in report
            assert 'summary_statistics' in report
            assert 'processing_results' in report
            assert 'configuration' in report
            
            # Check report file creation
            assert Path("processing_report.json").exists()


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_run(self, sample_config, create_test_files, temp_dir):
        """Test complete pipeline execution."""
        # Setup
        test_files = create_test_files(['.tif'])
        sample_config.input_folder = str(test_files[0].parent)
        sample_config.output_folder = str(temp_dir / "output")
        sample_config.model_path = str(temp_dir / "test_model.h5")
        sample_config.epochs = 1  # Minimal training for test
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock heavy operations
        mock_data = TestDataGenerator.create_bathymetric_grid(64)
        mock_input = mock_data[..., None]
        
        with patch.object(pipeline.processor, 'preprocess_bathymetric_grid') as mock_preprocess, \
             patch.object(pipeline, '_train_model') as mock_train, \
             patch.object(pipeline, '_process_single_file') as mock_process:
            
            mock_preprocess.return_value = (mock_input, (64, 64), {})
            mock_train.return_value = Mock()
            mock_process.return_value = {'filename': 'test.tif', 'ssim': 0.85}
            
            results = pipeline.run(
                input_folder=sample_config.input_folder,
                output_folder=sample_config.output_folder,
                model_path=sample_config.model_path
            )
            
            assert 'pipeline_info' in results
            assert results['pipeline_info']['total_files'] > 0
    
    @pytest.mark.integration
    def test_pipeline_error_recovery(self, sample_config, create_test_files, temp_dir):
        """Test pipeline error recovery."""
        test_files = create_test_files(['.tif'])
        sample_config.input_folder = str(test_files[0].parent)
        sample_config.output_folder = str(temp_dir / "output")
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock setup to fail
        with patch.object(pipeline, '_setup_environment', side_effect=Exception("Setup failed")):
            with pytest.raises(RuntimeError):
                pipeline.run(
                    input_folder=sample_config.input_folder,
                    output_folder=sample_config.output_folder,
                    model_path="test.h5"
                )
    
    def test_process_single_file_interactive(self, sample_config, create_test_files, temp_dir):
        """Test interactive single file processing."""
        test_files = create_test_files(['.tif'])
        model_path = temp_dir / "test_model.h5"
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock dependencies
        mock_model = Mock()
        mock_data = TestDataGenerator.create_bathymetric_grid(64)
        mock_input = mock_data[..., None]
        mock_prediction = mock_data[None, ..., None]
        
        mock_model.predict.return_value = mock_prediction
        
        with patch.object(pipeline.model_builder, 'load_model', return_value=mock_model), \
             patch.object(pipeline.processor, 'preprocess_bathymetric_grid') as mock_preprocess, \
             patch.object(pipeline.visualizer, 'plot_comparison'), \
             patch.object(pipeline.visualizer, 'plot_difference_map'):
            
            mock_preprocess.return_value = (mock_input, (64, 64), {})
            
            # Create dummy model file
            model_path.write_text("dummy")
            
            results = pipeline.process_single_file_interactive(
                file_path=str(test_files[0]),
                model_path=str(model_path),
                show_plots=False
            )
            
            assert results['processing_successful'] is True
            assert 'filename' in results
            assert 'ssim' in results


class TestPipelineUtilityFunctions:
    """Test pipeline utility functions."""
    
    def test_run_pipeline_from_config(self, sample_config, temp_dir):
        """Test running pipeline from configuration file."""
        config_file = temp_dir / "test_config.json"
        sample_config.save(str(config_file))
        
        # Mock the pipeline run
        with patch('bathymetric_cae.core.pipeline.BathymetricCAEPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.run.return_value = {'test': 'result'}
            mock_pipeline_class.return_value = mock_pipeline
            
            result = run_pipeline_from_config(str(config_file), epochs=50)
            
            assert result == {'test': 'result'}
            mock_pipeline_class.assert_called_once()
    
    def test_run_pipeline_from_config_override(self, sample_config, temp_dir):
        """Test running pipeline with parameter overrides."""
        config_file = temp_dir / "test_config.json"
        sample_config.save(str(config_file))
        
        with patch('bathymetric_cae.core.pipeline.BathymetricCAEPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            run_pipeline_from_config(str(config_file), epochs=150, batch_size=32)
            
            # Check that config was created with overrides
            mock_pipeline_class.assert_called_once()
            called_config = mock_pipeline_class.call_args[0][0]
            assert called_config.epochs == 150
            assert called_config.batch_size == 32
    
    def test_validate_pipeline_requirements(self):
        """Test pipeline requirements validation."""
        requirements = validate_pipeline_requirements()
        
        assert isinstance(requirements, dict)
        assert 'all_requirements_met' in requirements
        
        # Should check for key dependencies
        expected_deps = ['tensorflow', 'gdal', 'numpy', 'matplotlib', 'seaborn', 'scikit-image']
        for dep in expected_deps:
            assert dep in requirements


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_pipeline_memory_management(self, sample_config, create_test_files):
        """Test pipeline memory management."""
        test_files = create_test_files(['.tif'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock to avoid actual processing
        with patch.object(pipeline, '_get_valid_files', return_value=test_files), \
             patch.object(pipeline, '_analyze_sample_file', return_value={'channels': 1}), \
             patch.object(pipeline, '_get_or_train_model', return_value=Mock()), \
             patch.object(pipeline, '_process_files', return_value={'successful_files': [], 'failed_files': [], 'processing_stats': []}), \
             patch.object(pipeline, '_generate_final_report', return_value={}):
            
            # Should complete without memory issues
            pipeline.run("input", "output", "model.h5")
    
    def test_pipeline_progress_monitoring(self, sample_config, create_test_files, caplog):
        """Test pipeline progress monitoring and logging."""
        test_files = create_test_files(['.tif', '.asc'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with patch.object(pipeline, '_process_single_file') as mock_process:
            mock_process.return_value = {'filename': 'test.tif', 'ssim': 0.85}
            
            pipeline._process_files(Mock(), test_files, "output")
            
            # Should log progress
            assert "Processing file" in caplog.text
    
    def test_pipeline_error_logging(self, sample_config, create_test_files, caplog):
        """Test comprehensive error logging."""
        test_files = create_test_files(['.tif'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with patch.object(pipeline, '_process_single_file', side_effect=Exception("Test error")):
            pipeline._process_files(Mock(), test_files, "output")
            
            # Should log errors
            assert "Failed to process" in caplog.text
            assert "Test error" in caplog.text


class TestPipelineEdgeCases:
    """Test pipeline edge cases and boundary conditions."""
    
    def test_pipeline_empty_input_folder(self, sample_config, temp_dir):
        """Test pipeline with empty input folder."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with pytest.raises(ValueError, match="No valid files found"):
            pipeline.run(str(empty_dir), "output", "model.h5")
    
    def test_pipeline_all_files_fail(self, sample_config, create_test_files, caplog):
        """Test pipeline when all files fail processing."""
        test_files = create_test_files(['.tif', '.asc'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with patch.object(pipeline, '_get_valid_files', return_value=test_files), \
             patch.object(pipeline, '_analyze_sample_file', return_value={'channels': 1}), \
             patch.object(pipeline, '_get_or_train_model', return_value=Mock()), \
             patch.object(pipeline, '_process_single_file', side_effect=Exception("All fail")):
            
            results = pipeline._process_files(Mock(), test_files, "output")
            
            assert results['success_rate'] == 0.0
            assert len(results['failed_files']) == len(test_files)
    
    def test_pipeline_model_loading_failure(self, sample_config, create_test_files, temp_dir):
        """Test pipeline when model loading fails."""
        test_files = create_test_files(['.tif'])
        nonexistent_model = temp_dir / "nonexistent.h5"
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        with patch.object(pipeline, '_get_valid_files', return_value=test_files), \
             patch.object(pipeline, '_analyze_sample_file', return_value={'channels': 1}), \
             patch.object(pipeline.model_builder, 'load_model', side_effect=Exception("Load failed")), \
             patch.object(pipeline, '_train_model', return_value=Mock()) as mock_train:
            
            pipeline._get_or_train_model(str(nonexistent_model), test_files, {'channels': 1})
            
            # Should fall back to training
            mock_train.assert_called_once()
    
    def test_pipeline_disk_space_handling(self, sample_config, create_test_files, temp_dir):
        """Test pipeline behavior when disk space is low."""
        test_files = create_test_files(['.tif'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Mock disk space error during save
        with patch.object(pipeline, '_save_processed_data', side_effect=OSError("No space left")):
            with pytest.raises(OSError):
                pipeline._process_single_file(Mock(), test_files[0], temp_dir)
    
    def test_pipeline_interrupted_processing(self, sample_config, create_test_files):
        """Test pipeline behavior when processing is interrupted."""
        test_files = create_test_files(['.tif'])
        
        pipeline = BathymetricCAEPipeline(sample_config)
        
        # Simulate KeyboardInterrupt
        with patch.object(pipeline, '_process_single_file', side_effect=KeyboardInterrupt("User interrupt")):
            with pytest.raises(KeyboardInterrupt):
                pipeline._process_files(Mock(), test_files, "output")


class TestPipelineConfigurationValidation:
    """Test pipeline configuration validation and error handling."""
    
    def test_pipeline_invalid_config_paths(self, sample_config):
        """Test pipeline with invalid configuration paths."""
        # Test with non-existent input folder
        invalid_config = sample_config.update(input_folder="/nonexistent/path")
        
        pipeline = BathymetricCAEPipeline(invalid_config)
        
        with pytest.raises(FileNotFoundError):
            pipeline.run(
                input_folder=invalid_config.input_folder,
                output_folder="output",
                model_path="model.h5"
            )
    
    def test_pipeline_config_memory_optimization(self, sample_config):
        """Test pipeline memory optimization based on config."""
        # Test with memory-constrained config
        memory_config = sample_config.update(
            batch_size=2,
            grid_size=64,
            base_filters=8
        )
        
        pipeline = BathymetricCAEPipeline(memory_config)
        
        assert pipeline.config.batch_size == 2
        assert pipeline.config.grid_size == 64
        assert pipeline.config.base_filters == 8
    
    def test_pipeline_config_gpu_settings(self, sample_config):
        """Test pipeline GPU configuration handling."""
        gpu_config = sample_config.update(
            gpu_memory_growth=True,
            use_mixed_precision=True
        )
        
        pipeline = BathymetricCAEPipeline(gpu_config)
        
        assert pipeline.config.gpu_memory_growth is True
        assert pipeline.config.use_mixed_precision is True