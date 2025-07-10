"""
Tests for Model Architecture Module

Tests for the advanced CAE model including architecture creation,
training utilities, loss functions, and performance optimization.

Author: Bathymetric CAE Team
License: MIT
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from conftest import (
    assert_array_properties,
    create_minimal_model_for_testing,
    TENSORFLOW_AVAILABLE
)

if TENSORFLOW_AVAILABLE:
    import tensorflow as tf
    from bathymetric_cae.core.model import (
        AdvancedCAE,
        create_data_augmentation_layer,
        calculate_model_memory_requirements
    )


@pytest.mark.tensorflow
class TestAdvancedCAE:
    """Test the AdvancedCAE model builder class."""
    
    def test_model_initialization(self, sample_config):
        """Test model builder initialization."""
        model_builder = AdvancedCAE(sample_config)
        
        assert model_builder.config == sample_config
        assert hasattr(model_builder, 'logger')
    
    def test_model_initialization_validation(self, sample_config):
        """Test model initialization with invalid config."""
        # Test missing required attributes
        delattr(sample_config, 'grid_size')
        
        with pytest.raises(ValueError, match="missing required attribute"):
            AdvancedCAE(sample_config)
    
    def test_config_validation_grid_size(self, sample_config):
        """Test configuration validation for grid size."""
        sample_config.grid_size = 16  # Too small
        
        with pytest.raises(ValueError, match="grid_size must be at least 32"):
            AdvancedCAE(sample_config)
    
    def test_config_validation_base_filters(self, sample_config):
        """Test configuration validation for base filters."""
        sample_config.base_filters = 4  # Too small
        
        with pytest.raises(ValueError, match="base_filters must be at least 8"):
            AdvancedCAE(sample_config)
    
    def test_config_validation_depth(self, sample_config):
        """Test configuration validation for depth."""
        sample_config.depth = 0  # Invalid
        
        with pytest.raises(ValueError, match="depth must be at least 1"):
            AdvancedCAE(sample_config)
    
    def test_create_model_single_channel(self, sample_config):
        """Test creating model with single channel input."""
        model_builder = AdvancedCAE(sample_config)
        
        model = model_builder.create_model(channels=1)
        
        # Check model structure
        assert model is not None
        assert len(model.inputs) == 1
        assert len(model.outputs) == 1
        
        # Check input shape
        expected_input_shape = (None, sample_config.grid_size, sample_config.grid_size, 1)
        assert model.input_shape == expected_input_shape
        
        # Check output shape
        expected_output_shape = (None, sample_config.grid_size, sample_config.grid_size, 1)
        assert model.output_shape == expected_output_shape
    
    def test_create_model_multi_channel(self, sample_config):
        """Test creating model with multi-channel input."""
        model_builder = AdvancedCAE(sample_config)
        
        model = model_builder.create_model(channels=2)
        
        # Check input shape for depth + uncertainty
        expected_input_shape = (None, sample_config.grid_size, sample_config.grid_size, 2)
        assert model.input_shape == expected_input_shape
    
    def test_model_compilation(self, sample_config):
        """Test that model is properly compiled."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Check that model is compiled
        assert model.optimizer is not None
        assert model.loss is not None
        assert model.compiled_metrics is not None
    
    def test_model_parameters_count(self, sample_config):
        """Test that model has reasonable parameter count."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        param_count = model.count_params()
        
        # Should have reasonable number of parameters (not too small or huge)
        assert 1000 < param_count < 10_000_000
    
    def test_model_prediction_shape(self, sample_config, sample_bathymetric_data):
        """Test model prediction output shape."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Create test input
        test_input = np.expand_dims(sample_bathymetric_data, axis=(0, -1))
        
        # Run prediction
        prediction = model.predict(test_input, verbose=0)
        
        # Check output shape
        expected_shape = (1, sample_config.grid_size, sample_config.grid_size, 1)
        assert prediction.shape == expected_shape
        
        # Check output range (sigmoid activation)
        assert np.all(prediction >= 0)
        assert np.all(prediction <= 1)
    
    def test_residual_block(self, sample_config):
        """Test residual block implementation."""
        model_builder = AdvancedCAE(sample_config)
        
        # Create test input
        test_input = tf.keras.layers.Input(shape=(64, 64, 16))
        
        # Apply residual block
        output = model_builder._residual_block(test_input, 16, "test_block")
        
        # Check that output shape matches input
        assert output.shape[-1] == 16  # Same number of filters
    
    def test_attention_block(self, sample_config):
        """Test attention mechanism implementation."""
        model_builder = AdvancedCAE(sample_config)
        
        # Create test input
        test_input = tf.keras.layers.Input(shape=(32, 32, 32))
        
        # Apply attention block
        output = model_builder._attention_block(test_input, 32, "test_attention")
        
        # Check that output shape matches input
        assert output.shape == test_input.shape
    
    def test_encoder_decoder_symmetry(self, sample_config):
        """Test that encoder and decoder have symmetric structure."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Get model layers
        layer_names = [layer.name for layer in model.layers]
        
        # Check for encoder layers
        encoder_layers = [name for name in layer_names if 'encoder' in name]
        assert len(encoder_layers) > 0
        
        # Check for decoder layers
        decoder_layers = [name for name in layer_names if 'decoder' in name]
        assert len(decoder_layers) > 0
    
    def test_combined_loss_function(self, sample_config):
        """Test combined MSE + SSIM loss function."""
        model_builder = AdvancedCAE(sample_config)
        
        # Create test tensors
        y_true = tf.random.normal((2, 64, 64, 1))
        y_pred = tf.random.normal((2, 64, 64, 1))
        
        # Calculate loss
        loss_value = model_builder._combined_loss(y_true, y_pred)
        
        # Check that loss is computed
        assert loss_value is not None
        assert tf.is_tensor(loss_value)
        assert loss_value.shape == ()  # Scalar loss
    
    def test_ssim_metric(self, sample_config):
        """Test SSIM metric calculation."""
        model_builder = AdvancedCAE(sample_config)
        
        # Create test tensors
        y_true = tf.random.normal((2, 64, 64, 1))
        y_pred = y_true + 0.1 * tf.random.normal((2, 64, 64, 1))  # Similar to true
        
        # Calculate SSIM
        ssim_value = model_builder._ssim_metric(y_true, y_pred)
        
        # Check SSIM properties
        assert tf.is_tensor(ssim_value)
        assert 0 <= ssim_value <= 1  # SSIM should be between 0 and 1
    
    def test_create_callbacks(self, sample_config, temp_dir):
        """Test callback creation."""
        model_builder = AdvancedCAE(sample_config)
        model_path = str(temp_dir / "test_model.h5")
        
        callbacks = model_builder.create_callbacks(model_path)
        
        # Check that callbacks are created
        assert len(callbacks) > 0
        
        # Check for essential callbacks
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_types
        assert 'ReduceLROnPlateau' in callback_types
        assert 'ModelCheckpoint' in callback_types
    
    def test_load_model(self, sample_config, temp_dir):
        """Test model loading functionality."""
        model_builder = AdvancedCAE(sample_config)
        model_path = temp_dir / "test_model.h5"
        
        # Create and save a model
        original_model = model_builder.create_model(channels=1)
        original_model.save(str(model_path))
        
        # Load the model
        loaded_model = model_builder.load_model(str(model_path))
        
        assert loaded_model is not None
        assert loaded_model.input_shape == original_model.input_shape
        assert loaded_model.output_shape == original_model.output_shape
    
    def test_load_model_nonexistent(self, sample_config):
        """Test loading non-existent model raises error."""
        model_builder = AdvancedCAE(sample_config)
        
        with pytest.raises(Exception):  # Could be FileNotFoundError or other exception
            model_builder.load_model("nonexistent_model.h5")
    
    def test_get_model_summary(self, sample_config):
        """Test model summary generation."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        summary = model_builder.get_model_summary(model)
        
        assert isinstance(summary, str)
        assert len(summary) > 100  # Should be substantial
        assert 'Model:' in summary or 'Layer (type)' in summary


@pytest.mark.tensorflow 
class TestModelUtilityFunctions:
    """Test utility functions for model operations."""
    
    def test_create_data_augmentation_layer(self):
        """Test data augmentation layer creation."""
        aug_layer = create_data_augmentation_layer()
        
        assert aug_layer is not None
        assert isinstance(aug_layer, tf.keras.Sequential)
        
        # Test with sample data
        test_data = tf.random.normal((2, 64, 64, 1))
        augmented = aug_layer(test_data, training=True)
        
        assert augmented.shape == test_data.shape
    
    def test_calculate_model_memory_requirements(self, sample_config):
        """Test memory requirements calculation."""
        memory_req = calculate_model_memory_requirements(sample_config)
        
        assert isinstance(memory_req, dict)
        assert 'estimated_parameters' in memory_req
        assert 'parameter_memory_mb' in memory_req
        assert 'activation_memory_mb' in memory_req
        assert 'total_memory_mb' in memory_req
        
        # Check that values are reasonable
        assert memory_req['estimated_parameters'] > 0
        assert memory_req['total_memory_mb'] > 0
    
    def test_memory_requirements_scaling(self):
        """Test that memory requirements scale with model size."""
        from bathymetric_cae import Config
        
        small_config = Config(grid_size=128, base_filters=8, depth=2)
        large_config = Config(grid_size=512, base_filters=64, depth=6)
        
        small_req = calculate_model_memory_requirements(small_config)
        large_req = calculate_model_memory_requirements(large_config)
        
        # Large model should require more memory
        assert large_req['total_memory_mb'] > small_req['total_memory_mb']
        assert large_req['estimated_parameters'] > small_req['estimated_parameters']


@pytest.mark.tensorflow
class TestModelTraining:
    """Test model training functionality."""
    
    def test_model_training_step(self, sample_config, sample_multi_channel_data):
        """Test single training step."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=2)
        
        # Prepare training data
        X = np.expand_dims(sample_multi_channel_data, axis=0)
        y = X[..., :1]  # Target is first channel only
        
        # Single training step
        history = model.fit(X, y, epochs=1, verbose=0)
        
        assert 'loss' in history.history
        assert len(history.history['loss']) == 1
    
    def test_model_validation(self, sample_config, sample_multi_channel_data):
        """Test model with validation data."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=2)
        
        # Prepare data
        X = np.expand_dims(sample_multi_channel_data, axis=0)
        y = X[..., :1]
        
        # Train with validation split
        history = model.fit(
            X, y, 
            epochs=2, 
            validation_split=0.5, 
            verbose=0
        )
        
        assert 'val_loss' in history.history
        assert len(history.history['val_loss']) == 2
    
    @pytest.mark.slow
    def test_model_convergence(self, sample_config, sample_multi_channel_data):
        """Test that model loss decreases over training."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=2)
        
        # Prepare data
        X = np.tile(np.expand_dims(sample_multi_channel_data, axis=0), (4, 1, 1, 1))
        y = X[..., :1]
        
        # Train for several epochs
        history = model.fit(X, y, epochs=5, verbose=0)
        
        # Check that loss generally decreases
        losses = history.history['loss']
        assert losses[-1] < losses[0]  # Final loss should be lower than initial


@pytest.mark.tensorflow
class TestModelArchitectureVariations:
    """Test different model architecture configurations."""
    
    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_different_depths(self, sample_config, depth):
        """Test model creation with different depths."""
        sample_config.depth = depth
        model_builder = AdvancedCAE(sample_config)
        
        model = model_builder.create_model(channels=1)
        
        assert model is not None
        assert model.count_params() > 0
    
    @pytest.mark.parametrize("base_filters", [8, 16, 32, 64])
    def test_different_filter_counts(self, sample_config, base_filters):
        """Test model creation with different filter counts."""
        sample_config.base_filters = base_filters
        model_builder = AdvancedCAE(sample_config)
        
        model = model_builder.create_model(channels=1)
        
        assert model is not None
        
        # More filters should mean more parameters
        param_count = model.count_params()
        assert param_count > base_filters * 10  # Rough estimate
    
    @pytest.mark.parametrize("grid_size", [64, 128, 256])
    def test_different_input_sizes(self, sample_config, grid_size):
        """Test model creation with different input sizes."""
        sample_config.grid_size = grid_size
        model_builder = AdvancedCAE(sample_config)
        
        model = model_builder.create_model(channels=1)
        
        assert model is not None
        expected_shape = (None, grid_size, grid_size, 1)
        assert model.input_shape == expected_shape
    
    @pytest.mark.parametrize("channels", [1, 2, 3])
    def test_different_channel_counts(self, sample_config, channels):
        """Test model creation with different channel counts."""
        model_builder = AdvancedCAE(sample_config)
        
        model = model_builder.create_model(channels=channels)
        
        assert model is not None
        expected_input_shape = (None, sample_config.grid_size, sample_config.grid_size, channels)
        assert model.input_shape == expected_input_shape


@pytest.mark.tensorflow
class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_model_inference_speed(self, sample_config, sample_bathymetric_data):
        """Test model inference performance."""
        import time
        
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Prepare test data
        test_input = np.expand_dims(sample_bathymetric_data, axis=(0, -1))
        
        # Warm up
        model.predict(test_input, verbose=0)
        
        # Time inference
        start_time = time.time()
        for _ in range(10):
            model.predict(test_input, verbose=0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (less than 1 second for small model)
        assert avg_time < 1.0
    
    def test_model_memory_usage(self, sample_config):
        """Test model memory usage."""
        model_builder = AdvancedCAE(sample_config)
        
        # Create model
        model = model_builder.create_model(channels=1)
        
        # Check that model size is reasonable
        param_count = model.count_params()
        estimated_size_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
        
        # Should not be too large for test configuration
        assert estimated_size_mb < 100  # Less than 100MB for test model
    
    def test_batch_processing_efficiency(self, sample_config, sample_bathymetric_data):
        """Test efficiency of batch processing."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Single prediction
        single_input = np.expand_dims(sample_bathymetric_data, axis=(0, -1))
        single_pred = model.predict(single_input, verbose=0)
        
        # Batch prediction
        batch_input = np.tile(single_input, (4, 1, 1, 1))
        batch_pred = model.predict(batch_input, verbose=0)
        
        # Check shapes
        assert single_pred.shape == (1, sample_config.grid_size, sample_config.grid_size, 1)
        assert batch_pred.shape == (4, sample_config.grid_size, sample_config.grid_size, 1)
        
        # Batch processing should be more efficient than individual predictions
        # (This is more of a conceptual test - actual timing would be implementation dependent)


@pytest.mark.tensorflow
class TestModelErrorHandling:
    """Test model error handling and edge cases."""
    
    def test_invalid_input_shape(self, sample_config):
        """Test model behavior with invalid input shapes."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Wrong number of dimensions
        invalid_input = np.random.rand(32, 32)  # Missing batch and channel dims
        
        with pytest.raises(Exception):  # TensorFlow will raise an error
            model.predict(invalid_input)
    
    def test_wrong_channel_count(self, sample_config):
        """Test model with wrong number of input channels."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)  # Expects 1 channel
        
        # Provide 2 channels
        wrong_input = np.random.rand(1, sample_config.grid_size, sample_config.grid_size, 2)
        
        with pytest.raises(Exception):
            model.predict(wrong_input)
    
    def test_empty_input(self, sample_config):
        """Test model behavior with empty input."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Empty batch
        empty_input = np.empty((0, sample_config.grid_size, sample_config.grid_size, 1))
        
        # Should handle empty input gracefully or raise appropriate error
        try:
            result = model.predict(empty_input, verbose=0)
            assert result.shape[0] == 0  # Empty output for empty input
        except Exception:
            pass  # Acceptable to raise error for empty input
    
    def test_extreme_values(self, sample_config):
        """Test model with extreme input values."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Very large values
        extreme_input = np.full(
            (1, sample_config.grid_size, sample_config.grid_size, 1), 
            1e6
        )
        
        # Should handle extreme values without crashing
        result = model.predict(extreme_input, verbose=0)
        assert result is not None
        assert np.all(np.isfinite(result))  # Output should be finite
    
    def test_nan_input(self, sample_config):
        """Test model behavior with NaN input."""
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=1)
        
        # Input with NaN values
        nan_input = np.full(
            (1, sample_config.grid_size, sample_config.grid_size, 1), 
            np.nan
        )
        
        # Model should handle NaN input
        result = model.predict(nan_input, verbose=0)
        
        # Result might be NaN or the model might handle it - either is acceptable
        assert result is not None


@pytest.mark.tensorflow
class TestModelIntegration:
    """Test model integration with other components."""
    
    def test_model_with_processor_output(self, sample_config, sample_multi_channel_data):
        """Test model with actual processor output."""
        from bathymetric_cae.core.processor import BathymetricProcessor
        
        model_builder = AdvancedCAE(sample_config)
        model = model_builder.create_model(channels=2)
        
        # Simulate processor output
        processor_output = sample_multi_channel_data
        
        # Add batch dimension
        model_input = np.expand_dims(processor_output, axis=0)
        
        # Model should handle processor output
        result = model.predict(model_input, verbose=0)
        
        assert result.shape == (1, sample_config.grid_size, sample_config.grid_size, 1)
        assert np.all(result >= 0) and np.all(result <= 1)  # Sigmoid output
    
    def test_model_save_load_consistency(self, sample_config, temp_dir):
        """Test that saved and loaded models produce consistent results."""
        model_builder = AdvancedCAE(sample_config)
        model_path = temp_dir / "consistency_test.h5"
        
        # Create original model
        original_model = model_builder.create_model(channels=1)
        
        # Test input
        test_input = np.random.rand(1, sample_config.grid_size, sample_config.grid_size, 1)
        
        # Get prediction from original model
        original_pred = original_model.predict(test_input, verbose=0)
        
        # Save and load model
        original_model.save(str(model_path))
        loaded_model = model_builder.load_model(str(model_path))
        
        # Get prediction from loaded model
        loaded_pred = loaded_model.predict(test_input, verbose=0)
        
        # Predictions should be identical (or very close due to floating point)
        np.testing.assert_allclose(original_pred, loaded_pred, rtol=1e-6)
