# tests/test_models_architectures.py
"""
Test model architectures.
"""

import pytest
import numpy as np
import tensorflow as tf

from config.config import Config
from models.architectures import AdvancedCAE, UncertaintyCAE, LightweightCAE, create_model_variant


class TestModelArchitectures:
    """Test model architecture creation and functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for model testing."""
        return Config(
            grid_size=32,
            base_filters=8,
            depth=2,
            dropout_rate=0.1,
            learning_rate=0.001,
            epochs=1,
            batch_size=1
        )
    
    def test_advanced_cae_creation(self, sample_config):
        """Test AdvancedCAE model creation."""
        architecture = AdvancedCAE(sample_config)
        model = architecture.create_model((32, 32, 1))
        
        assert model is not None
        assert len(model.layers) > 5
        assert model.input_shape == (None, 32, 32, 1)
    
    def test_uncertainty_cae_creation(self, sample_config):
        """Test UncertaintyCAE model creation."""
        architecture = UncertaintyCAE(sample_config)
        model = architecture.create_model((32, 32, 1))
        
        assert model is not None
        assert len(model.outputs) == 2  # Depth + uncertainty outputs
    
    def test_lightweight_cae_creation(self, sample_config):
        """Test LightweightCAE model creation."""
        architecture = LightweightCAE(sample_config)
        model = architecture.create_model((32, 32, 1))
        
        assert model is not None
        assert model.count_params() < 100000  # Should be lightweight
    
    def test_model_variant_factory(self, sample_config):
        """Test model variant factory function."""
        model = create_model_variant(sample_config, 'advanced', (32, 32, 1))
        assert model is not None
        
        model = create_model_variant(sample_config, 'uncertainty', (32, 32, 1))
        assert model is not None
        
        model = create_model_variant(sample_config, 'lightweight', (32, 32, 1))
        assert model is not None
        
        with pytest.raises(ValueError):
            create_model_variant(sample_config, 'invalid_type', (32, 32, 1))
    
    def test_model_prediction(self, sample_config):
        """Test model prediction functionality."""
        architecture = AdvancedCAE(sample_config)
        model = architecture.create_model((32, 32, 1))
        
        # Create sample input
        input_data = np.random.random((1, 32, 32, 1)).astype(np.float32)
        
        # Test prediction
        prediction = model.predict(input_data, verbose=0)
        assert prediction.shape == (1, 32, 32, 1)
        assert not np.isnan(prediction).any()
