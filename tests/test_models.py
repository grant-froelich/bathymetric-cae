# tests/test_models.py
"""Tests for model architectures."""

import pytest
import numpy as np
import tensorflow as tf

from bathymetric_cae.models import (
    BathymetricEnsemble,
    create_model_by_type,
    get_model_variants
)
from bathymetric_cae.config import Config
from tests import create_temp_config


class TestModelArchitectures:
    """Test model architectures."""
    
    def test_model_creation_by_type(self):
        """Test creating models by type."""
        config = create_temp_config()
        input_shape = (config.grid_size, config.grid_size, 1)
        
        model_types = ['lightweight', 'standard', 'robust']
        
        for model_type in model_types:
            model = create_model_by_type(model_type, config, input_shape)
            
            assert isinstance(model, tf.keras.Model)
            assert model.input_shape == (None, *input_shape)
            assert model.output_shape == (None, config.grid_size, config.grid_size, 1)
    
    def test_model_variants_info(self):
        """Test getting model variants information."""
        config = create_temp_config()
        variants = get_model_variants(config)
        
        assert isinstance(variants, dict)
        assert 'lightweight' in variants
        assert 'standard' in variants
        assert all('description' in info for info in variants.values())
    
    def test_model_prediction(self):
        """Test model prediction capability."""
        config = create_temp_config()
        input_shape = (config.grid_size, config.grid_size, 1)
        
        model = create_model_by_type('lightweight', config, input_shape)
        
        # Create test input
        test_input = np.random.random((1, *input_shape)).astype(np.float32)
        
        # Test prediction
        prediction = model.predict(test_input, verbose=0)
        
        assert prediction.shape == (1, config.grid_size, config.grid_size, 1)
        assert np.all(prediction >= 0) and np.all(prediction <= 1)  # sigmoid output


class TestBathymetricEnsemble:
    """Test ensemble functionality."""
    
    def test_ensemble_creation(self):
        """Test ensemble creation."""
        config = create_temp_config()
        ensemble = BathymetricEnsemble(config)
        
        models = ensemble.create_ensemble(channels=1)
        
        assert len(models) == config.ensemble_size
        assert all(isinstance(m, tf.keras.Model) for m in models)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        config = create_temp_config()
        ensemble = BathymetricEnsemble(config)
        
        # Create ensemble
        models = ensemble.create_ensemble(channels=1)
        
        # Create test input
        test_input = np.random.random((1, config.grid_size, config.grid_size, 1)).astype(np.float32)
        
        # Test prediction
        prediction, metrics = ensemble.predict_ensemble(test_input)
        
        assert prediction.shape == test_input.shape
        assert isinstance(metrics, dict)
        assert 'ensemble_size' in metrics
