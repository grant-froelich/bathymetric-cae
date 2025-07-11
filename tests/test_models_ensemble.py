# tests/test_models_ensemble.py
"""
Test ensemble model functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from config.config import Config
from models.ensemble import BathymetricEnsemble


class TestBathymetricEnsemble:
    """Test ensemble model functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for ensemble testing."""
        return Config(
            grid_size=32,
            ensemble_size=2,
            base_filters=8,
            depth=2,
            enable_constitutional_constraints=True,
            ssim_weight=0.25,
            roughness_weight=0.25,
            feature_preservation_weight=0.25,
            consistency_weight=0.25
        )
    
    def test_ensemble_creation(self, sample_config):
        """Test ensemble creation."""
        ensemble = BathymetricEnsemble(sample_config)
        models = ensemble.create_ensemble(channels=1)
        
        assert len(models) == sample_config.ensemble_size
        assert len(ensemble.weights) == sample_config.ensemble_size
        assert sum(ensemble.weights) == pytest.approx(1.0)
    
    @patch('models.ensemble.BathymetricEnsemble._calculate_ensemble_metrics')
    def test_ensemble_prediction(self, mock_metrics, sample_config):
        """Test ensemble prediction."""
        ensemble = BathymetricEnsemble(sample_config)
        
        # Mock models
        mock_model1 = Mock()
        mock_model2 = Mock()
        mock_model1.predict.return_value = np.ones((1, 32, 32, 1)) * 0.4
        mock_model2.predict.return_value = np.ones((1, 32, 32, 1)) * 0.6
        
        ensemble.models = [mock_model1, mock_model2]
        ensemble.weights = [0.5, 0.5]
        
        # Mock metrics
        mock_metrics.return_value = {'composite_quality': 0.8}
        
        # Test prediction
        input_data = np.random.random((1, 32, 32, 1)).astype(np.float32)
        prediction, metrics = ensemble.predict_ensemble(input_data)
        
        assert prediction.shape == (1, 32, 32, 1)
        assert np.allclose(prediction, 0.5)  # Average of 0.4 and 0.6
        assert 'composite_quality' in metrics
    
    def test_weight_updates(self, sample_config):
        """Test ensemble weight updates."""
        ensemble = BathymetricEnsemble(sample_config)
        ensemble.create_ensemble(channels=1)
        
        # Update weights based on performance
        performance_scores = [0.8, 0.6]
        ensemble.update_weights_from_performance(performance_scores)
        
        # Better performing model should have higher weight
        assert ensemble.weights[0] > ensemble.weights[1]
        assert sum(ensemble.weights) == pytest.approx(1.0)