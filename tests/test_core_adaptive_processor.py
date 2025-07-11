# tests/test_core_adaptive_processor.py
"""
Test adaptive processing functionality.
"""

import pytest
import numpy as np

from core.adaptive_processor import SeafloorClassifier, AdaptiveProcessor
from core.enums import SeafloorType


class TestSeafloorClassifier:
    """Test seafloor classification."""
    
    def test_shallow_coastal_classification(self):
        """Test classification of shallow coastal data."""
        classifier = SeafloorClassifier()
        depth_data = np.full((10, 10), -50.0)  # 50m depth
        
        result = classifier.classify(depth_data)
        assert result == SeafloorType.SHALLOW_COASTAL
    
    def test_deep_ocean_classification(self):
        """Test classification of deep ocean data."""
        classifier = SeafloorClassifier()
        depth_data = np.full((10, 10), -3000.0)  # 3000m depth
        
        result = classifier.classify(depth_data)
        assert result == SeafloorType.DEEP_OCEAN
    
    def test_seamount_classification(self):
        """Test classification of seamount data."""
        classifier = SeafloorClassifier()
        # Create seamount-like data (high relief in deep water)
        depth_data = np.full((10, 10), -2000.0)
        depth_data[4:6, 4:6] = -1000.0  # Peak
        
        result = classifier.classify(depth_data)
        assert result == SeafloorType.SEAMOUNT
    
    def test_invalid_data_classification(self):
        """Test classification with invalid data."""
        classifier = SeafloorClassifier()
        depth_data = np.full((10, 10), np.nan)
        
        result = classifier.classify(depth_data)
        assert result == SeafloorType.UNKNOWN


class TestAdaptiveProcessor:
    """Test adaptive processor functionality."""
    
    def test_shallow_coastal_parameters(self):
        """Test parameters for shallow coastal processing."""
        processor = AdaptiveProcessor()
        depth_data = np.full((10, 10), -50.0)
        
        params = processor.get_processing_parameters(depth_data)
        
        assert params['smoothing_factor'] == 0.3
        assert params['edge_preservation'] == 0.8
        assert params['feature_preservation_weight'] == 0.9
    
    def test_deep_ocean_parameters(self):
        """Test parameters for deep ocean processing."""
        processor = AdaptiveProcessor()
        depth_data = np.full((10, 10), -3000.0)
        
        params = processor.get_processing_parameters(depth_data)
        
        assert params['smoothing_factor'] == 0.7
        assert params['edge_preservation'] == 0.4
        assert params['feature_preservation_weight'] == 0.6
    
    def test_parameters_format(self):
        """Test that parameters have expected keys."""
        processor = AdaptiveProcessor()
        depth_data = np.full((10, 10), -1000.0)
        
        params = processor.get_processing_parameters(depth_data)
        
        expected_keys = {
            'smoothing_factor', 'edge_preservation', 'noise_threshold',
            'gradient_constraint', 'feature_preservation_weight'
        }
        assert set(params.keys()) == expected_keys