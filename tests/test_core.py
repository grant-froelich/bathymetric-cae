# tests/test_core.py
"""Tests for core functionality."""

import pytest
import numpy as np

from bathymetric_cae.core import (
    SeafloorClassifier,
    AdaptiveProcessor,
    BathymetricQualityMetrics,
    SeafloorType,
    QualityLevel
)
from tests import create_test_data


class TestSeafloorClassifier:
    """Test seafloor classification."""
    
    def test_classifier_creation(self):
        """Test classifier initialization."""
        classifier = SeafloorClassifier()
        
        assert hasattr(classifier, 'feature_extractor')
        assert hasattr(classifier, 'classification_rules')
    
    def test_depth_classification(self):
        """Test classification based on depth."""
        classifier = SeafloorClassifier()
        
        # Test shallow data
        shallow_data = np.full((100, 100), -50)  # 50m depth
        classification = classifier.classify(shallow_data)
        assert classification == SeafloorType.SHALLOW_COASTAL
        
        # Test deep data  
        deep_data = np.full((100, 100), -3000)  # 3000m depth
        classification = classifier.classify(deep_data)
        assert classification == SeafloorType.DEEP_OCEAN
    
    def test_classification_with_confidence(self):
        """Test classification with confidence scoring."""
        classifier = SeafloorClassifier()
        
        data = create_test_data((100, 100))
        classification, confidence = classifier.classify_with_confidence(data)
        
        assert isinstance(classification, SeafloorType)
        assert 0 <= confidence <= 1


class TestAdaptiveProcessor:
    """Test adaptive processing."""
    
    def test_processor_creation(self):
        """Test processor initialization."""
        processor = AdaptiveProcessor()
        
        assert hasattr(processor, 'seafloor_classifier')
        assert hasattr(processor, 'processing_strategies')
    
    def test_parameter_generation(self):
        """Test adaptive parameter generation."""
        processor = AdaptiveProcessor()
        
        data = create_test_data((100, 100))
        params = processor.get_processing_parameters(data)
        
        assert isinstance(params, dict)
        assert 'seafloor_type' in params
        assert 'smoothing_factor' in params
        assert 'edge_preservation' in params
    
    def test_strategy_for_type(self):
        """Test getting parameters for specific type."""
        processor = AdaptiveProcessor()
        
        params = processor.get_processing_parameters_for_type(SeafloorType.SHALLOW_COASTAL)
        
        assert params['seafloor_type'] == 'shallow_coastal'
        assert 'smoothing_factor' in params


class TestQualityMetrics:
    """Test quality metrics calculation."""
    
    def test_metrics_creation(self):
        """Test metrics calculator initialization."""
        metrics = BathymetricQualityMetrics()
        
        assert hasattr(metrics, 'metrics')
        assert hasattr(metrics, 'additional_metrics')
    
    def test_metrics_calculation(self):
        """Test quality metrics calculation."""
        metrics = BathymetricQualityMetrics()
        
        original = create_test_data((100, 100))
        processed = original + np.random.normal(0, 0.1, original.shape)
        
        results = metrics.calculate_all_metrics(original, processed)
        
        assert isinstance(results, dict)
        assert 'ssim' in results
        assert 'composite_quality' in results
        assert all(0 <= v <= 1 for v in results.values() if isinstance(v, (int, float)))
    
    def test_quality_level_assessment(self):
        """Test quality level assessment."""
        metrics = BathymetricQualityMetrics()
        
        # Test different quality levels
        excellent_score = 0.95
        poor_score = 0.3
        
        assert metrics.assess_quality_level(excellent_score) == QualityLevel.EXCELLENT
        assert metrics.assess_quality_level(poor_score) == QualityLevel.POOR