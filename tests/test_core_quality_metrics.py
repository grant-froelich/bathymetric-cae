# tests/test_core_quality_metrics.py
"""
Test quality metrics functionality.
"""

import pytest
import numpy as np

from core.quality_metrics import BathymetricQualityMetrics


class TestBathymetricQualityMetrics:
    """Test bathymetric quality metrics."""
    
    def test_calculate_roughness(self):
        """Test roughness calculation."""
        # Smooth data should have low roughness
        smooth_data = np.ones((10, 10))
        roughness = BathymetricQualityMetrics.calculate_roughness(smooth_data)
        assert roughness == 0.0
        
        # Rough data should have higher roughness
        rough_data = np.random.random((10, 10))
        roughness = BathymetricQualityMetrics.calculate_roughness(rough_data)
        assert roughness > 0.0
    
    def test_calculate_feature_preservation(self):
        """Test feature preservation calculation."""
        # Identical data should have perfect preservation
        data = np.random.random((10, 10))
        preservation = BathymetricQualityMetrics.calculate_feature_preservation(data, data)
        assert preservation == pytest.approx(1.0, abs=1e-6)
        
        # Very different data should have low preservation
        data1 = np.ones((10, 10))
        data2 = np.zeros((10, 10))
        preservation = BathymetricQualityMetrics.calculate_feature_preservation(data1, data2)
        assert preservation < 0.5
    
    def test_calculate_depth_consistency(self):
        """Test depth consistency calculation."""
        # Consistent data should have high consistency
        consistent_data = np.ones((10, 10))
        consistency = BathymetricQualityMetrics.calculate_depth_consistency(consistent_data)
        assert consistency > 0.9
        
        # Inconsistent data should have lower consistency
        inconsistent_data = np.random.random((10, 10)) * 100
        consistency = BathymetricQualityMetrics.calculate_depth_consistency(inconsistent_data)
        assert consistency < 0.9
    
    def test_calculate_hydrographic_standards_compliance(self):
        """Test hydrographic standards compliance."""
        depth_data = np.full((10, 10), -100.0)  # 100m depth
        compliance = BathymetricQualityMetrics.calculate_hydrographic_standards_compliance(depth_data)
        assert 0.0 <= compliance <= 1.0
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        invalid_data = np.full((10, 10), np.nan)
        
        # Should return 0.0 for invalid data
        roughness = BathymetricQualityMetrics.calculate_roughness(invalid_data)
        assert roughness == 0.0
        
        consistency = BathymetricQualityMetrics.calculate_depth_consistency(invalid_data)
        assert consistency == 0.0
