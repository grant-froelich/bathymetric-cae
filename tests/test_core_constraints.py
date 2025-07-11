# tests/test_core_constraints.py
"""
Test constitutional constraints functionality.
"""

import pytest
import numpy as np

from core.constraints import BathymetricConstraints


class TestBathymetricConstraints:
    """Test bathymetric constraints."""
    
    def test_validate_depth_continuity(self):
        """Test depth continuity validation."""
        # Smooth data should have no violations
        smooth_data = np.linspace(0, 10, 100).reshape(10, 10)
        violations = BathymetricConstraints.validate_depth_continuity(smooth_data)
        assert np.sum(violations) == 0
        
        # Data with steep gradients should have violations
        steep_data = np.zeros((10, 10))
        steep_data[5, 5] = 100  # Steep spike
        violations = BathymetricConstraints.validate_depth_continuity(steep_data, max_gradient=0.01)
        assert np.sum(violations) > 0
    
    def test_preserve_depth_features(self):
        """Test depth feature preservation."""
        # Create data with features
        original = np.zeros((10, 10))
        original[4:6, 4:6] = 10  # Feature
        
        # Data with preserved features
        preserved = original.copy()
        violations = BathymetricConstraints.preserve_depth_features(original, preserved)
        assert np.sum(violations) == 0
        
        # Data with lost features
        lost_features = np.zeros((10, 10))
        violations = BathymetricConstraints.preserve_depth_features(original, lost_features)
        assert np.sum(violations) > 0
    
    def test_enforce_monotonicity(self):
        """Test monotonicity enforcement."""
        # Increasing data
        increasing_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        violations = BathymetricConstraints.enforce_monotonicity(increasing_data, 'increasing')
        assert np.sum(violations) == 0
        
        # Non-monotonic data
        non_monotonic = np.array([[1, 3, 2], [4, 6, 5], [7, 9, 8]])
        violations = BathymetricConstraints.enforce_monotonicity(non_monotonic, 'increasing')
        assert np.sum(violations) > 0
