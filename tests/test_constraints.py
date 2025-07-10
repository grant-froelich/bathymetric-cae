# tests/test_constraints.py
"""Tests for constitutional constraints module."""

import pytest
import numpy as np

from bathymetric_cae.core.constraints import (
    BathymetricConstraints,
    DepthContinuityConstraint,
    FeaturePreservationConstraint,
    PhysicalPlausibilityConstraint
)
from bathymetric_cae.core.enums import SeafloorType


class TestBathymetricConstraints:
    """Test bathymetric constraints."""
    
    def test_depth_continuity_constraint(self):
        """Test depth continuity validation."""
        constraint = DepthContinuityConstraint(max_gradient=0.1)
        
        # Valid smooth data
        smooth_data = np.ones((100, 100)) * -50
        valid, violation = constraint.validate(smooth_data)
        assert valid is True
        assert violation is None
        
        # Invalid data with large gradients
        steep_data = np.zeros((100, 100))
        steep_data[50:, :] = -1000  # Large step
        valid, violation = constraint.validate(steep_data)
        assert valid is False
        assert violation is not None
        assert violation.constraint_type == "depth_continuity"
    
    def test_feature_preservation_constraint(self):
        """Test feature preservation validation."""
        constraint = FeaturePreservationConstraint()
        
        # Create original with features
        original = np.random.normal(-100, 20, (100, 100))
        
        # Good preservation (same data)
        valid, violation = constraint.validate(original, original=original)
        assert valid is True
        
        # Poor preservation (heavily smoothed)
        from scipy import ndimage
        heavily_smoothed = ndimage.gaussian_filter(original, sigma=10)
        valid, violation = constraint.validate(heavily_smoothed, original=original)
        assert valid is False
        assert violation.constraint_type == "feature_preservation"
    
    def test_physical_plausibility_constraint(self):
        """Test physical plausibility validation."""
        constraint = PhysicalPlausibilityConstraint()
        
        # Valid depths
        valid_data = np.random.uniform(-6000, 0, (100, 100))
        valid, violation = constraint.validate(valid_data)
        assert valid is True
        
        # Invalid depths (too deep)
        invalid_data = np.full((100, 100), -15000)
        valid, violation = constraint.validate(invalid_data)
        assert valid is False
        assert violation.constraint_type == "physical_plausibility"
    
    def test_constraint_correction(self):
        """Test constraint correction application."""
        constraints = BathymetricConstraints()
        
        # Create problematic data
        original = np.random.normal(-100, 20, (100, 100))
        processed = original.copy()
        processed[50:, :] = -15000  # Make part physically implausible
        
        # Apply corrections
        corrected = constraints.apply_corrections(original, processed)
        
        # Should be different from processed (corrected)
        assert not np.array_equal(processed, corrected)
        # Should not have extremely deep values
        assert np.all(corrected >= -11000)
