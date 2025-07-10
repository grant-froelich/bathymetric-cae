"""
Adaptive processing based on seafloor type classification.
"""

import numpy as np
from typing import Dict
from .enums import SeafloorType


class SeafloorClassifier:
    """Classify seafloor type for adaptive processing."""
    
    def __init__(self):
        self.depth_ranges = {
            SeafloorType.SHALLOW_COASTAL: (0, 200),
            SeafloorType.CONTINENTAL_SHELF: (200, 2000),
            SeafloorType.DEEP_OCEAN: (2000, 6000),
            SeafloorType.ABYSSAL_PLAIN: (6000, 11000),
            SeafloorType.SEAMOUNT: (0, 6000)  # Special case based on topology
        }
    
    def classify(self, depth_data: np.ndarray) -> SeafloorType:
        """Classify seafloor type based on depth and topological features."""
        try:
            # Basic depth-based classification
            mean_depth = np.mean(depth_data[np.isfinite(depth_data)])
            depth_range = np.ptp(depth_data[np.isfinite(depth_data)])
            
            # Check for seamount (high relief in deep water)
            if mean_depth > 1000 and depth_range > 500:
                return SeafloorType.SEAMOUNT
            
            # Classify based on depth ranges
            for seafloor_type, (min_depth, max_depth) in self.depth_ranges.items():
                if min_depth <= mean_depth <= max_depth:
                    return seafloor_type
            
            return SeafloorType.UNKNOWN
            
        except Exception:
            return SeafloorType.UNKNOWN


class AdaptiveProcessor:
    """Adaptive processing based on seafloor type and local characteristics."""
    
    def __init__(self):
        self.seafloor_classifier = SeafloorClassifier()
        self.processing_strategies = {
            SeafloorType.SHALLOW_COASTAL: self._shallow_coastal_strategy,
            SeafloorType.DEEP_OCEAN: self._deep_ocean_strategy,
            SeafloorType.CONTINENTAL_SHELF: self._continental_shelf_strategy,
            SeafloorType.SEAMOUNT: self._seamount_strategy,
            SeafloorType.ABYSSAL_PLAIN: self._abyssal_plain_strategy,
            SeafloorType.UNKNOWN: self._default_strategy
        }
    
    def get_processing_parameters(self, depth_data: np.ndarray) -> Dict:
        """Get adaptive processing parameters based on seafloor type."""
        seafloor_type = self.seafloor_classifier.classify(depth_data)
        return self.processing_strategies[seafloor_type](depth_data)
    
    def _shallow_coastal_strategy(self, depth_data: np.ndarray) -> Dict:
        """Processing strategy for shallow coastal areas."""
        return {
            'smoothing_factor': 0.3,
            'edge_preservation': 0.8,
            'noise_threshold': 0.1,
            'gradient_constraint': 0.05,
            'feature_preservation_weight': 0.9
        }
    
    def _deep_ocean_strategy(self, depth_data: np.ndarray) -> Dict:
        """Processing strategy for deep ocean areas."""
        return {
            'smoothing_factor': 0.7,
            'edge_preservation': 0.4,
            'noise_threshold': 0.2,
            'gradient_constraint': 0.1,
            'feature_preservation_weight': 0.6
        }
    
    def _continental_shelf_strategy(self, depth_data: np.ndarray) -> Dict:
        """Processing strategy for continental shelf areas."""
        return {
            'smoothing_factor': 0.5,
            'edge_preservation': 0.6,
            'noise_threshold': 0.15,
            'gradient_constraint': 0.08,
            'feature_preservation_weight': 0.75
        }
    
    def _seamount_strategy(self, depth_data: np.ndarray) -> Dict:
        """Processing strategy for seamount areas."""
        return {
            'smoothing_factor': 0.2,
            'edge_preservation': 0.9,
            'noise_threshold': 0.05,
            'gradient_constraint': 0.15,
            'feature_preservation_weight': 0.95
        }
    
    def _abyssal_plain_strategy(self, depth_data: np.ndarray) -> Dict:
        """Processing strategy for abyssal plain areas."""
        return {
            'smoothing_factor': 0.8,
            'edge_preservation': 0.3,
            'noise_threshold': 0.25,
            'gradient_constraint': 0.05,
            'feature_preservation_weight': 0.5
        }
    
    def _default_strategy(self, depth_data: np.ndarray) -> Dict:
        """Default processing strategy for unknown areas."""
        return {
            'smoothing_factor': 0.5,
            'edge_preservation': 0.6,
            'noise_threshold': 0.15,
            'gradient_constraint': 0.1,
            'feature_preservation_weight': 0.7
        }
