"""
Adaptive processing based on seafloor type classification.
FIXED: Enhanced versions that prevent tuple error and maintain enum integrity.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from .enums import SeafloorType


class EnhancedSeafloorClassifier:
    """Enhanced seafloor classifier with confidence scoring and error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.depth_ranges = {
            SeafloorType.SHALLOW_COASTAL: (0, 200),
            SeafloorType.CONTINENTAL_SHELF: (200, 2000),
            SeafloorType.DEEP_OCEAN: (2000, 6000),
            SeafloorType.ABYSSAL_PLAIN: (6000, 11000),
            SeafloorType.SEAMOUNT: (0, 6000)  # Special case based on topology
        }
    
    def classify(self, depth_data: np.ndarray) -> SeafloorType:
        """Classify seafloor type based on depth and topological features.
        
        FIXED: Always returns a SeafloorType enum, never a tuple.
        """
        try:
            # Validate input data
            if depth_data is None or depth_data.size == 0:
                self.logger.warning("Empty or None depth data provided")
                return SeafloorType.UNKNOWN
            
            # Handle all NaN data
            valid_data = depth_data[np.isfinite(depth_data)]
            if len(valid_data) == 0:
                self.logger.warning("All depth data values are NaN or infinite")
                return SeafloorType.UNKNOWN
            
            # Basic depth-based classification
            mean_depth = np.abs(np.mean(valid_data))  # Use absolute values for depth
            depth_range = np.ptp(valid_data)
            
            # Log classification details for debugging
            self.logger.debug(f"Mean depth: {mean_depth:.1f}m, Range: {depth_range:.1f}m")
            
            # Check for seamount (high relief in deep water)
            if mean_depth > 1000 and depth_range > 500:
                seafloor_type = SeafloorType.SEAMOUNT
                confidence = min(1.0, (depth_range - 500) / 1000)
            else:
                # Classify based on depth ranges
                seafloor_type = SeafloorType.UNKNOWN
                confidence = 0.0
                
                for st_type, (min_depth, max_depth) in self.depth_ranges.items():
                    if min_depth <= mean_depth <= max_depth:
                        seafloor_type = st_type
                        # Calculate confidence based on how central the depth is in the range
                        range_center = (min_depth + max_depth) / 2
                        range_width = max_depth - min_depth
                        distance_from_center = abs(mean_depth - range_center)
                        confidence = max(0.5, 1.0 - (distance_from_center / (range_width / 2)))
                        break
            
            # Ensure we always return a valid enum
            if not isinstance(seafloor_type, SeafloorType):
                self.logger.error(f"Classification returned invalid type: {type(seafloor_type)}")
                seafloor_type = SeafloorType.UNKNOWN
                confidence = 0.0
            
            self.logger.debug(f"Classified as {seafloor_type.value} with confidence {confidence:.2f}")
            
            return seafloor_type
            
        except Exception as e:
            self.logger.error(f"Error in seafloor classification: {e}")
            return SeafloorType.UNKNOWN
    
    def classify_with_confidence(self, depth_data: np.ndarray) -> Tuple[SeafloorType, float]:
        """Classify seafloor type and return confidence score.
        
        Note: This method DOES return a tuple by design, but the main classify() method
        only returns the enum to prevent tuple errors in the pipeline.
        """
        try:
            seafloor_type = self.classify(depth_data)
            
            # Calculate confidence based on data quality and classification certainty
            if seafloor_type == SeafloorType.UNKNOWN:
                confidence = 0.0
            else:
                # Base confidence calculation
                valid_data = depth_data[np.isfinite(depth_data)]
                data_quality = len(valid_data) / depth_data.size if depth_data.size > 0 else 0
                confidence = min(1.0, data_quality * 1.2)  # Boost good data quality
            
            return seafloor_type, confidence
            
        except Exception as e:
            self.logger.error(f"Error in classification with confidence: {e}")
            return SeafloorType.UNKNOWN, 0.0


class EnhancedAdaptiveProcessor:
    """Enhanced adaptive processor with improved parameter calculation."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.seafloor_classifier = EnhancedSeafloorClassifier()
        self.processing_strategies = {
            SeafloorType.SHALLOW_COASTAL: self._shallow_coastal_strategy,
            SeafloorType.DEEP_OCEAN: self._deep_ocean_strategy,
            SeafloorType.CONTINENTAL_SHELF: self._continental_shelf_strategy,
            SeafloorType.SEAMOUNT: self._seamount_strategy,
            SeafloorType.ABYSSAL_PLAIN: self._abyssal_plain_strategy,
            SeafloorType.UNKNOWN: self._default_strategy
        }
    
    def get_processing_parameters(self, depth_data: np.ndarray) -> Dict:
        """Get adaptive processing parameters based on seafloor type.
        
        FIXED: Ensures seafloor_type is properly handled as enum, not tuple.
        """
        try:
            # Get seafloor type classification (guaranteed to be enum, not tuple)
            seafloor_type = self.seafloor_classifier.classify(depth_data)
            
            # Verify we have a valid enum
            if not isinstance(seafloor_type, SeafloorType):
                self.logger.error(f"Invalid seafloor type returned: {type(seafloor_type)}")
                seafloor_type = SeafloorType.UNKNOWN
            
            # Get base parameters from strategy
            params = self.processing_strategies[seafloor_type](depth_data)
            
            # Add metadata
            params['seafloor_type'] = seafloor_type.value  # Store string value
            params['classification_confidence'] = 1.0  # Default confidence
            
            # Calculate additional adaptive parameters
            self._add_adaptive_enhancements(params, depth_data, seafloor_type)
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error getting processing parameters: {e}")
            # Return safe default parameters
            return self._default_strategy(depth_data)
    
    def _add_adaptive_enhancements(self, params: Dict, depth_data: np.ndarray, 
                                 seafloor_type: SeafloorType):
        """Add enhanced adaptive parameters based on data characteristics."""
        try:
            valid_data = depth_data[np.isfinite(depth_data)]
            if len(valid_data) == 0:
                return
            
            # Calculate data characteristics
            data_std = np.std(valid_data)
            data_range = np.ptp(valid_data)
            mean_depth = np.abs(np.mean(valid_data))
            
            # Add compression ratio based on seafloor type
            if seafloor_type == SeafloorType.SHALLOW_COASTAL:
                params['compression_ratio'] = '6.0:1'
                params['grid_size'] = 8192
            elif seafloor_type == SeafloorType.DEEP_OCEAN:
                params['compression_ratio'] = '8.0:1'  
                params['grid_size'] = 4096
            elif seafloor_type == SeafloorType.SEAMOUNT:
                params['compression_ratio'] = '4.0:1'
                params['grid_size'] = 16384
            else:
                params['compression_ratio'] = '6.0:1'
                params['grid_size'] = 8192
            
            # Adaptive confidence based on data quality
            data_completeness = len(valid_data) / depth_data.size
            noise_level = data_std / max(abs(mean_depth), 1.0)
            
            confidence = min(1.0, data_completeness * (1.0 - min(noise_level, 0.5)))
            params['classification_confidence'] = confidence
            
        except Exception as e:
            self.logger.error(f"Error adding adaptive enhancements: {e}")
            # Add safe defaults
            params['compression_ratio'] = '6.0:1'
            params['grid_size'] = 8192
            params['classification_confidence'] = 0.5
    
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
            'feature_preservation_weight': 0.7,
            'seafloor_type': 'unknown',
            'compression_ratio': '6.0:1',
            'grid_size': 8192,
            'classification_confidence': 0.5
        }


# For backward compatibility, also provide the original class names
class SeafloorClassifier(EnhancedSeafloorClassifier):
    """Backward compatibility alias."""
    pass


class AdaptiveProcessor(EnhancedAdaptiveProcessor):
    """Backward compatibility alias."""
    pass
