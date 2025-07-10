"""
Seafloor classification module for adaptive bathymetric processing.

This module contains classes for classifying seafloor types based on 
bathymetric characteristics to enable adaptive processing strategies.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from scipy import ndimage
from sklearn.cluster import KMeans

from .enums import SeafloorType, DEFAULT_DEPTH_RANGES


class SeafloorFeatureExtractor:
    """Extract features from bathymetric data for classification."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_depth_statistics(self, depth_data: np.ndarray) -> Dict[str, float]:
        """Extract basic depth statistics."""
        try:
            finite_data = depth_data[np.isfinite(depth_data)]
            
            if len(finite_data) == 0:
                return self._get_default_stats()
            
            return {
                'mean_depth': float(np.mean(finite_data)),
                'median_depth': float(np.median(finite_data)),
                'std_depth': float(np.std(finite_data)),
                'min_depth': float(np.min(finite_data)),
                'max_depth': float(np.max(finite_data)),
                'depth_range': float(np.ptp(finite_data)),
                'depth_percentile_25': float(np.percentile(finite_data, 25)),
                'depth_percentile_75': float(np.percentile(finite_data, 75))
            }
        except Exception as e:
            self.logger.error(f"Error extracting depth statistics: {e}")
            return self._get_default_stats()
    
    def extract_topographic_features(self, depth_data: np.ndarray) -> Dict[str, float]:
        """Extract topographic features."""
        try:
            finite_data = depth_data[np.isfinite(depth_data)]
            
            if len(finite_data) == 0:
                return self._get_default_topo_features()
            
            # Calculate gradients
            gradient = np.gradient(depth_data)
            slope = np.sqrt(gradient[0]**2 + gradient[1]**2)
            finite_slope = slope[np.isfinite(slope)]
            
            # Calculate curvature using Laplacian
            laplacian = ndimage.laplace(depth_data.astype(np.float64))
            finite_laplacian = laplacian[np.isfinite(laplacian)]
            
            # Calculate rugosity (terrain roughness)
            rugosity = self._calculate_rugosity(depth_data)
            
            # Calculate terrain variation
            terrain_variation = self._calculate_terrain_variation(depth_data)
            
            return {
                'mean_slope': float(np.mean(finite_slope)) if len(finite_slope) > 0 else 0.0,
                'max_slope': float(np.max(finite_slope)) if len(finite_slope) > 0 else 0.0,
                'slope_std': float(np.std(finite_slope)) if len(finite_slope) > 0 else 0.0,
                'mean_curvature': float(np.mean(finite_laplacian)) if len(finite_laplacian) > 0 else 0.0,
                'curvature_std': float(np.std(finite_laplacian)) if len(finite_laplacian) > 0 else 0.0,
                'rugosity': rugosity,
                'terrain_variation': terrain_variation,
                'relief_ratio': self._calculate_relief_ratio(depth_data)
            }
        except Exception as e:
            self.logger.error(f"Error extracting topographic features: {e}")
            return self._get_default_topo_features()
    
    def _calculate_rugosity(self, depth_data: np.ndarray) -> float:
        """Calculate surface rugosity."""
        try:
            # Simple rugosity calculation using standard deviation of slopes
            gradient = np.gradient(depth_data)
            slope = np.sqrt(gradient[0]**2 + gradient[1]**2)
            return float(np.std(slope[np.isfinite(slope)]))
        except Exception:
            return 0.0
    
    def _calculate_terrain_variation(self, depth_data: np.ndarray) -> float:
        """Calculate terrain variation index."""
        try:
            # Use local standard deviation as terrain variation measure
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            
            # Calculate local mean
            local_mean = ndimage.convolve(depth_data, kernel, mode='constant', cval=0.0)
            
            # Calculate local variance
            local_var = ndimage.convolve((depth_data - local_mean)**2, kernel, mode='constant', cval=0.0)
            
            # Return mean of local standard deviations
            local_std = np.sqrt(local_var)
            return float(np.mean(local_std[np.isfinite(local_std)]))
        except Exception:
            return 0.0
    
    def _calculate_relief_ratio(self, depth_data: np.ndarray) -> float:
        """Calculate relief ratio (range/area)."""
        try:
            finite_data = depth_data[np.isfinite(depth_data)]
            if len(finite_data) == 0:
                return 0.0
            
            depth_range = np.ptp(finite_data)
            area = len(finite_data)  # Simplified area calculation
            
            return float(depth_range / max(area, 1))
        except Exception:
            return 0.0
    
    def _get_default_stats(self) -> Dict[str, float]:
        """Get default depth statistics for error cases."""
        return {
            'mean_depth': 0.0, 'median_depth': 0.0, 'std_depth': 0.0,
            'min_depth': 0.0, 'max_depth': 0.0, 'depth_range': 0.0,
            'depth_percentile_25': 0.0, 'depth_percentile_75': 0.0
        }
    
    def _get_default_topo_features(self) -> Dict[str, float]:
        """Get default topographic features for error cases."""
        return {
            'mean_slope': 0.0, 'max_slope': 0.0, 'slope_std': 0.0,
            'mean_curvature': 0.0, 'curvature_std': 0.0,
            'rugosity': 0.0, 'terrain_variation': 0.0, 'relief_ratio': 0.0
        }


class SeafloorClassificationRules:
    """Rule-based seafloor classification logic."""
    
    def __init__(self):
        self.depth_ranges = DEFAULT_DEPTH_RANGES
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def classify_by_depth(self, depth_stats: Dict[str, float]) -> SeafloorType:
        """Classify based on depth statistics."""
        mean_depth = depth_stats.get('mean_depth', 0)
        
        # Basic depth-based classification
        for seafloor_type, (min_depth, max_depth) in self.depth_ranges.items():
            if min_depth <= abs(mean_depth) <= max_depth:
                return seafloor_type
        
        return SeafloorType.UNKNOWN
    
    def classify_by_topography(self, depth_stats: Dict[str, float], 
                              topo_features: Dict[str, float]) -> SeafloorType:
        """Classify based on topographic features."""
        mean_depth = abs(depth_stats.get('mean_depth', 0))
        depth_range = depth_stats.get('depth_range', 0)
        rugosity = topo_features.get('rugosity', 0)
        mean_slope = topo_features.get('mean_slope', 0)
        relief_ratio = topo_features.get('relief_ratio', 0)
        
        # Seamount detection (high relief in deep water)
        if (mean_depth > 1000 and depth_range > 500 and 
            (rugosity > 0.1 or mean_slope > 0.05)):
            return SeafloorType.SEAMOUNT
        
        # Abyssal plain (deep, flat areas)
        if (mean_depth > 4000 and rugosity < 0.02 and 
            mean_slope < 0.01 and depth_range < 100):
            return SeafloorType.ABYSSAL_PLAIN
        
        # Shallow coastal (shallow, variable)
        if mean_depth < 200 and rugosity > 0.05:
            return SeafloorType.SHALLOW_COASTAL
        
        # Continental shelf
        if 200 <= mean_depth <= 2000:
            return SeafloorType.CONTINENTAL_SHELF
        
        # Deep ocean (general deep areas)
        if mean_depth > 2000:
            return SeafloorType.DEEP_OCEAN
        
        return SeafloorType.UNKNOWN
    
    def refine_classification(self, initial_type: SeafloorType, 
                            depth_stats: Dict[str, float],
                            topo_features: Dict[str, float]) -> SeafloorType:
        """Refine classification using additional rules."""
        # Add specific refinement rules here
        
        # Example: Distinguish between continental shelf types
        if initial_type == SeafloorType.CONTINENTAL_SHELF:
            slope = topo_features.get('mean_slope', 0)
            if slope > 0.1:  # Steep continental slope
                return SeafloorType.CONTINENTAL_SHELF
        
        # Example: Distinguish seamount from continental shelf
        if initial_type == SeafloorType.CONTINENTAL_SHELF:
            relief_ratio = topo_features.get('relief_ratio', 0)
            if relief_ratio > 0.001:  # High relief ratio might indicate seamount
                return SeafloorType.SEAMOUNT
        
        return initial_type


class SeafloorClassifier:
    """Classify seafloor type for adaptive processing."""
    
    def __init__(self, use_ml_classification: bool = False):
        self.feature_extractor = SeafloorFeatureExtractor()
        self.classification_rules = SeafloorClassificationRules()
        self.use_ml_classification = use_ml_classification
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ML classifier (if enabled)
        self.ml_classifier = None
        if use_ml_classification:
            self._initialize_ml_classifier()
    
    def classify(self, depth_data: np.ndarray) -> SeafloorType:
        """Classify seafloor type based on depth and topological features."""
        try:
            # Validate input
            if depth_data is None or depth_data.size == 0:
                return SeafloorType.UNKNOWN
            
            # Extract features
            depth_stats = self.feature_extractor.extract_depth_statistics(depth_data)
            topo_features = self.feature_extractor.extract_topographic_features(depth_data)
            
            # Classify using rules
            depth_classification = self.classification_rules.classify_by_depth(depth_stats)
            topo_classification = self.classification_rules.classify_by_topography(
                depth_stats, topo_features)
            
            # Choose best classification
            if depth_classification == topo_classification:
                final_classification = depth_classification
            elif topo_classification != SeafloorType.UNKNOWN:
                final_classification = topo_classification
            else:
                final_classification = depth_classification
            
            # Refine classification
            final_classification = self.classification_rules.refine_classification(
                final_classification, depth_stats, topo_features)
            
            # Use ML classification if available
            if self.use_ml_classification and self.ml_classifier:
                ml_classification = self._ml_classify(depth_stats, topo_features)
                if ml_classification != SeafloorType.UNKNOWN:
                    final_classification = ml_classification
            
            self.logger.debug(f"Classified seafloor as: {final_classification.value}")
            
            return final_classification
            
        except Exception as e:
            self.logger.error(f"Error classifying seafloor: {e}")
            return SeafloorType.UNKNOWN
    
    def classify_with_confidence(self, depth_data: np.ndarray) -> Tuple[SeafloorType, float]:
        """Classify with confidence score."""
        try:
            classification = self.classify(depth_data)
            confidence = self._calculate_confidence(depth_data, classification)
            return classification, confidence
        except Exception as e:
            self.logger.error(f"Error in classification with confidence: {e}")
            return SeafloorType.UNKNOWN, 0.0
    
    def _calculate_confidence(self, depth_data: np.ndarray, 
                            classification: SeafloorType) -> float:
        """Calculate confidence in classification."""
        try:
            if classification == SeafloorType.UNKNOWN:
                return 0.0
            
            # Extract features
            depth_stats = self.feature_extractor.extract_depth_statistics(depth_data)
            mean_depth = abs(depth_stats.get('mean_depth', 0))
            
            # Calculate confidence based on how well depth fits expected range
            expected_range = classification.depth_range
            
            if expected_range[0] <= mean_depth <= expected_range[1]:
                # Within expected range - high confidence
                range_size = expected_range[1] - expected_range[0]
                center = (expected_range[0] + expected_range[1]) / 2
                distance_from_center = abs(mean_depth - center)
                confidence = 1.0 - (distance_from_center / (range_size / 2))
                return max(0.5, min(1.0, confidence))  # Clamp between 0.5 and 1.0
            else:
                # Outside expected range - lower confidence
                return 0.3
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _initialize_ml_classifier(self):
        """Initialize ML classifier (placeholder for future implementation)."""
        # This is a placeholder for a more sophisticated ML-based classifier
        # Could use trained models for seafloor classification
        self.logger.info("ML classification not yet implemented")
        self.ml_classifier = None
    
    def _ml_classify(self, depth_stats: Dict[str, float], 
                    topo_features: Dict[str, float]) -> SeafloorType:
        """ML-based classification (placeholder)."""
        # Placeholder for ML classification
        return SeafloorType.UNKNOWN
    
    def get_classification_features(self, depth_data: np.ndarray) -> Dict[str, float]:
        """Get all features used for classification."""
        try:
            depth_stats = self.feature_extractor.extract_depth_statistics(depth_data)
            topo_features = self.feature_extractor.extract_topographic_features(depth_data)
            
            # Combine all features
            all_features = {**depth_stats, **topo_features}
            
            return all_features
        except Exception as e:
            self.logger.error(f"Error extracting classification features: {e}")
            return {}
    
    def classify_regions(self, depth_data: np.ndarray, 
                        region_size: int = 128) -> np.ndarray:
        """Classify different regions of the data separately."""
        try:
            height, width = depth_data.shape
            classification_map = np.full((height, width), SeafloorType.UNKNOWN.value, dtype=object)
            
            # Process in overlapping regions
            step_size = region_size // 2
            
            for y in range(0, height - region_size + 1, step_size):
                for x in range(0, width - region_size + 1, step_size):
                    # Extract region
                    region = depth_data[y:y+region_size, x:x+region_size]
                    
                    # Classify region
                    region_type = self.classify(region)
                    
                    # Assign to classification map
                    classification_map[y:y+region_size, x:x+region_size] = region_type.value
            
            return classification_map
            
        except Exception as e:
            self.logger.error(f"Error in regional classification: {e}")
            return np.full(depth_data.shape, SeafloorType.UNKNOWN.value, dtype=object)


def create_seafloor_classifier(config: Optional[Dict] = None) -> SeafloorClassifier:
    """Factory function to create a seafloor classifier."""
    if config is None:
        config = {}
    
    use_ml = config.get('use_ml_classification', False)
    return SeafloorClassifier(use_ml_classification=use_ml)
