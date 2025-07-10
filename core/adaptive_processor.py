"""
Adaptive processing strategies based on seafloor type and local characteristics.

This module implements adaptive processing parameters that are automatically
adjusted based on the classified seafloor type and local data characteristics.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod

from .enums import SeafloorType, ProcessingStrategy
from .seafloor_classifier import SeafloorClassifier


class ProcessingParameters:
    """Container for processing parameters."""
    
    def __init__(self, **kwargs):
        # Default parameters
        self.smoothing_factor = kwargs.get('smoothing_factor', 0.5)
        self.edge_preservation = kwargs.get('edge_preservation', 0.6)
        self.noise_threshold = kwargs.get('noise_threshold', 0.15)
        self.gradient_constraint = kwargs.get('gradient_constraint', 0.1)
        self.feature_preservation_weight = kwargs.get('feature_preservation_weight', 0.7)
        self.uncertainty_weight = kwargs.get('uncertainty_weight', 0.3)
        self.blend_factor = kwargs.get('blend_factor', 0.5)
        self.kernel_size = kwargs.get('kernel_size', 3)
        
        # Advanced parameters
        self.adaptive_smoothing = kwargs.get('adaptive_smoothing', True)
        self.edge_enhancement = kwargs.get('edge_enhancement', False)
        self.multi_scale_processing = kwargs.get('multi_scale_processing', False)
        self.uncertainty_guided = kwargs.get('uncertainty_guided', True)
        
        # Validation
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate parameter ranges."""
        constraints = {
            'smoothing_factor': (0.0, 1.0),
            'edge_preservation': (0.0, 1.0),
            'noise_threshold': (0.0, 1.0),
            'gradient_constraint': (0.0, 1.0),
            'feature_preservation_weight': (0.0, 1.0),
            'uncertainty_weight': (0.0, 1.0),
            'blend_factor': (0.0, 1.0)
        }
        
        for param, (min_val, max_val) in constraints.items():
            value = getattr(self, param)
            if not min_val <= value <= max_val:
                setattr(self, param, np.clip(value, min_val, max_val))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, **kwargs):
        """Update parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._validate_parameters()


class AdaptiveStrategy(ABC):
    """Abstract base class for adaptive processing strategies."""
    
    @abstractmethod
    def get_parameters(self, depth_data: np.ndarray, 
                      local_characteristics: Optional[Dict] = None) -> ProcessingParameters:
        """Get processing parameters for this strategy."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class ShallowCoastalStrategy(AdaptiveStrategy):
    """Processing strategy for shallow coastal areas."""
    
    def get_parameters(self, depth_data: np.ndarray, 
                      local_characteristics: Optional[Dict] = None) -> ProcessingParameters:
        """Get parameters optimized for shallow coastal processing."""
        # Shallow coastal areas need careful feature preservation
        base_params = {
            'smoothing_factor': 0.3,
            'edge_preservation': 0.8,
            'noise_threshold': 0.1,
            'gradient_constraint': 0.05,
            'feature_preservation_weight': 0.9,
            'uncertainty_weight': 0.4,
            'adaptive_smoothing': True,
            'edge_enhancement': True
        }
        
        # Adjust based on local characteristics
        if local_characteristics:
            # Higher noise threshold if high variability detected
            if local_characteristics.get('high_variability', False):
                base_params['noise_threshold'] = 0.15
                base_params['smoothing_factor'] = 0.4
            
            # Lower smoothing if many small features
            if local_characteristics.get('small_features', False):
                base_params['smoothing_factor'] = 0.2
                base_params['feature_preservation_weight'] = 0.95
        
        return ProcessingParameters(**base_params)
    
    def get_strategy_name(self) -> str:
        return "shallow_coastal"


class DeepOceanStrategy(AdaptiveStrategy):
    """Processing strategy for deep ocean areas."""
    
    def get_parameters(self, depth_data: np.ndarray, 
                      local_characteristics: Optional[Dict] = None) -> ProcessingParameters:
        """Get parameters optimized for deep ocean processing."""
        # Deep ocean can tolerate more aggressive smoothing
        base_params = {
            'smoothing_factor': 0.7,
            'edge_preservation': 0.4,
            'noise_threshold': 0.2,
            'gradient_constraint': 0.1,
            'feature_preservation_weight': 0.6,
            'uncertainty_weight': 0.5,
            'adaptive_smoothing': True,
            'multi_scale_processing': True
        }
        
        # Adjust based on local characteristics
        if local_characteristics:
            # Less smoothing if significant features detected
            if local_characteristics.get('significant_features', False):
                base_params['smoothing_factor'] = 0.5
                base_params['feature_preservation_weight'] = 0.8
            
            # More aggressive if very smooth area
            if local_characteristics.get('very_smooth', False):
                base_params['smoothing_factor'] = 0.8
                base_params['noise_threshold'] = 0.25
        
        return ProcessingParameters(**base_params)
    
    def get_strategy_name(self) -> str:
        return "deep_ocean"


class ContinentalShelfStrategy(AdaptiveStrategy):
    """Processing strategy for continental shelf areas."""
    
    def get_parameters(self, depth_data: np.ndarray, 
                      local_characteristics: Optional[Dict] = None) -> ProcessingParameters:
        """Get parameters optimized for continental shelf processing."""
        # Balanced approach for continental shelf
        base_params = {
            'smoothing_factor': 0.5,
            'edge_preservation': 0.6,
            'noise_threshold': 0.15,
            'gradient_constraint': 0.08,
            'feature_preservation_weight': 0.75,
            'uncertainty_weight': 0.4,
            'adaptive_smoothing': True,
            'edge_enhancement': False
        }
        
        # Adjust based on local characteristics
        if local_characteristics:
            slope = local_characteristics.get('mean_slope', 0)
            
            # Steeper areas need more feature preservation
            if slope > 0.05:
                base_params['edge_preservation'] = 0.8
                base_params['feature_preservation_weight'] = 0.9
                base_params['gradient_constraint'] = 0.12
            
            # Flatter areas can use more smoothing
            elif slope < 0.02:
                base_params['smoothing_factor'] = 0.6
                base_params['edge_preservation'] = 0.4
        
        return ProcessingParameters(**base_params)
    
    def get_strategy_name(self) -> str:
        return "continental_shelf"


class SeamountStrategy(AdaptiveStrategy):
    """Processing strategy for seamount areas."""
    
    def get_parameters(self, depth_data: np.ndarray, 
                      local_characteristics: Optional[Dict] = None) -> ProcessingParameters:
        """Get parameters optimized for seamount processing."""
        # Seamounts require maximum feature preservation
        base_params = {
            'smoothing_factor': 0.2,
            'edge_preservation': 0.9,
            'noise_threshold': 0.05,
            'gradient_constraint': 0.15,
            'feature_preservation_weight': 0.95,
            'uncertainty_weight': 0.3,
            'adaptive_smoothing': True,
            'edge_enhancement': True,
            'multi_scale_processing': True
        }
        
        # Adjust based on local characteristics
        if local_characteristics:
            relief = local_characteristics.get('relief_ratio', 0)
            
            # Very high relief needs even more careful processing
            if relief > 0.001:
                base_params['smoothing_factor'] = 0.1
                base_params['edge_preservation'] = 0.95
                base_params['feature_preservation_weight'] = 0.98
        
        return ProcessingParameters(**base_params)
    
    def get_strategy_name(self) -> str:
        return "seamount"


class AbyssalPlainStrategy(AdaptiveStrategy):
    """Processing strategy for abyssal plain areas."""
    
    def get_parameters(self, depth_data: np.ndarray, 
                      local_characteristics: Optional[Dict] = None) -> ProcessingParameters:
        """Get parameters optimized for abyssal plain processing."""
        # Abyssal plains are very flat, allow aggressive smoothing
        base_params = {
            'smoothing_factor': 0.8,
            'edge_preservation': 0.3,
            'noise_threshold': 0.25,
            'gradient_constraint': 0.05,
            'feature_preservation_weight': 0.5,
            'uncertainty_weight': 0.6,
            'adaptive_smoothing': False,  # Uniform smoothing for flat areas
            'multi_scale_processing': False
        }
        
        # Adjust based on local characteristics
        if local_characteristics:
            # If any significant features detected, be more conservative
            if local_characteristics.get('unexpected_features', False):
                base_params['smoothing_factor'] = 0.6
                base_params['feature_preservation_weight'] = 0.7
                base_params['edge_preservation'] = 0.5
        
        return ProcessingParameters(**base_params)
    
    def get_strategy_name(self) -> str:
        return "abyssal_plain"


class DefaultStrategy(AdaptiveStrategy):
    """Default processing strategy for unknown areas."""
    
    def get_parameters(self, depth_data: np.ndarray, 
                      local_characteristics: Optional[Dict] = None) -> ProcessingParameters:
        """Get balanced default parameters."""
        base_params = {
            'smoothing_factor': 0.5,
            'edge_preservation': 0.6,
            'noise_threshold': 0.15,
            'gradient_constraint': 0.1,
            'feature_preservation_weight': 0.7,
            'uncertainty_weight': 0.4,
            'adaptive_smoothing': True
        }
        
        return ProcessingParameters(**base_params)
    
    def get_strategy_name(self) -> str:
        return "default"


class LocalCharacteristicsAnalyzer:
    """Analyze local characteristics of bathymetric data."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(self, depth_data: np.ndarray) -> Dict[str, Any]:
        """Analyze local characteristics of the data."""
        try:
            characteristics = {}
            
            # Basic statistics
            finite_data = depth_data[np.isfinite(depth_data)]
            if len(finite_data) == 0:
                return characteristics
            
            # Calculate gradients and slopes
            gradient = np.gradient(depth_data)
            slope = np.sqrt(gradient[0]**2 + gradient[1]**2)
            finite_slope = slope[np.isfinite(slope)]
            
            characteristics.update({
                'mean_slope': float(np.mean(finite_slope)) if len(finite_slope) > 0 else 0.0,
                'max_slope': float(np.max(finite_slope)) if len(finite_slope) > 0 else 0.0,
                'slope_std': float(np.std(finite_slope)) if len(finite_slope) > 0 else 0.0,
                'depth_range': float(np.ptp(finite_data)),
                'depth_std': float(np.std(finite_data))
            })
            
            # Feature detection
            characteristics.update(self._detect_features(depth_data, characteristics))
            
            # Noise estimation
            characteristics['estimated_noise'] = self._estimate_noise_level(depth_data)
            
            # Quality indicators
            characteristics.update(self._assess_data_quality(depth_data, characteristics))
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing local characteristics: {e}")
            return {}
    
    def _detect_features(self, depth_data: np.ndarray, stats: Dict) -> Dict[str, bool]:
        """Detect various seafloor features."""
        features = {}
        
        try:
            # High variability indicator
            features['high_variability'] = stats.get('depth_std', 0) > 10.0
            
            # Small features (high local variation)
            features['small_features'] = stats.get('slope_std', 0) > 0.02
            
            # Significant features (large depth changes)
            features['significant_features'] = stats.get('depth_range', 0) > 100.0
            
            # Very smooth area
            features['very_smooth'] = (stats.get('slope_std', 0) < 0.005 and 
                                     stats.get('depth_std', 0) < 5.0)
            
            # Unexpected features in what should be smooth areas
            mean_slope = stats.get('mean_slope', 0)
            features['unexpected_features'] = (mean_slope < 0.01 and 
                                             stats.get('max_slope', 0) > 0.05)
            
        except Exception as e:
            self.logger.error(f"Error detecting features: {e}")
        
        return features
    
    def _estimate_noise_level(self, depth_data: np.ndarray) -> float:
        """Estimate noise level in the data."""
        try:
            # Use high-frequency component as noise estimate
            from scipy import ndimage
            
            # Apply high-pass filter
            low_pass = ndimage.gaussian_filter(depth_data, sigma=2)
            high_freq = depth_data - low_pass
            
            # Estimate noise as standard deviation of high-frequency component
            finite_hf = high_freq[np.isfinite(high_freq)]
            if len(finite_hf) > 0:
                return float(np.std(finite_hf))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error estimating noise level: {e}")
            return 0.1  # Default noise estimate
    
    def _assess_data_quality(self, depth_data: np.ndarray, stats: Dict) -> Dict[str, Any]:
        """Assess overall data quality indicators."""
        quality = {}
        
        try:
            # Data completeness
            total_pixels = depth_data.size
            valid_pixels = np.sum(np.isfinite(depth_data))
            quality['completeness'] = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0
            
            # Smoothness indicator
            quality['smoothness'] = 1.0 / (1.0 + stats.get('slope_std', 0))
            
            # Consistency indicator (low noise)
            quality['consistency'] = 1.0 / (1.0 + stats.get('estimated_noise', 0.1))
            
            # Overall quality score
            quality['overall_quality'] = (
                0.4 * quality['completeness'] +
                0.3 * quality['smoothness'] +
                0.3 * quality['consistency']
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
        
        return quality


class AdaptiveProcessor:
    """Adaptive processing based on seafloor type and local characteristics."""
    
    def __init__(self, use_local_analysis: bool = True):
        self.seafloor_classifier = SeafloorClassifier()
        self.characteristics_analyzer = LocalCharacteristicsAnalyzer() if use_local_analysis else None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize strategies
        self.processing_strategies = {
            SeafloorType.SHALLOW_COASTAL: ShallowCoastalStrategy(),
            SeafloorType.DEEP_OCEAN: DeepOceanStrategy(),
            SeafloorType.CONTINENTAL_SHELF: ContinentalShelfStrategy(),
            SeafloorType.SEAMOUNT: SeamountStrategy(),
            SeafloorType.ABYSSAL_PLAIN: AbyssalPlainStrategy(),
            SeafloorType.UNKNOWN: DefaultStrategy()
        }
    
    def get_processing_parameters(self, depth_data: np.ndarray) -> Dict[str, Any]:
        """Get adaptive processing parameters based on seafloor type."""
        try:
            # Classify seafloor type
            seafloor_type, confidence = self.seafloor_classifier.classify_with_confidence(depth_data)
            
            # Analyze local characteristics if enabled
            local_characteristics = None
            if self.characteristics_analyzer:
                local_characteristics = self.characteristics_analyzer.analyze(depth_data)
            
            # Get strategy
            strategy = self.processing_strategies.get(seafloor_type, self.processing_strategies[SeafloorType.UNKNOWN])
            
            # Get parameters
            params = strategy.get_parameters(depth_data, local_characteristics)
            
            # Add metadata
            result = params.to_dict()
            result.update({
                'seafloor_type': seafloor_type.value,
                'classification_confidence': confidence,
                'strategy_name': strategy.get_strategy_name(),
                'local_characteristics': local_characteristics or {}
            })
            
            self.logger.debug(f"Adaptive parameters for {seafloor_type.value}: {strategy.get_strategy_name()}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting adaptive parameters: {e}")
            # Return default parameters
            default_strategy = DefaultStrategy()
            default_params = default_strategy.get_parameters(depth_data)
            result = default_params.to_dict()
            result.update({
                'seafloor_type': SeafloorType.UNKNOWN.value,
                'classification_confidence': 0.0,
                'strategy_name': 'error_fallback',
                'local_characteristics': {}
            })
            return result
    
    def get_processing_parameters_for_type(self, seafloor_type: SeafloorType, 
                                         depth_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get parameters for a specific seafloor type."""
        strategy = self.processing_strategies.get(seafloor_type, self.processing_strategies[SeafloorType.UNKNOWN])
        
        # Analyze local characteristics if data provided
        local_characteristics = None
        if depth_data is not None and self.characteristics_analyzer:
            local_characteristics = self.characteristics_analyzer.analyze(depth_data)
        
        params = strategy.get_parameters(depth_data or np.array([]), local_characteristics)
        
        result = params.to_dict()
        result.update({
            'seafloor_type': seafloor_type.value,
            'strategy_name': strategy.get_strategy_name(),
            'local_characteristics': local_characteristics or {}
        })
        
        return result
    
    def add_custom_strategy(self, seafloor_type: SeafloorType, strategy: AdaptiveStrategy):
        """Add or replace a processing strategy."""
        self.processing_strategies[seafloor_type] = strategy
        self.logger.info(f"Added custom strategy for {seafloor_type.value}: {strategy.get_strategy_name()}")
    
    def get_available_strategies(self) -> Dict[str, str]:
        """Get list of available strategies."""
        return {
            seafloor_type.value: strategy.get_strategy_name()
            for seafloor_type, strategy in self.processing_strategies.items()
        }


def create_adaptive_processor(config: Optional[Dict] = None) -> AdaptiveProcessor:
    """Factory function to create an adaptive processor."""
    if config is None:
        config = {}
    
    use_local_analysis = config.get('use_local_analysis', True)
    return AdaptiveProcessor(use_local_analysis=use_local_analysis)