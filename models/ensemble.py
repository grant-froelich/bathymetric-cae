"""
Ensemble model architecture and prediction management.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.metrics import structural_similarity as ssim

from config.config import Config
from core.adaptive_processor import AdaptiveProcessor
from core.quality_metrics import BathymetricQualityMetrics
from core.constraints import BathymetricConstraints
from review.expert_system import ExpertReviewSystem
from .architectures import create_model_variant


class BathymetricEnsemble:
    """Ensemble of models for improved robustness."""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = []
        self.weights = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.adaptive_processor = AdaptiveProcessor()
        self.quality_metrics = BathymetricQualityMetrics()
        self.expert_review = ExpertReviewSystem() if config.enable_expert_review else None
    
    def create_ensemble(self, channels: int) -> List[tf.keras.Model]:
        """Create ensemble of diverse models."""
        models = []
        
        for i in range(self.config.ensemble_size):
            # Create model with slight variations
            model_config = self._get_model_variation(i)
            model = self._create_model_variant(channels, model_config)
            models.append(model)
        
        self.models = models
        self.weights = [1.0 / len(models)] * len(models)  # Equal weights initially
        
        return models
    
    def _get_model_variation(self, index: int) -> Dict:
        """Get model configuration variation."""
        variations = [
            {'base_filters': 24, 'depth': 3, 'dropout_rate': 0.1},
            {'base_filters': 32, 'depth': 4, 'dropout_rate': 0.2},
            {'base_filters': 48, 'depth': 5, 'dropout_rate': 0.3}
        ]
        
        return variations[index % len(variations)]
    
    def _create_model_variant(self, channels: int, variant_config: Dict) -> tf.keras.Model:
        """Create a model variant with specific configuration."""
        
        input_shape = (self.config.grid_size, self.config.grid_size, channels)
        return create_model_variant(self.config, 'advanced', input_shape, variant_config)
    
    def predict_ensemble(self, input_data: np.ndarray, 
                        adaptive_params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Make ensemble prediction with adaptive processing."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(input_data, verbose=0)
            predictions.append(pred)
        
        # Weighted ensemble average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Apply constitutional constraints if enabled
        if self.config.enable_constitutional_constraints:
            ensemble_pred = self._apply_constitutional_constraints(
                input_data, ensemble_pred, adaptive_params
            )
        
        # Calculate prediction uncertainty
        prediction_std = np.std(predictions, axis=0)
        
        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_ensemble_metrics(
            input_data, ensemble_pred, prediction_std
        )
        
        return ensemble_pred, ensemble_metrics
    
    def _apply_constitutional_constraints(self, original: np.ndarray, 
                                        prediction: np.ndarray,
                                        adaptive_params: Optional[Dict] = None) -> np.ndarray:
        """Apply constitutional AI constraints."""
        if adaptive_params is None:
            adaptive_params = {}
        
        constraints = BathymetricConstraints()
        
        # Check depth continuity
        continuity_violations = constraints.validate_depth_continuity(
            prediction[0, :, :, 0], 
            adaptive_params.get('gradient_constraint', 0.1)
        )
        
        # Check feature preservation
        feature_violations = constraints.preserve_depth_features(
            original[0, :, :, 0], 
            prediction[0, :, :, 0],
            adaptive_params.get('feature_preservation_weight', 0.05)
        )
        
        # Apply corrections where violations occur
        if np.any(continuity_violations) or np.any(feature_violations):
            # Blend with original data in violation areas
            blend_factor = 0.5
            violation_mask = continuity_violations | feature_violations
            
            corrected = prediction.copy()
            corrected[0, violation_mask, 0] = (
                blend_factor * original[0, violation_mask, 0] + 
                (1 - blend_factor) * prediction[0, violation_mask, 0]
            )
            
            return corrected
        
        return prediction
    
    def _calculate_ensemble_metrics(self, original: np.ndarray, 
                                  prediction: np.ndarray,
                                  uncertainty: np.ndarray) -> Dict:
        """Calculate comprehensive ensemble metrics."""
        orig_data = original[0, :, :, 0]
        pred_data = prediction[0, :, :, 0]
        
        metrics = {
            'ssim': self._calculate_ssim_safe(orig_data, pred_data),
            'roughness': self.quality_metrics.calculate_roughness(pred_data),
            'feature_preservation': self.quality_metrics.calculate_feature_preservation(orig_data, pred_data),
            'consistency': self.quality_metrics.calculate_depth_consistency(pred_data),
            'hydrographic_compliance': self.quality_metrics.calculate_hydrographic_standards_compliance(pred_data),
            'prediction_uncertainty': float(np.mean(uncertainty)),
            'prediction_confidence': float(1.0 / (1.0 + np.mean(uncertainty)))
        }
        
        # Calculate composite quality score
        metrics['composite_quality'] = (
            self.config.ssim_weight * metrics['ssim'] +
            self.config.roughness_weight * (1.0 - metrics['roughness']) +
            self.config.feature_preservation_weight * metrics['feature_preservation'] +
            self.config.consistency_weight * metrics['consistency']
        )
        
        return metrics
    
    def _calculate_ssim_safe(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate SSIM with comprehensive error handling."""
        try:
            if original.shape != cleaned.shape:
                return 0.0
            
            data_range = float(max(cleaned.max() - cleaned.min(), 1e-8))
            
            return ssim(
                original.astype(np.float64),
                cleaned.astype(np.float64),
                data_range=data_range,
                gaussian_weights=True,
                win_size=min(7, min(original.shape[-2:]))
            )
        except Exception as e:
            self.logger.error(f"Error calculating SSIM: {e}")
            return 0.0
    
    def update_weights_from_performance(self, performance_scores: List[float]):
        """Update ensemble weights based on performance."""
        # Softmax weighting based on performance
        exp_scores = np.exp(np.array(performance_scores))
        self.weights = exp_scores / np.sum(exp_scores)
        
        self.logger.info(f"Updated ensemble weights: {self.weights}")
