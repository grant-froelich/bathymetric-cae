"""
Ensemble model architecture and prediction management.
Fixed to handle missing models and provide fallback behavior.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models

from config.config import Config
from core.adaptive_processor import AdaptiveProcessor
from core.quality_metrics import BathymetricQualityMetrics
from core.constraints import BathymetricConstraints
from models.architectures import create_model_variant


class BathymetricEnsemble:
    """Ensemble of models for improved robustness."""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = []
        self.weights = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.adaptive_processor = AdaptiveProcessor()
        self.quality_metrics = BathymetricQualityMetrics()
        
        # Initialize expert review system if enabled
        try:
            if config.enable_expert_review:
                from review.expert_system import ExpertReviewSystem
                self.expert_review = ExpertReviewSystem()
            else:
                self.expert_review = None
        except Exception as e:
            self.logger.warning(f"Could not initialize expert review system: {e}")
            self.expert_review = None
    
    def create_ensemble(self, channels: int) -> List[tf.keras.Model]:
        """Create ensemble of diverse models with error handling."""
        models = []
        
        self.logger.info(f"Creating ensemble of {self.config.ensemble_size} models...")
        
        for i in range(self.config.ensemble_size):
            try:
                # Create model with slight variations
                model_config = self._get_model_variation(i)
                model = self._create_model_variant(channels, model_config, i)
                models.append(model)
                self.logger.info(f"Created ensemble model {i+1}/{self.config.ensemble_size}")
                
            except Exception as e:
                self.logger.error(f"Failed to create ensemble model {i+1}: {e}")
                # Create fallback simple model
                try:
                    fallback_model = self._create_fallback_model(channels, i)
                    models.append(fallback_model)
                    self.logger.warning(f"Using fallback model for ensemble {i+1}")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback model creation also failed: {fallback_error}")
                    # Skip this model and continue
                    continue
        
        if not models:
            self.logger.error("No models created successfully - creating single fallback model")
            models = [self._create_emergency_fallback(channels)]
        
        self.models = models
        self.weights = [1.0 / len(models)] * len(models)  # Equal weights initially
        
        self.logger.info(f"Ensemble created with {len(models)} models")
        return models
    
    def _get_model_variation(self, index: int) -> Dict:
        """Get model configuration variation with safe parameters."""
        # Use safer, smaller parameters to avoid memory issues
        variations = [
            {'base_filters': 16, 'depth': 2, 'dropout_rate': 0.1, 'name': f'lightweight_{index}'},
            {'base_filters': 24, 'depth': 3, 'dropout_rate': 0.2, 'name': f'standard_{index}'},
            {'base_filters': 32, 'depth': 3, 'dropout_rate': 0.25, 'name': f'enhanced_{index}'}
        ]
        
        return variations[index % len(variations)]
    
    def _create_model_variant(self, channels: int, variant_config: Dict, index: int) -> tf.keras.Model:
        """Create a model variant with comprehensive error handling."""
        input_shape = (self.config.grid_size, self.config.grid_size, channels)
        
        try:
            # Use lightweight architecture for ensemble models
            model = create_model_variant(self.config, 'lightweight', input_shape, variant_config)
            model._name = variant_config.get('name', f'ensemble_model_{index}')
            return model
            
        except Exception as e:
            self.logger.warning(f"Standard model creation failed: {e}")
            return self._create_fallback_model(channels, index)
    
    def _create_fallback_model(self, channels: int, index: int) -> tf.keras.Model:
        """Create a simple fallback model."""
        try:
            input_shape = (self.config.grid_size, self.config.grid_size, channels)
            
            inputs = tf.keras.Input(shape=input_shape)
            
            # Very simple autoencoder
            x = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            
            # Bottleneck
            encoded = tf.keras.layers.Conv2D(4, 3, activation='relu', padding='same')(x)
            
            # Decoder
            x = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(encoded)
            x = tf.keras.layers.UpSampling2D(2)(x)
            
            # Output layer - single channel for depth
            outputs = tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
            
            model = tf.keras.Model(inputs, outputs, name=f'fallback_model_{index}')
            
            # Compile with simple settings
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info(f"Created fallback model {index}")
            return model
            
        except Exception as e:
            self.logger.error(f"Fallback model creation failed: {e}")
            return self._create_emergency_fallback(channels)
    
    def _create_emergency_fallback(self, channels: int) -> tf.keras.Model:
        """Create the simplest possible model as last resort."""
        try:
            input_shape = (self.config.grid_size, self.config.grid_size, channels)
            
            inputs = tf.keras.Input(shape=input_shape)
            
            # Identity-like model that just passes through first channel
            if channels > 1:
                # Take only the first channel (depth)
                x = tf.keras.layers.Lambda(lambda x: x[..., :1])(inputs)
            else:
                x = inputs
            
            # Minimal processing
            outputs = tf.keras.layers.Conv2D(1, 1, activation='linear', padding='same')(x)
            
            model = tf.keras.Model(inputs, outputs, name='emergency_fallback')
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            self.logger.warning("Created emergency fallback model")
            return model
            
        except Exception as e:
            self.logger.critical(f"Emergency fallback model creation failed: {e}")
            raise RuntimeError("Cannot create any functional model")
    
    def predict_ensemble(self, input_data: np.ndarray, 
                        adaptive_params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Make ensemble prediction with comprehensive error handling."""
        
        if not self.models:
            raise RuntimeError("No models available for prediction")
        
        predictions = []
        successful_predictions = 0
        
        self.logger.debug(f"Making ensemble prediction with {len(self.models)} models")
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            try:
                # Ensure input data is float32
                model_input = input_data.astype(np.float32)
                
                pred = model.predict(model_input, verbose=0)
                
                # Ensure prediction is the right shape
                if pred.shape != (input_data.shape[0], input_data.shape[1], input_data.shape[2], 1):
                    self.logger.warning(f"Model {i} prediction shape mismatch: {pred.shape}")
                    # Reshape if possible
                    if pred.size == input_data.shape[0] * input_data.shape[1] * input_data.shape[2]:
                        pred = pred.reshape(input_data.shape[0], input_data.shape[1], input_data.shape[2], 1)
                    else:
                        continue  # Skip this prediction
                
                predictions.append(pred)
                successful_predictions += 1
                
            except Exception as e:
                self.logger.warning(f"Model {i} prediction failed: {e}")
                continue
        
        if not predictions:
            self.logger.error("All model predictions failed - using input as fallback")
            # Return first channel of input as emergency fallback
            fallback_pred = input_data[..., :1]
            return fallback_pred, {'composite_quality': 0.0, 'prediction_source': 'fallback'}
        
        # Calculate weighted ensemble average
        if len(predictions) != len(self.weights):
            # Adjust weights for successful predictions
            active_weights = self.weights[:len(predictions)]
            weight_sum = sum(active_weights)
            if weight_sum > 0:
                active_weights = [w / weight_sum for w in active_weights]
            else:
                active_weights = [1.0 / len(predictions)] * len(predictions)
        else:
            active_weights = self.weights
        
        ensemble_pred = np.average(predictions, axis=0, weights=active_weights)
        
        # Apply constitutional constraints if enabled
        if self.config.enable_constitutional_constraints:
            try:
                ensemble_pred = self._apply_constitutional_constraints(
                    input_data, ensemble_pred, adaptive_params
                )
            except Exception as e:
                self.logger.warning(f"Constitutional constraints failed: {e}")
        
        # Calculate prediction uncertainty
        if len(predictions) > 1:
            prediction_std = np.std(predictions, axis=0)
        else:
            prediction_std = np.zeros_like(ensemble_pred)
        
        # Calculate ensemble metrics
        try:
            ensemble_metrics = self._calculate_ensemble_metrics(
                input_data, ensemble_pred, prediction_std
            )
            ensemble_metrics['successful_models'] = successful_predictions
            ensemble_metrics['total_models'] = len(self.models)
        except Exception as e:
            self.logger.error(f"Metric calculation failed: {e}")
            ensemble_metrics = {
                'composite_quality': 0.5,  # Default score
                'successful_models': successful_predictions,
                'total_models': len(self.models),
                'prediction_source': 'ensemble_with_errors'
            }
        
        # Cleanup memory
        del predictions
        import gc
        gc.collect()
        
        return ensemble_pred, ensemble_metrics
    
    def _apply_constitutional_constraints(self, original: np.ndarray, 
                                        prediction: np.ndarray,
                                        adaptive_params: Optional[Dict] = None) -> np.ndarray:
        """Apply constitutional AI constraints with error handling."""
        if adaptive_params is None:
            adaptive_params = {}
        
        try:
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
            
        except Exception as e:
            self.logger.warning(f"Constitutional constraints error: {e}")
        
        return prediction
    
    def _calculate_ensemble_metrics(self, original: np.ndarray, 
                                  prediction: np.ndarray,
                                  uncertainty: np.ndarray) -> Dict:
        """Calculate comprehensive ensemble metrics with error handling."""
        try:
            orig_data = original[0, :, :, 0]
            pred_data = prediction[0, :, :, 0]
            
            metrics = {
                'ssim': self._calculate_ssim_safe(orig_data, pred_data),
                'roughness': self.quality_metrics.calculate_roughness(pred_data),
                'feature_preservation': self.quality_metrics.calculate_feature_preservation(orig_data, pred_data),
                'consistency': self.quality_metrics.calculate_depth_consistency(pred_data),
                'hydrographic_compliance': self.quality_metrics.calculate_hydrographic_standards_compliance(pred_data),
                'prediction_uncertainty': float(np.mean(uncertainty)) if uncertainty.size > 0 else 0.0,
                'prediction_confidence': float(1.0 / (1.0 + np.mean(uncertainty))) if uncertainty.size > 0 else 1.0
            }
            
            # Calculate composite quality score
            metrics['composite_quality'] = (
                self.config.ssim_weight * metrics['ssim'] +
                self.config.roughness_weight * max(0, 1.0 - metrics['roughness']) +
                self.config.feature_preservation_weight * metrics['feature_preservation'] +
                self.config.consistency_weight * metrics['consistency']
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble metrics: {e}")
            return {
                'composite_quality': 0.5,
                'ssim': 0.0,
                'roughness': 1.0,
                'feature_preservation': 0.0,
                'consistency': 0.0,
                'hydrographic_compliance': 0.0,
                'prediction_uncertainty': 1.0,
                'prediction_confidence': 0.0
            }
    
    def _calculate_ssim_safe(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate SSIM with comprehensive error handling."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            if original.shape != cleaned.shape:
                self.logger.warning("Shape mismatch in SSIM calculation")
                return 0.0
            
            # Ensure finite values
            if not (np.isfinite(original).all() and np.isfinite(cleaned).all()):
                self.logger.warning("Non-finite values in SSIM calculation")
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
        try:
            if len(performance_scores) != len(self.weights):
                self.logger.warning("Performance scores length mismatch with weights")
                return
            
            # Softmax weighting based on performance
            exp_scores = np.exp(np.array(performance_scores))
            self.weights = exp_scores / np.sum(exp_scores)
            
            self.logger.info(f"Updated ensemble weights: {self.weights}")
        except Exception as e:
            self.lo