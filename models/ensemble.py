"""
Ensemble model architecture for improved bathymetric processing robustness.

This module implements ensemble learning approaches for bathymetric data cleaning
using multiple diverse models to improve prediction reliability and uncertainty estimation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from ..core import BathymetricConstraints, AdaptiveProcessor, BathymetricQualityMetrics
from ..config import Config


class ModelVariant:
    """Configuration for model variants in ensemble."""
    
    def __init__(self, name: str, base_filters: int, depth: int, 
                 dropout_rate: float, activation: str = 'relu'):
        self.name = name
        self.base_filters = base_filters
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.activation = activation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'base_filters': self.base_filters,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        }


class BathymetricEnsemble:
    """Ensemble of models for improved robustness."""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = []
        self.weights = []
        self.model_variants = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize supporting components
        self.adaptive_processor = AdaptiveProcessor() if config.enable_adaptive_processing else None
        self.quality_metrics = BathymetricQualityMetrics()
        self.constraints = BathymetricConstraints() if config.enable_constitutional_constraints else None
        
        # Performance tracking
        self.training_history = []
        self.validation_scores = []
    
    def create_ensemble(self, channels: int) -> List[tf.keras.Model]:
        """Create ensemble of diverse models."""
        self.logger.info(f"Creating ensemble of {self.config.ensemble_size} models")
        
        # Define model variants
        self.model_variants = self._define_model_variants()
        
        models = []
        for i in range(self.config.ensemble_size):
            variant_idx = i % len(self.model_variants)
            variant = self.model_variants[variant_idx]
            
            self.logger.info(f"Creating model {i+1}/{self.config.ensemble_size} with variant: {variant.name}")
            
            model = self._create_model_variant(channels, variant)
            models.append(model)
        
        self.models = models
        self.weights = [1.0 / len(models)] * len(models)  # Equal weights initially
        
        self.logger.info(f"Successfully created ensemble with {len(models)} models")
        return models
    
    def _define_model_variants(self) -> List[ModelVariant]:
        """Define different model architecture variants."""
        base_filters = self.config.base_filters
        base_depth = self.config.depth
        base_dropout = self.config.dropout_rate
        
        variants = [
            ModelVariant("lightweight", max(16, base_filters//2), max(2, base_depth-1), base_dropout*0.5),
            ModelVariant("standard", base_filters, base_depth, base_dropout),
            ModelVariant("robust", min(64, base_filters*2), min(6, base_depth+1), min(0.5, base_dropout*1.5)),
        ]
        
        # Add more variants if ensemble size is large
        if self.config.ensemble_size > 3:
            variants.extend([
                ModelVariant("deep", base_filters, min(7, base_depth+2), base_dropout, "swish"),
                ModelVariant("wide", min(96, base_filters*3), base_depth, base_dropout*0.8),
            ])
        
        return variants
    
    def _create_model_variant(self, channels: int, variant: ModelVariant) -> tf.keras.Model:
        """Create a model variant with specific configuration."""
        try:
            input_shape = (self.config.grid_size, self.config.grid_size, channels)
            input_layer = layers.Input(shape=input_shape, name=f"{variant.name}_input")
            
            # Encoder path
            x = input_layer
            skip_connections = []
            filters = variant.base_filters
            
            for i in range(variant.depth):
                # Convolutional block
                x = layers.Conv2D(
                    filters, (3, 3), 
                    activation=variant.activation, 
                    padding='same',
                    name=f"{variant.name}_dec_conv_{i}_1"
                )(x)
                x = layers.BatchNormalization(name=f"{variant.name}_dec_bn_{i}_1")(x)
                
                x = layers.Conv2D(
                    filters, (3, 3), 
                    activation=variant.activation, 
                    padding='same',
                    name=f"{variant.name}_dec_conv_{i}_2"
                )(x)
                x = layers.BatchNormalization(name=f"{variant.name}_dec_bn_{i}_2")(x)
                
                x = layers.Dropout(variant.dropout_rate, name=f"{variant.name}_dec_dropout_{i}")(x)
            
            # Output layer with uncertainty estimation
            if self.config.enable_uncertainty_estimation:
                # Main output (cleaned depth)
                depth_output = layers.Conv2D(
                    1, (1, 1), 
                    activation='linear', 
                    name=f"{variant.name}_depth_output"
                )(x)
                
                # Uncertainty output
                uncertainty_output = layers.Conv2D(
                    1, (1, 1), 
                    activation='softplus',  # Ensure positive uncertainty
                    name=f"{variant.name}_uncertainty_output"
                )(x)
                
                outputs = [depth_output, uncertainty_output]
                output_names = ['depth', 'uncertainty']
            else:
                # Single output
                output = layers.Conv2D(
                    1, (1, 1), 
                    activation='linear', 
                    name=f"{variant.name}_output"
                )(x)
                outputs = output
                output_names = ['depth']
            
            # Create model
            model = models.Model(
                inputs=input_layer, 
                outputs=outputs,
                name=f"bathymetric_cae_{variant.name}"
            )
            
            # Compile model
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=1e-4
            )
            
            if self.config.enable_uncertainty_estimation:
                # Custom loss for uncertainty-aware training
                model.compile(
                    optimizer=optimizer,
                    loss={
                        'depth': self._uncertainty_aware_loss,
                        'uncertainty': 'mse'
                    },
                    loss_weights={'depth': 1.0, 'uncertainty': 0.1},
                    metrics={
                        'depth': ['mae'],
                        'uncertainty': ['mae']
                    }
                )
            else:
                model.compile(
                    optimizer=optimizer,
                    loss='mse',
                    metrics=['mae']
                )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model variant {variant.name}: {e}")
            raise
    
    def _uncertainty_aware_loss(self, y_true, y_pred, uncertainty):
        """Custom loss function that considers prediction uncertainty."""
        # Negative log-likelihood assuming Gaussian distribution
        precision = 1.0 / (uncertainty + 1e-8)
        nll = 0.5 * tf.reduce_mean(precision * tf.square(y_true - y_pred) + tf.math.log(uncertainty + 1e-8))
        return nll
    
    def predict_ensemble(self, input_data: np.ndarray, 
                        adaptive_params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Make ensemble prediction with adaptive processing."""
        try:
            if len(self.models) == 0:
                raise ValueError("No models in ensemble")
            
            predictions = []
            uncertainties = []
            
            # Get predictions from each model
            for i, model in enumerate(self.models):
                try:
                    pred = model.predict(input_data, verbose=0)
                    
                    if isinstance(pred, list) and len(pred) == 2:
                        # Model with uncertainty estimation
                        depth_pred, uncertainty_pred = pred
                        predictions.append(depth_pred)
                        uncertainties.append(uncertainty_pred)
                    else:
                        # Model without uncertainty
                        predictions.append(pred)
                        uncertainties.append(np.ones_like(pred) * 0.1)  # Default uncertainty
                        
                except Exception as e:
                    self.logger.warning(f"Model {i} prediction failed: {e}")
                    continue
            
            if not predictions:
                raise RuntimeError("All models failed to make predictions")
            
            # Weighted ensemble average
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights[:len(predictions)])
            ensemble_uncertainty = np.average(uncertainties, axis=0, weights=self.weights[:len(predictions)])
            
            # Add ensemble uncertainty (prediction variance)
            prediction_variance = np.var(predictions, axis=0)
            total_uncertainty = ensemble_uncertainty + prediction_variance
            
            # Apply constitutional constraints if enabled
            if self.constraints and self.config.enable_constitutional_constraints:
                ensemble_pred = self._apply_constitutional_constraints(
                    input_data, ensemble_pred, adaptive_params
                )
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_ensemble_metrics(
                input_data, ensemble_pred, total_uncertainty, adaptive_params
            )
            
            return ensemble_pred, ensemble_metrics
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            raise
    
    def _apply_constitutional_constraints(self, original: np.ndarray, 
                                        prediction: np.ndarray,
                                        adaptive_params: Optional[Dict] = None) -> np.ndarray:
        """Apply constitutional AI constraints."""
        try:
            if adaptive_params is None:
                adaptive_params = {}
            
            # Validate and apply corrections
            corrected = self.constraints.apply_corrections(
                original[0, :, :, 0] if original.ndim == 4 else original,
                prediction[0, :, :, 0] if prediction.ndim == 4 else prediction,
                **adaptive_params
            )
            
            # Reshape back to original format if needed
            if prediction.ndim == 4:
                result = prediction.copy()
                result[0, :, :, 0] = corrected
                return result
            else:
                return corrected
                
        except Exception as e:
            self.logger.error(f"Error applying constitutional constraints: {e}")
            return prediction
    
    def _calculate_ensemble_metrics(self, original: np.ndarray, 
                                  prediction: np.ndarray,
                                  uncertainty: np.ndarray,
                                  adaptive_params: Optional[Dict] = None) -> Dict:
        """Calculate comprehensive ensemble metrics."""
        try:
            # Extract data for metric calculation
            orig_data = original[0, :, :, 0] if original.ndim == 4 else original
            pred_data = prediction[0, :, :, 0] if prediction.ndim == 4 else prediction
            
            # Calculate quality metrics
            quality_weights = self.config.get_quality_weights()
            metrics = self.quality_metrics.calculate_all_metrics(
                orig_data, pred_data, uncertainty[0, :, :, 0] if uncertainty.ndim == 4 else uncertainty,
                weights=quality_weights
            )
            
            # Add ensemble-specific metrics
            metrics.update({
                'ensemble_size': len(self.models),
                'ensemble_weights': self.weights.copy(),
                'mean_prediction_uncertainty': float(np.mean(uncertainty)),
                'max_prediction_uncertainty': float(np.max(uncertainty)),
                'prediction_confidence': float(1.0 / (1.0 + np.mean(uncertainty)))
            })
            
            # Add adaptive processing info
            if adaptive_params:
                metrics.update({
                    'adaptive_processing': True,
                    'seafloor_type': adaptive_params.get('seafloor_type', 'unknown'),
                    'processing_strategy': adaptive_params.get('strategy_name', 'unknown')
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble metrics: {e}")
            return {'error': str(e)}
    
    def update_weights_from_performance(self, performance_scores: List[float]):
        """Update ensemble weights based on performance."""
        try:
            if len(performance_scores) != len(self.models):
                self.logger.warning(f"Performance scores length ({len(performance_scores)}) "
                                  f"doesn't match number of models ({len(self.models)})")
                return
            
            # Convert to numpy array and handle edge cases
            scores = np.array(performance_scores)
            
            # Handle case where all scores are the same
            if np.std(scores) < 1e-10:
                self.weights = [1.0 / len(self.models)] * len(self.models)
                return
            
            # Apply softmax weighting with temperature
            temperature = 2.0  # Controls weight distribution sharpness
            exp_scores = np.exp(scores / temperature)
            self.weights = (exp_scores / np.sum(exp_scores)).tolist()
            
            self.logger.info(f"Updated ensemble weights: {[f'{w:.3f}' for w in self.weights]}")
            
        except Exception as e:
            self.logger.error(f"Error updating ensemble weights: {e}")
    
    def save_ensemble(self, base_path: str):
        """Save all models in the ensemble."""
        try:
            base_path = Path(base_path)
            base_path.parent.mkdir(parents=True, exist_ok=True)
            
            ensemble_info = {
                'ensemble_size': len(self.models),
                'weights': self.weights,
                'model_variants': [variant.to_dict() for variant in self.model_variants],
                'config': self.config.to_dict()
            }
            
            # Save ensemble metadata
            import json
            with open(f"{base_path}_ensemble_info.json", 'w') as f:
                json.dump(ensemble_info, f, indent=2)
            
            # Save individual models
            for i, model in enumerate(self.models):
                model_path = f"{base_path}_model_{i}.h5"
                model.save(model_path)
                self.logger.info(f"Saved model {i+1}/{len(self.models)} to {model_path}")
            
            self.logger.info(f"Ensemble saved successfully to {base_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble: {e}")
            raise
    
    def load_ensemble(self, base_path: str):
        """Load ensemble from saved files."""
        try:
            base_path = Path(base_path)
            
            # Load ensemble metadata
            import json
            with open(f"{base_path}_ensemble_info.json", 'r') as f:
                ensemble_info = json.load(f)
            
            self.weights = ensemble_info['weights']
            ensemble_size = ensemble_info['ensemble_size']
            
            # Load individual models
            models = []
            for i in range(ensemble_size):
                model_path = f"{base_path}_model_{i}.h5"
                if Path(model_path).exists():
                    model = tf.keras.models.load_model(model_path, compile=False)
                    models.append(model)
                    self.logger.info(f"Loaded model {i+1}/{ensemble_size} from {model_path}")
                else:
                    self.logger.warning(f"Model file not found: {model_path}")
            
            if len(models) == 0:
                raise FileNotFoundError("No model files found")
            
            self.models = models
            
            # Adjust weights if some models failed to load
            if len(models) < ensemble_size:
                self.weights = self.weights[:len(models)]
                self.weights = [w / sum(self.weights) for w in self.weights]
            
            self.logger.info(f"Ensemble loaded successfully with {len(models)} models")
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble: {e}")
            raise
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary information about the ensemble."""
        summary = {
            'ensemble_size': len(self.models),
            'weights': self.weights.copy(),
            'total_parameters': sum(model.count_params() for model in self.models),
            'model_variants': [variant.to_dict() for variant in self.model_variants],
            'training_history_length': len(self.training_history),
            'validation_scores': self.validation_scores.copy() if self.validation_scores else []
        }
        
        # Add individual model info
        summary['models'] = []
        for i, model in enumerate(self.models):
            model_info = {
                'index': i,
                'name': model.name,
                'parameters': model.count_params(),
                'weight': self.weights[i] if i < len(self.weights) else 0.0
            }
            summary['models'].append(model_info)
        
        return summary
    
    def cross_validate_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                               cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation on the ensemble."""
        try:
            from sklearn.model_selection import KFold
            
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                self.logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
                
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Train ensemble on fold
                fold_models = self.create_ensemble(X_train.shape[-1])
                
                for model in fold_models:
                    model.fit(
                        X_fold_train, y_fold_train,
                        validation_data=(X_fold_val, y_fold_val),
                        epochs=min(20, self.config.epochs),  # Reduced epochs for CV
                        batch_size=self.config.batch_size,
                        verbose=0
                    )
                
                # Evaluate fold
                fold_predictions, _ = self.predict_ensemble(X_fold_val)
                fold_score = self.quality_metrics.calculate_all_metrics(
                    y_fold_val[0, :, :, 0], fold_predictions[0, :, :, 0]
                )['composite_quality']
                
                cv_scores.append(fold_score)
                self.logger.info(f"Fold {fold + 1} score: {fold_score:.4f}")
            
            cv_results = {
                'mean_score': float(np.mean(cv_scores)),
                'std_score': float(np.std(cv_scores)),
                'scores': cv_scores
            }
            
            self.logger.info(f"Cross-validation completed: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            return {'error': str(e)}


def create_bathymetric_ensemble(config: Config) -> BathymetricEnsemble:
    """Factory function to create a bathymetric ensemble."""
    return BathymetricEnsemble(config)
                    padding='same',
                    name=f"{variant.name}_enc_conv_{i}_1"
                )(x)
                x = layers.BatchNormalization(name=f"{variant.name}_enc_bn_{i}_1")(x)
                
                x = layers.Conv2D(
                    filters, (3, 3), 
                    activation=variant.activation, 
                    padding='same',
                    name=f"{variant.name}_enc_conv_{i}_2"
                )(x)
                x = layers.BatchNormalization(name=f"{variant.name}_enc_bn_{i}_2")(x)
                
                # Store skip connection
                skip_connections.append(x)
                
                # Downsampling (except for last layer)
                if i < variant.depth - 1:
                    x = layers.MaxPooling2D((2, 2), name=f"{variant.name}_enc_pool_{i}")(x)
                    x = layers.Dropout(variant.dropout_rate, name=f"{variant.name}_enc_dropout_{i}")(x)
                
                filters = min(512, filters * 2)  # Cap at 512 filters
            
            # Bottleneck
            x = layers.Conv2D(
                filters, (3, 3), 
                activation=variant.activation, 
                padding='same',
                name=f"{variant.name}_bottleneck_1"
            )(x)
            x = layers.BatchNormalization(name=f"{variant.name}_bottleneck_bn")(x)
            x = layers.Dropout(variant.dropout_rate * 1.5, name=f"{variant.name}_bottleneck_dropout")(x)
            
            # Decoder path
            for i in range(variant.depth - 1, -1, -1):
                filters = max(variant.base_filters, filters // 2)
                
                # Upsampling
                if i < variant.depth - 1:
                    x = layers.Conv2DTranspose(
                        filters, (2, 2), 
                        strides=(2, 2), 
                        padding='same',
                        name=f"{variant.name}_dec_upsample_{i}"
                    )(x)
                
                # Skip connection
                if i < len(skip_connections):
                    skip = skip_connections[i]
                    x = layers.Concatenate(name=f"{variant.name}_dec_concat_{i}")([x, skip])
                
                # Convolutional block
                x = layers.Conv2D(
                    filters, (3, 3), 
                    activation=variant.activation, 