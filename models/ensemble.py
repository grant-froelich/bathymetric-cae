"""
Enhanced Ensemble Model with Bathymetric Feature Preservation
============================================================

This module implements an enhanced ensemble model that incorporates bathymetric
feature preservation capabilities for underwater terrain reconstruction.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from core.loss_functions import bathymetric_feature_preservation_loss
from models.architectures import create_model_variant
from config.config import Config


class BathymetricEnsemble:
    """Enhanced ensemble model with feature preservation capabilities."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced ensemble model.
        
        Args:
            config: Model configuration object
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.models = []
        self.ensemble_weights = None
        self.feature_extractors = {}
        
    def create_ensemble_model(self, input_shape: tuple, index: int) -> tf.keras.Model:
        """
        Create individual ensemble model with enhanced architecture.
        
        Args:
            input_shape: Input tensor shape
            index: Model index in ensemble
            
        Returns:
            Compiled Keras model with feature preservation capabilities
        """
        # Import new loss functions
        from core.loss_functions import bathymetric_feature_preservation_loss
        
        # Create variant config for feature preservation
        variant_config = {
            'base_filters': self.config.base_filters,
            'depth': self.config.depth,
            'dropout_rate': self.config.dropout_rate,
            'name': f'feature_preserving_model_{index}',
            'preserve_anthropogenic': self.config.preserve_anthropogenic_features,
            'preserve_geological': self.config.preserve_geological_features,
            'feature_weight': getattr(self.config, 'feature_preservation_weight', 0.3),
            'edge_threshold': getattr(self.config, 'edge_detection_threshold', 0.1),
            'smoothness_factor': getattr(self.config, 'bathymetric_smoothness_factor', 0.2)
        }
        
        try:
            # Use feature-preserving architecture
            model = create_model_variant(
                self.config,
                self.config.model_variant,  # Use config setting
                input_shape,
                variant_config
            )
            
            # Apply custom loss function if enabled
            if self.config.use_edge_preserving_loss:
                custom_loss = bathymetric_feature_preservation_loss(self.config)
                
                # Enhanced metrics for bathymetric analysis
                custom_metrics = [
                    'anthropogenic_preservation',
                    'geological_preservation',
                    self._bathymetric_gradient_metric,
                    self._feature_continuity_metric,
                    self._depth_accuracy_metric
                ]
                
                # Compile with enhanced configuration
                model.compile(
                    optimizer=self._get_enhanced_optimizer(),
                    loss=custom_loss,
                    metrics=model.metrics + custom_metrics,
                    loss_weights=self._get_loss_weights()
                )
            else:
                # Standard compilation
                model.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
                
            # Add feature extraction layers if configured
            if getattr(self.config, 'extract_bathymetric_features', False):
                model = self._add_feature_extraction_layers(model, variant_config)
                
            self.logger.info(f"Enhanced ensemble model {index} created successfully")
            return model
            
        except Exception as e:
            self.logger.warning(f"Enhanced model creation failed: {e}")
            return self._create_fallback_model(input_shape[-1], index)
    
    def _get_enhanced_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Create enhanced optimizer for bathymetric feature preservation.
        
        Returns:
            Configured optimizer instance
        """
        learning_rate = getattr(self.config, 'learning_rate', 0.001)
        
        # Use learning rate scheduling for better convergence
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        
        # Enhanced optimizer configuration
        if getattr(self.config, 'use_adamw', False):
            return tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
        else:
            return tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
    
    def _get_loss_weights(self) -> Dict[str, float]:
        """
        Get loss weights for multi-objective optimization.
        
        Returns:
            Dictionary of loss weights
        """
        return {
            'bathymetric_reconstruction': 1.0,
            'feature_preservation': getattr(self.config, 'feature_weight', 0.3),
            'edge_preservation': getattr(self.config, 'edge_weight', 0.2),
            'smoothness_regularization': getattr(self.config, 'smoothness_weight', 0.1)
        }
    
    def _add_feature_extraction_layers(self, model: tf.keras.Model, 
                                     variant_config: Dict) -> tf.keras.Model:
        """
        Add feature extraction layers for bathymetric analysis.
        
        Args:
            model: Base model
            variant_config: Variant configuration
            
        Returns:
            Model with added feature extraction layers
        """
        # Extract intermediate features
        feature_layers = []
        for i, layer in enumerate(model.layers):
            if 'conv' in layer.name.lower() and i % 3 == 0:
                feature_layers.append(layer.output)
        
        # Create feature extraction outputs
        if feature_layers:
            feature_extractor = tf.keras.Model(
                inputs=model.input,
                outputs=feature_layers + [model.output],
                name=f"feature_extractor_{variant_config['name']}"
            )
            
            # Store for ensemble analysis
            self.feature_extractors[variant_config['name']] = feature_extractor
            
        return model
    
    def _create_fallback_model(self, output_channels: int, index: int) -> tf.keras.Model:
        """
        Create fallback model when enhanced creation fails.
        
        Args:
            output_channels: Number of output channels
            index: Model index
            
        Returns:
            Simple fallback model
        """
        self.logger.info(f"Creating fallback model {index}")
        
        inputs = tf.keras.layers.Input(shape=self.config.input_shape)
        
        # Simple CNN architecture
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='linear')(x)
        
        model = tf.keras.Model(inputs, outputs, name=f'fallback_model_{index}')
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def _bathymetric_gradient_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Custom metric for bathymetric gradient preservation.
        
        Args:
            y_true: Ground truth bathymetry
            y_pred: Predicted bathymetry
            
        Returns:
            Gradient preservation metric
        """
        # Calculate gradients
        grad_true_x = tf.image.sobel_edges(y_true)[..., 0]
        grad_pred_x = tf.image.sobel_edges(y_pred)[..., 0]
        grad_true_y = tf.image.sobel_edges(y_true)[..., 1]
        grad_pred_y = tf.image.sobel_edges(y_pred)[..., 1]
        
        # Compute gradient similarity
        grad_similarity = tf.reduce_mean(
            tf.nn.cosine_similarity(
                tf.concat([grad_true_x, grad_true_y], axis=-1),
                tf.concat([grad_pred_x, grad_pred_y], axis=-1),
                axis=-1
            )
        )
        
        return grad_similarity
    
    @staticmethod
    def _feature_continuity_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Custom metric for feature continuity assessment.
        
        Args:
            y_true: Ground truth bathymetry
            y_pred: Predicted bathymetry
            
        Returns:
            Feature continuity metric
        """
        # Calculate second derivatives (curvature)
        laplacian_true = tf.nn.conv2d(
            y_true,
            tf.constant([[[[-1., -1., -1.]], [[-1., 8., -1.]], [[-1., -1., -1.]]]]),
            strides=1,
            padding='SAME'
        )
        
        laplacian_pred = tf.nn.conv2d(
            y_pred,
            tf.constant([[[[-1., -1., -1.]], [[-1., 8., -1.]], [[-1., -1., -1.]]]]),
            strides=1,
            padding='SAME'
        )
        
        # Measure continuity preservation
        continuity = 1.0 - tf.reduce_mean(tf.abs(laplacian_true - laplacian_pred))
        return tf.maximum(continuity, 0.0)
    
    @staticmethod
    def _depth_accuracy_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Custom metric for depth-weighted accuracy.
        
        Args:
            y_true: Ground truth bathymetry
            y_pred: Predicted bathymetry
            
        Returns:
            Depth-weighted accuracy metric
        """
        # Weight errors by depth (deeper areas get higher weight)
        depth_weights = tf.abs(y_true) + 1.0  # Add 1 to avoid zero weights
        weighted_error = tf.abs(y_true - y_pred) * depth_weights
        
        # Normalize by total weight
        accuracy = 1.0 - (tf.reduce_sum(weighted_error) / tf.reduce_sum(depth_weights))
        return tf.maximum(accuracy, 0.0)
    
    def build_ensemble(self, input_shape: tuple, num_models: int = 5) -> List[tf.keras.Model]:
        """
        Build complete ensemble with enhanced models.
        
        Args:
            input_shape: Input tensor shape
            num_models: Number of models in ensemble
            
        Returns:
            List of compiled ensemble models
        """
        self.models = []
        
        for i in range(num_models):
            self.logger.info(f"Creating ensemble model {i+1}/{num_models}")
            model = self.create_ensemble_model(input_shape, i)
            self.models.append(model)
        
        self.logger.info(f"Ensemble with {len(self.models)} models created successfully")
        return self.models
    
    def predict_ensemble(self, x: np.ndarray, 
                        return_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make ensemble predictions with uncertainty estimation.
        
        Args:
            x: Input data
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Tuple of (predictions, uncertainty) if return_uncertainty=True,
            otherwise just predictions
        """
        if not self.models:
            raise ValueError("Ensemble not built. Call build_ensemble() first.")
        
        predictions = []
        for model in self.models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate ensemble mean
        ensemble_mean = np.mean(predictions, axis=0)
        
        if return_uncertainty:
            # Calculate uncertainty as standard deviation
            ensemble_std = np.std(predictions, axis=0)
            return ensemble_mean, ensemble_std
        
        return ensemble_mean
    
    def save_ensemble(self, save_dir: str) -> None:
        """
        Save ensemble models to directory.
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = save_path / f"ensemble_model_{i}.keras"
            model.save(model_path)
            self.logger.info(f"Saved model {i} to {model_path}")
    
    def load_ensemble(self, save_dir: str) -> None:
        """
        Load ensemble models from directory.
        
        Args:
            save_dir: Directory containing saved models
        """
        save_path = Path(save_dir)
        self.models = []
        
        model_files = sorted(save_path.glob("ensemble_model_*.keras"))
        
        for model_file in model_files:
            model = tf.keras.models.load_model(
                model_file,
                custom_objects={
                    'bathymetric_feature_preservation_loss': bathymetric_feature_preservation_loss,
                    '_bathymetric_gradient_metric': self._bathymetric_gradient_metric,
                    '_feature_continuity_metric': self._feature_continuity_metric,
                    '_depth_accuracy_metric': self._depth_accuracy_metric
                }
            )
            self.models.append(model)
            self.logger.info(f"Loaded model from {model_file}")


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration for bathymetric reconstruction
    class ExampleConfig:
        def __init__(self):
            self.base_filters = 64
            self.depth = 5
            self.dropout_rate = 0.1
            self.model_variant = 'unet'
            self.use_edge_preserving_loss = True
            self.preserve_anthropogenic_features = True
            self.preserve_geological_features = True
            self.feature_preservation_weight = 0.3
            self.edge_detection_threshold = 0.1
            self.bathymetric_smoothness_factor = 0.2
            self.extract_bathymetric_features = True
            self.input_shape = (256, 256, 3)
            self.learning_rate = 0.001
            self.use_adamw = True
    
    # Initialize enhanced ensemble
    config = ExampleConfig()
    ensemble = EnhancedEnsembleModel(config)
    
    # Build ensemble
    input_shape = (256, 256, 3)
    models = ensemble.build_ensemble(input_shape, num_models=3)
    
    print(f"Enhanced ensemble with {len(models)} models created successfully!")
    print("Features: Bathymetric preservation, uncertainty estimation, custom metrics")
