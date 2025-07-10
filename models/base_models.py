"""
Base model architectures for Enhanced Bathymetric CAE Processing.

This module contains individual model architectures and variants
used in the ensemble system.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import logging
from typing import Dict, Optional, Tuple, Any

from ..config import Config


class BaseCAE:
    """Base class for Convolutional Autoencoder architectures."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_model(self, input_shape: Tuple[int, int, int], name: str = "base_cae") -> tf.keras.Model:
        """Create the base CAE model."""
        raise NotImplementedError("Subclasses must implement create_model")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'grid_size': self.config.grid_size,
            'base_filters': self.config.base_filters,
            'depth': self.config.depth,
            'dropout_rate': self.config.dropout_rate
        }


class LightweightCAE(BaseCAE):
    """Lightweight CAE for fast processing."""
    
    def create_model(self, input_shape: Tuple[int, int, int], name: str = "lightweight_cae") -> tf.keras.Model:
        """Create lightweight CAE model."""
        try:
            input_layer = layers.Input(shape=input_shape, name=f"{name}_input")
            
            # Reduced parameters for lightweight model
            base_filters = max(16, self.config.base_filters // 2)
            depth = max(2, self.config.depth - 1)
            dropout = self.config.dropout_rate * 0.5
            
            # Encoder
            x = input_layer
            skip_connections = []
            filters = base_filters
            
            for i in range(depth):
                # Single conv block per level for speed
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}")(x)
                
                skip_connections.append(x)
                
                if i < depth - 1:
                    x = layers.MaxPooling2D((2, 2), name=f"{name}_enc_pool_{i}")(x)
                    x = layers.Dropout(dropout, name=f"{name}_enc_dropout_{i}")(x)
                
                filters = min(256, filters * 2)  # Cap at 256 for lightweight
            
            # Bottleneck
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                             name=f"{name}_bottleneck")(x)
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn")(x)
            
            # Decoder
            for i in range(depth - 1, -1, -1):
                filters = max(base_filters, filters // 2)
                
                if i < depth - 1:
                    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), 
                                             padding='same', name=f"{name}_dec_upsample_{i}")(x)
                
                # Skip connection
                if i < len(skip_connections):
                    x = layers.Concatenate(name=f"{name}_dec_concat_{i}")([x, skip_connections[i]])
                
                # Single conv block
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_dec_conv_{i}")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}")(x)
                x = layers.Dropout(dropout, name=f"{name}_dec_dropout_{i}")(x)
            
            # Output
            output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                  name=f"{name}_output")(x)
            
            model = models.Model(inputs=input_layer, outputs=output, name=name)
            
            self.logger.info(f"Created lightweight CAE: {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating lightweight CAE: {e}")
            raise


class StandardCAE(BaseCAE):
    """Standard CAE with balanced performance."""
    
    def create_model(self, input_shape: Tuple[int, int, int], name: str = "standard_cae") -> tf.keras.Model:
        """Create standard CAE model."""
        try:
            input_layer = layers.Input(shape=input_shape, name=f"{name}_input")
            
            # Standard parameters
            base_filters = self.config.base_filters
            depth = self.config.depth
            dropout = self.config.dropout_rate
            
            # Encoder
            x = input_layer
            skip_connections = []
            filters = base_filters
            
            for i in range(depth):
                # Double conv block
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_1")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_2")(x)
                
                skip_connections.append(x)
                
                if i < depth - 1:
                    x = layers.MaxPooling2D((2, 2), name=f"{name}_enc_pool_{i}")(x)
                    x = layers.Dropout(dropout, name=f"{name}_enc_dropout_{i}")(x)
                
                filters = min(512, filters * 2)
            
            # Bottleneck
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                             name=f"{name}_bottleneck_1")(x)
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_1")(x)
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                             name=f"{name}_bottleneck_2")(x)
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_2")(x)
            x = layers.Dropout(dropout * 1.5, name=f"{name}_bottleneck_dropout")(x)
            
            # Decoder
            for i in range(depth - 1, -1, -1):
                filters = max(base_filters, filters // 2)
                
                if i < depth - 1:
                    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), 
                                             padding='same', name=f"{name}_dec_upsample_{i}")(x)
                
                # Skip connection
                if i < len(skip_connections):
                    x = layers.Concatenate(name=f"{name}_dec_concat_{i}")([x, skip_connections[i]])
                
                # Double conv block
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_dec_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_1")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_dec_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_2")(x)
                x = layers.Dropout(dropout, name=f"{name}_dec_dropout_{i}")(x)
            
            # Output
            output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                  name=f"{name}_output")(x)
            
            model = models.Model(inputs=input_layer, outputs=output, name=name)
            
            self.logger.info(f"Created standard CAE: {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating standard CAE: {e}")
            raise


class RobustCAE(BaseCAE):
    """Robust CAE with enhanced capacity."""
    
    def create_model(self, input_shape: Tuple[int, int, int], name: str = "robust_cae") -> tf.keras.Model:
        """Create robust CAE model."""
        try:
            input_layer = layers.Input(shape=input_shape, name=f"{name}_input")
            
            # Enhanced parameters for robustness
            base_filters = min(64, self.config.base_filters * 2)
            depth = min(6, self.config.depth + 1)
            dropout = min(0.5, self.config.dropout_rate * 1.5)
            
            # Encoder with residual connections
            x = input_layer
            skip_connections = []
            filters = base_filters
            
            for i in range(depth):
                # Residual block
                shortcut = x
                
                # First conv block
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_1")(x)
                
                # Second conv block
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_2")(x)
                
                # Residual connection if same dimensions
                if shortcut.shape[-1] == filters and i > 0:
                    x = layers.Add(name=f"{name}_enc_residual_{i}")([x, shortcut])
                
                skip_connections.append(x)
                
                if i < depth - 1:
                    x = layers.MaxPooling2D((2, 2), name=f"{name}_enc_pool_{i}")(x)
                    x = layers.Dropout(dropout, name=f"{name}_enc_dropout_{i}")(x)
                
                filters = min(1024, filters * 2)  # Higher capacity
            
            # Enhanced bottleneck
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                             name=f"{name}_bottleneck_1")(x)
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_1")(x)
            
            x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same',
                             name=f"{name}_bottleneck_2")(x)  # 1x1 conv for feature mixing
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_2")(x)
            
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                             name=f"{name}_bottleneck_3")(x)
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_3")(x)
            x = layers.Dropout(dropout * 1.5, name=f"{name}_bottleneck_dropout")(x)
            
            # Decoder with attention
            for i in range(depth - 1, -1, -1):
                filters = max(base_filters, filters // 2)
                
                if i < depth - 1:
                    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), 
                                             padding='same', name=f"{name}_dec_upsample_{i}")(x)
                
                # Skip connection with attention
                if i < len(skip_connections):
                    skip = skip_connections[i]
                    
                    # Simple attention mechanism
                    attention = layers.Conv2D(1, (1, 1), activation='sigmoid', 
                                            name=f"{name}_attention_{i}")(skip)
                    attended_skip = layers.Multiply(name=f"{name}_attended_skip_{i}")([skip, attention])
                    
                    x = layers.Concatenate(name=f"{name}_dec_concat_{i}")([x, attended_skip])
                
                # Enhanced conv block
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_dec_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_1")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_dec_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_2")(x)
                
                x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same',
                                 name=f"{name}_dec_conv_{i}_3")(x)  # Additional 1x1 conv
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_3")(x)
                
                x = layers.Dropout(dropout, name=f"{name}_dec_dropout_{i}")(x)
            
            # Output with refinement
            x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                             name=f"{name}_pre_output")(x)
            x = layers.BatchNormalization(name=f"{name}_pre_output_bn")(x)
            
            output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                  name=f"{name}_output")(x)
            
            model = models.Model(inputs=input_layer, outputs=output, name=name)
            
            self.logger.info(f"Created robust CAE: {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating robust CAE: {e}")
            raise


class DeepCAE(BaseCAE):
    """Deep CAE with many layers."""
    
    def create_model(self, input_shape: Tuple[int, int, int], name: str = "deep_cae") -> tf.keras.Model:
        """Create deep CAE model."""
        try:
            input_layer = layers.Input(shape=input_shape, name=f"{name}_input")
            
            # Deep architecture parameters
            base_filters = self.config.base_filters
            depth = min(7, self.config.depth + 2)
            dropout = self.config.dropout_rate
            
            # Encoder
            x = input_layer
            skip_connections = []
            filters = base_filters
            
            for i in range(depth):
                # Triple conv block for deeper features
                x = layers.Conv2D(filters, (3, 3), activation='swish', padding='same',
                                 name=f"{name}_enc_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_1")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='swish', padding='same',
                                 name=f"{name}_enc_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_2")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='swish', padding='same',
                                 name=f"{name}_enc_conv_{i}_3")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_3")(x)
                
                skip_connections.append(x)
                
                if i < depth - 1:
                    x = layers.MaxPooling2D((2, 2), name=f"{name}_enc_pool_{i}")(x)
                    x = layers.Dropout(dropout, name=f"{name}_enc_dropout_{i}")(x)
                
                filters = min(512, filters * 2)
            
            # Deep bottleneck
            for j in range(3):
                x = layers.Conv2D(filters, (3, 3), activation='swish', padding='same',
                                 name=f"{name}_bottleneck_{j}")(x)
                x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_{j}")(x)
                if j < 2:
                    x = layers.Dropout(dropout, name=f"{name}_bottleneck_dropout_{j}")(x)
            
            # Decoder
            for i in range(depth - 1, -1, -1):
                filters = max(base_filters, filters // 2)
                
                if i < depth - 1:
                    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), 
                                             padding='same', name=f"{name}_dec_upsample_{i}")(x)
                
                # Skip connection
                if i < len(skip_connections):
                    x = layers.Concatenate(name=f"{name}_dec_concat_{i}")([x, skip_connections[i]])
                
                # Triple conv block
                x = layers.Conv2D(filters, (3, 3), activation='swish', padding='same',
                                 name=f"{name}_dec_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_1")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='swish', padding='same',
                                 name=f"{name}_dec_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_2")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='swish', padding='same',
                                 name=f"{name}_dec_conv_{i}_3")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_3")(x)
                
                x = layers.Dropout(dropout, name=f"{name}_dec_dropout_{i}")(x)
            
            # Output
            output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                  name=f"{name}_output")(x)
            
            model = models.Model(inputs=input_layer, outputs=output, name=name)
            
            self.logger.info(f"Created deep CAE: {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating deep CAE: {e}")
            raise


class WideCAE(BaseCAE):
    """Wide CAE with more filters per layer."""
    
    def create_model(self, input_shape: Tuple[int, int, int], name: str = "wide_cae") -> tf.keras.Model:
        """Create wide CAE model."""
        try:
            input_layer = layers.Input(shape=input_shape, name=f"{name}_input")
            
            # Wide architecture parameters
            base_filters = min(96, self.config.base_filters * 3)
            depth = self.config.depth
            dropout = self.config.dropout_rate * 0.8  # Less dropout for wider model
            
            # Encoder
            x = input_layer
            skip_connections = []
            filters = base_filters
            
            for i in range(depth):
                # Wide conv blocks
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_1")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_2")(x)
                
                # Additional parallel conv for width
                x_parallel = layers.Conv2D(filters // 2, (1, 1), activation='relu', 
                                          padding='same', name=f"{name}_enc_parallel_{i}")(x)
                x = layers.Concatenate(name=f"{name}_enc_wide_concat_{i}")([x, x_parallel])
                
                skip_connections.append(x)
                
                if i < depth - 1:
                    x = layers.MaxPooling2D((2, 2), name=f"{name}_enc_pool_{i}")(x)
                    x = layers.Dropout(dropout, name=f"{name}_enc_dropout_{i}")(x)
                
                filters = min(768, int(filters * 1.5))  # Gradual increase for wide model
            
            # Wide bottleneck
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                             name=f"{name}_bottleneck_1")(x)
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_1")(x)
            
            # Parallel processing in bottleneck
            x1 = layers.Conv2D(filters // 2, (1, 1), activation='relu', padding='same',
                              name=f"{name}_bottleneck_1x1")(x)
            x3 = layers.Conv2D(filters // 2, (3, 3), activation='relu', padding='same',
                              name=f"{name}_bottleneck_3x3")(x)
            x5 = layers.Conv2D(filters // 4, (5, 5), activation='relu', padding='same',
                              name=f"{name}_bottleneck_5x5")(x)
            
            x = layers.Concatenate(name=f"{name}_bottleneck_concat")([x1, x3, x5])
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_2")(x)
            x = layers.Dropout(dropout * 1.5, name=f"{name}_bottleneck_dropout")(x)
            
            # Decoder
            for i in range(depth - 1, -1, -1):
                filters = max(base_filters, int(filters / 1.5))
                
                if i < depth - 1:
                    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), 
                                             padding='same', name=f"{name}_dec_upsample_{i}")(x)
                
                # Skip connection
                if i < len(skip_connections):
                    x = layers.Concatenate(name=f"{name}_dec_concat_{i}")([x, skip_connections[i]])
                
                # Wide conv blocks
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_dec_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_1")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_dec_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_dec_bn_{i}_2")(x)
                
                # Additional parallel conv
                x_parallel = layers.Conv2D(filters // 2, (1, 1), activation='relu', 
                                          padding='same', name=f"{name}_dec_parallel_{i}")(x)
                x = layers.Concatenate(name=f"{name}_dec_wide_concat_{i}")([x, x_parallel])
                
                x = layers.Dropout(dropout, name=f"{name}_dec_dropout_{i}")(x)
            
            # Output
            output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                  name=f"{name}_output")(x)
            
            model = models.Model(inputs=input_layer, outputs=output, name=name)
            
            self.logger.info(f"Created wide CAE: {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating wide CAE: {e}")
            raise


class UncertaintyCAE(BaseCAE):
    """CAE with uncertainty estimation."""
    
    def create_model(self, input_shape: Tuple[int, int, int], name: str = "uncertainty_cae") -> tf.keras.Model:
        """Create CAE with uncertainty estimation."""
        try:
            input_layer = layers.Input(shape=input_shape, name=f"{name}_input")
            
            # Standard parameters
            base_filters = self.config.base_filters
            depth = self.config.depth
            dropout = self.config.dropout_rate
            
            # Shared encoder
            x = input_layer
            skip_connections = []
            filters = base_filters
            
            for i in range(depth):
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}_1")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_1")(x)
                
                x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                                 name=f"{name}_enc_conv_{i}_2")(x)
                x = layers.BatchNormalization(name=f"{name}_enc_bn_{i}_2")(x)
                
                skip_connections.append(x)
                
                if i < depth - 1:
                    x = layers.MaxPooling2D((2, 2), name=f"{name}_enc_pool_{i}")(x)
                    x = layers.Dropout(dropout, name=f"{name}_enc_dropout_{i}")(x)
                
                filters = min(512, filters * 2)
            
            # Shared bottleneck
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                             name=f"{name}_bottleneck_1")(x)
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_1")(x)
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                             name=f"{name}_bottleneck_2")(x)
            x = layers.BatchNormalization(name=f"{name}_bottleneck_bn_2")(x)
            x = layers.Dropout(dropout * 1.5, name=f"{name}_bottleneck_dropout")(x)
            
            # Split into two decoders
            # Main decoder for depth prediction
            x_main = x
            for i in range(depth - 1, -1, -1):
                filters_main = max(base_filters, filters // 2)
                
                if i < depth - 1:
                    x_main = layers.Conv2DTranspose(filters_main, (2, 2), strides=(2, 2), 
                                                   padding='same', name=f"{name}_main_upsample_{i}")(x_main)
                
                if i < len(skip_connections):
                    x_main = layers.Concatenate(name=f"{name}_main_concat_{i}")([x_main, skip_connections[i]])
                
                x_main = layers.Conv2D(filters_main, (3, 3), activation='relu', padding='same',
                                      name=f"{name}_main_conv_{i}_1")(x_main)
                x_main = layers.BatchNormalization(name=f"{name}_main_bn_{i}_1")(x_main)
                
                x_main = layers.Conv2D(filters_main, (3, 3), activation='relu', padding='same',
                                      name=f"{name}_main_conv_{i}_2")(x_main)
                x_main = layers.BatchNormalization(name=f"{name}_main_bn_{i}_2")(x_main)
                x_main = layers.Dropout(dropout, name=f"{name}_main_dropout_{i}")(x_main)
            
            # Uncertainty decoder
            x_unc = x
            for i in range(depth - 1, -1, -1):
                filters_unc = max(base_filters // 2, filters // 4)  # Smaller for uncertainty
                
                if i < depth - 1:
                    x_unc = layers.Conv2DTranspose(filters_unc, (2, 2), strides=(2, 2), 
                                                  padding='same', name=f"{name}_unc_upsample_{i}")(x_unc)
                
                if i < len(skip_connections):
                    # Use reduced skip connections for uncertainty
                    skip_reduced = layers.Conv2D(filters_unc, (1, 1), activation='relu',
                                               name=f"{name}_unc_skip_reduce_{i}")(skip_connections[i])
                    x_unc = layers.Concatenate(name=f"{name}_unc_concat_{i}")([x_unc, skip_reduced])
                
                x_unc = layers.Conv2D(filters_unc, (3, 3), activation='relu', padding='same',
                                     name=f"{name}_unc_conv_{i}_1")(x_unc)
                x_unc = layers.BatchNormalization(name=f"{name}_unc_bn_{i}_1")(x_unc)
                
                x_unc = layers.Conv2D(filters_unc, (3, 3), activation='relu', padding='same',
                                     name=f"{name}_unc_conv_{i}_2")(x_unc)
                x_unc = layers.BatchNormalization(name=f"{name}_unc_bn_{i}_2")(x_unc)
                x_unc = layers.Dropout(dropout * 1.2, name=f"{name}_unc_dropout_{i}")(x_unc)  # Higher dropout for uncertainty
            
            # Outputs
            depth_output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                        name=f"{name}_depth_output")(x_main)
            
            uncertainty_output = layers.Conv2D(1, (1, 1), activation='softplus', padding='same',
                                              name=f"{name}_uncertainty_output")(x_unc)
            
            model = models.Model(inputs=input_layer, outputs=[depth_output, uncertainty_output], name=name)
            
            self.logger.info(f"Created uncertainty CAE: {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating uncertainty CAE: {e}")
            raise


# Model factory functions
def create_model_by_type(model_type: str, config: Config, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Factory function to create models by type."""
    model_classes = {
        'lightweight': LightweightCAE,
        'standard': StandardCAE,
        'robust': RobustCAE,
        'deep': DeepCAE,
        'wide': WideCAE,
        'uncertainty': UncertaintyCAE
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    model_class = model_classes[model_type]
    model_builder = model_class(config)
    
    return model_builder.create_model(input_shape, name=f"{model_type}_cae")


def get_model_variants(config: Config) -> Dict[str, Dict[str, Any]]:
    """Get available model variants with their configurations."""
    return {
        'lightweight': {
            'description': 'Fast model with reduced parameters',
            'parameters': 'Low',
            'speed': 'Fast',
            'accuracy': 'Good'
        },
        'standard': {
            'description': 'Balanced performance and accuracy',
            'parameters': 'Medium',
            'speed': 'Medium',
            'accuracy': 'Very Good'
        },
        'robust': {
            'description': 'Enhanced model with attention mechanisms',
            'parameters': 'High',
            'speed': 'Slow',
            'accuracy': 'Excellent'
        },
        'deep': {
            'description': 'Deep architecture for complex patterns',
            'parameters': 'High',
            'speed': 'Slow',
            'accuracy': 'Excellent'
        },
        'wide': {
            'description': 'Wide architecture with parallel processing',
            'parameters': 'Very High',
            'speed': 'Very Slow',
            'accuracy': 'Excellent'
        },
        'uncertainty': {
            'description': 'Dual-output model with uncertainty estimation',
            'parameters': 'High',
            'speed': 'Slow',
            'accuracy': 'Very Good + Uncertainty'
        }
    }


def compile_model(model: tf.keras.Model, config: Config, 
                 uncertainty_aware: bool = False) -> tf.keras.Model:
    """Compile model with appropriate loss and optimizer."""
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=1e-4
    )
    
    if uncertainty_aware:
        # Custom loss for uncertainty-aware models
        def uncertainty_loss(y_true, y_pred):
            # Assuming y_pred contains both prediction and uncertainty
            return tf.keras.losses.mse(y_true, y_pred)
        
        model.compile(
            optimizer=optimizer,
            loss={'depth_output': 'mse', 'uncertainty_output': 'mse'},
            loss_weights={'depth_output': 1.0, 'uncertainty_output': 0.1},
            metrics={'depth_output': ['mae'], 'uncertainty_output': ['mae']}
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    return model