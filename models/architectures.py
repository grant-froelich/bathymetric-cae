"""
Model architecture definitions for Enhanced Bathymetric CAE Processing.
Fixed for channel consistency and Windows compatibility.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Optional
from config.config import Config


class AdvancedCAE:
    """Advanced Convolutional Autoencoder for bathymetric data processing."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_model(self, input_shape: tuple, variant_config: Optional[Dict] = None) -> tf.keras.Model:
        """Create advanced CAE model with specified configuration."""
        if variant_config is None:
            variant_config = {
                'base_filters': self.config.base_filters,
                'depth': self.config.depth,
                'dropout_rate': self.config.dropout_rate
            }
        
        input_layer = layers.Input(shape=input_shape)
        
        # Encoder path
        encoded, skip_connections = self._build_encoder(input_layer, variant_config)
        
        # Decoder path  
        decoded = self._build_decoder((encoded, skip_connections), variant_config, input_shape)
        
        # Create model
        model = models.Model(input_layer, decoded, name=f"AdvancedCAE_v{variant_config.get('version', 1)}")
        
        # Compile model with simple loss to avoid channel issues
        optimizer = tf.keras.optimizers.AdamW(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Use simple MSE loss instead of custom loss
            metrics=['mae']
        )
        
        return model
    
    def _build_encoder(self, input_layer, variant_config: Dict):
        """Build encoder part of the network."""
        x = input_layer
        filters = variant_config['base_filters']
        
        # Progressive downsampling with residual connections
        skip_connections = []
        
        for i in range(variant_config['depth']):
            # Residual block
            residual = x
            
            # First convolution
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            
            # Second convolution
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            
            # Residual connection (adjust channels if needed)
            if residual.shape[-1] != filters:
                residual = layers.Conv2D(filters, (1, 1), padding='same')(residual)
            x = layers.Add()([x, residual])
            
            skip_connections.append(x)
            
            # Downsampling
            x = layers.MaxPooling2D((2, 2), padding='same')(x)
            x = layers.Dropout(variant_config['dropout_rate'])(x)
            
            filters *= 2
        
        # Bottleneck
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(variant_config['dropout_rate'])(x)
        
        return x, skip_connections
    
    def _build_decoder(self, encoded_input, variant_config: Dict, original_shape: tuple):
        """Build decoder part of the network."""
        x, skip_connections = encoded_input
        filters = variant_config['base_filters'] * (2 ** variant_config['depth'])
        
        # Progressive upsampling with skip connections
        for i in range(variant_config['depth']):
            filters //= 2
            
            # Upsampling
            x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2),
                                     activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                # Ensure compatible shapes
                if x.shape[1:3] != skip.shape[1:3]:
                    skip = layers.Resizing(x.shape[1], x.shape[2])(skip)
                x = layers.Concatenate()([x, skip])
            
            # Convolution block
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(variant_config['dropout_rate'])(x)
        
        # Output layer - always output single channel for depth
        output = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)
        
        return output


class UncertaintyCAE(AdvancedCAE):
    """CAE variant that estimates prediction uncertainty."""
    
    def create_model(self, input_shape: tuple, variant_config: Optional[Dict] = None) -> tf.keras.Model:
        """Create uncertainty-aware CAE model."""
        if variant_config is None:
            variant_config = {
                'base_filters': self.config.base_filters,
                'depth': self.config.depth,
                'dropout_rate': self.config.dropout_rate
            }
        
        input_layer = layers.Input(shape=input_shape)
        
        # Shared encoder
        encoded, skip_connections = self._build_encoder(input_layer, variant_config)
        
        # Dual decoder heads
        depth_output = self._build_decoder((encoded, skip_connections), variant_config, input_shape)
        uncertainty_output = self._build_uncertainty_head(encoded, variant_config, input_shape)
        
        # Create model with dual outputs
        model = models.Model(
            input_layer, 
            [depth_output, uncertainty_output],
            name="UncertaintyCAE"
        )
        
        # Compile with proper list format for multiple outputs
        optimizer = tf.keras.optimizers.AdamW(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=['mse', 'mse'],  # Use list for multiple outputs
            loss_weights=[1.0, 0.1],  # Use list for multiple outputs
            metrics=['mae']
        )
        
        return model
    
    def _build_uncertainty_head(self, encoded_input, variant_config: Dict, original_shape: tuple):
        """Build uncertainty estimation head."""
        x = encoded_input
        filters = variant_config['base_filters'] * (2 ** variant_config['depth'])
        
        # Simple upsampling for uncertainty
        for i in range(variant_config['depth']):
            filters //= 2
            x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2),
                                     activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(variant_config['dropout_rate'])(x)
        
        # Uncertainty output (positive values)
        uncertainty = layers.Conv2D(1, (3, 3), activation='softplus', 
                                   padding='same', name='uncertainty_output')(x)
        
        return uncertainty


class LightweightCAE(AdvancedCAE):
    """Lightweight CAE variant for resource-constrained environments."""
    
    def create_model(self, input_shape: tuple, variant_config: Optional[Dict] = None) -> tf.keras.Model:
        """Create lightweight CAE model."""
        if variant_config is None:
            variant_config = {
                'base_filters': max(16, self.config.base_filters // 2),
                'depth': max(2, self.config.depth - 1),
                'dropout_rate': self.config.dropout_rate
            }
        
        input_layer = layers.Input(shape=input_shape)
        
        # Simplified encoder-decoder
        x = input_layer
        filters = variant_config['base_filters']
        
        # Encoder
        for i in range(variant_config['depth']):
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2), padding='same')(x)
            filters *= 2
        
        # Decoder
        for i in range(variant_config['depth']):
            filters //= 2
            x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2),
                                     activation='relu', padding='same')(x)
        
        # Output - single channel
        output = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)
        
        model = models.Model(input_layer, output, name="LightweightCAE")
        
        # Simple compilation
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model


def create_model_variant(config: Config, variant_type: str, input_shape: tuple, 
                        variant_config: Optional[Dict] = None) -> tf.keras.Model:
    """Factory function to create different model variants."""
    
    if variant_type.lower() == 'advanced':
        architecture = AdvancedCAE(config)
    elif variant_type.lower() == 'uncertainty':
        architecture = UncertaintyCAE(config)
    elif variant_type.lower() == 'lightweight':
        architecture = LightweightCAE(config)
    else:
        raise ValueError(f"Unknown variant type: {variant_type}")
    
    return architecture.create_model(input_shape, variant_config)


def get_model_variants_for_ensemble(config: Config, input_shape: tuple) -> list:
    """Get diverse model variants for ensemble."""
    variants = []
    
    # For small ensemble sizes, use lightweight models
    if config.ensemble_size == 1:
        variants.append(create_model_variant(config, 'lightweight', input_shape, {
            'base_filters': config.base_filters,
            'depth': max(2, config.depth - 1),
            'dropout_rate': config.dropout_rate,
            'version': 1
        }))
    else:
        # Advanced variant
        variants.append(create_model_variant(config, 'advanced', input_shape, {
            'base_filters': config.base_filters,
            'depth': config.depth,
            'dropout_rate': config.dropout_rate,
            'version': 1
        }))
        
        # Wider variant
        if len(variants) < config.ensemble_size:
            variants.append(create_model_variant(config, 'advanced', input_shape, {
                'base_filters': int(config.base_filters * 1.5),
                'depth': max(2, config.depth - 1),
                'dropout_rate': config.dropout_rate * 0.8,
                'version': 2
            }))
        
        # Deeper variant
        if len(variants) < config.ensemble_size:
            variants.append(create_model_variant(config, 'advanced', input_shape, {
                'base_filters': max(16, config.base_filters // 2),
                'depth': config.depth + 1,
                'dropout_rate': config.dropout_rate * 1.2,
                'version': 3
            }))
    
    return variants[:config.ensemble_size]