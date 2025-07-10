"""
Advanced CAE Model Architecture Module

This module contains the implementation of the Advanced Convolutional Autoencoder
with modern architecture features including residual blocks, attention mechanisms,
and advanced loss functions.

Author: Bathymetric CAE Team
License: MIT
"""

import logging
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model
from typing import Tuple, List, Optional

from ..utils.logging_utils import log_function_call


class AdvancedCAE:
    """
    Advanced Convolutional Autoencoder with modern architecture.
    
    This class implements a sophisticated CAE architecture with:
    - Residual connections for better gradient flow
    - Attention mechanisms for feature enhancement
    - Skip connections in decoder for detail preservation
    - Combined loss function with MSE and SSIM
    - Advanced optimization techniques
    
    Attributes:
        config: Configuration object containing model parameters
        logger: Logger instance for this model
    """
    
    def __init__(self, config):
        """
        Initialize the Advanced CAE model builder.
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info(
            f"Initialized AdvancedCAE with grid_size={config.grid_size}, "
            f"base_filters={config.base_filters}, depth={config.depth}"
        )
    
    def _validate_config(self):
        """Validate model configuration parameters."""
        required_attrs = ['grid_size', 'base_filters', 'depth', 'dropout_rate', 'learning_rate']
        
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValueError(f"Configuration missing required attribute: {attr}")
        
        if self.config.grid_size < 32:
            raise ValueError("grid_size must be at least 32")
        if self.config.base_filters < 8:
            raise ValueError("base_filters must be at least 8")
        if self.config.depth < 1:
            raise ValueError("depth must be at least 1")
    
    @log_function_call
    def create_model(self, channels: int = 2) -> tf.keras.Model:
        """
        Create advanced CAE model with attention mechanism.
        
        Args:
            channels: Number of input channels (1 for depth only, 2 for depth+uncertainty)
            
        Returns:
            tf.keras.Model: Compiled model ready for training
        """
        self.logger.info(f"Creating model with {channels} input channels")
        
        # Input layer
        input_layer = layers.Input(
            shape=(self.config.grid_size, self.config.grid_size, channels),
            name='input_layer'
        )
        
        # Encoder with residual connections
        encoder_outputs = self._build_encoder(input_layer)
        
        # Decoder with skip connections
        decoder_output = self._build_decoder(encoder_outputs)
        
        # Final output layer
        output_layer = layers.Conv2D(
            1, (3, 3), 
            activation='sigmoid', 
            padding='same',
            name='output_layer'
        )(decoder_output)
        
        # Create model
        model = models.Model(input_layer, output_layer, name='AdvancedCAE')
        
        # Compile with advanced optimizer
        self._compile_model(model)
        
        # Log model summary
        self.logger.info(f"Model created with {model.count_params():,} parameters")
        
        return model
    
    def _build_encoder(self, input_layer) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Build encoder with residual blocks.
        
        Args:
            input_layer: Input tensor
            
        Returns:
            Tuple of (encoded_tensor, skip_connections)
        """
        x = input_layer
        skip_connections = []
        
        filters = self.config.base_filters
        
        for i in range(self.config.depth):
            # Residual block
            x = self._residual_block(x, filters, f'encoder_block_{i}')
            skip_connections.append(x)
            
            # Downsample
            x = layers.MaxPooling2D((2, 2), padding='same', name=f'encoder_pool_{i}')(x)
            x = layers.Dropout(self.config.dropout_rate, name=f'encoder_dropout_{i}')(x)
            
            filters *= 2
        
        # Bottleneck with attention
        x = self._attention_block(x, filters, 'bottleneck_attention')
        
        return x, skip_connections
    
    def _build_decoder(self, encoder_outputs: Tuple[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        """
        Build decoder with skip connections.
        
        Args:
            encoder_outputs: Tuple of (encoded_tensor, skip_connections)
            
        Returns:
            tf.Tensor: Decoded tensor
        """
        x, skip_connections = encoder_outputs
        
        filters = self.config.base_filters * (2 ** self.config.depth)
        
        for i in range(self.config.depth):
            filters //= 2
            
            # Upsample
            x = layers.Conv2DTranspose(
                filters, (3, 3), 
                strides=(2, 2), 
                padding='same',
                activation='relu',
                name=f'decoder_upsample_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'decoder_bn_{i}')(x)
            
            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                
                # Ensure compatible shapes
                if x.shape[1:3] != skip.shape[1:3]:
                    skip = layers.Resizing(
                        x.shape[1], x.shape[2], 
                        name=f'skip_resize_{i}'
                    )(skip)
                
                x = layers.Concatenate(name=f'skip_concat_{i}')([x, skip])
            
            # Residual block
            x = self._residual_block(x, filters, f'decoder_block_{i}')
            x = layers.Dropout(self.config.dropout_rate, name=f'decoder_dropout_{i}')(x)
        
        return x
    
    def _residual_block(self, x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        """
        Residual block with batch normalization.
        
        Args:
            x: Input tensor
            filters: Number of filters
            name: Block name prefix
            
        Returns:
            tf.Tensor: Output tensor
        """
        shortcut = x
        
        # First conv
        x = layers.Conv2D(
            filters, (3, 3), 
            padding='same', 
            name=f'{name}_conv1'
        )(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Activation('relu', name=f'{name}_relu1')(x)
        
        # Second conv
        x = layers.Conv2D(
            filters, (3, 3), 
            padding='same', 
            name=f'{name}_conv2'
        )(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        
        # Adjust shortcut if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, (1, 1), 
                padding='same',
                name=f'{name}_shortcut'
            )(shortcut)
        
        # Add shortcut
        x = layers.Add(name=f'{name}_add')([x, shortcut])
        x = layers.Activation('relu', name=f'{name}_relu2')(x)
        
        return x
    
    def _attention_block(self, x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        """
        Spatial attention block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            name: Block name
            
        Returns:
            tf.Tensor: Output tensor with attention applied
        """
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D(name=f'{name}_avg_pool')(x)
        max_pool = layers.GlobalMaxPooling2D(name=f'{name}_max_pool')(x)
        
        avg_pool = layers.Reshape((1, 1, filters), name=f'{name}_avg_reshape')(avg_pool)
        max_pool = layers.Reshape((1, 1, filters), name=f'{name}_max_reshape')(max_pool)
        
        # Shared MLP
        reduction_ratio = 8
        shared_layer_one = layers.Conv2D(
            filters // reduction_ratio, (1, 1), 
            activation='relu',
            name=f'{name}_shared_conv1'
        )
        shared_layer_two = layers.Conv2D(
            filters, (1, 1), 
            activation='sigmoid',
            name=f'{name}_shared_conv2'
        )
        
        avg_out = shared_layer_two(shared_layer_one(avg_pool))
        max_out = shared_layer_two(shared_layer_one(max_pool))
        
        channel_attention = layers.Add(name=f'{name}_channel_add')([avg_out, max_out])
        x = layers.Multiply(name=f'{name}_channel_mult')([x, channel_attention])
        
        return x
    
    def _compile_model(self, model: tf.keras.Model):
        """
        Compile model with advanced optimizer and loss function.
        
        Args:
            model: Model to compile
        """
        # Advanced optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=self._combined_loss,
            metrics=['mae', self._ssim_metric]
        )
        
        self.logger.info("Model compiled with AdamW optimizer and combined loss")
    
    def _combined_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Combined loss function with MSE and SSIM.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            tf.Tensor: Combined loss value
        """
        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        
        # Weighted combination
        return mse_loss + 0.1 * ssim_loss
    
    def _ssim_metric(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        SSIM metric for monitoring.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            tf.Tensor: SSIM value
        """
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    def create_callbacks(self, model_path: str) -> List[callbacks.Callback]:
        """
        Create training callbacks.
        
        Args:
            model_path: Path to save model
            
        Returns:
            List[callbacks.Callback]: List of configured callbacks
        """
        callback_list = [
            callbacks.TensorBoard(
                log_dir=self.config.log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            callbacks.CSVLogger('training_log.csv', append=True)
        ]
        
        # Add learning rate scheduler if configured
        if hasattr(self.config, 'use_cosine_decay') and self.config.use_cosine_decay:
            cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=self.config.epochs
            )
            lr_scheduler = callbacks.LearningRateScheduler(
                lambda epoch: cosine_decay(epoch)
            )
            callback_list.append(lr_scheduler)
        
        return callback_list
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            tf.keras.Model: Loaded model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model cannot be loaded
        """
        try:
            model = load_model(
                model_path, 
                custom_objects={
                    '_combined_loss': self._combined_loss,
                    '_ssim_metric': self._ssim_metric
                },
                compile=False
            )
            
            # Recompile with current settings
            self._compile_model(model)
            
            self.logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def get_model_summary(self, model: tf.keras.Model) -> str:
        """
        Get detailed model summary.
        
        Args:
            model: Model to summarize
            
        Returns:
            str: Model summary string
        """
        import io
        
        summary_buffer = io.StringIO()
        model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        summary = summary_buffer.getvalue()
        summary_buffer.close()
        
        return summary


def create_data_augmentation_layer() -> tf.keras.Sequential:
    """
    Create data augmentation layer for training robustness.
    
    Returns:
        tf.keras.Sequential: Data augmentation layer
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.GaussianNoise(0.01)
    ], name="data_augmentation")


def calculate_model_memory_requirements(config) -> dict:
    """
    Calculate estimated memory requirements for model.
    
    Args:
        config: Configuration object
        
    Returns:
        dict: Memory requirements estimation
    """
    # Rough estimation based on model architecture
    grid_size = config.grid_size
    base_filters = config.base_filters
    depth = config.depth
    
    # Estimate parameter count
    param_count = 0
    filters = base_filters
    
    # Encoder parameters
    for i in range(depth):
        param_count += filters * filters * 9 * 2  # Conv layers in residual block
        param_count += filters * 2  # BatchNorm parameters
        filters *= 2
    
    # Decoder parameters (roughly similar)
    param_count *= 2
    
    # Memory estimation (very rough)
    param_memory_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
    activation_memory_mb = grid_size * grid_size * base_filters * depth * 4 / (1024 * 1024)
    
    return {
        'estimated_parameters': param_count,
        'parameter_memory_mb': param_memory_mb,
        'activation_memory_mb': activation_memory_mb,
        'total_memory_mb': param_memory_mb + activation_memory_mb
    }
