"""
Enhanced Bathymetric CAE Processing Pipeline v2.0
=========================================================================

Complete pipeline implementation with robust BAG generation capabilities.
This version includes the full robust export system that handles:
- Direct BAG creation with comprehensive error checking
- Multiple fallback strategies for reliable file generation
- Format preservation (input format = output format)
- Enhanced error handling and logging
"""

import logging
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from scipy import ndimage
from osgeo import gdal
import json
import gc
import subprocess
import warnings

# Suppress TensorFlow warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# Matplotlib with fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available - training plots will be skipped")

from config.config import Config
from processing.data_processor import BathymetricProcessor
from models.ensemble import BathymetricEnsemble
from core.adaptive_processor import AdaptiveProcessor
from core.quality_metrics import BathymetricQualityMetrics
from utils.visualization import create_enhanced_visualization, plot_training_history
from review.expert_system import ExpertReviewSystem


class DepthScaler:
    """Handles proper scaling and denormalization of depth data."""
    
    def __init__(self):
        self.depth_p1: Optional[float] = None
        self.depth_p99: Optional[float] = None
        self.uncertainty_p1: Optional[float] = None
        self.uncertainty_p99: Optional[float] = None
        self.original_depth_range: Optional[Tuple[float, float]] = None
        self.original_uncertainty_range: Optional[Tuple[float, float]] = None
        
    def fit_and_normalize_depth(self, depth_data: np.ndarray) -> np.ndarray:
        """Fit scaler to depth data and return normalized version."""
        valid_depth = depth_data[np.isfinite(depth_data)]
        if len(valid_depth) == 0:
            raise ValueError("No valid depth data for scaling")
        
        # Use robust percentile-based scaling (p1-p99)
        self.depth_p1 = float(np.percentile(valid_depth, 1))
        self.depth_p99 = float(np.percentile(valid_depth, 99))
        self.original_depth_range = (self.depth_p1, self.depth_p99)
        
        # Avoid division by zero
        range_val = self.depth_p99 - self.depth_p1
        if range_val == 0:
            range_val = 1.0
        
        normalized = (depth_data - self.depth_p1) / range_val
        return np.clip(normalized, 0, 1)
    
    def fit_and_normalize_uncertainty(self, uncertainty_data: np.ndarray) -> np.ndarray:
        """Fit scaler to uncertainty data and return normalized version."""
        valid_uncertainty = uncertainty_data[np.isfinite(uncertainty_data)]
        if len(valid_uncertainty) == 0:
            return uncertainty_data
        
        self.uncertainty_p1 = float(np.percentile(valid_uncertainty, 1))
        self.uncertainty_p99 = float(np.percentile(valid_uncertainty, 99))
        self.original_uncertainty_range = (self.uncertainty_p1, self.uncertainty_p99)
        
        range_val = self.uncertainty_p99 - self.uncertainty_p1
        if range_val == 0:
            range_val = 1.0
        
        normalized = (uncertainty_data - self.uncertainty_p1) / range_val
        return np.clip(normalized, 0, 1)
    
    def denormalize_depth(self, normalized_data: np.ndarray) -> np.ndarray:
        """Convert normalized data back to original depth scale."""
        if self.depth_p1 is None or self.depth_p99 is None:
            raise ValueError("Scaler not fitted")
        
        range_val = self.depth_p99 - self.depth_p1
        if range_val == 0:
            range_val = 1.0
        
        return normalized_data * range_val + self.depth_p1
    
    def get_scaling_metadata(self) -> Dict:
        """Get scaling metadata for reporting."""
        return {
            'depth_range': self.original_depth_range,
            'uncertainty_range': self.original_uncertainty_range,
            'scaling_method': 'robust_percentile_p1_p99'
        }


class EnhancedBathymetricCAEPipeline:
    """Main pipeline with robust BAG export capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tf_version = tf.__version__
        self.global_scaler = DepthScaler()
        self.file_scalers = {}
        
        # Initialize processors
        self.data_processor = BathymetricProcessor(config)
        self.ensemble = BathymetricEnsemble(config)
        self.adaptive_processor = AdaptiveProcessor()
        self.quality_metrics = BathymetricQualityMetrics()
        
        # Expert review system 
        self.expert_review_system = ExpertReviewSystem() if getattr(config, 'enable_expert_review', False) else None
        if self.expert_review_system:
            self.logger.info("Expert review system initialized")
        
        # Configure TensorFlow
        self._configure_gpu()
        
        # Setup logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger.info("=== Enhanced Bathymetric CAE Pipeline Initialized ===")
        self.logger.info(f"TensorFlow version: {self.tf_version}")
        self.logger.info(f"Config: grid_size={self.config.grid_size}, ensemble_size={self.config.ensemble_size}")
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
            except RuntimeError as e:
                self.logger.warning(f"GPU configuration failed: {e}")
        else:
            self.logger.info("No GPUs found, using CPU")
    
    def run(self, input_folder: str, output_folder: str, model_name: str = "enhanced_bathymetric_cae.keras"):
        """Run the complete enhanced pipeline with proper scaling."""
        try:
            self.logger.info("=== STARTING PROCESSING PIPELINE ===")
            
            # Setup paths
            input_path = Path(input_folder)
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find input files
            supported_extensions = ['.bag', '.tif', '.tiff', '.asc']
            file_list = []
            for ext in supported_extensions:
                file_list.extend(input_path.glob(f"*{ext}"))
            
            if not file_list:
                raise ValueError(f"No supported files found in {input_folder}")
            
            self.logger.info(f"Found {len(file_list)} files to process")
            
            # Adapt grid size to input data dimensions
            self._adapt_grid_size_to_input(file_list)
            
            # Train ensemble models
            self.logger.info("Training ensemble models...")
            ensemble_models = self._train_ensemble_fixed(file_list)
            
            # Process files with enhanced approach
            self.logger.info("Processing files with ensemble models...")
            self._process_files_enhanced_fixed(ensemble_models, file_list, str(output_path))
            
            # Save ensemble model
            model_path = output_path / model_name
            self._save_ensemble_model(ensemble_models, str(model_path))
            
            # Generate summary report
            self._generate_processing_summary(file_list, str(output_path))
            
            # Generate expert review report and export JSON files
            if self.expert_review_system:
                try:
                    pending_reviews = self.expert_review_system.get_pending_reviews()
                    self.logger.info(f"Expert review: {len(pending_reviews)} files flagged for review")
                    
                    # Export expert review data to JSON files
                    expert_review_folder = output_path / "expert_reviews"
                    success = self.expert_review_system.export_review_data(str(expert_review_folder))
                    if success:
                        self.logger.info(f"Expert review data exported to {expert_review_folder}/")
                    else:
                        self.logger.warning("Failed to export expert review data")
                        
                except Exception as e:
                    self.logger.warning(f"Expert review report failed: {e}")
            
            self.logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _train_ensemble_fixed(self, file_list: List[Path]) -> List:
        """Train ensemble models with proper data scaling and model saving."""
        self.logger.info("Loading and preprocessing training data...")
        
        # Collect all training data for global scaling
        all_depth_data = []
        all_uncertainty_data = []
        
        for file_path in file_list:
            try:
                # Use original depth data for training scaling
                original_depth = self.data_processor.get_original_depth_data(file_path)
        
                # Resize to match grid size
                if original_depth.shape != (self.config.grid_size, self.config.grid_size):
                    from scipy import ndimage
                    original_depth = ndimage.zoom(
                        original_depth, 
                        (self.config.grid_size / original_depth.shape[0], 
                        self.config.grid_size / original_depth.shape[1]), 
                        order=1
                    )
        
                all_depth_data.append(original_depth)
        
                # Get uncertainty data from processed data if needed
                data, shape, metadata = self.data_processor.preprocess_bathymetric_grid(file_path)
                if data.shape[-1] == 2:  # Has uncertainty
                    uncertainty = data[:, :, 1]
                    all_uncertainty_data.append(uncertainty)
                
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path.name}: {e}")
                continue
        
        if not all_depth_data:
            raise RuntimeError("No valid training data found")
        
        # Create global scaler
        combined_depth = np.concatenate([d.flatten() for d in all_depth_data])
        normalized_depth = self.global_scaler.fit_and_normalize_depth(combined_depth)
        
        if all_uncertainty_data:
            combined_uncertainty = np.concatenate([u.flatten() for u in all_uncertainty_data])
            self.global_scaler.fit_and_normalize_uncertainty(combined_uncertainty)
        
        self.logger.debug(f"Global scaling fitted - Depth range: [{self.global_scaler.original_depth_range[0]:.1f}, {self.global_scaler.original_depth_range[1]:.1f}]m")
        
        # Prepare training data
        training_inputs = []
        training_targets = []
        
        for file_path in file_list:
            try:
                data, shape, metadata = self.data_processor.preprocess_bathymetric_grid(file_path)
                
                # Create individual file scaler using ORIGINAL depth data
                file_scaler = DepthScaler()
                
                # Get original depth data for proper scaler fitting
                original_depth_for_scaler = self.data_processor.get_original_depth_data(file_path)
                if original_depth_for_scaler.shape != (self.config.grid_size, self.config.grid_size):
                    original_depth_for_scaler = ndimage.zoom(
                        original_depth_for_scaler, 
                        (self.config.grid_size / original_depth_for_scaler.shape[0], 
                        self.config.grid_size / original_depth_for_scaler.shape[1]), 
                        order=1
                    )
                
                if data.shape[-1] == 2:  # Multi-channel
                    depth = data[:, :, 0]
                    uncertainty = data[:, :, 1]
                    
                    # Fit scaler on ORIGINAL data, apply to preprocessed data
                    file_scaler.fit_and_normalize_depth(original_depth_for_scaler)  # Fit on original
                    self.logger.debug(f"File: {file_path.name}")
                    self.logger.debug(f"Original depth for scaler range: {np.min(original_depth_for_scaler):.1f} to {np.max(original_depth_for_scaler):.1f}m")
                    self.logger.debug(f"Scaler fitted p1: {file_scaler.depth_p1:.1f}, p99: {file_scaler.depth_p99:.1f}")
                    self.logger.debug(f"Preprocessed depth range: {np.min(depth):.1f} to {np.max(depth):.1f}")
                    depth_norm = (depth - file_scaler.depth_p1) / (file_scaler.depth_p99 - file_scaler.depth_p1)
                    depth_norm = np.clip(depth_norm, 0, 1)
                    uncertainty_norm = file_scaler.fit_and_normalize_uncertainty(uncertainty)
                    
                    input_data = np.stack([depth_norm, uncertainty_norm], axis=-1)
                    target_data = depth_norm  # Target is the depth channel
                else:  # Single channel
                    depth = data[:, :, 0] if len(data.shape) == 3 else data
                    # Fit scaler on ORIGINAL data, apply to preprocessed data
                    file_scaler.fit_and_normalize_depth(original_depth_for_scaler)  # Fit on original
                    depth_norm = (depth - file_scaler.depth_p1) / (file_scaler.depth_p99 - file_scaler.depth_p1)
                    depth_norm = np.clip(depth_norm, 0, 1)
                    
                    input_data = np.expand_dims(depth_norm, axis=-1)
                    target_data = depth_norm
                
                training_inputs.append(input_data)
                training_targets.append(target_data)
                
                # Store the scaler for this file
                self.file_scalers[str(file_path)] = file_scaler
                
            except Exception as e:
                self.logger.warning(f"Failed to prepare training data for {file_path.name}: {e}")
                continue
        
        # Convert to training arrays
        training_inputs = np.array(training_inputs)
        training_targets = np.array(training_targets)
        
        if len(training_targets.shape) == 3:
            training_targets = np.expand_dims(training_targets, axis=-1)
        
        self.logger.info(f"Training data shape: {training_inputs.shape} -> {training_targets.shape}")
        
        # Train ensemble models
        ensemble_models = []
        input_channels = training_inputs.shape[-1]
        
        for i in range(self.config.ensemble_size):
            self.logger.info(f"Training ensemble model {i+1}/{self.config.ensemble_size}")
            
            try:
                # Create model variant
                current_config = {
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate * (0.9 ** i)  # Slight variation
                }
                
                model_instance = self.ensemble.create_ensemble(input_channels)[i % len(self.ensemble.create_ensemble(input_channels))]
                
                # Setup callbacks
                model_save_path = Path(f"temp_model_{i}.keras")
                callbacks_list = [
                    tf.keras.callbacks.EarlyStopping(
                        patience=15,
                        restore_best_weights=True,
                        monitor='val_loss' if self.config.validation_split > 0 else 'loss'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        factor=0.5, 
                        patience=5,
                        monitor='val_loss' if self.config.validation_split > 0 else 'loss'
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        str(model_save_path),
                        save_best_only=True,
                        monitor='val_loss' if self.config.validation_split > 0 else 'loss',
                        mode='min',
                        verbose=1,
                        save_weights_only=False
                    )
                ]
                
                # Train model
                self.logger.info(f"Training model {i+1} with {current_config['epochs']} epochs")
                
                history = model_instance.fit(
                    training_inputs, training_targets,
                    epochs=current_config['epochs'],
                    batch_size=current_config['batch_size'],
                    validation_split=self.config.validation_split,
                    callbacks=callbacks_list,
                    verbose=1
                )
                
                # Verify model was saved and load it
                if model_save_path.exists():
                    self.logger.info(f"Model {i+1} saved successfully to {model_save_path}")
                    
                    try:
                        saved_model = tf.keras.models.load_model(str(model_save_path))
                        ensemble_models.append(saved_model)
                        self.logger.info(f"Model {i+1} loaded and verified successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to load saved model {i+1}: {e}")
                        ensemble_models.append(model_instance)
                else:
                    self.logger.warning(f"Model {i+1} was not saved to disk, using in-memory version")
                    ensemble_models.append(model_instance)
                
                # Save training history plot if available
                if hasattr(history, 'history') and history.history and MATPLOTLIB_AVAILABLE:
                    try:
                        self.logger.debug(f"Attempting to create training history plot for model {i+1}")
                        plt.figure(figsize=(12, 4))
                        
                        plt.subplot(1, 2, 1)
                        plt.plot(history.history['loss'], label='Training Loss')
                        if 'val_loss' in history.history:
                            plt.plot(history.history['val_loss'], label='Validation Loss')
                            # Add learning rate change indicators
                            if 'lr' in history.history:
                                lr_changes = []
                                for epoch, lr in enumerate(history.history['lr']):
                                    if epoch > 0 and lr != history.history['lr'][epoch-1]:
                                        lr_changes.append(epoch)
                            
                                for i, change_epoch in enumerate(lr_changes):
                                    label = 'LR Reduced' if i == 0 else None  # Only label first occurrence
                                    plt.axvline(x=change_epoch, color='red', linestyle='--', alpha=0.7, 
                                            linewidth=1, label=label)
                        plt.title(f'Model {i+1} Training Loss')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend()
                        
                        plt.subplot(1, 2, 2)
                        plt.plot(history.history['mae'], label='Training MAE')
                        if 'val_mae' in history.history:
                            plt.plot(history.history['val_mae'], label='Validation MAE')
                        # Add learning rate change indicators
                        if 'lr' in history.history:
                            lr_changes = []
                            for epoch, lr in enumerate(history.history['lr']):
                                if epoch > 0 and lr != history.history['lr'][epoch-1]:
                                    lr_changes.append(epoch)
                            
                            for i, change_epoch in enumerate(lr_changes):
                                label = 'LR Reduced' if i == 0 else None  # Only label first occurrence
                                plt.axvline(x=change_epoch, color='red', linestyle='--', alpha=0.7, 
                                    linewidth=1, label=label)
                        plt.title(f'Model {i+1} Training MAE')
                        plt.xlabel('Epoch')
                        plt.ylabel('MAE')
                        plt.legend()
                        
                        plot_dir = Path("plots")
                        plot_dir.mkdir(exist_ok=True)
                        plt.savefig(plot_dir / f"training_history_ensemble_{i}.png", 
                                  dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        self.logger.info(f"Training history plot saved for model {i+1}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save training history plot for model {i+1}: {e}")
                
                self.logger.info(f"Model {i+1} training completed successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to train model {i+1}: {e}")
                # Create a fallback model
                try:
                    fallback_model = self._create_simple_model(training_inputs.shape[-1])
                    ensemble_models.append(fallback_model)
                    self.logger.warning(f"Added untrained fallback model for ensemble {i+1}")
                except Exception as fallback_error:
                    self.logger.error(f"Failed to create fallback model {i+1}: {fallback_error}")
                    continue
        
        if not ensemble_models:
            raise RuntimeError("No models were created successfully")
        
        self.logger.info(f"Ensemble training completed with {len(ensemble_models)} models")
        return ensemble_models
 
    def _robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """Non-destructive normalization that preserves bathymetric features."""
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return np.full_like(data, 0.5)
    
        # Use full min-max range to preserve all bathymetric features
        data_min, data_max = np.min(valid_data), np.max(valid_data)
    
        if data_max == data_min:
            return np.full_like(data, 0.5)
    
        # Normalize without clipping - preserves peaks and valleys
        normalized = (data - data_min) / (data_max - data_min)
    
        # Only handle invalid values
        normalized = np.where(np.isfinite(normalized), normalized, 0.5)
    
        return normalized

 
    def _create_simple_model(self, input_channels: int):
        """Create a simple model that properly handles the input channels."""
        inputs = tf.keras.layers.Input(shape=(self.config.grid_size, self.config.grid_size, input_channels))
        
        # Simple autoencoder architecture
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        outputs = tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _clean_invalid_data(self, data: np.ndarray) -> np.ndarray:
        """Clean invalid data (NaN, inf) by replacing with valid values."""
        invalid_mask = ~np.isfinite(data)
        if np.any(invalid_mask):
            valid_data = data[~invalid_mask]
            if len(valid_data) > 0:
                fill_value = np.mean(valid_data)
                data[invalid_mask] = fill_value
            else:
                data.fill(0)
        return data
        
    def _adapt_grid_size_to_input(self, file_list: List[Path]):
        """Adapt grid size to match input data dimensions with memory and performance considerations.
        Only runs if adaptive processing is enabled via --enable-adaptive flag.
        """
        # Check if adaptive processing is enabled
        if not getattr(self.config, 'enable_adaptive_processing', False):
            self.logger.info(f"Adaptive processing disabled. Using fixed grid_size={self.config.grid_size}")
            return
            
        if not file_list:
            return
            
        self.logger.info("Adaptive processing enabled. Analyzing input data for optimal grid size...")
            
        # Get the first file's dimensions to set adaptive grid size
        first_file = file_list[0]
        try:
            original_depth_data = self.data_processor.get_original_depth_data(first_file)
            input_height, input_width = original_depth_data.shape
            total_nodes = input_height * input_width
            
            self.logger.info(f"Input data dimensions: {input_height}x{input_width} ({total_nodes:,} nodes)")
            
            # Handle very large datasets (>5M nodes)
            if total_nodes > 5_000_000:
                self.logger.warning(f"Large dataset detected ({total_nodes:,} nodes). Implementing memory-safe processing.")
                
                # For very large datasets, use tiling strategy
                if total_nodes > 25_000_000:  # >25M nodes
                    adapted_size = 2048
                    self.config.batch_size = 1  # Force single batch
                    self.config.ensemble_size = min(self.config.ensemble_size, 3)  # Limit ensemble
                    self.logger.warning(f"Massive dataset: Using tiled processing with {adapted_size}x{adapted_size} tiles")
                    
                elif total_nodes > 10_000_000:  # 10-25M nodes
                    adapted_size = 2048
                    self.config.batch_size = min(self.config.batch_size, 2)
                    self.logger.warning(f"Very large dataset: Limited to {adapted_size}x{adapted_size} with batch_size={self.config.batch_size}")
                    
                else:  # 5-10M nodes
                    adapted_size = 1024
                    self.config.batch_size = min(self.config.batch_size, 4)
                    self.logger.info(f"Large dataset: Using {adapted_size}x{adapted_size} with batch_size={self.config.batch_size}")
                    
                # Enable memory optimizations for large datasets
                self.config.use_mixed_precision = True
                self.config.gpu_memory_growth = True
                
            else:
                # Normal adaptive sizing for smaller datasets
                max_dim = max(input_height, input_width)
                valid_sizes = [128, 256, 512, 1024, 2048]
                adapted_size = min(valid_sizes, key=lambda x: abs(x - max_dim))
            
            # Memory safety check
            estimated_memory_gb = self._estimate_memory_usage(adapted_size, self.config.batch_size, self.config.ensemble_size)
            available_memory_gb = self._get_available_memory_gb()
            
            if estimated_memory_gb > available_memory_gb * 0.8:  # Use max 80% of available memory
                self.logger.warning(f"Estimated memory usage ({estimated_memory_gb:.1f}GB) exceeds 80% of available memory ({available_memory_gb:.1f}GB)")
                
                # Auto-adjust parameters to fit memory
                while estimated_memory_gb > available_memory_gb * 0.8 and adapted_size > 128:
                    if adapted_size > 512:
                        adapted_size = adapted_size // 2
                    else:
                        self.config.batch_size = max(1, self.config.batch_size // 2)
                    
                    estimated_memory_gb = self._estimate_memory_usage(adapted_size, self.config.batch_size, self.config.ensemble_size)
                
                self.logger.info(f"Adjusted to grid_size={adapted_size}, batch_size={self.config.batch_size} for memory safety")
            
            # Update config if different from current
            if adapted_size != self.config.grid_size:
                self.logger.info(f"Adapting grid size from {self.config.grid_size} to {adapted_size} based on input dimensions ({input_height}x{input_width})")
                self.config.grid_size = adapted_size
            else:
                self.logger.info(f"Grid size {self.config.grid_size} is already optimal for input dimensions")
            
            self.logger.info(f"Final adaptive configuration: grid_size={self.config.grid_size}, batch_size={self.config.batch_size}, ensemble_size={self.config.ensemble_size}")
            self.logger.info(f"Estimated memory usage: {estimated_memory_gb:.1f}GB")
                
        except Exception as e:
            self.logger.warning(f"Adaptive grid sizing failed: {e}, using configured grid_size={self.config.grid_size}")
    
    def _estimate_memory_usage(self, grid_size: int, batch_size: int, ensemble_size: int) -> float:
        """Estimate memory usage in GB for given parameters."""
        # Rough estimation based on typical model sizes and data
        bytes_per_float32 = 4
        
        # Input data memory (batch_size * grid_size^2 * channels * float32)
        input_memory = batch_size * grid_size * grid_size * 2 * bytes_per_float32
        
        # Model memory (approximate based on typical CNN architectures)
        model_params = grid_size * grid_size * 32 * 0.1  # Rough estimate
        model_memory = model_params * bytes_per_float32 * ensemble_size
        
        # Working memory for training (gradients, activations, etc.)
        working_memory = input_memory * 4  # Rough multiplier for training overhead
        
        total_bytes = input_memory + model_memory + working_memory
        return total_bytes / (1024**3)  # Convert to GB
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.available / (1024**3)
        except ImportError:
            self.logger.warning("psutil not available, estimating 8GB available memory")
            return 8.0
        except Exception:
            self.logger.warning("Could not determine available memory, estimating 8GB")
            return 8.0
    
    def _process_files_enhanced_fixed(self, ensemble_models: List, file_list: List[Path], output_folder: str):
        """Process files with proper denormalization before saving."""
        output_path = Path(output_folder)
        
        for file_path in file_list:
            try:
                self.logger.info(f"Processing {file_path.name}...")
                
                # Get the scaler that was created during training
                # This scaler is already fitted with the correct original depth range
                file_scaler = self.file_scalers.get(str(file_path))
                if file_scaler is None:
                    self.logger.warning(f"No scaler found for {file_path.name}, using global scaler")
                    file_scaler = self.global_scaler
            
                # Load original depth data for logging
                original_depth_data = self.data_processor.get_original_depth_data(file_path)
                if original_depth_data.shape != (self.config.grid_size, self.config.grid_size):
                    original_depth_data = ndimage.zoom(
                        original_depth_data, 
                        (self.config.grid_size / original_depth_data.shape[0], 
                        self.config.grid_size / original_depth_data.shape[1]), 
                        order=1
                    )
                
                # Log the scaler's ACTUAL depth range for verification
                valid_depths = original_depth_data[np.isfinite(original_depth_data)]
                actual_min, actual_max = np.min(valid_depths), np.max(valid_depths)
                self.logger.debug(f"Using scaler with range: {file_scaler.depth_p1:.1f} to {file_scaler.depth_p99:.1f}m (actual file range: {actual_min:.1f} to {actual_max:.1f}m)")
                
                # Process with ensemble
                enhanced_data, quality_metrics, adaptive_params = self._process_single_file_fixed(
                    file_path, ensemble_models, file_scaler
                )
                
                # Save enhanced data with proper scaling
                output_file_path = output_path / f"enhanced_{file_path.name}"
                self._save_enhanced_data_fixed(
                    enhanced_data, file_path, output_file_path, 
                    quality_metrics, adaptive_params, file_scaler
                )
                
                self.logger.info(f"Completed processing {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                continue
    
    def _process_single_file_fixed(self, file_path: Path, ensemble_models: List, scaler: DepthScaler) -> Tuple[np.ndarray, Dict, Dict]:
        """Process a single file with ensemble models - ADAPTIVE TO AVAILABLE CHANNELS."""
        # Load original depth data without any preprocessing
        original_depth_data = self.data_processor.get_original_depth_data(file_path)
        if original_depth_data.shape != (self.config.grid_size, self.config.grid_size):
            from scipy import ndimage
            original_depth_data = ndimage.zoom(
                original_depth_data, 
                (self.config.grid_size / original_depth_data.shape[0], 
                self.config.grid_size / original_depth_data.shape[1]), 
                order=1
            )
        
        
        # Load and preprocess data
        data, shape, metadata = self.data_processor.preprocess_bathymetric_grid(file_path)
        
        # Get adaptive parameters
        adaptive_params = self.adaptive_processor.get_processing_parameters(original_depth_data)
        
        # Store original data for comparison
        if data.shape[-1] == 2:  # Multi-channel
            original_depth = data[:, :, 0].copy()
            original_uncertainty = data[:, :, 1].copy()
            has_uncertainty = True
        else:  # Single channel
            original_depth = (data[:, :, 0] if len(data.shape) == 3 else data).copy()
            original_uncertainty = None
            has_uncertainty = False
            
        # Use the scaler that was already fitted during training
        # Don't refit - just use the existing scaler to normalize
        if has_uncertainty:
            depth_norm = (data[:, :, 0] - scaler.depth_p1) / (scaler.depth_p99 - scaler.depth_p1)
            depth_norm = np.clip(depth_norm, 0, 1)
            # Don't refit uncertainty scaler - use existing values if available
            if hasattr(scaler, 'uncertainty_p1') and scaler.uncertainty_p1 is not None:
                uncertainty_norm = (data[:, :, 1] - scaler.uncertainty_p1) / (scaler.uncertainty_p99 - scaler.uncertainty_p1)
                uncertainty_norm = np.clip(uncertainty_norm, 0, 1)
            else:
                uncertainty_norm = scaler.fit_and_normalize_uncertainty(data[:, :, 1])
            input_data = np.stack([depth_norm, uncertainty_norm], axis=-1)
        else:
            depth_norm = (original_depth - scaler.depth_p1) / (scaler.depth_p99 - scaler.depth_p1)
            depth_norm = np.clip(depth_norm, 0, 1)
            input_data = np.expand_dims(depth_norm, axis=-1)
        
        # Clean input data
        input_data = self._clean_invalid_data(input_data)
        
        # Prepare for prediction
        input_batch = np.expand_dims(input_data, axis=0)
        
        # Ensemble prediction
        predictions = []
        for i, model in enumerate(ensemble_models):
            try:
                prediction = model.predict(input_batch, verbose=0)
                predictions.append(prediction[0])
            except Exception as e:
                self.logger.warning(f"Model {i+1} prediction failed: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("All ensemble models failed to predict")
        
        # Average ensemble predictions
        enhanced_normalized = np.mean(predictions, axis=0)
        enhanced_normalized = enhanced_normalized[:, :, 0]  # Remove channel dimension
        enhanced_normalized = np.clip(enhanced_normalized, 0, 1)  # Ensure valid normalized range
        
        self.logger.debug(f"Enhanced normalized range: {np.min(enhanced_normalized):.3f} to {np.max(enhanced_normalized):.3f}")
        self.logger.debug(f"Scaler p1: {scaler.depth_p1:.1f}, p99: {scaler.depth_p99:.1f}")
        
        # Denormalize to original scale
        enhanced_depth_real = scaler.denormalize_depth(enhanced_normalized)
        
        # Calculate quality metrics
        try:
            ssim = self.quality_metrics.calculate_ssim_safe(original_depth_data, enhanced_depth_real)
            roughness = self.quality_metrics.calculate_roughness(enhanced_depth_real)
            feature_preservation = self.quality_metrics.calculate_feature_preservation(original_depth_data, enhanced_depth_real)
            consistency = self.quality_metrics.calculate_depth_consistency(enhanced_depth_real)
    
            quality_metrics = {
                'ssim': float(ssim),
                'roughness': float(roughness),
                'feature_preservation': float(feature_preservation),
                'consistency': float(consistency),
                'mae': float(np.mean(np.abs(original_depth_data - enhanced_depth_real))),
                'rmse': float(np.sqrt(np.mean((original_depth_data - enhanced_depth_real)**2)))
            }
            
            # Calculate composite score manually
            individual_scores = [
                quality_metrics['ssim'],
                1.0 - quality_metrics['roughness'],  # Lower roughness is better
                quality_metrics['feature_preservation'],
                quality_metrics['consistency']
            ]
            quality_metrics['composite_quality'] = float(np.mean(individual_scores))
        
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            quality_metrics = {
                'ssim': 0.0, 'roughness': 1.0, 'feature_preservation': 0.0,
                'consistency': 0.0, 'composite_quality': 0.0, 'mae': 0.0, 'rmse': 0.0
            }
        
        # Flag for expert review if enabled  # ADD THIS SECTION
        if self.expert_review_system and quality_metrics.get('composite_quality', 1.0) < getattr(self.config, 'quality_threshold', 0.7):
            try:
                self.expert_review_system.flag_for_review(
                    filename=file_path.name,
                    region=(0, 0, self.config.grid_size, self.config.grid_size),
                    flag_type=f"low_quality_{quality_metrics['composite_quality']:.3f}",
                    confidence=1.0 - quality_metrics['composite_quality']
                )
                self.logger.info(f"Flagged {file_path.name} for expert review")
            except Exception as e:
                self.logger.warning(f"Expert review flagging failed: {e}")
        
        # Create visualization with uncertainty if available
        uncertainty_for_viz = None
        if has_uncertainty and original_uncertainty is not None:
            uncertainty_for_viz = original_uncertainty
        
        try:
            self.logger.info(f"Attempting to create enhanced visualization for {file_path.name}")
            create_enhanced_visualization(
                original_depth_data, enhanced_depth_real, uncertainty_for_viz,
                quality_metrics, file_path, adaptive_params
            )
        except Exception as e:
            self.logger.warning(f"Failed to create visualization for {file_path.name}: {e}")
            import traceback
            self.logger.warning(f"Visualization error traceback: {traceback.format_exc()}")
        
        self.logger.info(f"Quality metrics - SSIM: {quality_metrics.get('ssim', 0):.4f}, "
                        f"Composite: {quality_metrics.get('composite_quality', 0):.4f}")
        
        return enhanced_depth_real, quality_metrics, adaptive_params
    
    def _save_enhanced_data_fixed(self, data: np.ndarray, input_file_path: Path, 
                                 output_path: Path, quality_metrics: Dict, 
                                 adaptive_params: Dict, scaler: DepthScaler):
        """Save enhanced data with robust BAG export handling - FIXED VERSION."""
        try:
            # Read original file metadata
            dataset = gdal.Open(str(input_file_path))
            if dataset is None:
                raise RuntimeError(f"Cannot read original file: {input_file_path}")
            
            geotransform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()
            original_metadata = dataset.GetMetadata()
            original_shape = (dataset.RasterYSize, dataset.RasterXSize)
            
            # Check if original file has uncertainty (BAG files)
            has_uncertainty = dataset.RasterCount >= 2
            uncertainty_data = None
            original_depth_description = "depth"  # Default fallback
            original_uncertainty_description = "uncertainty"  # Default fallback
            # Read original band descriptions
            try:
                original_depth_band = dataset.GetRasterBand(1)
                if original_depth_band:
                    desc = original_depth_band.GetDescription()
                    if desc:
                        original_depth_description = desc
                    self.logger.debug(f"Original depth band description: '{original_depth_description}'")
            except Exception as e:
                self.logger.warning(f"Could not read original depth band description: {e}")
            
            if has_uncertainty:
                uncertainty_band = dataset.GetRasterBand(2)
                uncertainty_data = uncertainty_band.ReadAsArray()
                # Read uncertainty band description
                try:
                    desc = uncertainty_band.GetDescription()
                    if desc:
                        original_uncertainty_description = desc
                    self.logger.debug(f"Original uncertainty band description: '{original_uncertainty_description}'")
                except Exception as e:
                    self.logger.warning(f"Could not read original uncertainty band description: {e}")
            
            dataset = None  # Close immediately after reading metadata
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format and use robust creation
            input_extension = input_file_path.suffix.lower()
            
            if input_extension == '.bag':
                success = self._create_robust_bag(
                    data, output_path, geotransform, projection, 
                    original_metadata, uncertainty_data, quality_metrics,
                    original_depth_description, original_uncertainty_description
                )
            elif input_extension in ['.tif', '.tiff']:
                success = self._create_robust_geotiff(
                    data, output_path, geotransform, projection, 
                    original_metadata, quality_metrics
                )
            elif input_extension == '.asc':
                success = self._create_robust_ascii(
                    data, output_path, geotransform, 
                    original_metadata, quality_metrics
                )
            else:
                # Default to GeoTIFF for unknown formats
                self.logger.warning(f"Unknown format {input_extension}, defaulting to GeoTIFF")
                output_path = output_path.with_suffix('.tif')
                success = self._create_robust_geotiff(
                    data, output_path, geotransform, projection, 
                    original_metadata, quality_metrics
                )
            
            if success:
                self.logger.info(f"Successfully saved enhanced data: {output_path}")
            else:
                self.logger.error(f"Failed to save enhanced data: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving enhanced data: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _create_robust_bag(self, enhanced_data: np.ndarray, output_path: Path,
                          geotransform: tuple, projection: str, metadata: dict,
                          uncertainty_data: Optional[np.ndarray] = None,
                          quality_metrics: Optional[Dict] = None,
                          depth_description: str = "depth",
                          uncertainty_description: str = "uncertainty") -> bool:
        """Create robust BAG file with multiple fallback strategies."""
        
        # Strategy 1: Direct BAG creation with robust error handling
        if self._try_direct_bag_creation(enhanced_data, output_path, geotransform, 
                                       projection, metadata, uncertainty_data,
                                       depth_description, uncertainty_description):
            return True
        
        # Strategy 2: Create GeoTIFF first, then convert to BAG
        temp_tiff = output_path.with_suffix('.temp.tif')
        try:
            if self._create_robust_geotiff(enhanced_data, temp_tiff, geotransform, 
                                         projection, metadata, quality_metrics):
                if self._convert_tiff_to_bag(temp_tiff, output_path, metadata):
                    temp_tiff.unlink(missing_ok=True)
                    return True
            temp_tiff.unlink(missing_ok=True)
        except Exception as e:
            self.logger.warning(f"GeoTIFF→BAG conversion failed: {e}")
            temp_tiff.unlink(missing_ok=True)
        
        # Strategy 3: Fallback to GeoTIFF with BAG extension warning
        self.logger.warning("BAG creation failed, creating GeoTIFF instead")
        fallback_path = output_path.with_suffix('.tif')
        return self._create_robust_geotiff(enhanced_data, fallback_path, geotransform, 
                                         projection, metadata, quality_metrics)

    def _try_direct_bag_creation(self, enhanced_data: np.ndarray, output_path: Path,
                               geotransform: tuple, projection: str, metadata: dict,
                               uncertainty_data: Optional[np.ndarray] = None,
                               depth_description: str = "depth",
                               uncertainty_description: str = "uncertainty") -> bool:
        """Attempt direct BAG creation with robust error handling."""
        
        # Initialize variables to avoid UnboundLocalError
        elevation_band = None
        uncertainty_band = None
        dataset = None
        
        try:
            self.logger.info(f"Starting direct BAG creation for {output_path}")
            height, width = enhanced_data.shape
            self.logger.debug(f"Data shape: {height}x{width}")
            nodata_value = -9999.0
            
            # Create BAG driver
            driver = gdal.GetDriverByName('BAG')
            if driver is None:
                self.logger.warning("BAG driver not available")
                return False
            
            # Create dataset - BAG requires 2 bands minimum
            bands = 2 
            self.logger.info(f"Creating BAG dataset: {output_path}, {width}x{height}, {bands} bands")
            
            dataset = driver.Create(
                str(output_path),
                width, height, bands,
                gdal.GDT_Float32
            )
            
            if dataset is None:
                self.logger.warning("BAG dataset creation returned None")
                return False
                
            self.logger.info(f"BAG dataset created successfully")
            
            # Set geospatial information
            self.logger.debug(f"Setting geotransform: {geotransform}")
            dataset.SetGeoTransform(geotransform)
            self.logger.debug(f"Setting projection: {projection[:50]}...")
            dataset.SetProjection(projection)
            
            # Write elevation data (band 1) with robust handling
            self.logger.debug(f"Getting elevation band from dataset")
            elevation_band = dataset.GetRasterBand(1)
            self.logger.debug(f"Elevation band result: {elevation_band}")
            if elevation_band is None:
                self.logger.warning("Failed to get elevation band from BAG dataset")
                return False
            
            self.logger.debug(f"Got elevation band successfully")
                
            # Prepare elevation data
            elevation_output = enhanced_data.astype(np.float32)
            elevation_invalid = ~np.isfinite(elevation_output)
            if np.any(elevation_invalid):
                elevation_output[elevation_invalid] = nodata_value
                
            elevation_band.SetNoDataValue(nodata_value)
            elevation_band.SetDescription(depth_description)
            
            # Write elevation data with proper error checking
            self.logger.debug(f"Enhanced data shape: {enhanced_data.shape}, elevation_output shape: {elevation_output.shape}")
            write_result = elevation_band.WriteArray(elevation_output)
            if write_result != 0:
                self.logger.warning(f"WriteArray returned error code: {write_result}")
            
            # Force flush after writing elevation
            elevation_band.FlushCache()
            
            # Write uncertainty data if available (band 2)
            if uncertainty_data is not None:
                # Ensure uncertainty data matches enhanced data shape
                if uncertainty_data.shape != enhanced_data.shape:
                    from scipy import ndimage
                    uncertainty_data = ndimage.zoom(
                        uncertainty_data,
                        (enhanced_data.shape[0] / uncertainty_data.shape[0],
                         enhanced_data.shape[1] / uncertainty_data.shape[1]),
                        order=1
                    )
                uncertainty_output = uncertainty_data.astype(np.float32)
                uncertainty_invalid = ~np.isfinite(uncertainty_output)
                if np.any(uncertainty_invalid):
                    uncertainty_output[uncertainty_invalid] = nodata_value
            else:
                # Create dummy uncertainty data if none provided
                uncertainty_output = np.full_like(elevation_output, 0.5)

            uncertainty_band = dataset.GetRasterBand(2)
            if uncertainty_band is None:
                self.logger.warning("Failed to get uncertainty band from BAG dataset")
                return False
                
            uncertainty_band.SetNoDataValue(nodata_value)
            uncertainty_band.SetDescription(uncertainty_description)
            self.logger.debug(f"Uncertainty output shape: {uncertainty_output.shape}")
            uncertainty_band.WriteArray(uncertainty_output)
            uncertainty_band.FlushCache()
            
            # Set comprehensive metadata
            processing_metadata = {
                'PROCESSING_DATE': datetime.datetime.now().isoformat(),
                'PROCESSING_SOFTWARE': 'Enhanced Bathymetric CAE v2.0',
                'ENHANCEMENT_METHOD': 'Convolutional Autoencoder',
                'ROBUST_BAG_CREATION': 'Direct creation with error handling'
            }
            
            # Set BAG-specific metadata
            processing_metadata.update({
                'BAG_CREATION_DATE': datetime.datetime.now().isoformat() + 'Z',
                'BAG_VERSION': '1.6.2',
                'BAG_DATUM': 'WGS84',
                'BAG_COORDINATE_SYSTEM': projection
            })
            
            for key, value in processing_metadata.items():
                dataset.SetMetadataItem(key, str(value), 'PROCESSING')
            
            # Force final flush and close with proper cleanup
            if uncertainty_band is not None:
                uncertainty_band.FlushCache()
                uncertainty_band = None
            if elevation_band is not None:
                elevation_band.FlushCache()
                elevation_band = None
            if dataset is not None:
                dataset.FlushCache()
                dataset = None
            
            # Verify the file was created properly
            verify_dataset = gdal.Open(str(output_path), gdal.GA_ReadOnly)
            if verify_dataset is None:
                self.logger.warning("Direct BAG creation verification failed")
                return False
            
            self.logger.info("✅ Direct BAG creation successful!")
            verify_dataset = None
            return True
            
        except Exception as e:
            self.logger.warning(f"Direct BAG creation failed: {e}")
            import traceback
            self.logger.warning(f"Full traceback: {traceback.format_exc()}")
            return False
            
        except Exception as e:
            self.logger.warning(f"Direct BAG creation failed: {e}")
            return False

    def _convert_tiff_to_bag(self, tiff_path: Path, bag_path: Path, metadata: Dict) -> bool:
        """Convert GeoTIFF to BAG using external tools."""
        
        try:
            cmd = [
                'gdal_translate',
                '-of', 'BAG',
                '-scale', '1', '1', '1', '1',
                '-co', 'VAR_TITLE=Enhanced Bathymetric Data',
                '-co', 'VAR_ABSTRACT=Enhanced bathymetric surface',
                '-co', f'VAR_DATETIME={datetime.datetime.now().isoformat()}Z',
                '-co', 'VAR_VERTICAL_DATUM=MLLW',
                str(tiff_path),
                str(bag_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and bag_path.exists():
                # Verify the converted file
                test_ds = gdal.Open(str(bag_path))
                if test_ds is not None:
                    self.logger.info("✅ GeoTIFF → BAG conversion successful!")
                    test_ds = None
                    return True
            
            self.logger.warning(f"gdal_translate failed: {result.stderr}")
            return False
            
        except Exception as e:
            self.logger.warning(f"TIFF to BAG conversion error: {e}")
            return False

    def _create_robust_geotiff(self, enhanced_data: np.ndarray, output_path: Path,
                              geotransform: tuple, projection: str, metadata: dict,
                              quality_metrics: Optional[Dict] = None) -> bool:
        """Create robust GeoTIFF file."""
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            height, width = enhanced_data.shape
            nodata_value = -9999.0
            creation_options = [
                'COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'
            ]
            
            # Prepare data
            output_data = enhanced_data.astype(np.float32)
            invalid_mask = ~np.isfinite(output_data)
            if np.any(invalid_mask):
                output_data[invalid_mask] = nodata_value
            
            # Create GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(
                str(output_path),
                width, height, 1,
                gdal.GDT_Float32,
                creation_options
            )
            
            if dataset is None:
                self.logger.error("Failed to create GeoTIFF")
                return False
            
            # Set geospatial information
            dataset.SetGeoTransform(geotransform)
            dataset.SetProjection(projection)
            
            # Write data with robust handling
            band = dataset.GetRasterBand(1)
            band.SetNoDataValue(nodata_value)
            band.SetDescription('Enhanced Bathymetry (m)')
            
            write_result = band.WriteArray(output_data)
            if write_result != 0:
                self.logger.warning(f"WriteArray returned error code: {write_result}")
            
            # Force flush after writing
            band.FlushCache()
            
            # Set statistics
            valid_data = output_data[output_data != nodata_value]
            if len(valid_data) > 0:
                band.SetStatistics(
                    float(np.min(valid_data)),
                    float(np.max(valid_data)),
                    float(np.mean(valid_data)),
                    float(np.std(valid_data))
                )
            
            # Set comprehensive metadata
            band_metadata = {
                'PROCESSING_SOFTWARE': 'Enhanced Bathymetric CAE v2.0 FIXED',
                'CREATION_DATE': datetime.datetime.now().isoformat(),
                'DATA_TYPE': 'Enhanced Bathymetry',
                'UNITS': 'meters'
            }
            
            if quality_metrics:
                band_metadata.update({
                    'QUALITY_SSIM': str(quality_metrics.get('ssim', 0)),
                    'QUALITY_COMPOSITE': str(quality_metrics.get('composite_quality', 0))
                })
            
            band.SetMetadata(band_metadata)
            
            # Force final flush and close
            band.FlushCache()
            band = None
            dataset.FlushCache()
            dataset = None
            
            self.logger.info("✅ GeoTIFF created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating GeoTIFF: {e}")
            return False

    def _create_robust_ascii(self, enhanced_data: np.ndarray, output_path: Path,
                            geotransform: tuple, metadata: dict,
                            quality_metrics: Optional[Dict] = None) -> bool:
        """Create robust ESRI ASCII Grid file."""
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            height, width = enhanced_data.shape
            nodata_value = -9999.0
            
            # Prepare data
            output_data = enhanced_data.astype(np.float32)
            invalid_mask = ~np.isfinite(output_data)
            if np.any(invalid_mask):
                output_data[invalid_mask] = nodata_value
            
            # Extract geotransform parameters
            xllcorner = geotransform[0]
            yllcorner = geotransform[3] + (height * geotransform[5])
            cellsize = abs(geotransform[1])
            
            # Write ASCII Grid format
            with open(output_path, 'w') as f:
                f.write(f"ncols         {width}\n")
                f.write(f"nrows         {height}\n")
                f.write(f"xllcorner     {xllcorner:.10f}\n")
                f.write(f"yllcorner     {yllcorner:.10f}\n")
                f.write(f"cellsize      {cellsize:.10f}\n")
                f.write(f"NODATA_value  {nodata_value}\n")
                
                # Write data row by row
                for row in output_data:
                    f.write(' '.join(f'{val:.6f}' for val in row) + '\n')
            
            self.logger.info("✅ ASCII Grid created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating ASCII Grid: {e}")
            return False
    
    def _save_ensemble_model(self, ensemble_models: List, model_path: str):
        """Save ensemble models with comprehensive error handling."""
        try:
            # Create ensemble save directory
            model_path_obj = Path(model_path)
            ensemble_dir = model_path_obj.parent / "ensemble_models"
            ensemble_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual models with proper error handling
            saved_count = 0
            for i, model in enumerate(ensemble_models):
                try:
                    individual_path = ensemble_dir / f"ensemble_model_{i}.keras"  # Use .keras format
                    
                    # Check if model has save method (Keras model)
                    if hasattr(model, 'save'):
                        model.save(str(individual_path))
                        self.logger.info(f"Saved ensemble model {i+1} to {individual_path}")
                        saved_count += 1
                    else:
                        self.logger.warning(f"Model {i} does not have save method, skipping")
                        
                except Exception as e:
                    self.logger.error(f"Failed to save individual model {i}: {e}")
                    continue
            
            # Also save the main model path for compatibility
            if saved_count > 0:
                try:
                    # Save the first model as the main model for compatibility
                    if hasattr(ensemble_models[0], 'save'):
                        main_model_path = model_path_obj.with_suffix('.keras')  # Ensure .keras extension
                        ensemble_models[0].save(str(main_model_path))
                        self.logger.info(f"Saved main model to {main_model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save main model: {e}")
            
            # Save ensemble metadata
            metadata = {
                'ensemble_size': len(ensemble_models),
                'saved_models': saved_count,
                'model_type': 'Enhanced Bathymetric CAE Ensemble',
                'tensorflow_version': self.tf_version,
                'grid_size': self.config.grid_size,
                'scaling_fixed': True,
                'created_date': datetime.datetime.now().isoformat(),
                'model_files': [f"ensemble_model_{i}.keras" for i in range(saved_count)],
                'model_format': 'keras',  # Track format
                'input_channels': getattr(ensemble_models[0], 'input_shape', [None, None, None, 2])[-1] if ensemble_models else 2
            }
            
            metadata_path = ensemble_dir / "ensemble_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Ensemble saved: {saved_count}/{len(ensemble_models)} models in {ensemble_dir}")
            
            if saved_count == 0:
                self.logger.error("No models were saved successfully!")
            else:
                self.logger.info(f"Ensemble metadata saved: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save ensemble model: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _generate_processing_summary(self, file_list: List[Path], output_folder: str):
        """Generate comprehensive processing summary."""
        try:
            summary = {
                'processing_date': datetime.datetime.now().isoformat(),
                'pipeline_version': 'Enhanced Bathymetric CAE v2.0',
                'tensorflow_version': self.tf_version,
                'configuration': {
                    'grid_size': self.config.grid_size,
                    'ensemble_size': self.config.ensemble_size,
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size
                },
                'input_files': [str(f) for f in file_list],
                'total_files_processed': len(file_list),
                'scaling_issues_fixed': True,
                'depth_scaling_method': 'robust_percentile',
                'output_folder': output_folder
            }
            
            # Add scaling information if available
            if hasattr(self, 'global_scaler') and self.global_scaler:
                summary['global_scaling'] = self.global_scaler.get_scaling_metadata()
            
            # Save summary
            summary_path = Path("enhanced_processing_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Processing summary saved: {summary_path}")
            
            # Print summary to console
            print("\n" + "="*60)
            print("ENHANCED BATHYMETRIC CAE PROCESSING SUMMARY")
            print("="*60)
            print(f"Files processed: {len(file_list)}")
            print(f"Output folder: {output_folder}")
            
            if hasattr(self, 'global_scaler') and self.global_scaler.original_depth_range:
                print(f"Original depth range: {self.global_scaler.original_depth_range[0]:.1f} to {self.global_scaler.original_depth_range[1]:.1f}m")
                print(f"Scaling method: Robust percentile (p1-p99)")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Failed to generate processing summary: {e}")


# Example usage and testing functions
def main():
    """Main function for testing the pipeline."""
    try:
        from config.config import Config
        
        # Test configuration
        config = Config(
            grid_size=256,
            epochs=10,
            ensemble_size=2,
            batch_size=2,
            validation_split=0.2
        )
        
        # Create pipeline
        pipeline = EnhancedBathymetricCAEPipeline(config)
        print("Pipeline created successfully!")
        
        # Test with synthetic data if available
        print("Pipeline ready for processing.")
        print("Use: pipeline.run(input_folder, output_folder, model_name)")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")


if __name__ == "__main__":
    main()