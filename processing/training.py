"""
Model training utilities for Enhanced Bathymetric CAE Processing.

This module provides training infrastructure including callbacks,
training loops, and model management for the ensemble system.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import datetime

from ..config import Config
from ..utils.logging_utils import get_performance_logger
from ..utils.visualization import plot_training_history
from .memory_utils import memory_monitor, cleanup_memory


class ModelTrainer:
    """Training infrastructure for bathymetric models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.performance_logger = get_performance_logger("training")
        
        # Training history storage
        self.training_histories = []
        self.best_models = []
    
    def train_ensemble(self, models: List[tf.keras.Model], 
                      X_train: np.ndarray, y_train: np.ndarray,
                      model_base_path: str) -> List[tf.keras.Model]:
        """Train ensemble of models."""
        self.logger.info(f"Training ensemble of {len(models)} models")
        
        trained_models = []
        
        for i, model in enumerate(models):
            self.logger.info(f"Training model {i+1}/{len(models)}")
            
            try:
                # Train individual model
                trained_model, history = self._train_single_model(
                    model, X_train, y_train, f"{model_base_path}_model_{i}"
                )
                
                trained_models.append(trained_model)
                self.training_histories.append(history)
                
                # Plot training history
                plot_training_history(history, f"training_history_model_{i}.png")
                
                # Cleanup after each model
                cleanup_memory()
                
            except Exception as e:
                self.logger.error(f"Error in quality assessment: {e}")


class LearningRateSchedulerCallback(callbacks.Callback):
    """Custom learning rate scheduler."""
    
    def __init__(self, schedule_fn):
        super().__init__()
        self.schedule_fn = schedule_fn
        self.logger = logging.getLogger("lr_scheduler")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Update learning rate at beginning of epoch."""
        try:
            new_lr = self.schedule_fn(epoch)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            self.logger.debug(f"Epoch {epoch}: Learning rate set to {new_lr:.2e}")
        except Exception as e:
            self.logger.error(f"Error updating learning rate: {e}")


def create_cosine_decay_schedule(initial_lr: float, decay_steps: int, alpha: float = 0.0):
    """Create cosine decay learning rate schedule."""
    def schedule(epoch):
        return initial_lr * (1 + np.cos(np.pi * epoch / decay_steps)) / 2 * (1 - alpha) + alpha
    return schedule


def create_exponential_decay_schedule(initial_lr: float, decay_rate: float, decay_steps: int):
    """Create exponential decay learning rate schedule."""
    def schedule(epoch):
        return initial_lr * (decay_rate ** (epoch / decay_steps))
    return schedule


class AdaptiveBatchSizeCallback(callbacks.Callback):
    """Callback to adaptively adjust batch size during training."""
    
    def __init__(self, initial_batch_size: int, memory_threshold: float = 0.8):
        super().__init__()
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger("adaptive_batch_size")
    
    def on_epoch_end(self, epoch, logs=None):
        """Adjust batch size based on memory usage."""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100
            
            if memory_percent > self.memory_threshold and self.current_batch_size > 1:
                # Reduce batch size
                self.current_batch_size = max(1, self.current_batch_size // 2)
                self.logger.warning(f"High memory usage ({memory_percent:.1%}), "
                                   f"reducing batch size to {self.current_batch_size}")
            
            elif memory_percent < self.memory_threshold * 0.5 and self.current_batch_size < self.initial_batch_size:
                # Increase batch size
                self.current_batch_size = min(self.initial_batch_size, self.current_batch_size * 2)
                self.logger.info(f"Low memory usage ({memory_percent:.1%}), "
                                f"increasing batch size to {self.current_batch_size}")
                
        except Exception as e:
            self.logger.error(f"Error in adaptive batch size: {e}")


class ModelCheckpointWithMetrics(callbacks.Callback):
    """Enhanced model checkpoint that saves additional metrics."""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score = float('inf')
        self.logger = logging.getLogger("model_checkpoint")
    
    def on_epoch_end(self, epoch, logs=None):
        """Save model with enhanced metrics."""
        try:
            current_score = logs.get(self.monitor, float('inf'))
            
            should_save = not self.save_best_only or current_score < self.best_score
            
            if should_save:
                # Save model
                self.model.save(self.filepath)
                
                # Save additional metrics
                metrics_path = self.filepath.replace('.h5', '_metrics.json')
                metrics_data = {
                    'epoch': epoch,
                    'score': float(current_score),
                    'all_metrics': {k: float(v) for k, v in logs.items()},
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                with open(metrics_path, 'w') as f:
                    import json
                    json.dump(metrics_data, f, indent=2)
                
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.logger.info(f"New best model saved at epoch {epoch}: {self.monitor}={current_score:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error saving model checkpoint: {e}")


def create_training_data_generator(X_data: np.ndarray, y_data: np.ndarray, 
                                 batch_size: int, augment: bool = False):
    """Create data generator for training with optional augmentation."""
    
    def data_generator():
        """Generator function for training data."""
        while True:
            # Shuffle data
            indices = np.random.permutation(len(X_data))
            
            for start_idx in range(0, len(X_data), batch_size):
                end_idx = min(start_idx + batch_size, len(X_data))
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_data[batch_indices]
                y_batch = y_data[batch_indices]
                
                if augment:
                    X_batch, y_batch = apply_data_augmentation(X_batch, y_batch)
                
                yield X_batch, y_batch
    
    return data_generator


def apply_data_augmentation(X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation to training batch."""
    
    # Simple augmentations for bathymetric data
    augmented_X = X_batch.copy()
    augmented_y = y_batch.copy()
    
    for i in range(len(X_batch)):
        # Random horizontal flip
        if np.random.random() > 0.5:
            augmented_X[i] = np.fliplr(augmented_X[i])
            augmented_y[i] = np.fliplr(augmented_y[i])
        
        # Random vertical flip
        if np.random.random() > 0.5:
            augmented_X[i] = np.flipud(augmented_X[i])
            augmented_y[i] = np.flipud(augmented_y[i])
        
        # Random rotation (90 degree increments only for bathymetric data)
        if np.random.random() > 0.75:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            augmented_X[i] = np.rot90(augmented_X[i], k)
            augmented_y[i] = np.rot90(augmented_y[i], k)
    
    return augmented_X, augmented_y


def setup_distributed_training(strategy_name: str = 'mirrored'):
    """Setup distributed training strategy."""
    try:
        if strategy_name == 'mirrored':
            strategy = tf.distribute.MirroredStrategy()
        elif strategy_name == 'multi_worker':
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()  # Default strategy
        
        logger = logging.getLogger("distributed_training")
        logger.info(f"Number of replicas in sync: {strategy.num_replicas_in_sync}")
        
        return strategy
        
    except Exception as e:
        logger = logging.getLogger("distributed_training")
        logger.error(f"Error setting up distributed training: {e}")
        return tf.distribute.get_strategy()  # Fallback to default


def create_advanced_optimizer(learning_rate: float, optimizer_type: str = 'adamw'):
    """Create advanced optimizer with better performance."""
    
    if optimizer_type.lower() == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    elif optimizer_type.lower() == 'radam':
        # Rectified Adam
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    elif optimizer_type.lower() == 'lion':
        # Would require custom implementation or additional library
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    return optimizer


def save_training_config(config: Config, save_path: str):
    """Save training configuration for reproducibility."""
    
    training_config = {
        'model_config': config.get_model_config(),
        'training_config': config.get_training_config(),
        'quality_weights': config.get_quality_weights(),
        'timestamp': datetime.datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'random_seeds': {
            'numpy': np.random.get_state()[1][0],  # Get first element of random state
            'tensorflow': tf.random.get_global_generator().state.numpy().tolist()
        }
    }
    
    with open(save_path, 'w') as f:
        import json
        json.dump(training_config, f, indent=2)


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration."""
    
    with open(config_path, 'r') as f:
        import json
        return json.load(f)


def validate_training_data(X_data: np.ndarray, y_data: np.ndarray) -> bool:
    """Validate training data before starting training."""
    
    logger = logging.getLogger("data_validation")
    errors = []
    
    # Shape validation
    if X_data.shape[0] != y_data.shape[0]:
        errors.append(f"Sample count mismatch: X={X_data.shape[0]}, y={y_data.shape[0]}")
    
    # Data type validation
    if not np.issubdtype(X_data.dtype, np.floating):
        errors.append(f"X_data should be floating point, got {X_data.dtype}")
    
    if not np.issubdtype(y_data.dtype, np.floating):
        errors.append(f"y_data should be floating point, got {y_data.dtype}")
    
    # Range validation
    if np.any(np.isnan(X_data)) or np.any(np.isinf(X_data)):
        errors.append("X_data contains NaN or infinite values")
    
    if np.any(np.isnan(y_data)) or np.any(np.isinf(y_data)):
        errors.append("y_data contains NaN or infinite values")
    
    # Size validation
    if X_data.size == 0 or y_data.size == 0:
        errors.append("Training data is empty")
    
    if errors:
        for error in errors:
            logger.error(f"Training data validation error: {error}")
        return False
    
    logger.info("Training data validation passed")
    return True


# Training utilities for different scenarios
class TrainingScenarios:
    """Predefined training scenarios for different use cases."""
    
    @staticmethod
    def quick_training(config: Config) -> Config:
        """Configure for quick training/testing."""
        config.epochs = 10
        config.batch_size = 16
        config.early_stopping_patience = 3
        config.reduce_lr_patience = 2
        return config
    
    @staticmethod
    def production_training(config: Config) -> Config:
        """Configure for production-quality training."""
        config.epochs = 200
        config.batch_size = 8
        config.early_stopping_patience = 20
        config.reduce_lr_patience = 10
        return config
    
    @staticmethod
    def fine_tuning(config: Config) -> Config:
        """Configure for fine-tuning existing models."""
        config.epochs = 50
        config.learning_rate = config.learning_rate * 0.1  # Lower learning rate
        config.early_stopping_patience = 10
        return config
    
    @staticmethod
    def memory_constrained(config: Config) -> Config:
        """Configure for memory-constrained training."""
        config.batch_size = max(1, config.batch_size // 2)
        config.use_mixed_precision = True
        return configf"Failed to train model {i+1}: {e}")
                # Continue with other models
                continue
        
        if not trained_models:
            raise RuntimeError("No models were successfully trained")
        
        self.logger.info(f"Successfully trained {len(trained_models)} models")
        return trained_models
    
    def _train_single_model(self, model: tf.keras.Model, 
                           X_train: np.ndarray, y_train: np.ndarray,
                           model_path: str) -> Tuple[tf.keras.Model, Any]:
        """Train a single model with callbacks and monitoring."""
        
        # Setup callbacks
        model_callbacks = self._setup_callbacks(f"{model_path}.h5")
        
        # Training with memory monitoring
        with memory_monitor(f"Training {Path(model_path).name}"):
            self.performance_logger.start_timer(f"train_{Path(model_path).name}")
            
            try:
                history = model.fit(
                    X_train, y_train,
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    validation_split=self.config.validation_split,
                    callbacks=model_callbacks,
                    verbose=1,
                    shuffle=True
                )
                
                # Load best weights if early stopping was triggered
                if any(isinstance(cb, callbacks.EarlyStopping) for cb in model_callbacks):
                    try:
                        model.load_weights(f"{model_path}.h5")
                        self.logger.info("Loaded best weights from early stopping")
                    except Exception as e:
                        self.logger.warning(f"Could not load best weights: {e}")
                
                return model, history
                
            finally:
                training_time = self.performance_logger.end_timer(f"train_{Path(model_path).name}")
                self.logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    def _setup_callbacks(self, model_path: str) -> List[callbacks.Callback]:
        """Setup training callbacks."""
        callback_list = []
        
        # TensorBoard
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=self.config.log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callback_list.append(tensorboard_callback)
        
        # Early Stopping
        early_stopping_callback = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callback_list.append(early_stopping_callback)
        
        # Reduce Learning Rate
        reduce_lr_callback = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
            verbose=1,
            mode='min'
        )
        callback_list.append(reduce_lr_callback)
        
        # Model Checkpoint
        checkpoint_callback = callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        )
        callback_list.append(checkpoint_callback)
        
        # CSV Logger
        csv_logger_callback = callbacks.CSVLogger(
            f'{model_path}_training_log.csv',
            append=True
        )
        callback_list.append(csv_logger_callback)
        
        # Custom callbacks
        callback_list.extend(self._create_custom_callbacks(model_path))
        
        return callback_list
    
    def _create_custom_callbacks(self, model_path: str) -> List[callbacks.Callback]:
        """Create custom training callbacks."""
        custom_callbacks = []
        
        # Memory monitoring callback
        memory_callback = MemoryMonitorCallback()
        custom_callbacks.append(memory_callback)
        
        # Training progress callback
        progress_callback = TrainingProgressCallback(
            log_frequency=max(1, self.config.epochs // 10)
        )
        custom_callbacks.append(progress_callback)
        
        # Quality assessment callback (if validation data includes quality metrics)
        quality_callback = QualityAssessmentCallback()
        custom_callbacks.append(quality_callback)
        
        return custom_callbacks
    
    def cross_validate_model(self, model_fn, X_data: np.ndarray, y_data: np.ndarray,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on a single model."""
        from sklearn.model_selection import KFold
        
        self.logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        cv_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_data)):
            self.logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X_data[train_idx], X_data[val_idx]
            y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]
            
            # Create fresh model
            model = model_fn()
            
            # Reduced epochs for CV
            original_epochs = self.config.epochs
            self.config.epochs = min(20, original_epochs)
            
            try:
                # Train on fold
                model, history = self._train_single_model(
                    model, X_train_fold, y_train_fold, f"cv_fold_{fold}"
                )
                
                # Evaluate on validation set
                val_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)
                if isinstance(val_loss, list):
                    val_loss = val_loss[0]  # Get main loss if multiple outputs
                
                cv_scores.append(val_loss)
                cv_histories.append(history)
                
                self.logger.info(f"Fold {fold + 1} validation loss: {val_loss:.4f}")
                
            finally:
                # Restore original epochs
                self.config.epochs = original_epochs
        
        cv_results = {
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'scores': cv_scores,
            'histories': cv_histories
        }
        
        self.logger.info(f"Cross-validation completed: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        return cv_results
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process."""
        if not self.training_histories:
            return {'error': 'No training completed'}
        
        summary = {
            'models_trained': len(self.training_histories),
            'total_epochs': sum(len(h.history['loss']) for h in self.training_histories),
            'best_val_losses': [min(h.history['val_loss']) for h in self.training_histories],
            'final_val_losses': [h.history['val_loss'][-1] for h in self.training_histories],
            'training_times': []  # Would need to track this separately
        }
        
        # Calculate averages
        summary['mean_best_val_loss'] = float(np.mean(summary['best_val_losses']))
        summary['mean_final_val_loss'] = float(np.mean(summary['final_val_losses']))
        
        return summary


class MemoryMonitorCallback(callbacks.Callback):
    """Callback to monitor memory usage during training."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("memory_monitor")
    
    def on_epoch_end(self, epoch, logs=None):
        """Monitor memory at end of each epoch."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if epoch % 10 == 0:  # Log every 10 epochs
                self.logger.info(f"Epoch {epoch}: Memory usage {memory_mb:.1f}MB")
                
                # Trigger cleanup if memory is high
                if memory_mb > 2000:  # 2GB threshold
                    self.logger.warning("High memory usage detected, triggering cleanup")
                    cleanup_memory()
                    
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {e}")


class TrainingProgressCallback(callbacks.Callback):
    """Callback for enhanced training progress reporting."""
    
    def __init__(self, log_frequency: int = 10):
        super().__init__()
        self.log_frequency = log_frequency
        self.logger = logging.getLogger("training_progress")
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        """Initialize training progress tracking."""
        self.start_time = datetime.datetime.now()
        self.logger.info("Training started")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log training progress."""
        if epoch % self.log_frequency == 0:
            elapsed = datetime.datetime.now() - self.start_time
            
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            lr = logs.get('lr', 0)
            
            self.logger.info(
                f"Epoch {epoch}: loss={loss:.4f}, val_loss={val_loss:.4f}, "
                f"lr={lr:.2e}, elapsed={elapsed}"
            )
    
    def on_train_end(self, logs=None):
        """Log training completion."""
        total_time = datetime.datetime.now() - self.start_time
        self.logger.info(f"Training completed in {total_time}")


class QualityAssessmentCallback(callbacks.Callback):
    """Callback for quality assessment during training."""
    
    def __init__(self, assessment_frequency: int = 20):
        super().__init__()
        self.assessment_frequency = assessment_frequency
        self.logger = logging.getLogger("quality_assessment")
    
    def on_epoch_end(self, epoch, logs=None):
        """Assess model quality periodically."""
        if epoch % self.assessment_frequency == 0:
            try:
                # This would require access to validation data and quality metrics
                # Simplified implementation for now
                val_loss = logs.get('val_loss', 0)
                
                if val_loss < 0.01:
                    quality = "excellent"
                elif val_loss < 0.05:
                    quality = "good"
                elif val_loss < 0.1:
                    quality = "acceptable"
                else:
                    quality = "poor"
                
                self.logger.info(f"Epoch {epoch}: Quality assessment - {quality} (val_loss: {val_loss:.4f})")
                
            except Exception as e:
                self.logger.error(