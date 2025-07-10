"""
Memory management utilities for Enhanced Bathymetric CAE Processing.

This module provides memory monitoring, GPU optimization, and cleanup utilities
to handle large datasets efficiently.
"""

import gc
import logging
import psutil
import tensorflow as tf
from contextlib import contextmanager
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


@contextmanager
def memory_monitor(operation_name: str):
    """Context manager to monitor memory usage during operations."""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_gpu_memory = get_gpu_memory_usage()
    
    logger.debug(f"{operation_name} - Starting memory: {start_memory:.1f}MB RAM")
    if start_gpu_memory:
        logger.debug(f"{operation_name} - Starting GPU memory: {start_gpu_memory:.1f}MB")
    
    try:
        yield
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_gpu_memory = get_gpu_memory_usage()
        
        memory_diff = end_memory - start_memory
        
        log_msg = f"{operation_name} - Memory: {start_memory:.1f}MB → {end_memory:.1f}MB ({memory_diff:+.1f}MB)"
        
        if start_gpu_memory and end_gpu_memory:
            gpu_diff = end_gpu_memory - start_gpu_memory
            log_msg += f", GPU: {start_gpu_memory:.1f}MB → {end_gpu_memory:.1f}MB ({gpu_diff:+.1f}MB)"
        
        logger.info(log_msg)


def optimize_gpu_memory():
    """Optimize GPU memory usage for TensorFlow."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                # Enable memory growth to avoid allocating all GPU memory at once
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Optional: Set memory limit
                # tf.config.experimental.set_memory_limit(gpu, 4096)  # 4GB limit
            
            logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            
            # Log GPU information
            for i, gpu in enumerate(gpus):
                logger.info(f"GPU {i}: {gpu.name}")
        else:
            logger.info("No GPU detected, using CPU")
            
    except RuntimeError as e:
        logger.warning(f"GPU memory optimization failed: {e}")
        logger.info("This is normal if GPU is already in use by another process")
    except Exception as e:
        logger.error(f"Unexpected error in GPU optimization: {e}")


def get_gpu_memory_usage() -> Optional[float]:
    """Get current GPU memory usage in MB."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return None
        
        # Get memory info for first GPU
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        if 'device_name' in gpu_details:
            # This is a simplified approach - TensorFlow doesn't provide direct memory usage
            # In practice, you might want to use nvidia-ml-py for more detailed GPU monitoring
            return None
        
    except Exception as e:
        logger.debug(f"Could not get GPU memory usage: {e}")
        return None


def cleanup_memory():
    """Perform memory cleanup operations."""
    try:
        # Clear TensorFlow session if any
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Log memory status
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.debug(f"Memory cleanup: collected {collected} objects, current RAM: {current_memory:.1f}MB")
        
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")


def get_memory_info() -> Dict[str, Any]:
    """Get comprehensive memory information."""
    try:
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # GPU memory (if available)
        gpu_info = {}
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for i, gpu in enumerate(gpus):
                    gpu_info[f'gpu_{i}'] = {
                        'name': gpu.name,
                        'available': True
                    }
        except Exception:
            gpu_info = {'available': False}
        
        return {
            'system': {
                'total_mb': system_memory.total / 1024 / 1024,
                'available_mb': system_memory.available / 1024 / 1024,
                'used_mb': system_memory.used / 1024 / 1024,
                'percent_used': system_memory.percent
            },
            'process': {
                'rss_mb': process_memory.rss / 1024 / 1024,
                'vms_mb': process_memory.vms / 1024 / 1024
            },
            'gpu': gpu_info
        }
        
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {'error': str(e)}


def check_memory_threshold(threshold_mb: float = 1000.0) -> bool:
    """Check if process memory usage exceeds threshold."""
    try:
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if current_memory > threshold_mb:
            logger.warning(f"Memory usage ({current_memory:.1f}MB) exceeds threshold ({threshold_mb}MB)")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking memory threshold: {e}")
        return False


def force_memory_cleanup():
    """Force aggressive memory cleanup."""
    try:
        # Clear TensorFlow backend
        tf.keras.backend.clear_session()
        
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        # Log memory after cleanup
        process = psutil.Process()
        memory_after = process.memory_info().rss / 1024 / 1024
        
        logger.info(f"Forced memory cleanup completed, current RAM: {memory_after:.1f}MB")
        
    except Exception as e:
        logger.error(f"Force memory cleanup failed: {e}")


class MemoryTracker:
    """Track memory usage throughout processing."""
    
    def __init__(self, name: str = "memory_tracker"):
        self.name = name
        self.logger = logging.getLogger(f"memory.{name}")
        self.checkpoints = []
        self.start_memory = None
    
    def start_tracking(self):
        """Start memory tracking."""
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory tracking started: {self.start_memory:.1f}MB")
        except Exception as e:
            self.logger.error(f"Error starting memory tracking: {e}")
    
    def checkpoint(self, label: str):
        """Add a memory checkpoint."""
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            
            checkpoint_data = {
                'label': label,
                'memory_mb': current_memory,
                'delta_mb': current_memory - (self.start_memory or 0)
            }
            
            self.checkpoints.append(checkpoint_data)
            self.logger.info(f"Checkpoint '{label}': {current_memory:.1f}MB ({checkpoint_data['delta_mb']:+.1f}MB)")
            
        except Exception as e:
            self.logger.error(f"Error adding checkpoint '{label}': {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory tracking summary."""
        try:
            if not self.checkpoints:
                return {'error': 'No checkpoints recorded'}
            
            max_memory = max(cp['memory_mb'] for cp in self.checkpoints)
            min_memory = min(cp['memory_mb'] for cp in self.checkpoints)
            
            return {
                'start_memory_mb': self.start_memory,
                'max_memory_mb': max_memory,
                'min_memory_mb': min_memory,
                'memory_range_mb': max_memory - min_memory,
                'checkpoints': self.checkpoints
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory summary: {e}")
            return {'error': str(e)}


def set_tensorflow_memory_limit(limit_mb: int):
    """Set TensorFlow GPU memory limit."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_limit(gpu, limit_mb)
            logger.info(f"Set TensorFlow GPU memory limit to {limit_mb}MB")
        else:
            logger.warning("No GPUs available for memory limit setting")
            
    except RuntimeError as e:
        logger.error(f"Cannot set memory limit after GPU initialization: {e}")
    except Exception as e:
        logger.error(f"Error setting TensorFlow memory limit: {e}")


def enable_mixed_precision():
    """Enable mixed precision training for memory optimization."""
    try:
        from tensorflow.keras.mixed_precision import Policy
        
        policy = Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        logger.info("Mixed precision training enabled (float16)")
        
    except Exception as e:
        logger.error(f"Error enabling mixed precision: {e}")


@contextmanager
def temporary_memory_limit(limit_mb: int):
    """Temporarily set lower memory usage for operations."""
    original_limit = None
    
    try:
        # Note: This is a conceptual implementation
        # Actual memory limiting would require more sophisticated approaches
        yield
        
    except Exception as e:
        logger.error(f"Error in temporary memory limit context: {e}")
        raise
    finally:
        # Restore original settings if needed
        pass


def log_memory_stats():
    """Log current memory statistics."""
    try:
        memory_info = get_memory_info()
        
        logger.info("Memory Statistics:")
        logger.info(f"  System - Total: {memory_info['system']['total_mb']:.1f}MB, "
                   f"Used: {memory_info['system']['used_mb']:.1f}MB "
                   f"({memory_info['system']['percent_used']:.1f}%)")
        logger.info(f"  Process - RSS: {memory_info['process']['rss_mb']:.1f}MB, "
                   f"VMS: {memory_info['process']['vms_mb']:.1f}MB")
        
        if memory_info['gpu'].get('available', False):
            logger.info("  GPU: Available")
            for gpu_id, gpu_info in memory_info['gpu'].items():
                if gpu_id.startswith('gpu_'):
                    logger.info(f"    {gpu_id}: {gpu_info['name']}")
        else:
            logger.info("  GPU: Not available")
            
    except Exception as e:
        logger.error(f"Error logging memory stats: {e}")


# Memory management configuration
MEMORY_CONFIG = {
    'cleanup_threshold_mb': 2000,  # Cleanup when process exceeds this
    'warning_threshold_mb': 1500,  # Warning when process exceeds this
    'gpu_memory_growth': True,     # Enable GPU memory growth
    'mixed_precision': True,       # Enable mixed precision by default
    'auto_cleanup_interval': 10    # Auto cleanup every N processed files
}


def configure_memory_management(config: Optional[Dict[str, Any]] = None):
    """Configure memory management settings."""
    if config:
        MEMORY_CONFIG.update(config)
    
    logger.info(f"Memory management configured: {MEMORY_CONFIG}")
    
    # Apply GPU optimizations
    if MEMORY_CONFIG['gpu_memory_growth']:
        optimize_gpu_memory()
    
    # Enable mixed precision if requested
    if MEMORY_CONFIG['mixed_precision']:
        enable_mixed_precision()