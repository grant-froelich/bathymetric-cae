"""
Memory Management Utilities Module

This module provides utilities for monitoring and optimizing memory usage
during bathymetric data processing and model training.

Author: Bathymetric CAE Team
License: MIT
"""

import gc
import logging
import psutil
from contextlib import contextmanager
from typing import Optional, Dict, Any
import warnings


def get_memory_info() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dict[str, float]: Memory information in MB
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }
    except Exception as e:
        logging.warning(f"Could not get memory info: {e}")
        return {'error': str(e)}


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory usage information if available.
    
    Returns:
        Dict[str, Any]: GPU memory information
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return {'gpus': 0, 'message': 'No GPUs available'}
        
        gpu_info = []
        for i, gpu in enumerate(gpus):
            try:
                # This requires TensorFlow 2.x and may not work in all configurations
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                gpu_info.append({
                    'gpu_id': i,
                    'name': gpu.name,
                    'current_mb': memory_info['current'] / 1024 / 1024,
                    'peak_mb': memory_info['peak'] / 1024 / 1024
                })
            except Exception:
                gpu_info.append({
                    'gpu_id': i,
                    'name': gpu.name,
                    'status': 'Memory info not available'
                })
        
        return {'gpus': len(gpus), 'gpu_info': gpu_info}
    
    except ImportError:
        return {'error': 'TensorFlow not available'}
    except Exception as e:
        return {'error': str(e)}


@contextmanager
def memory_monitor(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to monitor memory usage during an operation.
    
    Args:
        operation_name: Name of the operation being monitored
        logger: Logger instance (uses default if None)
    
    Yields:
        Dict: Memory information at start and during operation
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get initial memory info
    start_memory = get_memory_info()
    start_rss = start_memory.get('rss_mb', 0)
    
    logger.debug(f"{operation_name} - Starting memory: {start_rss:.1f}MB")
    
    memory_info = {'start_memory': start_memory}
    
    try:
        yield memory_info
    finally:
        # Get final memory info
        end_memory = get_memory_info()
        end_rss = end_memory.get('rss_mb', 0)
        memory_diff = end_rss - start_rss
        
        memory_info['end_memory'] = end_memory
        memory_info['memory_diff_mb'] = memory_diff
        
        logger.info(f"{operation_name} - Memory: {start_rss:.1f}MB → {end_rss:.1f}MB "
                   f"({memory_diff:+.1f}MB)")


def force_garbage_collection():
    """
    Force garbage collection and log memory before/after.
    
    Returns:
        Dict: Memory information before and after GC
    """
    logger = logging.getLogger(__name__)
    
    before = get_memory_info()
    before_rss = before.get('rss_mb', 0)
    
    # Force garbage collection
    collected = gc.collect()
    
    after = get_memory_info()
    after_rss = after.get('rss_mb', 0)
    freed = before_rss - after_rss
    
    logger.debug(f"Garbage collection: {collected} objects collected, "
                f"{freed:.1f}MB freed ({before_rss:.1f} → {after_rss:.1f}MB)")
    
    return {
        'before': before,
        'after': after,
        'objects_collected': collected,
        'memory_freed_mb': freed
    }


def check_memory_threshold(threshold_percent: float = 85.0) -> bool:
    """
    Check if memory usage exceeds threshold.
    
    Args:
        threshold_percent: Memory usage threshold percentage
        
    Returns:
        bool: True if memory usage exceeds threshold
    """
    memory_info = get_memory_info()
    current_percent = memory_info.get('percent', 0)
    
    if current_percent > threshold_percent:
        logging.warning(f"Memory usage high: {current_percent:.1f}% > {threshold_percent}%")
        return True
    
    return False


def optimize_memory_usage(aggressive: bool = False):
    """
    Optimize memory usage by cleaning up and configuring settings.
    
    Args:
        aggressive: If True, perform more aggressive cleanup
    """
    logger = logging.getLogger(__name__)
    
    with memory_monitor("Memory optimization", logger):
        # Force garbage collection
        force_garbage_collection()
        
        if aggressive:
            # More aggressive cleanup
            import sys
            
            # Clear module caches
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            
            # Force multiple GC cycles
            for _ in range(3):
                gc.collect()
        
        # Configure TensorFlow memory settings if available
        try:
            configure_tensorflow_memory()
        except Exception as e:
            logger.debug(f"TensorFlow memory configuration failed: {e}")


def configure_tensorflow_memory(enable_growth: bool = True):
    """
    Configure TensorFlow memory settings.
    
    Args:
        enable_growth: Enable GPU memory growth
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and enable_growth:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"GPU memory growth enabled for {gpu.name}")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    logging.warning(f"Could not set memory growth for {gpu.name}: {e}")
        
        # Set mixed precision policy for memory efficiency
        try:
            from tensorflow.keras.mixed_precision import Policy
            policy = Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Mixed precision policy enabled for memory efficiency")
        except Exception as e:
            logging.debug(f"Could not set mixed precision policy: {e}")
            
    except ImportError:
        logging.debug("TensorFlow not available for memory configuration")


class MemoryProfiler:
    """
    Class for profiling memory usage during operations.
    
    Provides detailed memory tracking and reporting capabilities.
    """
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize memory profiler.
        
        Args:
            name: Name of the operation being profiled
        """
        self.name = name
        self.memory_snapshots = []
        self.logger = logging.getLogger(__name__)
    
    def snapshot(self, label: str = None):
        """
        Take a memory snapshot.
        
        Args:
            label: Optional label for the snapshot
        """
        memory_info = get_memory_info()
        gpu_info = get_gpu_memory_info()
        
        snapshot = {
            'label': label or f"snapshot_{len(self.memory_snapshots)}",
            'timestamp': self._get_timestamp(),
            'memory': memory_info,
            'gpu': gpu_info
        }
        
        self.memory_snapshots.append(snapshot)
        
        if label:
            rss_mb = memory_info.get('rss_mb', 0)
            self.logger.debug(f"{self.name} - {label}: {rss_mb:.1f}MB")
    
    def get_peak_memory(self) -> float:
        """
        Get peak memory usage from snapshots.
        
        Returns:
            float: Peak memory usage in MB
        """
        if not self.memory_snapshots:
            return 0.0
        
        return max(
            snapshot['memory'].get('rss_mb', 0) 
            for snapshot in self.memory_snapshots
        )
    
    def get_memory_growth(self) -> float:
        """
        Get memory growth from first to last snapshot.
        
        Returns:
            float: Memory growth in MB
        """
        if len(self.memory_snapshots) < 2:
            return 0.0
        
        first = self.memory_snapshots[0]['memory'].get('rss_mb', 0)
        last = self.memory_snapshots[-1]['memory'].get('rss_mb', 0)
        
        return last - first
    
    def report(self) -> Dict:
        """
        Generate memory usage report.
        
        Returns:
            Dict: Detailed memory usage report
        """
        if not self.memory_snapshots:
            return {'error': 'No snapshots available'}
        
        peak_memory = self.get_peak_memory()
        memory_growth = self.get_memory_growth()
        
        report = {
            'operation': self.name,
            'num_snapshots': len(self.memory_snapshots),
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': memory_growth,
            'snapshots': self.memory_snapshots
        }
        
        self.logger.info(f"{self.name} Memory Report - Peak: {peak_memory:.1f}MB, "
                        f"Growth: {memory_growth:+.1f}MB")
        
        return report
    
    def _get_timestamp(self):
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()


def get_optimal_batch_size(model_memory_mb: float, available_memory_mb: float, 
                          safety_factor: float = 0.7) -> int:
    """
    Estimate optimal batch size based on available memory.
    
    Args:
        model_memory_mb: Memory required for model in MB
        available_memory_mb: Available memory in MB
        safety_factor: Safety factor to prevent OOM (0.5-0.8)
        
    Returns:
        int: Recommended batch size
    """
    if model_memory_mb <= 0 or available_memory_mb <= 0:
        return 1
    
    # Estimate memory per batch (very rough approximation)
    memory_per_batch = model_memory_mb * 0.1  # Assume 10% of model size per batch
    
    # Calculate optimal batch size with safety factor
    usable_memory = available_memory_mb * safety_factor
    optimal_batch_size = max(1, int(usable_memory / memory_per_batch))
    
    # Limit to reasonable values
    return min(optimal_batch_size, 32)


def memory_efficient_data_loading(batch_size: int = None, prefetch_buffer: int = 2):
    """
    Configure memory-efficient data loading parameters.
    
    Args:
        batch_size: Batch size (auto-detect if None)
        prefetch_buffer: Prefetch buffer size
        
    Returns:
        Dict: Optimized data loading configuration
    """
    memory_info = get_memory_info()
    available_mb = memory_info.get('available_mb', 1000)
    
    if batch_size is None:
        # Estimate reasonable batch size
        batch_size = get_optimal_batch_size(
            model_memory_mb=500,  # Rough estimate
            available_memory_mb=available_mb
        )
    
    config = {
        'batch_size': batch_size,
        'prefetch_buffer': prefetch_buffer,
        'memory_info': memory_info
    }
    
    logging.info(f"Memory-efficient loading config: batch_size={batch_size}, "
                f"available_memory={available_mb:.1f}MB")
    
    return config


def cleanup_tensorflow_session():
    """Clean up TensorFlow session and free memory."""
    try:
        import tensorflow as tf
        
        # Clear session
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        force_garbage_collection()
        
        logging.debug("TensorFlow session cleaned up")
        
    except ImportError:
        logging.debug("TensorFlow not available for cleanup")
    except Exception as e:
        logging.warning(f"Error cleaning up TensorFlow session: {e}")


def set_memory_warnings(threshold_percent: float = 80.0):
    """
    Set up memory usage warnings.
    
    Args:
        threshold_percent: Memory threshold for warnings
    """
    def memory_warning_filter(record):
        if check_memory_threshold(threshold_percent):
            memory_info = get_memory_info()
            record.msg = f"HIGH MEMORY USAGE ({memory_info.get('percent', 0):.1f}%) - {record.msg}"
        return True
    
    # Add filter to root logger
    logging.getLogger().addFilter(memory_warning_filter)