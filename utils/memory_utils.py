"""
Memory management and monitoring utilities.
"""

import logging
import psutil
import tensorflow as tf
from contextlib import contextmanager


@contextmanager
def memory_monitor(operation_name: str):
    """Context manager to monitor memory usage."""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = end_memory - start_memory
        logging.info(f"{operation_name} - Memory: {start_memory:.1f}MB → {end_memory:.1f}MB "
                    f"({memory_diff:+.1f}MB)")

def log_memory_usage(operation_name: str = "Current"):
    """Log current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Also log GPU memory if available
        gpu_info = ""
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                gpu_info = f", GPU: {gpu_memory['current'] / 1024 / 1024:.1f}MB"
        except:
            pass
            
        logging.info(f"{operation_name} - Memory: {memory_mb:.1f}MB{gpu_info}")
        
        # Warning for high memory usage
        if memory_mb > 8000:  # 8GB
            logging.warning(f"High memory usage detected: {memory_mb:.1f}MB")
            
    except ImportError:
        logging.debug("psutil not available for memory monitoring")
        
def optimize_gpu_memory():
    """Optimize GPU memory usage."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logging.warning(f"GPU memory optimization failed: {e}")
    else:
        logging.info("No GPU detected, using CPU")
