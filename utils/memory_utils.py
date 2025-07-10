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
