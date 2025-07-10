"""
GPU Configuration Utilities Module

This module provides utilities for configuring and optimizing GPU usage
for the bathymetric CAE processing pipeline.

Author: Bathymetric CAE Team
License: MIT
"""

import logging
from typing import List, Dict, Optional, Any


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return detailed information.
    
    Returns:
        Dict[str, Any]: GPU availability information
    """
    try:
        import tensorflow as tf
        
        # Get physical GPU devices
        physical_gpus = tf.config.list_physical_devices('GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        
        gpu_info = {
            'tensorflow_version': tf.__version__,
            'gpu_available': len(physical_gpus) > 0,
            'num_physical_gpus': len(physical_gpus),
            'num_logical_gpus': len(logical_gpus),
            'physical_gpus': [gpu.name for gpu in physical_gpus],
            'logical_gpus': [gpu.name for gpu in logical_gpus],
            'cuda_available': tf.test.is_built_with_cuda(),
            'gpu_support': tf.test.is_gpu_available()
        }
        
        # Get detailed GPU information
        if physical_gpus:
            detailed_info = []
            for i, gpu in enumerate(physical_gpus):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    detailed_info.append({
                        'id': i,
                        'name': gpu.name,
                        'details': details
                    })
                except Exception as e:
                    detailed_info.append({
                        'id': i,
                        'name': gpu.name,
                        'error': str(e)
                    })
            
            gpu_info['detailed_gpu_info'] = detailed_info
        
        return gpu_info
    
    except ImportError:
        return {
            'error': 'TensorFlow not available',
            'gpu_available': False
        }
    except Exception as e:
        return {
            'error': str(e),
            'gpu_available': False
        }


def configure_gpu_memory(enable_growth: bool = True, memory_limit_mb: Optional[int] = None) -> bool:
    """
    Configure GPU memory settings.
    
    Args:
        enable_growth: Enable GPU memory growth
        memory_limit_mb: Set memory limit in MB (None for no limit)
        
    Returns:
        bool: True if configuration was successful
    """
    logger = logging.getLogger(__name__)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.info("No GPUs available for configuration")
            return False
        
        for gpu in gpus:
            try:
                if enable_growth:
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU memory growth enabled for {gpu.name}")
                
                if memory_limit_mb is not None:
                    # Set memory limit
                    tf.config.experimental.set_memory_limit(gpu, memory_limit_mb)
                    logger.info(f"GPU memory limit set to {memory_limit_mb}MB for {gpu.name}")
                
            except RuntimeError as e:
                # Memory configuration must be set before GPUs have been initialized
                logger.warning(f"Could not configure GPU {gpu.name}: {e}")
                return False
        
        return True
    
    except ImportError:
        logger.debug("TensorFlow not available for GPU configuration")
        return False
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
        return False


def optimize_gpu_performance() -> Dict[str, Any]:
    """
    Optimize GPU performance settings.
    
    Returns:
        Dict[str, Any]: Configuration results
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    try:
        import tensorflow as tf
        
        # Configure GPU memory growth
        memory_config = configure_gpu_memory(enable_growth=True)
        results['memory_growth'] = memory_config
        
        # Enable mixed precision for better performance
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            results['mixed_precision'] = True
            logger.info("Mixed precision training enabled")
        except Exception as e:
            results['mixed_precision'] = False
            logger.debug(f"Could not enable mixed precision: {e}")
        
        # Configure XLA compilation (experimental)
        try:
            tf.config.optimizer.set_jit(True)
            results['xla_enabled'] = True
            logger.info("XLA compilation enabled")
        except Exception as e:
            results['xla_enabled'] = False
            logger.debug(f"Could not enable XLA: {e}")
        
        # Set thread configuration for optimal performance
        try:
            tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
            tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
            results['threading_optimized'] = True
            logger.debug("Threading configuration optimized")
        except Exception as e:
            results['threading_optimized'] = False
            logger.debug(f"Could not optimize threading: {e}")
        
        return results
    
    except ImportError:
        logger.debug("TensorFlow not available for GPU optimization")
        return {'error': 'TensorFlow not available'}
    except Exception as e:
        logger.error(f"Error optimizing GPU performance: {e}")
        return {'error': str(e)}


def disable_gpu() -> bool:
    """
    Disable GPU usage and force CPU-only computation.
    
    Returns:
        bool: True if GPU was successfully disabled
    """
    logger = logging.getLogger(__name__)
    
    try:
        import tensorflow as tf
        
        # Hide GPU devices
        tf.config.set_visible_devices([], 'GPU')
        
        # Verify GPU is disabled
        visible_devices = tf.config.get_visible_devices()
        gpu_devices = [d for d in visible_devices if d.device_type == 'GPU']
        
        if not gpu_devices:
            logger.info("GPU disabled, using CPU-only computation")
            return True
        else:
            logger.warning("Failed to disable GPU")
            return False
    
    except ImportError:
        logger.debug("TensorFlow not available")
        return False
    except Exception as e:
        logger.error(f"Error disabling GPU: {e}")
        return False


def get_gpu_memory_usage() -> List[Dict[str, Any]]:
    """
    Get current GPU memory usage for all available GPUs.
    
    Returns:
        List[Dict[str, Any]]: Memory usage information for each GPU
    """
    logger = logging.getLogger(__name__)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return []
        
        memory_usage = []
        for i, gpu in enumerate(gpus):
            try:
                # Get memory info (may not work in all TensorFlow versions)
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                
                usage_info = {
                    'gpu_id': i,
                    'name': gpu.name,
                    'current_mb': memory_info['current'] / (1024 * 1024),
                    'peak_mb': memory_info['peak'] / (1024 * 1024),
                    'limit_mb': memory_info.get('limit', 0) / (1024 * 1024) if memory_info.get('limit') else None
                }
                
                memory_usage.append(usage_info)
                
            except Exception as e:
                logger.debug(f"Could not get memory info for GPU {i}: {e}")
                memory_usage.append({
                    'gpu_id': i,
                    'name': gpu.name,
                    'error': str(e)
                })
        
        return memory_usage
    
    except ImportError:
        logger.debug("TensorFlow not available")
        return []
    except Exception as e:
        logger.error(f"Error getting GPU memory usage: {e}")
        return []


def log_gpu_info():
    """Log detailed GPU information."""
    logger = logging.getLogger(__name__)
    
    gpu_info = check_gpu_availability()
    
    logger.info("=== GPU Information ===")
    
    if gpu_info.get('gpu_available'):
        logger.info(f"GPUs available: {gpu_info['num_physical_gpus']}")
        logger.info(f"CUDA support: {gpu_info.get('cuda_available', 'Unknown')}")
        
        # Log each GPU
        for gpu_name in gpu_info.get('physical_gpus', []):
            logger.info(f"GPU: {gpu_name}")
        
        # Log memory usage
        memory_usage = get_gpu_memory_usage()
        for usage in memory_usage:
            if 'error' not in usage:
                logger.info(f"GPU {usage['gpu_id']} memory: "
                          f"{usage['current_mb']:.1f}MB current, "
                          f"{usage['peak_mb']:.1f}MB peak")
    else:
        logger.info("No GPUs available")
        if 'error' in gpu_info:
            logger.debug(f"GPU check error: {gpu_info['error']}")
    
    logger.info("=" * 24)


def setup_gpu_environment(config) -> Dict[str, Any]:
    """
    Setup GPU environment based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Dict[str, Any]: Setup results
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    # Log GPU information
    log_gpu_info()
    
    # Check if GPU should be disabled
    if hasattr(config, 'disable_gpu') and config.disable_gpu:
        results['gpu_disabled'] = disable_gpu()
        return results
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    results['gpu_available'] = gpu_info.get('gpu_available', False)
    
    if results['gpu_available']:
        # Configure GPU memory
        if hasattr(config, 'gpu_memory_growth'):
            memory_limit = getattr(config, 'gpu_memory_limit_mb', None)
            results['memory_configured'] = configure_gpu_memory(
                enable_growth=config.gpu_memory_growth,
                memory_limit_mb=memory_limit
            )
        
        # Optimize GPU performance
        if hasattr(config, 'optimize_gpu') and config.optimize_gpu:
            optimization_results = optimize_gpu_performance()
            results.update(optimization_results)
    
    else:
        logger.info("No GPU available, using CPU computation")
    
    return results


def create_gpu_strategy():
    """
    Create appropriate distribution strategy for GPU usage.
    
    Returns:
        TensorFlow distribution strategy
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            # Multi-GPU strategy
            strategy = tf.distribute.MirroredStrategy()
            logging.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
        elif len(gpus) == 1:
            # Single GPU strategy
            strategy = tf.distribute.get_strategy()  # Default strategy
            logging.info("Using single GPU strategy")
        else:
            # CPU strategy
            strategy = tf.distribute.get_strategy()  # Default strategy
            logging.info("Using CPU strategy")
        
        return strategy
    
    except ImportError:
        logging.error("TensorFlow not available")
        return None
    except Exception as e:
        logging.error(f"Error creating GPU strategy: {e}")
        return None


def monitor_gpu_utilization(duration_seconds: int = 60, interval_seconds: int = 5):
    """
    Monitor GPU utilization over time.
    
    Args:
        duration_seconds: Total monitoring duration
        interval_seconds: Monitoring interval
        
    Returns:
        List[Dict]: GPU utilization data
    """
    import time
    
    logger = logging.getLogger(__name__)
    utilization_data = []
    
    logger.info(f"Monitoring GPU utilization for {duration_seconds} seconds...")
    
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        timestamp = time.time()
        memory_usage = get_gpu_memory_usage()
        
        utilization_data.append({
            'timestamp': timestamp,
            'memory_usage': memory_usage
        })
        
        time.sleep(interval_seconds)
    
    logger.info(f"GPU monitoring completed, {len(utilization_data)} samples collected")
    return utilization_data


def get_recommended_gpu_settings(model_size_mb: float, dataset_size_gb: float) -> Dict[str, Any]:
    """
    Get recommended GPU settings based on model and dataset size.
    
    Args:
        model_size_mb: Estimated model size in MB
        dataset_size_gb: Dataset size in GB
        
    Returns:
        Dict[str, Any]: Recommended settings
    """
    gpu_info = check_gpu_availability()
    
    recommendations = {
        'use_gpu': gpu_info.get('gpu_available', False),
        'enable_memory_growth': True,
        'use_mixed_precision': True,
        'enable_xla': False  # Conservative default
    }
    
    if model_size_mb > 1000:  # Large model
        recommendations.update({
            'batch_size': 4,
            'memory_limit_mb': None,  # Use all available memory
            'enable_xla': True
        })
    elif model_size_mb > 500:  # Medium model
        recommendations.update({
            'batch_size': 8,
            'memory_limit_mb': 8192  # 8GB limit
        })
    else:  # Small model
        recommendations.update({
            'batch_size': 16,
            'memory_limit_mb': 4096  # 4GB limit
        })
    
    # Adjust for large datasets
    if dataset_size_gb > 10:
        recommendations['batch_size'] = max(2, recommendations['batch_size'] // 2)
    
    return recommendations
