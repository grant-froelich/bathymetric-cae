# utils/__init__.py
"""Utility modules for bathymetric CAE package."""

from .logging_utils import (
    setup_logging, get_logger, log_system_info, log_configuration,
    ContextLogger, log_function_call, configure_tensorflow_logging, silence_warnings
)
from .memory_utils import (
    get_memory_info, get_gpu_memory_info, memory_monitor, force_garbage_collection,
    check_memory_threshold, optimize_memory_usage, configure_tensorflow_memory,
    MemoryProfiler, get_optimal_batch_size, memory_efficient_data_loading,
    cleanup_tensorflow_session, set_memory_warnings
)
from .gpu_utils import (
    check_gpu_availability, configure_gpu_memory, optimize_gpu_performance,
    disable_gpu, get_gpu_memory_usage, log_gpu_info, setup_gpu_environment,
    create_gpu_strategy, monitor_gpu_utilization, get_recommended_gpu_settings
)
from .file_utils import (
    get_valid_files, validate_paths, ensure_directory, get_file_info,
    calculate_file_hash, safe_copy_file, clean_filename, get_unique_filename,
    batch_rename_files, organize_files_by_extension, find_duplicate_files,
    cleanup_empty_directories, FileManager, monitor_directory
)

__all__ = [
    # Logging utilities
    'setup_logging', 'get_logger', 'log_system_info', 'log_configuration',
    'ContextLogger', 'log_function_call', 'configure_tensorflow_logging', 'silence_warnings',
    
    # Memory utilities
    'get_memory_info', 'get_gpu_memory_info', 'memory_monitor', 'force_garbage_collection',
    'check_memory_threshold', 'optimize_memory_usage', 'configure_tensorflow_memory',
    'MemoryProfiler', 'get_optimal_batch_size', 'memory_efficient_data_loading',
    'cleanup_tensorflow_session', 'set_memory_warnings',
    
    # GPU utilities
    'check_gpu_availability', 'configure_gpu_memory', 'optimize_gpu_performance',
    'disable_gpu', 'get_gpu_memory_usage', 'log_gpu_info', 'setup_gpu_environment',
    'create_gpu_strategy', 'monitor_gpu_utilization', 'get_recommended_gpu_settings',
    
    # File utilities
    'get_valid_files', 'validate_paths', 'ensure_directory', 'get_file_info',
    'calculate_file_hash', 'safe_copy_file', 'clean_filename', 'get_unique_filename',
    'batch_rename_files', 'organize_files_by_extension', 'find_duplicate_files',
    'cleanup_empty_directories', 'FileManager', 'monitor_directory'
]

# ================================================================================

