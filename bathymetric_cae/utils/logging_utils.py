"""
Logging Utilities Module

This module provides enhanced logging capabilities with colored output,
file logging, and proper formatting for the bathymetric CAE pipeline.

Author: Bathymetric CAE Team
License: MIT
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    
    Provides color-coded log levels for better readability in terminal output.
    Colors are applied based on log level severity.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: Log record to format
            
        Returns:
            str: Formatted log message with colors
        """
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "bathymetric_processing.log",
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
    colored_output: bool = True
) -> logging.Logger:
    """
    Enhanced logging setup with colored output and file logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Name of log file
        log_dir: Directory for log files
        console_output: Enable console output
        file_output: Enable file output
        colored_output: Enable colored console output
        
    Returns:
        logging.Logger: Configured root logger
        
    Raises:
        ValueError: If log_level is invalid
    """
    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    root_logger.setLevel(numeric_level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    
    handlers = []
    
    # File handler
    if file_output:
        file_handler = logging.FileHandler(log_path / log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if colored_output:
            colored_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(simple_formatter)
        
        handlers.append(console_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Test logging setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, "
               f"Console: {console_output}, File: {file_output}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def log_system_info():
    """Log system information for debugging purposes."""
    import platform
    import sys
    
    logger = get_logger(__name__)
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Try to log TensorFlow version
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        if tf.config.list_physical_devices('GPU'):
            for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
                logger.info(f"GPU {i}: {gpu}")
    except ImportError:
        logger.warning("TensorFlow not available")
    
    # Try to log GDAL version
    try:
        from osgeo import gdal
        logger.info(f"GDAL version: {gdal.__version__}")
    except ImportError:
        logger.warning("GDAL not available")
    
    logger.info("=" * 30)


def log_configuration(config) -> None:
    """
    Log configuration parameters.
    
    Args:
        config: Configuration object to log
    """
    logger = get_logger(__name__)
    
    logger.info("=== Configuration ===")
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 22)


class ContextLogger:
    """
    Context manager for logging operations with timing.
    
    Logs the start and end of operations with execution time.
    """
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize context logger.
        
        Args:
            operation_name: Name of the operation being logged
            logger: Logger instance (uses default if None)
        """
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Enter context and log start."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and log completion."""
        import time
        
        if self.start_time is not None:
            duration = time.time() - self.start_time
            
            if exc_type is None:
                self.logger.info(f"Completed {self.operation_name} in {duration:.2f} seconds")
            else:
                self.logger.error(f"Failed {self.operation_name} after {duration:.2f} seconds: {exc_val}")
        
        return False  # Don't suppress exceptions


def log_function_call(func):
    """
    Decorator to log function calls with arguments and execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with logging
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        args_str = ', '.join([str(arg) for arg in args[:3]])  # Limit to first 3 args
        if len(args) > 3:
            args_str += '...'
        
        kwargs_str = ', '.join([f"{k}={v}" for k, v in list(kwargs.items())[:3]])
        if len(kwargs) > 3:
            kwargs_str += '...'
        
        params = ', '.join(filter(None, [args_str, kwargs_str]))
        logger.debug(f"Calling {func.__name__}({params})")
        
        # Execute function with timing
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {duration:.3f} seconds")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {duration:.3f} seconds: {e}")
            raise
    
    return wrapper


def configure_tensorflow_logging():
    """Configure TensorFlow logging to reduce verbosity."""
    import os
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        pass


def silence_warnings():
    """Silence common warnings that don't affect functionality."""
    import warnings
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Configure TensorFlow logging
    configure_tensorflow_logging()
