"""
Logging utilities with proper setup and error handling.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name for console output
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(log_level="INFO", log_file="bathymetric_processing.log"):
    """Setup comprehensive logging with file and console handlers."""
    
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create full log file path
    log_file_path = log_dir / log_file
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Test write to ensure file handler works
        test_logger = logging.getLogger('setup_test')
        test_logger.info("Logging system initialized successfully")
        
    except Exception as e:
        print(f"Warning: Could not create file handler for {log_file_path}: {e}")
    
    # Console handler
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    except Exception as e:
        print(f"Warning: Could not create console handler: {e}")
    
    # Configure specific loggers to prevent spam
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Log initialization success
    logger = logging.getLogger('logging_utils')
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file_path}")
    
    return str(log_file_path)


def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_system_info():
    """Log basic system information."""
    logger = logging.getLogger('system_info')
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Log environment variables that might affect processing
    env_vars = ['GDAL_DATA', 'PROJ_LIB', 'CUDA_VISIBLE_DEVICES']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        logger.debug(f"{var}: {value}")


def configure_tensorflow_logging():
    """Configure TensorFlow logging to reduce noise."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        # Disable TensorFlow warnings
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        
        logger = logging.getLogger('tensorflow_config')
        logger.debug("TensorFlow logging configured")
        
    except ImportError:
        logger = logging.getLogger('tensorflow_config')
        logger.warning("TensorFlow not available for logging configuration")


def create_timestamped_log_file(base_name="processing"):
    """Create a timestamped log file name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.log"


# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()