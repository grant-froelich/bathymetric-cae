"""
Logging utilities for Enhanced Bathymetric CAE Processing.

This module provides enhanced logging setup with colored output,
file rotation, and structured logging capabilities.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Store original levelname
        original_levelname = record.levelname
        
        # Add color to levelname
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the record
        formatted = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"performance.{name}")
        self.timings = {}
        self.counters = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.timings[operation] = datetime.datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and log duration."""
        if operation not in self.timings:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return 0.0
        
        start_time = self.timings[operation]
        duration = (datetime.datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"{operation} completed in {duration:.3f} seconds")
        del self.timings[operation]
        
        return duration
    
    def increment_counter(self, counter: str, value: int = 1):
        """Increment a counter."""
        if counter not in self.counters:
            self.counters[counter] = 0
        self.counters[counter] += value
    
    def log_counter(self, counter: str):
        """Log current counter value."""
        if counter in self.counters:
            self.logger.info(f"{counter}: {self.counters[counter]}")
        else:
            self.logger.warning(f"Counter '{counter}' does not exist")
    
    def log_all_counters(self):
        """Log all counter values."""
        if self.counters:
            self.logger.info("Performance counters:")
            for counter, value in self.counters.items():
                self.logger.info(f"  {counter}: {value}")
        else:
            self.logger.info("No performance counters recorded")
    
    def reset_counters(self):
        """Reset all counters."""
        self.counters.clear()


class ProcessingProgressLogger:
    """Logger for tracking processing progress."""
    
    def __init__(self, total_items: int, name: str = "processing"):
        self.total_items = total_items
        self.current_item = 0
        self.logger = logging.getLogger(f"progress.{name}")
        self.start_time = datetime.datetime.now()
        self.last_log_time = self.start_time
        self.log_interval = 10  # Log every 10 seconds
    
    def update(self, items_completed: int = 1, message: Optional[str] = None):
        """Update progress and log if needed."""
        self.current_item += items_completed
        current_time = datetime.datetime.now()
        
        # Log progress at intervals or on completion
        time_since_last_log = (current_time - self.last_log_time).total_seconds()
        
        if (time_since_last_log >= self.log_interval or 
            self.current_item >= self.total_items or
            self.current_item % max(1, self.total_items // 10) == 0):
            
            self._log_progress(message)
            self.last_log_time = current_time
    
    def _log_progress(self, message: Optional[str] = None):
        """Log current progress."""
        if self.total_items > 0:
            percentage = (self.current_item / self.total_items) * 100
            
            # Calculate ETA
            elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
            if self.current_item > 0:
                eta_seconds = (elapsed / self.current_item) * (self.total_items - self.current_item)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta = "unknown"
            
            log_msg = f"Progress: {self.current_item}/{self.total_items} ({percentage:.1f}%) - ETA: {eta}"
            if message:
                log_msg += f" - {message}"
            
            self.logger.info(log_msg)
        else:
            log_msg = f"Processed: {self.current_item} items"
            if message:
                log_msg += f" - {message}"
            self.logger.info(log_msg)


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_json: bool = False,
                 max_file_size: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 5) -> Dict[str, logging.Handler]:
    """Enhanced logging setup with multiple handlers."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    handlers = {}
    
    # Console handler with colors
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        root_logger.addHandler(console_handler)
        handlers['console'] = console_handler
    
    # File handler with rotation
    if log_file is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/bathymetric_processing_{timestamp}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    
    # Choose formatter based on JSON setting
    if enable_json:
        file_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s'
        )
    
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    handlers['file'] = file_handler
    
    # Error file handler (errors only)
    error_log_file = log_file.replace('.log', '_errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    handlers['error'] = error_handler
    
    # Performance log handler
    perf_log_file = log_file.replace('.log', '_performance.log')
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(file_formatter)
    
    # Add filter to only log performance messages
    class PerformanceFilter(logging.Filter):
        def filter(self, record):
            return record.name.startswith('performance.')
    
    perf_handler.addFilter(PerformanceFilter())
    root_logger.addHandler(perf_handler)
    handlers['performance'] = perf_handler
    
    # Progress log handler
    progress_handler = logging.StreamHandler(sys.stdout)
    progress_handler.setLevel(logging.INFO)
    progress_formatter = logging.Formatter('%(message)s')
    progress_handler.setFormatter(progress_formatter)
    
    # Add filter to only log progress messages
    class ProgressFilter(logging.Filter):
        def filter(self, record):
            return record.name.startswith('progress.')
    
    progress_handler.addFilter(ProgressFilter())
    root_logger.addHandler(progress_handler)
    handlers['progress'] = progress_handler
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    logger.info(f"Log handlers: {list(handlers.keys())}")
    
    return handlers


def get_logger(name: str, extra_fields: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Get a logger with optional extra fields."""
    logger = logging.getLogger(name)
    
    if extra_fields:
        # Create adapter to add extra fields
        logger = logging.LoggerAdapter(logger, extra_fields)
    
    return logger


def log_system_info():
    """Log system and environment information."""
    import platform
    import psutil
    import os
    
    logger = logging.getLogger("system_info")
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU cores: {psutil.cpu_count()}")
    logger.info(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU information
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"  GPUs: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                logger.info(f"    GPU {i}: {gpu.name}")
        else:
            logger.info("  GPUs: None detected")
    except ImportError:
        logger.info("  GPUs: TensorFlow not available")
    
    # Environment variables
    relevant_env_vars = ['CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL', 'OMP_NUM_THREADS']
    env_info = []
    for var in relevant_env_vars:
        value = os.environ.get(var)
        if value:
            env_info.append(f"{var}={value}")
    
    if env_info:
        logger.info(f"  Environment: {', '.join(env_info)}")


def setup_file_logging(log_file: str, level: str = "INFO") -> logging.FileHandler:
    """Setup standalone file logging."""
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    handler.setFormatter(formatter)
    
    return handler


def create_memory_usage_logger() -> logging.Logger:
    """Create logger for memory usage tracking."""
    logger = logging.getLogger("memory_usage")
    
    def log_memory_usage(operation: str = ""):
        """Log current memory usage."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        rss_mb = memory_info.rss / (1024 * 1024)
        vms_mb = memory_info.vms / (1024 * 1024)
        
        message = f"Memory usage: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB"
        if operation:
            message = f"{operation} - {message}"
        
        logger.info(message)
    
    # Add convenience method
    logger.log_usage = log_memory_usage
    
    return logger


# Convenience functions
def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger instance."""
    return PerformanceLogger(name)


def get_progress_logger(total_items: int, name: str = "processing") -> ProcessingProgressLogger:
    """Get a progress logger instance."""
    return ProcessingProgressLogger(total_items, name)
