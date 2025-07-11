# tests/test_utils_logging_utils.py
"""
Test logging utilities.
"""

import pytest
import logging
import tempfile
from pathlib import Path

from utils.logging_utils import setup_logging, ColoredFormatter


class TestLoggingUtils:
    """Test logging utilities."""
    
    def test_colored_formatter(self):
        """Test colored formatter functionality."""
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
        
        # Create a log record
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert 'INFO' in formatted
        assert 'Test message' in formatted
    
    def test_setup_logging(self, temp_dir):
        """Test logging setup."""
        log_file = "test_log.log"
        
        setup_logging(log_level="DEBUG", log_file=log_file)
        
        # Test that logger is configured
        logger = logging.getLogger('test_logger')
        logger.info("Test log message")
        
        # Check that log file was created
        log_path = Path("logs") / log_file
        assert log_path.exists()
        
        # Check log content
        with open(log_path, 'r') as f:
            content = f.read()
            assert "Test log message" in content