# tests/test_utils_memory_utils.py
"""
Test memory management utilities.
"""

import pytest
import logging
from unittest.mock import patch, Mock

from utils.memory_utils import memory_monitor, log_memory_usage, optimize_gpu_memory


class TestMemoryUtils:
    """Test memory management utilities."""
    
    @patch('utils.memory_utils.psutil.Process')
    def test_memory_monitor(self, mock_process):
        """Test memory monitoring context manager."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        with memory_monitor("test_operation"):
            pass
        
        # Should have been called twice (start and end)
        assert mock_process.return_value.memory_info.call_count == 2
    
    @patch('utils.memory_utils.psutil.Process')
    def test_log_memory_usage(self, mock_process, caplog):
        """Test memory usage logging."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        with caplog.at_level(logging.INFO):
            log_memory_usage("Test Operation")
        
        assert "Test Operation - Memory: 100.0MB" in caplog.text
    
    @patch('utils.memory_utils.tf.config.list_physical_devices')
    @patch('utils.memory_utils.tf.config.experimental.set_memory_growth')
    def test_optimize_gpu_memory(self, mock_set_growth, mock_list_devices, caplog):
        """Test GPU memory optimization."""
        # Mock GPU available
        mock_gpu = Mock()
        mock_list_devices.return_value = [mock_gpu]
        
        with caplog.at_level(logging.INFO):
            optimize_gpu_memory()
        
        mock_set_growth.assert_called_once_with(mock_gpu, True)
        assert "GPU memory growth enabled for 1 GPU(s)" in caplog.text
    
    @patch('utils.memory_utils.tf.config.list_physical_devices')
    def test_optimize_gpu_memory_no_gpu(self, mock_list_devices, caplog):
        """Test GPU memory optimization with no GPU."""
        mock_list_devices.return_value = []
        
        with caplog.at_level(logging.INFO):
            optimize_gpu_memory()
        
        assert "No GPU detected, using CPU" in caplog.text