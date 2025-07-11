# tests/utils/test_helpers.py

import numpy as np
import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import Mock
import tensorflow as tf


class TestHelpers:
    """Advanced test helper utilities."""
    
    @staticmethod
    def create_mock_model(input_shape=(64, 64, 1), output_shape=None):
        """Create a mock TensorFlow model for testing."""
        if output_shape is None:
            output_shape = input_shape
            
        model = Mock()
        model.input_shape = (None,) + input_shape
        model.predict.return_value = np.random.random((1,) + output_shape)
        model.fit.return_value = Mock()
        model.evaluate.return_value = [0.1, 0.05]  # loss, mae
        model.count_params.return_value = 10000
        
        return model
    
    @staticmethod
    def assert_array_properties(array, expected_shape=None, expected_dtype=None, 
                               expected_range=None, no_nan=True, no_inf=True):
        """Assert multiple array properties at once."""
        if expected_shape:
            assert array.shape == expected_shape
        if expected_dtype:
            assert array.dtype == expected_dtype
        if expected_range:
            assert expected_range[0] <= array.min() <= array.max() <= expected_range[1]
        if no_nan:
            assert not np.isnan(array).any()
        if no_inf:
            assert not np.isinf(array).any()
    
    @staticmethod
    def create_gradient_data(shape, direction='horizontal'):
        """Create test data with known gradients."""
        if direction == 'horizontal':
            return np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1))
        elif direction == 'vertical':
            return np.tile(np.linspace(0, 1, shape[0])[:, None], (1, shape[1]))
        elif direction == 'radial':
            center = (shape[0]//2, shape[1]//2)
            y, x = np.ogrid[:shape[0], :shape[1]]
            return np.sqrt((x - center[1])**2 + (y - center[0])**2) / max(shape)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    @staticmethod
    @contextmanager
    def temporary_config_override(config, **overrides):
        """Temporarily override configuration values."""
        original_values = {}
        for key, value in overrides.items():
            if hasattr(config, key):
                original_values[key] = getattr(config, key)
                setattr(config, key, value)
        
        try:
            yield config
        finally:
            for key, value in original_values.items():
                setattr(config, key, value)
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure function execution time."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def compare_arrays_with_tolerance(arr1, arr2, rtol=1e-5, atol=1e-8):
        """Compare arrays with relative and absolute tolerance."""
        return np.allclose(arr1, arr2, rtol=rtol, atol=atol)