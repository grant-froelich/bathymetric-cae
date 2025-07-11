# tests/test_performance.py
"""
Performance and stress tests.
"""

import pytest
import time
import numpy as np
from pathlib import Path

from config.config import Config
from models.architectures import create_model_variant
from core.quality_metrics import BathymetricQualityMetrics


class TestPerformance:
    """Performance and stress tests."""
    
    def test_model_inference_speed(self):
        """Test model inference speed."""
        config = Config(grid_size=128, base_filters=16, depth=3)
        model = create_model_variant(config, 'lightweight', (128, 128, 1))
        
        # Prepare test data
        test_data = np.random.random((10, 128, 128, 1)).astype(np.float32)
        
        # Time inference
        start_time = time.time()
        predictions = model.predict(test_data, verbose=0)
        inference_time = time.time() - start_time
        
        # Should complete inference in reasonable time
        assert inference_time < 30  # 30 seconds for 10 samples
        assert predictions.shape == test_data.shape
    
    def test_quality_metrics_performance(self):
        """Test quality metrics calculation performance."""
        # Large data for stress testing
        large_data = np.random.random((512, 512)).astype(np.float32)
        
        metrics = BathymetricQualityMetrics()
        
        # Time quality metric calculations
        start_time = time.time()
        
        roughness = metrics.calculate_roughness(large_data)
        feature_preservation = metrics.calculate_feature_preservation(large_data, large_data)
        consistency = metrics.calculate_depth_consistency(large_data)
        compliance = metrics.calculate_hydrographic_standards_compliance(large_data)
        
        calculation_time = time.time() - start_time
        
        # Should complete calculations in reasonable time
        assert calculation_time < 10  # 10 seconds for 512x512 data
        assert all(0 <= metric <= 1 for metric in [roughness, feature_preservation, 
                                                   consistency, compliance] if metric is not None)
    
    def test_memory_usage_with_large_datasets(self):
        """Test memory usage with large datasets."""
        import psutil
        process = psutil.Process()
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        large_dataset = [np.random.random((256, 256)).astype(np.float32) for _ in range(50)]
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del large_dataset
        import gc
        gc.collect()
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should be released after cleanup
        memory_increase = peak_memory - initial_memory
        memory_released = peak_memory - final_memory
        
        assert memory_increase > 0  # Should use memory for large dataset
        assert memory_released > memory_increase * 0.5  # Should release most memory