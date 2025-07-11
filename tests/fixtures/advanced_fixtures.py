# tests/fixtures/advanced_fixtures.py

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from contextlib import contextmanager
import tensorflow as tf

from config.config import Config
from core.enums import SeafloorType
from tests.factories.data_factory import BathymetricDataFactory, ConfigurationFactory
from tests.utils.performance_monitor import PerformanceMonitor


class ModelTestFixture:
    """Advanced fixture for model testing."""
    
    def __init__(self):
        self.models = {}
        self.test_data = {}
        self.performance_data = {}
    
    def create_test_model(self, model_type: str, input_shape=(64, 64, 1)):
        """Create a test model of specified type."""
        if model_type in self.models:
            return self.models[model_type]
        
        # Create minimal functional model for testing
        inputs = tf.keras.Input(shape=input_shape)
        
        if model_type == 'simple':
            # Simple autoencoder
            x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(x)
            encoded = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            
            x = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(encoded)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            outputs = tf.keras.layers.Conv2D(input_shape[-1], 3, activation='sigmoid', padding='same')(x)
            
            model = tf.keras.Model(inputs, outputs)
            
        elif model_type == 'uncertainty':
            # Model with dual outputs
            x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            
            # Reshape back to spatial
            x = tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation='relu')(x)
            x = tf.keras.layers.Reshape(input_shape[:2] + (1,))(x)
            
            depth_output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='depth')(x)
            uncertainty_output = tf.keras.layers.Conv2D(1, 1, activation='softplus', name='uncertainty')(x)
            
            model = tf.keras.Model(inputs, [depth_output, uncertainty_output])
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.models[model_type] = model
        return model
    
    def create_test_data(self, data_type: str, shape=(10, 64, 64, 1)):
        """Create test data of specified type."""
        if data_type in self.test_data:
            return self.test_data[data_type]
        
        if data_type == 'training':
            # Realistic training data
            data = []
            for i in range(shape[0]):
                factory = BathymetricDataFactory(
                    width=shape[2], 
                    height=shape[1],
                    seed=42 + i
                )
                sample = factory.build()
                normalized_depth = (sample['depth_data'] + 6000) / 6000  # Normalize to 0-1
                data.append(normalized_depth)
            
            self.test_data[data_type] = np.stack(data).reshape(shape)
            
        elif data_type == 'validation':
            # Simple validation data
            np.random.seed(123)
            self.test_data[data_type] = np.random.random(shape).astype(np.float32)
            
        elif data_type == 'noisy':
            # Data with noise for testing robustness
            base_data = self.create_test_data('training', shape)
            noise = np.random.normal(0, 0.1, shape)
            self.test_data[data_type] = np.clip(base_data + noise, 0, 1)
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return self.test_data[data_type]
    
    def benchmark_model_performance(self, model, test_data, operation='inference'):
        """Benchmark model performance."""
        with PerformanceMonitor().monitor() as monitor:
            if operation == 'inference':
                predictions = model.predict(test_data, verbose=0)
            elif operation == 'training':
                model.fit(test_data, test_data, epochs=1, verbose=0)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        results = monitor.get_results()
        self.performance_data[f"{model.name}_{operation}"] = results
        return results
    
    def cleanup(self):
        """Clean up test models and data."""
        for model in self.models.values():
            del model
        
        self.models.clear()
        self.test_data.clear()
        self.performance_data.clear()
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()


class DataProcessingFixture:
    """Advanced fixture for data processing testing."""
    
    def __init__(self):
        self.temp_dirs = []
        self.test_files = []
        self.mock_datasets = {}
    
    def create_temp_workspace(self) -> Path:
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        workspace = Path(temp_dir)
        self.temp_dirs.append(workspace)
        
        # Create standard directory structure
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "logs").mkdir()
        (workspace / "plots").mkdir()
        (workspace / "expert_reviews").mkdir()
        
        return workspace
    
    def create_mock_bathymetric_files(self, workspace: Path, num_files: int = 3):
        """Create mock bathymetric files for testing."""
        from tests.factories.data_factory import TestFileFactory
        
        input_dir = workspace / "input"
        seafloor_types = [SeafloorType.SHALLOW_COASTAL, SeafloorType.DEEP_OCEAN, SeafloorType.SEAMOUNT]
        
        for i in range(num_files):
            seafloor_type = seafloor_types[i % len(seafloor_types)]
            
            if i % 2 == 0:
                # Create BAG file
                file_path = input_dir / f"test_{seafloor_type.value}_{i}.bag"
                TestFileFactory.create_test_bag_file(file_path, seafloor_type)
            else:
                # Create GeoTIFF file
                file_path = input_dir / f"test_{seafloor_type.value}_{i}.tif"
                TestFileFactory.create_test_geotiff(file_path, seafloor_type)
            
            self.test_files.append(file_path)
        
        return self.test_files
    
    @contextmanager
    def mock_gdal_environment(self):
        """Context manager for mocking GDAL environment."""
        from tests.utils.mock_gdal import mock_gdal_open, mock_gdal_get_driver_by_name
        
        with patch('processing.data_processor.gdal.Open', side_effect=mock_gdal_open), \
             patch('processing.data_processor.gdal.GetDriverByName', side_effect=mock_gdal_get_driver_by_name):
            yield
    
    def create_test_configuration(self, **overrides) -> Config:
        """Create test configuration with overrides."""
        config_data = ConfigurationFactory.build()
        config_data.update(overrides)
        
        # Convert to Config object
        config = Config(**config_data)
        return config
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        self.temp_dirs.clear()
        self.test_files.clear()
        self.mock_datasets.clear()


class PipelineTestFixture:
    """Advanced fixture for pipeline testing."""
    
    def __init__(self):
        self.model_fixture = ModelTestFixture()
        self.data_fixture = DataProcessingFixture()
        self.pipeline_configs = {}
        self.pipeline_results = {}
    
    def setup_complete_test_environment(self, test_name: str):
        """Setup complete test environment for pipeline testing."""
        workspace = self.data_fixture.create_temp_workspace()
        
        # Create test files
        test_files = self.data_fixture.create_mock_bathymetric_files(workspace, 3)
        
        # Create test configuration
        config = self.data_fixture.create_test_configuration(
            input_folder=str(workspace / "input"),
            output_folder=str(workspace / "output"),
            epochs=2,
            batch_size=1,
            ensemble_size=1
        )
        
        # Create test models
        test_model = self.model_fixture.create_test_model('simple')
        
        environment = {
            'workspace': workspace,
            'config': config,
            'test_files': test_files,
            'test_model': test_model
        }
        
        self.pipeline_configs[test_name] = environment
        return environment
    
    @contextmanager
    def pipeline_test_context(self, test_name: str):
        """Context manager for pipeline testing."""
        environment = self.setup_complete_test_environment(test_name)
        
        try:
            with self.data_fixture.mock_gdal_environment():
                yield environment
        finally:
            # Store results for analysis
            if test_name not in self.pipeline_results:
                self.pipeline_results[test_name] = {}
            
            # Capture any output files
            output_dir = environment['workspace'] / "output"
            if output_dir.exists():
                output_files = list(output_dir.glob("*"))
                self.pipeline_results[test_name]['output_files'] = output_files
    
    def assert_pipeline_outputs(self, test_name: str, expected_files: int = None):
        """Assert pipeline produced expected outputs."""
        if test_name not in self.pipeline_results:
            raise AssertionError(f"No results found for test: {test_name}")
        
        results = self.pipeline_results[test_name]
        
        if expected_files is not None:
            output_files = results.get('output_files', [])
            assert len(output_files) >= expected_files, \
                f"Expected at least {expected_files} output files, got {len(output_files)}"
    
    def cleanup(self):
        """Clean up all test fixtures."""
        self.model_fixture.cleanup()
        self.data_fixture.cleanup()
        self.pipeline_configs.clear()
        self.pipeline_results.clear()


class PerformanceTestFixture:
    """Advanced fixture for performance testing."""
    
    def __init__(self):
        self.monitors = {}
        self.benchmarks = {}
        self.thresholds = {
            'inference_time': 5.0,  # seconds
            'memory_usage': 1000,   # MB
            'training_time': 30.0   # seconds per epoch
        }
    
    @contextmanager
    def performance_monitor(self, test_name: str):
        """Monitor performance during test execution."""
        monitor = PerformanceMonitor()
        
        with monitor.monitor() as m:
            yield m
        
        results = monitor.get_results()
        self.monitors[test_name] = results
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function execution."""
        import time
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        benchmark_data = {
            'execution_time': execution_time,
            'function_name': func.__name__,
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        }
        
        function_key = f"{func.__module__}.{func.__name__}"
        if function_key not in self.benchmarks:
            self.benchmarks[function_key] = []
        
        self.benchmarks[function_key].append(benchmark_data)
        
        return result, execution_time
    
    def assert_performance_thresholds(self, test_name: str):
        """Assert performance meets defined thresholds."""
        if test_name not in self.monitors:
            raise AssertionError(f"No performance data for test: {test_name}")
        
        results = self.monitors[test_name]
        
        if 'execution_time' in results and results['execution_time']:
            assert results['execution_time'] < self.thresholds['inference_time'], \
                f"Execution time {results['execution_time']:.2f}s exceeds threshold {self.thresholds['inference_time']}s"
        
        if 