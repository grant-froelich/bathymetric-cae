# tests/fixtures/mock_fixtures.py

import pytest
import numpy as np
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import tensorflow as tf

from tests.utils.mock_gdal import MockGDALDataset, MockGDALDriver


class MockGDALFixture:
    """Comprehensive GDAL mocking fixture."""
    
    def __init__(self):
        self.mock_datasets = {}
        self.mock_drivers = {}
        self.file_registry = {}
    
    def register_file(self, filename: str, data, file_type: str = "tiff", 
                     geotransform=None, projection=None, metadata=None):
        """Register a file with mock data."""
        if isinstance(data, np.ndarray):
            data = [data]  # Convert single array to list
        
        mock_dataset = MockGDALDataset(
            data=data,
            geotransform=geotransform or (0, 1, 0, 0, 0, -1),
            projection=projection or 'EPSG:4326',
            metadata=metadata or {}
        )
        
        self.mock_datasets[str(filename)] = mock_dataset
        self.file_registry[str(filename)] = {
            'type': file_type,
            'bands': len(data),
            'shape': data[0].shape if data else (0, 0)
        }
        
        return mock_dataset
    
    def create_bathymetric_file_mock(self, filename: str, seafloor_type: str = "continental_shelf"):
        """Create mock bathymetric file with realistic data."""
        from tests.factories.data_factory import BathymetricDataFactory
        from core.enums import SeafloorType
        
        # Generate realistic data
        factory = BathymetricDataFactory(
            width=64, height=64,
            seafloor_type=SeafloorType(seafloor_type)
        )
        test_data = factory.build()
        
        if filename.endswith('.bag'):
            # BAG file with depth and uncertainty
            data = [test_data['depth_data'], test_data['uncertainty_data']]
            file_type = "bag"
        else:
            # Single band file
            data = [test_data['depth_data']]
            file_type = "tiff"
        
        return self.register_file(filename, data, file_type,
                                metadata={'SEAFLOOR_TYPE': seafloor_type})
    
    @contextmanager
    def mock_gdal_environment(self):
        """Context manager for comprehensive GDAL mocking."""
        def mock_open(filename):
            filename_str = str(filename)
            if filename_str in self.mock_datasets:
                return self.mock_datasets[filename_str]
            else:
                # Return None for unknown files (GDAL behavior)
                return None
        
        def mock_get_driver(driver_name):
            if driver_name not in self.mock_drivers:
                self.mock_drivers[driver_name] = MockGDALDriver(driver_name)
            return self.mock_drivers[driver_name]
        
        with patch('processing.data_processor.gdal.Open', side_effect=mock_open), \
             patch('processing.data_processor.gdal.GetDriverByName', side_effect=mock_get_driver), \
             patch('processing.pipeline.gdal.Open', side_effect=mock_open), \
             patch('processing.pipeline.gdal.GetDriverByName', side_effect=mock_get_driver):
            yield self
    
    def get_file_info(self, filename: str):
        """Get information about registered mock file."""
        return self.file_registry.get(str(filename))
    
    def list_registered_files(self):
        """List all registered mock files."""
        return list(self.file_registry.keys())
    
    def cleanup(self):
        """Clean up mock data."""
        self.mock_datasets.clear()
        self.mock_drivers.clear()
        self.file_registry.clear()


class MockTensorFlowFixture:
    """Comprehensive TensorFlow mocking fixture."""
    
    def __init__(self):
        self.mock_models = {}
        self.training_histories = {}
        self.prediction_results = {}
    
    def create_mock_model(self, model_name: str, input_shape=(64, 64, 1), 
                         output_shape=None, model_type="single_output"):
        """Create a comprehensive mock TensorFlow model."""
        if output_shape is None:
            output_shape = input_shape
        
        mock_model = Mock()
        mock_model.name = model_name
        mock_model.input_shape = (None,) + input_shape
        
        # Mock model methods
        if model_type == "single_output":
            mock_model.predict.return_value = np.random.random((1,) + output_shape).astype(np.float32)
        elif model_type == "dual_output":
            # For uncertainty models with dual outputs
            depth_output = np.random.random((1,) + output_shape).astype(np.float32)
            uncertainty_output = np.random.random((1,) + output_shape).astype(np.float32) * 0.5
            mock_model.predict.return_value = [depth_output, uncertainty_output]
        
        # Mock training history
        mock_history = Mock()
        mock_history.history = {
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3],
            'mae': [0.5, 0.4, 0.3, 0.2, 0.1],
            'val_mae': [0.6, 0.5, 0.4, 0.3, 0.2]
        }
        mock_model.fit.return_value = mock_history
        
        # Mock other model properties
        mock_model.evaluate.return_value = [0.2, 0.1]  # [loss, mae]
        mock_model.count_params.return_value = 50000
        mock_model.summary.return_value = None
        mock_model.compile.return_value = None
        mock_model.save.return_value = None
        
        # Mock layer access
        mock_model.layers = [Mock() for _ in range(10)]  # Mock 10 layers
        
        self.mock_models[model_name] = mock_model
        self.training_histories[model_name] = mock_history
        
        return mock_model
    
    def create_mock_ensemble(self, ensemble_size: int = 3, input_shape=(64, 64, 1)):
        """Create mock ensemble of models."""
        ensemble_models = []
        
        for i in range(ensemble_size):
            model_name = f"ensemble_model_{i}"
            mock_model = self.create_mock_model(model_name, input_shape)
            ensemble_models.append(mock_model)
        
        return ensemble_models
    
    @contextmanager
    def mock_tensorflow_environment(self):
        """Context manager for TensorFlow mocking."""
        
        def mock_model_load(filepath, compile=True):
            # Return a mock model when loading
            model_name = Path(filepath).stem
            if model_name in self.mock_models:
                return self.mock_models[model_name]
            else:
                # Create new mock model
                return self.create_mock_model(model_name)
        
        with patch('tensorflow.keras.models.load_model', side_effect=mock_model_load), \
             patch('models.architectures.tf.keras.Model') as mock_model_class, \
             patch('tensorflow.keras.backend.clear_session'):
            
            # Configure the Model class mock
            mock_model_class.side_effect = lambda inputs, outputs, name=None: \
                self.create_mock_model(name or "test_model")
            
            yield self
    
    def set_prediction_result(self, model_name: str, result):
        """Set specific prediction result for a model."""
        if model_name in self.mock_models:
            self.mock_models[model_name].predict.return_value = result
            self.prediction_results[model_name] = result
    
    def set_training_history(self, model_name: str, history_dict):
        """Set specific training history for a model."""
        if model_name in self.mock_models:
            mock_history = Mock()
            mock_history.history = history_dict
            self.mock_models[model_name].fit.return_value = mock_history
            self.training_histories[model_name] = mock_history
    
    def cleanup(self):
        """Clean up mock models and data."""
        self.mock_models.clear()
        self.training_histories.clear()
        self.prediction_results.clear()


class MockDatabaseFixture:
    """Mock database fixture for expert review system testing."""
    
    def __init__(self):
        self.temp_db_files = []
        self.mock_connections = {}
        self.test_data = {
            'flagged_regions': [],
            'expert_reviews': []
        }
    
    def create_temp_database(self) -> str:
        """Create temporary SQLite database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        db_path = temp_file.name
        self.temp_db_files.append(db_path)
        
        # Initialize database schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE flagged_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                x_start INTEGER,
                y_start INTEGER,
                x_end INTEGER,
                y_end INTEGER,
                flag_type TEXT,
                confidence REAL,
                reviewed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE expert_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                region_id TEXT,
                expert_rating INTEGER,
                quality_score REAL,
                comments TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        return db_path
    
    def populate_test_data(self, db_path: str, num_flagged: int = 5, num_reviews: int = 2):
        """Populate database with test data."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Add flagged regions
        for i in range(num_flagged):
            cursor.execute('''
                INSERT INTO flagged_regions 
                (filename, x_start, y_start, x_end, y_end, flag_type, confidence, reviewed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f'test_file_{i}.bag',
                0, 0, 100, 100,
                'low_quality',
                0.8,
                i < num_reviews  # First num_reviews are marked as reviewed
            ))
        
        # Add expert reviews
        for i in range(num_reviews):
            cursor.execute('''
                INSERT INTO expert_reviews
                (filename, region_id, expert_rating, quality_score, comments)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                f'test_file_{i}.bag',
                f'region_{i}',
                3,  # Rating 1-5
                0.75,
                f'Test review comment {i}'
            ))
        
        conn.commit()
        conn.close()
    
    @contextmanager
    def mock_database_environment(self, populate_data: bool = True):
        """Context manager for database mocking."""
        db_path = self.create_temp_database()
        
        if populate_data:
            self.populate_test_data(db_path)
        
        try:
            # Mock the database path in expert review system
            with patch('review.expert_system.ExpertReviewSystem.__init__') as mock_init:
                mock_init.return_value = None  # Don't call real __init__
                
                # Create mock expert review system
                mock_system = Mock()
                mock_system.db_path = db_path
                
                # Setup mock methods to use real database
                def mock_flag_for_review(filename, region, flag_type, confidence):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO flagged_regions 
                        (filename, x_start, y_start, x_end, y_end, flag_type, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (filename, region[0], region[1], region[2], region[3], flag_type, confidence))
                    conn.commit()
                    conn.close()
                
                def mock_get_pending_reviews():
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM flagged_regions WHERE reviewed = FALSE')
                    results = cursor.fetchall()
                    conn.close()
                    
                    columns = ['id', 'filename', 'x_start', 'y_start', 'x_end', 'y_end', 
                              'flag_type', 'confidence', 'reviewed']
                    return [dict(zip(columns, row)) for row in results]
                
                mock_system.flag_for_review = mock_flag_for_review
                mock_system.get_pending_reviews = mock_get_pending_reviews
                
                yield mock_system, db_path
                
        finally:
            # Cleanup is handled in cleanup method
            pass
    
    def get_database_contents(self, db_path: str):
        """Get current database contents for inspection."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        contents = {}
        
        # Get flagged regions
        cursor.execute('SELECT * FROM flagged_regions')
        contents['flagged_regions'] = cursor.fetchall()
        
        # Get expert reviews
        cursor.execute('SELECT * FROM expert_reviews')
        contents['expert_reviews'] = cursor.fetchall()
        
        conn.close()
        return contents
    
    def cleanup(self):
        """Clean up temporary database files."""
        for db_file in self.temp_db_files:
            try:
                Path(db_file).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors
        
        self.temp_db_files.clear()
        self.mock_connections.clear()
        self.test_data.clear()


class IntegratedMockFixture:
    """Integrated fixture combining all mock capabilities."""
    
    def __init__(self):
        self.gdal_fixture = MockGDALFixture()
        self.tf_fixture = MockTensorFlowFixture()
        self.db_fixture = MockDatabaseFixture()
    
    @contextmanager
    def complete_mock_environment(self, test_scenario: str = "standard"):
        """Complete mock environment for integration testing."""
        
        # Setup test files based on scenario
        if test_scenario == "standard":
            self.gdal_fixture.create_bathymetric_file_mock("test_shallow.bag", "shallow_coastal")
            self.gdal_fixture.create_bathymetric_file_mock("test_deep.tif", "deep_ocean")
            self.gdal_fixture.create_bathymetric_file_mock("test_seamount.bag", "seamount")
        
        elif test_scenario == "error_handling":
            # Create files that might cause processing errors
            self.gdal_fixture.register_file("corrupt_file.bag", 
                                          np.full((64, 64), np.nan))
            self.gdal_fixture.register_file("empty_file.tif", 
                                          np.zeros((0, 0)))
        
        elif test_scenario == "performance":
            # Create larger files for performance testing
            large_data = np.random.random((256, 256)).astype(np.float32)
            for i in range(10):
                self.gdal_fixture.register_file(f"large_file_{i}.tif", large_data)
        
        # Setup mock models
        ensemble_models = self.tf_fixture.create_mock_ensemble(3)
        
        # Combine all mock environments
        with self.gdal_fixture.mock_gdal_environment(), \
             self.tf_fixture.mock_tensorflow_environment(), \
             self.db_fixture.mock_database_environment() as (mock_db, db_path):
            
            yield {
                'gdal_fixture': self.gdal_fixture,
                'tf_fixture': self.tf_fixture,
                'db_fixture': self.db_fixture,
                'mock_database': mock_db,
                'db_path': db_path,
                'ensemble_models': ensemble_models
            }
    
    def cleanup(self):
        """Clean up all mock fixtures."""
        self.gdal_fixture.cleanup()
        self.tf_fixture.cleanup()
        self.db_fixture.cleanup()