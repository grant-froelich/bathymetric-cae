"""
Tests for Configuration Module

Tests for the bathymetric CAE configuration management including
validation, serialization, and parameter handling.

Author: Bathymetric CAE Team
License: MIT
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from bathymetric_cae.config.config import (
    Config, 
    load_default_config,
    load_config_from_file,
    create_config_from_args
)


class TestConfig:
    """Test the Config class."""
    
    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = Config()
        
        # Test default values
        assert config.epochs == 100
        assert config.batch_size == 8
        assert config.grid_size == 512
        assert config.validation_split == 0.2
        assert config.learning_rate == 0.001
        assert config.base_filters == 32
        assert config.depth == 4
        assert config.dropout_rate == 0.2
        
        # Test supported formats
        assert '.bag' in config.supported_formats
        assert '.tif' in config.supported_formats
        assert '.tiff' in config.supported_formats
        assert '.asc' in config.supported_formats
        assert '.xyz' in config.supported_formats
    
    def test_custom_config_creation(self):
        """Test creation of custom configuration."""
        config = Config(
            epochs=50,
            batch_size=16,
            grid_size=256,
            learning_rate=0.002
        )
        
        assert config.epochs == 50
        assert config.batch_size == 16
        assert config.grid_size == 256
        assert config.learning_rate == 0.002
        
        # Other values should remain default
        assert config.depth == 4
        assert config.dropout_rate == 0.2
    
    def test_config_validation_valid(self):
        """Test validation with valid parameters."""
        # This should not raise any exception
        config = Config(
            epochs=100,
            batch_size=8,
            validation_split=0.2,
            grid_size=512,
            learning_rate=0.001
        )
        config.validate()  # Should pass without exception
    
    def test_config_validation_invalid_epochs(self):
        """Test validation with invalid epochs."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            Config(epochs=-10)
    
    def test_config_validation_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            Config(batch_size=0)
    
    def test_config_validation_invalid_validation_split(self):
        """Test validation with invalid validation split."""
        with pytest.raises(ValueError, match="validation_split must be between 0 and 1"):
            Config(validation_split=1.5)
        
        with pytest.raises(ValueError, match="validation_split must be between 0 and 1"):
            Config(validation_split=-0.1)
    
    def test_config_validation_invalid_grid_size(self):
        """Test validation with invalid grid size."""
        with pytest.raises(ValueError, match="grid_size must be at least 32"):
            Config(grid_size=16)
    
    def test_config_validation_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            Config(learning_rate=-0.001)
    
    def test_config_validation_multiple_errors(self):
        """Test validation with multiple errors."""
        with pytest.raises(ValueError) as exc_info:
            Config(epochs=-1, batch_size=0, learning_rate=-0.001)
        
        error_message = str(exc_info.value)
        assert "epochs must be positive" in error_message
        assert "batch_size must be positive" in error_message
        assert "learning_rate must be positive" in error_message
    
    def test_config_save_and_load(self, temp_dir):
        """Test saving and loading configuration."""
        config = Config(
            epochs=75,
            batch_size=12,
            grid_size=256,
            learning_rate=0.0005
        )
        
        config_file = temp_dir / "test_config.json"
        
        # Save configuration
        config.save(config_file)
        assert config_file.exists()
        
        # Load configuration
        loaded_config = Config.load(config_file)
        
        # Verify loaded configuration
        assert loaded_config.epochs == 75
        assert loaded_config.batch_size == 12
        assert loaded_config.grid_size == 256
        assert loaded_config.learning_rate == 0.0005
    
    def test_config_save_creates_directory(self, temp_dir):
        """Test that save creates parent directories."""
        nested_dir = temp_dir / "nested" / "config"
        config_file = nested_dir / "config.json"
        
        config = Config()
        config.save(config_file)
        
        assert config_file.exists()
        assert config_file.parent.exists()
    
    def test_config_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config.load("nonexistent_config.json")
    
    def test_config_load_invalid_json(self, temp_dir):
        """Test loading from invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{ invalid json content")
        
        with pytest.raises(ValueError, match="Invalid configuration file"):
            Config.load(invalid_file)
    
    def test_config_update(self):
        """Test updating configuration."""
        config = Config(epochs=100, batch_size=8)
        
        updated_config = config.update(epochs=150, learning_rate=0.002)
        
        # Original config should be unchanged
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        
        # Updated config should have new values
        assert updated_config.epochs == 150
        assert updated_config.learning_rate == 0.002
        assert updated_config.batch_size == 8  # Unchanged value
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config(epochs=50, batch_size=16)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['epochs'] == 50
        assert config_dict['batch_size'] == 16
        assert 'learning_rate' in config_dict
    
    def test_config_str_representation(self):
        """Test string representation of configuration."""
        config = Config(epochs=50, batch_size=16, grid_size=256)
        config_str = str(config)
        
        assert 'epochs=50' in config_str
        assert 'batch_size=16' in config_str
        assert 'grid_size=256' in config_str
    
    def test_config_repr_representation(self):
        """Test repr representation of configuration."""
        config = Config(epochs=50)
        config_repr = repr(config)
        
        assert 'Config(' in config_repr
        assert 'epochs' in config_repr


class TestConfigUtilityFunctions:
    """Test utility functions for configuration management."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_default_config()
        
        assert isinstance(config, Config)
        assert config.epochs == 100  # Default value
        assert config.batch_size == 8  # Default value
    
    def test_load_config_from_file_existing(self, temp_dir):
        """Test loading configuration from existing file."""
        config_data = {
            "epochs": 75,
            "batch_size": 12,
            "learning_rate": 0.0005
        }
        
        config_file = temp_dir / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = load_config_from_file(config_file)
        
        assert config.epochs == 75
        assert config.batch_size == 12
        assert config.learning_rate == 0.0005
    
    def test_load_config_from_file_nonexistent(self, capfd):
        """Test loading configuration from non-existent file."""
        config = load_config_from_file("nonexistent.json")
        
        # Should return default config
        assert isinstance(config, Config)
        assert config.epochs == 100  # Default value
        
        # Should print warning message
        captured = capfd.readouterr()
        assert "not found" in captured.out
    
    def test_load_config_from_file_invalid(self, temp_dir, capfd):
        """Test loading configuration from invalid file."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{ invalid json")
        
        config = load_config_from_file(invalid_file)
        
        # Should return default config
        assert isinstance(config, Config)
        
        # Should print error message
        captured = capfd.readouterr()
        assert "Error loading configuration" in captured.out
    
    def test_create_config_from_args(self):
        """Test creating configuration from command line arguments."""
        # Mock command line arguments
        class MockArgs:
            epochs = 150
            batch_size = 16
            learning_rate = 0.002
            grid_size = None  # Should not override
            nonexistent_arg = "should_be_ignored"
        
        args = MockArgs()
        config = create_config_from_args(args)
        
        assert config.epochs == 150
        assert config.batch_size == 16
        assert config.learning_rate == 0.002
        assert config.grid_size == 512  # Should remain default
    
    def test_create_config_from_args_with_base_config(self):
        """Test creating configuration from args with base config."""
        base_config = Config(epochs=100, batch_size=8)
        
        class MockArgs:
            epochs = 200
            learning_rate = 0.003
            batch_size = None  # Should not override
        
        args = MockArgs()
        config = create_config_from_args(base_config, args)
        
        assert config.epochs == 200  # Overridden
        assert config.learning_rate == 0.003  # Overridden
        assert config.batch_size == 8  # From base config


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""
    
    def test_json_serialization(self, temp_dir):
        """Test JSON serialization with various data types."""
        config = Config(
            epochs=100,
            batch_size=8,
            validation_split=0.2,
            supported_formats=['.bag', '.tif'],
            gpu_memory_growth=True
        )
        
        config_file = temp_dir / "serialization_test.json"
        config.save(config_file)
        
        # Verify JSON file content
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        assert data['epochs'] == 100
        assert data['batch_size'] == 8
        assert data['validation_split'] == 0.2
        assert data['supported_formats'] == ['.bag', '.tif']
        assert data['gpu_memory_growth'] is True
    
    def test_special_values_serialization(self, temp_dir):
        """Test serialization of special values like tf.data.AUTOTUNE."""
        config = Config()
        
        # Mock tf.data.AUTOTUNE if TensorFlow is available
        with patch('tensorflow.data.AUTOTUNE', 'AUTOTUNE_MOCK'):
            config.prefetch_buffer_size = 'AUTOTUNE_MOCK'
            
            config_file = temp_dir / "special_values_test.json"
            config.save(config_file)
            
            # Should save successfully without errors
            assert config_file.exists()
    
    def test_roundtrip_serialization(self, temp_dir):
        """Test complete roundtrip serialization."""
        original_config = Config(
            epochs=123,
            batch_size=17,
            grid_size=384,
            learning_rate=0.00123,
            dropout_rate=0.15,
            supported_formats=['.bag', '.tif', '.asc']
        )
        
        config_file = temp_dir / "roundtrip_test.json"
        
        # Save and load
        original_config.save(config_file)
        loaded_config = Config.load(config_file)
        
        # Compare all attributes
        original_dict = original_config.to_dict()
        loaded_dict = loaded_config.to_dict()
        
        # Check key attributes
        for key in ['epochs', 'batch_size', 'grid_size', 'learning_rate', 'dropout_rate']:
            assert original_dict[key] == loaded_dict[key], f"Mismatch in {key}"
        
        assert original_dict['supported_formats'] == loaded_dict['supported_formats']


class TestConfigEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_valid_values(self):
        """Test configuration with minimum valid values."""
        config = Config(
            epochs=1,
            batch_size=1,
            validation_split=0.01,
            grid_size=32,
            learning_rate=1e-10,
            base_filters=1,
            depth=1,
            dropout_rate=0.0
        )
        
        # Should not raise validation errors
        config.validate()
    
    def test_maximum_reasonable_values(self):
    """Test configuration with large but reasonable values."""
    config = Config(
        epochs=10000,
        batch_size=1024,
        validation_split=0.99,
        grid_size=2048,
        learning_rate=1.0,
        base_filters=512,
        depth=20,
        dropout_rate=0.99
    )
    
    # Should not raise validation errors for large but valid values
    config.validate()
    
    # Verify the values are set correctly
    assert config.epochs == 10000
    assert config.batch_size == 1024
    assert config.validation_split == 0.99
    assert config.grid_size == 2048
    assert config.learning_rate == 1.0
    assert config.base_filters == 512
    assert config.depth == 20
    assert config.dropout_rate == 0.99