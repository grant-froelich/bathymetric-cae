# tests/test_config.py
"""Tests for configuration module."""

import pytest
import json
import tempfile
from pathlib import Path

from bathymetric_cae.config import Config, load_config_from_file, merge_configs


class TestConfig:
    """Test configuration functionality."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = Config()
        
        assert config.grid_size == 512
        assert config.epochs == 100
        assert config.batch_size == 8
        assert config.ensemble_size == 3
        assert len(config.supported_formats) > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Valid configuration should not raise
        config.validate()
        
        # Invalid epochs should raise
        config.epochs = -1
        with pytest.raises(ValueError, match="epochs must be positive"):
            config.validate()
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Create and save config
            config = Config()
            config.epochs = 150
            config.save(config_path)
            
            # Load config
            loaded_config = load_config_from_file(config_path)
            
            assert loaded_config.epochs == 150
            assert loaded_config.grid_size == config.grid_size
            
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_config_update_from_dict(self):
        """Test updating configuration from dictionary."""
        config = Config()
        original_epochs = config.epochs
        
        updates = {'epochs': 200, 'batch_size': 16}
        config.update_from_dict(updates)
        
        assert config.epochs == 200
        assert config.batch_size == 16
        assert config.grid_size == 512  # Unchanged
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = Config()
        overrides = {'epochs': 300, 'learning_rate': 0.01}
        
        merged = merge_configs(base_config, overrides)
        
        assert merged.epochs == 300
        assert merged.learning_rate == 0.01
        assert merged.grid_size == base_config.grid_size
