# tests/test_config.py
"""
Test configuration management.
"""

import pytest
import json
import tempfile
from pathlib import Path

from config.config import Config


class TestConfig:
    """Test configuration functionality."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = Config()
        assert config.epochs == 100
        assert config.batch_size == 8
        assert config.grid_size == 512
        assert config.ensemble_size == 3
        assert config.enable_adaptive_processing is True
    
    def test_config_validation_valid(self):
        """Test configuration validation with valid parameters."""
        config = Config(
            epochs=50,
            batch_size=4,
            validation_split=0.2,
            grid_size=128,
            learning_rate=0.001,
            ensemble_size=3
        )
        # Should not raise exception
        config.validate()
    
    def test_config_validation_invalid_epochs(self):
        """Test configuration validation with invalid epochs."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            Config(epochs=0)
    
    def test_config_validation_invalid_validation_split(self):
        """Test configuration validation with invalid validation split."""
        with pytest.raises(ValueError, match="validation_split must be between 0 and 1"):
            Config(validation_split=1.5)
    
    def test_config_validation_invalid_weights(self):
        """Test configuration validation with invalid weight sum."""
        with pytest.raises(ValueError, match="Quality metric weights must sum to 1.0"):
            Config(
                ssim_weight=0.5,
                roughness_weight=0.5,
                feature_preservation_weight=0.5,
                consistency_weight=0.5
            )
    
    def test_config_save_load(self, temp_dir):
        """Test saving and loading configuration."""
        config = Config(epochs=75, batch_size=16)
        config_path = temp_dir / "test_config.json"
        
        # Save configuration
        config.save(str(config_path))
        assert config_path.exists()
        
        # Load configuration
        loaded_config = Config.load(str(config_path))
        assert loaded_config.epochs == 75
        assert loaded_config.batch_size == 16
