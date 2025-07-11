# tests/test_cli_interface.py
"""
Test command line interface.
"""

import pytest
import argparse

from cli.interface import create_argument_parser, update_config_from_args
from config.config import Config


class TestCLIInterface:
    """Test command line interface functionality."""
    
    def test_argument_parser_creation(self):
        """Test argument parser creation."""
        parser = create_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        
        # Test some key arguments
        args = parser.parse_args(['--epochs', '50', '--batch-size', '16'])
        assert args.epochs == 50
        assert args.batch_size == 16
    
    def test_config_update_from_args(self):
        """Test updating configuration from arguments."""
        config = Config()
        
        # Mock arguments
        args = argparse.Namespace(
            epochs=75,
            batch_size=32,
            enable_adaptive=True,
            input='test_input',
            grid_size=256
        )
        
        updated_config = update_config_from_args(config, args)
        
        assert updated_config.epochs == 75
        assert updated_config.batch_size == 32
        assert updated_config.enable_adaptive_processing is True
        assert updated_config.input_folder == 'test_input'
        assert updated_config.grid_size == 256
    
    def test_boolean_flag_handling(self):
        """Test boolean flag handling."""
        parser = create_argument_parser()
        
        # Test boolean flags
        args = parser.parse_args(['--enable-adaptive', '--enable-expert-review', '--no-gpu'])
        assert args.enable_adaptive is True
        assert args.enable_expert_review is True
        assert args.no_gpu is True