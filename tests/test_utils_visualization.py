# tests/test_utils_visualization.py
"""
Test visualization utilities.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
from pathlib import Path
from unittest.mock import patch, Mock

from utils.visualization import create_enhanced_visualization, plot_training_history


class TestVisualization:
    """Test visualization utilities."""
    
    def test_create_enhanced_visualization(self, temp_dir):
        """Test enhanced visualization creation."""
        # Create sample data
        original = np.random.random((64, 64))
        cleaned = original + np.random.random((64, 64)) * 0.1
        uncertainty = np.random.random((64, 64)) * 0.5
        
        metrics = {
            'ssim': 0.85,
            'feature_preservation': 0.78,
            'consistency': 0.82,
            'composite_quality': 0.81,
            'hydrographic_compliance': 0.79
        }
        
        adaptive_params = {
            'seafloor_type': 'continental_shelf',
            'smoothing_factor': 0.5,
            'edge_preservation': 0.6
        }
        
        file_path = Path("test_file.bag")
        
        # Ensure plots directory exists
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Test visualization creation
        create_enhanced_visualization(
            original, cleaned, uncertainty, metrics, file_path, adaptive_params
        )
        
        # Check that plot file was created
        expected_plot = plots_dir / f"enhanced_comparison_{file_path.stem}.png"
        assert expected_plot.exists()
        
        # Cleanup
        expected_plot.unlink(missing_ok=True)
    
    def test_plot_training_history(self, temp_dir):
        """Test training history plotting."""
        # Mock training history
        mock_history = Mock()
        mock_history.history = {
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3],
            'mae': [0.5, 0.4, 0.3, 0.2, 0.1],
            'val_mae': [0.6, 0.5, 0.4, 0.3, 0.2]
        }
        
        # Ensure plots directory exists
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Test plotting
        filename = "test_training_history.png"
        plot_training_history(mock_history, filename)
        
        # Check that plot file was created
        expected_plot = plots_dir / filename
        assert expected_plot.exists()
        
        # Cleanup
        expected_plot.unlink(missing_ok=True)
