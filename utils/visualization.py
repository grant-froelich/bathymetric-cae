"""
Visualization and plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Optional


def create_enhanced_visualization(original: np.ndarray, cleaned: np.ndarray,
                                 uncertainty: Optional[np.ndarray], metrics: Dict,
                                 file_path: Path, adaptive_params: Dict):
    """Create enhanced visualization with quality metrics."""
    logger = logging.getLogger(__name__)
    
    n_plots = 4 if uncertainty is not None else 3
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data
    im1 = axes[0, 0].imshow(original, cmap='viridis', aspect='equal')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Cleaned data
    im2 = axes[0, 1].imshow(cleaned, cmap='viridis', aspect='equal')
    axes[0, 1].set_title('Enhanced Cleaned Data')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference map
    diff = cleaned - original
    im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', aspect='equal')
    axes[1, 0].set_title('Difference (Cleaned - Original)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Quality metrics display
    axes[1, 1].axis('off')
    metrics_text = f"""Quality Metrics:
    
SSIM: {metrics.get('ssim', 0):.4f}
Feature Preservation: {metrics.get('feature_preservation', 0):.4f}
Consistency: {metrics.get('consistency', 0):.4f}
Composite Quality: {metrics.get('composite_quality', 0):.4f}
Hydrographic Compliance: {metrics.get('hydrographic_compliance', 0):.4f}

Seafloor Type: {adaptive_params.get('seafloor_type', 'unknown')}
Smoothing Factor: {adaptive_params.get('smoothing_factor', 0):.2f}
Edge Preservation: {adaptive_params.get('edge_preservation', 0):.2f}
    """
    
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Enhanced Processing Results: {file_path.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_filename = f"plots/enhanced_comparison_{file_path.stem}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Enhanced visualization saved: {plot_filename}")


def plot_training_history(history, filename: str):
    """Plot training history for a model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].legend()
    
    # MAE
    if 'mae' in history.history:
        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
