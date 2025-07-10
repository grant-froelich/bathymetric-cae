"""
Visualization utilities for Enhanced Bathymetric CAE Processing.

This module provides comprehensive plotting and visualization functions
for quality assessment, comparison plots, and processing dashboards.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def create_enhanced_visualization(original: np.ndarray, cleaned: np.ndarray,
                                uncertainty: Optional[np.ndarray], metrics: Dict,
                                file_path: Path, adaptive_params: Dict):
    """Create enhanced visualization with quality metrics."""
    try:
        n_subplots = 4 if uncertainty is not None else 3
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original data
        im1 = axes[0, 0].imshow(original, cmap='viridis', aspect='equal')
        axes[0, 0].set_title('Original Data', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        # Cleaned data
        im2 = axes[0, 1].imshow(cleaned, cmap='viridis', aspect='equal')
        axes[0, 1].set_title('Enhanced Cleaned Data', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # Difference map
        diff = cleaned - original
        diff_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', aspect='equal', 
                               vmin=-diff_max, vmax=diff_max)
        axes[1, 0].set_title('Difference (Cleaned - Original)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        
        # Quality metrics display
        axes[1, 1].axis('off')
        
        # Create metrics text
        metrics_text = _format_metrics_text(metrics, adaptive_params)
        
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Add quality level indicator
        quality_score = metrics.get('composite_quality', 0)
        quality_color = _get_quality_color(quality_score)
        
        # Quality score circle
        circle = patches.Circle((0.8, 0.8), 0.1, transform=axes[1, 1].transAxes,
                               facecolor=quality_color, edgecolor='black', linewidth=2)
        axes[1, 1].add_patch(circle)
        axes[1, 1].text(0.8, 0.8, f'{quality_score:.2f}', transform=axes[1, 1].transAxes,
                        ha='center', va='center', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Enhanced Processing Results: {file_path.name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"plots/enhanced_comparison_{file_path.stem}.png"
        Path("plots").mkdir(exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced visualization saved: {plot_filename}")
        
    except Exception as e:
        logger.error(f"Error creating enhanced visualization: {e}")
        plt.close('all')  # Cleanup on error


def create_comparison_plot(original: np.ndarray, processed: np.ndarray, 
                         title: str = "Comparison", save_path: Optional[str] = None) -> str:
    """Create a side-by-side comparison plot."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        im1 = axes[0].imshow(original, cmap='viridis', aspect='equal')
        axes[0].set_title('Original', fontsize=14)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Processed
        im2 = axes[1].imshow(processed, cmap='viridis', aspect='equal')
        axes[1].set_title('Processed', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = processed - original
        diff_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='equal',
                            vmin=-diff_max, vmax=diff_max)
        axes[2].set_title('Difference', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plot_path = save_path
        else:
            plot_path = "plots/comparison.png"
            Path("plots").mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return plot_path
        
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
        plt.close('all')
        return ""


def create_quality_dashboard(processing_stats: List[Dict], save_path: str = "plots/quality_dashboard.png"):
    """Create a comprehensive quality assessment dashboard."""
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # Extract quality metrics
        quality_scores = [s.get('composite_quality', 0) for s in processing_stats]
        ssim_scores = [s.get('ssim', 0) for s in processing_stats]
        feature_scores = [s.get('feature_preservation', 0) for s in processing_stats]
        consistency_scores = [s.get('consistency', 0) for s in processing_stats]
        
        # 1. Quality Score Distribution
        plt.subplot(2, 3, 1)
        plt.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(quality_scores), color='red', linestyle='--', label=f'Mean: {np.mean(quality_scores):.3f}')
        plt.xlabel('Composite Quality Score')
        plt.ylabel('Frequency')
        plt.title('Quality Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Quality vs SSIM
        plt.subplot(2, 3, 2)
        plt.scatter(ssim_scores, quality_scores, alpha=0.6, s=50)
        plt.xlabel('SSIM Score')
        plt.ylabel('Composite Quality')
        plt.title('Quality vs SSIM Correlation')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(ssim_scores, quality_scores)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        # 3. Metric Comparison Box Plot
        plt.subplot(2, 3, 3)
        metrics_data = [quality_scores, ssim_scores, feature_scores, consistency_scores]
        metrics_labels = ['Quality', 'SSIM', 'Features', 'Consistency']
        
        bp = plt.boxplot(metrics_data, labels=metrics_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.ylabel('Score')
        plt.title('Metric Distributions')
        plt.grid(True, alpha=0.3)
        
        # 4. Quality Categories Pie Chart
        plt.subplot(2, 3, 4)
        excellent = len([q for q in quality_scores if q > 0.9])
        good = len([q for q in quality_scores if 0.8 <= q <= 0.9])
        acceptable = len([q for q in quality_scores if 0.7 <= q < 0.8])
        poor = len([q for q in quality_scores if q < 0.7])
        
        sizes = [excellent, good, acceptable, poor]
        labels = ['Excellent\n(>0.9)', 'Good\n(0.8-0.9)', 'Acceptable\n(0.7-0.8)', 'Poor\n(<0.7)']
        colors = ['green', 'lightgreen', 'yellow', 'red']
        
        # Only include non-zero categories
        non_zero = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        
        plt.title('Quality Categories')
        
        # 5. Seafloor Type Distribution
        plt.subplot(2, 3, 5)
        seafloor_types = [s.get('seafloor_type', 'unknown') for s in processing_stats]
        unique_types, counts = np.unique(seafloor_types, return_counts=True)
        
        bars = plt.bar(range(len(unique_types)), counts, color='lightsteelblue', edgecolor='black')
        plt.xticks(range(len(unique_types)), unique_types, rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title('Seafloor Type Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 6. Processing Time Analysis
        plt.subplot(2, 3, 6)
        if any('processing_time' in s for s in processing_stats):
            # This would require processing time data
            plt.text(0.5, 0.5, 'Processing Time\nAnalysis\n(Not implemented)', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray'))
        else:
            # Summary statistics
            summary_text = f"""Summary Statistics:
            
Total Files: {len(processing_stats)}
Mean Quality: {np.mean(quality_scores):.3f}
Std Quality: {np.std(quality_scores):.3f}
Min Quality: {np.min(quality_scores):.3f}
Max Quality: {np.max(quality_scores):.3f}

High Quality (>0.8): {len([q for q in quality_scores if q > 0.8])}
Low Quality (<0.6): {len([q for q in quality_scores if q < 0.6])}
            """
            
            plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray'))
            plt.axis('off')
        
        plt.suptitle('Quality Assessment Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save dashboard
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Quality dashboard saved: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error creating quality dashboard: {e}")
        plt.close('all')
        return ""


def plot_training_history(history, filename: str = "training_history.png"):
    """Plot training history for a model."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        if 'mae' in history.history:
            axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
            axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
            axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].axis('off')
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
        
        # Training summary
        axes[1, 1].axis('off')
        
        # Calculate training summary
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        best_val_loss = min(history.history['val_loss'])
        epochs_trained = len(history.history['loss'])
        
        summary_text = f"""Training Summary:
        
Epochs: {epochs_trained}
Final Loss: {final_loss:.4f}
Final Val Loss: {final_val_loss:.4f}
Best Val Loss: {best_val_loss:.4f}

Overfitting: {'Yes' if final_val_loss > best_val_loss * 1.1 else 'No'}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = f"plots/{filename}"
        Path("plots").mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved: {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")
        plt.close('all')
        return ""


def save_processing_plots(original: np.ndarray, processed: np.ndarray,
                         metrics: Dict, filename: str):
    """Save comprehensive processing plots."""
    try:
        # Create plots directory
        plots_dir = Path("plots") / "processing"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual plots
        base_name = Path(filename).stem
        
        # 1. Comparison plot
        comparison_path = create_comparison_plot(
            original, processed, 
            title=f"Processing Results: {filename}",
            save_path=str(plots_dir / f"{base_name}_comparison.png")
        )
        
        # 2. Statistical plots
        _create_statistical_plots(original, processed, plots_dir / f"{base_name}_stats.png")
        
        # 3. Quality metrics visualization
        _create_metrics_plot(metrics, plots_dir / f"{base_name}_metrics.png")
        
        logger.info(f"Processing plots saved for {filename}")
        
        return {
            'comparison': comparison_path,
            'statistics': str(plots_dir / f"{base_name}_stats.png"),
            'metrics': str(plots_dir / f"{base_name}_metrics.png")
        }
        
    except Exception as e:
        logger.error(f"Error saving processing plots: {e}")
        return {}


def _format_metrics_text(metrics: Dict, adaptive_params: Dict) -> str:
    """Format metrics text for display."""
    quality_score = metrics.get('composite_quality', 0)
    seafloor_type = adaptive_params.get('seafloor_type', 'unknown')
    
    metrics_text = f"""Quality Metrics:

SSIM: {metrics.get('ssim', 0):.4f}
Feature Preservation: {metrics.get('feature_preservation', 0):.4f}
Consistency: {metrics.get('consistency', 0):.4f}
Composite Quality: {quality_score:.4f}
Hydrographic Compliance: {metrics.get('hydrographic_compliance', 0):.4f}

Seafloor Type: {seafloor_type.replace('_', ' ').title()}
Smoothing Factor: {adaptive_params.get('smoothing_factor', 0):.2f}
Edge Preservation: {adaptive_params.get('edge_preservation', 0):.2f}

Quality Level: {_get_quality_level(quality_score)}
    """
    
    return metrics_text


def _get_quality_color(score: float) -> str:
    """Get color based on quality score."""
    if score >= 0.9:
        return 'green'
    elif score >= 0.8:
        return 'lightgreen'
    elif score >= 0.7:
        return 'yellow'
    elif score >= 0.6:
        return 'orange'
    else:
        return 'red'


def _get_quality_level(score: float) -> str:
    """Get quality level description."""
    if score >= 0.9:
        return 'Excellent'
    elif score >= 0.8:
        return 'Good'
    elif score >= 0.7:
        return 'Acceptable'
    elif score >= 0.6:
        return 'Fair'
    else:
        return 'Poor'


def _create_statistical_plots(original: np.ndarray, processed: np.ndarray, save_path: Path):
    """Create statistical analysis plots."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Histograms
        axes[0, 0].hist(original.flatten(), bins=50, alpha=0.7, label='Original', density=True)
        axes[0, 0].hist(processed.flatten(), bins=50, alpha=0.7, label='Processed', density=True)
        axes[0, 0].set_xlabel('Depth Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Depth Value Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot
        sample_indices = np.random.choice(original.size, min(10000, original.size), replace=False)
        orig_sample = original.flat[sample_indices]
        proc_sample = processed.flat[sample_indices]
        
        axes[0, 1].scatter(orig_sample, proc_sample, alpha=0.5, s=1)
        axes[0, 1].plot([orig_sample.min(), orig_sample.max()], 
                       [orig_sample.min(), orig_sample.max()], 'r--', label='y=x')
        axes[0, 1].set_xlabel('Original Depth')
        axes[0, 1].set_ylabel('Processed Depth')
        axes[0, 1].set_title('Original vs Processed')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Difference statistics
        diff = processed - original
        axes[1, 0].hist(diff.flatten(), bins=50, alpha=0.7, color='purple')
        axes[1, 0].axvline(np.mean(diff), color='red', linestyle='--', label=f'Mean: {np.mean(diff):.3f}')
        axes[1, 0].axvline(np.median(diff), color='orange', linestyle='--', label=f'Median: {np.median(diff):.3f}')
        axes[1, 0].set_xlabel('Difference (Processed - Original)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Difference Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        
        stats_text = f"""Statistical Summary:
        
Original:
  Mean: {np.mean(original):.3f}
  Std: {np.std(original):.3f}
  Min: {np.min(original):.3f}
  Max: {np.max(original):.3f}

Processed:
  Mean: {np.mean(processed):.3f}
  Std: {np.std(processed):.3f}
  Min: {np.min(processed):.3f}
  Max: {np.max(processed):.3f}

Difference:
  Mean: {np.mean(diff):.3f}
  Std: {np.std(diff):.3f}
  RMSE: {np.sqrt(np.mean(diff**2)):.3f}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Statistical Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating statistical plots: {e}")
        plt.close('all')


def _create_metrics_plot(metrics: Dict, save_path: Path):
    """Create quality metrics visualization."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Metrics radar chart
        metric_names = ['SSIM', 'Feature\nPreservation', 'Consistency', 'Hydrographic\nCompliance']
        metric_values = [
            metrics.get('ssim', 0),
            metrics.get('feature_preservation', 0),
            metrics.get('consistency', 0),
            metrics.get('hydrographic_compliance', 0)
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        metric_values += metric_values[:1]
        
        axes[0, 0].plot(angles, metric_values, 'o-', linewidth=2, label='Scores')
        axes[0, 0].fill(angles, metric_values, alpha=0.25)
        axes[0, 0].set_xticks(angles[:-1])
        axes[0, 0].set_xticklabels(metric_names)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title('Quality Metrics Radar')
        axes[0, 0].grid(True)
        
        # 2. Metrics bar chart
        bar_names = ['SSIM', 'Features', 'Consistency', 'Compliance', 'Composite']
        bar_values = [
            metrics.get('ssim', 0),
            metrics.get('feature_preservation', 0),
            metrics.get('consistency', 0),
            metrics.get('hydrographic_compliance', 0),
            metrics.get('composite_quality', 0)
        ]
        
        bars = axes[0, 1].bar(bar_names, bar_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Quality Metrics')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, bar_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Error metrics
        error_names = ['MAE', 'RMSE']
        error_values = [
            metrics.get('mae', 0),
            metrics.get('rmse', 0)
        ]
        
        if any(error_values):
            axes[1, 0].bar(error_names, error_values, color=['orange', 'red'])
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].set_title('Error Metrics')
            
            for i, (name, value) in enumerate(zip(error_names, error_values)):
                axes[1, 0].text(i, value + max(error_values) * 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1, 0].axis('off')
        
        # 4. Quality assessment
        axes[1, 1].axis('off')
        
        composite_quality = metrics.get('composite_quality', 0)
        quality_level = _get_quality_level(composite_quality)
        quality_color = _get_quality_color(composite_quality)
        
        # Create quality gauge
        theta = np.linspace(0, np.pi, 100)
        r = 0.8
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        axes[1, 1].plot(x, y, 'k-', linewidth=3)
        
        # Quality sectors
        sectors = [
            (0, 0.6, 'red'),
            (0.6, 0.7, 'orange'),
            (0.7, 0.8, 'yellow'),
            (0.8, 0.9, 'lightgreen'),
            (0.9, 1.0, 'green')
        ]
        
        for start, end, color in sectors:
            start_angle = (1 - start) * np.pi
            end_angle = (1 - end) * np.pi
            sector_theta = np.linspace(start_angle, end_angle, 20)
            sector_x = r * np.cos(sector_theta)
            sector_y = r * np.sin(sector_theta)
            axes[1, 1].fill_between(sector_x, 0, sector_y, color=color, alpha=0.7)
        
        # Quality needle
        needle_angle = (1 - composite_quality) * np.pi
        needle_x = r * 0.9 * np.cos(needle_angle)
        needle_y = r * 0.9 * np.sin(needle_angle)
        axes[1, 1].arrow(0, 0, needle_x, needle_y, head_width=0.05, head_length=0.05,
                        fc='black', ec='black', linewidth=3)
        
        axes[1, 1].set_xlim(-1, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].set_title(f'Quality Gauge: {quality_level}\n({composite_quality:.3f})')
        
        plt.suptitle('Quality Metrics Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating metrics plot: {e}")
        plt.close('all')


def create_ensemble_comparison_plot(predictions: List[np.ndarray], 
                                  ensemble_result: np.ndarray,
                                  uncertainty: np.ndarray,
                                  save_path: str = "plots/ensemble_comparison.png"):
    """Create ensemble comparison visualization."""
    try:
        n_models = len(predictions)
        cols = min(3, n_models + 2)  # +2 for ensemble and uncertainty
        rows = (n_models + 3) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Individual model predictions
        for i, pred in enumerate(predictions):
            row, col = divmod(i, cols)
            im = axes[row, col].imshow(pred[0, :, :, 0], cmap='viridis', aspect='equal')
            axes[row, col].set_title(f'Model {i+1}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # Ensemble result
        ensemble_row, ensemble_col = divmod(n_models, cols)
        im = axes[ensemble_row, ensemble_col].imshow(ensemble_result[0, :, :, 0], cmap='viridis', aspect='equal')
        axes[ensemble_row, ensemble_col].set_title('Ensemble Result')
        axes[ensemble_row, ensemble_col].axis('off')
        plt.colorbar(im, ax=axes[ensemble_row, ensemble_col], shrink=0.8)
        
        # Uncertainty
        if n_models + 1 < rows * cols:
            unc_row, unc_col = divmod(n_models + 1, cols)
            im = axes[unc_row, unc_col].imshow(uncertainty[0, :, :, 0], cmap='Reds', aspect='equal')
            axes[unc_row, unc_col].set_title('Prediction Uncertainty')
            axes[unc_row, unc_col].axis('off')
            plt.colorbar(im, ax=axes[unc_row, unc_col], shrink=0.8)
        
        # Hide unused subplots
        for i in range(n_models + 2, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].axis('off')
        
        plt.suptitle('Ensemble Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Ensemble comparison plot saved: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error creating ensemble comparison plot: {e}")
        plt.close('all')
        return ""


def create_seafloor_classification_plot(depth_data: np.ndarray, 
                                       classification_map: np.ndarray,
                                       seafloor_type: str,
                                       save_path: str = "plots/seafloor_classification.png"):
    """Create seafloor classification visualization."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original depth data
        im1 = axes[0].imshow(depth_data, cmap='viridis', aspect='equal')
        axes[0].set_title('Depth Data')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Classification map
        # Note: This would need proper implementation based on classification_map format
        axes[1].text(0.5, 0.5, f'Classified as:\n{seafloor_type.replace("_", " ").title()}', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1].set_title('Seafloor Classification')
        axes[1].axis('off')
        
        plt.suptitle('Seafloor Type Classification', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    except Exception as e:
        logger.error(f"Error creating seafloor classification plot: {e}")
        plt.close('all')
        return ""


# Configuration for plot styling
PLOT_CONFIG = {
    'style': 'seaborn-v0_8',
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': 12,
    'title_size': 14,
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}


def configure_plotting(config: Optional[Dict] = None):
    """Configure plotting parameters."""
    if config:
        PLOT_CONFIG.update(config)
    
    plt.style.use(PLOT_CONFIG['style'])
    plt.rcParams.update({
        'figure.figsize': PLOT_CONFIG['figure_size'],
        'figure.dpi': PLOT_CONFIG['dpi'],
        'font.size': PLOT_CONFIG['font_size'],
        'axes.titlesize': PLOT_CONFIG['title_size'],
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    logger.info(f"Plotting configured with: {PLOT_CONFIG}")


# Cleanup function
def cleanup_plots():
    """Close all open plots and clear memory."""
    plt.close('all')
    logger.debug("All plots closed and memory cleared")
    