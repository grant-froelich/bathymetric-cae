"""
Visualization Utilities Module

This module provides comprehensive visualization capabilities for the
bathymetric CAE pipeline, including training history plots, data comparisons,
and result analysis.

Author: Bathymetric CAE Team
License: MIT
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class Visualizer:
    """
    Enhanced visualization utilities for bathymetric data processing.
    
    This class provides methods for creating various plots and visualizations
    including training history, data comparisons, statistical analysis, and
    model performance metrics.
    
    Attributes:
        style: Matplotlib style to use for plots
        logger: Logger instance for this visualizer
        figure_size: Default figure size for plots
        dpi: Default DPI for saved figures
    """
    
    def __init__(
        self, 
        style: str = 'seaborn-v0_8',
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 300
    ):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figure_size: Default figure size (width, height)
            dpi: DPI for saved figures
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if requested style not available
            plt.style.use('default')
            self.logger.warning(f"Style '{style}' not available, using default")
        
        # Configure seaborn
        sns.set_palette("husl")
        
        self.logger.info(f"Visualizer initialized with style: {style}")
    
    def plot_training_history(
        self, 
        history: Any, 
        save_path: str = 'training_history.png',
        show_plot: bool = True
    ) -> None:
        """
        Plot comprehensive training history.
        
        Args:
            history: Training history object from Keras
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        self.logger.info("Plotting training history...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Loss plot
        self._plot_metric(
            axes[0, 0], history, 'loss', 'Model Loss', 
            xlabel='Epoch', ylabel='Loss'
        )
        
        # MAE plot
        if 'mae' in history.history:
            self._plot_metric(
                axes[0, 1], history, 'mae', 'Mean Absolute Error',
                xlabel='Epoch', ylabel='MAE'
            )
        else:
            axes[0, 1].text(0.5, 0.5, 'MAE not available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Mean Absolute Error')
        
        # SSIM plot
        ssim_key = self._find_ssim_key(history.history)
        if ssim_key:
            self._plot_metric(
                axes[1, 0], history, ssim_key, 'Structural Similarity Index',
                xlabel='Epoch', ylabel='SSIM'
            )
        else:
            axes[1, 0].text(0.5, 0.5, 'SSIM not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Structural Similarity Index')
        
        # Learning Rate plot
        if 'lr' in history.history:
            axes[1, 1].semilogy(history.history['lr'], linewidth=2, color='orange')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning rate not logged', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate')
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        self.logger.info(f"Training history saved as '{save_path}'")
    
    def _plot_metric(
        self, 
        ax: plt.Axes, 
        history: Any, 
        metric: str, 
        title: str,
        xlabel: str = 'Epoch',
        ylabel: str = 'Value'
    ) -> None:
        """
        Plot a single metric from training history.
        
        Args:
            ax: Matplotlib axes object
            history: Training history object
            metric: Metric name to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        train_key = metric
        val_key = f'val_{metric}'
        
        if train_key in history.history:
            ax.plot(history.history[train_key], label=f'Training {metric.upper()}', 
                   linewidth=2, alpha=0.8)
        
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f'Validation {metric.upper()}', 
                   linewidth=2, alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _find_ssim_key(self, history_dict: Dict) -> Optional[str]:
        """Find SSIM metric key in history dictionary."""
        possible_keys = ['_ssim_metric', 'ssim', 'ssim_metric', 'structural_similarity']
        for key in possible_keys:
            if key in history_dict:
                return key
        return None
    
    def plot_comparison(
        self, 
        original: np.ndarray, 
        cleaned: np.ndarray, 
        uncertainty: Optional[np.ndarray] = None,
        filename: str = 'comparison.png',
        show_plot: bool = True,
        title: Optional[str] = None
    ) -> None:
        """
        Plot comparison between original and cleaned data.
        
        Args:
            original: Original bathymetric data
            cleaned: Cleaned/processed data
            uncertainty: Optional uncertainty data
            filename: Output filename
            show_plot: Whether to display the plot
            title: Optional title for the plot
        """
        n_plots = 3 if uncertainty is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        if n_plots == 2:
            axes = [axes] if not hasattr(axes, '__len__') else axes
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Original data
        im1 = axes[0].imshow(original, cmap='viridis', aspect='equal')
        axes[0].set_title('Original Data')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Add statistics to original plot
        self._add_statistics_text(axes[0], original, 'Original')
        
        # Cleaned data
        im2 = axes[1].imshow(cleaned, cmap='viridis', aspect='equal')
        axes[1].set_title('Cleaned Data')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Add statistics to cleaned plot
        self._add_statistics_text(axes[1], cleaned, 'Cleaned')
        
        # Uncertainty (if available)
        if uncertainty is not None:
            im3 = axes[2].imshow(uncertainty, cmap='plasma', aspect='equal')
            axes[2].set_title('Uncertainty')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Add statistics to uncertainty plot
            self._add_statistics_text(axes[2], uncertainty, 'Uncertainty')
        
        plt.tight_layout()
        
        # Save plot
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        self.logger.info(f"Comparison plot saved as '{filename}'")
    
    def _add_statistics_text(self, ax: plt.Axes, data: np.ndarray, label: str) -> None:
        """Add statistical information as text on the plot."""
        stats_text = (
            f"{label} Stats:\n"
            f"Min: {np.min(data):.2f}\n"
            f"Max: {np.max(data):.2f}\n"
            f"Mean: {np.mean(data):.2f}\n"
            f"Std: {np.std(data):.2f}"
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_difference_map(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        filename: str = 'difference_map.png',
        show_plot: bool = True
    ) -> None:
        """
        Plot difference map between original and cleaned data.
        
        Args:
            original: Original data
            cleaned: Cleaned data
            filename: Output filename
            show_plot: Whether to display the plot
        """
        difference = cleaned - original
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Difference Analysis', fontsize=14, fontweight='bold')
        
        # Difference map
        im1 = axes[0].imshow(difference, cmap='RdBu_r', aspect='equal')
        axes[0].set_title('Difference Map (Cleaned - Original)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Histogram of differences
        axes[1].hist(difference.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].set_title('Distribution of Differences')
        axes[1].set_xlabel('Difference Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = (
            f"Difference Stats:\n"
            f"Mean: {np.mean(difference):.4f}\n"
            f"Std: {np.std(difference):.4f}\n"
            f"Min: {np.min(difference):.4f}\n"
            f"Max: {np.max(difference):.4f}\n"
            f"RMS: {np.sqrt(np.mean(difference**2)):.4f}"
        )
        
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        self.logger.info(f"Difference map saved as '{filename}'")
    
    def plot_data_distribution(
        self,
        data_dict: Dict[str, np.ndarray],
        filename: str = 'data_distribution.png',
        show_plot: bool = True
    ) -> None:
        """
        Plot data distribution comparison.
        
        Args:
            data_dict: Dictionary of data arrays with labels as keys
            filename: Output filename
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(data_dict)))
        
        # Histograms
        for i, (label, data) in enumerate(data_dict.items()):
            axes[0, 0].hist(data.flatten(), bins=50, alpha=0.6, 
                           label=label, color=colors[i])
        
        axes[0, 0].set_title('Data Distributions')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plots
        box_data = [data.flatten() for data in data_dict.values()]
        box_plot = axes[0, 1].boxplot(box_data, labels=list(data_dict.keys()), patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[0, 1].set_title('Box Plot Comparison')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plots (if only 2 datasets)
        if len(data_dict) == 2:
            data_items = list(data_dict.items())
            label1, data1 = data_items[0]
            label2, data2 = data_items[1]
            
            # Sample data for Q-Q plot if too large
            if data1.size > 10000:
                indices = np.random.choice(data1.size, 10000, replace=False)
                data1_sample = data1.flatten()[indices]
            else:
                data1_sample = data1.flatten()
            
            if data2.size > 10000:
                indices = np.random.choice(data2.size, 10000, replace=False)
                data2_sample = data2.flatten()[indices]
            else:
                data2_sample = data2.flatten()
            
            self._plot_qq(axes[1, 0], data1_sample, data2_sample, label1, label2)
        else:
            axes[1, 0].text(0.5, 0.5, 'Q-Q plot requires exactly 2 datasets',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Q-Q Plot')
        
        # Statistics table
        self._plot_statistics_table(axes[1, 1], data_dict)
        
        plt.tight_layout()
        
        # Save plot
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        self.logger.info(f"Data distribution plot saved as '{filename}'")
    
    def _plot_qq(self, ax: plt.Axes, data1: np.ndarray, data2: np.ndarray, 
                 label1: str, label2: str) -> None:
        """Plot Q-Q plot for two datasets."""
        from scipy import stats
        
        # Calculate quantiles
        quantiles = np.linspace(0, 1, min(len(data1), len(data2), 1000))
        q1 = np.quantile(data1, quantiles)
        q2 = np.quantile(data2, quantiles)
        
        # Plot Q-Q
        ax.scatter(q1, q2, alpha=0.6, s=20)
        
        # Add reference line
        min_val = min(np.min(q1), np.min(q2))
        max_val = max(np.max(q1), np.max(q2))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_xlabel(f'{label1} Quantiles')
        ax.set_ylabel(f'{label2} Quantiles')
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # Calculate correlation
        corr, _ = stats.pearsonr(q1, q2)
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_statistics_table(self, ax: plt.Axes, data_dict: Dict[str, np.ndarray]) -> None:
        """Plot statistics table."""
        ax.axis('off')
        
        # Calculate statistics
        stats_data = []
        for label, data in data_dict.items():
            flat_data = data.flatten()
            stats_data.append([
                label,
                f"{np.mean(flat_data):.3f}",
                f"{np.std(flat_data):.3f}",
                f"{np.min(flat_data):.3f}",
                f"{np.max(flat_data):.3f}",
                f"{np.median(flat_data):.3f}"
            ])
        
        # Create table
        headers = ['Dataset', 'Mean', 'Std', 'Min', 'Max', 'Median']
        table = ax.table(cellText=stats_data, colLabels=headers,
                        cellLoc='center', loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Statistical Summary', fontweight='bold')
    
    def plot_processing_summary(
        self,
        processing_stats: List[Dict],
        filename: str = 'processing_summary.png',
        show_plot: bool = True
    ) -> None:
        """
        Plot processing summary statistics.
        
        Args:
            processing_stats: List of processing statistics dictionaries
            filename: Output filename
            show_plot: Whether to display the plot
        """
        if not processing_stats:
            self.logger.warning("No processing statistics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Processing Summary', fontsize=16, fontweight='bold')
        
        # Extract data
        ssim_scores = [s.get('ssim', 0) for s in processing_stats if 'ssim' in s]
        filenames = [s.get('filename', f'File_{i}') for i, s in enumerate(processing_stats)]
        
        # SSIM distribution
        if ssim_scores:
            axes[0, 0].hist(ssim_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('SSIM Score Distribution')
            axes[0, 0].set_xlabel('SSIM Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add statistics
            mean_ssim = np.mean(ssim_scores)
            std_ssim = np.std(ssim_scores)
            axes[0, 0].axvline(mean_ssim, color='red', linestyle='--', 
                              label=f'Mean: {mean_ssim:.3f}')
            axes[0, 0].legend()
        
        # SSIM over files
        if ssim_scores and len(ssim_scores) <= 50:  # Only plot if not too many files
            axes[0, 1].bar(range(len(ssim_scores)), ssim_scores, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('SSIM Scores by File')
            axes[0, 1].set_xlabel('File Index')
            axes[0, 1].set_ylabel('SSIM Score')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Too many files to display individually',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('SSIM Scores by File')
        
        # Processing success rate
        total_files = len(processing_stats)
        successful_files = len([s for s in processing_stats if 'ssim' in s])
        success_rate = (successful_files / total_files) * 100 if total_files > 0 else 0
        
        labels = ['Successful', 'Failed']
        sizes = [successful_files, total_files - successful_files]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = axes[1, 0].pie(sizes, labels=labels, colors=colors, 
                                                 autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Processing Success Rate')
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
Processing Summary:
Total files: {total_files}
Successful: {successful_files}
Failed: {total_files - successful_files}
Success rate: {success_rate:.1f}%

SSIM Statistics:
Mean: {np.mean(ssim_scores):.4f}
Std: {np.std(ssim_scores):.4f}
Min: {np.min(ssim_scores):.4f}
Max: {np.max(ssim_scores):.4f}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        self.logger.info(f"Processing summary plot saved as '{filename}'")
    
    def create_report_figure(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        uncertainty: Optional[np.ndarray],
        history: Any,
        stats: Dict,
        filename: str = 'complete_report.png'
    ) -> None:
        """
        Create a comprehensive report figure.
        
        Args:
            original: Original data
            cleaned: Cleaned data
            uncertainty: Uncertainty data (optional)
            history: Training history
            stats: Processing statistics
            filename: Output filename
        """
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Bathymetric Processing Report', fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Data visualizations
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Original data
        im1 = ax1.imshow(original, cmap='viridis', aspect='equal')
        ax1.set_title('Original Data')
        ax1.axis('off')
        
        # Cleaned data
        im2 = ax2.imshow(cleaned, cmap='viridis', aspect='equal')
        ax2.set_title('Cleaned Data')
        ax2.axis('off')
        
        # Difference or uncertainty
        if uncertainty is not None:
            im3 = ax3.imshow(uncertainty, cmap='plasma', aspect='equal')
            ax3.set_title('Uncertainty')
        else:
            difference = cleaned - original
            im3 = ax3.imshow(difference, cmap='RdBu_r', aspect='equal')
            ax3.set_title('Difference')
        ax3.axis('off')
        
        # Training history (if available)
        if history is not None:
            ax4 = fig.add_subplot(gs[1, :2])
            if 'loss' in history.history:
                ax4.plot(history.history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in history.history:
                ax4.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            ax4.set_title('Training Loss')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Statistics
        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis('off')
        
        stats_text = self._format_stats_text(stats)
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Complete report saved as '{filename}'")
    
    def _format_stats_text(self, stats: Dict) -> str:
        """Format statistics dictionary as text."""
        lines = ["Processing Statistics:"]
        
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)


def setup_matplotlib_backend():
    """Setup matplotlib backend for headless environments."""
    import matplotlib
    
    # Try to use Agg backend for headless environments
    try:
        matplotlib.use('Agg')
    except:
        pass  # Use default backend if Agg not available


def save_plots_as_pdf(plot_files: List[str], output_pdf: str) -> None:
    """
    Combine multiple plot files into a single PDF.
    
    Args:
        plot_files: List of plot file paths
        output_pdf: Output PDF file path
    """
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(output_pdf) as pdf:
            for plot_file in plot_files:
                if Path(plot_file).exists():
                    fig = plt.figure()
                    img = plt.imread(plot_file)
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        logging.info(f"Combined plots saved to PDF: {output_pdf}")
        
    except ImportError:
        logging.warning("Cannot create PDF - matplotlib PDF backend not available")
    except Exception as e:
        logging.error(f"Error creating PDF: {e}")
