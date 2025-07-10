# visualization/__init__.py
"""Visualization module for bathymetric CAE package."""

from .visualizer import Visualizer, setup_matplotlib_backend, save_plots_as_pdf

__all__ = ['Visualizer', 'setup_matplotlib_backend', 'save_plots_as_pdf']

# ================================================================================