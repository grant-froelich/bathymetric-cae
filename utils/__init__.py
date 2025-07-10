# utils/__init__.py  
"""Utilities for Enhanced Bathymetric CAE Processing."""

from .logging_utils import (
    setup_logging,
    get_logger,
    get_performance_logger,
    get_progress_logger,
    log_system_info,
    ColoredFormatter,
    JSONFormatter,
    PerformanceLogger,
    ProcessingProgressLogger,
    create_memory_usage_logger
)

from .cli import (
    create_argument_parser,
    update_config_from_args,
    validate_arguments,
    print_argument_summary,
    create_config_from_args,
    parse_and_validate_arguments,
    get_validated_config
)

from .visualization import (
    create_enhanced_visualization,
    create_comparison_plot,
    create_quality_dashboard,
    plot_training_history,
    save_processing_plots,
    create_ensemble_comparison_plot,
    create_seafloor_classification_plot,
    configure_plotting,
    cleanup_plots
)

__all__ = [
    # Logging
    'setup_logging',
    'get_logger', 
    'get_performance_logger',
    'get_progress_logger',
    'log_system_info',
    'ColoredFormatter',
    'JSONFormatter',
    'PerformanceLogger',
    'ProcessingProgressLogger',
    'create_memory_usage_logger',
    
    # CLI
    'create_argument_parser',
    'update_config_from_args',
    'validate_arguments',
    'print_argument_summary',
    'create_config_from_args', 
    'parse_and_validate_arguments',
    'get_validated_config',
    
    # Visualization
    'create_enhanced_visualization',
    'create_comparison_plot',
    'create_quality_dashboard',
    'plot_training_history',
    'save_processing_plots',
    'create_ensemble_comparison_plot',
    'create_seafloor_classification_plot',
    'configure_plotting',
    'cleanup_plots'
]