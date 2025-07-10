"""
Command-line interface utilities for Enhanced Bathymetric CAE Processing.

This module provides argument parsing and configuration management for the CLI.
"""

import argparse
from typing import Dict, Any

from ..config import Config


def create_argument_parser() -> argparse.ArgumentParser:
    """Create enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced Bathymetric Grid Processing using Advanced Ensemble CAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic usage with enhanced features
  python main.py
  
  # Custom paths with expert review enabled
  python main.py --input /path/to/input --output /path/to/output --enable-expert-review
  
  # Load custom configuration
  python main.py --config custom_config.json
  
  # Training with ensemble and adaptive processing
  python main.py --epochs 200 --ensemble-size 5 --enable-adaptive
        """
    )
    
    # I/O Arguments
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', '--input-folder', type=str, 
                         help='Path to input folder containing bathymetric files')
    io_group.add_argument('--output', '--output-folder', type=str,
                         help='Path to output folder for cleaned files')
    io_group.add_argument('--model', '--model-path', type=str,
                         help='Path to save/load model file')
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str,
                             help='Path to JSON configuration file')
    config_group.add_argument('--save-config', type=str,
                             help='Save current configuration to file')
    
    # Training Parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, help='Training batch size')
    train_group.add_argument('--learning-rate', type=float, help='Learning rate')
    train_group.add_argument('--validation-split', type=float, 
                            help='Validation split ratio (0-1)')
    
    # Model Architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--grid-size', type=int, help='Input grid size')
    model_group.add_argument('--base-filters', type=int, help='Base number of filters')
    model_group.add_argument('--depth', type=int, help='Model depth')
    model_group.add_argument('--dropout-rate', type=float, help='Dropout rate')
    model_group.add_argument('--ensemble-size', type=int, help='Number of models in ensemble')
    
    # Enhanced Features
    features_group = parser.add_argument_group('Enhanced Features')
    features_group.add_argument('--enable-adaptive', action='store_true',
                               help='Enable adaptive processing based on seafloor type')
    features_group.add_argument('--enable-expert-review', action='store_true',
                               help='Enable expert review system')
    features_group.add_argument('--enable-constitutional', action='store_true',
                               help='Enable constitutional AI constraints')
    features_group.add_argument('--quality-threshold', type=float,
                               help='Quality threshold for expert review flagging')
    features_group.add_argument('--enable-uncertainty', action='store_true',
                               help='Enable uncertainty estimation')
    
    # Processing Options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--max-workers', type=int, help='Maximum number of worker processes')
    proc_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                           help='Logging level')
    proc_group.add_argument('--no-gpu', action='store_true',
                           help='Disable GPU usage')
    proc_group.add_argument('--memory-limit', type=int,
                           help='GPU memory limit in MB')
    
    # Quality Metric Weights
    weights_group = parser.add_argument_group('Quality Metric Weights')
    weights_group.add_argument('--ssim-weight', type=float, help='SSIM weight in composite score')
    weights_group.add_argument('--roughness-weight', type=float, help='Roughness weight in composite score')
    weights_group.add_argument('--feature-weight', type=float, help='Feature preservation weight')
    weights_group.add_argument('--consistency-weight', type=float, help='Consistency weight')
    
    # Hydrographic Standards
    hydro_group = parser.add_argument_group('Hydrographic Standards')
    hydro_group.add_argument('--iho-standard', choices=['special_order', 'order_1a', 'order_1b', 'order_2'],
                            help='IHO standard for hydrographic compliance checking')
    
    # Advanced Options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument('--cross-validate', action='store_true',
                               help='Perform cross-validation during training')
    advanced_group.add_argument('--cv-folds', type=int, default=5,
                               help='Number of cross-validation folds')
    advanced_group.add_argument('--early-stopping-patience', type=int,
                               help='Early stopping patience')
    advanced_group.add_argument('--reduce-lr-patience', type=int,
                               help='Reduce learning rate patience')
    advanced_group.add_argument('--mixed-precision', action='store_true',
                               help='Enable mixed precision training')
    
    # Output Options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--save-plots', action='store_true',
                             help='Save visualization plots')
    output_group.add_argument('--save-metrics', action='store_true',
                             help='Save detailed metrics to CSV')
    output_group.add_argument('--output-format', choices=['tif', 'bag', 'asc'],
                             help='Output file format')
    
    return parser


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Update configuration with command line arguments."""
    # Create argument mapping
    arg_mapping = {
        # Basic parameters
        'epochs': 'epochs',
        'batch_size': 'batch_size',
        'learning_rate': 'learning_rate',
        'validation_split': 'validation_split',
        
        # Model architecture
        'grid_size': 'grid_size',
        'base_filters': 'base_filters',
        'depth': 'depth',
        'dropout_rate': 'dropout_rate',
        'ensemble_size': 'ensemble_size',
        
        # Processing options
        'max_workers': 'max_workers',
        'log_level': 'log_level',
        
        # Enhanced features
        'enable_adaptive': 'enable_adaptive_processing',
        'enable_expert_review': 'enable_expert_review',
        'enable_constitutional': 'enable_constitutional_constraints',
        'enable_uncertainty': 'enable_uncertainty_estimation',
        'quality_threshold': 'quality_threshold',
        
        # Quality weights
        'ssim_weight': 'ssim_weight',
        'roughness_weight': 'roughness_weight',
        'feature_weight': 'feature_preservation_weight',
        'consistency_weight': 'consistency_weight',
        
        # Callbacks
        'early_stopping_patience': 'early_stopping_patience',
        'reduce_lr_patience': 'reduce_lr_patience',
        
        # Performance
        'mixed_precision': 'use_mixed_precision'
    }
    
    # Update configuration with non-None arguments
    updates = {}
    for arg_name, config_attr in arg_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            updates[config_attr] = arg_value
    
    # Handle special cases
    if hasattr(args, 'iho_standard') and args.iho_standard:
        updates['iho_standard'] = args.iho_standard
    
    # Apply updates
    if updates:
        config.update_from_dict(updates)
    
    return config


def validate_arguments(args: argparse.Namespace) -> Dict[str, str]:
    """Validate command line arguments and return any errors."""
    errors = {}
    
    # Validate numeric ranges
    if hasattr(args, 'epochs') and args.epochs is not None:
        if args.epochs <= 0:
            errors['epochs'] = "Must be positive"
    
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        if args.batch_size <= 0:
            errors['batch_size'] = "Must be positive"
    
    if hasattr(args, 'validation_split') and args.validation_split is not None:
        if not 0 < args.validation_split < 1:
            errors['validation_split'] = "Must be between 0 and 1"
    
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        if args.learning_rate <= 0:
            errors['learning_rate'] = "Must be positive"
    
    if hasattr(args, 'dropout_rate') and args.dropout_rate is not None:
        if not 0 <= args.dropout_rate < 1:
            errors['dropout_rate'] = "Must be between 0 and 1"
    
    if hasattr(args, 'quality_threshold') and args.quality_threshold is not None:
        if not 0 <= args.quality_threshold <= 1:
            errors['quality_threshold'] = "Must be between 0 and 1"
    
    # Validate quality weights sum to 1.0 if all provided
    quality_weights = []
    weight_args = ['ssim_weight', 'roughness_weight', 'feature_weight', 'consistency_weight']
    
    for weight_arg in weight_args:
        if hasattr(args, weight_arg):
            weight_value = getattr(args, weight_arg)
            if weight_value is not None:
                if weight_value < 0:
                    errors[weight_arg] = "Must be non-negative"
                quality_weights.append(weight_value)
            else:
                quality_weights.append(None)
    
    # Check if all weights are provided and sum to 1.0
    if all(w is not None for w in quality_weights):
        weight_sum = sum(quality_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            errors['quality_weights'] = f"Quality weights must sum to 1.0, got {weight_sum:.3f}"
    
    # Validate file paths if provided
    if hasattr(args, 'input') and args.input is not None:
        from pathlib import Path
        if not Path(args.input).exists():
            errors['input'] = f"Input folder does not exist: {args.input}"
    
    if hasattr(args, 'config') and args.config is not None:
        from pathlib import Path
        if not Path(args.config).exists():
            errors['config'] = f"Configuration file does not exist: {args.config}"
    
    return errors


def print_argument_summary(args: argparse.Namespace):
    """Print a summary of the provided arguments."""
    print("\nArgument Summary:")
    print("-" * 40)
    
    # Group arguments by category
    categories = {
        'Input/Output': ['input', 'output', 'model', 'config'],
        'Training': ['epochs', 'batch_size', 'learning_rate', 'validation_split'],
        'Model Architecture': ['grid_size', 'base_filters', 'depth', 'dropout_rate', 'ensemble_size'],
        'Enhanced Features': ['enable_adaptive', 'enable_expert_review', 'enable_constitutional', 'enable_uncertainty'],
        'Processing': ['max_workers', 'log_level', 'no_gpu'],
        'Quality Weights': ['ssim_weight', 'roughness_weight', 'feature_weight', 'consistency_weight']
    }
    
    for category, arg_names in categories.items():
        category_args = []
        for arg_name in arg_names:
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    category_args.append(f"  {arg_name}: {value}")
        
        if category_args:
            print(f"\n{category}:")
            for arg_str in category_args:
                print(arg_str)


def create_config_from_args(args: argparse.Namespace) -> Config:
    """Create a new configuration from command line arguments."""
    config = Config()
    config = update_config_from_args(config, args)
    return config


def handle_config_conflicts(config: Config, args: argparse.Namespace) -> Config:
    """Handle conflicts between config file and command line arguments."""
    # Command line arguments take precedence over config file
    # This is already handled in update_config_from_args, but we can add
    # logging here to inform the user about overrides
    
    import logging
    logger = logging.getLogger(__name__)
    
    overrides = []
    
    # Check for common overrides
    if hasattr(args, 'epochs') and args.epochs is not None:
        if args.epochs != config.epochs:
            overrides.append(f"epochs: {config.epochs} -> {args.epochs}")
    
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        if args.batch_size != config.batch_size:
            overrides.append(f"batch_size: {config.batch_size} -> {args.batch_size}")
    
    if hasattr(args, 'ensemble_size') and args.ensemble_size is not None:
        if args.ensemble_size != config.ensemble_size:
            overrides.append(f"ensemble_size: {config.ensemble_size} -> {args.ensemble_size}")
    
    if overrides:
        logger.info("Command line arguments override config file:")
        for override in overrides:
            logger.info(f"  {override}")
    
    return config


def setup_argument_groups(parser: argparse.ArgumentParser) -> Dict[str, argparse._ArgumentGroup]:
    """Setup and return argument groups for organized help display."""
    groups = {}
    
    # Define argument groups
    group_definitions = [
        ('input_output', 'Input/Output Options'),
        ('configuration', 'Configuration Management'),
        ('training', 'Training Parameters'),
        ('model', 'Model Architecture'),
        ('features', 'Enhanced Features'),
        ('processing', 'Processing Options'),
        ('quality', 'Quality Metrics'),
        ('hydrographic', 'Hydrographic Standards'),
        ('advanced', 'Advanced Options'),
        ('output', 'Output Options')
    ]
    
    for group_name, group_title in group_definitions:
        groups[group_name] = parser.add_argument_group(group_title)
    
    return groups


def add_expert_review_arguments(parser: argparse.ArgumentParser):
    """Add expert review specific arguments."""
    expert_group = parser.add_argument_group('Expert Review System')
    
    expert_group.add_argument('--review-threshold', type=float, default=0.7,
                             help='Quality threshold for automatic expert review flagging')
    expert_group.add_argument('--review-db-path', type=str,
                             help='Path to expert review database')
    expert_group.add_argument('--auto-flag-low-quality', action='store_true',
                             help='Automatically flag low quality results for review')
    expert_group.add_argument('--review-report', action='store_true',
                             help='Generate expert review report')


def add_adaptive_processing_arguments(parser: argparse.ArgumentParser):
    """Add adaptive processing specific arguments."""
    adaptive_group = parser.add_argument_group('Adaptive Processing')
    
    adaptive_group.add_argument('--force-seafloor-type', 
                               choices=['shallow_coastal', 'deep_ocean', 'continental_shelf', 
                                       'seamount', 'abyssal_plain'],
                               help='Force specific seafloor type (override classification)')
    adaptive_group.add_argument('--adaptive-smoothing', action='store_true',
                               help='Enable adaptive smoothing based on local characteristics')
    adaptive_group.add_argument('--local-analysis', action='store_true', default=True,
                               help='Enable local characteristics analysis')


def add_visualization_arguments(parser: argparse.ArgumentParser):
    """Add visualization specific arguments."""
    viz_group = parser.add_argument_group('Visualization Options')
    
    viz_group.add_argument('--create-plots', action='store_true',
                          help='Create visualization plots')
    viz_group.add_argument('--plot-format', choices=['png', 'pdf', 'svg'], default='png',
                          help='Format for saved plots')
    viz_group.add_argument('--plot-dpi', type=int, default=300,
                          help='DPI for saved plots')
    viz_group.add_argument('--show-plots', action='store_true',
                          help='Display plots interactively')


def create_comprehensive_parser() -> argparse.ArgumentParser:
    """Create a comprehensive argument parser with all options."""
    parser = create_argument_parser()
    
    # Add specialized argument groups
    add_expert_review_arguments(parser)
    add_adaptive_processing_arguments(parser)
    add_visualization_arguments(parser)
    
    return parser


def parse_and_validate_arguments() -> tuple:
    """Parse and validate command line arguments."""
    parser = create_comprehensive_parser()
    args = parser.parse_args()
    
    # Validate arguments
    errors = validate_arguments(args)
    
    if errors:
        print("Argument validation errors:")
        for arg, error in errors.items():
            print(f"  {arg}: {error}")
        parser.print_help()
        return None, errors
    
    return args, None


# Convenience function for simple CLI usage
def get_validated_config() -> tuple:
    """Get validated configuration from command line arguments."""
    args, errors = parse_and_validate_arguments()
    
    if errors:
        return None, errors
    
    # Load or create configuration
    if args.config:
        try:
            from ..config import load_config_from_file
            config = load_config_from_file(args.config)
        except Exception as e:
            return None, {'config': f"Error loading config file: {e}"}
    else:
        config = Config()
    
    # Update with command line arguments
    config = update_config_from_args(config, args)
    
    return (config, args), None