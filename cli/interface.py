"""
Command line interface for Enhanced Bathymetric CAE Processing.
"""

import argparse
import logging
from config.config import Config


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
    
    # Feature preservation arguments
    model_group.add_argument('--model-variant', choices=['advanced', 'uncertainty', 'lightweight', 'feature_preserving'], 
                            default='feature_preserving', help='Model architecture variant to use')
    model_group.add_argument('--preserve-anthropogenic', action='store_true', default=True, 
                            help='Enable preservation of man-made features')
    model_group.add_argument('--preserve-geological', action='store_true', default=True, 
                            help='Enable preservation of geological features')
    model_group.add_argument('--use-edge-preserving-loss', action='store_true', default=True, 
                            help='Use custom edge-preserving loss function')
    model_group.add_argument('--anthropogenic-priority', type=float, default=1.0, 
                            help='Priority weight for anthropogenic features (0.0-1.0)')
    
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
    
    # Processing Options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--max-workers', type=int, help='Maximum number of worker processes')
    proc_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                           help='Logging level')
    proc_group.add_argument('--no-gpu', action='store_true',
                           help='Disable GPU usage')
    
    # Quality Metric Weights
    weights_group = parser.add_argument_group('Quality Metric Weights')
    weights_group.add_argument('--ssim-weight', type=float, help='SSIM weight in composite score')
    weights_group.add_argument('--roughness-weight', type=float, help='Roughness weight in composite score')
    weights_group.add_argument('--feature-weight', type=float, help='Feature preservation weight')
    weights_group.add_argument('--consistency-weight', type=float, help='Consistency weight')
    
    return parser


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Update configuration with command line arguments."""
    # Argument name mappings
    arg_mappings = {
        'input': 'input_folder',
        'output': 'output_folder',
        'model': 'model_path',
        'feature_weight': 'feature_preservation_weight',
        'enable_adaptive': 'enable_adaptive_processing',
        'enable_constitutional': 'enable_constitutional_constraints'
    }
    
    # Special handling for flags that don't map directly to config
    special_flags = {'no_gpu', 'save_config', 'config'}
    
    for key, value in vars(args).items():
        if value is not None and key not in special_flags:
            # Use mapping if available, otherwise convert dashes
            config_key = arg_mappings.get(key, key.replace('-', '_'))
            
            if hasattr(config, config_key):
                setattr(config, config_key, value)
            else:
                logging.warning(f"Unknown config attribute: {config_key}")
    
    return config