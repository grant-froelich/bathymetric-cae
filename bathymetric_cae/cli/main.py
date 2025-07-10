"""
Command Line Interface Module

This module provides the main command-line interface for the bathymetric CAE
processing pipeline, including argument parsing and execution management.

Author: Bathymetric CAE Team
License: MIT
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from ..config.config import Config, create_config_from_args
from ..core.pipeline import BathymetricCAEPipeline, validate_pipeline_requirements
from ..utils.logging_utils import setup_logging, log_system_info, silence_warnings
from ..utils.gpu_utils import disable_gpu


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create enhanced command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Bathymetric Grid Processing using Advanced Convolutional Autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default configuration
  python -m bathymetric_cae
  
  # Custom paths
  python -m bathymetric_cae --input /path/to/input --output /path/to/output
  
  # Load custom configuration
  python -m bathymetric_cae --config custom_config.json
  
  # Training with custom parameters
  python -m bathymetric_cae --epochs 200 --batch-size 16 --learning-rate 0.0001
  
  # Process single file interactively
  python -m bathymetric_cae --single-file input.bag --model trained_model.h5
  
  # Validate requirements only
  python -m bathymetric_cae --validate-requirements
        """
    )
    
    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--single-file', type=str, metavar='FILE',
        help='Process a single file interactively'
    )
    mode_group.add_argument(
        '--validate-requirements', action='store_true',
        help='Validate system requirements and exit'
    )
    
    # I/O Arguments
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        '--input', '--input-folder', type=str, dest='input_folder',
        help='Path to input folder containing bathymetric files'
    )
    io_group.add_argument(
        '--output', '--output-folder', type=str, dest='output_folder',
        help='Path to output folder for cleaned files'
    )
    io_group.add_argument(
        '--model', '--model-path', type=str, dest='model_path',
        help='Path to save/load model file'
    )
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', type=str,
        help='Path to JSON configuration file'
    )
    config_group.add_argument(
        '--save-config', type=str,
        help='Save current configuration to file and exit'
    )
    
    # Training Parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument(
        '--epochs', type=int,
        help='Number of training epochs'
    )
    train_group.add_argument(
        '--batch-size', type=int, dest='batch_size',
        help='Training batch size'
    )
    train_group.add_argument(
        '--learning-rate', type=float, dest='learning_rate',
        help='Learning rate'
    )
    train_group.add_argument(
        '--validation-split', type=float, dest='validation_split',
        help='Validation split ratio (0-1)'
    )
    train_group.add_argument(
        '--force-retrain', action='store_true',
        help='Force model retraining even if model exists'
    )
    
    # Model Architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        '--grid-size', type=int, dest='grid_size',
        help='Input grid size'
    )
    model_group.add_argument(
        '--base-filters', type=int, dest='base_filters',
        help='Base number of filters'
    )
    model_group.add_argument(
        '--depth', type=int,
        help='Model depth'
    )
    model_group.add_argument(
        '--dropout-rate', type=float, dest='dropout_rate',
        help='Dropout rate'
    )
    
    # Processing Options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument(
        '--max-workers', type=int, dest='max_workers',
        help='Maximum number of worker processes'
    )
    proc_group.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], dest='log_level',
        help='Logging level'
    )
    proc_group.add_argument(
        '--no-gpu', action='store_true',
        help='Disable GPU usage'
    )
    proc_group.add_argument(
        '--silent', action='store_true',
        help='Suppress all non-error output'
    )
    proc_group.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output (sets log level to DEBUG)'
    )
    
    # Visualization Options
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument(
        '--no-plots', action='store_true',
        help='Disable plot generation'
    )
    viz_group.add_argument(
        '--show-plots', action='store_true',
        help='Show plots interactively (for single file processing)'
    )
    
    return parser


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """
    Setup logging based on command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    # Determine log level
    if args.verbose:
        log_level = 'DEBUG'
    elif args.silent:
        log_level = 'ERROR'
    elif hasattr(args, 'log_level') and args.log_level:
        log_level = args.log_level
    else:
        log_level = 'INFO'
    
    # Setup logging
    console_output = not args.silent
    setup_logging(
        log_level=log_level,
        console_output=console_output,
        file_output=True,
        colored_output=console_output
    )
    
    # Silence warnings if not in debug mode
    if log_level != 'DEBUG':
        silence_warnings()


def validate_requirements_command() -> int:
    """
    Validate system requirements and return exit code.
    
    Returns:
        int: Exit code (0 if all requirements met, 1 otherwise)
    """
    print("Validating system requirements...")
    print("=" * 50)
    
    requirements = validate_pipeline_requirements()
    
    all_good = True
    
    # Check each requirement
    for req, status in requirements.items():
        if req == 'all_requirements_met':
            continue
        
        if req.endswith('_version'):
            continue
        
        status_str = "✓ OK" if status else "✗ MISSING"
        print(f"{req:<20}: {status_str}")
        
        if req.endswith('_version') and req.replace('_version', '') in requirements:
            version = requirements[req]
            print(f"{'Version':<20}: {version}")
        
        if not status:
            all_good = False
    
    print("=" * 50)
    
    if all_good:
        print("✓ All requirements satisfied!")
        return 0
    else:
        print("✗ Some requirements are missing. Please install missing dependencies.")
        return 1


def process_single_file_command(
    file_path: str, 
    config: Config, 
    show_plots: bool = False
) -> int:
    """
    Process a single file interactively.
    
    Args:
        file_path: Path to file to process
        config: Configuration object
        show_plots: Whether to show plots
        
    Returns:
        int: Exit code
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Processing single file: {file_path}")
        
        # Validate file exists
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return 1
        
        # Validate model exists
        if not Path(config.model_path).exists():
            logger.error(f"Model file not found: {config.model_path}")
            logger.info("Please train a model first using batch processing mode")
            return 1
        
        # Create pipeline and process file
        pipeline = BathymetricCAEPipeline(config)
        
        results = pipeline.process_single_file_interactive(
            file_path=file_path,
            model_path=config.model_path,
            show_plots=show_plots
        )
        
        if results['processing_successful']:
            logger.info("Single file processing completed successfully!")
            return 0
        else:
            logger.error(f"Processing failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logging.error(f"Single file processing failed: {e}")
        return 1


def run_main_pipeline(config: Config, force_retrain: bool = False) -> int:
    """
    Run the main processing pipeline.
    
    Args:
        config: Configuration object
        force_retrain: Force model retraining
        
    Returns:
        int: Exit code
    """
    try:
        logger = logging.getLogger(__name__)
        
        # Log system information
        log_system_info()
        
        # Create and run pipeline
        pipeline = BathymetricCAEPipeline(config)
        
        results = pipeline.run(
            input_folder=config.input_folder,
            output_folder=config.output_folder,
            model_path=config.model_path,
            force_retrain=force_retrain
        )
        
        # Log final results
        logger.info("Pipeline execution completed successfully!")
        logger.info(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        logging.debug("Full traceback:", exc_info=True)
        return 1


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command line arguments (uses sys.argv if None)
        
    Returns:
        int: Exit code
    """
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    
    # Setup logging first
    setup_logging_from_args(args)
    logger = logging.getLogger(__name__)
    
    try:
        # Handle special commands first
        if args.validate_requirements:
            return validate_requirements_command()
        
        # Load or create configuration
        if args.config:
            try:
                config = Config.load(args.config)
                logger.info(f"Loaded configuration from: {args.config}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                return 1
        else:
            config = Config()
        
        # Update configuration with command line arguments
        config = create_config_from_args(config, args)
        
        # Save configuration if requested
        if args.save_config:
            try:
                config.save(args.save_config)
                print(f"Configuration saved to {args.save_config}")
                return 0
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
                return 1
        
        # Configure GPU settings
        if args.no_gpu:
            logger.info("GPU disabled by user request")
            disable_gpu()
        
        # Handle single file processing
        if args.single_file:
            return process_single_file_command(
                file_path=args.single_file,
                config=config,
                show_plots=args.show_plots
            )
        
        # Run main pipeline
        return run_main_pipeline(config, args.force_retrain)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


def create_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Base configuration
        args: Parsed command line arguments
        
    Returns:
        Config: Updated configuration
    """
    # Update only non-None arguments
    for key, value in vars(args).items():
        if value is not None:
            # Convert argument names to config attribute names
            config_key = key.replace('-', '_')
            if hasattr(config, config_key):
                setattr(config, config_key, value)
    
    return config


# Entry point for package execution
if __name__ == "__main__":
    sys.exit(main())
