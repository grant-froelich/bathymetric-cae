#!/usr/bin/env python3
"""
Main entry point for Enhanced Bathymetric CAE Processing.

This script provides the command-line interface and orchestrates the entire
processing pipeline.
"""

import sys
import logging
from pathlib import Path

# Add the current directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, load_config_from_file
from processing import EnhancedBathymetricCAEPipeline
from utils import setup_logging, create_argument_parser, update_config_from_args
from utils.memory_utils import optimize_gpu_memory


def main():
    """Enhanced main function with comprehensive error handling."""
    exit_code = 0
    
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Load configuration
        if args.config:
            try:
                config = load_config_from_file(args.config)
                print(f"Loaded configuration from: {args.config}")
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
                config = Config()
        else:
            config = Config()
        
        # Update configuration with command line arguments
        config = update_config_from_args(config, args)
        
        # Save configuration if requested
        if args.save_config:
            try:
                config.save(args.save_config)
                print(f"Configuration saved to: {args.save_config}")
            except Exception as e:
                print(f"Error saving configuration: {e}")
        
        # Setup logging
        setup_logging(config.log_level)
        logger = logging.getLogger(__name__)
        
        # Log startup information
        logger.info("="*60)
        logger.info("Enhanced Bathymetric CAE Pipeline v2.0")
        logger.info("="*60)
        
        # Check TensorFlow availability
        try:
            import tensorflow as tf
            logger.info(f"TensorFlow version: {tf.__version__}")
            logger.info(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
            
            # Disable GPU if requested
            if args.no_gpu:
                tf.config.set_visible_devices([], 'GPU')
                logger.info("GPU disabled by user request")
        
        except ImportError:
            logger.error("TensorFlow not available")
            return 1
        
        # Setup GPU memory optimization
        try:
            optimize_gpu_memory()
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
        
        # Log enabled features
        features_enabled = []
        if config.enable_adaptive_processing:
            features_enabled.append("Adaptive Processing")
        if config.enable_expert_review:
            features_enabled.append("Expert Review System")
        if config.enable_constitutional_constraints:
            features_enabled.append("Constitutional Constraints")
        
        logger.info(f"Enhanced features enabled: {', '.join(features_enabled) if features_enabled else 'None'}")
        
        # Validate required paths
        input_folder = getattr(args, 'input', None) or config.input_folder
        output_folder = getattr(args, 'output', None) or config.output_folder
        model_path = getattr(args, 'model', None) or config.model_path
        
        if not input_folder or not output_folder:
            logger.error("Input and output folders must be specified")
            return 1
        
        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Model path: {model_path}")
        
        # Validate input folder exists
        if not Path(input_folder).exists():
            logger.error(f"Input folder does not exist: {input_folder}")
            return 1
        
        # Create and run the enhanced pipeline
        logger.info("Initializing enhanced processing pipeline...")
        
        pipeline = EnhancedBathymetricCAEPipeline(config)
        
        logger.info("Starting processing pipeline...")
        pipeline.run(input_folder, output_folder, model_path)
        
        logger.info("="*60)
        logger.info("Enhanced processing pipeline completed successfully!")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        exit_code = 1
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Pipeline failed with error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        print(f"Error: {e}")
        exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
