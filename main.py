#!/usr/bin/env python3
"""
Enhanced Bathymetric CAE Processing - Main Entry Point
"""

import os
import warnings
import logging
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Enable mixed precision for better GPU performance
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

from config.config import Config
from cli.interface import create_argument_parser, update_config_from_args
from processing.pipeline import EnhancedBathymetricCAEPipeline
from utils.logging_utils import setup_logging


def main():
    """Enhanced main function with comprehensive error handling."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = Config.load(args.config)
        else:
            config = Config()
        
        # Update with command line arguments
        config = update_config_from_args(config, args)
        
        # Save configuration if requested
        if args.save_config:
            config.save(args.save_config)
            print(f"Configuration saved to {args.save_config}")
        
        # Setup logging
        setup_logging(config.log_level)
        logger = logging.getLogger(__name__)
        
        # Log system information
        logger.info(f"Starting Enhanced Bathymetric CAE Pipeline v2.0")
        logger.info(f"Python version: {tf.__version__}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        
        # Log enabled features
        features_enabled = []
        if config.enable_adaptive_processing:
            features_enabled.append("Adaptive Processing")
        if config.enable_expert_review:
            features_enabled.append("Expert Review")
        if config.enable_constitutional_constraints:
            features_enabled.append("Constitutional Constraints")
        
        logger.info(f"Enhanced features enabled: {', '.join(features_enabled) if features_enabled else 'None'}")
        
        # Disable GPU if requested
        if args.no_gpu:
            tf.config.set_visible_devices([], 'GPU')
            logger.info("GPU disabled by user request")
        
        # Set paths
        input_folder = args.input or config.input_folder
        output_folder = args.output or config.output_folder
        model_path = args.model or config.model_path
        
        # Create and run enhanced pipeline
        pipeline = EnhancedBathymetricCAEPipeline(config)
        pipeline.run(input_folder, output_folder, model_path)
        
        logger.info("Enhanced processing pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Enhanced pipeline failed: {e}")
        logging.debug("Full traceback:", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())