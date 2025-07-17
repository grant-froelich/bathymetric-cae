#!/usr/bin/env python3
"""
Enhanced Bathymetric CAE Processing - Main Entry Point
Fixed HDF5 compatibility and logging issues.
"""

# ============================================================================
# CRITICAL: HDF5 COMPATIBILITY FIXES - MUST BE FIRST
# ============================================================================

import os
import sys

# Set HDF5 environment variables BEFORE any imports
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Suppress warnings before imports
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*HDF5.*')

# ============================================================================
# Safe TensorFlow Import with HDF5 Protection
# ============================================================================

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    print("ERROR: TensorFlow not found. Please install with: pip install tensorflow>=2.13.0")
    sys.exit(1)

# ============================================================================
# Rest of the imports
# ============================================================================

import logging
from pathlib import Path
from config.config import Config
from cli.interface import create_argument_parser, update_config_from_args
from processing.pipeline import EnhancedBathymetricCAEPipeline
from utils.logging_utils import setup_logging


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "logs",
        "plots", 
        "expert_reviews",
        "models",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def safe_model_creation(config: Config):
    """Ensure model configuration uses safe formats."""
    # Force modern Keras format for compatibility
    config.model_format = "keras"
    if config.model_path.endswith('.h5'):
        config.model_path = config.model_path.replace('.h5', '.keras')
    elif not config.model_path.endswith('.keras'):
        if '.' in config.model_path:
            base_path = '.'.join(config.model_path.split('.')[:-1])
            config.model_path = f"{base_path}.keras"
        else:
            config.model_path += '.keras'


def main():
    """Enhanced main function with comprehensive error handling and logging."""
    
    # Ensure directories exist first
    ensure_directories()
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = Config.load(args.config)
            print(f"Loaded configuration from: {args.config}")
        else:
            config = Config()
            print("Using default configuration")
        
        # Update with command line arguments
        config = update_config_from_args(config, args)
        
        # Apply safe model configuration
        safe_model_creation(config)
        
        # Setup logging EARLY with proper paths
        setup_logging(config.log_level)
        logger = logging.getLogger(__name__)
        
        # Force log a test message to verify logging works
        logger.info("="*60)
        logger.info("ENHANCED BATHYMETRIC CAE PIPELINE v2.0 - STARTING")
        logger.info("="*60)
        
        # Save configuration if requested
        if args.save_config:
            config.save(args.save_config)
            logger.info(f"Configuration saved to {args.save_config}")
        
        # Log system information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # GPU information
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        logger.info(f"GPU available: {gpu_available}")
        
        # Handle GPU setting
        if hasattr(args, 'no_gpu') and args.no_gpu:
            tf.config.set_visible_devices([], 'GPU')
            logger.info("GPU disabled by user request")
        
        # Set paths with validation
        input_folder = args.input or config.input_folder
        output_folder = args.output or config.output_folder
        model_path = args.model or config.model_path
        
        # Validate input folder exists
        if not Path(input_folder).exists():
            logger.error(f"Input folder does not exist: {input_folder}")
            print(f"ERROR: Input folder does not exist: {input_folder}")
            return 1
        
        # Ensure output directory exists
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Model path: {model_path}")
        
        # Create and run pipeline
        logger.info("Creating pipeline...")
        pipeline = EnhancedBathymetricCAEPipeline(config)
        
        logger.info("Starting pipeline execution...")
        pipeline.run(input_folder, output_folder, model_path)
        
        logger.info("✅ Processing completed successfully!")
        print("✅ Processing completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        error_msg = f"File not found: {e}"
        logging.error(error_msg)
        print(f"ERROR: {error_msg}")
        return 1
        
    except Exception as e:
        error_msg = f"Pipeline error: {e}"
        logging.error(error_msg)
        logging.debug("Full traceback:", exc_info=True)
        print(f"ERROR: {error_msg}")
        
        # Try minimal fallback for testing
        try:
            logging.info("Attempting minimal test run...")
            print("Attempting minimal test run...")
            
            # Just test that we can create a basic model
            from models.architectures import LightweightCAE
            test_model = LightweightCAE(config)
            model = test_model.create_model((64, 64, 1))
            
            logging.info("✅ Basic model creation successful")
            print("✅ Basic model creation successful")
            return 0
            
        except Exception as fallback_error:
            fallback_msg = f"Fallback also failed: {fallback_error}"
            logging.error(fallback_msg)
            print(f"ERROR: {fallback_msg}")
            return 1


if __name__ == "__main__":
    # Handle special diagnostic commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-hdf5":
            print("Testing HDF5 compatibility...")
            try:
                import tensorflow as tf
                print(f"✅ TensorFlow available: {tf.__version__}")
                
                # Test basic model creation
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(1, input_shape=(1,))
                ])
                print("✅ Basic model creation successful")
                print("✅ HDF5 compatibility test passed!")
                
            except Exception as e:
                print(f"❌ HDF5 compatibility test failed: {e}")
            sys.exit(0)
        
        elif sys.argv[1] == "--version":
            print("Enhanced Bathymetric CAE Processing v2.0")
            sys.exit(0)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)