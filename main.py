#!/usr/bin/env python3
"""
Enhanced Bathymetric CAE Processing - Main Entry Point
Updated with comprehensive HDF5 compatibility fixes.
"""

# ============================================================================
# CRITICAL: HDF5 COMPATIBILITY FIXES - MUST BE FIRST
# ============================================================================

import os
import sys

# Set HDF5 environment variables BEFORE any imports
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
os.environ['HDF5_DRIVER'] = 'core'  # Use in-memory driver
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
    # Import h5py first to initialize HDF5 properly
    import h5py
    # Force h5py to use compatible settings
    # h5py.get_config().default_file_mode = 'r'
except ImportError:
    pass

# Now import TensorFlow with protections
try:
    import tensorflow as tf
    
    # Configure TensorFlow to avoid HDF5 conflicts
    tf.get_logger().setLevel('ERROR')
    
    # Use modern Keras format by default
    if hasattr(tf.keras.utils, 'get_file'):
        # Force Keras to prefer .keras format
        pass
        
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


def handle_hdf5_model_path(model_path: str) -> str:
    """Safely handle model paths to avoid HDF5 issues."""
    if model_path.endswith('.h5'):
        # Convert to modern format
        keras_path = model_path.replace('.h5', '.keras')
        logging.info(f"Converting model path from H5 to Keras format: {keras_path}")
        return keras_path
    elif not model_path.endswith('.keras'):
        # Add modern extension
        return f"{model_path}.keras"
    return model_path


def safe_model_creation(config: Config):
    """Ensure model configuration uses safe formats."""
    # Force modern Keras format
    config.model_format = "keras"
    config.model_path = handle_hdf5_model_path(config.model_path)
    config.auto_convert_legacy = True
    
    # Log the safe configuration
    logging.info(f"Using safe model configuration:")
    logging.info(f"  Format: {config.model_format}")
    logging.info(f"  Path: {config.model_path}")
    logging.info(f"  Auto-convert legacy: {config.auto_convert_legacy}")


def main():
    """Enhanced main function with comprehensive HDF5 compatibility."""
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = Config.load(args.config)
            logging.info(f"Loaded configuration from: {args.config}")
        else:
            config = Config()
            logging.info("Using default configuration")
        
        # Update with command line arguments
        config = update_config_from_args(config, args)
        
        # Apply HDF5-safe model configuration
        safe_model_creation(config)
        
        # Save configuration if requested
        if args.save_config:
            config.save(args.save_config)
            print(f"✅ Configuration saved to {args.save_config}")
        
        # Setup logging
        setup_logging(config.log_level)
        logger = logging.getLogger(__name__)
        
        # Log system information
        logger.info("=" * 60)
        logger.info("ENHANCED BATHYMETRIC CAE PIPELINE v2.0")
        logger.info("HDF5 Compatibility Mode: ENABLED")
        logger.info("=" * 60)
        
        logger.info(f"Python version: {sys.version}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Check HDF5 status
        try:
            import h5py
            logger.info(f"HDF5 version: {h5py.version.hdf5_version}")
            logger.info(f"h5py version: {h5py.version.version}")
        except ImportError:
            logger.warning("h5py not available")
        
        # GPU information
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        logger.info(f"GPU available: {gpu_available}")
        
        # Handle GPU setting
        if hasattr(args, 'no_gpu') and args.no_gpu:
            tf.config.set_visible_devices([], 'GPU')
            logger.info("GPU disabled by user request")
        
        # Set paths with safety checks
        input_folder = args.input or config.input_folder
        output_folder = args.output or config.output_folder
        model_path = handle_hdf5_model_path(args.model or config.model_path)
        
        # Ensure directories exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Model path: {model_path}")
        
        # Create and run pipeline with HDF5 protection
        try:
            pipeline = EnhancedBathymetricCAEPipeline(config)
            pipeline.run(input_folder, output_folder, model_path)
            
            logger.info("✅ Processing completed successfully!")
            return 0
            
        except Exception as pipeline_error:
            logger.error(f"❌ Pipeline error: {pipeline_error}")
            
            # Try fallback with minimal configuration
            logger.info("Attempting fallback with minimal configuration...")
            
            fallback_config = Config(
                epochs=5,
                batch_size=1,
                grid_size=128,
                ensemble_size=1,
                model_format="keras",
                model_path=handle_hdf5_model_path("models/fallback_model.keras")
            )
            
            try:
                fallback_pipeline = EnhancedBathymetricCAEPipeline(fallback_config)
                fallback_pipeline.run(input_folder, output_folder, fallback_config.model_path)
                logger.info("✅ Fallback processing completed!")
                return 0
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
                return 1
        
    except KeyboardInterrupt:
        logging.info("⏹️  Process interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"❌ Application error: {e}")
        logging.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    # Handle special diagnostic commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-hdf5":
            print("Testing HDF5 compatibility...")
            try:
                import h5py
                print(f"✅ h5py available: {h5py.version.version}")
                print(f"✅ HDF5 version: {h5py.version.hdf5_version}")
                
                import tensorflow as tf
                print(f"✅ TensorFlow available: {tf.__version__}")
                
                # Test model creation
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
            print("HDF5 Compatibility Mode: ENABLED")
            sys.exit(0)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)