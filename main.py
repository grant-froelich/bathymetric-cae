#!/usr/bin/env python3
"""
Enhanced Bathymetric CAE Processing - Main Entry Point
Updated with modern Keras format support and warning suppression.
"""

import os
import sys
import warnings
import logging
from pathlib import Path

# ============================================================================
# WARNING SUPPRESSION (Apply before any TensorFlow imports)
# ============================================================================

# Environment variables for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Filter INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization warnings
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings

# Python warnings suppression
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*GDAL.*')
warnings.filterwarnings('ignore', message='.*HDF5.*')
warnings.filterwarnings('ignore', message='.*legacy.*')

# GDAL Exception handling (prevents FutureWarning)
try:
    from osgeo import gdal
    gdal.UseExceptions()  # Explicitly enable exceptions
except ImportError:
    pass

# ============================================================================
# TensorFlow Configuration
# ============================================================================

import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy

# Configure TensorFlow before other imports
tf.get_logger().setLevel('ERROR')  # Suppress TF logging
tf.autograph.set_verbosity(0)      # Suppress autograph verbosity

# Suppress specific TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configure mixed precision for better performance
try:
    policy = Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
except Exception:
    # Fallback if mixed precision not supported
    pass

# Optimize TensorFlow for CPU if no GPU
if len(tf.config.list_physical_devices('GPU')) == 0:
    tf.config.optimizer.set_jit(True)  # Enable XLA for CPU

# ============================================================================
# Main Application Imports
# ============================================================================

from config.config import Config
from cli.interface import create_argument_parser, update_config_from_args
from processing.pipeline import EnhancedBathymetricCAEPipeline
from utils.logging_utils import setup_logging


def setup_cpu_optimization():
    """Setup additional CPU optimizations to reduce warnings."""
    try:
        # Set thread configuration for better CPU performance
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
        
        # Enable JIT compilation
        tf.config.optimizer.set_jit(True)
        
    except Exception as e:
        logging.debug(f"CPU optimization setup failed: {e}")


def setup_model_format_migration(config: Config):
    """Handle automatic migration from legacy H5 to modern Keras format."""
    if config.model_format == "keras" and config.auto_convert_legacy:
        # Check if legacy H5 models exist and need conversion
        legacy_paths = []
        modern_paths = []
        
        for i in range(config.ensemble_size):
            h5_path = f"{config.model_path.replace('.keras', '')}_ensemble_{i}.h5"
            keras_path = f"{config.model_path.replace('.keras', '')}_ensemble_{i}.keras"
            
            if Path(h5_path).exists() and not Path(keras_path).exists():
                legacy_paths.append(h5_path)
                modern_paths.append(keras_path)
        
        if legacy_paths:
            logging.info(f"Found {len(legacy_paths)} legacy H5 models. Auto-conversion enabled.")
            logging.info("Legacy models will be converted to modern Keras format during loading.")


def validate_environment():
    """Validate the environment and dependencies."""
    issues = []
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        logging.info(f"TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            logging.info(f"GPU available: {len(gpu_devices)} device(s)")
        else:
            logging.info("GPU available: False (using CPU)")
            
    except ImportError:
        issues.append("TensorFlow not available")
    
    # Check GDAL
    try:
        from osgeo import gdal
        logging.info(f"GDAL version: {gdal.__version__}")
    except ImportError:
        issues.append("GDAL not available - some file formats may not work")
    
    # Check essential packages
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError as e:
        issues.append(f"Missing essential package: {e}")
    
    if issues:
        logging.warning("Environment issues detected:")
        for issue in issues:
            logging.warning(f"  - {issue}")
    
    return len(issues) == 0


def main():
    """Enhanced main function with modern Keras format support and comprehensive error handling."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration (with modern format support)
        if args.config:
            config = Config.load(args.config)
            logging.info(f"Loaded configuration from: {args.config}")
        else:
            config = Config()
            logging.info("Using default configuration")
        
        # Update with command line arguments
        config = update_config_from_args(config, args)
        
        # Handle model format settings
        if hasattr(args, 'model_format'):
            config.set_model_format(args.model_format)
        
        # Save configuration if requested
        if args.save_config:
            config.save(args.save_config)
            print(f"✅ Configuration saved to {args.save_config}")
            if config.is_modern_format():
                print(f"   Using modern Keras format (.keras)")
            else:
                print(f"   Using legacy H5 format (.h5)")
        
        # Setup logging with warning suppression
        setup_logging(config.log_level)
        logger = logging.getLogger(__name__)
        
        # Suppress additional logging from common sources
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('absl').setLevel(logging.ERROR)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
        # Log system information
        logger.info(f"Starting Enhanced Bathymetric CAE Pipeline v2.0")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        logger.info(f"Model format: {config.model_format.upper()} ({config.get_model_extension()})")
        
        # Validate environment
        if not validate_environment():
            logger.warning("Environment validation detected issues. Proceeding anyway...")
        
        # Setup CPU optimization
        setup_cpu_optimization()
        
        # Setup model format migration
        setup_model_format_migration(config)
        
        # Log enabled features
        features_enabled = []
        if config.enable_adaptive_processing:
            features_enabled.append("Adaptive Processing")
        if config.enable_expert_review:
            features_enabled.append("Expert Review")
        if config.enable_constitutional_constraints:
            features_enabled.append("Constitutional Constraints")
        if config.is_modern_format():
            features_enabled.append("Modern Keras Format")
        
        logger.info(f"Enhanced features enabled: {', '.join(features_enabled) if features_enabled else 'None'}")
        
        # Handle GPU setting
        if hasattr(args, 'no_gpu') and args.no_gpu:
            tf.config.set_visible_devices([], 'GPU')
            logger.info("GPU disabled by user request")
        
        # Set paths
        input_folder = args.input or config.input_folder
        output_folder = args.output or config.output_folder
        model_path = args.model or config.model_path
        
        # Ensure model path uses correct format
        if config.model_format == "keras" and model_path.endswith('.h5'):
            model_path = model_path.replace('.h5', '.keras')
            logger.info(f"Updated model path to modern format: {model_path}")
        elif config.model_format == "h5" and model_path.endswith('.keras'):
            model_path = model_path.replace('.keras', '.h5')
            logger.info(f"Updated model path to legacy format: {model_path}")
        
        # Setup environment
        _setup_environment(config)
        
        # Create and run enhanced pipeline
        pipeline = EnhancedBathymetricCAEPipeline(config)
        pipeline.run(input_folder, output_folder, model_path)
        
        logger.info("✅ Enhanced processing pipeline completed successfully!")
        
        # Log final summary
        if config.is_modern_format():
            logger.info(f"📁 Models saved in modern Keras format ({config.get_model_extension()})")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("⏹️  Process interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"❌ Enhanced pipeline failed: {e}")
        logging.debug("Full traceback:", exc_info=True)
        return 1


def _setup_environment(config):
    """Setup processing environment with modern format support."""
    # Create all required directories
    directories = [
        config.output_folder,
        "logs",
        "plots", 
        "expert_reviews",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Log model format info
    logging.info(f"Model format: {config.model_format}")
    logging.info(f"Model extension: {config.get_model_extension()}")
    if config.auto_convert_legacy:
        logging.info("Auto-conversion of legacy models: ENABLED")


def create_quick_test_config_file():
    """Create a quick test configuration file with modern format."""
    config = Config.create_quick_test_config()
    config.save("quick_test_config.json")
    print("✅ Quick test configuration created: quick_test_config.json")
    print(f"   Model format: {config.model_format}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Grid size: {config.grid_size}")
    return config


if __name__ == "__main__":
    # Handle special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-quick-config":
            create_quick_test_config_file()
            sys.exit(0)
        elif sys.argv[1] == "--migrate-config" and len(sys.argv) > 2:
            legacy_path = sys.argv[2]
            try:
                config, new_path = Config.migrate_legacy_config(legacy_path)
                print(f"✅ Configuration migrated: {legacy_path} -> {new_path}")
                print(f"   Model format updated to: {config.model_format}")
            except Exception as e:
                print(f"❌ Migration failed: {e}")
                sys.exit(1)
            sys.exit(0)
        elif sys.argv[1] == "--version":
            print("Enhanced Bathymetric CAE Processing v2.0")
            print("Modern Keras Format Support: ✅")
            print("Warning Suppression: ✅")
            sys.exit(0)
    
    # Run main application
    exit(main())