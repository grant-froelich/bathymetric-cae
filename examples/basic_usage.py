# examples/basic_usage.py
"""
Basic usage example for Enhanced Bathymetric CAE Processing.

This example demonstrates the simplest way to process bathymetric data
using the default configuration.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bathymetric_cae import Config, quick_process
from bathymetric_cae.processing import EnhancedBathymetricCAEPipeline
from bathymetric_cae.utils import setup_logging


def basic_processing_example():
    """Example of basic bathymetric data processing."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Example 1: Quick processing with defaults
    print("Example 1: Quick Processing")
    print("-" * 30)
    
    try:
        quick_process(
            input_path="data/input",
            output_path="data/output_basic",
            ensemble_size=3,
            enable_adaptive_processing=True
        )
        print("✓ Quick processing completed successfully!")
    except Exception as e:
        print(f"✗ Quick processing failed: {e}")
    
    # Example 2: Using configuration
    print("\nExample 2: Configuration-based Processing")
    print("-" * 40)
    
    # Create and customize configuration
    config = Config()
    config.ensemble_size = 3
    config.epochs = 50  # Reduced for faster example
    config.batch_size = 4
    config.enable_adaptive_processing = True
    config.enable_expert_review = True
    config.quality_threshold = 0.7
    
    # Save configuration for reuse
    config.save("examples/basic_config.json")
    print("Configuration saved to: examples/basic_config.json")
    
    # Process using configuration
    try:
        pipeline = EnhancedBathymetricCAEPipeline(config)
        pipeline.run(
            input_folder="data/input",
            output_folder="data/output_configured",
            model_path="models/basic_example"
        )
        print("✓ Configuration-based processing completed!")
    except Exception as e:
        print(f"✗ Configuration-based processing failed: {e}")
    
    print("\nBasic usage examples completed!")


if __name__ == "__main__":
    basic_processing_example()

