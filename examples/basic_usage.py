"""
Basic Usage Example for Bathymetric CAE Pipeline

This example demonstrates the most straightforward way to use the bathymetric CAE
pipeline for processing bathymetric data with default settings.

Author: Bathymetric CAE Team
License: MIT
"""

import os
import sys
from pathlib import Path

# Add the package to the path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent))

from bathymetric_cae import (
    BathymetricCAEPipeline, 
    Config, 
    setup_logging,
    quick_start
)


def basic_pipeline_example():
    """
    Basic pipeline usage with default configuration.
    """
    print("=== Basic Pipeline Usage Example ===")
    
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Define input and output directories
    input_folder = "./sample_data/input"
    output_folder = "./sample_data/output"
    
    # Create directories if they don't exist
    Path(input_folder).mkdir(parents=True, exist_ok=True)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Method 1: Using the quick_start function (simplest approach)
    print("\n--- Method 1: Quick Start ---")
    try:
        results = quick_start(
            input_folder=input_folder,
            output_folder=output_folder,
            epochs=10,  # Small number for demo
            batch_size=4
        )
        
        print(f"Quick start completed!")
        print(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"Quick start failed: {e}")
        print("This is expected if no input files are present")
    
    # Method 2: Using configuration object (more control)
    print("\n--- Method 2: Configuration Object ---")
    
    # Create configuration
    config = Config(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path="basic_model.h5",
        epochs=20,
        batch_size=8,
        grid_size=256,  # Smaller for faster processing
        log_level="INFO"
    )
    
    print(f"Configuration created:")
    print(f"  Input folder: {config.input_folder}")
    print(f"  Output folder: {config.output_folder}")
    print(f"  Model path: {config.model_path}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Grid size: {config.grid_size}")
    
    # Create and run pipeline
    try:
        pipeline = BathymetricCAEPipeline(config)
        
        results = pipeline.run(
            input_folder=config.input_folder,
            output_folder=config.output_folder,
            model_path=config.model_path
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Total files processed: {results['pipeline_info']['total_files']}")
        print(f"Successful: {results['pipeline_info']['successful_files']}")
        print(f"Failed: {results['pipeline_info']['failed_files']}")
        print(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
        
        if results['summary_statistics']:
            stats = results['summary_statistics']
            print(f"\nQuality Metrics:")
            print(f"  Mean SSIM: {stats['mean_ssim']:.4f}")
            print(f"  SSIM Std: {stats['std_ssim']:.4f}")
            print(f"  SSIM Range: {stats['min_ssim']:.4f} - {stats['max_ssim']:.4f}")
    
    except FileNotFoundError as e:
        print(f"Input folder not found or empty: {e}")
        print("Please add some bathymetric files (.bag, .tif, .asc) to the input folder")
    
    except Exception as e:
        print(f"Pipeline failed: {e}")
        print("Check the logs for more details")


def basic_single_file_example():
    """
    Basic single file processing example.
    """
    print("\n=== Single File Processing Example ===")
    
    # Example file path (you would replace this with an actual file)
    example_file = "./sample_data/example_bathymetry.bag"
    model_path = "basic_model.h5"
    
    # Check if example file exists
    if not Path(example_file).exists():
        print(f"Example file not found: {example_file}")
        print("To use this example:")
        print("1. Place a bathymetric file in the sample_data folder")
        print("2. Update the example_file path above")
        print("3. Train a model first using the basic pipeline")
        return
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        print("Please train a model first using the basic pipeline example above")
        return
    
    try:
        # Create configuration
        config = Config()
        
        # Create pipeline
        pipeline = BathymetricCAEPipeline(config)
        
        # Process single file
        results = pipeline.process_single_file_interactive(
            file_path=example_file,
            model_path=model_path,
            show_plots=True  # Set to False if running headless
        )
        
        if results['processing_successful']:
            print(f"Single file processing completed!")
            print(f"File: {results['filename']}")
            print(f"SSIM score: {results['ssim']:.4f}")
            print(f"Original shape: {results['original_shape']}")
            print(f"Has uncertainty: {results['has_uncertainty']}")
        else:
            print(f"Processing failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Single file processing failed: {e}")


def basic_configuration_save_load():
    """
    Example of saving and loading configurations.
    """
    print("\n=== Configuration Save/Load Example ===")
    
    # Create a custom configuration
    config = Config(
        input_folder="./my_input_data",
        output_folder="./my_output_data",
        epochs=50,
        batch_size=16,
        grid_size=512,
        base_filters=64,
        learning_rate=0.0005,
        dropout_rate=0.1
    )
    
    # Save configuration
    config_file = "my_basic_config.json"
    try:
        config.save(config_file)
        print(f"Configuration saved to: {config_file}")
        
        # Load configuration
        loaded_config = Config.load(config_file)
        print(f"Configuration loaded from: {config_file}")
        
        # Verify configuration
        print(f"Loaded config - Epochs: {loaded_config.epochs}")
        print(f"Loaded config - Batch size: {loaded_config.batch_size}")
        print(f"Loaded config - Grid size: {loaded_config.grid_size}")
        
        # Clean up
        Path(config_file).unlink()
        print(f"Cleaned up config file: {config_file}")
        
    except Exception as e:
        print(f"Configuration save/load failed: {e}")


def setup_sample_environment():
    """
    Setup sample environment for testing.
    """
    print("\n=== Setting Up Sample Environment ===")
    
    # Create sample directory structure
    sample_dir = Path("./sample_data")
    input_dir = sample_dir / "input"
    output_dir = sample_dir / "output"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created sample directories:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Create a sample README file
    readme_content = """
# Sample Data Directory

This directory is set up for testing the Bathymetric CAE pipeline.

## Usage:
1. Place your bathymetric files (.bag, .tif, .asc, .xyz) in the 'input' folder
2. Run the basic usage examples
3. Check the 'output' folder for processed results

## Supported Formats:
- BAG (Bathymetric Attributed Grid) - .bag
- GeoTIFF - .tif, .tiff  
- ASCII Grid - .asc
- XYZ Point Cloud - .xyz

## Note:
This is a sample directory created by the basic usage example.
"""
    
    readme_file = sample_dir / "README.md"
    readme_file.write_text(readme_content)
    print(f"Created README: {readme_file}")
    
    return str(input_dir), str(output_dir)


def main():
    """
    Main function to run all basic examples.
    """
    print("Bathymetric CAE - Basic Usage Examples")
    print("=" * 50)
    
    # Setup sample environment
    input_dir, output_dir = setup_sample_environment()
    
    # Run basic examples
    basic_pipeline_example()
    basic_single_file_example()
    basic_configuration_save_load()
    
    print("\n" + "=" * 50)
    print("Basic examples completed!")
    print("\nNext steps:")
    print("1. Add some bathymetric files to:", input_dir)
    print("2. Run the examples again to see actual processing")
    print("3. Check the other example files for more advanced usage")
    print("4. Explore the custom_config.py example for configuration options")


if __name__ == "__main__":
    main()
