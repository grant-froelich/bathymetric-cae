"""
Single File Processing Example for Bathymetric CAE Pipeline

This example demonstrates interactive single file processing including:
- File validation and analysis
- Interactive processing with visualization
- Quality assessment and metrics
- Custom output formats and locations
- Error handling and debugging

Author: Bathymetric CAE Team
License: MIT
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add the package to the path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent))

from bathymetric_cae import (
    BathymetricCAEPipeline,
    BathymetricProcessor,
    AdvancedCAE,
    Visualizer,
    Config,
    setup_logging,
    get_logger,
    memory_monitor
)


def setup_single_file_environment():
    """
    Setup environment for single file processing demonstration.
    """
    print("=== Setting Up Single File Processing Environment ===")
    
    # Create directory structure
    demo_dir = Path("./single_file_demo")
    input_dir = demo_dir / "input"
    output_dir = demo_dir / "output"
    models_dir = demo_dir / "models"
    plots_dir = demo_dir / "plots"
    
    for directory in [input_dir, output_dir, models_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return {
        "demo_dir": demo_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "models_dir": models_dir,
        "plots_dir": plots_dir
    }


def create_demo_file(input_dir: Path) -> Path:
    """
    Create a demo bathymetric file for testing.
    """
    print("\n=== Creating Demo File ===")
    
    demo_file = input_dir / "demo_bathymetry.tif"
    
    # Create a simple demo file (placeholder)
    # In real usage, this would be an actual bathymetric data file
    demo_content = """# Demo Bathymetric File
# This is a placeholder for demonstration purposes
# In real usage, this would be a proper bathymetric data file
# Supported formats: .bag, .tif, .tiff, .asc, .xyz
"""
    
    demo_file.write_text(demo_content)
    print(f"Created demo file: {demo_file}")
    print("Note: This is a placeholder file for demonstration")
    print("For real processing, use actual bathymetric data files")
    
    return demo_file


def analyze_single_file(file_path: Path, config: Config) -> Dict[str, Any]:
    """
    Analyze a single bathymetric file before processing.
    """
    print(f"\n=== Analyzing File: {file_path.name} ===")
    
    # Basic file information
    file_info = {
        "path": str(file_path),
        "name": file_path.name,
        "size_mb": file_path.stat().st_size / (1024 * 1024),
        "extension": file_path.suffix.lower()
    }
    
    print(f"File information:")
    print(f"  Name: {file_info['name']}")
    print(f"  Size: {file_info['size_mb']:.2f} MB")
    print(f"  Extension: {file_info['extension']}")
    
    # Try to analyze with processor
    try:
        processor = BathymetricProcessor(config)
        
        # Check if file format is supported
        if not processor.validate_file_format(file_path):
            print(f"⚠️  Warning: File format {file_info['extension']} may not be fully supported")
            file_info["format_supported"] = False
            return file_info
        
        file_info["format_supported"] = True
        
        # Get detailed file information
        try:
            detailed_info = processor.get_file_info(file_path)
            file_info.update(detailed_info)
            
            if "error" not in detailed_info:
                print(f"  Width: {detailed_info.get('width', 'Unknown')}")
                print(f"  Height: {detailed_info.get('height', 'Unknown')}")
                print(f"  Bands: {detailed_info.get('bands', 'Unknown')}")
                print(f"  Data type: {detailed_info.get('data_type', 'Unknown')}")
        
        except Exception as e:
            print(f"  Could not get detailed info: {e}")
            file_info["analysis_error"] = str(e)
    
    except Exception as e:
        print(f"  Analysis failed: {e}")
        file_info["analysis_error"] = str(e)
    
    return file_info


def interactive_processing_demo(file_path: Path, dirs: Dict[str, Path]):
    """
    Demonstrate interactive processing with visualization.
    """
    print(f"\n=== Interactive Processing Demo ===")
    
    # Create configuration
    config = Config(
        grid_size=256,  # Smaller for demo
        batch_size=4,
        epochs=5,  # Very few epochs for demo
        base_filters=16,
        depth=2,
        log_level="INFO"
    )
    
    model_path = dirs["models_dir"] / "demo_model.h5"
    
    # Check if we have a pre-trained model
    if not model_path.exists():
        print("No pre-trained model found. For interactive processing:")
        print("1. First train a model using batch processing")
        print("2. Or use a pre-trained model")
        print("\nCreating a minimal demo model instead...")
        
        # Create a minimal model for demonstration
        try:
            model_builder = AdvancedCAE(config)
            model = model_builder.create_model(channels=1)  # Single channel for demo
            model.save(str(model_path))
            print(f"Demo model created: {model_path}")
        except Exception as e:
            print(f"Could not create demo model: {e}")
            return None
    
    # Process the file
    try:
        pipeline = BathymetricCAEPipeline(config)
        
        output_path = dirs["output_dir"] / f"processed_{file_path.stem}.tif"
        
        print(f"Processing file: {file_path.name}")
        print(f"Output will be saved to: {output_path}")
        
        # Note: In real scenario with actual bathymetric data, this would work
        print("\nNote: This demo uses placeholder files")
        print("For real processing, replace with actual bathymetric data")
        
        # Simulate processing results
        simulated_results = {
            "filename": file_path.name,
            "ssim": 0.8756,
            "original_shape": (256, 256),
            "has_uncertainty": False,
            "processing_successful": True,
            "output_path": str(output_path)
        }
        
        return simulated_results
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return {
            "filename": file_path.name,
            "error": str(e),
            "processing_successful": False
        }


def quality_assessment_demo(results: Dict[str, Any]):
    """
    Demonstrate quality assessment of processed results.
    """
    print(f"\n=== Quality Assessment Demo ===")
    
    if not results.get("processing_successful", False):
        print("Cannot assess quality - processing failed")
        return
    
    # Simulated quality metrics
    quality_metrics = {
        "ssim": results.get("ssim", 0.8756),
        "mse": 0.0234,
        "psnr": 34.56,
        "mae": 0.0123,
        "correlation": 0.9234
    }
    
    print("Quality Metrics:")
    print(f"  SSIM (Structural Similarity): {quality_metrics['ssim']:.4f}")
    print(f"  MSE (Mean Squared Error): {quality_metrics['mse']:.4f}")
    print(f"  PSNR (Peak Signal-to-Noise Ratio): {quality_metrics['psnr']:.2f} dB")
    print(f"  MAE (Mean Absolute Error): {quality_metrics['mae']:.4f}")
    print(f"  Correlation: {quality_metrics['correlation']:.4f}")
    
    # Quality assessment
    ssim = quality_metrics['ssim']
    if ssim > 0.9:
        quality_rating = "Excellent"
        quality_color = "🟢"
    elif ssim > 0.8:
        quality_rating = "Good"
        quality_color = "🟡"
    elif ssim > 0.7:
        quality_rating = "Fair"
        quality_color = "🟠"
    else:
        quality_rating = "Poor"
        quality_color = "🔴"
    
    print(f"\nOverall Quality: {quality_color} {quality_rating} (SSIM: {ssim:.4f})")
    
    return quality_metrics


def visualization_demo(file_path: Path, results: Dict[str, Any], dirs: Dict[str, Path]):
    """
    Demonstrate visualization capabilities.
    """
    print(f"\n=== Visualization Demo ===")
    
    if not results.get("processing_successful", False):
        print("Cannot create visualizations - processing failed")
        return
    
    # Create visualizer
    visualizer = Visualizer()
    
    # Simulate data for visualization
    import numpy as np
    
    # Create sample data (in real scenario, this would be actual bathymetric data)
    np.random.seed(42)  # For reproducible demo
    size = 128
    
    # Simulate original bathymetric data
    original_data = np.random.rand(size, size) * 100 - 50  # Depth values
    
    # Simulate cleaned data (slightly smoothed)
    from scipy import ndimage
    cleaned_data = ndimage.gaussian_filter(original_data, sigma=0.5)
    
    # Simulate uncertainty data
    uncertainty_data = np.random.rand(size, size) * 2  # Uncertainty values
    
    print("Creating visualizations...")
    
    # Comparison plot
    comparison_file = dirs["plots_dir"] / f"comparison_{file_path.stem}.png"
    try:
        visualizer.plot_comparison(
            original=original_data,
            cleaned=cleaned_data,
            uncertainty=uncertainty_data,
            filename=str(comparison_file),
            show_plot=False,  # Set to True for interactive display
            title=f"Processing Results: {file_path.name}"
        )
        print(f"✓ Comparison plot saved: {comparison_file}")
    except Exception as e:
        print(f"✗ Could not create comparison plot: {e}")
    
    # Difference map
    difference_file = dirs["plots_dir"] / f"difference_{file_path.stem}.png"
    try:
        visualizer.plot_difference_map(
            original=original_data,
            cleaned=cleaned_data,
            filename=str(difference_file),
            show_plot=False
        )
        print(f"✓ Difference map saved: {difference_file}")
    except Exception as e:
        print(f"✗ Could not create difference map: {e}")
    
    # Data distribution analysis
    distribution_file = dirs["plots_dir"] / f"distribution_{file_path.stem}.png"
    try:
        data_dict = {
            "Original": original_data,
            "Cleaned": cleaned_data,
            "Uncertainty": uncertainty_data
        }
        
        visualizer.plot_data_distribution(
            data_dict=data_dict,
            filename=str(distribution_file),
            show_plot=False
        )
        print(f"✓ Distribution analysis saved: {distribution_file}")
    except Exception as e:
        print(f"✗ Could not create distribution analysis: {e}")
    
    print(f"All visualizations saved to: {dirs['plots_dir']}")
    
    return {
        "comparison_plot": str(comparison_file),
        "difference_map": str(difference_file),
        "distribution_plot": str(distribution_file)
    }


def custom_output_formats_demo(results: Dict[str, Any], dirs: Dict[str, Path]):
    """
    Demonstrate custom output formats and metadata preservation.
    """
    print(f"\n=== Custom Output Formats Demo ===")
    
    if not results.get("processing_successful", False):
        print("Cannot demonstrate output formats - processing failed")
        return
    
    # Example of different output formats
    output_formats = {
        "geotiff": {
            "extension": ".tif",
            "description": "GeoTIFF with preserved geospatial metadata",
            "use_case": "Standard geospatial workflows"
        },
        "ascii_grid": {
            "extension": ".asc",
            "description": "ESRI ASCII Grid format",
            "use_case": "Import into GIS software"
        },
        "numpy": {
            "extension": ".npy",
            "description": "NumPy binary format",
            "use_case": "Further processing in Python"
        },
        "csv": {
            "extension": ".csv",
            "description": "Comma-separated values",
            "use_case": "Data analysis and visualization"
        }
    }
    
    print("Available output formats:")
    for format_name, info in output_formats.items():
        print(f"  {format_name.upper()}:")
        print(f"    Extension: {info['extension']}")
        print(f"    Description: {info['description']}")
        print(f"    Use case: {info['use_case']}")
        print()
    
    # Demonstrate metadata preservation
    sample_metadata = {
        "processing_date": "2024-01-15T10:30:00Z",
        "software_version": "bathymetric-cae v1.0.0",
        "model_architecture": "Advanced CAE with attention",
        "input_file": results.get("filename", "unknown"),
        "quality_metrics": {
            "ssim": results.get("ssim", 0.0),
            "processing_time": "45.2 seconds"
        },
        "geospatial_info": {
            "coordinate_system": "EPSG:4326",
            "bounds": [-180, -90, 180, 90],
            "resolution": "30m"
        }
    }
    
    print("Metadata preservation example:")
    import json
    print(json.dumps(sample_metadata, indent=2))


def error_handling_demo():
    """
    Demonstrate error handling and debugging capabilities.
    """
    print(f"\n=== Error Handling and Debugging Demo ===")
    
    print("Common error scenarios and handling strategies:")
    
    error_scenarios = [
        {
            "error": "File not found",
            "handling": "Validate file path and permissions",
            "prevention": "Check file existence before processing"
        },
        {
            "error": "Unsupported file format",
            "handling": "Convert to supported format or skip file",
            "prevention": "Validate file extensions beforehand"
        },
        {
            "error": "Corrupted file data",
            "handling": "Apply data cleaning and fallback methods",
            "prevention": "Implement robust data validation"
        },
        {
            "error": "Out of memory",
            "handling": "Reduce batch size or grid resolution",
            "prevention": "Monitor memory usage and optimize"
        },
        {
            "error": "GPU memory error",
            "handling": "Fallback to CPU processing",
            "prevention": "Configure GPU memory growth"
        },
        {
            "error": "Model loading failure",
            "handling": "Use fallback model or retrain",
            "prevention": "Validate model compatibility"
        }
    ]
    
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"{i}. {scenario['error']}:")
        print(f"   Handling: {scenario['handling']}")
        print(f"   Prevention: {scenario['prevention']}")
        print()
    
    # Demonstrate error logging
    print("Error logging example:")
    logger = get_logger(__name__)
    
    try:
        # Simulate an error
        raise ValueError("Demo error for logging demonstration")
    except Exception as e:
        logger.error(f"Caught demo error: {e}")
        logger.debug("This would include full traceback in debug mode")
        print("✓ Error logged successfully")


def performance_monitoring_demo():
    """
    Demonstrate performance monitoring for single file processing.
    """
    print(f"\n=== Performance Monitoring Demo ===")
    
    # Memory monitoring example
    print("Memory monitoring during processing:")
    
    with memory_monitor("Single file processing simulation"):
        # Simulate some processing work
        import numpy as np
        
        print("  Allocating memory for data simulation...")
        data = np.random.rand(1000, 1000)
        
        print("  Simulating data processing...")
        time.sleep(0.5)  # Simulate processing time
        
        result = np.mean(data)
        print(f"  Processing complete. Result: {result:.4f}")
        
        # Clean up
        del data
    
    # Performance metrics
    performance_metrics = {
        "file_loading_time": 2.3,
        "preprocessing_time": 5.7,
        "inference_time": 8.2,
        "postprocessing_time": 1.8,
        "total_time": 18.0,
        "memory_peak_mb": 1234.5,
        "gpu_utilization": 87.3
    }
    
    print("\nPerformance metrics example:")
    for metric, value in performance_metrics.items():
        if "time" in metric:
            print(f"  {metric.replace('_', ' ').title()}: {value:.1f} seconds")
        elif "memory" in metric:
            print(f"  {metric.replace('_', ' ').title()}: {value:.1f} MB")
        else:
            print(f"  {metric.replace('_', ' ').title()}: {value:.1f}%")


def batch_vs_single_comparison():
    """
    Compare batch processing vs single file processing.
    """
    print(f"\n=== Batch vs Single File Processing Comparison ===")
    
    comparison_table = [
        ["Aspect", "Single File", "Batch Processing"],
        ["Use Case", "Interactive analysis", "Production workflows"],
        ["Speed", "Immediate results", "Optimized throughput"],
        ["Memory Usage", "Lower", "Higher (batch optimization)"],
        ["Visualization", "Real-time plots", "Summary reports"],
        ["Error Handling", "Immediate feedback", "Robust recovery"],
        ["Monitoring", "Detailed progress", "Aggregate statistics"],
        ["Resource Usage", "On-demand", "Sustained utilization"],
        ["Flexibility", "High customization", "Standardized workflow"]
    ]
    
    # Print comparison table
    col_widths = [max(len(row[i]) for row in comparison_table) + 2 for i in range(3)]
    
    for i, row in enumerate(comparison_table):
        formatted_row = "|".join(f" {cell:<{col_widths[j]-1}}" for j, cell in enumerate(row))
        print(f"|{formatted_row}|")
        
        if i == 0:  # Header separator
            separator = "|".join("-" * col_widths[j] for j in range(3))
            print(f"|{separator}|")


def create_processing_report(file_info: Dict[str, Any], results: Dict[str, Any], 
                           quality_metrics: Dict[str, Any], dirs: Dict[str, Path]):
    """
    Create a comprehensive processing report for the single file.
    """
    print(f"\n=== Creating Processing Report ===")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_info": file_info,
        "processing_results": results,
        "quality_metrics": quality_metrics,
        "processing_successful": results.get("processing_successful", False)
    }
    
    # Save JSON report
    report_file = dirs["output_dir"] / f"report_{Path(file_info['name']).stem}.json"
    
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Processing report saved: {report_file}")
    
    # Create summary text
    summary_file = dirs["output_dir"] / f"summary_{Path(file_info['name']).stem}.txt"
    
    summary_content = f"""
BATHYMETRIC CAE PROCESSING SUMMARY
=================================

File: {file_info['name']}
Processed: {report['timestamp']}

FILE INFORMATION:
- Size: {file_info.get('size_mb', 0):.2f} MB
- Format: {file_info.get('extension', 'unknown')}
- Supported: {file_info.get('format_supported', False)}

PROCESSING RESULTS:
- Success: {results.get('processing_successful', False)}
- Output: {results.get('output_path', 'N/A')}

QUALITY METRICS:
- SSIM: {quality_metrics.get('ssim', 0):.4f}
- MSE: {quality_metrics.get('mse', 0):.4f}
- Correlation: {quality_metrics.get('correlation', 0):.4f}

Generated by Bathymetric CAE v1.0.0
"""
    
    summary_file.write_text(summary_content)
    print(f"Processing summary saved: {summary_file}")
    
    return report


def main():
    """
    Main function to demonstrate single file processing capabilities.
    """
    print("Bathymetric CAE - Single File Processing Examples")
    print("=" * 60)
    
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Setup environment
    dirs = setup_single_file_environment()
    
    # Create demo file
    demo_file = create_demo_file(dirs["input_dir"])
    
    # Create configuration
    config = Config(
        input_folder=str(dirs["input_dir"]),
        output_folder=str(dirs["output_dir"]),
        grid_size=256,
        batch_size=4,
        log_level="INFO"
    )
    
    # Analyze the file
    file_info = analyze_single_file(demo_file, config)
    
    # Run interactive processing demo
    results = interactive_processing_demo(demo_file, dirs)
    
    if results:
        # Quality assessment
        quality_metrics = quality_assessment_demo(results)
        
        # Visualization demo
        plot_info = visualization_demo(demo_file, results, dirs)
        
        # Custom output formats
        custom_output_formats_demo(results, dirs)
        
        # Create comprehensive report
        report = create_processing_report(file_info, results, quality_metrics, dirs)
    
    # Demonstrate other concepts
    error_handling_demo()
    performance_monitoring_demo()
    batch_vs_single_comparison()
    
    print("\n" + "=" * 60)
    print("Single File Processing Examples Completed!")
    print("\nKey Features Demonstrated:")
    print("1. File analysis and validation")
    print("2. Interactive processing with immediate feedback")
    print("3. Comprehensive quality assessment")
    print("4. Rich visualization capabilities")
    print("5. Custom output formats and metadata preservation")
    print("6. Robust error handling and debugging")
    print("7. Performance monitoring and optimization")
    print("8. Detailed reporting and documentation")
    
    print(f"\nDemo files created in: {dirs['demo_dir']}")
    print("For real processing:")
    print("1. Replace demo files with actual bathymetric data")
    print("2. Train or load a proper model")
    print("3. Adjust configuration for your specific needs")


if __name__ == "__main__":
    main()
