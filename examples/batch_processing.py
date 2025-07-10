"""
Batch Processing Example for Bathymetric CAE Pipeline

This example demonstrates advanced batch processing capabilities including:
- Large-scale file processing
- Progress monitoring and logging
- Error recovery and resumption
- Performance optimization
- Quality analysis and reporting

Author: Bathymetric CAE Team
License: MIT
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the package to the path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent))

from bathymetric_cae import (
    BathymetricCAEPipeline,
    BathymetricProcessor,
    Config,
    setup_logging,
    get_logger,
    memory_monitor,
    get_memory_info,
    FileManager
)


def setup_batch_processing_environment():
    """
    Setup environment for batch processing demonstration.
    """
    print("=== Setting Up Batch Processing Environment ===")
    
    # Create directory structure
    base_dir = Path("./batch_processing_demo")
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    reports_dir = base_dir / "reports"
    logs_dir = base_dir / "logs"
    
    for directory in [input_dir, output_dir, reports_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create sample file structure info
    info_file = base_dir / "README.md"
    info_content = """
# Batch Processing Demo Environment

This directory contains the structure for demonstrating batch processing capabilities.

## Directory Structure:
- `input/`: Place your bathymetric files here (.bag, .tif, .asc, .xyz)
- `output/`: Processed files will be saved here
- `reports/`: Processing reports and statistics
- `logs/`: Detailed processing logs

## Usage:
1. Add bathymetric files to the input directory
2. Run the batch processing example
3. Check output directory for results
4. Review reports for quality metrics and statistics

## Supported File Formats:
- BAG (Bathymetric Attributed Grid): .bag
- GeoTIFF: .tif, .tiff
- ASCII Grid: .asc
- XYZ Point Cloud: .xyz
"""
    
    info_file.write_text(info_content)
    print(f"Created info file: {info_file}")
    
    return {
        "base_dir": base_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "reports_dir": reports_dir,
        "logs_dir": logs_dir
    }


def create_batch_config(dirs: Dict[str, Path]) -> Config:
    """
    Create optimized configuration for batch processing.
    """
    print("\n=== Creating Batch Processing Configuration ===")
    
    # Get system information for optimization
    memory_info = get_memory_info()
    available_memory_gb = memory_info.get('available_mb', 4000) / 1024
    
    print(f"Available memory: {available_memory_gb:.1f} GB")
    
    # Optimize batch size based on available memory
    if available_memory_gb > 16:
        batch_size = 16
        grid_size = 512
        base_filters = 32
    elif available_memory_gb > 8:
        batch_size = 8
        grid_size = 512
        base_filters = 32
    else:
        batch_size = 4
        grid_size = 256
        base_filters = 16
    
    config = Config(
        # I/O Paths
        input_folder=str(dirs["input_dir"]),
        output_folder=str(dirs["output_dir"]),
        model_path=str(dirs["base_dir"] / "batch_model.h5"),
        log_dir=str(dirs["logs_dir"] / "tensorboard"),
        
        # Training Parameters
        epochs=100,
        batch_size=batch_size,
        validation_split=0.2,
        learning_rate=0.001,
        
        # Model Architecture
        grid_size=grid_size,
        base_filters=base_filters,
        depth=4,
        dropout_rate=0.2,
        
        # Callback Parameters
        early_stopping_patience=15,
        reduce_lr_patience=10,
        reduce_lr_factor=0.5,
        min_lr=1e-7,
        
        # Performance Settings
        gpu_memory_growth=True,
        use_mixed_precision=True,
        max_workers=4,
        
        # Logging
        log_level="INFO"
    )
    
    print(f"Batch config created:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Base filters: {config.base_filters}")
    print(f"  Max workers: {config.max_workers}")
    
    return config


def simulate_batch_files(input_dir: Path, num_files: int = 5):
    """
    Create dummy files to simulate a batch processing scenario.
    """
    print(f"\n=== Simulating {num_files} Files for Demo ===")
    
    # Create dummy file info (in real scenario, these would be actual bathymetric files)
    file_info = []
    
    for i in range(num_files):
        filename = f"bathymetry_sample_{i+1:03d}.bag"
        file_path = input_dir / filename
        
        # Create a small dummy file (just for demonstration)
        # In real usage, these would be actual bathymetric data files
        dummy_content = f"# Dummy bathymetric file {i+1}\n# This would contain actual bathymetric data\n"
        file_path.write_text(dummy_content)
        
        file_info.append({
            "filename": filename,
            "path": str(file_path),
            "size_mb": file_path.stat().st_size / (1024 * 1024)
        })
        
        print(f"Created dummy file: {filename}")
    
    return file_info


def analyze_input_files(input_dir: Path) -> Dict[str, Any]:
    """
    Analyze input files before processing.
    """
    print("\n=== Analyzing Input Files ===")
    
    from bathymetric_cae.utils.file_utils import get_valid_files
    from bathymetric_cae import SUPPORTED_FORMATS
    
    # Get valid files
    try:
        valid_files = get_valid_files(input_dir, SUPPORTED_FORMATS)
    except Exception as e:
        print(f"Error finding files: {e}")
        valid_files = []
    
    if not valid_files:
        print("No valid bathymetric files found in input directory")
        print("Note: This demo creates dummy files for demonstration")
        return {"total_files": 0, "analysis": "No valid files"}
    
    # Analyze files
    total_size = 0
    file_types = {}
    file_details = []
    
    for file_path in valid_files:
        file_info = {
            "name": file_path.name,
            "size_mb": file_path.stat().st_size / (1024 * 1024),
            "extension": file_path.suffix.lower()
        }
        
        total_size += file_info["size_mb"]
        ext = file_info["extension"]
        file_types[ext] = file_types.get(ext, 0) + 1
        file_details.append(file_info)
    
    analysis = {
        "total_files": len(valid_files),
        "total_size_mb": total_size,
        "file_types": file_types,
        "file_details": file_details,
        "average_size_mb": total_size / len(valid_files) if valid_files else 0
    }
    
    print(f"Analysis Results:")
    print(f"  Total files: {analysis['total_files']}")
    print(f"  Total size: {analysis['total_size_mb']:.2f} MB")
    print(f"  Average size: {analysis['average_size_mb']:.2f} MB")
    print(f"  File types: {analysis['file_types']}")
    
    return analysis


def run_batch_processing(config: Config, dirs: Dict[str, Path]) -> Dict[str, Any]:
    """
    Run the main batch processing pipeline with monitoring.
    """
    print("\n=== Running Batch Processing Pipeline ===")
    
    logger = get_logger(__name__)
    
    # Initialize pipeline
    pipeline = BathymetricCAEPipeline(config)
    
    # Monitor processing
    start_time = time.time()
    
    with memory_monitor("Batch Processing", logger):
        try:
            # Run pipeline
            results = pipeline.run(
                input_folder=config.input_folder,
                output_folder=config.output_folder,
                model_path=config.model_path
            )
            
            processing_time = time.time() - start_time
            results["processing_time_seconds"] = processing_time
            results["processing_time_minutes"] = processing_time / 60
            
            print(f"\nBatch processing completed in {processing_time/60:.1f} minutes")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                "error": str(e),
                "processing_time_seconds": time.time() - start_time,
                "success": False
            }


def generate_batch_report(results: Dict[str, Any], dirs: Dict[str, Path], input_analysis: Dict[str, Any]):
    """
    Generate comprehensive batch processing report.
    """
    print("\n=== Generating Batch Processing Report ===")
    
    report_file = dirs["reports_dir"] / "batch_processing_report.json"
    html_report_file = dirs["reports_dir"] / "batch_processing_report.html"
    
    # Create comprehensive report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_analysis": input_analysis,
        "processing_results": results,
        "system_info": get_memory_info()
    }
    
    # Save JSON report
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"JSON report saved: {report_file}")
    
    # Generate HTML report
    html_content = generate_html_report(report)
    html_report_file.write_text(html_content)
    
    print(f"HTML report saved: {html_report_file}")
    
    # Print summary to console
    print_batch_summary(results, input_analysis)


def generate_html_report(report: Dict[str, Any]) -> str:
    """
    Generate HTML report for batch processing.
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Bathymetric CAE Batch Processing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
        .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
        .error {{ background-color: #f8d7da; border-color: #f5c6cb; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 1.2em; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Bathymetric CAE Batch Processing Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Processing Summary</h2>
        <table>
            <tr><td>Total Files Processed:</td><td class="metric">{total_files}</td></tr>
            <tr><td>Successful:</td><td class="metric" style="color: green;">{successful_files}</td></tr>
            <tr><td>Failed:</td><td class="metric" style="color: red;">{failed_files}</td></tr>
            <tr><td>Success Rate:</td><td class="metric">{success_rate:.1f}%</td></tr>
            <tr><td>Processing Time:</td><td class="metric">{processing_time:.1f} minutes</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Quality Metrics</h2>
        <table>
            <tr><td>Mean SSIM:</td><td class="metric">{mean_ssim:.4f}</td></tr>
            <tr><td>SSIM Standard Deviation:</td><td class="metric">{std_ssim:.4f}</td></tr>
            <tr><td>SSIM Range:</td><td class="metric">{min_ssim:.4f} - {max_ssim:.4f}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Input Analysis</h2>
        <table>
            <tr><td>Total Input Files:</td><td>{input_total_files}</td></tr>
            <tr><td>Total Size:</td><td>{total_size:.2f} MB</td></tr>
            <tr><td>Average File Size:</td><td>{avg_size:.2f} MB</td></tr>
            <tr><td>File Types:</td><td>{file_types}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>System Information</h2>
        <table>
            <tr><td>Available Memory:</td><td>{available_memory:.1f} MB</td></tr>
            <tr><td>Memory Usage:</td><td>{memory_percent:.1f}%</td></tr>
        </table>
    </div>
</body>
</html>
"""
    
    # Extract data for template
    results = report.get("processing_results", {})
    input_analysis = report.get("input_analysis", {})
    system_info = report.get("system_info", {})
    
    pipeline_info = results.get("pipeline_info", {})
    summary_stats = results.get("summary_statistics", {})
    
    return html_template.format(
        timestamp=report["timestamp"],
        total_files=pipeline_info.get("total_files", 0),
        successful_files=pipeline_info.get("successful_files", 0),
        failed_files=pipeline_info.get("failed_files", 0),
        success_rate=pipeline_info.get("success_rate", 0),
        processing_time=results.get("processing_time_minutes", 0),
        mean_ssim=summary_stats.get("mean_ssim", 0),
        std_ssim=summary_stats.get("std_ssim", 0),
        min_ssim=summary_stats.get("min_ssim", 0),
        max_ssim=summary_stats.get("max_ssim", 0),
        input_total_files=input_analysis.get("total_files", 0),
        total_size=input_analysis.get("total_size_mb", 0),
        avg_size=input_analysis.get("average_size_mb", 0),
        file_types=str(input_analysis.get("file_types", {})),
        available_memory=system_info.get("available_mb", 0),
        memory_percent=system_info.get("percent", 0)
    )


def print_batch_summary(results: Dict[str, Any], input_analysis: Dict[str, Any]):
    """
    Print batch processing summary to console.
    """
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    
    if "error" in results:
        print(f"❌ Processing failed: {results['error']}")
        print(f"Processing time: {results.get('processing_time_seconds', 0):.1f} seconds")
        return
    
    pipeline_info = results.get("pipeline_info", {})
    summary_stats = results.get("summary_statistics", {})
    
    print(f"📊 INPUT ANALYSIS:")
    print(f"   Total files found: {input_analysis.get('total_files', 0)}")
    print(f"   Total size: {input_analysis.get('total_size_mb', 0):.2f} MB")
    print(f"   File types: {input_analysis.get('file_types', {})}")
    
    print(f"\n🔄 PROCESSING RESULTS:")
    print(f"   Total processed: {pipeline_info.get('total_files', 0)}")
    print(f"   Successful: {pipeline_info.get('successful_files', 0)}")
    print(f"   Failed: {pipeline_info.get('failed_files', 0)}")
    print(f"   Success rate: {pipeline_info.get('success_rate', 0):.1f}%")
    print(f"   Processing time: {results.get('processing_time_minutes', 0):.1f} minutes")
    
    if summary_stats:
        print(f"\n📈 QUALITY METRICS:")
        print(f"   Mean SSIM: {summary_stats.get('mean_ssim', 0):.4f}")
        print(f"   SSIM Std: {summary_stats.get('std_ssim', 0):.4f}")
        print(f"   SSIM Range: {summary_stats.get('min_ssim', 0):.4f} - {summary_stats.get('max_ssim', 0):.4f}")
    
    print("=" * 60)


def monitor_batch_progress():
    """
    Example of monitoring batch processing progress.
    """
    print("\n=== Batch Progress Monitoring Example ===")
    
    # This would be implemented in a real scenario with callbacks or separate monitoring
    print("Progress monitoring features:")
    print("1. Real-time file processing status")
    print("2. Memory usage tracking")
    print("3. GPU utilization monitoring") 
    print("4. ETA calculation based on processing speed")
    print("5. Error rate monitoring and alerting")
    print("6. Automatic recovery from transient failures")
    
    # Simulate progress updates
    total_files = 10
    for i in range(total_files + 1):
        progress = (i / total_files) * 100
        bar_length = 30
        filled_length = int(bar_length * i // total_files)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        
        print(f"\rProgress: |{bar}| {progress:.1f}% ({i}/{total_files})", end="", flush=True)
        time.sleep(0.1)  # Simulate processing time
    
    print("\n✅ Progress monitoring complete")


def batch_error_recovery_example():
    """
    Example of error recovery in batch processing.
    """
    print("\n=== Batch Error Recovery Example ===")
    
    print("Error recovery strategies:")
    print("1. Skip corrupted files and continue processing")
    print("2. Retry failed files with different parameters")
    print("3. Save partial results to prevent data loss")
    print("4. Generate detailed error reports for manual review")
    print("5. Resume processing from last successful file")
    print("6. Automatic notification of critical errors")
    
    # Example error scenarios
    error_scenarios = [
        "Corrupted file format",
        "Insufficient memory for large file",
        "GPU out of memory error", 
        "Network connectivity issues",
        "Disk space exhaustion",
        "Invalid geospatial metadata"
    ]
    
    print("\nCommon error scenarios and handling:")
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"{i}. {scenario}")
        print(f"   → Automatic retry with fallback parameters")
        print(f"   → Log detailed error information")
        print(f"   → Continue with remaining files")


def optimize_batch_performance():
    """
    Example of batch processing performance optimization.
    """
    print("\n=== Batch Performance Optimization ===")
    
    memory_info = get_memory_info()
    
    print("Performance optimization strategies:")
    print(f"1. Memory management (Available: {memory_info.get('available_mb', 0):.0f} MB)")
    print("2. Adaptive batch sizing based on file sizes")
    print("3. Parallel processing with optimal worker count")
    print("4. GPU memory optimization and cleanup")
    print("5. Disk I/O optimization for large files")
    print("6. Caching strategies for repeated operations")
    
    # Example optimization recommendations
    available_gb = memory_info.get('available_mb', 4000) / 1024
    
    if available_gb > 16:
        recommendations = [
            "Use large batch sizes (16-32)",
            "Enable high-resolution processing (1024x1024)",
            "Use maximum parallel workers",
            "Enable mixed precision training"
        ]
    elif available_gb > 8:
        recommendations = [
            "Use medium batch sizes (8-16)",
            "Use standard resolution (512x512)",
            "Use moderate parallel workers",
            "Enable mixed precision training"
        ]
    else:
        recommendations = [
            "Use small batch sizes (2-4)",
            "Use reduced resolution (256x256)",
            "Limit parallel workers",
            "Consider CPU-only processing"
        ]
    
    print(f"\nRecommendations for {available_gb:.1f}GB system:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")


def main():
    """
    Main function to demonstrate batch processing capabilities.
    """
    print("Bathymetric CAE - Batch Processing Examples")
    print("=" * 60)
    
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Setup environment
    dirs = setup_batch_processing_environment()
    
    # Create batch configuration
    config = create_batch_config(dirs)
    
    # Save configuration for reference
    config_file = dirs["base_dir"] / "batch_config.json"
    config.save(str(config_file))
    print(f"Configuration saved: {config_file}")
    
    # Simulate input files for demonstration
    file_info = simulate_batch_files(dirs["input_dir"], num_files=3)
    
    # Analyze input files
    input_analysis = analyze_input_files(dirs["input_dir"])
    
    # Demonstrate monitoring and optimization concepts
    monitor_batch_progress()
    batch_error_recovery_example()
    optimize_batch_performance()
    
    # Note about actual processing
    print(f"\n{'='*60}")
    print("NOTE: Actual batch processing disabled in this demo")
    print("To run real batch processing:")
    print("1. Add actual bathymetric files to:", dirs["input_dir"])
    print("2. Uncomment the processing section below")
    print("3. Run the script with real data")
    print(f"{'='*60}")
    
    # Uncomment these lines to run actual processing:
    # results = run_batch_processing(config, dirs)
    # generate_batch_report(results, dirs, input_analysis)
    
    # Create a sample report instead
    sample_results = {
        "pipeline_info": {
            "total_files": 3,
            "successful_files": 3,
            "failed_files": 0,
            "success_rate": 100.0
        },
        "summary_statistics": {
            "mean_ssim": 0.8542,
            "std_ssim": 0.0234,
            "min_ssim": 0.8234,
            "max_ssim": 0.8876
        },
        "processing_time_minutes": 15.5
    }
    
    generate_batch_report(sample_results, dirs, input_analysis)
    
    print("\nBatch processing example completed!")
    print(f"Check the reports directory: {dirs['reports_dir']}")


if __name__ == "__main__":
    main()
