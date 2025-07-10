"""
Custom Configuration Example for Bathymetric CAE Pipeline

This example demonstrates advanced configuration options, including:
- Custom model architectures
- Advanced training parameters
- Performance optimization settings
- Multiple configuration scenarios

Author: Bathymetric CAE Team
License: MIT
"""

import sys
from pathlib import Path

# Add the package to the path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent))

from bathymetric_cae import (
    Config, 
    BathymetricCAEPipeline,
    setup_logging,
    get_memory_info,
    check_gpu_availability
)


def create_high_performance_config():
    """
    Configuration for high-performance processing with powerful hardware.
    """
    print("=== High Performance Configuration ===")
    
    config = Config(
        # I/O Settings
        input_folder="./data/high_res_bathymetry",
        output_folder="./results/high_performance",
        model_path="high_performance_model.h5",
        
        # Training Parameters - Optimized for quality
        epochs=200,
        batch_size=32,  # Large batch for stable gradients
        validation_split=0.15,
        learning_rate=0.0001,  # Lower LR for fine-tuning
        
        # Model Architecture - Large and deep
        grid_size=1024,  # High resolution
        base_filters=64,  # More filters for complex features
        depth=6,  # Deeper network
        dropout_rate=0.15,  # Moderate regularization
        
        # Callback Parameters
        early_stopping_patience=25,  # More patience for convergence
        reduce_lr_patience=12,
        reduce_lr_factor=0.3,  # Aggressive LR reduction
        min_lr=1e-8,
        
        # Performance Settings
        gpu_memory_growth=True,
        use_mixed_precision=True,
        max_workers=8,  # Parallel processing
        
        # Logging
        log_level="INFO"
    )
    
    print(f"High Performance Config Created:")
    print(f"  Grid Size: {config.grid_size}")
    print(f"  Base Filters: {config.base_filters}")
    print(f"  Depth: {config.depth}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    
    return config


def create_memory_efficient_config():
    """
    Configuration for systems with limited memory.
    """
    print("\n=== Memory Efficient Configuration ===")
    
    config = Config(
        # I/O Settings
        input_folder="./data/bathymetry",
        output_folder="./results/memory_efficient",
        model_path="memory_efficient_model.h5",
        
        # Training Parameters - Memory optimized
        epochs=100,
        batch_size=4,  # Small batch to save memory
        validation_split=0.2,
        learning_rate=0.002,  # Higher LR to compensate for small batch
        
        # Model Architecture - Compact
        grid_size=256,  # Smaller input size
        base_filters=16,  # Fewer filters
        depth=3,  # Shallower network
        dropout_rate=0.3,  # Higher dropout for regularization
        
        # Callback Parameters
        early_stopping_patience=15,
        reduce_lr_patience=8,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
        
        # Performance Settings
        gpu_memory_growth=True,
        use_mixed_precision=True,
        max_workers=2,  # Limited parallel processing
        
        # Logging
        log_level="INFO"
    )
    
    print(f"Memory Efficient Config Created:")
    print(f"  Grid Size: {config.grid_size}")
    print(f"  Base Filters: {config.base_filters}")
    print(f"  Depth: {config.depth}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Max Workers: {config.max_workers}")
    
    return config


def create_rapid_prototyping_config():
    """
    Configuration for fast prototyping and testing.
    """
    print("\n=== Rapid Prototyping Configuration ===")
    
    config = Config(
        # I/O Settings
        input_folder="./data/test_samples",
        output_folder="./results/prototype",
        model_path="prototype_model.h5",
        
        # Training Parameters - Fast training
        epochs=20,  # Very few epochs
        batch_size=8,
        validation_split=0.3,  # More validation data
        learning_rate=0.01,  # High LR for fast convergence
        
        # Model Architecture - Simple
        grid_size=128,  # Very small for speed
        base_filters=8,  # Minimal filters
        depth=2,  # Very shallow
        dropout_rate=0.1,  # Low dropout
        
        # Callback Parameters - Aggressive
        early_stopping_patience=5,  # Stop early
        reduce_lr_patience=3,
        reduce_lr_factor=0.5,
        min_lr=1e-5,
        
        # Performance Settings
        gpu_memory_growth=True,
        use_mixed_precision=False,  # Disable for simplicity
        max_workers=1,  # Single thread for simplicity
        
        # Logging
        log_level="DEBUG"  # Verbose for debugging
    )
    
    print(f"Rapid Prototyping Config Created:")
    print(f"  Grid Size: {config.grid_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Early Stopping Patience: {config.early_stopping_patience}")
    
    return config


def create_production_config():
    """
    Configuration for production deployment.
    """
    print("\n=== Production Configuration ===")
    
    config = Config(
        # I/O Settings
        input_folder="/data/production/input",
        output_folder="/data/production/output",
        model_path="/models/production_model.h5",
        
        # Training Parameters - Balanced
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        learning_rate=0.0005,
        
        # Model Architecture - Proven settings
        grid_size=512,
        base_filters=32,
        depth=4,
        dropout_rate=0.2,
        
        # Callback Parameters - Stable
        early_stopping_patience=20,
        reduce_lr_patience=10,
        reduce_lr_factor=0.5,
        min_lr=1e-7,
        
        # Performance Settings
        gpu_memory_growth=True,
        use_mixed_precision=True,
        max_workers=4,
        
        # Logging - Production level
        log_level="WARNING"  # Minimal logging for production
    )
    
    print(f"Production Config Created:")
    print(f"  All paths use absolute production paths")
    print(f"  Balanced performance and reliability settings")
    print(f"  Minimal logging for production environment")
    
    return config


def adaptive_configuration():
    """
    Create configuration that adapts to system capabilities.
    """
    print("\n=== Adaptive Configuration ===")
    
    # Check system capabilities
    memory_info = get_memory_info()
    gpu_info = check_gpu_availability()
    
    print(f"System Analysis:")
    print(f"  Available Memory: {memory_info.get('available_mb', 0):.0f} MB")
    print(f"  GPU Available: {gpu_info.get('gpu_available', False)}")
    
    # Adapt configuration based on system
    if memory_info.get('available_mb', 0) > 16000:  # > 16GB
        print("  Detected high-memory system - using large config")
        base_config = create_high_performance_config()
    elif memory_info.get('available_mb', 0) > 8000:  # > 8GB
        print("  Detected medium-memory system - using balanced config")
        base_config = Config()  # Default config
    else:
        print("  Detected low-memory system - using efficient config")
        base_config = create_memory_efficient_config()
    
    # Adjust for GPU availability
    if not gpu_info.get('gpu_available', False):
        print("  No GPU detected - adjusting for CPU processing")
        base_config.batch_size = max(1, base_config.batch_size // 2)
        base_config.grid_size = min(256, base_config.grid_size)
        base_config.use_mixed_precision = False
    
    return base_config


def configuration_validation_example():
    """
    Example of configuration validation and error handling.
    """
    print("\n=== Configuration Validation Example ===")
    
    # Test valid configuration
    try:
        valid_config = Config(
            epochs=100,
            batch_size=8,
            learning_rate=0.001,
            grid_size=512
        )
        print("✓ Valid configuration created successfully")
    except Exception as e:
        print(f"✗ Valid configuration failed: {e}")
    
    # Test invalid configurations
    invalid_configs = [
        {"epochs": -10, "description": "Negative epochs"},
        {"batch_size": 0, "description": "Zero batch size"},
        {"learning_rate": -0.001, "description": "Negative learning rate"},
        {"validation_split": 1.5, "description": "Invalid validation split"},
        {"grid_size": 16, "description": "Too small grid size"}
    ]
    
    for invalid_params in invalid_configs:
        description = invalid_params.pop("description")
        try:
            Config(**invalid_params)
            print(f"✗ {description} - Should have failed but didn't")
        except ValueError as e:
            print(f"✓ {description} - Correctly caught: {str(e)[:50]}...")
        except Exception as e:
            print(f"? {description} - Unexpected error: {e}")


def configuration_comparison():
    """
    Compare different configuration scenarios.
    """
    print("\n=== Configuration Comparison ===")
    
    configs = {
        "High Performance": create_high_performance_config(),
        "Memory Efficient": create_memory_efficient_config(),
        "Rapid Prototype": create_rapid_prototyping_config()
    }
    
    print("\nConfiguration Comparison Table:")
    print("-" * 80)
    print(f"{'Setting':<20} {'High Perf':<12} {'Memory Eff':<12} {'Rapid Proto':<12}")
    print("-" * 80)
    
    settings = ['grid_size', 'base_filters', 'depth', 'batch_size', 'epochs']
    
    for setting in settings:
        values = [str(getattr(config, setting)) for config in configs.values()]
        print(f"{setting:<20} {values[0]:<12} {values[1]:<12} {values[2]:<12}")
    
    print("-" * 80)


def save_and_load_configurations():
    """
    Example of saving and loading different configurations.
    """
    print("\n=== Save and Load Configurations ===")
    
    configs = {
        "high_performance": create_high_performance_config(),
        "memory_efficient": create_memory_efficient_config(),
        "rapid_prototype": create_rapid_prototyping_config(),
        "production": create_production_config()
    }
    
    # Save all configurations
    saved_files = []
    for name, config in configs.items():
        filename = f"config_{name}.json"
        try:
            config.save(filename)
            saved_files.append(filename)
            print(f"✓ Saved {name} configuration to {filename}")
        except Exception as e:
            print(f"✗ Failed to save {name}: {e}")
    
    # Load and verify configurations
    print("\nVerifying saved configurations:")
    for filename in saved_files:
        try:
            loaded_config = Config.load(filename)
            print(f"✓ Loaded {filename} - Grid size: {loaded_config.grid_size}")
        except Exception as e:
            print(f"✗ Failed to load {filename}: {e}")
    
    # Clean up
    for filename in saved_files:
        try:
            Path(filename).unlink()
            print(f"✓ Cleaned up {filename}")
        except Exception as e:
            print(f"✗ Failed to clean up {filename}: {e}")


def main():
    """
    Main function to demonstrate custom configuration examples.
    """
    print("Bathymetric CAE - Custom Configuration Examples")
    print("=" * 60)
    
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Run configuration examples
    create_high_performance_config()
    create_memory_efficient_config() 
    create_rapid_prototyping_config()
    create_production_config()
    
    adaptive_config = adaptive_configuration()
    print(f"Adaptive config selected - Grid size: {adaptive_config.grid_size}")
    
    configuration_validation_example()
    configuration_comparison()
    save_and_load_configurations()
    
    print("\n" + "=" * 60)
    print("Custom Configuration Examples Completed!")
    print("\nKey Takeaways:")
    print("1. Adapt configuration to your system capabilities")
    print("2. Use high-performance config for quality, memory-efficient for constraints")
    print("3. Rapid prototyping config for quick testing and iteration")
    print("4. Production config for stable, reliable deployment")
    print("5. Always validate configurations before use")
    print("6. Save configurations for reproducible experiments")


if __name__ == "__main__":
    main()
