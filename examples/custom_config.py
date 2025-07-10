# examples/custom_config.py
"""
Custom configuration example for Enhanced Bathymetric CAE Processing.

This example shows how to create custom configurations for different
processing scenarios.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bathymetric_cae import Config


def create_high_quality_config():
    """Create configuration for high-quality processing."""
    
    config = Config()
    
    # High-quality settings
    config.ensemble_size = 5
    config.epochs = 200
    config.batch_size = 4  # Smaller for stability
    config.learning_rate = 0.0005  # Lower for better convergence
    config.early_stopping_patience = 25
    
    # Enable all quality features
    config.enable_adaptive_processing = True
    config.enable_expert_review = True
    config.enable_constitutional_constraints = True
    
    # Strict quality requirements
    config.quality_threshold = 0.9
    
    # Quality metric weights favoring feature preservation
    config.ssim_weight = 0.3
    config.roughness_weight = 0.1
    config.feature_preservation_weight = 0.4
    config.consistency_weight = 0.2
    
    # Performance optimizations
    config.use_mixed_precision = True
    config.gpu_memory_growth = True
    
    config.save("examples/high_quality_config.json")
    print("High-quality configuration saved!")
    return config


def create_fast_processing_config():
    """Create configuration for fast processing."""
    
    config = Config()
    
    # Speed-optimized settings
    config.ensemble_size = 1  # Single model
    config.epochs = 25
    config.batch_size = 16  # Larger batches
    config.grid_size = 256  # Smaller grid
    config.learning_rate = 0.001
    
    # Minimal quality checks
    config.enable_adaptive_processing = False
    config.enable_expert_review = False
    config.enable_constitutional_constraints = False
    
    # Lower quality threshold
    config.quality_threshold = 0.5
    
    # Performance settings
    config.use_mixed_precision = True
    config.max_workers = 8
    
    config.save("examples/fast_processing_config.json")
    print("Fast processing configuration saved!")
    return config


def create_memory_optimized_config():
    """Create configuration for memory-constrained systems."""
    
    config = Config()
    
    # Memory-conscious settings
    config.ensemble_size = 2
    config.epochs = 100
    config.batch_size = 2  # Very small batches
    config.grid_size = 256  # Reduced grid size
    config.base_filters = 16  # Fewer filters
    config.depth = 3  # Shallower network
    
    # Enable memory optimizations
    config.use_mixed_precision = True
    config.gpu_memory_growth = True
    
    # Standard quality settings
    config.enable_adaptive_processing = True
    config.quality_threshold = 0.7
    
    config.save("examples/memory_optimized_config.json")
    print("Memory-optimized configuration saved!")
    return config


def create_research_config():
    """Create configuration for research/experimental use."""
    
    config = Config()
    
    # Research settings
    config.ensemble_size = 7  # Large ensemble
    config.epochs = 500
    config.batch_size = 6
    config.learning_rate = 0.0001  # Very careful learning
    config.early_stopping_patience = 50
    
    # All features enabled
    config.enable_adaptive_processing = True
    config.enable_expert_review = True
    config.enable_constitutional_constraints = True
    
    # Very strict quality
    config.quality_threshold = 0.95
    
    # Comprehensive logging
    config.log_level = "DEBUG"
    
    # Advanced quality weights
    config.ssim_weight = 0.25
    config.roughness_weight = 0.15
    config.feature_preservation_weight = 0.35
    config.consistency_weight = 0.25
    
    config.save("examples/research_config.json")
    print("Research configuration saved!")
    return config


def main():
    """Create all example configurations."""
    
    print("Creating custom configuration examples...")
    print("=" * 50)
    
    # Create examples directory
    Path("examples").mkdir(exist_ok=True)
    
    # Create different configurations
    create_high_quality_config()
    create_fast_processing_config()
    create_memory_optimized_config()
    create_research_config()
    
    print("\nAll custom configurations created!")
    print("\nUsage examples:")
    print("  High Quality: python main.py --config examples/high_quality_config.json")
    print("  Fast:         python main.py --config examples/fast_processing_config.json")
    print("  Memory Opt:   python main.py --config examples/memory_optimized_config.json")
    print("  Research:     python main.py --config examples/research_config.json")


if __name__ == "__main__":
    main()
