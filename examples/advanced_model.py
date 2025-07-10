"""
Advanced Model Configuration Example for Bathymetric CAE

This example demonstrates advanced model configuration options including:
- Custom model architectures
- Advanced training strategies
- Transfer learning and fine-tuning
- Model ensembles and comparison
- Architecture optimization

Author: Bathymetric CAE Team
License: MIT
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the package to the path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent))

from bathymetric_cae import (
    AdvancedCAE,
    Config,
    setup_logging,
    get_logger,
    calculate_model_memory_requirements,
    get_memory_info
)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.callbacks import LearningRateScheduler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - some examples will be skipped")


def demonstrate_model_architectures():
    """
    Demonstrate different model architecture configurations.
    """
    print("=== Model Architecture Variations ===")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping model demonstrations")
        return
    
    # Architecture variations
    architectures = {
        "compact": {
            "description": "Compact model for limited resources",
            "config": Config(
                grid_size=128,
                base_filters=8,
                depth=2,
                dropout_rate=0.1
            )
        },
        "standard": {
            "description": "Standard balanced architecture",
            "config": Config(
                grid_size=256,
                base_filters=16,
                depth=3,
                dropout_rate=0.2
            )
        },
        "large": {
            "description": "Large model for high-quality results",
            "config": Config(
                grid_size=512,
                base_filters=32,
                depth=4,
                dropout_rate=0.2
            )
        },
        "deep": {
            "description": "Deep architecture with many layers",
            "config": Config(
                grid_size=256,
                base_filters=16,
                depth=6,
                dropout_rate=0.3
            )
        },
        "wide": {
            "description": "Wide architecture with many filters",
            "config": Config(
                grid_size=256,
                base_filters=64,
                depth=3,
                dropout_rate=0.2
            )
        }
    }
    
    print("Architecture comparison:")
    print("-" * 80)
    print(f"{'Name':<12} {'Description':<30} {'Params':<10} {'Memory':<12}")
    print("-" * 80)
    
    for name, arch_info in architectures.items():
        config = arch_info["config"]
        description = arch_info["description"]
        
        # Calculate memory requirements
        memory_req = calculate_model_memory_requirements(config)
        
        print(f"{name:<12} {description:<30} "
              f"{memory_req['estimated_parameters']:<10,} "
              f"{memory_req['total_memory_mb']:<8.1f} MB")
    
    print("-" * 80)
    
    return architectures


def create_custom_architecture_example():
    """
    Example of creating a completely custom architecture.
    """
    print("\n=== Custom Architecture Example ===")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping custom architecture")
        return None
    
    def create_custom_cae(input_shape=(256, 256, 2), name="CustomCAE"):
        """
        Create a custom CAE architecture with unique features.
        """
        inputs = layers.Input(shape=input_shape, name="input")
        
        # Custom encoder with pyramid pooling
        x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Multi-scale feature extraction
        pool1 = layers.MaxPooling2D(2)(x)  # 1/2 scale
        pool2 = layers.MaxPooling2D(4)(x)  # 1/4 scale
        pool3 = layers.MaxPooling2D(8)(x)  # 1/8 scale
        
        # Process each scale
        feat1 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
        feat2 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool2)
        feat3 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool3)
        
        # Upsample and combine features
        up2 = layers.UpSampling2D(2)(feat2)
        up3 = layers.UpSampling2D(4)(feat3)
        
        # Resize to match feat1 dimensions
        feat2_resized = layers.Resizing(feat1.shape[1], feat1.shape[2])(up2)
        feat3_resized = layers.Resizing(feat1.shape[1], feat1.shape[2])(up3)
        
        # Combine multi-scale features
        combined = layers.Concatenate()([feat1, feat2_resized, feat3_resized])
        
        # Decoder
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
        
        model = models.Model(inputs, outputs, name=name)
        return model
    
    # Create the custom model
    try:
        custom_model = create_custom_cae()
        
        print(f"Custom model created successfully:")
        print(f"  Total parameters: {custom_model.count_params():,}")
        print(f"  Input shape: {custom_model.input_shape}")
        print(f"  Output shape: {custom_model.output_shape}")
        
        # Print model summary
        print("\nModel architecture summary:")
        custom_model.summary(line_length=100)
        
        return custom_model
        
    except Exception as e:
        print(f"Failed to create custom model: {e}")
        return None


def demonstrate_training_strategies():
    """
    Demonstrate advanced training strategies and techniques.
    """
    print("\n=== Advanced Training Strategies ===")
    
    strategies = {
        "progressive_training": {
            "description": "Start with low resolution, gradually increase",
            "phases": [
                {"epochs": 20, "grid_size": 128, "lr": 0.01},
                {"epochs": 30, "grid_size": 256, "lr": 0.001},
                {"epochs": 50, "grid_size": 512, "lr": 0.0001}
            ]
        },
        "curriculum_learning": {
            "description": "Train on easy examples first, then harder ones",
            "phases": [
                {"epochs": 25, "data_complexity": "low", "lr": 0.001},
                {"epochs": 35, "data_complexity": "medium", "lr": 0.0005},
                {"epochs": 40, "data_complexity": "high", "lr": 0.0001}
            ]
        },
        "cyclic_learning": {
            "description": "Use cyclic learning rates for better convergence",
            "base_lr": 0.0001,
            "max_lr": 0.01,
            "cycle_length": 20
        },
        "warm_restart": {
            "description": "Periodic learning rate restarts",
            "initial_lr": 0.001,
            "restart_epochs": [30, 60, 90],
            "lr_multiplier": 0.5
        }
    }
    
    print("Training strategy options:")
    for name, strategy in strategies.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {strategy['description']}")
        
        if 'phases' in strategy:
            print("  Training phases:")
            for i, phase in enumerate(strategy['phases'], 1):
                print(f"    Phase {i}: {phase}")
        else:
            print(f"  Parameters: {strategy}")


def implement_transfer_learning_example():
    """
    Example of implementing transfer learning with pre-trained models.
    """
    print("\n=== Transfer Learning Example ===")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping transfer learning")
        return
    
    def create_transfer_learning_setup():
        """
        Setup for transfer learning scenario.
        """
        scenarios = {
            "domain_adaptation": {
                "description": "Adapt model from one geographic region to another",
                "source_domain": "Arctic bathymetry",
                "target_domain": "Tropical bathymetry",
                "strategy": "Fine-tune last layers only"
            },
            "resolution_transfer": {
                "description": "Transfer from low-res to high-res processing",
                "source_resolution": "256x256",
                "target_resolution": "1024x1024",
                "strategy": "Progressive upsampling with layer addition"
            },
            "sensor_adaptation": {
                "description": "Adapt between different sensor types",
                "source_sensor": "Multibeam sonar",
                "target_sensor": "LiDAR bathymetry",
                "strategy": "Feature layer adaptation"
            },
            "uncertainty_transfer": {
                "description": "Transfer from single-band to uncertainty-aware",
                "source_channels": 1,
                "target_channels": 2,
                "strategy": "Channel expansion with pre-trained weights"
            }
        }
        
        print("Transfer learning scenarios:")
        for name, scenario in scenarios.items():
            print(f"\n{name.upper()}:")
            for key, value in scenario.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        return scenarios
    
    # Demonstrate transfer learning implementation
    scenarios = create_transfer_learning_setup()
    
    def implement_layer_freezing_strategy():
        """
        Example of layer freezing for transfer learning.
        """
        print("\nLayer freezing strategies:")
        
        strategies = [
            {
                "name": "Freeze encoder, train decoder",
                "frozen_layers": "encoder_*",
                "trainable_layers": "decoder_*",
                "use_case": "New output domain, similar input features"
            },
            {
                "name": "Freeze feature layers, train classifier",
                "frozen_layers": "conv2d_*, batch_normalization_*",
                "trainable_layers": "dense_*, output_*",
                "use_case": "Similar features, different output interpretation"
            },
            {
                "name": "Progressive unfreezing",
                "strategy": "Gradually unfreeze layers during training",
                "phases": "Start with output layers, work backwards",
                "use_case": "Maximum knowledge transfer with adaptation"
            }
        ]
        
        for strategy in strategies:
            print(f"\n  Strategy: {strategy['name']}")
            for key, value in strategy.items():
                if key != 'name':
                    print(f"    {key.replace('_', ' ').title()}: {value}")
    
    implement_layer_freezing_strategy()


def demonstrate_model_ensembles():
    """
    Demonstrate model ensemble techniques.
    """
    print("\n=== Model Ensemble Techniques ===")
    
    ensemble_methods = {
        "simple_averaging": {
            "description": "Average predictions from multiple models",
            "models": ["compact_model", "standard_model", "large_model"],
            "combination": "mean(predictions)",
            "pros": ["Simple to implement", "Reduces overfitting"],
            "cons": ["All models equally weighted"]
        },
        "weighted_averaging": {
            "description": "Weighted average based on model performance",
            "weights": {"model_1": 0.4, "model_2": 0.35, "model_3": 0.25},
            "combination": "weighted_mean(predictions, weights)",
            "pros": ["Better models have more influence"],
            "cons": ["Requires validation for weight selection"]
        },
        "stacking": {
            "description": "Meta-model learns to combine base models",
            "base_models": ["cae_1", "cae_2", "cae_3"],
            "meta_model": "simple_neural_network",
            "pros": ["Learns optimal combination", "Can be non-linear"],
            "cons": ["More complex", "Requires additional training"]
        },
        "boosting": {
            "description": "Sequential training focusing on difficult examples",
            "approach": "Train models sequentially on residuals",
            "combination": "additive_ensemble",
            "pros": ["Focuses on hard examples", "Often high performance"],
            "cons": ["Prone to overfitting", "Computationally expensive"]
        }
    }
    
    print("Ensemble methods comparison:")
    for method, details in ensemble_methods.items():
        print(f"\n{method.upper()}:")
        print(f"  Description: {details['description']}")
        if 'pros' in details:
            print(f"  Pros: {', '.join(details['pros'])}")
            print(f"  Cons: {', '.join(details['cons'])}")
    
    def implement_simple_ensemble_example():
        """
        Example implementation of a simple ensemble.
        """
        print("\nSimple ensemble implementation example:")
        
        # Pseudo-code for ensemble implementation
        ensemble_code = '''
def create_ensemble_predictor(model_paths, weights=None):
    """Create ensemble predictor from multiple models."""
    models = [load_model(path) for path in model_paths]
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    def ensemble_predict(input_data):
        predictions = []
        for model in models:
            pred = model.predict(input_data)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    return ensemble_predict

# Usage example
ensemble = create_ensemble_predictor([
    "model_1.h5", "model_2.h5", "model_3.h5"
], weights=[0.4, 0.35, 0.25])

result = ensemble(input_bathymetric_data)
'''
        
        print(ensemble_code)
    
    implement_simple_ensemble_example()


def optimize_model_performance():
    """
    Demonstrate model performance optimization techniques.
    """
    print("\n=== Model Performance Optimization ===")
    
    optimization_techniques = {
        "quantization": {
            "description": "Reduce model precision for faster inference",
            "methods": ["Post-training quantization", "Quantization-aware training"],
            "benefits": ["Smaller model size", "Faster inference", "Lower memory usage"],
            "trade_offs": ["Slight accuracy loss"]
        },
        "pruning": {
            "description": "Remove unnecessary model weights",
            "methods": ["Magnitude-based pruning", "Structured pruning"],
            "benefits": ["Smaller models", "Faster inference"],
            "trade_offs": ["Requires fine-tuning", "Complex implementation"]
        },
        "knowledge_distillation": {
            "description": "Train smaller student model from larger teacher",
            "process": "Teacher model guides student training",
            "benefits": ["Compact models", "Maintains performance"],
            "trade_offs": ["Requires teacher model", "Additional training"]
        },
        "mixed_precision": {
            "description": "Use FP16 and FP32 strategically",
            "implementation": "Automatic mixed precision (AMP)",
            "benefits": ["Faster training", "Lower memory usage"],
            "trade_offs": ["Requires modern hardware"]
        }
    }
    
    print("Performance optimization techniques:")
    for technique, details in optimization_techniques.items():
        print(f"\n{technique.upper()}:")
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key.title()}: {', '.join(value)}")
            else:
                print(f"  {key.title()}: {value}")


def model_architecture_search():
    """
    Demonstrate automated model architecture search concepts.
    """
    print("\n=== Neural Architecture Search (NAS) Concepts ===")
    
    nas_approaches = {
        "grid_search": {
            "description": "Systematic search over predefined parameter grid",
            "parameters": ["depth", "width", "kernel_sizes", "activation_functions"],
            "complexity": "Low",
            "computational_cost": "Medium"
        },
        "random_search": {
            "description": "Random sampling of architecture parameters",
            "efficiency": "Better than grid search for high dimensions",
            "complexity": "Low",
            "computational_cost": "Medium"
        },
        "bayesian_optimization": {
            "description": "Use Bayesian methods to guide architecture search",
            "advantage": "Efficient exploration of search space",
            "complexity": "Medium",
            "computational_cost": "Medium-High"
        },
        "evolutionary_search": {
            "description": "Evolutionary algorithms for architecture optimization",
            "process": "Mutation and crossover of architectures",
            "complexity": "High",
            "computational_cost": "High"
        }
    }
    
    print("Architecture search approaches:")
    for approach, details in nas_approaches.items():
        print(f"\n{approach.upper()}:")
        for key, value in details.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Example search space definition
    print("\nExample search space for bathymetric CAE:")
    search_space = {
        "depth": [2, 3, 4, 5, 6],
        "base_filters": [8, 16, 32, 64],
        "kernel_sizes": [(3, 3), (5, 5), (7, 7)],
        "activation": ["relu", "elu", "swish"],
        "normalization": ["batch", "layer", "instance"],
        "dropout_rate": [0.0, 0.1, 0.2, 0.3]
    }
    
    for param, values in search_space.items():
        print(f"  {param}: {values}")


def main():
    """
    Main function to demonstrate advanced model configuration.
    """
    print("Advanced Model Configuration Examples")
    print("=" * 50)
    
    # Run all demonstrations
    demonstrate_model_architectures()
    create_custom_architecture_example()
    demonstrate_training_strategies()
    implement_transfer_learning_example()
    demonstrate_model_ensembles()
    optimize_model_performance()
    model_architecture_search()
    
    print("\nAdvanced model examples completed!")


if __name__ == "__main__":
    main()