# Bathymetric CAE Project Structure

This document provides a comprehensive overview of the Bathymetric CAE project structure, including detailed descriptions of each module, their responsibilities, and interconnections.

## Directory Structure

```
bathymetric_cae/
├── LICENSE                           # MIT License
├── README.md                         # Main project documentation
├── requirements.txt                  # Python dependencies
├── setup.py                         # Package installation configuration
│
├── bathymetric_cae/                  # Main package directory
│   ├── __init__.py                   # Package initialization and public API
│   ├── __main__.py                   # Module entry point for CLI execution
│   │
│   ├── cli/                          # Command Line Interface
│   │   ├── __init__.py               # CLI module initialization
│   │   └── main.py                   # Main CLI implementation and argument parsing
│   │
│   ├── config/                       # Configuration Management
│   │   ├── __init__.py               # Config module initialization
│   │   ├── config.py                 # Configuration class with validation
│   │   └── default_config.json       # Default configuration parameters
│   │
│   ├── core/                         # Core Processing Components
│   │   ├── __init__.py               # Core module initialization
│   │   ├── processor.py              # Bathymetric data processor
│   │   ├── model.py                  # Advanced CAE model architecture
│   │   └── pipeline.py               # Main processing pipeline orchestration
│   │
│   ├── utils/                        # Utility Modules
│   │   ├── __init__.py               # Utils module initialization
│   │   ├── logging_utils.py          # Enhanced logging with colored output
│   │   ├── memory_utils.py           # Memory monitoring and optimization
│   │   ├── gpu_utils.py              # GPU configuration and optimization
│   │   └── file_utils.py             # File handling and validation utilities
│   │
│   └── visualization/                # Visualization Components
│       ├── __init__.py               # Visualization module initialization
│       └── visualizer.py             # Comprehensive plotting and analysis
│
├── tests/                            # Test Suite
│   ├── __init__.py                   # Test package initialization
│   ├── conftest.py                   # Pytest configuration and fixtures
│   ├── test_config.py                # Configuration module tests
│   ├── test_model.py                 # Model architecture tests
│   ├── test_pipeline.py              # Pipeline integration tests
│   └── test_processor.py             # Data processing tests
│
├── examples/                         # Usage Examples
│   ├── __init__.py                   # Examples module with metadata
│   ├── basic_usage.py                # Basic pipeline usage demonstration
│   ├── custom_config.py              # Advanced configuration examples
│   ├── batch_processing.py           # Large-scale batch processing
│   ├── single_file.py                # Interactive single file processing
│   └── advanced_model.py             # Custom model architectures
│
├── docs/                             # Documentation
│   ├── index.txt                     # Main documentation index
│   ├── installation.md              # Installation guide
│   ├── quickstart.md                # Quick start guide
│   ├── usage.md                      # Comprehensive usage guide
│   ├── project_structure.md          # This file
│   └── test_documentation.md         # Testing documentation and guidelines
│
└── scripts/                          # Setup and Deployment Scripts
    ├── install_deps.sh               # Dependency installation script
    └── setup_env.py                  # Environment setup utility
```

## Module Descriptions

### Core Package (`bathymetric_cae/`)

#### **`__init__.py`** - Package Initialization
- **Purpose**: Defines the public API and package metadata
- **Key Components**:
  - Version information and package constants (`__version__ = "1.0.0"`)
  - Public API exports for easy importing
  - Supported file formats definitions (BAG, GeoTIFF, ASCII Grid, XYZ)
  - Quick start function for immediate usage
- **Dependencies**: All core modules
- **Usage**: `from bathymetric_cae import BathymetricCAEPipeline, Config`

#### **`__main__.py`** - Module Entry Point
- **Purpose**: Enables package execution as a module (`python -m bathymetric_cae`)
- **Functionality**: Direct bridge to CLI interface
- **Usage**: `python -m bathymetric_cae --input data/ --output results/`

### Command Line Interface (`cli/`)

#### **`main.py`** - CLI Implementation
- **Purpose**: Complete command-line interface with argument parsing
- **Key Features**:
  - Comprehensive argument parser with help documentation
  - Multiple operation modes (batch, single-file, validation)
  - Configuration file loading and command-line overrides
  - Logging setup and system validation
  - Error handling and user-friendly messages
- **Functions**:
  - `create_argument_parser()`: Enhanced argument parser creation
  - `main()`: Primary CLI entry point
  - `validate_requirements_command()`: System validation
  - `process_single_file_command()`: Interactive processing
  - `run_main_pipeline()`: Batch processing execution

### Configuration Management (`config/`)

#### **`config.py`** - Configuration Class
- **Purpose**: Type-safe configuration management with validation
- **Key Features**:
  - Dataclass-based configuration with type hints
  - Comprehensive validation with detailed error messages
  - JSON serialization and deserialization
  - Configuration merging and updating
  - Default value management
- **Classes**:
  - `Config`: Main configuration class with validation
- **Functions**:
  - `load_default_config()`: Load default settings
  - `load_config_from_file()`: Load with fallback handling
  - `create_config_from_args()`: Command-line integration

#### **`default_config.json`** - Default Parameters
- **Purpose**: Default configuration values for all pipeline components
- **Sections**:
  - I/O paths and file handling
  - Training parameters and model architecture
  - Performance optimization settings
  - Logging and callback configuration

### Core Processing (`core/`)

#### **`processor.py`** - Bathymetric Data Processor
- **Purpose**: Handles preprocessing of various bathymetric file formats
- **Key Features**:
  - Multi-format support (BAG, GeoTIFF, ASCII Grid, XYZ)
  - Advanced data validation and cleaning
  - Uncertainty data processing for BAG files
  - Robust normalization using percentile-based scaling
  - Geospatial metadata preservation
  - Memory-efficient batch processing
- **Classes**:
  - `BathymetricProcessor`: Main processing class
- **Functions**:
  - `get_supported_formats()`: Format listing
  - `validate_gdal_installation()`: GDAL validation

#### **`model.py`** - Advanced CAE Architecture
- **Purpose**: Implements sophisticated Convolutional Autoencoder architecture
- **Key Features**:
  - Residual blocks with skip connections
  - Spatial attention mechanisms
  - Multi-channel input support (depth + uncertainty)
  - Combined loss function (MSE + SSIM)
  - Advanced optimization with AdamW and mixed precision
  - Configurable architecture depth and filter counts
  - Built-in data augmentation layers
- **Classes**:
  - `AdvancedCAE`: Model builder with modern architecture
- **Functions**:
  - `create_data_augmentation_layer()`: Data augmentation
  - `calculate_model_memory_requirements()`: Memory estimation

#### **`pipeline.py`** - Processing Pipeline
- **Purpose**: Orchestrates the complete bathymetric processing workflow
- **Key Features**:
  - End-to-end workflow management
  - Intelligent model training vs. loading decisions
  - Batch file processing with error recovery
  - Comprehensive progress monitoring and logging
  - Statistical analysis and quality metrics calculation
  - Automatic report generation with visualizations
  - Memory optimization and cleanup management
- **Classes**:
  - `BathymetricCAEPipeline`: Main pipeline orchestrator
- **Functions**:
  - `run_pipeline_from_config()`: Configuration-driven execution
  - `validate_pipeline_requirements()`: System validation

### Utility Modules (`utils/`)

#### **`logging_utils.py`** - Enhanced Logging System
- **Purpose**: Provides sophisticated logging with visual enhancements
- **Key Features**:
  - Colored console output with severity-based formatting
  - Simultaneous file and console logging
  - Context managers for operation timing
  - Function call decorators for automatic logging
  - System information logging for debugging
  - TensorFlow logging configuration and warning suppression
- **Classes**:
  - `ColoredFormatter`: Console color formatting
  - `ContextLogger`: Context manager for timed operations
- **Functions**:
  - `setup_logging()`: Complete logging configuration
  - `log_function_call()`: Decorator for function timing

#### **`memory_utils.py`** - Memory Management
- **Purpose**: Comprehensive memory monitoring and optimization
- **Key Features**:
  - Real-time memory usage tracking with process monitoring
  - GPU memory information and optimization
  - Context managers for memory leak detection
  - Garbage collection optimization and forced cleanup
  - Memory threshold warnings and alerts
  - Batch size optimization based on available memory
  - TensorFlow session cleanup utilities
- **Classes**:
  - `MemoryProfiler`: Detailed memory usage profiling
- **Functions**:
  - `memory_monitor()`: Context manager for monitoring
  - `get_optimal_batch_size()`: Memory-based optimization

#### **`gpu_utils.py`** - GPU Configuration
- **Purpose**: GPU detection, configuration, and optimization
- **Key Features**:
  - Automatic GPU detection and capability assessment
  - Memory growth configuration and limit setting
  - Mixed precision training setup for performance
  - XLA compilation enablement
  - Multi-GPU strategy creation and management
  - GPU utilization monitoring and reporting
  - Recommended settings based on model complexity
- **Functions**:
  - `check_gpu_availability()`: Comprehensive GPU detection
  - `optimize_gpu_performance()`: Performance optimization
  - `get_recommended_gpu_settings()`: Intelligent recommendations

#### **`file_utils.py`** - File Management
- **Purpose**: Robust file operations and validation utilities
- **Key Features**:
  - Comprehensive file validation and information extraction
  - Safe file operations with integrity verification
  - Batch file renaming and organization tools
  - Duplicate detection using hash comparison
  - Directory cleanup and maintenance utilities
  - File monitoring for automated processing workflows
  - Cross-platform path handling and permissions
- **Classes**:
  - `FileManager`: High-level file management operations
- **Functions**:
  - `get_valid_files()`: Format-aware file discovery
  - `safe_copy_file()`: Verified file copying

### Visualization (`visualization/`)

#### **`visualizer.py`** - Comprehensive Plotting
- **Purpose**: Rich visualization capabilities for analysis and monitoring
- **Key Features**:
  - Training history visualization with multiple metrics
  - Data comparison plots (original vs. processed vs. uncertainty)
  - Statistical analysis and distribution comparisons
  - Difference mapping and error analysis
  - Processing summary reports with success rates
  - Quality metrics visualization (SSIM, correlation)
  - Customizable plot themes and export formats
  - PDF report generation capabilities
- **Classes**:
  - `Visualizer`: Main visualization engine with customizable styling
- **Functions**:
  - `setup_matplotlib_backend()`: Headless environment support
  - `save_plots_as_pdf()`: Multi-plot PDF generation

### Test Suite (`tests/`)

#### **`conftest.py`** - Test Configuration
- **Purpose**: Pytest configuration, fixtures, and test utilities
- **Key Features**:
  - Environment setup for consistent testing
  - Mock data generators for various scenarios
  - Test fixtures for temporary directories and configurations
  - Conditional test markers for different environments
  - Helper functions for assertion validation
  - Integration with TensorFlow and GDAL when available
- **Classes**:
  - `TestDataGenerator`: Synthetic data creation for testing

#### **Test Modules** - Comprehensive Testing
- **`test_config.py`**: Configuration validation and serialization
- **`test_model.py`**: Model architecture and training functionality  
- **`test_pipeline.py`**: End-to-end pipeline integration
- **`test_processor.py`**: Data processing and validation

### Examples (`examples/`)

#### **`__init__.py`** - Example Metadata
- **Purpose**: Provides comprehensive example catalog and guidance
- **Key Features**:
  - Complete example listing with descriptions
  - Difficulty levels and time estimates
  - Requirement specifications for each example
  - Category-based organization
  - Environment validation utilities
- **Functions**:
  - `list_examples()`: Available example enumeration
  - `print_example_guide()`: Comprehensive usage guide

#### **Example Scripts** - Practical Demonstrations
- **`basic_usage.py`**: Fundamental pipeline usage patterns
- **`custom_config.py`**: Advanced configuration and optimization
- **`batch_processing.py`**: Large-scale processing workflows
- **`single_file.py`**: Interactive processing with visualization
- **`advanced_model.py`**: Custom architectures and training strategies

### Documentation (`docs/`)

#### **Documentation Suite**
- **`index.txt`**: Main documentation with feature overview and comprehensive package introduction
- **`installation.md`**: Platform-specific installation guides with dependency management
- **`quickstart.md`**: Rapid introduction and basic usage patterns
- **`usage.md`**: Comprehensive usage guide with advanced patterns, workflows, and integration examples
- **`project_structure.md`**: This architectural overview and module documentation
- **`test_documentation.md`**: Testing documentation, guidelines, and best practices

### Scripts (`scripts/`)

#### **Setup and Deployment Utilities**
- **`install_deps.sh`**: Automated dependency installation with platform detection
- **`setup_env.py`**: Environment configuration and project structure creation

## Architecture Patterns

### Modular Design
The project follows a clean modular architecture with clear separation of concerns:

- **Configuration Layer**: Centralized, validated parameter management
- **Core Processing**: Domain-specific bathymetric data handling
- **Model Architecture**: Advanced deep learning components
- **Utility Services**: Cross-cutting concerns (logging, memory, GPU)
- **Presentation Layer**: Visualization and user interfaces

### Dependency Flow
```
CLI ────────────► Pipeline ────────────► Processor
 │                   │                      │
 ▼                   ▼                      ▼
Config ──────────► Model ──────────────► Utils
 │                   │                      │
 ▼                   ▼                      ▼
Validation ────► Training ──────────► Visualization
```

### Error Handling Strategy
- **Graceful Degradation**: Continue processing when individual files fail
- **Comprehensive Logging**: Detailed error tracking and debugging information
- **User-Friendly Messages**: Clear error descriptions with suggested solutions
- **Recovery Mechanisms**: Automatic retry and fallback strategies

### Performance Optimization
- **Memory Management**: Intelligent monitoring and cleanup
- **GPU Utilization**: Automatic detection and optimization
- **Parallel Processing**: Multi-threaded file operations
- **Caching Strategies**: Intermediate result storage and reuse

### Extensibility Points
- **Custom Loss Functions**: Pluggable loss function architecture
- **Model Architectures**: Configurable depth, filters, and connections
- **Data Formats**: Extensible file format support
- **Visualization**: Customizable plotting and analysis tools

## Integration Interfaces

### Python API
```python
# High-level pipeline interface
from bathymetric_cae import BathymetricCAEPipeline, Config

# Component-level access
from bathymetric_cae.core import BathymetricProcessor, AdvancedCAE
from bathymetric_cae.utils import memory_monitor, check_gpu_availability
```

### Command Line Interface
```bash
# Direct execution
bathymetric-cae --input data/ --output results/

# Module execution
python -m bathymetric_cae --config custom.json
```

### Configuration Interface
```json
{
  "input_folder": "./data",
  "model_architecture": {
    "grid_size": 512,
    "base_filters": 32,
    "depth": 4
  },
  "training": {
    "epochs": 100,
    "batch_size": 8
  }
}
```

## Quality Assurance

### Testing Strategy
- **Unit Tests**: Individual component validation with mocks and fixtures
- **Integration Tests**: End-to-end workflow verification
- **Performance Tests**: Memory and speed benchmarking
- **Conditional Testing**: Environment-specific test execution (TensorFlow, GDAL, GPU)

### Code Quality
- **Type Hints**: Comprehensive type annotations throughout
- **Documentation**: Detailed docstrings and inline comments
- **Linting**: Code style enforcement
- **Error Handling**: Robust exception management

### Test Organization
- **Fixtures**: Centralized test data and environment setup in `conftest.py`
- **Markers**: Conditional test execution based on available dependencies
- **Mock Strategy**: Comprehensive mocking of external dependencies
- **Data Generation**: Synthetic test data for reproducible testing

## Key Design Decisions

### Configuration Management
- **Location**: Integrated within the main package (`bathymetric_cae/config/`)
- **Validation**: Comprehensive parameter validation with meaningful error messages
- **Serialization**: JSON-based with special value handling
- **Flexibility**: Command-line overrides and programmatic updates

### Package Structure
- **Self-Contained**: All components within the main package directory
- **Clear Separation**: Distinct modules for core functionality, utilities, and interfaces
- **Extensibility**: Easy addition of new formats, models, and utilities
- **Testing**: Comprehensive test coverage with realistic scenarios

### Error Handling
- **Graceful Degradation**: System continues operation when individual components fail
- **Detailed Logging**: Rich error information for debugging
- **User-Friendly**: Clear error messages with actionable suggestions
- **Recovery**: Automatic retry and fallback mechanisms

This modular architecture ensures maintainability, testability, and extensibility while providing a comprehensive solution for bathymetric data processing using advanced machine learning techniques. The clear separation of concerns and well-defined interfaces make it easy to understand, extend, and maintain the codebase.