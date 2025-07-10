"""
Examples Module for Bathymetric CAE

This module contains comprehensive examples demonstrating various aspects
of the bathymetric CAE pipeline including basic usage, advanced configuration,
batch processing, single file processing, and performance optimization.

Author: Bathymetric CAE Team
License: MIT
"""

from pathlib import Path

# Example module information
EXAMPLES_VERSION = "1.0.0"
EXAMPLES_DIR = Path(__file__).parent

# Available examples
AVAILABLE_EXAMPLES = {
    "basic_usage": {
        "module": "basic_usage",
        "description": "Basic pipeline usage with default settings",
        "difficulty": "Beginner",
        "estimated_time": "5-10 minutes",
        "requirements": ["Sample data files"]
    },
    "custom_config": {
        "module": "custom_config", 
        "description": "Advanced configuration options and optimization",
        "difficulty": "Intermediate",
        "estimated_time": "10-15 minutes",
        "requirements": ["Understanding of model parameters"]
    },
    "batch_processing": {
        "module": "batch_processing",
        "description": "Large-scale batch processing and monitoring",
        "difficulty": "Intermediate",
        "estimated_time": "15-20 minutes", 
        "requirements": ["Multiple bathymetric files"]
    },
    "single_file_processing": {
        "module": "single_file_processing",
        "description": "Interactive single file processing with visualization",
        "difficulty": "Beginner",
        "estimated_time": "5-10 minutes",
        "requirements": ["Single bathymetric file", "Trained model"]
    },
    "advanced_model_config": {
        "module": "advanced_model_config",
        "description": "Advanced model architectures and training strategies",
        "difficulty": "Advanced",
        "estimated_time": "20-30 minutes",
        "requirements": ["Deep learning knowledge", "TensorFlow"]
    },
    "gpu_optimization": {
        "module": "gpu_optimization",
        "description": "GPU optimization and performance tuning",
        "difficulty": "Intermediate",
        "estimated_time": "10-15 minutes",
        "requirements": ["GPU hardware"]
    },
    "visualization_examples": {
        "module": "visualization_examples",
        "description": "Comprehensive visualization and analysis techniques",
        "difficulty": "Beginner",
        "estimated_time": "10-15 minutes",
        "requirements": ["Processed results"]
    }
}

# Example categories
EXAMPLE_CATEGORIES = {
    "getting_started": ["basic_usage", "single_file_processing"],
    "configuration": ["custom_config", "advanced_model_config"],
    "processing": ["batch_processing", "single_file_processing"],
    "optimization": ["gpu_optimization", "advanced_model_config"],
    "visualization": ["visualization_examples", "single_file_processing"]
}


def list_examples():
    """
    List all available examples with descriptions.
    
    Returns:
        dict: Dictionary of available examples
    """
    return AVAILABLE_EXAMPLES


def get_example_info(example_name: str) -> dict:
    """
    Get detailed information about a specific example.
    
    Args:
        example_name: Name of the example
        
    Returns:
        dict: Example information
        
    Raises:
        KeyError: If example doesn't exist
    """
    if example_name not in AVAILABLE_EXAMPLES:
        raise KeyError(f"Example '{example_name}' not found. Available examples: {list(AVAILABLE_EXAMPLES.keys())}")
    
    return AVAILABLE_EXAMPLES[example_name]


def get_examples_by_category(category: str) -> list:
    """
    Get examples by category.
    
    Args:
        category: Category name
        
    Returns:
        list: List of example names in the category
        
    Raises:
        KeyError: If category doesn't exist
    """
    if category not in EXAMPLE_CATEGORIES:
        raise KeyError(f"Category '{category}' not found. Available categories: {list(EXAMPLE_CATEGORIES.keys())}")
    
    return EXAMPLE_CATEGORIES[category]


def get_examples_by_difficulty(difficulty: str) -> list:
    """
    Get examples by difficulty level.
    
    Args:
        difficulty: Difficulty level (Beginner, Intermediate, Advanced)
        
    Returns:
        list: List of example names at the specified difficulty
    """
    return [
        name for name, info in AVAILABLE_EXAMPLES.items()
        if info.get("difficulty", "").lower() == difficulty.lower()
    ]


def print_example_guide():
    """Print a comprehensive guide to all examples."""
    
    print("Bathymetric CAE Examples Guide")
    print("=" * 50)
    print()
    
    # Print by category
    for category, examples in EXAMPLE_CATEGORIES.items():
        print(f"{category.upper().replace('_', ' ')}:")
        
        for example_name in examples:
            if example_name in AVAILABLE_EXAMPLES:
                info = AVAILABLE_EXAMPLES[example_name]
                print(f"  📁 {example_name}")
                print(f"     Description: {info['description']}")
                print(f"     Difficulty: {info['difficulty']}")
                print(f"     Time: {info['estimated_time']}")
                print(f"     Requirements: {', '.join(info['requirements'])}")
                print()
        
        print()
    
    print("GETTING STARTED RECOMMENDATIONS:")
    print("1. Start with 'basic_usage' for your first experience")
    print("2. Try 'single_file_processing' for interactive analysis")
    print("3. Explore 'custom_config' to understand configuration options")
    print("4. Use 'batch_processing' for production workflows")
    print("5. Study 'advanced_model_config' for deep customization")
    print()
    
    print("RUNNING EXAMPLES:")
    print("Each example can be run independently:")
    print("  python examples/basic_usage.py")
    print("  python examples/custom_config.py")
    print("  python -m examples.batch_processing")
    print()


def create_example_summary():
    """
    Create a summary of all examples for documentation.
    
    Returns:
        str: Formatted summary text
    """
    summary_lines = [
        "# Bathymetric CAE Examples Summary",
        "",
        "This directory contains comprehensive examples demonstrating the capabilities",
        "of the Bathymetric CAE pipeline.",
        "",
        "## Available Examples",
        ""
    ]
    
    for name, info in AVAILABLE_EXAMPLES.items():
        summary_lines.extend([
            f"### {name.replace('_', ' ').title()}",
            f"- **File:** `{info['module']}.py`",
            f"- **Description:** {info['description']}",
            f"- **Difficulty:** {info['difficulty']}",
            f"- **Estimated Time:** {info['estimated_time']}",
            f"- **Requirements:** {', '.join(info['requirements'])}",
            ""
        ])
    
    summary_lines.extend([
        "## Example Categories",
        ""
    ])
    
    for category, examples in EXAMPLE_CATEGORIES.items():
        category_title = category.replace('_', ' ').title()
        example_list = ', '.join(examples)
        summary_lines.extend([
            f"### {category_title}",
            f"Examples: {example_list}",
            ""
        ])
    
    summary_lines.extend([
        "## Quick Start Guide",
        "",
        "1. **Begin with basics:** Run `basic_usage.py` to understand the fundamentals",
        "2. **Try single file processing:** Use `single_file_processing.py` for interactive analysis", 
        "3. **Explore configuration:** Study `custom_config.py` for advanced options",
        "4. **Scale up:** Use `batch_processing.py` for production workflows",
        "5. **Optimize:** Apply `gpu_optimization.py` and `advanced_model_config.py`",
        "",
        "## Running Examples",
        "",
        "Each example is self-contained and can be run independently:",
        "",
        "```bash",
        "# Basic approaches",
        "python examples/basic_usage.py",
        "python examples/single_file_processing.py",
        "",
        "# Advanced usage", 
        "python examples/batch_processing.py",
        "python examples/advanced_model_config.py",
        "```",
        "",
        "## Requirements",
        "",
        "- Python 3.8+",
        "- bathymetric-cae package installed",
        "- Sample bathymetric data files (for realistic examples)",
        "- GPU (recommended for performance examples)",
        "",
        "## Support",
        "",
        "For questions about examples:",
        "1. Check the inline documentation in each example file",
        "2. Review the main package documentation",
        "3. Open an issue on the project GitHub repository"
    ])
    
    return "\n".join(summary_lines)


def validate_example_environment():
    """
    Validate that the environment is set up correctly for running examples.
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        "package_available": False,
        "tensorflow_available": False,
        "gdal_available": False,
        "matplotlib_available": False,
        "examples_directory": False,
        "all_files_present": False
    }
    
    # Check package availability
    try:
        import bathymetric_cae
        validation_results["package_available"] = True
    except ImportError:
        pass
    
    # Check TensorFlow
    try:
        import tensorflow
        validation_results["tensorflow_available"] = True
    except ImportError:
        pass
    
    # Check GDAL
    try:
        from osgeo import gdal
        validation_results["gdal_available"] = True
    except ImportError:
        pass
    
    # Check matplotlib
    try:
        import matplotlib
        validation_results["matplotlib_available"] = True
    except ImportError:
        pass
    
    # Check examples directory
    validation_results["examples_directory"] = EXAMPLES_DIR.exists()
    
    # Check if all example files are present
    expected_files = [f"{module}.py" for module in AVAILABLE_EXAMPLES.keys()]
    all_present = all((EXAMPLES_DIR / filename).exists() for filename in expected_files)
    validation_results["all_files_present"] = all_present
    
    return validation_results


# Public API
__all__ = [
    "AVAILABLE_EXAMPLES",
    "EXAMPLE_CATEGORIES", 
    "list_examples",
    "get_example_info",
    "get_examples_by_category",
    "get_examples_by_difficulty",
    "print_example_guide",
    "create_example_summary",
    "validate_example_environment"
]


# Auto-generate README if run as script
if __name__ == "__main__":
    print_example_guide()
    
    # Optionally create README file
    readme_content = create_example_summary()
    readme_path = EXAMPLES_DIR / "README.md"
    
    try:
        readme_path.write_text(readme_content)
        print(f"Generated README.md at: {readme_path}")
    except Exception as e:
        print(f"Could not create README.md: {e}")
    
    # Validate environment
    print("\nEnvironment Validation:")
    validation = validate_example_environment()
    
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    if all(validation.values()):
        print("\n🎉 Environment is ready for running examples!")
    else:
        print("\n⚠️  Some requirements are missing. Check installation guide.")
