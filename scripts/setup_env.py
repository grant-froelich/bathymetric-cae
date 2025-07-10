"""
Environment Setup Script for Bathymetric CAE

This script sets up the development and runtime environment for the
Bathymetric CAE package, including configuration, directory structure,
and initial validation.

Author: Bathymetric CAE Team
License: MIT
"""

import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

# Add package to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from bathymetric_cae import (
        Config, 
        validate_installation,
        setup_logging,
        get_memory_info,
        check_gpu_availability
    )
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_colored(message: str, color: str = Colors.WHITE):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.ENDC}")


def print_header(title: str):
    """Print formatted header."""
    print_colored("\n" + "=" * 60, Colors.BLUE)
    print_colored(f" {title}", Colors.BOLD + Colors.BLUE)
    print_colored("=" * 60, Colors.BLUE)


def print_success(message: str):
    """Print success message."""
    print_colored(f"✓ {message}", Colors.GREEN)


def print_warning(message: str):
    """Print warning message."""
    print_colored(f"⚠ {message}", Colors.YELLOW)


def print_error(message: str):
    """Print error message."""
    print_colored(f"✗ {message}", Colors.RED)


def print_info(message: str):
    """Print info message."""
    print_colored(f"ℹ {message}", Colors.CYAN)


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        return False, version_str
    
    return True, version_str


def check_system_info() -> Dict[str, str]:
    """Get system information."""
    return {
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor() or "Unknown",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation()
    }


def check_dependencies() -> Dict[str, Dict[str, any]]:
    """Check availability of required dependencies."""
    dependencies = {
        "numpy": {"required": True, "available": False, "version": None},
        "tensorflow": {"required": True, "available": False, "version": None},
        "gdal": {"required": True, "available": False, "version": None},
        "matplotlib": {"required": True, "available": False, "version": None},
        "seaborn": {"required": True, "available": False, "version": None},
        "scikit-image": {"required": True, "available": False, "version": None},
        "joblib": {"required": True, "available": False, "version": None},
        "psutil": {"required": True, "available": False, "version": None},
        "pytest": {"required": False, "available": False, "version": None},
        "sphinx": {"required": False, "available": False, "version": None}
    }
    
    for dep_name, dep_info in dependencies.items():
        try:
            if dep_name == "gdal":
                from osgeo import gdal
                module = gdal
                version = gdal.__version__
            else:
                module = __import__(dep_name)
                version = getattr(module, '__version__', 'Unknown')
            
            dep_info["available"] = True
            dep_info["version"] = version
            
        except ImportError:
            pass
    
    return dependencies


def create_directory_structure(base_path: Path) -> Dict[str, Path]:
    """Create project directory structure."""
    directories = {
        "data": base_path / "data",
        "input": base_path / "data" / "input",
        "output": base_path / "data" / "output", 
        "models": base_path / "models",
        "logs": base_path / "logs",
        "plots": base_path / "plots",
        "reports": base_path / "reports",
        "config": base_path / "config",
        "temp": base_path / "temp"
    }
    
    created_dirs = {}
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            created_dirs[name] = path
            print_success(f"Created directory: {path}")
        except PermissionError:
            print_error(f"Permission denied creating: {path}")
        except Exception as e:
            print_error(f"Failed to create {path}: {e}")
    
    return created_dirs


def create_default_config(config_dir: Path) -> Optional[Path]:
    """Create default configuration file."""
    if not PACKAGE_AVAILABLE:
        print_warning("Package not available, skipping config creation")
        return None
    
    try:
        config = Config()
        config_file = config_dir / "default_config.json"
        config.save(str(config_file))
        print_success(f"Created default config: {config_file}")
        return config_file
    except Exception as e:
        print_error(f"Failed to create config: {e}")
        return None


def create_gitignore(base_path: Path) -> Optional[Path]:
    """Create .gitignore file for the project."""
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
data/input/*
!data/input/.gitkeep
data/output/*
!data/output/.gitkeep
models/*.h5
models/*.hdf5
logs/*
!logs/.gitkeep
plots/*.png
plots/*.jpg
plots/*.pdf
reports/*.html
reports/*.pdf
temp/*
!temp/.gitkeep

# System files
.DS_Store
Thumbs.db
"""
    
    try:
        gitignore_file = base_path / ".gitignore"
        gitignore_file.write_text(gitignore_content.strip())
        print_success(f"Created .gitignore: {gitignore_file}")
        return gitignore_file
    except Exception as e:
        print_error(f"Failed to create .gitignore: {e}")
        return None


def create_readme(base_path: Path, project_name: str) -> Optional[Path]:
    """Create project README file."""
    readme_content = f"""# {project_name}

Bathymetric data processing project using the Bathymetric CAE package.

## Project Structure

- `data/input/` - Input bathymetric files
- `data/output/` - Processed output files
- `models/` - Trained model files
- `logs/` - Training and processing logs
- `plots/` - Generated visualizations
- `reports/` - Processing reports
- `config/` - Configuration files
- `temp/` - Temporary files

## Quick Start

1. Place your bathymetric files in `data/input/`
2. Configure processing parameters in `config/`
3. Run processing pipeline:

```bash
# Basic processing
bathymetric-cae --input data/input --output data/output

# Custom configuration
bathymetric-cae --config config/my_config.json
```

## Configuration

Edit `config/default_config.json` to customize:
- Model architecture parameters
- Training settings
- Input/output paths
- Performance options

## Examples

See the bathymetric_cae package examples for detailed usage patterns.

## Support

For issues and questions:
- Check the package documentation
- Review example scripts
- Open an issue on the project repository

Generated by Bathymetric CAE setup script.
"""
    
    try:
        readme_file = base_path / "README.md"
        readme_file.write_text(readme_content)
        print_success(f"Created README: {readme_file}")
        return readme_file
    except Exception as e:
        print_error(f"Failed to create README: {e}")
        return None


def create_placeholder_files(directories: Dict[str, Path]):
    """Create placeholder files in empty directories."""
    placeholder_content = "# This file ensures the directory is tracked by git\n"
    
    placeholder_dirs = ["input", "output", "logs", "temp"]
    
    for dir_name in placeholder_dirs:
        if dir_name in directories:
            placeholder_file = directories[dir_name] / ".gitkeep"
            try:
                placeholder_file.write_text(placeholder_content)
                print_success(f"Created placeholder: {placeholder_file}")
            except Exception as e:
                print_warning(f"Failed to create placeholder in {dir_name}: {e}")


def validate_environment() -> Dict[str, bool]:
    """Validate the complete environment setup."""
    validation_results = {}
    
    # Check Python version
    python_ok, python_version = check_python_version()
    validation_results["python_version"] = python_ok
    
    if python_ok:
        print_success(f"Python version {python_version} is compatible")
    else:
        print_error(f"Python version {python_version} is not compatible (3.8+ required)")
    
    # Check dependencies
    dependencies = check_dependencies()
    required_available = all(
        dep["available"] for dep in dependencies.values() 
        if dep["required"]
    )
    validation_results["dependencies"] = required_available
    
    if required_available:
        print_success("All required dependencies are available")
    else:
        print_error("Some required dependencies are missing")
        for name, dep in dependencies.items():
            if dep["required"] and not dep["available"]:
                print_error(f"  Missing: {name}")
    
    # Check package installation
    if PACKAGE_AVAILABLE:
        try:
            pkg_validation = validate_installation()
            validation_results["package"] = pkg_validation.get("all_requirements_met", False)
            
            if validation_results["package"]:
                print_success("Bathymetric CAE package validation passed")
            else:
                print_warning("Bathymetric CAE package validation had issues")
        except Exception as e:
            validation_results["package"] = False
            print_error(f"Package validation failed: {e}")
    else:
        validation_results["package"] = False
        print_error("Bathymetric CAE package not available")
    
    return validation_results


def print_system_summary():
    """Print comprehensive system summary."""
    print_header("System Information")
    
    # System info
    system_info = check_system_info()
    for key, value in system_info.items():
        print_info(f"{key.replace('_', ' ').title()}: {value}")
    
    # Memory info
    if PACKAGE_AVAILABLE:
        try:
            memory_info = get_memory_info()
            print_info(f"Available Memory: {memory_info.get('available_mb', 0):.0f} MB")
            print_info(f"Total Memory: {memory_info.get('total_mb', 0):.0f} MB")
        except Exception:
            print_warning("Could not get memory information")
    
    # GPU info
    if PACKAGE_AVAILABLE:
        try:
            gpu_info = check_gpu_availability()
            if gpu_info.get("gpu_available", False):
                print_success(f"GPU Available: {gpu_info.get('num_physical_gpus', 0)} GPU(s)")
            else:
                print_info("GPU: Not available (CPU processing will be used)")
        except Exception:
            print_warning("Could not get GPU information")


def print_dependency_summary():
    """Print dependency status summary."""
    print_header("Dependency Status")
    
    dependencies = check_dependencies()
    
    for name, dep in dependencies.items():
        status_icon = "✓" if dep["available"] else "✗"
        status_color = Colors.GREEN if dep["available"] else Colors.RED
        required_text = " (Required)" if dep["required"] else " (Optional)"
        version_text = f" v{dep['version']}" if dep["version"] else ""
        
        print_colored(
            f"{status_icon} {name}{version_text}{required_text}", 
            status_color
        )


def create_sample_script(base_path: Path) -> Optional[Path]:
    """Create a sample processing script."""
    script_content = '''#!/usr/bin/env python3
"""
Sample Bathymetric Processing Script

This script demonstrates basic usage of the Bathymetric CAE package
for processing bathymetric data files.
"""

from pathlib import Path
from bathymetric_cae import Config, BathymetricCAEPipeline, setup_logging

def main():
    """Main processing function."""
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Define paths
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    model_path = Path("models/bathymetric_model.h5")
    
    # Create configuration
    config = Config(
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        model_path=str(model_path),
        epochs=50,  # Adjust as needed
        batch_size=8,
        grid_size=256
    )
    
    # Create and run pipeline
    pipeline = BathymetricCAEPipeline(config)
    
    try:
        results = pipeline.run(
            input_folder=str(input_dir),
            output_folder=str(output_dir),
            model_path=str(model_path)
        )
        
        print(f"Processing completed successfully!")
        print(f"Success rate: {results['pipeline_info']['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
    
    try:
        script_file = base_path / "process_bathymetry.py"
        script_file.write_text(script_content)
        script_file.chmod(0o755)  # Make executable
        print_success(f"Created sample script: {script_file}")
        return script_file
    except Exception as e:
        print_error(f"Failed to create sample script: {e}")
        return None


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup environment for Bathymetric CAE project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--project-name", 
        default="bathymetric_project",
        help="Name for the project"
    )
    
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path.cwd(),
        help="Base path for project setup"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample processing script"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    print_header("Bathymetric CAE Environment Setup")
    
    # Print system summary
    print_system_summary()
    
    # Print dependency summary
    print_dependency_summary()
    
    # Validate environment if requested
    if not args.skip_validation:
        print_header("Environment Validation")
        validation_results = validate_environment()
        
        if not all(validation_results.values()):
            print_warning("Some validation checks failed. Consider running the installation script.")
    
    # Setup project structure
    print_header("Creating Project Structure")
    
    project_path = args.base_path / args.project_name
    project_path.mkdir(exist_ok=True)
    print_success(f"Project directory: {project_path}")
    
    # Create directories
    directories = create_directory_structure(project_path)
    
    # Create configuration files
    if "config" in directories:
        create_default_config(directories["config"])
    
    # Create project files
    create_gitignore(project_path)
    create_readme(project_path, args.project_name)
    create_placeholder_files(directories)
    
    # Create sample script if requested
    if args.create_sample:
        create_sample_script(project_path)
    
    # Final summary
    print_header("Setup Complete")
    print_success(f"Project '{args.project_name}' setup complete!")
    print_info(f"Project location: {project_path}")
    print_info("Next steps:")
    print_info("1. Place bathymetric files in data/input/")
    print_info("2. Customize configuration in config/")
    print_info("3. Run processing with bathymetric-cae command")
    
    if args.create_sample:
        print_info("4. Or run the sample script: python process_bathymetry.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
