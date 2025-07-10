#!/bin/bash

# Bathymetric CAE Dependency Installation Script
# 
# This script installs all required dependencies for the Bathymetric CAE package
# including GDAL, TensorFlow, and other scientific computing libraries.
#
# Author: Bathymetric CAE Team
# License: MIT

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PYTHON_VERSION=""
INSTALL_GPU=false
INSTALL_DEV=false
INSTALL_DOCS=false
USE_CONDA=false
VERBOSE=false
DRY_RUN=false

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print usage
print_usage() {
    cat << EOF
Bathymetric CAE Dependency Installation Script

Usage: $0 [OPTIONS]

OPTIONS:
    -p, --python VERSION    Specify Python version (e.g., 3.8, 3.9, 3.10)
    -g, --gpu              Install GPU support (CUDA, cuDNN)
    -d, --dev              Install development dependencies
    -o, --docs             Install documentation dependencies
    -c, --conda            Use conda instead of pip
    -v, --verbose          Verbose output
    -n, --dry-run          Show commands without executing
    -h, --help             Show this help message

EXAMPLES:
    $0                     # Basic installation
    $0 --gpu --dev         # Install with GPU and development support
    $0 --conda --python 3.9  # Use conda with Python 3.9
    $0 --dry-run           # Preview installation commands

EOF
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Python version
get_python_version() {
    if [ -n "$PYTHON_VERSION" ]; then
        echo "$PYTHON_VERSION"
    else
        python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.8"
    fi
}

# Function to check system requirements
check_system_requirements() {
    print_color $BLUE "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_color $GREEN "✓ Linux detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_color $GREEN "✓ macOS detected"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_color $GREEN "✓ Windows detected"
    else
        print_color $RED "✗ Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check Python
    if command_exists python3; then
        PYTHON_CMD="python3"
        PYTHON_VER=$(get_python_version)
        print_color $GREEN "✓ Python $PYTHON_VER found"
    elif command_exists python; then
        PYTHON_CMD="python"
        PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
        if [[ "$PYTHON_VER" == "2."* ]]; then
            print_color $RED "✗ Python 2 detected. Python 3.8+ required."
            exit 1
        fi
        print_color $GREEN "✓ Python $PYTHON_VER found"
    else
        print_color $RED "✗ Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check if Python version is supported
    case "$PYTHON_VER" in
        3.8|3.9|3.10|3.11)
            print_color $GREEN "✓ Python version $PYTHON_VER is supported"
            ;;
        *)
            print_color $YELLOW "⚠ Python version $PYTHON_VER may not be fully tested"
            ;;
    esac
    
    # Check package managers
    if $USE_CONDA; then
        if command_exists conda; then
            print_color $GREEN "✓ Conda found"
            PACKAGE_MANAGER="conda"
        else
            print_color $RED "✗ Conda not found. Install Anaconda/Miniconda or use pip instead."
            exit 1
        fi
    else
        if command_exists pip || command_exists pip3; then
            print_color $GREEN "✓ pip found"
            PACKAGE_MANAGER="pip"
        else
            print_color $RED "✗ pip not found. Please install pip."
            exit 1
        fi
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    print_color $BLUE "Installing system dependencies..."
    
    case "$OS" in
        linux)
            if command_exists apt-get; then
                # Ubuntu/Debian
                print_color $YELLOW "Installing system packages (Ubuntu/Debian)..."
                if [ "$DRY_RUN" = true ]; then
                    echo "sudo apt-get update"
                    echo "sudo apt-get install -y gdal-bin libgdal-dev python3-dev build-essential"
                else
                    sudo apt-get update
                    sudo apt-get install -y gdal-bin libgdal-dev python3-dev build-essential
                fi
            elif command_exists yum; then
                # RHEL/CentOS
                print_color $YELLOW "Installing system packages (RHEL/CentOS)..."
                if [ "$DRY_RUN" = true ]; then
                    echo "sudo yum install -y gdal gdal-devel python3-devel gcc gcc-c++"
                else
                    sudo yum install -y gdal gdal-devel python3-devel gcc gcc-c++
                fi
            elif command_exists pacman; then
                # Arch Linux
                print_color $YELLOW "Installing system packages (Arch Linux)..."
                if [ "$DRY_RUN" = true ]; then
                    echo "sudo pacman -S --noconfirm gdal python base-devel"
                else
                    sudo pacman -S --noconfirm gdal python base-devel
                fi
            else
                print_color $YELLOW "⚠ Unknown Linux distribution. Please install GDAL manually."
            fi
            ;;
        macos)
            if command_exists brew; then
                print_color $YELLOW "Installing system packages (macOS with Homebrew)..."
                if [ "$DRY_RUN" = true ]; then
                    echo "brew install gdal"
                else
                    brew install gdal
                fi
            else
                print_color $YELLOW "⚠ Homebrew not found. Please install GDAL manually or install Homebrew."
            fi
            ;;
        windows)
            print_color $YELLOW "⚠ Windows detected. Please install GDAL manually or use conda."
            print_color $YELLOW "  Recommended: Use conda-forge channel for GDAL installation"
            ;;
    esac
}

# Function to setup virtual environment
setup_virtual_environment() {
    print_color $BLUE "Setting up virtual environment..."
    
    ENV_NAME="bathymetric_cae"
    
    if $USE_CONDA; then
        if [ "$DRY_RUN" = true ]; then
            echo "conda create -n $ENV_NAME python=$PYTHON_VER -y"
            echo "conda activate $ENV_NAME"
        else
            conda create -n $ENV_NAME python=$PYTHON_VER -y
            print_color $GREEN "✓ Conda environment '$ENV_NAME' created"
            print_color $YELLOW "To activate: conda activate $ENV_NAME"
        fi
    else
        if [ "$DRY_RUN" = true ]; then
            echo "$PYTHON_CMD -m venv $ENV_NAME"
            echo "source $ENV_NAME/bin/activate"
        else
            $PYTHON_CMD -m venv $ENV_NAME
            print_color $GREEN "✓ Virtual environment '$ENV_NAME' created"
            print_color $YELLOW "To activate: source $ENV_NAME/bin/activate"
        fi
    fi
}

# Function to install Python dependencies
install_python_dependencies() {
    print_color $BLUE "Installing Python dependencies..."
    
    # Core dependencies
    CORE_DEPS=(
        "numpy>=1.21.0"
        "matplotlib>=3.5.0"
        "seaborn>=0.11.0"
        "scikit-image>=0.19.0"
        "joblib>=1.1.0"
        "psutil>=5.8.0"
        "pillow>=8.0.0"
        "scipy>=1.7.0"
    )
    
    # TensorFlow installation
    if $INSTALL_GPU; then
        TF_PACKAGE="tensorflow[and-cuda]>=2.13.0"
    else
        TF_PACKAGE="tensorflow>=2.13.0"
    fi
    
    # GDAL installation
    if $USE_CONDA; then
        GDAL_PACKAGE="gdal"
    else
        GDAL_PACKAGE="gdal>=3.4.0"
    fi
    
    if $USE_CONDA; then
        # Conda installation
        if [ "$DRY_RUN" = true ]; then
            echo "conda install -c conda-forge ${CORE_DEPS[*]} $GDAL_PACKAGE -y"
            echo "conda install -c conda-forge $TF_PACKAGE -y"
        else
            conda install -c conda-forge "${CORE_DEPS[@]}" $GDAL_PACKAGE -y
            conda install -c conda-forge $TF_PACKAGE -y
        fi
    else
        # Pip installation
        if [ "$DRY_RUN" = true ]; then
            echo "pip install --upgrade pip"
            echo "pip install ${CORE_DEPS[*]}"
            echo "pip install $TF_PACKAGE"
            echo "pip install $GDAL_PACKAGE"
        else
            pip install --upgrade pip
            pip install "${CORE_DEPS[@]}"
            pip install $TF_PACKAGE
            
            # GDAL installation with pip can be tricky
            if ! pip install $GDAL_PACKAGE; then
                print_color $YELLOW "⚠ GDAL pip installation failed. Trying alternative approaches..."
                
                # Try system GDAL version
                GDAL_VERSION=$(gdal-config --version 2>/dev/null || echo "")
                if [ -n "$GDAL_VERSION" ]; then
                    pip install "gdal==$GDAL_VERSION" || pip install gdal --no-binary gdal
                else
                    print_color $RED "✗ GDAL installation failed. Please install manually."
                fi
            fi
        fi
    fi
    
    # Development dependencies
    if $INSTALL_DEV; then
        DEV_DEPS=(
            "pytest>=6.0.0"
            "pytest-cov>=2.12.0"
            "black>=21.0.0"
            "flake8>=3.9.0"
            "mypy>=0.910"
        )
        
        if $USE_CONDA; then
            if [ "$DRY_RUN" = true ]; then
                echo "conda install -c conda-forge ${DEV_DEPS[*]} -y"
            else
                conda install -c conda-forge "${DEV_DEPS[@]}" -y
            fi
        else
            if [ "$DRY_RUN" = true ]; then
                echo "pip install ${DEV_DEPS[*]}"
            else
                pip install "${DEV_DEPS[@]}"
            fi
        fi
    fi
    
    # Documentation dependencies
    if $INSTALL_DOCS; then
        DOCS_DEPS=(
            "sphinx>=4.0.0"
            "sphinx-rtd-theme>=1.0.0"
            "myst-parser>=0.17.0"
        )
        
        if $USE_CONDA; then
            if [ "$DRY_RUN" = true ]; then
                echo "conda install -c conda-forge ${DOCS_DEPS[*]} -y"
            else
                conda install -c conda-forge "${DOCS_DEPS[@]}" -y
            fi
        else
            if [ "$DRY_RUN" = true ]; then
                echo "pip install ${DOCS_DEPS[*]}"
            else
                pip install "${DOCS_DEPS[@]}"
            fi
        fi
    fi
}

# Function to install bathymetric CAE package
install_bathymetric_cae() {
    print_color $BLUE "Installing Bathymetric CAE package..."
    
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        # Install from source
        if [ "$DRY_RUN" = true ]; then
            echo "pip install -e ."
        else
            pip install -e .
            print_color $GREEN "✓ Bathymetric CAE installed from source"
        fi
    else
        # Install from PyPI
        if [ "$DRY_RUN" = true ]; then
            echo "pip install bathymetric-cae"
        else
            pip install bathymetric-cae
            print_color $GREEN "✓ Bathymetric CAE installed from PyPI"
        fi
    fi
}

# Function to verify installation
verify_installation() {
    print_color $BLUE "Verifying installation..."
    
    if [ "$DRY_RUN" = true ]; then
        echo "python -c \"import bathymetric_cae; print('Installation verified')\""
        echo "python -c \"from bathymetric_cae import validate_installation; print(validate_installation())\""
        return
    fi
    
    # Test basic import
    if $PYTHON_CMD -c "import bathymetric_cae; print('✓ Bathymetric CAE import successful')" 2>/dev/null; then
        print_color $GREEN "✓ Basic import test passed"
    else
        print_color $RED "✗ Basic import test failed"
        return 1
    fi
    
    # Test dependencies
    if $PYTHON_CMD -c "import tensorflow; print('✓ TensorFlow import successful')" 2>/dev/null; then
        print_color $GREEN "✓ TensorFlow test passed"
    else
        print_color $YELLOW "⚠ TensorFlow import failed"
    fi
    
    if $PYTHON_CMD -c "from osgeo import gdal; print('✓ GDAL import successful')" 2>/dev/null; then
        print_color $GREEN "✓ GDAL test passed"
    else
        print_color $YELLOW "⚠ GDAL import failed"
    fi
    
    # Run package validation
    if $PYTHON_CMD -c "from bathymetric_cae import validate_installation; result = validate_installation(); print(f'Requirements: {result[\"all_requirements_met\"]}')" 2>/dev/null; then
        print_color $GREEN "✓ Package validation completed"
    else
        print_color $YELLOW "⚠ Package validation had issues"
    fi
}

# Function to print post-installation instructions
print_post_install_instructions() {
    print_color $BLUE "Post-installation instructions:"
    echo
    
    if $USE_CONDA; then
        print_color $YELLOW "1. Activate the environment:"
        echo "   conda activate bathymetric_cae"
    else
        print_color $YELLOW "1. Activate the virtual environment:"
        echo "   source bathymetric_cae/bin/activate  # Linux/macOS"
        echo "   bathymetric_cae\\Scripts\\activate     # Windows"
    fi
    
    print_color $YELLOW "2. Verify installation:"
    echo "   python -c \"import bathymetric_cae; print('Success!')\""
    
    print_color $YELLOW "3. Run examples:"
    echo "   python examples/basic_usage.py"
    
    print_color $YELLOW "4. Get help:"
    echo "   bathymetric-cae --help"
    echo "   python -m bathymetric_cae --validate-requirements"
    
    if $INSTALL_GPU; then
        print_color $YELLOW "5. Test GPU support:"
        echo "   python -c \"import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))\""
    fi
    
    echo
    print_color $GREEN "Installation complete! 🎉"
    print_color $BLUE "For documentation, visit: https://bathymetric-cae.readthedocs.io/"
}

# Main installation function
main() {
    print_color $BLUE "Bathymetric CAE Dependency Installation"
    print_color $BLUE "======================================"
    echo
    
    # Check system requirements
    check_system_requirements
    echo
    
    # Install system dependencies
    install_system_dependencies
    echo
    
    # Setup virtual environment
    setup_virtual_environment
    echo
    
    # Install Python dependencies
    install_python_dependencies
    echo
    
    # Install the package
    install_bathymetric_cae
    echo
    
    # Verify installation
    verify_installation
    echo
    
    # Print instructions
    print_post_install_instructions
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -g|--gpu)
            INSTALL_GPU=true
            shift
            ;;
        -d|--dev)
            INSTALL_DEV=true
            shift
            ;;
        -o|--docs)
            INSTALL_DOCS=true
            shift
            ;;
        -c|--conda)
            USE_CONDA=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            print_color $YELLOW "DRY RUN MODE - Commands will be shown but not executed"
            echo
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Run main installation
main