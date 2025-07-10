#!/bin/bash

# Enhanced Bathymetric CAE Environment Setup Script (Bash Version)
# =================================================================
# 
# This script sets up the complete environment for the Enhanced Bathymetric CAE system
# for Unix-like systems (Linux, macOS, WSL)
#
# Usage: ./setup_environment.sh [options]
#
# Author: Enhanced Bathymetric CAE Team
# Version: 2.0
# Date: 2025-07-08

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration defaults
PYTHON_VERSION="3.9"
INSTALL_GPU=true
INSTALL_OPTIONAL=true
CREATE_VENV=true
VENV_NAME="bathymetric_cae"
FORCE_REINSTALL=false
TEST_INSTALLATION=true
SETUP_SAMPLE_DATA=true
PROJECT_ROOT=$(pwd)

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_fail() {
    echo -e "${RED}❌${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Enhanced Bathymetric CAE Environment Setup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -p, --python-version    Minimum Python version (default: 3.9)
    --no-gpu               Skip GPU-specific installations
    --no-optional          Skip optional dependencies
    --no-venv              Skip virtual environment creation
    --venv-name NAME       Virtual environment name (default: bathymetric_cae)
    --force-reinstall      Force reinstall all packages
    --no-test              Skip installation testing
    --no-sample-data       Skip sample data creation

Examples:
    $0                          # Basic setup
    $0 --no-gpu                 # Setup without GPU support
    $0 --no-venv                # Setup without virtual environment
    $0 --force-reinstall        # Force complete reinstall
    $0 --venv-name my_env       # Custom virtual environment name

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -p|--python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --no-gpu)
                INSTALL_GPU=false
                shift
                ;;
            --no-optional)
                INSTALL_OPTIONAL=false
                shift
                ;;
            --no-venv)
                CREATE_VENV=false
                shift
                ;;
            --venv-name)
                VENV_NAME="$2"
                shift 2
                ;;
            --force-reinstall)
                FORCE_REINSTALL=true
                shift
                ;;
            --no-test)
                TEST_INSTALLATION=false
                shift
                ;;
            --no-sample-data)
                SETUP_SAMPLE_DATA=false
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &> /dev/null; then
            PACKAGE_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            PACKAGE_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            PACKAGE_MANAGER="dnf"
        elif command -v pacman &> /dev/null; then
            PACKAGE_MANAGER="pacman"
        else
            PACKAGE_MANAGER="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PACKAGE_MANAGER="brew"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        PACKAGE_MANAGER="unknown"
    else
        OS="unknown"
        PACKAGE_MANAGER="unknown"
    fi
    
    print_info "Detected OS: $OS with package manager: $PACKAGE_MANAGER"
}

# Function to check prerequisites
check_prerequisites() {
    print_step "📋 Step 1: Checking Prerequisites"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python $PYTHON_VERSION or higher."
        exit 1
    fi
    
    CURRENT_PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_success "Python version: $CURRENT_PYTHON_VERSION"
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip not found. Please install pip."
        exit 1
    fi
    print_success "pip is available"
    
    # Check git (optional)
    if command -v git &> /dev/null; then
        print_success "Git is available"
    else
        print_warning "Git not found - version control won't be available"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
    if [[ $AVAILABLE_GB -lt 5 ]]; then
        print_warning "Low disk space: ${AVAILABLE_GB}GB available (5GB recommended)"
    else
        print_success "Disk space: ${AVAILABLE_GB}GB available"
    fi
    
    # Check internet connection
    if ping -c 1 google.com &> /dev/null; then
        print_success "Internet connection available"
    else
        print_warning "Internet connection test failed"
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    print_step "🌍 Step 2: Installing System Dependencies"
    
    case $PACKAGE_MANAGER in
        apt)
            print_info "Installing dependencies via apt..."
            sudo apt-get update
            sudo apt-get install -y \
                python3-dev \
                python3-pip \
                python3-venv \
                build-essential \
                gdal-bin \
                libgdal-dev \
                python3-gdal \
                libproj-dev \
                proj-data \
                proj-bin \
                libgeos-dev \
                libspatialite-dev \
                sqlite3 \
                libsqlite3-dev \
                libhdf5-dev \
                libnetcdf-dev \
                git
            ;;
        yum|dnf)
            print_info "Installing dependencies via $PACKAGE_MANAGER..."
            sudo $PACKAGE_MANAGER install -y \
                python3-devel \
                python3-pip \
                gcc \
                gcc-c++ \
                gdal \
                gdal-devel \
                python3-gdal \
                proj \
                proj-devel \
                geos \
                geos-devel \
                sqlite \
                sqlite-devel \
                hdf5-devel \
                netcdf-devel \
                git
            ;;
        brew)
            print_info "Installing dependencies via Homebrew..."
            if ! command -v brew &> /dev/null; then
                print_error "Homebrew not found. Please install Homebrew first:"
                print_info "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            
            brew update
            brew install \
                python@3.9 \
                gdal \
                proj \
                geos \
                spatialite-tools \
                sqlite \
                hdf5 \
                netcdf \
                git
            ;;
        *)
            print_warning "Unknown package manager. You may need to install system dependencies manually:"
            print_info "Required: python3-dev, gdal, proj, geos, sqlite, hdf5, netcdf"
            ;;
    esac
    
    print_success "System dependencies installation completed"
}

# Function to create virtual environment
create_virtual_environment() {
    if [[ "$CREATE_VENV" == false ]]; then
        print_info "Skipping virtual environment creation"
        return 0
    fi
    
    print_step "🐍 Step 3: Creating Virtual Environment"
    
    VENV_PATH="$PROJECT_ROOT/$VENV_NAME"
    
    if [[ -d "$VENV_PATH" ]] && [[ "$FORCE_REINSTALL" == false ]]; then
        print_warning "Virtual environment already exists: $VENV_PATH"
        read -p "Overwrite existing virtual environment? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Using existing virtual environment"
            return 0
        else
            rm -rf "$VENV_PATH"
        fi
    fi
    
    print_info "Creating virtual environment: $VENV_PATH"
    $PYTHON_CMD -m venv "$VENV_PATH"
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Function to get pip command
get_pip_cmd() {
    if [[ "$CREATE_VENV" == true ]]; then
        echo "$PROJECT_ROOT/$VENV_NAME/bin/pip"
    else
        echo "pip3"
    fi
}

# Function to install Python packages
install_python_packages() {
    print_step "📦 Step 4: Installing Python Packages"
    
    PIP_CMD=$(get_pip_cmd)
    
    # Activate virtual environment if it exists
    if [[ "$CREATE_VENV" == true ]]; then
        source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    fi
    
    print_info "Installing core dependencies..."
    
    # Core packages
    CORE_PACKAGES=(
        "numpy>=1.21.0"
        "matplotlib>=3.5.0"
        "seaborn>=0.11.0"
        "scikit-learn>=1.1.0"
        "scipy>=1.7.0"
        "opencv-python>=4.5.0"
        "Pillow>=8.0.0"
        "psutil>=5.8.0"
        "joblib>=1.1.0"
    )
    
    for package in "${CORE_PACKAGES[@]}"; do
        print_info "Installing $package..."
        $PIP_CMD install "$package"
    done
    
    print_success "Core packages installed"
    
    # Machine Learning packages
    print_info "Installing machine learning packages..."
    
    if [[ "$INSTALL_GPU" == true ]] && check_gpu_support; then
        print_info "Installing TensorFlow with GPU support..."
        $PIP_CMD install "tensorflow[and-cuda]>=2.13.0"
    else
        print_info "Installing TensorFlow (CPU only)..."
        $PIP_CMD install "tensorflow>=2.13.0"
    fi
    
    $PIP_CMD install "scikit-image>=0.19.0"
    $PIP_CMD install "keras>=2.13.0"
    
    print_success "Machine learning packages installed"
    
    # Geospatial packages
    print_info "Installing geospatial packages..."
    
    GEOSPATIAL_PACKAGES=(
        "GDAL>=3.4.0"
        "rasterio>=1.3.0"
        "fiona>=1.8.0"
        "shapely>=1.8.0"
        "pyproj>=3.3.0"
        "geopandas>=0.11.0"
    )
    
    for package in "${GEOSPATIAL_PACKAGES[@]}"; do
        print_info "Installing $package..."
        if ! $PIP_CMD install "$package"; then
            print_warning "Failed to install $package - continuing anyway"
        fi
    done
    
    print_success "Geospatial packages installation completed"
    
    # Optional packages
    if [[ "$INSTALL_OPTIONAL" == true ]]; then
        print_info "Installing optional packages..."
        
        OPTIONAL_PACKAGES=(
            "jupyter>=1.0.0"
            "ipykernel>=6.0.0"
            "plotly>=5.0.0"
            "streamlit>=1.0.0"
            "pytest>=7.0.0"
            "black>=22.0.0"
            "flake8>=4.0.0"
        )
        
        for package in "${OPTIONAL_PACKAGES[@]}"; do
            print_info "Installing $package..."
            if ! $PIP_CMD install "$package"; then
                print_warning "Failed to install optional package $package"
            fi
        done
        
        print_success "Optional packages installation completed"
    fi
}

# Function to check GPU support
check_gpu_support() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_success "NVIDIA GPU detected"
            return 0
        fi
    fi
    
    print_info "No NVIDIA GPU detected"
    return 1
}

# Function to create directory structure
create_directory_structure() {
    print_step "📁 Step 5: Creating Directory Structure"
    
    DIRECTORIES=(
        "data/input"
        "data/output"
        "data/processed"
        "data/sample"
        "models"
        "logs"
        "plots"
        "reports"
        "configs"
        "scripts"
        "tests"
        "docs"
        "expert_reviews"
        "notebooks"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        touch "$PROJECT_ROOT/$dir/.gitkeep"
        print_success "Created directory: $dir"
    done
}

# Function to create configuration files
create_configuration_files() {
    print_step "⚙️ Step 6: Creating Configuration Files"
    
    # Create default configuration
    cat > "$PROJECT_ROOT/configs/default_config.json" << 'EOF'
{
  "_config_info": {
    "name": "Enhanced Bathymetric CAE Configuration",
    "version": "2.0",
    "description": "Default configuration for Enhanced Bathymetric Grid Processing using Advanced Ensemble CAE",
    "created_date": "2025-07-08"
  },
  "input_folder": "./data/input",
  "output_folder": "./data/output",
  "model_path": "./models/cae_model_with_uncertainty.h5",
  "log_dir": "./logs/fit",
  "log_level": "INFO",
  "epochs": 100,
  "batch_size": 8,
  "validation_split": 0.2,
  "learning_rate": 0.001,
  "grid_size": 512,
  "base_filters": 32,
  "depth": 4,
  "dropout_rate": 0.2,
  "early_stopping_patience": 15,
  "reduce_lr_patience": 8,
  "reduce_lr_factor": 0.5,
  "min_lr": 1e-7,
  "supported_formats": [".bag", ".tif", ".tiff", ".asc", ".xyz"],
  "min_patch_size": 32,
  "max_workers": -1,
  "gpu_memory_growth": true,
  "use_mixed_precision": true,
  "prefetch_buffer_size": "AUTOTUNE",
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "enable_constitutional_constraints": true,
  "quality_threshold": 0.7,
  "auto_flag_threshold": 0.5,
  "ensemble_size": 3,
  "ssim_weight": 0.3,
  "roughness_weight": 0.2,
  "feature_preservation_weight": 0.3,
  "consistency_weight": 0.2
}
EOF
    print_success "Created default configuration"
    
    # Create development configuration
    cat > "$PROJECT_ROOT/configs/development_config.json" << 'EOF'
{
  "_config_info": {
    "name": "Enhanced Bathymetric CAE Development Configuration",
    "version": "2.0",
    "description": "Development configuration for testing and debugging"
  },
  "input_folder": "./data/sample",
  "output_folder": "./data/output",
  "model_path": "./models/dev_model.h5",
  "log_level": "DEBUG",
  "epochs": 10,
  "batch_size": 4,
  "grid_size": 256,
  "ensemble_size": 2,
  "enable_expert_review": false,
  "quality_threshold": 0.5
}
EOF
    print_success "Created development configuration"
    
    # Create requirements.txt
    cat > "$PROJECT_ROOT/requirements.txt" << 'EOF'
# Enhanced Bathymetric CAE Dependencies
# Core dependencies
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
scipy>=1.7.0
opencv-python>=4.5.0
Pillow>=8.0.0
psutil>=5.8.0
joblib>=1.1.0

# Machine Learning
tensorflow>=2.13.0
scikit-image>=0.19.0
keras>=2.13.0

# Geospatial
GDAL>=3.4.0
rasterio>=1.3.0
fiona>=1.8.0
shapely>=1.8.0
pyproj>=3.3.0
geopandas>=0.11.0

# Optional dependencies
jupyter>=1.0.0
plotly>=5.0.0
streamlit>=1.0.0
pytest>=7.0.0
EOF
    print_success "Created requirements.txt"
    
    # Create environment file
    cat > "$PROJECT_ROOT/.env" << EOF
# Enhanced Bathymetric CAE Environment Variables
BATHYMETRIC_CAE_ROOT=$PROJECT_ROOT
BATHYMETRIC_DATA_DIR=$PROJECT_ROOT/data
BATHYMETRIC_MODELS_DIR=$PROJECT_ROOT/models
BATHYMETRIC_LOGS_DIR=$PROJECT_ROOT/logs

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true

# GDAL Configuration
GDAL_DATA=/usr/share/gdal
PROJ_LIB=/usr/share/proj

# Python paths
PYTHONPATH=$PROJECT_ROOT
EOF
    
    if [[ "$CREATE_VENV" == true ]]; then
        cat >> "$PROJECT_ROOT/.env" << EOF

# Virtual Environment
VIRTUAL_ENV=$PROJECT_ROOT/$VENV_NAME
PATH=$PROJECT_ROOT/$VENV_NAME/bin:\$PATH
EOF
    fi
    
    print_success "Created environment file (.env)"
}

# Function to setup sample data
setup_sample_data() {
    if [[ "$SETUP_SAMPLE_DATA" == false ]]; then
        print_info "Skipping sample data setup"
        return 0
    fi
    
    print_step "📊 Step 7: Setting Up Sample Data"
    
    # Create a simple Python script to generate sample data
    cat > "$PROJECT_ROOT/scripts/generate_sample_data.py" << 'EOF'
#!/usr/bin/env python3
"""Generate sample bathymetric data for testing."""

import numpy as np
from pathlib import Path
import json

def create_sample_data():
    """Create synthetic bathymetric data."""
    # Create synthetic bathymetric grid
    size = 512
    x, y = np.meshgrid(np.linspace(0, 10, size), np.linspace(0, 10, size))
    
    # Create realistic bathymetry with multiple features
    depth = (
        50 * np.sin(0.5 * x) * np.cos(0.3 * y) +
        100 * np.exp(-0.1 * ((x - 5)**2 + (y - 5)**2)) +
        200 + 0.1 * (x**2 + y**2)
    )
    
    # Add noise
    noise = np.random.normal(0, 5, depth.shape)
    noisy_depth = depth + noise
    
    # Add some NaN values to simulate missing data
    mask = np.random.random(depth.shape) < 0.02
    noisy_depth[mask] = np.nan
    
    # Save as numpy arrays
    sample_dir = Path("data/sample")
    sample_dir.mkdir(exist_ok=True)
    
    np.save(sample_dir / "sample_bathymetry_clean.npy", depth)
    np.save(sample_dir / "sample_bathymetry_noisy.npy", noisy_depth)
    
    # Create ASCII grid format
    ascii_content = f"""ncols         {size}
nrows         {size}
xllcorner     0.0
yllcorner     0.0
cellsize      0.02
NODATA_value  -9999
"""
    
    for i in range(size):
        row_data = []
        for j in range(size):
            if np.isnan(depth[i, j]):
                row_data.append("-9999")
            else:
                row_data.append(f"{depth[i, j]:.2f}")
        ascii_content += " ".join(row_data) + "\n"
    
    with open(sample_dir / "sample_bathymetry.asc", "w") as f:
        f.write(ascii_content)
    
    print("Sample data created successfully!")

if __name__ == "__main__":
    create_sample_data()
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/generate_sample_data.py"
    
    # Activate virtual environment if it exists
    if [[ "$CREATE_VENV" == true ]]; then
        source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    fi
    
    # Generate sample data
    $PYTHON_CMD "$PROJECT_ROOT/scripts/generate_sample_data.py"
    
    print_success "Sample data setup completed"
}

# Function to test installation
test_installation() {
    if [[ "$TEST_INSTALLATION" == false ]]; then
        print_info "Skipping installation test"
        return 0
    fi
    
    print_step "🧪 Step 8: Testing Installation"
    
    # Activate virtual environment if it exists
    if [[ "$CREATE_VENV" == true ]]; then
        source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    fi
    
    # Test core imports
    TEST_PACKAGES=(
        "numpy"
        "matplotlib"
        "sklearn"
        "scipy"
        "cv2"
        "tensorflow"
        "skimage"
    )
    
    for package in "${TEST_PACKAGES[@]}"; do
        if $PYTHON_CMD -c "import $package" 2>/dev/null; then
            print_success "$package import successful"
        else
            print_fail "$package import failed"
            return 1
        fi
    done
    
    # Test optional imports
    OPTIONAL_PACKAGES=(
        "osgeo.gdal"
        "rasterio"
        "geopandas"
    )
    
    for package in "${OPTIONAL_PACKAGES[@]}"; do
        if $PYTHON_CMD -c "import $package" 2>/dev/null; then
            print_success "$package (optional) import successful"
        else
            print_warning "$package (optional) import failed"
        fi
    done
    
    # Test TensorFlow GPU
    GPU_TEST_OUTPUT=$($PYTHON_CMD -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null)
    if [[ "$GPU_TEST_OUTPUT" -gt 0 ]]; then
        print_success "TensorFlow GPU support: $GPU_TEST_OUTPUT GPU(s) detected"
    else
        print_info "TensorFlow running in CPU mode"
    fi
    
    print_success "Installation test completed"
}

# Function to generate setup report
generate_setup_report() {
    print_step "📄 Step 9: Generating Setup Report"
    
    REPORT_FILE="$PROJECT_ROOT/setup_report.txt"
    
    cat > "$REPORT_FILE" << EOF
Enhanced Bathymetric CAE Setup Report
=====================================

Setup Date: $(date)
System: $(uname -a)
Python Version: $($PYTHON_CMD --version 2>&1)
Project Root: $PROJECT_ROOT

Configuration:
- Virtual Environment: $CREATE_VENV
- Environment Name: $VENV_NAME
- GPU Support: $INSTALL_GPU
- Optional Packages: $INSTALL_OPTIONAL
- Sample Data: $SETUP_SAMPLE_DATA

Directory Structure:
$(find "$PROJECT_ROOT" -type d -name ".*" -prune -o -type d -print | head -20)

Package Versions:
EOF
    
    # Add package versions if available
    if [[ "$CREATE_VENV" == true ]]; then
        source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    fi
    
    for package in numpy matplotlib tensorflow scipy sklearn; do
        version=$($PYTHON_CMD -c "import $package; print($package.__version__)" 2>/dev/null || echo "not installed")
        echo "$package: $version" >> "$REPORT_FILE"
    done
    
    print_success "Setup report saved: $REPORT_FILE"
}

# Function to print next steps
print_next_steps() {
    print_step "🎉 SETUP COMPLETE! Next Steps:"
    echo "============================================"
    
    if [[ "$CREATE_VENV" == true ]]; then
        print_info "1. Activate virtual environment:"
        echo "   source $PROJECT_ROOT/$VENV_NAME/bin/activate"
    fi
    
    print_info "2. Navigate to project directory:"
    echo "   cd $PROJECT_ROOT"
    
    print_info "3. Test the installation:"
    echo "   python -c \"import tensorflow as tf; print('TensorFlow version:', tf.__version__)\""
    
    print_info "4. Run with sample data:"
    echo "   python enhanced_bathymetric_cae_v2.py --config configs/development_config.json"
    
    print_info "5. Key directories created:"
    echo "   📁 data/input      - Place your bathymetric files here"
    echo "   📁 data/output     - Processed files will be saved here"
    echo "   📁 models          - Trained models"
    echo "   📁 configs         - Configuration files"
    
    print_info "6. Configuration files:"
    echo "   ⚙️  configs/default_config.json      - Default settings"
    echo "   ⚙️  configs/development_config.json  - Development/testing"
    
    echo "============================================"
}

# Main execution function
main() {
    echo "🚀 Enhanced Bathymetric CAE Environment Setup"
    echo "=============================================="
    
    # Parse arguments
    parse_args "$@"
    
    # Detect operating system
    detect_os
    
    # Run setup steps
    check_prerequisites
    install_system_dependencies
    create_virtual_environment
    install_python_packages
    create_directory_structure
    create_configuration_files
    setup_sample_data
    test_installation
    generate_setup_report
    
    print_next_steps
    
    print_success "Environment setup completed successfully!"
}

# Run main function with all arguments
main "$@"