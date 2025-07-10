#!/usr/bin/env python3
"""
Enhanced Bathymetric CAE Environment Setup Script
================================================

This script automatically sets up the complete environment for the Enhanced Bathymetric CAE system,
including all dependencies, directory structure, and configuration files.

Usage:
    python setup_environment.py [options]

Author: Enhanced Bathymetric CAE Team
Version: 2.0
Date: 2025-07-08
"""

import os
import sys
import subprocess
import platform
import json
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from dataclasses import dataclass
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SetupConfig:
    """Configuration for environment setup."""
    python_version: str = "3.9"
    install_gpu: bool = True
    install_optional: bool = True
    create_venv: bool = True
    venv_name: str = "bathymetric_cae"
    force_reinstall: bool = False
    offline_mode: bool = False
    test_installation: bool = True
    setup_sample_data: bool = True

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# Apply colored formatting
console_handler = logging.getLogger().handlers[0]
console_handler.setFormatter(ColoredFormatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))

class EnvironmentSetup:
    """Main environment setup class."""
    
    def __init__(self, config: SetupConfig):
        self.config = config
        self.system_info = self._get_system_info()
        self.project_root = Path.cwd()
        self.setup_errors = []
        
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'processor': platform.processor()
        }
    
    def run_setup(self) -> bool:
        """Run the complete environment setup."""
        logger.info("🚀 Starting Enhanced Bathymetric CAE Environment Setup")
        logger.info(f"System: {self.system_info['platform']} {self.system_info['architecture']}")
        logger.info(f"Python: {self.system_info['python_version']}")
        
        try:
            # Step 1: Check prerequisites
            logger.info("\n📋 Step 1: Checking Prerequisites")
            if not self._check_prerequisites():
                return False
            
            # Step 2: Create virtual environment
            if self.config.create_venv:
                logger.info("\n🐍 Step 2: Creating Virtual Environment")
                if not self._create_virtual_environment():
                    return False
            
            # Step 3: Install core dependencies
            logger.info("\n📦 Step 3: Installing Core Dependencies")
            if not self._install_core_dependencies():
                return False
            
            # Step 4: Install geospatial dependencies
            logger.info("\n🌍 Step 4: Installing Geospatial Dependencies")
            if not self._install_geospatial_dependencies():
                return False
            
            # Step 5: Install machine learning dependencies
            logger.info("\n🤖 Step 5: Installing Machine Learning Dependencies")
            if not self._install_ml_dependencies():
                return False
            
            # Step 6: Install optional dependencies
            if self.config.install_optional:
                logger.info("\n🔧 Step 6: Installing Optional Dependencies")
                if not self._install_optional_dependencies():
                    logger.warning("Some optional dependencies failed to install")
            
            # Step 7: Create directory structure
            logger.info("\n📁 Step 7: Creating Directory Structure")
            self._create_directory_structure()
            
            # Step 8: Create configuration files
            logger.info("\n⚙️  Step 8: Creating Configuration Files")
            self._create_configuration_files()
            
            # Step 9: Setup sample data
            if self.config.setup_sample_data:
                logger.info("\n📊 Step 9: Setting Up Sample Data")
                self._setup_sample_data()
            
            # Step 10: Test installation
            if self.config.test_installation:
                logger.info("\n🧪 Step 10: Testing Installation")
                if not self._test_installation():
                    logger.warning("Installation test failed, but setup may still be functional")
            
            # Step 11: Generate setup report
            logger.info("\n📄 Step 11: Generating Setup Report")
            self._generate_setup_report()
            
            logger.info("\n✅ Environment setup completed successfully!")
            self._print_next_steps()
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            self.setup_errors.append(str(e))
            return False
    
    def _check_prerequisites(self) -> bool:
        """Check system prerequisites."""
        logger.info("Checking system prerequisites...")
        
        # Check Python version
        python_version = tuple(map(int, platform.python_version().split('.')))
        required_version = tuple(map(int, self.config.python_version.split('.')))
        
        if python_version < required_version:
            logger.error(f"Python {self.config.python_version}+ required, found {platform.python_version()}")
            return False
        
        logger.info(f"✓ Python version: {platform.python_version()}")
        
        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            logger.info("✓ pip is available")
        except subprocess.CalledProcessError:
            logger.error("❌ pip is not available")
            return False
        
        # Check internet connection (unless offline mode)
        if not self.config.offline_mode:
            if not self._check_internet_connection():
                logger.error("❌ Internet connection required for online installation")
                return False
            logger.info("✓ Internet connection available")
        
        # Check available disk space
        disk_space = shutil.disk_usage(self.project_root)
        free_gb = disk_space.free / (1024**3)
        if free_gb < 5:
            logger.warning(f"⚠️  Low disk space: {free_gb:.1f}GB free (5GB recommended)")
        else:
            logger.info(f"✓ Disk space: {free_gb:.1f}GB available")
        
        # Check for GDAL system libraries (platform specific)
        self._check_gdal_system_libs()
        
        return True
    
    def _check_internet_connection(self) -> bool:
        """Check internet connectivity."""
        try:
            urllib.request.urlopen('https://pypi.org', timeout=10)
            return True
        except Exception:
            return False
    
    def _check_gdal_system_libs(self):
        """Check for GDAL system libraries."""
        if self.system_info['platform'] == 'Windows':
            logger.info("ℹ️  Windows detected - GDAL will be installed via pip")
        elif self.system_info['platform'] == 'Darwin':  # macOS
            if shutil.which('brew'):
                logger.info("✓ Homebrew detected - can install GDAL dependencies")
            else:
                logger.warning("⚠️  Homebrew not found - GDAL installation may fail")
        elif self.system_info['platform'] == 'Linux':
            # Check for common package managers
            if shutil.which('apt-get'):
                logger.info("✓ apt package manager detected")
            elif shutil.which('yum') or shutil.which('dnf'):
                logger.info("✓ yum/dnf package manager detected")
            else:
                logger.warning("⚠️  Package manager not detected - manual GDAL installation may be required")
    
    def _create_virtual_environment(self) -> bool:
        """Create virtual environment."""
        venv_path = self.project_root / self.config.venv_name
        
        if venv_path.exists() and not self.config.force_reinstall:
            logger.info(f"Virtual environment already exists: {venv_path}")
            response = input("Overwrite existing virtual environment? (y/N): ")
            if response.lower() != 'y':
                logger.info("Using existing virtual environment")
                return True
            else:
                shutil.rmtree(venv_path)
        
        try:
            logger.info(f"Creating virtual environment: {venv_path}")
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True)
            
            # Get activation script path
            if self.system_info['platform'] == 'Windows':
                activate_script = venv_path / "Scripts" / "activate.bat"
                pip_path = venv_path / "Scripts" / "pip.exe"
            else:
                activate_script = venv_path / "bin" / "activate"
                pip_path = venv_path / "bin" / "pip"
            
            logger.info(f"✓ Virtual environment created")
            logger.info(f"Activation script: {activate_script}")
            
            # Upgrade pip in virtual environment
            subprocess.run([
                str(pip_path), "install", "--upgrade", "pip"
            ], check=True)
            
            logger.info("✓ pip upgraded in virtual environment")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False
    
    def _get_pip_path(self) -> str:
        """Get pip path for current environment."""
        if self.config.create_venv:
            venv_path = self.project_root / self.config.venv_name
            if self.system_info['platform'] == 'Windows':
                return str(venv_path / "Scripts" / "pip.exe")
            else:
                return str(venv_path / "bin" / "pip")
        else:
            return sys.executable + " -m pip"
    
    def _install_package(self, package: str, pip_args: List[str] = None) -> bool:
        """Install a package using pip."""
        if pip_args is None:
            pip_args = []
        
        pip_path = self._get_pip_path()
        cmd = [pip_path, "install"] + pip_args + [package]
        
        try:
            logger.info(f"Installing {package}...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"Installation output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            logger.debug(f"Error output: {e.stderr}")
            self.setup_errors.append(f"Failed to install {package}: {e}")
            return False
    
    def _install_core_dependencies(self) -> bool:
        """Install core Python dependencies."""
        core_packages = [
            "numpy>=1.21.0",
            "matplotlib>=3.5.0", 
            "seaborn>=0.11.0",
            "scikit-learn>=1.1.0",
            "scipy>=1.7.0",
            "opencv-python>=4.5.0",
            "Pillow>=8.0.0",
            "psutil>=5.8.0",
            "joblib>=1.1.0",
            "pathlib",
            "argparse",
            "dataclasses; python_version<'3.7'",
            "typing-extensions"
        ]
        
        success = True
        for package in core_packages:
            if not self._install_package(package):
                success = False
        
        return success
    
    def _install_geospatial_dependencies(self) -> bool:
        """Install geospatial dependencies."""
        logger.info("Installing geospatial dependencies...")
        
        # Platform-specific GDAL installation
        if not self._install_gdal():
            return False
        
        # Other geospatial packages
        geo_packages = [
            "rasterio>=1.3.0",
            "fiona>=1.8.0", 
            "shapely>=1.8.0",
            "pyproj>=3.3.0",
            "geopandas>=0.11.0"
        ]
        
        success = True
        for package in geo_packages:
            if not self._install_package(package):
                logger.warning(f"Optional geospatial package failed: {package}")
                # Don't fail for optional packages
        
        return success
    
    def _install_gdal(self) -> bool:
        """Install GDAL with platform-specific handling."""
        logger.info("Installing GDAL...")
        
        if self.system_info['platform'] == 'Windows':
            # On Windows, try GDAL from pip first
            if self._install_package("GDAL>=3.4.0"):
                return True
            else:
                logger.warning("GDAL pip installation failed, trying alternative...")
                # Try installing from wheel
                gdal_wheels = [
                    "https://download.lfd.uci.edu/pythonlibs/archived/GDAL-3.4.3-cp39-cp39-win_amd64.whl",
                    "https://download.lfd.uci.edu/pythonlibs/archived/GDAL-3.4.3-cp310-cp310-win_amd64.whl"
                ]
                
                for wheel_url in gdal_wheels:
                    try:
                        if self._install_package(wheel_url):
                            logger.info("✓ GDAL installed from wheel")
                            return True
                    except:
                        continue
                
                logger.error("GDAL installation failed on Windows")
                return False
        
        elif self.system_info['platform'] == 'Darwin':  # macOS
            # Try installing system dependencies first
            if shutil.which('brew'):
                try:
                    logger.info("Installing GDAL system dependencies via Homebrew...")
                    subprocess.run(['brew', 'install', 'gdal'], check=True)
                    logger.info("✓ GDAL system libraries installed")
                except subprocess.CalledProcessError:
                    logger.warning("Homebrew GDAL installation failed")
            
            # Then install Python bindings
            return self._install_package("GDAL>=3.4.0")
        
        elif self.system_info['platform'] == 'Linux':
            # Install system dependencies
            self._install_linux_gdal_deps()
            
            # Install Python bindings
            return self._install_package("GDAL>=3.4.0")
        
        else:
            logger.warning(f"Unknown platform: {self.system_info['platform']}")
            return self._install_package("GDAL>=3.4.0")
    
    def _install_linux_gdal_deps(self):
        """Install GDAL system dependencies on Linux."""
        logger.info("Installing GDAL system dependencies...")
        
        # Detect package manager and install
        if shutil.which('apt-get'):
            try:
                subprocess.run([
                    'sudo', 'apt-get', 'update'
                ], check=True)
                subprocess.run([
                    'sudo', 'apt-get', 'install', '-y',
                    'gdal-bin', 'libgdal-dev', 'python3-gdal'
                ], check=True)
                logger.info("✓ GDAL dependencies installed via apt")
            except subprocess.CalledProcessError:
                logger.warning("apt-get GDAL installation failed")
        
        elif shutil.which('yum'):
            try:
                subprocess.run([
                    'sudo', 'yum', 'install', '-y',
                    'gdal', 'gdal-devel', 'python3-gdal'
                ], check=True)
                logger.info("✓ GDAL dependencies installed via yum")
            except subprocess.CalledProcessError:
                logger.warning("yum GDAL installation failed")
        
        elif shutil.which('dnf'):
            try:
                subprocess.run([
                    'sudo', 'dnf', 'install', '-y',
                    'gdal', 'gdal-devel', 'python3-gdal'
                ], check=True)
                logger.info("✓ GDAL dependencies installed via dnf")
            except subprocess.CalledProcessError:
                logger.warning("dnf GDAL installation failed")
    
    def _install_ml_dependencies(self) -> bool:
        """Install machine learning dependencies."""
        logger.info("Installing machine learning dependencies...")
        
        # Install TensorFlow
        if self.config.install_gpu:
            if not self._install_tensorflow_gpu():
                logger.warning("GPU TensorFlow installation failed, falling back to CPU")
                if not self._install_package("tensorflow>=2.13.0"):
                    return False
        else:
            if not self._install_package("tensorflow>=2.13.0"):
                return False
        
        # Install scikit-image
        if not self._install_package("scikit-image>=0.19.0"):
            return False
        
        # Install additional ML packages
        ml_packages = [
            "keras>=2.13.0",
            "tensorboard>=2.13.0"
        ]
        
        for package in ml_packages:
            self._install_package(package)  # Don't fail if these don't install
        
        return True
    
    def _install_tensorflow_gpu(self) -> bool:
        """Install TensorFlow with GPU support."""
        logger.info("Installing TensorFlow with GPU support...")
        
        # Check for CUDA
        if not self._check_cuda():
            logger.warning("CUDA not detected, installing CPU version")
            return False
        
        # Install TensorFlow GPU
        return self._install_package("tensorflow[and-cuda]>=2.13.0")
    
    def _check_cuda(self) -> bool:
        """Check for CUDA installation."""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ NVIDIA GPU detected")
                return True
        except FileNotFoundError:
            pass
        
        logger.info("NVIDIA GPU not detected")
        return False
    
    def _install_optional_dependencies(self) -> bool:
        """Install optional dependencies."""
        optional_packages = [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "streamlit>=1.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0"
        ]
        
        success = True
        for package in optional_packages:
            if not self._install_package(package):
                logger.warning(f"Optional package failed: {package}")
                # Don't fail setup for optional packages
        
        return success
    
    def _create_directory_structure(self):
        """Create project directory structure."""
        directories = [
            "data/input",
            "data/output", 
            "data/processed",
            "data/sample",
            "models",
            "logs",
            "plots",
            "reports",
            "configs",
            "scripts",
            "tests",
            "docs",
            "expert_reviews",
            "notebooks"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created directory: {directory}")
        
        # Create .gitkeep files for empty directories
        for directory in directories:
            gitkeep_path = self.project_root / directory / ".gitkeep"
            if not any((self.project_root / directory).iterdir()):
                gitkeep_path.touch()
    
    def _create_configuration_files(self):
        """Create configuration files."""
        configs_dir = self.project_root / "configs"
        
        # Create default configuration
        default_config = self._get_default_config()
        config_path = configs_dir / "default_config.json"
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"✓ Created default configuration: {config_path}")
        
        # Create development configuration
        dev_config = default_config.copy()
        dev_config.update({
            "log_level": "DEBUG",
            "epochs": 10,
            "batch_size": 4,
            "grid_size": 256,
            "enable_expert_review": False
        })
        dev_config_path = configs_dir / "development_config.json"
        with open(dev_config_path, 'w') as f:
            json.dump(dev_config, f, indent=2)
        logger.info(f"✓ Created development configuration: {dev_config_path}")
        
        # Create production configuration
        prod_config = default_config.copy()
        prod_config.update({
            "log_level": "INFO",
            "epochs": 200,
            "batch_size": 16,
            "grid_size": 1024,
            "ensemble_size": 5,
            "enable_expert_review": True
        })
        prod_config_path = configs_dir / "production_config.json"
        with open(prod_config_path, 'w') as f:
            json.dump(prod_config, f, indent=2)
        logger.info(f"✓ Created production configuration: {prod_config_path}")
        
        # Create environment file
        self._create_environment_file()
        
        # Create requirements file
        self._create_requirements_file()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration dictionary."""
        return {
            "_config_info": {
                "name": "Enhanced Bathymetric CAE Configuration",
                "version": "2.0",
                "description": "Default configuration for Enhanced Bathymetric Grid Processing using Advanced Ensemble CAE",
                "created_date": "2025-07-08"
            },
            "input_folder": str(self.project_root / "data" / "input"),
            "output_folder": str(self.project_root / "data" / "output"),
            "model_path": str(self.project_root / "models" / "cae_model_with_uncertainty.h5"),
            "log_dir": str(self.project_root / "logs" / "fit"),
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
            "gpu_memory_growth": True,
            "use_mixed_precision": True,
            "prefetch_buffer_size": "AUTOTUNE",
            "enable_adaptive_processing": True,
            "enable_expert_review": True,
            "enable_constitutional_constraints": True,
            "quality_threshold": 0.7,
            "auto_flag_threshold": 0.5,
            "ensemble_size": 3,
            "ssim_weight": 0.3,
            "roughness_weight": 0.2,
            "feature_preservation_weight": 0.3,
            "consistency_weight": 0.2
        }
    
    def _create_environment_file(self):
        """Create environment variables file."""
        env_content = f"""# Enhanced Bathymetric CAE Environment Variables
# Set these environment variables for your system

# Project paths
BATHYMETRIC_CAE_ROOT={self.project_root}
BATHYMETRIC_DATA_DIR={self.project_root}/data
BATHYMETRIC_MODELS_DIR={self.project_root}/models
BATHYMETRIC_LOGS_DIR={self.project_root}/logs

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true

# GDAL Configuration  
GDAL_DATA=/usr/share/gdal
PROJ_LIB=/usr/share/proj

# Python paths
PYTHONPATH={self.project_root}

# Virtual environment (if created)
"""
        
        if self.config.create_venv:
            venv_path = self.project_root / self.config.venv_name
            if self.system_info['platform'] == 'Windows':
                env_content += f"VIRTUAL_ENV={venv_path}\n"
                env_content += f"PATH={venv_path}/Scripts;$PATH\n"
            else:
                env_content += f"VIRTUAL_ENV={venv_path}\n"
                env_content += f"PATH={venv_path}/bin:$PATH\n"
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        logger.info(f"✓ Created environment file: {env_file}")
    
    def _create_requirements_file(self):
        """Create requirements.txt file."""
        requirements = [
            "# Enhanced Bathymetric CAE Dependencies",
            "# Core dependencies",
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.1.0",
            "scipy>=1.7.0",
            "opencv-python>=4.5.0",
            "Pillow>=8.0.0",
            "psutil>=5.8.0",
            "joblib>=1.1.0",
            "",
            "# Machine Learning",
            "tensorflow>=2.13.0",
            "scikit-image>=0.19.0",
            "keras>=2.13.0",
            "",
            "# Geospatial",
            "GDAL>=3.4.0",
            "rasterio>=1.3.0",
            "fiona>=1.8.0",
            "shapely>=1.8.0",
            "pyproj>=3.3.0",
            "geopandas>=0.11.0",
            "",
            "# Optional dependencies",
            "jupyter>=1.0.0",
            "plotly>=5.0.0",
            "streamlit>=1.0.0",
            "pytest>=7.0.0",
            ""
        ]
        
        requirements_file = self.project_root / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(requirements))
        logger.info(f"✓ Created requirements file: {requirements_file}")
    
    def _setup_sample_data(self):
        """Setup sample data for testing."""
        sample_dir = self.project_root / "data" / "sample"
        
        # Create sample bathymetric data
        logger.info("Creating sample bathymetric data...")
        self._create_sample_bathymetric_data(sample_dir)
        
        # Create sample configuration files
        sample_configs = {
            "test_config.json": {
                "epochs": 5,
                "batch_size": 2,
                "grid_size": 128,
                "enable_expert_review": False
            }
        }
        
        for config_name, config_data in sample_configs.items():
            config_path = sample_dir / config_name
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"✓ Created sample config: {config_name}")
    
    def _create_sample_bathymetric_data(self, sample_dir: Path):
        """Create synthetic sample bathymetric data."""
        try:
            import numpy as np
            from scipy import ndimage
            
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
            
            # Save as different formats
            self._save_sample_data(sample_dir, "sample_bathymetry_clean.npy", depth)
            self._save_sample_data(sample_dir, "sample_bathymetry_noisy.npy", noisy_depth)
            
            # Create a simple ASCII grid format
            ascii_content = f"""ncols         {size}
nrows         {size}
xllcorner     0.0
yllcorner     0.0
cellsize      0.02
NODATA_value  -9999
"""
            ascii_content += '\n'.join([' '.join([
                str(depth[i, j]) if not np.isnan(depth[i, j]) else "-9999"
                for j in range(size)
            ]) for i in range(size)])
            
            ascii_file = sample_dir / "sample_bathymetry.asc"
            with open(ascii_file, 'w') as f:
                f.write(ascii_content)
            
            logger.info("✓ Created sample bathymetric data files")
            
        except ImportError as e:
            logger.warning(f"Could not create sample data: {e}")
    
    def _save_sample_data(self, sample_dir: Path, filename: str, data):
        """Save sample data file."""
        try:
            import numpy as np
            filepath = sample_dir / filename
            np.save(filepath, data)
            logger.info(f"✓ Created sample file: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save {filename}: {e}")
    
    def _test_installation(self) -> bool:
        """Test the installation by importing key packages."""
        logger.info("Testing package imports...")
        
        test_packages = [
            ("numpy", "NumPy"),
            ("matplotlib", "Matplotlib"),
            ("sklearn", "Scikit-learn"),
            ("scipy", "SciPy"),
            ("cv2", "OpenCV"),
            ("tensorflow", "TensorFlow"),
            ("skimage", "Scikit-image")
        ]
        
        optional_packages = [
            ("osgeo.gdal", "GDAL"),
            ("rasterio", "Rasterio"),
            ("geopandas", "GeoPandas")
        ]
        
        success = True
        
        # Test core packages
        for package, name in test_packages:
            try:
                __import__(package)
                logger.info(f"✓ {name} import successful")
            except ImportError as e:
                logger.error(f"❌ {name} import failed: {e}")
                success = False
        
        # Test optional packages
        for package, name in optional_packages:
            try:
                __import__(package)
                logger.info(f"✓ {name} (optional) import successful")
            except ImportError as e:
                logger.warning(f"⚠️  {name} (optional) import failed: {e}")
        
        # Test TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"✓ TensorFlow GPU support: {len(gpus)} GPU(s) detected")
            else:
                logger.info("ℹ️  TensorFlow CPU-only mode")
        except Exception as e:
            logger.warning(f"TensorFlow GPU test failed: {e}")
        
        # Test basic functionality
        if success:
            success = self._test_basic_functionality()
        
        return success
    
    def _test_basic_functionality(self) -> bool:
        """Test basic functionality of key components."""
        logger.info("Testing basic functionality...")
        
        try:
            import numpy as np
            import tensorflow as tf
            from pathlib import Path
            
            # Test numpy operations
            test_array = np.random.random((100, 100))
            processed = np.mean(test_array)
            logger.info("✓ NumPy operations working")
            
            # Test TensorFlow basic operations
            x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            y = tf.matmul(x, x)
            logger.info("✓ TensorFlow operations working")
            
            # Test file I/O
            test_file = self.project_root / "test_file.tmp"
            test_file.write_text("test")
            test_file.unlink()
            logger.info("✓ File I/O operations working")
            
            return True
            
        except Exception as e:
            logger.error(f"Basic functionality test failed: {e}")
            return False
    
    def _generate_setup_report(self):
        """Generate a comprehensive setup report."""
        report = {
            "setup_date": str(datetime.datetime.now()),
            "system_info": self.system_info,
            "config": {
                "python_version": self.config.python_version,
                "install_gpu": self.config.install_gpu,
                "install_optional": self.config.install_optional,
                "create_venv": self.config.create_venv,
                "venv_name": self.config.venv_name
            },
            "project_root": str(self.project_root),
            "setup_errors": self.setup_errors,
            "status": "completed" if not self.setup_errors else "completed_with_errors"
        }
        
        # Add package versions
        report["package_versions"] = self._get_package_versions()
        
        # Save report
        report_file = self.project_root / "setup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Setup report saved: {report_file}")
    
    def _get_package_versions(self) -> Dict:
        """Get installed package versions."""
        versions = {}
        
        packages_to_check = [
            "numpy", "matplotlib", "sklearn", "scipy", "tensorflow",
            "opencv-python", "gdal", "rasterio", "geopandas"
        ]
        
        for package in packages_to_check:
            try:
                if package == "sklearn":
                    import sklearn
                    versions[package] = sklearn.__version__
                elif package == "opencv-python":
                    import cv2
                    versions[package] = cv2.__version__
                elif package == "gdal":
                    try:
                        from osgeo import gdal
                        versions[package] = gdal.VersionInfo()
                    except ImportError:
                        versions[package] = "not_installed"
                else:
                    module = __import__(package)
                    versions[package] = getattr(module, '__version__', 'unknown')
            except ImportError:
                versions[package] = "not_installed"
            except Exception as e:
                versions[package] = f"error: {e}"
        
        return versions
    
    def _print_next_steps(self):
        """Print next steps for the user."""
        logger.info("\n" + "="*60)
        logger.info("🎉 SETUP COMPLETE! Next Steps:")
        logger.info("="*60)
        
        if self.config.create_venv:
            venv_path = self.project_root / self.config.venv_name
            if self.system_info['platform'] == 'Windows':
                activate_cmd = f"{venv_path}\\Scripts\\activate.bat"
            else:
                activate_cmd = f"source {venv_path}/bin/activate"
            
            logger.info(f"1. Activate virtual environment:")
            logger.info(f"   {activate_cmd}")
        
        logger.info(f"\n2. Navigate to project directory:")
        logger.info(f"   cd {self.project_root}")
        
        logger.info(f"\n3. Test the installation:")
        logger.info(f"   python -c \"import tensorflow as tf; print('TensorFlow version:', tf.__version__)\"")
        
        logger.info(f"\n4. Run with sample data:")
        logger.info(f"   python enhanced_bathymetric_cae_v2.py --config configs/development_config.json")
        
        logger.info(f"\n5. Key directories created:")
        logger.info(f"   📁 data/input      - Place your bathymetric files here")
        logger.info(f"   📁 data/output     - Processed files will be saved here")
        logger.info(f"   📁 models          - Trained models")
        logger.info(f"   📁 configs         - Configuration files")
        logger.info(f"   📁 logs            - Training and processing logs")
        
        logger.info(f"\n6. Configuration files:")
        logger.info(f"   ⚙️  configs/default_config.json      - Default settings")
        logger.info(f"   ⚙️  configs/development_config.json  - Development/testing")
        logger.info(f"   ⚙️  configs/production_config.json   - Production settings")
        
        if self.setup_errors:
            logger.info(f"\n⚠️  Setup completed with {len(self.setup_errors)} errors:")
            for error in self.setup_errors:
                logger.info(f"   - {error}")
            logger.info(f"   Check setup_report.json for details")
        
        logger.info(f"\n📚 Documentation:")
        logger.info(f"   - README files in docs/ directory")
        logger.info(f"   - Sample data in data/sample/")
        logger.info(f"   - Example notebooks in notebooks/")
        
        logger.info("="*60)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced Bathymetric CAE Environment Setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic setup with virtual environment
  python setup_environment.py
  
  # Setup without GPU support
  python setup_environment.py --no-gpu
  
  # Setup without virtual environment
  python setup_environment.py --no-venv
  
  # Force reinstall everything
  python setup_environment.py --force-reinstall
  
  # Minimal setup (no optional packages)
  python setup_environment.py --no-optional
        """
    )
    
    parser.add_argument('--python-version', type=str, default="3.9",
                       help='Minimum Python version required')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Skip GPU-specific installations')
    parser.add_argument('--no-optional', action='store_true',
                       help='Skip optional dependencies')
    parser.add_argument('--no-venv', action='store_true',
                       help='Skip virtual environment creation')
    parser.add_argument('--venv-name', type=str, default="bathymetric_cae",
                       help='Virtual environment name')
    parser.add_argument('--force-reinstall', action='store_true',
                       help='Force reinstall all packages')
    parser.add_argument('--offline', action='store_true',
                       help='Offline installation mode')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip installation testing')
    parser.add_argument('--no-sample-data', action='store_true',
                       help='Skip sample data creation')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    return parser


def main():
    """Main setup function."""
    import datetime
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create setup configuration
    config = SetupConfig(
        python_version=args.python_version,
        install_gpu=not args.no_gpu,
        install_optional=not args.no_optional,
        create_venv=not args.no_venv,
        venv_name=args.venv_name,
        force_reinstall=args.force_reinstall,
        offline_mode=args.offline,
        test_installation=not args.no_test,
        setup_sample_data=not args.no_sample_data
    )
    
    # Run setup
    try:
        setup = EnvironmentSetup(config)
        success = setup.run_setup()
        
        if success:
            logger.info("🎉 Environment setup completed successfully!")
            return 0
        else:
            logger.error("❌ Environment setup failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())


# ============================================================================
# SETUP SCRIPT USAGE GUIDE
# ============================================================================
"""
ENHANCED BATHYMETRIC CAE ENVIRONMENT SETUP SCRIPT
=================================================

FEATURES:
- Automatic dependency installation
- Virtual environment creation
- Platform-specific optimizations
- GPU support detection and installation
- GDAL geospatial library setup
- Sample data generation
- Configuration file creation
- Installation testing and validation

BASIC USAGE:
===========

1. Basic setup (recommended):
   python setup_environment.py

2. Setup without GPU support:
   python setup_environment.py --no-gpu

3. Setup without virtual environment:
   python setup_environment.py --no-venv

4. Minimal setup (core packages only):
   python setup_environment.py --no-optional

5. Force complete reinstall:
   python setup_environment.py --force-reinstall

ADVANCED OPTIONS:
================

--python-version X.Y    Set minimum Python version (default: 3.9)
--venv-name NAME        Set virtual environment name
--offline               Offline installation mode
--no-test              Skip installation testing
--no-sample-data       Skip sample data creation
--log-level LEVEL      Set logging verbosity

PLATFORM-SPECIFIC NOTES:
========================

Windows:
- GDAL installed via pip wheels
- Virtual environment in {project}/bathymetric_cae/
- Activation: {project}/bathymetric_cae/Scripts/activate.bat

macOS:
- Homebrew used for system dependencies
- Virtual environment in {project}/bathymetric_cae/
- Activation: source {project}/bathymetric_cae/bin/activate

Linux:
- System package manager (apt/yum/dnf) for dependencies
- Virtual environment in {project}/bathymetric_cae/
- Activation: source {project}/bathymetric_cae/bin/activate

WHAT GETS INSTALLED:
===================

Core Dependencies:
- numpy, matplotlib, seaborn
- scikit-learn, scipy
- opencv-python, Pillow
- psutil, joblib

Machine Learning:
- tensorflow (with GPU support if available)
- scikit-image, keras

Geospatial:
- GDAL, rasterio, fiona
- shapely, pyproj, geopandas

Optional:
- jupyter, plotly, streamlit
- pytest, black, flake8
- sphinx documentation tools

DIRECTORY STRUCTURE CREATED:
===========================
project_root/
├── data/
│   ├── input/          # Input bathymetric files
│   ├── output/         # Processed output files
│   ├── processed/      # Intermediate processed data
│   └── sample/         # Sample test data
├── models/             # Trained model files
├── logs/               # Training and processing logs
├── plots/              # Generated visualizations
├── reports/            # Processing reports
├── configs/            # Configuration files
├── scripts/            # Utility scripts
├── tests/              # Test files
├── docs/               # Documentation
├── expert_reviews/     # Expert review database
├── notebooks/          # Jupyter notebooks
└── bathymetric_cae/    # Virtual environment (if created)

CONFIGURATION FILES CREATED:
============================
- configs/default_config.json      # Default settings
- configs/development_config.json  # Development/testing
- configs/production_config.json   # Production settings
- requirements.txt                  # Package dependencies
- .env                             # Environment variables
- setup_report.json                # Installation report

TROUBLESHOOTING:
===============

1. GDAL Installation Issues:
   - Windows: Install Visual C++ Redistributable
   - macOS: Install Xcode command line tools: xcode-select --install
   - Linux: Install system packages: sudo apt-get install gdal-bin libgdal-dev

2. TensorFlow GPU Issues:
   - Install NVIDIA drivers and CUDA toolkit
   - Use --no-gpu flag for CPU-only installation

3. Permission Issues:
   - Use sudo for system package installations (Linux/macOS)
   - Run as administrator (Windows)

4. Memory Issues:
   - Close other applications during setup
   - Use --no-optional to reduce memory usage

5. Network Issues:
   - Check internet connection
   - Use --offline flag if packages are pre-downloaded

POST-SETUP VERIFICATION:
=======================

1. Test imports:
   python -c "import tensorflow as tf, numpy as np, cv2; print('All imports successful')"

2. Test TensorFlow GPU:
   python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

3. Test GDAL:
   python -c "from osgeo import gdal; print('GDAL version:', gdal.VersionInfo())"

4. Run sample processing:
   python enhanced_bathymetric_cae_v2.py --config configs/development_config.json

SUPPORT:
========
For issues or questions:
1. Check setup_report.json for detailed error information
2. Review log files in logs/ directory
3. Consult the documentation in docs/ directory
4. Submit issues to the project repository
"""