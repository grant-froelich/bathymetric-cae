# Quick Start Guide

Get up and running with Enhanced Bathymetric CAE Processing in 15 minutes! This guide walks you through your first processing job from installation to results.

## 🎯 What You'll Learn

By the end of this guide, you'll have:
- ✅ Installed the system
- ✅ Processed your first bathymetric dataset
- ✅ Generated quality reports and visualizations
- ✅ Understood the basic workflow

## ⏱️ Time Required: ~15 minutes

## 📋 Prerequisites

- Python 3.8+ installed
- 8GB+ RAM available
- Internet connection for downloads
- Basic command line familiarity

## 🚀 Step 1: Installation (3 minutes)

### Quick Install via pip

```bash
# Install the package
pip install bathymetric-cae

# Verify installation
bathymetric-cae --version
```

!!! success "Expected Output"
    ```
    Bathymetric CAE Processing v2.0.0
    ```

### Alternative: Development Install

If you want the latest features:

```bash
git clone https://github.com/noaa-ocs-hydrography/bathymetric-cae.git
cd bathymetric-cae
pip install -r requirements.txt
pip install -e .
```

## 📁 Step 2: Prepare Sample Data (2 minutes)

### Option A: Generate Test Data

```bash
# Create test data directory
mkdir -p ~/bathymetric_test/input

# Generate sample bathymetric files
python -c "
from tests.test_fixtures.sample_data_generator import TestDataGenerator
from pathlib import Path
TestDataGenerator.create_test_dataset(Path('~/bathymetric_test/input').expanduser(), 3)
print('✅ Sample data generated!')
"
```

### Option B: Use Your Own Data

Place your bathymetric files in a folder:
```bash
mkdir -p ~/bathymetric_test/input
# Copy your .bag, .tif, or .asc files to ~/bathymetric_test/input/
```

**Supported formats**: `.bag`, `.tif`, `.tiff`, `.asc`, `.xyz`

## ⚙️ Step 3: Basic Configuration (2 minutes)

### Create a Simple Configuration

```bash
# Generate default configuration
bathymetric-cae --save-config ~/bathymetric_test/config.json
```

### Customize for Quick Start

Edit the generated config for faster processing:

```json
{
  "input_folder": "~/bathymetric_test/input",
  "output_folder": "~/bathymetric_test/output",
  "epochs": 10,
  "batch_size": 2,
  "grid_size": 128,
  "ensemble_size": 2,
  "enable_adaptive_processing": true,
  "enable_expert_review": true,
  "enable_constitutional_constraints": true
}
```

!!! tip "Quick Start Settings"
    These settings prioritize speed over maximum quality f
