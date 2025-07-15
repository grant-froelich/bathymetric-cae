# Enhanced Bathymetric CAE Processing - Command Line Reference Guide

Complete guide to command line options with Windows Command Prompt examples, usage scenarios, and practical tips.

## 🎯 How to Run the Program

### Method 1: Direct Command (Recommended)
```cmd
bathymetric-cae [options]
```

### Method 2: Python Script
```cmd
python main.py [options]
```

### Method 3: Python Module
```cmd
python -m enhanced_bathymetric_cae [options]
```

## 📋 Complete Command Reference Table

| Option | Type | Default | Range/Options | Purpose | Quick Example |
|--------|------|---------|---------------|---------|---------------|
| **Input/Output** |
| `--input` | Path | `\\network_folder\input_bathymetric_files` | Any valid path | Input directory | `--input C:\SurveyData` |
| `--output` | Path | `\\network_folder\output_bathymetric_files` | Any valid path | Output directory | `--output C:\Results` |
| `--model` | Path | `cae_model_with_uncertainty.keras` | .keras/.h5 files | Model file path | `--model C:\Models\survey.keras` |
| `--config` | Path | None | .json files | Load config file | `--config settings.json` |
| `--save-config` | Path | None | .json files | Save config file | `--save-config my_settings.json` |
| **Training Parameters** |
| `--epochs` | Integer | 100 | 10-500 | Training iterations | `--epochs 200` |
| `--batch-size` | Integer | 8 | 1-32 | Samples per batch | `--batch-size 16` |
| `--learning-rate` | Float | 0.001 | 0.0001-0.01 | AI learning speed | `--learning-rate 0.0005` |
| `--validation-split` | Float | 0.2 | 0.1-0.3 | Validation ratio | `--validation-split 0.25` |
| **Model Architecture** |
| `--grid-size` | Integer | 512 | 128,256,512,1024,2048 | Processing resolution | `--grid-size 1024` |
| `--base-filters` | Integer | 32 | 16-64 | Model complexity | `--base-filters 48` |
| `--depth` | Integer | 4 | 2-6 | Model layers | `--depth 5` |
| `--dropout-rate` | Float | 0.2 | 0.0-0.5 | Regularization | `--dropout-rate 0.1` |
| `--ensemble-size` | Integer | 3 | 1-10 | Number of models | `--ensemble-size 5` |
| **Enhanced Features** |
| `--enable-adaptive` | Flag | False | True/False | Smart seafloor processing | `--enable-adaptive` |
| `--enable-expert-review` | Flag | False | True/False | Human quality control | `--enable-expert-review` |
| `--enable-constitutional` | Flag | False | True/False | AI safety constraints | `--enable-constitutional` |
| `--quality-threshold` | Float | 0.7 | 0.0-1.0 | Review trigger level | `--quality-threshold 0.85` |
| **Processing Options** |
| `--max-workers` | Integer | -1 | 1-32, -1=auto | Parallel workers | `--max-workers 8` |
| `--log-level` | Choice | INFO | DEBUG,INFO,WARNING,ERROR | Logging detail | `--log-level DEBUG` |
| `--no-gpu` | Flag | False | True/False | Disable GPU | `--no-gpu` |
| **Quality Weights** (Must sum to 1.0) |
| `--ssim-weight` | Float | 0.3 | 0.0-1.0 | Structural similarity | `--ssim-weight 0.4` |
| `--roughness-weight` | Float | 0.2 | 0.0-1.0 | Surface smoothness | `--roughness-weight 0.15` |
| `--feature-weight` | Float | 0.3 | 0.0-1.0 | Feature preservation | `--feature-weight 0.35` |
| `--consistency-weight` | Float | 0.2 | 0.0-1.0 | Depth consistency | `--consistency-weight 0.1` |

## 📋 Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Input/Output Commands](#inputoutput-commands)
- [Training Parameters](#training-parameters)
- [Model Architecture](#model-architecture)
- [Enhanced Features](#enhanced-features)
- [Processing Options](#processing-options)
- [Quality Control](#quality-control)
- [Common Usage Patterns](#common-usage-patterns)
- [Troubleshooting Commands](#troubleshooting-commands)

## 🚀 Quick Start Examples

### Basic Usage (Windows CMD)

**Method 1: Direct Command (Recommended)**
```cmd
REM Most basic command - uses all defaults
bathymetric-cae

REM Process specific folders
bathymetric-cae --input C:\BathymetryData\Survey2024 --output C:\ProcessedData\Survey2024

REM Load saved configuration
bathymetric-cae --config production_settings.json
```

**Method 2: Python Script**
```cmd
REM Most basic command using Python script
python main.py

REM Process specific folders using Python
python main.py --input C:\BathymetryData\Survey2024 --output C:\ProcessedData\Survey2024

REM Load configuration using Python
python main.py --config production_settings.json
```

**Method 3: Python Module**
```cmd
REM Run as Python module
python -m enhanced_bathymetric_cae

REM With parameters
python -m enhanced_bathymetric_cae --input C:\Data --output C:\Results
```

### Common Workflows

**Direct Command Examples:**
```cmd
REM Development mode (fast testing)
bathymetric-cae --epochs 25 --ensemble-size 1 --grid-size 256

REM Production mode (high quality)
bathymetric-cae --epochs 200 --ensemble-size 5 --enable-adaptive --enable-expert-review

REM Quick test run
bathymetric-cae --epochs 5 --batch-size 1 --grid-size 128
```

**Python Script Examples:**
```cmd
REM Development mode using Python script
python main.py --epochs 25 --ensemble-size 1 --grid-size 256

REM Production mode using Python script
python main.py --epochs 200 --ensemble-size 5 --enable-adaptive --enable-expert-review

REM Quick test run using Python script
python main.py --epochs 5 --batch-size 1 --grid-size 128
```

---

## 📁 Input/Output Commands

### `--input` / `--input-folder`
**Purpose**: Specify folder containing bathymetric files to process  
**Default**: `\\network_folder\input_bathymetric_files`  
**File Types**: `.bag`, `.tif`, `.tiff`, `.asc`, `.xyz`

**When to Use:**
- Processing data from a specific survey
- Working with files in a custom location
- Batch processing multiple files

**Windows Examples:**
```cmd
REM Direct command - Local drive
bathymetric-cae --input C:\Surveys\January2024

REM Python script - Network drive
python main.py --input "\\SurveyServer\Data\Multibeam\2024\Q1"

REM Direct command - Current directory
bathymetric-cae --input .

REM Python script - Relative path
python main.py --input .\InputData
```

**How to Modify:**
- Use **quotes** for paths with spaces: `"C:\Program Files\Survey Data"`
- Use **forward slashes** or **double backslashes**: `C:/Data` or `C:\\Data`
- Use **UNC paths** for network drives: `\\server\share\folder`

### `--output` / `--output-folder`
**Purpose**: Specify where processed files will be saved  
**Default**: `\\network_folder\output_bathymetric_files`

**When to Use:**
- Organizing results by date/survey
- Saving to different storage locations
- Keeping processed data separate from raw data

**Windows Examples:**
```cmd
REM Direct command - Standard output folder
bathymetric-cae --output C:\ProcessedSurveys\January2024

REM Python script - Timestamped output
python main.py --output "C:\Results\Survey_%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%"

REM Direct command - Network storage
bathymetric-cae --output "\\StorageServer\ProcessedData\Bathymetry"

REM Python script - Desktop folder
python main.py --output "%USERPROFILE%\Desktop\BathymetryResults"
```

### `--model` / `--model-path`
**Purpose**: Path to save/load trained AI model  
**Default**: `cae_model_with_uncertainty.keras`  
**Formats**: `.keras` (modern), `.h5` (legacy)

**When to Use:**
- Reusing trained models on similar data
- Saving models for specific survey types
- Loading pre-trained models

**Windows Examples:**
```cmd
REM Direct command - Modern format (recommended)
bathymetric-cae --model C:\Models\CoastalSurvey_2024.keras

REM Python script - Legacy format (auto-converted)
python main.py --model C:\Models\DeepOcean_trained.h5

REM Direct command - Timestamped model
bathymetric-cae --model "C:\Models\Model_%DATE%.keras"

REM Python script - Network model storage
python main.py --model "\\ModelServer\TrainedModels\bathymetric_ensemble.keras"
```

---

## ⚙️ Configuration Commands

### `--config`
**Purpose**: Load settings from JSON configuration file  
**Default**: None (uses built-in defaults)

**When to Use:**
- Standardizing processing across teams
- Saving complex parameter combinations
- Switching between processing modes

**Windows Examples:**
```cmd
REM Direct command - Load production settings
bathymetric-cae --config C:\Configs\production.json

REM Python script - Load survey-specific settings
python main.py --config "C:\Survey Configs\shallow_water.json"

REM Direct command - Combine config with overrides
bathymetric-cae --config base_settings.json --epochs 300
```

### `--save-config`
**Purpose**: Save current settings to JSON file  
**Default**: None

**When to Use:**
- Creating reusable configurations
- Documenting successful parameter combinations
- Sharing settings with team members

**Windows Examples:**
```cmd
REM Direct command - Save current settings
bathymetric-cae --save-config my_settings.json

REM Python script - Save production configuration
python main.py --epochs 200 --ensemble-size 5 --save-config production.json

REM Direct command - Save and run simultaneously
bathymetric-cae --save-config settings.json --input C:\TestData
```

---

## 🏋️ Training Parameters

### `--epochs`
**Purpose**: Number of AI training iterations  
**Default**: 100  
**Range**: 10-500

**When to Use:**
- **10-50**: Development and testing (faster results)
- **100-150**: Standard processing (balanced quality/speed)
- **200-300**: High-quality production (slower, better results)
- **300+**: Research-grade processing

**Windows Examples:**
```cmd
REM Direct command - Quick test
bathymetric-cae --epochs 25 --input C:\TestData

REM Python script - Standard processing
python main.py --epochs 100 --input C:\SurveyData

REM Direct command - High quality
bathymetric-cae --epochs 250 --input C:\ImportantSurvey

REM Python script - Research quality
python main.py --epochs 400 --input C:\ResearchData
```

**How to Modify Based on Usage:**
- **Fast testing**: Use 10-25 epochs
- **Quality concerns**: Increase to 200-300 epochs
- **Time constraints**: Decrease to 50-75 epochs

### `--batch-size`
**Purpose**: Number of data samples processed simultaneously  
**Default**: 8  
**Range**: 1-32

**When to Use:**
- **1-2**: Systems with < 8GB RAM
- **4-8**: Standard systems (8-16GB RAM)
- **16-32**: High-memory systems (> 16GB RAM + GPU)

**Windows Examples:**
```cmd
REM Direct command - Low memory system
bathymetric-cae --batch-size 2 --input C:\Data

REM Python script - Standard system
python main.py --batch-size 8 --input C:\Data

REM Direct command - High-performance system
bathymetric-cae --batch-size 16 --input C:\Data

REM Python script - GPU-optimized
python main.py --batch-size 32 --input C:\Data
```

**Memory Guidelines:**
- Monitor Task Manager while processing
- Reduce if getting "Out of Memory" errors
- Increase on powerful systems for faster processing

### `--learning-rate`
**Purpose**: AI model learning speed  
**Default**: 0.001  
**Range**: 0.0001-0.01

**When to Use:**
- **0.0001-0.0005**: Conservative learning (more stable)
- **0.001**: Standard learning (default)
- **0.002-0.005**: Fast learning (may be unstable)

**Windows Examples:**
```cmd
REM Direct command - Conservative (stable but slower)
bathymetric-cae --learning-rate 0.0005 --input C:\CriticalData

REM Python script - Standard
python main.py --learning-rate 0.001 --input C:\StandardData

REM Direct command - Aggressive (faster but riskier)
bathymetric-cae --learning-rate 0.003 --input C:\TestData
```

---

## 🏗️ Model Architecture

### `--grid-size`
**Purpose**: Processing resolution in pixels  
**Default**: 512  
**Options**: 128, 256, 512, 1024, 2048

**When to Use:**
- **128-256**: Fast development/testing, lower detail
- **512**: Balanced performance and quality (recommended)
- **1024**: High quality, slower processing
- **2048**: Maximum quality, very slow

**Windows Examples:**
```cmd
REM Direct command - Fast processing
bathymetric-cae --grid-size 256 --input C:\QuickTest

REM Python script - Standard quality
python main.py --grid-size 512 --input C:\StandardSurvey

REM Direct command - High resolution
bathymetric-cae --grid-size 1024 --input C:\DetailedSurvey

REM Python script - Maximum quality
python main.py --grid-size 2048 --input C:\ResearchSurvey
```

**Performance vs Quality Trade-offs:**
- Larger grid = Better quality but **much** slower
- Choose based on time available and quality requirements

### `--ensemble-size`
**Purpose**: Number of AI models working together  
**Default**: 3  
**Range**: 1-10

**When to Use:**
- **1**: Fastest processing, lowest accuracy
- **3**: Good balance (recommended)
- **5**: High accuracy, slower processing
- **7+**: Maximum accuracy, very slow

**Windows Examples:**
```cmd
REM Direct command - Single model (fastest)
bathymetric-cae --ensemble-size 1 --input C:\QuickJob

REM Python script - Standard ensemble
python main.py --ensemble-size 3 --input C:\NormalJob

REM Direct command - High accuracy
bathymetric-cae --ensemble-size 5 --input C:\QualityJob

REM Python script - Research grade
python main.py --ensemble-size 7 --input C:\ResearchJob
```

**Time vs Quality:**
- Each additional model roughly doubles processing time
- Benefits diminish after 5-7 models

---

## ✨ Enhanced Features

### `--enable-adaptive`
**Purpose**: Enable automatic seafloor type detection and smart processing  
**Default**: Disabled

**When to Use:**
- Processing diverse seafloor types
- Unknown or mixed survey areas
- Want automatic optimization

**What It Does:**
- Detects shallow coastal, deep ocean, seamount, etc.
- Adjusts processing parameters automatically
- Optimizes for each seafloor environment

**Windows Examples:**
```cmd
REM Direct command - Enable smart processing
bathymetric-cae --enable-adaptive --input C:\MixedSurvey

REM Python script - Combine with other features
python main.py --enable-adaptive --enable-expert-review --input C:\ComplexSurvey
```

### `--enable-expert-review`
**Purpose**: Enable human expert quality control system  
**Default**: Disabled

**When to Use:**
- Critical surveys requiring validation
- Quality control workflows
- Learning about data quality patterns

**What It Does:**
- Flags questionable results for human review
- Maintains database of expert assessments
- Generates review reports

**Windows Examples:**
```cmd
REM Direct command - Enable expert review
bathymetric-cae --enable-expert-review --input C:\CriticalSurvey

REM Python script - Set custom quality threshold
python main.py --enable-expert-review --quality-threshold 0.85 --input C:\HighStandardSurvey
```

### `--enable-constitutional`
**Purpose**: Enable AI safety constraints for data integrity  
**Default**: Disabled

**When to Use:**
- Preventing unrealistic results
- Ensuring physical plausibility
- Safety-critical applications

**What It Does:**
- Prevents physically impossible depth changes
- Preserves important seafloor features
- Ensures realistic gradient transitions

**Windows Examples:**
```cmd
REM Direct command - Enable safety constraints
bathymetric-cae --enable-constitutional --input C:\SafetyCriticalSurvey

REM Python script - Full feature set
python main.py --enable-adaptive --enable-expert-review --enable-constitutional --input C:\ProductionSurvey
```

---

## ⚡ Processing Options

### `--max-workers`
**Purpose**: Number of parallel processing workers  
**Default**: -1 (auto-detect CPU cores)  
**Range**: 1-32

**When to Use:**
- **1**: Memory-constrained systems
- **4-8**: Standard workstations
- **-1**: Let system decide (recommended)

**Windows Examples:**
```cmd
REM Direct command - Auto-detect cores
bathymetric-cae --max-workers -1 --input C:\Data

REM Python script - Limit for other work
python main.py --max-workers 4 --input C:\Data

REM Direct command - Single-threaded (memory limited)
bathymetric-cae --max-workers 1 --input C:\Data
```

### `--log-level`
**Purpose**: Control amount of information displayed  
**Default**: INFO  
**Options**: DEBUG, INFO, WARNING, ERROR

**When to Use:**
- **DEBUG**: Troubleshooting issues
- **INFO**: Normal operation (recommended)
- **WARNING**: Minimal output for production
- **ERROR**: Only show problems

**Windows Examples:**
```cmd
REM Direct command - Detailed troubleshooting
bathymetric-cae --log-level DEBUG --input C:\ProblemData

REM Python script - Normal operation
python main.py --log-level INFO --input C:\StandardData

REM Direct command - Quiet operation
bathymetric-cae --log-level WARNING --input C:\BatchData
```

### `--no-gpu`
**Purpose**: Force CPU-only processing (disable GPU)  
**Default**: GPU enabled if available

**When to Use:**
- GPU driver problems
- Memory limitations
- CPU-only systems
- Debugging GPU issues

**Windows Examples:**
```cmd
REM Direct command - Force CPU processing
bathymetric-cae --no-gpu --input C:\Data

REM Python script - CPU with more workers
python main.py --no-gpu --max-workers 8 --input C:\Data
```

---

## 📊 Quality Control Options

### `--quality-threshold`
**Purpose**: Set minimum quality score for expert review flagging  
**Default**: 0.7  
**Range**: 0.0-1.0

**When to Use:**
- **0.5-0.6**: Development/testing (relaxed standards)
- **0.7-0.8**: Standard production
- **0.85-0.9**: High-quality surveys
- **0.9+**: Research applications

**Windows Examples:**
```cmd
REM Direct command - Relaxed standards
bathymetric-cae --quality-threshold 0.6 --input C:\TestData

REM Python script - Standard production
python main.py --quality-threshold 0.75 --input C:\ProductionData

REM Direct command - High standards
bathymetric-cae --quality-threshold 0.9 --input C:\CriticalData
```

### Quality Metric Weights
**Purpose**: Control importance of different quality aspects in the composite quality score  
**Requirement**: All weights must sum to 1.0

#### Understanding Each Quality Metric

**🎯 SSIM Weight (Structural Similarity)**
- **What it measures**: How well the processed data maintains the overall structure of the original
- **Range**: 0.0-1.0 (higher = more importance to preserving structure)
- **Default**: 0.3 (30% of total quality score)
- **When to increase**: When preserving overall bathymetric patterns is critical
- **When to decrease**: When fine details matter more than overall structure

**🗻 Feature Weight (Feature Preservation)**
- **What it measures**: How well important seafloor features (ridges, valleys, peaks) are retained
- **Range**: 0.0-1.0 (higher = more importance to keeping seafloor features)
- **Default**: 0.3 (30% of total quality score)
- **When to increase**: Mapping underwater mountains, canyons, or complex terrain
- **When to decrease**: Processing flat areas where features are less important

**📏 Consistency Weight (Depth Consistency)**
- **What it measures**: How smooth and realistic depth transitions are
- **Range**: 0.0-1.0 (higher = more importance to realistic depth changes)
- **Default**: 0.2 (20% of total quality score)
- **When to increase**: Navigation safety applications requiring smooth depth transitions
- **When to decrease**: Research applications where small-scale variations matter

**🌊 Roughness Weight (Surface Smoothness)**
- **What it measures**: How much noise and unrealistic variation is removed
- **Range**: 0.0-1.0 (higher = more importance to noise reduction)
- **Default**: 0.2 (20% of total quality score)
- **When to increase**: Very noisy data or when smooth surfaces are expected
- **When to decrease**: Naturally rough terrain where some "noise" is actually real features

#### Practical Tuning Examples

**Scenario 1: Underwater Mountain Mapping (Seamounts)**
```cmd
REM Emphasize keeping dramatic terrain features
bathymetric-cae --feature-weight 0.5 --ssim-weight 0.3 --consistency-weight 0.1 --roughness-weight 0.1 --input C:\SeamountData

REM Why these weights?
REM feature-weight 0.5    = Most important: keep peaks, ridges, steep slopes
REM ssim-weight 0.3       = Important: maintain overall mountain structure  
REM consistency-weight 0.1 = Less important: steep changes are expected
REM roughness-weight 0.1   = Less important: rough terrain is natural
```

**Scenario 2: Harbor/Port Surveying (Navigation Safety)**
```cmd
REM Emphasize preserving man-made hazards while maintaining smooth natural areas
bathymetric-cae --feature-weight 0.4 --consistency-weight 0.3 --ssim-weight 0.2 --roughness-weight 0.1 --input C:\HarborSurvey

REM Why these weights?
REM feature-weight 0.4     = Most important: preserve piers, docks, debris, sunken vessels
REM consistency-weight 0.3 = Important: smooth depth transitions in open water areas
REM ssim-weight 0.2        = Important: maintain overall harbor structure
REM roughness-weight 0.1   = Low: avoid smoothing away small but critical man-made objects

REM Alternative for very cluttered ports with many structures
bathymetric-cae --feature-weight 0.5 --ssim-weight 0.3 --consistency-weight 0.15 --roughness-weight 0.05 --input C:\BusyPort
```

**Scenario 3: Open Ocean Navigation Routes**
```cmd
REM Emphasize smooth, consistent depths for safe navigation in open water
bathymetric-cae --consistency-weight 0.4 --roughness-weight 0.3 --ssim-weight 0.2 --feature-weight 0.1 --input C:\ShippingLanes

REM Why these weights?
REM consistency-weight 0.4 = Most important: smooth depth transitions for ship safety
REM roughness-weight 0.3   = Important: remove noise that could mislead navigators
REM ssim-weight 0.2        = Important: keep overall depth patterns
REM feature-weight 0.1     = Less important: open ocean has fewer critical small features
```

**Scenario 4: Very Noisy Sonar Data**
```cmd
REM Focus on cleaning up noise while keeping structure
bathymetric-cae --roughness-weight 0.4 --ssim-weight 0.3 --consistency-weight 0.2 --feature-weight 0.1 --input C:\NoisyData

REM Why these weights?
REM roughness-weight 0.4   = Most important: remove excessive noise
REM ssim-weight 0.3        = Important: maintain overall structural patterns
REM consistency-weight 0.2 = Important: ensure realistic depth variations
REM feature-weight 0.1     = Less important: may be hard to distinguish features from noise
```

**Scenario 5: Flat Abyssal Plain Mapping**
```cmd
REM Emphasize smoothness and consistency over features
bathymetric-cae --roughness-weight 0.35 --consistency-weight 0.35 --ssim-weight 0.25 --feature-weight 0.05 --input C:\AbyssalPlain

REM Why these weights?
REM roughness-weight 0.35   = High: remove noise from flat areas
REM consistency-weight 0.35 = High: ensure smooth, realistic gradual slopes
REM ssim-weight 0.25        = Moderate: maintain overall depth patterns
REM feature-weight 0.05     = Low: few significant features expected
```

**Scenario 6: Scientific Research (Preserve All Details)**
```cmd
REM Balanced approach preserving both structure and features
bathymetric-cae --ssim-weight 0.35 --feature-weight 0.35 --consistency-weight 0.2 --roughness-weight 0.1 --input C:\ResearchData

REM Why these weights?
REM ssim-weight 0.35        = High: preserve overall scientific accuracy
REM feature-weight 0.35     = High: keep all potential geological features
REM consistency-weight 0.2  = Moderate: realistic but allow natural variation
REM roughness-weight 0.1    = Low: minimize smoothing that might remove real data
```

**Scenario 7: Coastal Mapping (Complex Environment)**
```cmd
REM Balanced approach for mixed shallow/deep coastal areas
bathymetric-cae --feature-weight 0.4 --ssim-weight 0.3 --consistency-weight 0.2 --roughness-weight 0.1 --input C:\CoastalSurvey

REM Why these weights?
REM feature-weight 0.4      = High: preserve channels, shoals, reefs
REM ssim-weight 0.3         = Important: maintain coastal structure patterns
REM consistency-weight 0.2  = Moderate: allow for natural coastal complexity
REM roughness-weight 0.1    = Low: coastal areas naturally have variations
```

#### Quick Tuning Guidelines

| If you want... | Increase this weight | Decrease this weight |
|----------------|---------------------|---------------------|
| **Smoother results** | `--roughness-weight` | `--feature-weight` |
| **Keep sharp features** | `--feature-weight` | `--roughness-weight` |
| **Better for navigation** | `--feature-weight` + `--consistency-weight` | `--roughness-weight` |
| **Preserve man-made structures** | `--feature-weight` | `--roughness-weight` |
| **Scientific accuracy** | `--ssim-weight` + `--feature-weight` | `--roughness-weight` |
| **Remove more noise** | `--roughness-weight` | `--feature-weight` |
| **Preserve fine detail** | `--feature-weight` | `--roughness-weight` |

#### Testing Your Custom Weights

```cmd
REM Test different weight combinations on a small sample
bathymetric-cae --input C:\TestSample --output C:\Test1 --feature-weight 0.5 --ssim-weight 0.3 --consistency-weight 0.1 --roughness-weight 0.1 --epochs 50

REM Compare with different weights
bathymetric-cae --input C:\TestSample --output C:\Test2 --feature-weight 0.2 --ssim-weight 0.3 --consistency-weight 0.3 --roughness-weight 0.2 --epochs 50

REM Review the quality scores in the output JSON files to see which works better
```

#### Weight Validation

**✅ Valid Examples:**
```cmd
REM All weights sum to 1.0
--ssim-weight 0.25 --roughness-weight 0.25 --feature-weight 0.25 --consistency-weight 0.25

--ssim-weight 0.4 --roughness-weight 0.3 --feature-weight 0.2 --consistency-weight 0.1
```

**❌ Invalid Examples:**
```cmd
REM These will cause errors (weights sum to 1.2)
--ssim-weight 0.3 --roughness-weight 0.3 --feature-weight 0.3 --consistency-weight 0.3

REM These will cause errors (weights sum to 0.8)  
--ssim-weight 0.2 --roughness-weight 0.2 --feature-weight 0.2 --consistency-weight 0.2
```

---

## 🎯 Common Usage Patterns

### Development and Testing

**Direct Command Examples:**
```cmd
REM Quick development test
bathymetric-cae --epochs 10 --batch-size 1 --grid-size 128 --ensemble-size 1 --input C:\DevData

REM Feature testing
bathymetric-cae --epochs 25 --enable-adaptive --log-level DEBUG --input C:\TestFeatures

REM Performance testing
bathymetric-cae --epochs 50 --batch-size 8 --max-workers 4 --input C:\PerfTest
```

**Python Script Examples:**
```cmd
REM Quick development test using Python
python main.py --epochs 10 --batch-size 1 --grid-size 128 --ensemble-size 1 --input C:\DevData

REM Feature testing using Python
python main.py --epochs 25 --enable-adaptive --log-level DEBUG --input C:\TestFeatures

REM Performance testing using Python
python main.py --epochs 50 --batch-size 8 --max-workers 4 --input C:\PerfTest
```

### Production Processing

**Direct Command Examples:**
```cmd
REM Standard production
bathymetric-cae --epochs 150 --ensemble-size 3 --enable-adaptive --enable-expert-review --input "C:\Production Data" --output "C:\Production Results"

REM High-quality production
bathymetric-cae --epochs 200 --ensemble-size 5 --grid-size 1024 --quality-threshold 0.85 --enable-adaptive --enable-expert-review --enable-constitutional --input "C:\Critical Survey" --output "C:\Critical Results"

REM Batch production using saved settings
bathymetric-cae --config production.json --input "C:\Batch Data" --output "C:\Batch Results"
```

**Python Script Examples:**
```cmd
REM Standard production using Python
python main.py --epochs 150 --ensemble-size 3 --enable-adaptive --enable-expert-review --input "C:\Production Data" --output "C:\Production Results"

REM High-quality production using Python
python main.py --epochs 200 --ensemble-size 5 --grid-size 1024 --quality-threshold 0.85 --enable-adaptive --enable-expert-review --enable-constitutional --input "C:\Critical Survey" --output "C:\Critical Results"

REM Batch production using saved settings with Python
python main.py --config production.json --input "C:\Batch Data" --output "C:\Batch Results"
```

### Specialized Surveys

**Direct Command Examples:**
```cmd
REM Coastal mapping (high detail required)
bathymetric-cae --grid-size 1024 --feature-weight 0.4 --enable-adaptive --quality-threshold 0.8 --input "C:\Coastal Survey" --output "C:\Coastal Results"

REM Deep ocean survey (noise reduction focus)
bathymetric-cae --roughness-weight 0.4 --ensemble-size 5 --enable-constitutional --input "C:\Deep Ocean" --output "C:\Deep Ocean Clean"

REM Research quality processing
bathymetric-cae --epochs 300 --ensemble-size 7 --grid-size 1024 --quality-threshold 0.9 --enable-adaptive --enable-expert-review --enable-constitutional --input "C:\Research Data" --output "C:\Research Results"
```

**Python Script Examples:**
```cmd
REM Coastal mapping using Python (high detail required)
python main.py --grid-size 1024 --feature-weight 0.4 --enable-adaptive --quality-threshold 0.8 --input "C:\Coastal Survey" --output "C:\Coastal Results"

REM Deep ocean survey using Python (noise reduction focus)
python main.py --roughness-weight 0.4 --ensemble-size 5 --enable-constitutional --input "C:\Deep Ocean" --output "C:\Deep Ocean Clean"

REM Research quality processing using Python
python main.py --epochs 300 --ensemble-size 7 --grid-size 1024 --quality-threshold 0.9 --enable-adaptive --enable-expert-review --enable-constitutional --input "C:\Research Data" --output "C:\Research Results"
```

### Resource-Constrained Processing

**Direct Command Examples:**
```cmd
REM Low memory (< 8GB RAM)
bathymetric-cae --batch-size 1 --grid-size 256 --ensemble-size 1 --max-workers 1 --input C:\Data

REM CPU-only processing
bathymetric-cae --no-gpu --max-workers 8 --batch-size 4 --input C:\Data

REM Network storage with minimal logging
bathymetric-cae --input "\\Server\Input" --output "\\Server\Output" --log-level WARNING
```

**Python Script Examples:**
```cmd
REM Low memory using Python (< 8GB RAM)
python main.py --batch-size 1 --grid-size 256 --ensemble-size 1 --max-workers 1 --input C:\Data

REM CPU-only processing using Python
python main.py --no-gpu --max-workers 8 --batch-size 4 --input C:\Data

REM Network storage with minimal logging using Python
python main.py --input "\\Server\Input" --output "\\Server\Output" --log-level WARNING
```

---

## 🔧 Troubleshooting Commands

### Memory Issues

**Direct Command Examples:**
```cmd
REM Minimal memory usage
bathymetric-cae --batch-size 1 --grid-size 256 --ensemble-size 1 --max-workers 1 --no-gpu --input C:\Data

REM Check memory usage during processing
bathymetric-cae --log-level DEBUG --batch-size 2 --epochs 5 --input C:\SmallTest
```

**Python Script Examples:**
```cmd
REM Minimal memory usage using Python
python main.py --batch-size 1 --grid-size 256 --ensemble-size 1 --max-workers 1 --no-gpu --input C:\Data

REM Check memory usage during processing using Python
python main.py --log-level DEBUG --batch-size 2 --epochs 5 --input C:\SmallTest
```

### Performance Issues

**Direct Command Examples:**
```cmd
REM Profile performance
bathymetric-cae --epochs 10 --log-level DEBUG --max-workers 1 --input C:\PerfTest

REM GPU debugging
bathymetric-cae --log-level DEBUG --batch-size 1 --epochs 5 --input C:\GPUTest

REM Test without GPU
bathymetric-cae --no-gpu --log-level DEBUG --input C:\CPUTest
```

**Python Script Examples:**
```cmd
REM Profile performance using Python
python main.py --epochs 10 --log-level DEBUG --max-workers 1 --input C:\PerfTest

REM GPU debugging using Python
python main.py --log-level DEBUG --batch-size 1 --epochs 5 --input C:\GPUTest

REM Test without GPU using Python
python main.py --no-gpu --log-level DEBUG --input C:\CPUTest
```

### Quality Issues

**Direct Command Examples:**
```cmd
REM Debug quality problems
bathymetric-cae --epochs 100 --enable-adaptive --enable-constitutional --log-level DEBUG --input C:\QualityTest

REM Strict quality checking
bathymetric-cae --quality-threshold 0.9 --enable-expert-review --log-level INFO --input C:\StrictTest
```

**Python Script Examples:**
```cmd
REM Debug quality problems using Python
python main.py --epochs 100 --enable-adaptive --enable-constitutional --log-level DEBUG --input C:\QualityTest

REM Strict quality checking using Python
python main.py --quality-threshold 0.9 --enable-expert-review --log-level INFO --input C:\StrictTest
```

### File/Path Issues

**Direct Command Examples:**
```cmd
REM Test with current directory
bathymetric-cae --input . --output .\test_output --epochs 5

REM Verbose file processing
bathymetric-cae --log-level DEBUG --input "C:\Problem Data" --output "C:\Debug Results"

REM Check file format support
bathymetric-cae --log-level DEBUG --epochs 1 --input C:\SingleFile
```

**Python Script Examples:**
```cmd
REM Test with current directory using Python
python main.py --input . --output .\test_output --epochs 5

REM Verbose file processing using Python
python main.py --log-level DEBUG --input "C:\Problem Data" --output "C:\Debug Results"

REM Check file format support using Python
python main.py --log-level DEBUG --epochs 1 --input C:\SingleFile
```

---

## 💡 Windows CMD Pro Tips

### Environment Variables
```cmd
REM Set up commonly used paths
set SURVEY_DATA=C:\SurveyData
set RESULTS=C:\ProcessedResults
set CONFIG=C:\Configs\standard.json

REM Use in direct commands
bathymetric-cae --input %SURVEY_DATA% --output %RESULTS% --config %CONFIG%

REM Use in Python script commands
python main.py --input %SURVEY_DATA% --output %RESULTS% --config %CONFIG%
```

### Batch Files for Common Tasks

**Direct Command Batch File (quick_process.bat):**
```cmd
@echo off
echo Starting quick bathymetric processing...
bathymetric-cae --epochs 50 --ensemble-size 2 --input %1 --output %2
echo Processing complete!
pause

REM Usage: quick_process.bat "C:\Input Data" "C:\Output"
```

**Python Script Batch File (python_process.bat):**
```cmd
@echo off
echo Starting Python-based bathymetric processing...
python main.py --epochs 50 --ensemble-size 2 --input %1 --output %2
echo Processing complete!
pause

REM Usage: python_process.bat "C:\Input Data" "C:\Output"
```

### Combining with Other Windows Commands

**Direct Command Examples:**
```cmd
REM Create timestamped output folder
mkdir "C:\Results\%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%"
bathymetric-cae --input C:\Data --output "C:\Results\%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%"

REM Log processing to file
bathymetric-cae --input C:\Data --output C:\Results > processing_log.txt 2>&1
```

**Python Script Examples:**
```cmd
REM Create timestamped output folder and process with Python
mkdir "C:\Results\%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%"
python main.py --input C:\Data --output "C:\Results\%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%"

REM Log Python processing to file
python main.py --input C:\Data --output C:\Results > python_processing_log.txt 2>&1
```

---

## 🆘 Getting Help

### Help Commands

**Direct Command:**
```cmd
REM Show all available options
bathymetric-cae --help

REM Show version information
bathymetric-cae --version

REM Test installation
bathymetric-cae --epochs 1 --batch-size 1 --log-level DEBUG
```

**Python Script:**
```cmd
REM Show all available options using Python
python main.py --help

REM Show version information using Python
python main.py --version

REM Test installation using Python
python main.py --epochs 1 --batch-size 1 --log-level DEBUG
```

### Quick Diagnostics

**Direct Command:**
```cmd
REM Test basic functionality
bathymetric-cae --epochs 5 --batch-size 1 --grid-size 64 --input C:\TestData --log-level DEBUG

REM Check GPU availability
bathymetric-cae --log-level DEBUG --epochs 1 --batch-size 1

REM Verify file formats
bathymetric-cae --log-level DEBUG --input "C:\Sample Files"
```

**Python Script:**
```cmd
REM Test basic functionality using Python
python main.py --epochs 5 --batch-size 1 --grid-size 64 --input C:\TestData --log-level DEBUG

REM Check GPU availability using Python
python main.py --log-level DEBUG --epochs 1 --batch-size 1

REM Verify file formats using Python
python main.py --log-level DEBUG --input "C:\Sample Files"
```

---

## 📋 Quick Reference Summary

| Command | Quick Use | When to Change | Direct Command | Python Script |
|---------|-----------|----------------|----------------|---------------|
| `--epochs 100` | More = better quality | 25 for testing, 200+ for production | `bathymetric-cae --epochs 200` | `python main.py --epochs 200` |
| `--batch-size 8` | More = faster (needs RAM) | 1-2 for low memory, 16+ for high memory | `bathymetric-cae --batch-size 16` | `python main.py --batch-size 16` |
| `--grid-size 512` | More = better detail | 256 for speed, 1024 for quality | `bathymetric-cae --grid-size 1024` | `python main.py --grid-size 1024` |
| `--ensemble-size 3` | More = better accuracy | 1 for speed, 5+ for quality | `bathymetric-cae --ensemble-size 5` | `python main.py --ensemble-size 5` |
| `--enable-adaptive` | Smart processing | Always use unless testing | `bathymetric-cae --enable-adaptive` | `python main.py --enable-adaptive` |
| `--enable-expert-review` | Quality control | Use for important surveys | `bathymetric-cae --enable-expert-review` | `python main.py --enable-expert-review` |
| `--quality-threshold 0.7` | Higher = stricter | 0.6 for testing, 0.85+ for critical work | `bathymetric-cae --quality-threshold 0.85` | `python main.py --quality-threshold 0.85` |

## 🚀 When to Use Each Method

### Direct Command (`bathymetric-cae`)
- **Best for**: Production use, end users, simple workflows
- **Pros**: Clean, simple, professional appearance
- **Cons**: Requires proper installation/PATH setup

### Python Script (`python main.py`)
- **Best for**: Development, debugging, custom environments
- **Pros**: Always works if Python is installed, easier debugging
- **Cons**: Slightly more verbose, requires Python knowledge

### Python Module (`python -m enhanced_bathymetric_cae`)
- **Best for**: Package testing, virtual environments
- **Pros**: Works with package installations, explicit module loading
- **Cons**: Most verbose, mainly for advanced users

---

**Remember**: Start with defaults and adjust based on your specific needs for time, quality, and system resources!

For detailed documentation, visit: [Enhanced Bathymetric CAE Documentation](https://github.com/noaa-ocs-hydrography/bathymetric-cae/docs)