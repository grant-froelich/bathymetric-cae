#!/usr/bin/env python3
"""
enhanced_synthetic_bathymetry_generator.py
Enhanced Bathymetric CAE - Synthetic Data Generator with Realistic Noise Patterns
Based on makefakesurf.py by S. Greenaway and G. Rice

Generates realistic synthetic bathymetric data with:
- Multiple seafloor types (shallow coastal, deep ocean, seamount, etc.)
- Realistic acoustic noise patterns (sidelobe, refraction, bubble-sweep)
- Proper spectral characteristics 
- BAG and GeoTIFF output formats
- Uncertainty data for BAG files
- Configurable parameters for different survey scenarios
- Realistic geographic coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from osgeo import gdal
import warnings

# Suppress GDAL warnings
warnings.filterwarnings('ignore')

class SeafloorType(Enum):
    """Seafloor environment types with different characteristics."""
    SHALLOW_COASTAL = "shallow_coastal"
    CONTINENTAL_SHELF = "continental_shelf" 
    DEEP_OCEAN = "deep_ocean"
    SEAMOUNT = "seamount"
    ABYSSAL_PLAIN = "abyssal_plain"
    CANYON = "canyon"
    RIDGE = "ridge"

@dataclass
class NoiseConfig:
    """Configuration for acoustic noise patterns."""
    # Sidelobe noise parameters
    enable_sidelobe: bool = False
    sidelobe_amplitude: float = 0.3  # meters
    sidelobe_frequency: float = 8.0  # cycles per swath width
    sidelobe_outer_beam_factor: float = 2.0  # stronger in outer beams
    
    # Sound speed refraction noise parameters
    enable_refraction: bool = False
    refraction_smile_amplitude: float = 1.0  # meters
    refraction_asymmetry: float = 0.5  # asymmetric bias (0=symmetric, 1=fully asymmetric)
    refraction_temporal_drift: float = 0.2  # temporal drift in meters per survey line
    
    # Bubble sweep down noise parameters
    enable_bubble_sweep: bool = False
    bubble_sweep_amplitude: float = 0.8  # meters depth bias
    bubble_plume_width: float = 0.15  # fraction of swath width
    bubble_wake_length: float = 50.0  # meters
    bubble_random_factor: float = 0.3  # randomness in bubble effects
    
    # Survey pattern parameters (affects noise application)
    survey_line_spacing: float = 50.0  # meters between survey lines
    survey_speed: float = 8.0  # knots
    swath_width_factor: float = 4.0  # swath width as multiple of water depth

@dataclass
class BathymetryConfig:
    """Configuration for bathymetric data generation."""
    # Grid parameters
    width: int = 512
    height: int = 512
    resolution: float = 1.0  # meters per pixel
    
    # Depth parameters
    base_depth: float = -100.0  # meters (negative = below sea level)
    depth_range: float = 50.0   # meters variation
    
    # Spectral characteristics
    spectral_slope: float = -2.0  # Red noise slope (-1 to -3)
    roughness_scale: float = 1.0  # Multiplier for surface roughness
    
    # Noise parameters
    noise_level: float = 0.1      # Additive noise level
    uncertainty_base: float = 0.5  # Base uncertainty in meters
    uncertainty_scale: float = 0.02  # Uncertainty as fraction of depth
    
    # Feature parameters
    num_features: int = 3         # Number of bathymetric features
    feature_amplitude: float = 10.0  # Feature height variation
    
    # Geospatial coordinates (realistic survey locations)
    origin_x: float = 0.0
    origin_y: float = 0.0
    projection: str = "EPSG:32610"  # UTM Zone 10N (West Coast US)
    
    # Acoustic noise configuration
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)

class SurveyLocation:
    """Predefined realistic survey locations for bathymetric data."""
    
    # West Coast US locations (UTM Zone 10N - EPSG:32610)
    MONTEREY_BAY = {
        'name': 'Monterey Bay, CA',
        'utm_x': 603000,  # UTM Easting
        'utm_y': 4052000, # UTM Northing  
        'projection': 'EPSG:32610',
        'depth_range': (-200, -50),
        'description': 'Monterey Bay submarine canyon area'
    }
    
    SAN_FRANCISCO_BAY = {
        'name': 'San Francisco Bay, CA', 
        'utm_x': 551000,
        'utm_y': 4182000,
        'projection': 'EPSG:32610',
        'depth_range': (-60, -5),
        'description': 'San Francisco Bay entrance'
    }
    
    PUGET_SOUND = {
        'name': 'Puget Sound, WA',
        'utm_x': 548000,
        'utm_y': 5282000,
        'projection': 'EPSG:32610', 
        'depth_range': (-180, -20),
        'description': 'Puget Sound deep basin'
    }
    
    # East Coast US locations (UTM Zone 18N - EPSG:32618)
    CHESAPEAKE_BAY = {
        'name': 'Chesapeake Bay, MD',
        'utm_x': 382000,
        'utm_y': 4313000,
        'projection': 'EPSG:32618',
        'depth_range': (-50, -2),
        'description': 'Chesapeake Bay main channel'
    }
    
    CAPE_COD = {
        'name': 'Cape Cod, MA',
        'utm_x': 398000,
        'utm_y': 4635000,
        'projection': 'EPSG:32618',
        'depth_range': (-80, -10),
        'description': 'Cape Cod Bay approach'
    }
    
    # Gulf Coast (UTM Zone 15N - EPSG:32615)
    GULF_OF_AMERICA = {
        'name': 'Gulf of America, TX',
        'utm_x': 295000,
        'utm_y': 3230000,
        'projection': 'EPSG:32615',
        'depth_range': (-3000, -200),
        'description': 'Gulf of America continental slope'
    }
    
    # Great Lakes (UTM Zone 16N - EPSG:32616)
    LAKE_SUPERIOR = {
        'name': 'Lake Superior, MN',
        'utm_x': 698000,
        'utm_y': 5345000,
        'projection': 'EPSG:32616',
        'depth_range': (-400, -50),
        'description': 'Lake Superior deep basin'
    }
    
    # Alaska (UTM Zone 6N - EPSG:32606)
    COOK_INLET = {
        'name': 'Cook Inlet, AK',
        'utm_x': 355000,
        'utm_y': 6718000,
        'projection': 'EPSG:32606',
        'depth_range': (-80, -5),
        'description': 'Cook Inlet shipping channel'
    }
    
    # Hawaii (UTM Zone 4N - EPSG:32604)
    PEARL_HARBOR = {
        'name': 'Pearl Harbor, HI',
        'utm_x': 612000,
        'utm_y': 2365000,
        'projection': 'EPSG:32604',
        'depth_range': (-20, -3),
        'description': 'Pearl Harbor entrance channel'
    }
    
    @classmethod
    def get_all_locations(cls):
        """Get all predefined survey locations."""
        locations = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, dict) and 'name' in attr:
                locations[attr_name.lower()] = attr
        return locations
    
    @classmethod 
    def get_location_by_seafloor_type(cls, seafloor_type: SeafloorType):
        """Get appropriate survey location for seafloor type."""
        type_mapping = {
            SeafloorType.SHALLOW_COASTAL: cls.SAN_FRANCISCO_BAY,
            SeafloorType.CONTINENTAL_SHELF: cls.MONTEREY_BAY,
            SeafloorType.DEEP_OCEAN: cls.GULF_OF_AMERICA,
            SeafloorType.SEAMOUNT: cls.PEARL_HARBOR,
            SeafloorType.ABYSSAL_PLAIN: cls.GULF_OF_AMERICA,
            SeafloorType.CANYON: cls.MONTEREY_BAY,
            SeafloorType.RIDGE: cls.COOK_INLET
        }
        return type_mapping.get(seafloor_type, cls.MONTEREY_BAY)

class SyntheticBathymetryGenerator:
    """Generate realistic synthetic bathymetric data with acoustic noise patterns."""
    
    def __init__(self, config: BathymetryConfig):
        self.config = config
        self.seafloor_configs = self._initialize_seafloor_configs()
        
    def _initialize_seafloor_configs(self) -> Dict[SeafloorType, BathymetryConfig]:
        """Initialize configurations for different seafloor types."""
        configs = {}
        
        # Shallow coastal configuration
        configs[SeafloorType.SHALLOW_COASTAL] = BathymetryConfig(
            base_depth=-15.0,
            depth_range=12.0,
            spectral_slope=-1.5,
            roughness_scale=0.8,
            num_features=2,
            feature_amplitude=3.0,
            noise_config=self.config.noise_config
        )
        
        # Continental shelf configuration  
        configs[SeafloorType.CONTINENTAL_SHELF] = BathymetryConfig(
            base_depth=-75.0,
            depth_range=40.0,
            spectral_slope=-2.0,
            roughness_scale=1.0,
            num_features=3,
            feature_amplitude=8.0,
            noise_config=self.config.noise_config
        )
        
        # Deep ocean configuration
        configs[SeafloorType.DEEP_OCEAN] = BathymetryConfig(
            base_depth=-2500.0,
            depth_range=500.0,
            spectral_slope=-2.5,
            roughness_scale=1.2,
            num_features=4,
            feature_amplitude=50.0,
            noise_config=self.config.noise_config
        )
        
        # Seamount configuration
        configs[SeafloorType.SEAMOUNT] = BathymetryConfig(
            base_depth=-800.0,
            depth_range=600.0,
            spectral_slope=-1.8,
            roughness_scale=1.5,
            num_features=1,
            feature_amplitude=400.0,
            noise_config=self.config.noise_config
        )
        
        # Abyssal plain configuration
        configs[SeafloorType.ABYSSAL_PLAIN] = BathymetryConfig(
            base_depth=-4000.0,
            depth_range=100.0,
            spectral_slope=-2.8,
            roughness_scale=0.5,
            num_features=2,
            feature_amplitude=20.0,
            noise_config=self.config.noise_config
        )
        
        # Canyon configuration
        configs[SeafloorType.CANYON] = BathymetryConfig(
            base_depth=-150.0,
            depth_range=120.0,
            spectral_slope=-1.6,
            roughness_scale=2.0,
            num_features=1,
            feature_amplitude=80.0,
            noise_config=self.config.noise_config
        )
        
        # Ridge configuration
        configs[SeafloorType.RIDGE] = BathymetryConfig(
            base_depth=-50.0,
            depth_range=30.0,
            spectral_slope=-1.4,
            roughness_scale=1.8,
            num_features=1,
            feature_amplitude=25.0,
            noise_config=self.config.noise_config
        )
        
        return configs

    def generate_fractal_surface(self, config: BathymetryConfig) -> np.ndarray:
        """Generate fractal bathymetric surface using spectral synthesis."""
        print(f"   Generating {config.width}x{config.height} fractal surface...")
        
        # Create frequency grids
        kx = np.fft.fftfreq(config.width, d=config.resolution)
        ky = np.fft.fftfreq(config.height, d=config.resolution)
        kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
        
        # Calculate radial frequency (avoid division by zero)
        k_radial = np.sqrt(kx_2d**2 + ky_2d**2)
        k_radial[0, 0] = 1e-10  # Avoid division by zero at DC
        
        # Generate power spectrum with specified slope
        # P(k) ∝ k^β where β is the spectral slope
        power_spectrum = k_radial ** config.spectral_slope
        
        # Add low-frequency enhancement for realistic bathymetry
        low_freq_enhancement = 1.0 + 10.0 * np.exp(-k_radial * config.resolution * 50)
        power_spectrum *= low_freq_enhancement
        
        # Generate complex random phases
        phases = np.random.uniform(0, 2*np.pi, (config.width, config.height))
        
        # Create complex amplitude spectrum
        amplitude = np.sqrt(power_spectrum) * config.roughness_scale
        complex_spectrum = amplitude * np.exp(1j * phases)
        
        # Ensure Hermitian symmetry for real output
        complex_spectrum = self._ensure_hermitian_symmetry(complex_spectrum)
        
        # Generate surface via inverse FFT
        surface = np.real(np.fft.ifft2(complex_spectrum))
        
        # Normalize to desired depth range
        surface = surface - np.mean(surface)  # Zero mean
        surface = surface / np.std(surface) * (config.depth_range / 4.0)  # Scale to range
        surface += config.base_depth  # Add base depth
        
        return surface

    def _ensure_hermitian_symmetry(self, spectrum: np.ndarray) -> np.ndarray:
        """Ensure Hermitian symmetry for real-valued IFFT output."""
        height, width = spectrum.shape
        
        # Handle even/odd dimensions properly
        for i in range(1, height):
            for j in range(1, width):
                # Mirror index calculation
                i_mirror = (height - i) % height
                j_mirror = (width - j) % width
                
                # Enforce conjugate symmetry
                if i_mirror < i or (i_mirror == i and j_mirror < j):
                    spectrum[i_mirror, j_mirror] = np.conj(spectrum[i, j])
        
        return spectrum

    def add_bathymetric_features(self, surface: np.ndarray, config: BathymetryConfig) -> np.ndarray:
        """Add realistic bathymetric features to the surface."""
        print(f"   Adding {config.num_features} bathymetric features...")
        
        height, width = surface.shape
        
        for _ in range(config.num_features):
            # Random feature location
            center_x = np.random.randint(width // 4, 3 * width // 4)
            center_y = np.random.randint(height // 4, 3 * height // 4)
            
            # Feature size and shape parameters
            sigma_x = np.random.uniform(width * 0.05, width * 0.2)
            sigma_y = np.random.uniform(height * 0.05, height * 0.2)
            amplitude = np.random.uniform(-config.feature_amplitude, config.feature_amplitude)
            
            # Create coordinate grids
            y_coords, x_coords = np.ogrid[:height, :width]
            
            # Gaussian feature
            feature = amplitude * np.exp(
                -((x_coords - center_x)**2 / (2 * sigma_x**2) +
                  (y_coords - center_y)**2 / (2 * sigma_y**2))
            )
            
            # Add directional bias for canyon/ridge features
            if abs(amplitude) > config.feature_amplitude * 0.7:
                # Add linear trend for valley/ridge
                trend_direction = np.random.uniform(0, 2*np.pi)
                trend_strength = amplitude * 0.3
                x_trend = np.cos(trend_direction) * (x_coords - center_x) / width
                y_trend = np.sin(trend_direction) * (y_coords - center_y) / height
                feature += trend_strength * (x_trend + y_trend)
            
            surface += feature
            
        return surface

    def apply_acoustic_noise(self, surface: np.ndarray, config: BathymetryConfig) -> np.ndarray:
        """Apply realistic acoustic noise patterns to bathymetric data."""
        if not any([config.noise_config.enable_sidelobe, 
                   config.noise_config.enable_refraction,
                   config.noise_config.enable_bubble_sweep]):
            return surface
            
        print(f"   Applying acoustic noise patterns...")
        
        noisy_surface = surface.copy()
        height, width = surface.shape
        
        # Apply sidelobe noise
        if config.noise_config.enable_sidelobe:
            print(f"     • Sidelobe interference")
            sidelobe_noise = self._generate_sidelobe_noise(surface, config.noise_config, width, height)
            noisy_surface += sidelobe_noise
            
        # Apply refraction noise (smile/frown)
        if config.noise_config.enable_refraction:
            print(f"     • Sound speed refraction")
            refraction_noise = self._generate_refraction_noise(surface, config.noise_config, width, height)
            noisy_surface += refraction_noise
            
        # Apply bubble sweep noise
        if config.noise_config.enable_bubble_sweep:
            print(f"     • Bubble sweep artifacts")
            bubble_noise = self._generate_bubble_sweep_noise(surface, config.noise_config, width, height)
            noisy_surface += bubble_noise
            
        return noisy_surface

    def _generate_sidelobe_noise(self, surface: np.ndarray, noise_config: NoiseConfig, 
                                width: int, height: int) -> np.ndarray:
        """Generate sidelobe interference patterns."""
        sidelobe_noise = np.zeros((height, width))
        
        # Create across-track coordinate (beam angle proxy)
        across_track = np.linspace(-1, 1, width)
        
        for row in range(height):
            # Outer beam amplification
            beam_factor = 1.0 + noise_config.sidelobe_outer_beam_factor * np.abs(across_track)
            
            # Sinusoidal sidelobe pattern
            sidelobe_pattern = noise_config.sidelobe_amplitude * beam_factor * \
                              np.sin(2 * np.pi * noise_config.sidelobe_frequency * across_track)
            
            # Add random phase variation per survey line
            phase_shift = np.random.uniform(0, 2*np.pi)
            sidelobe_pattern *= np.cos(phase_shift)
            
            sidelobe_noise[row, :] = sidelobe_pattern
            
        return sidelobe_noise

    def _generate_refraction_noise(self, surface: np.ndarray, noise_config: NoiseConfig,
                                  width: int, height: int) -> np.ndarray:
        """Generate sound speed refraction artifacts (smile/frown)."""
        refraction_noise = np.zeros((height, width))
        
        # Create across-track coordinate
        across_track = np.linspace(-1, 1, width)
        
        # Parabolic refraction pattern (smile/frown)
        base_pattern = noise_config.refraction_smile_amplitude * across_track**2
        
        # Add asymmetry
        asymmetric_component = noise_config.refraction_asymmetry * \
                              noise_config.refraction_smile_amplitude * across_track
        
        for row in range(height):
            # Temporal drift
            drift = noise_config.refraction_temporal_drift * row / height
            
            # Random variation in refraction strength
            strength_variation = 1.0 + 0.3 * np.random.normal()
            
            refraction_pattern = strength_variation * (base_pattern + asymmetric_component) + drift
            refraction_noise[row, :] = refraction_pattern
            
        return refraction_noise

    def _generate_bubble_sweep_noise(self, surface: np.ndarray, noise_config: NoiseConfig,
                                    width: int, height: int) -> np.ndarray:
        """Generate bubble sweep down artifacts."""
        bubble_noise = np.zeros((height, width))
        
        # Random bubble plume locations
        num_plumes = max(1, int(height / 20))  # One plume every ~20 survey lines
        
        for _ in range(num_plumes):
            # Random plume start location
            plume_start_row = np.random.randint(0, height)
            plume_center_col = np.random.randint(int(width * 0.2), int(width * 0.8))
            
            # Plume dimensions
            plume_width = int(noise_config.bubble_plume_width * width)
            wake_length = int(noise_config.bubble_wake_length / 2.0)  # Assuming 2m resolution
            
            # Apply bubble effect
            for row_offset in range(wake_length):
                current_row = plume_start_row + row_offset
                if current_row >= height:
                    break
                    
                # Gaussian profile across plume width
                col_start = max(0, plume_center_col - plume_width // 2)
                col_end = min(width, plume_center_col + plume_width // 2)
                
                for col in range(col_start, col_end):
                    # Distance from plume center
                    distance = abs(col - plume_center_col) / (plume_width / 2)
                    
                    # Gaussian amplitude with decay along wake
                    amplitude = noise_config.bubble_sweep_amplitude * \
                              np.exp(-distance**2) * \
                              np.exp(-row_offset / (wake_length / 3))
                    
                    # Add randomness
                    amplitude *= (1.0 + noise_config.bubble_random_factor * np.random.normal())
                    
                    bubble_noise[current_row, col] += amplitude
                    
        return bubble_noise

    def generate_uncertainty_data(self, depth_data: np.ndarray, config: BathymetryConfig) -> np.ndarray:
        """Generate realistic uncertainty estimates for bathymetric data."""
        print(f"   Generating uncertainty estimates...")
        
        # Base uncertainty component
        uncertainty = np.full_like(depth_data, config.uncertainty_base)
        
        # Depth-dependent uncertainty (deeper = more uncertain)
        depth_uncertainty = config.uncertainty_scale * np.abs(depth_data)
        uncertainty += depth_uncertainty
        
        # Slope-dependent uncertainty (steeper slopes = more uncertain)
        gradient_y, gradient_x = np.gradient(depth_data)
        slope_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        slope_uncertainty = 0.1 * slope_magnitude
        uncertainty += slope_uncertainty
        
        # Add noise pattern dependent uncertainty
        if config.noise_config.enable_sidelobe:
            # Higher uncertainty in outer beams
            width = depth_data.shape[1]
            across_track = np.linspace(-1, 1, width)
            beam_uncertainty = 0.2 * (1.0 + 2.0 * np.abs(across_track))
            uncertainty += beam_uncertainty[np.newaxis, :]
            
        if config.noise_config.enable_bubble_sweep:
            # Add random uncertainty spikes for bubble artifacts
            bubble_mask = np.random.random(depth_data.shape) < 0.05  # 5% of points
            uncertainty[bubble_mask] += np.random.uniform(0.5, 2.0, np.sum(bubble_mask))
        
        # Ensure minimum uncertainty
        uncertainty = np.maximum(uncertainty, 0.1)
        
        # Add small random component
        uncertainty += np.random.normal(0, 0.05, uncertainty.shape)
        uncertainty = np.maximum(uncertainty, 0.1)  # Ensure positive
        
        return uncertainty

    def generate_bathymetry(self, seafloor_type: SeafloorType, 
                          override_config: Optional[BathymetryConfig] = None,
                          survey_location: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete bathymetric dataset."""
        
        # Use override config if provided, otherwise use preset
        if override_config is not None:
            config = override_config
            print(f"   Using custom configuration")
        else:
            config = self.seafloor_configs[seafloor_type]
            print(f"   Using {seafloor_type.value} configuration")
            
        # Override grid size if specified
        if hasattr(self.config, 'width') and hasattr(self.config, 'height'):
            config.width = self.config.width
            config.height = self.config.height
            config.resolution = self.config.resolution
            
        # Set geographic coordinates from survey location
        if survey_location:
            config.origin_x = survey_location['utm_x']
            config.origin_y = survey_location['utm_y'] 
            config.projection = survey_location['projection']
            # Override depth range if specified in location
            if 'depth_range' in survey_location:
                config.base_depth = np.mean(survey_location['depth_range'])
                config.depth_range = abs(survey_location['depth_range'][1] - survey_location['depth_range'][0])
        
        print(f"   Grid: {config.width}x{config.height} at {config.resolution}m resolution")
        print(f"   Location: {survey_location['name'] if survey_location else 'Default'}")
        print(f"   Depth range: {config.base_depth - config.depth_range/2:.1f} to {config.base_depth + config.depth_range/2:.1f}m")
        
        # Generate fractal surface
        surface = self.generate_fractal_surface(config)
        
        # Add bathymetric features
        surface = self.add_bathymetric_features(surface, config)
        
        # Apply acoustic noise if enabled
        surface = self.apply_acoustic_noise(surface, config)
        
        # Generate uncertainty data
        uncertainty = self.generate_uncertainty_data(surface, config)
        
        print(f"   ✓ Generated {surface.shape[0]}x{surface.shape[1]} bathymetric grid")
        print(f"   ✓ Depth range: {np.min(surface):.2f} to {np.max(surface):.2f} m")
        print(f"   ✓ Uncertainty range: {np.min(uncertainty):.3f} to {np.max(uncertainty):.3f} m")
        
        return surface, uncertainty

def create_noise_preset_configs() -> Dict[str, NoiseConfig]:
    """Create predefined noise configurations for different survey scenarios."""
    presets = {}
    
    # Clean data (no acoustic artifacts)
    presets['clean'] = NoiseConfig()
    
    # Light noise (minimal artifacts)
    presets['light'] = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=0.15,
        enable_refraction=True,
        refraction_smile_amplitude=0.3
    )
    
    # Moderate noise (typical survey conditions)
    presets['moderate'] = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=0.3,
        enable_refraction=True,
        refraction_smile_amplitude=0.8,
        enable_bubble_sweep=True,
        bubble_sweep_amplitude=0.4
    )
    
    # Heavy noise (challenging survey conditions)
    presets['heavy'] = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=0.6,
        sidelobe_outer_beam_factor=3.0,
        enable_refraction=True,
        refraction_smile_amplitude=1.5,
        refraction_asymmetry=0.8,
        enable_bubble_sweep=True,
        bubble_sweep_amplitude=1.2,
        bubble_random_factor=0.5
    )
    
    # Sidelobe only
    presets['sidelobe_only'] = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=0.5
    )
    
    # Refraction only  
    presets['refraction_only'] = NoiseConfig(
        enable_refraction=True,
        refraction_smile_amplitude=1.0
    )
    
    # Bubble sweep only
    presets['bubble_only'] = NoiseConfig(
        enable_bubble_sweep=True,
        bubble_sweep_amplitude=0.8
    )
    
    return presets

def save_bag_file(depth_data: np.ndarray, uncertainty_data: np.ndarray, 
                  filename: str, config: BathymetryConfig, metadata: Dict[str, str]):
    """Save bathymetric data to BAG format using GDAL."""
    
    print(f"   Saving BAG file: {filename}")
    
    try:
        # Ensure data is in the correct format
        height, width = depth_data.shape
        
        # Convert to float32 for BAG format
        depth_data_f32 = depth_data.astype(np.float32)
        uncertainty_data_f32 = uncertainty_data.astype(np.float32)
        
        # Create BAG file using GDAL
        driver = gdal.GetDriverByName('BAG')
        if driver is None:
            print(f"   ⚠️  BAG driver not available, saving as GeoTIFF instead")
            save_geotiff_file(depth_data, uncertainty_data, filename.replace('.bag', '.tif'), config, metadata)
            return
            
        dataset = driver.Create(filename, width, height, 2, gdal.GDT_Float32)
        
        if dataset is None:
            raise RuntimeError(f"Failed to create BAG file: {filename}")
            
        # Set geotransform (GDAL geotransform: [top-left x, pixel width, rotation, top-left y, rotation, pixel height])
        geotransform = [
            config.origin_x,           # Top-left X (UTM Easting)
            config.resolution,         # Pixel width (meters)
            0,                         # Rotation (0 for north-up)
            config.origin_y + (height * config.resolution),  # Top-left Y (UTM Northing)
            0,                         # Rotation (0 for north-up)  
            -config.resolution         # Pixel height (negative for north-up)
        ]
        dataset.SetGeoTransform(geotransform)
        
        # Set appropriate projection (UTM zones for bathymetry)
        dataset.SetProjection(config.projection)
        
        # Write depth data (band 1)
        depth_band = dataset.GetRasterBand(1)
        
        print(f"   Writing depth band...")
        write_result = depth_band.WriteArray(depth_data_f32)
        if write_result != 0:
            print(f"   ⚠️  WriteArray returned error code: {write_result}")
        
        depth_band.SetDescription('Depth')
        depth_band.SetNoDataValue(-9999)
        
        # Force flush after writing depth
        depth_band.FlushCache()
        
        # Write uncertainty data (band 2)
        uncertainty_band = dataset.GetRasterBand(2)
        print(f"   Writing uncertainty band...")
        uncertainty_band.WriteArray(uncertainty_data_f32)
        uncertainty_band.SetDescription('Uncertainty')
        uncertainty_band.SetNoDataValue(-9999)
        uncertainty_band.FlushCache()
        
        # Add comprehensive metadata
        for key, value in metadata.items():
            dataset.SetMetadataItem(key, str(value))
            
        # BAG-specific metadata
        dataset.SetMetadataItem('BAG_CREATION_DATE', metadata.get('CREATION_DATE', '2024-01-01T00:00:00Z'))
        dataset.SetMetadataItem('BAG_VERSION', '1.6.2')
        dataset.SetMetadataItem('BAG_DATUM', 'WGS84')
        dataset.SetMetadataItem('BAG_COORDINATE_SYSTEM', config.projection)
        
        # Force final flush and close
        dataset.FlushCache()
        dataset = None  # Close file
        
        print(f"   ✓ BAG file created successfully")
        
    except Exception as e:
        print(f"   ❌ Error creating BAG file: {e}")
        print(f"   Attempting to save as GeoTIFF instead...")
        save_geotiff_file(depth_data, uncertainty_data, filename.replace('.bag', '.tif'), config, metadata)

def save_geotiff_file(depth_data: np.ndarray, uncertainty_data: np.ndarray,
                     filename: str, config: BathymetryConfig, metadata: Dict[str, str]):
    """Save bathymetric data to GeoTIFF format."""
    
    print(f"   Saving GeoTIFF file: {filename}")
    
    try:
        height, width = depth_data.shape
        
        # Create GeoTIFF file
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(filename, width, height, 2, gdal.GDT_Float32,
                              ['COMPRESS=LZW', 'TILED=YES'])
        
        if dataset is None:
            raise RuntimeError(f"Failed to create GeoTIFF file: {filename}")
            
        # Set geotransform
        geotransform = [
            config.origin_x,
            config.resolution,
            0,
            config.origin_y + (height * config.resolution),
            0,
            -config.resolution
        ]
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(config.projection)
        
        # Write depth data (band 1)
        depth_band = dataset.GetRasterBand(1)
        depth_band.WriteArray(depth_data.astype(np.float32))
        depth_band.SetDescription('Depth (meters)')
        depth_band.SetNoDataValue(-9999)
        
        # Write uncertainty data (band 2)
        uncertainty_band = dataset.GetRasterBand(2)
        uncertainty_band.WriteArray(uncertainty_data.astype(np.float32))
        uncertainty_band.SetDescription('Uncertainty (meters)')
        uncertainty_band.SetNoDataValue(-9999)
        
        # Add metadata
        for key, value in metadata.items():
            dataset.SetMetadataItem(key, str(value))
            
        dataset.FlushCache()
        dataset = None
        
        print(f"   ✓ GeoTIFF file created successfully")
        
    except Exception as e:
        print(f"   ❌ Error creating GeoTIFF file: {e}")

def calculate_grid_size_for_nodes(target_nodes: int) -> Tuple[int, int]:
    """Calculate grid dimensions for a target number of nodes."""
    # Aim for roughly square grid
    side_length = int(np.sqrt(target_nodes))
    
    # Round to nearest power-of-2 friendly numbers for better performance
    if side_length <= 256:
        width = height = 256
    elif side_length <= 512:
        width = height = 512
    elif side_length <= 1024:
        width = height = 1024
    elif side_length <= 2048:
        width = height = 2048
    elif side_length <= 4096:
        width = height = 4096
    else:
        # For very large grids, use rectangular shapes
        width = 4096
        height = target_nodes // width
        
    return width, height

def generate_large_scale_data(target_nodes: int, resolution: float = 2.0, 
                            num_files: int = 5, noise_preset: str = 'moderate',
                            output_dir: str = 'large_bathymetry_data'):
    """Generate large-scale bathymetric datasets for testing."""
    
    output_dir = Path(output_dir)
    
    print(f"🏭 Large-Scale Bathymetric Data Generation")
    print(f"🎯 Target: {target_nodes:,} nodes per file")
    print(f"📏 Resolution: {resolution}m")
    print(f"📊 Files to create: {num_files}")
    print(f"🔊 Noise preset: {noise_preset}")
    print(f"📁 Output directory: {output_dir}")
    
    # Calculate grid size
    width, height = calculate_grid_size_for_nodes(target_nodes)
    actual_nodes = width * height
    
    print(f"\n📐 Grid Configuration:")
    print(f"   Dimensions: {width} x {height}")
    print(f"   Actual nodes: {actual_nodes:,}")
    print(f"   Coverage: {width * resolution:.1f}m x {height * resolution:.1f}m")
    print(f"   Area: {(width * resolution * height * resolution) / 1_000_000:.1f} km²")
    
    # Memory estimation
    estimated_size_mb = (actual_nodes * 4 * 2) / (1024 * 1024)  # 4 bytes * 2 bands
    print(f"   Estimated file size: {estimated_size_mb:.1f} MB per BAG file")
    
    if actual_nodes >= 5_000_000:
        print(f"\n⚠️  Very large dataset warning!")
        print(f"   This will create files with {actual_nodes:,} nodes each")
        print(f"   Memory usage: ~{estimated_size_mb * 3:.1f} MB per file during processing")
        print(f"   Consider processing files individually.")
    
    # Create custom configuration for large grids with noise
    large_config = BathymetryConfig(
        width=width,
        height=height,
        resolution=resolution,
        noise_config=noise_config
    )
    
    generator = SyntheticBathymetryGenerator(large_config)
    created_files = []
    seafloor_types = list(SeafloorType)
    
    for i in range(num_files):
        seafloor_type = seafloor_types[i % len(seafloor_types)]
        survey_location = survey_locations[i % len(survey_locations)]
        
        print(f"\n📊 Creating large BAG file {i+1}/{num_files}: {seafloor_type.value}")
        print(f"   Location: {survey_location['name']}")
        print(f"   Generating {actual_nodes:,} depth and uncertainty values...")
        
        try:
            # Generate data with progress indication and location
            depth_data, uncertainty_data = generator.generate_bathymetry(
                seafloor_type, large_config, survey_location
            )
            
            # Create comprehensive metadata including location and noise info
            metadata = {
                'SEAFLOOR_TYPE': seafloor_type.value,
                'CREATION_DATE': '2024-01-01T00:00:00Z',
                'GENERATOR': 'Enhanced Bathymetric CAE Large Grid Generator with Acoustic Noise',
                'NODE_COUNT': str(actual_nodes),
                'GRID_SIZE': f"{width}x{height}",
                'RESOLUTION': f"{resolution}m",
                'COVERAGE_AREA': f"{width * resolution:.1f}m x {height * resolution:.1f}m",
                'DEPTH_RANGE': f"{np.min(depth_data):.2f} to {np.max(depth_data):.2f} m",
                'UNCERTAINTY_RANGE': f"{np.min(uncertainty_data):.3f} to {np.max(uncertainty_data):.3f} m",
                'DATA_QUALITY': 'Synthetic high-resolution bathymetry with realistic acoustic noise',
                'PURPOSE': 'Large-scale testing of Enhanced Bathymetric CAE algorithms',
                'SURVEY_LOCATION': survey_location['name'],
                'NOISE_PRESET': noise_preset
            }
            
            # Save file
            filename = f"large_bathymetry_{seafloor_type.value}_{width}x{height}_{i+1:03d}.bag"
            filepath = output_dir / filename
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_bag_file(depth_data, uncertainty_data, str(filepath), large_config, metadata)
            created_files.append(filepath)
            
        except Exception as e:
            print(f"   ❌ Error creating file {i+1}: {e}")
            continue
    
    print(f"\n✅ Large-scale data generation complete!")
    print(f"   Created {len(created_files)} files")
    print(f"   Total nodes generated: {len(created_files) * actual_nodes:,}")
    print(f"   Output directory: {output_dir}")
    
    return created_files

def generate_bathymetry_files(num_files: int = 10, output_dir: str = 'synthetic_bathymetry', 
                            noise_preset: str = 'moderate', formats: List[str] = None):
    """Generate multiple synthetic bathymetric files with different seafloor types."""
    
    output_dir = Path(output_dir)
    
    if formats is None:
        formats = ['bag', 'tiff']
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get noise configuration
    noise_presets = create_noise_preset_configs()
    if noise_preset not in noise_presets:
        print(f"Warning: Unknown noise preset '{noise_preset}'. Available: {list(noise_presets.keys())}")
        noise_preset = 'clean'
    
    noise_config = noise_presets[noise_preset]
    
    # Create configuration with noise
    config = BathymetryConfig(noise_config=noise_config)
    generator = SyntheticBathymetryGenerator(config)
    
    created_files = []
    seafloor_types = list(SeafloorType)
    
    print(f"🔊 Using noise preset: {noise_preset}")
    
    for i in range(num_files):
        seafloor_type = seafloor_types[i % len(seafloor_types)]
        
        print(f"\n📊 Creating file {i+1}/{num_files}: {seafloor_type.value}")
        
        # Generate data
        depth_data, uncertainty_data = generator.generate_bathymetry(seafloor_type)
        
        # Create metadata
        noise_types = []
        if noise_config.enable_sidelobe:
            noise_types.append("sidelobe")
        if noise_config.enable_refraction:
            noise_types.append("refraction")
        if noise_config.enable_bubble_sweep:
            noise_types.append("bubble-sweep")
        
        metadata = {
            'SEAFLOOR_TYPE': seafloor_type.value,
            'CREATION_DATE': '2024-01-01T00:00:00Z',
            'GENERATOR': 'Enhanced Bathymetric CAE Synthetic Generator with Acoustic Noise',
            'DEPTH_RANGE': f"{np.min(depth_data):.2f} to {np.max(depth_data):.2f} m",
            'RESOLUTION': f"{generator.config.resolution} m",
            'NOISE_PRESET': noise_preset,
            'NOISE_PATTERNS': ', '.join(noise_types) if noise_types else 'none',
            'DATA_QUALITY': 'Synthetic bathymetry with realistic characteristics',
            'PURPOSE': 'Testing Enhanced Bathymetric CAE denoising algorithms'
        }
        
        # Save in requested formats
        for fmt in formats:
            if fmt.lower() == 'bag':
                filename = f"synthetic_bathymetry_{seafloor_type.value}_{i+1:03d}.bag"
                filepath = output_dir / filename
                save_bag_file(depth_data, uncertainty_data, str(filepath), config, metadata)
                created_files.append(filepath)
                
            elif fmt.lower() in ['tiff', 'tif', 'geotiff']:
                filename = f"synthetic_bathymetry_{seafloor_type.value}_{i+1:03d}.tif"
                filepath = output_dir / filename
                save_geotiff_file(depth_data, uncertainty_data, str(filepath), config, metadata)
                created_files.append(filepath)
    
    print(f"\n✅ Generated {len(created_files)} bathymetric files")
    print(f"📁 Output directory: {output_dir}")
    
    return created_files

def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description='Enhanced Synthetic Bathymetry Generator with Realistic Acoustic Noise',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 files with moderate noise
  python synthetic_bathymetry_generator.py --num-files 10 --noise-preset moderate
  
  # Generate large-scale data (1M nodes per file)
  python synthetic_bathymetry_generator.py --large-scale --target-nodes 1000000 --num-files 5
  
  # Generate with custom noise settings
  python synthetic_bathymetry_generator.py --enable-sidelobe --enable-refraction
  
  # Estimate memory requirements for large grids
  python synthetic_bathymetry_generator.py --estimate-only --target-nodes 5000000
  
  # List available survey locations
  python synthetic_bathymetry_generator.py --list-locations
        """)
    
    # Basic generation options
    parser.add_argument('--num-files', type=int, default=10,
                       help='Number of bathymetric files to generate (default: 10)')
    parser.add_argument('--output-dir', type=str, default='synthetic_bathymetry',
                       help='Output directory for generated files (default: synthetic_bathymetry)')
    parser.add_argument('--formats', nargs='+', choices=['bag', 'tiff'], default=['bag'],
                       help='Output formats (default: bag)')
    
    # Noise configuration options
    parser.add_argument('--noise-preset', type=str, default='moderate',
                       choices=['clean', 'light', 'moderate', 'heavy', 'sidelobe_only', 'refraction_only', 'bubble_only'],
                       help='Predefined noise configuration (default: moderate)')
    
    # Custom noise options
    parser.add_argument('--enable-sidelobe', action='store_true',
                       help='Enable sidelobe interference noise')
    parser.add_argument('--sidelobe-amplitude', type=float, default=0.3,
                       help='Sidelobe noise amplitude in meters (default: 0.3)')
    parser.add_argument('--enable-refraction', action='store_true',
                       help='Enable sound speed refraction artifacts')
    parser.add_argument('--refraction-amplitude', type=float, default=1.0,
                       help='Refraction artifact amplitude in meters (default: 1.0)')
    parser.add_argument('--enable-bubble-sweep', action='store_true',
                       help='Enable bubble sweep noise')
    parser.add_argument('--bubble-amplitude', type=float, default=0.8,
                       help='Bubble sweep amplitude in meters (default: 0.8)')
    
    # Large-scale generation options
    parser.add_argument('--large-scale', action='store_true',
                       help='Generate large-scale datasets for performance testing')
    parser.add_argument('--target-nodes', type=int, default=1000000,
                       help='Target number of nodes per file for large-scale generation (default: 1M)')
    parser.add_argument('--resolution', type=float, default=2.0,
                       help='Grid resolution in meters (default: 2.0)')
    
    # Utility options
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only estimate memory requirements and file sizes, don\'t generate data')
    parser.add_argument('--list-locations', action='store_true',
                       help='List available survey locations and exit')
    parser.add_argument('--location', type=str,
                       help='Specify survey location for generation')
    
    args = parser.parse_args()
    
    # Handle list locations request
    if args.list_locations:
        locations = SurveyLocation.get_all_locations()
        print("📍 Available Survey Locations:")
        print("=" * 50)
        for key, loc in locations.items():
            print(f"🗺️  {key}:")
            print(f"   Name: {loc['name']}")
            print(f"   Projection: {loc['projection']}")
            print(f"   UTM Coordinates: {loc['utm_x']}, {loc['utm_y']}")
            print(f"   Depth Range: {loc['depth_range'][0]}m to {loc['depth_range'][1]}m")
            print()
        
        print("Usage examples:")
        print("  --location monterey_bay")
        print("  --location chesapeake_bay")
        print("  --location gulf_of_america")
        return
    
    # Handle estimation mode
    if args.estimate_only:
        print(f"\n📊 Grid Size Estimation for {args.target_nodes:,} nodes:")
        width, height = calculate_grid_size_for_nodes(args.target_nodes)
        actual_nodes = width * height
        
        # Calculate estimates
        estimated_size_mb = (actual_nodes * 4 * 2) / (1024 * 1024)
        memory_req_mb = estimated_size_mb * 3
        
        print(f"📐 Required grid: {width} x {height}")
        print(f"🎯 Actual nodes: {actual_nodes:,}")
        print(f"📏 At {args.resolution}m resolution:")
        print(f"   Coverage: {width * args.resolution:.1f}m x {height * args.resolution:.1f}m")
        print(f"   Area: {(width * args.resolution * height * args.resolution) / 1_000_000:.1f} km²")
        print(f"💾 File size estimate: {estimated_size_mb:.1f} MB per BAG file")
        print(f"🧠 Memory requirement: ~{memory_req_mb:.1f} MB per file")
        print(f"⏱️  Processing time estimate: 3-8 minutes per file")
        
        if actual_nodes >= 1_000_000:
            print(f"⚠️  Large dataset warning: {actual_nodes:,} nodes")
            print(f"   • Consider processing with reduced batch size")
            print(f"   • Ensure adequate RAM ({memory_req_mb/1024:.1f} GB+)")
            print(f"   • Processing may take 10+ minutes per file")
        
        return
    
    # Determine noise configuration
    if any([args.enable_sidelobe, args.enable_refraction, args.enable_bubble_sweep]):
        # Custom noise configuration from command line arguments
        custom_noise_config = NoiseConfig(
            enable_sidelobe=args.enable_sidelobe,
            sidelobe_amplitude=args.sidelobe_amplitude,
            enable_refraction=args.enable_refraction,
            refraction_smile_amplitude=args.refraction_amplitude,
            enable_bubble_sweep=args.enable_bubble_sweep,
            bubble_sweep_amplitude=args.bubble_amplitude
        )
        noise_preset_name = 'custom'
        
        print(f"\n🔊 Using custom noise configuration:")
        if custom_noise_config.enable_sidelobe:
            print(f"   • Sidelobe: {custom_noise_config.sidelobe_amplitude}m amplitude")
        if custom_noise_config.enable_refraction:
            print(f"   • Refraction: {custom_noise_config.refraction_smile_amplitude}m amplitude")
        if custom_noise_config.enable_bubble_sweep:
            print(f"   • Bubble sweep: {custom_noise_config.bubble_sweep_amplitude}m amplitude")
    else:
        custom_noise_config = None
        noise_preset_name = args.noise_preset
    
    # Handle large-scale generation
    if args.large_scale:
        if custom_noise_config:
            # For large scale with custom noise, we need to modify the generate_large_scale_data function
            print("⚠️  Large-scale generation with custom noise not yet implemented")
            print("   Using moderate noise preset instead")
            noise_preset_name = 'moderate'
            
        generate_large_scale_data(
            target_nodes=args.target_nodes,
            resolution=args.resolution,
            num_files=args.num_files,
            noise_preset=noise_preset_name,
            output_dir=args.output_dir
        )
    else:
        # Standard generation
        if custom_noise_config:
            # Create temporary preset for custom config
            noise_presets = create_noise_preset_configs()
            noise_presets['custom'] = custom_noise_config
            
        generate_bathymetry_files(
            num_files=args.num_files,
            output_dir=args.output_dir,
            noise_preset=noise_preset_name,
            formats=args.formats
        )

if __name__ == "__main__":
    main()