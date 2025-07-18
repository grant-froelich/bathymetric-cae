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
    GULF_OF_MEXICO = {
        'name': 'Gulf of Mexico, TX',
        'utm_x': 295000,
        'utm_y': 3230000,
        'projection': 'EPSG:32615',
        'depth_range': (-3000, -200),
        'description': 'Gulf of Mexico continental slope'
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
            SeafloorType.DEEP_OCEAN: cls.GULF_OF_MEXICO,
            SeafloorType.SEAMOUNT: cls.PEARL_HARBOR,
            SeafloorType.ABYSSAL_PLAIN: cls.GULF_OF_MEXICO,
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
        
        # Shallow Coastal - high variability, complex features
        configs[SeafloorType.SHALLOW_COASTAL] = BathymetryConfig(
            base_depth=-20.0, depth_range=15.0, spectral_slope=-1.5,
            roughness_scale=2.0, noise_level=0.3, num_features=5,
            feature_amplitude=8.0, uncertainty_base=0.3, uncertainty_scale=0.03
        )
        
        # Continental Shelf - moderate slopes and features
        configs[SeafloorType.CONTINENTAL_SHELF] = BathymetryConfig(
            base_depth=-150.0, depth_range=100.0, spectral_slope=-2.0,
            roughness_scale=1.0, noise_level=0.15, num_features=3,
            feature_amplitude=20.0, uncertainty_base=0.5, uncertainty_scale=0.02
        )
        
        # Deep Ocean - steep slopes, major features
        configs[SeafloorType.DEEP_OCEAN] = BathymetryConfig(
            base_depth=-3000.0, depth_range=500.0, spectral_slope=-2.5,
            roughness_scale=0.8, noise_level=0.1, num_features=2,
            feature_amplitude=200.0, uncertainty_base=1.0, uncertainty_scale=0.01
        )
        
        # Seamount - volcanic cone features
        configs[SeafloorType.SEAMOUNT] = BathymetryConfig(
            base_depth=-2000.0, depth_range=1500.0, spectral_slope=-2.2,
            roughness_scale=1.2, noise_level=0.2, num_features=1,
            feature_amplitude=800.0, uncertainty_base=0.8, uncertainty_scale=0.015
        )
        
        # Abyssal Plain - very flat, minimal features
        configs[SeafloorType.ABYSSAL_PLAIN] = BathymetryConfig(
            base_depth=-5000.0, depth_range=20.0, spectral_slope=-3.0,
            roughness_scale=0.3, noise_level=0.05, num_features=0,
            feature_amplitude=5.0, uncertainty_base=0.2, uncertainty_scale=0.005
        )
        
        # Canyon - steep-sided valleys
        configs[SeafloorType.CANYON] = BathymetryConfig(
            base_depth=-1000.0, depth_range=800.0, spectral_slope=-1.8,
            roughness_scale=1.5, noise_level=0.25, num_features=1,
            feature_amplitude=400.0, uncertainty_base=1.2, uncertainty_scale=0.025
        )
        
        # Ridge - linear elevated features
        configs[SeafloorType.RIDGE] = BathymetryConfig(
            base_depth=-2500.0, depth_range=1000.0, spectral_slope=-2.3,
            roughness_scale=1.3, noise_level=0.18, num_features=2,
            feature_amplitude=500.0, uncertainty_base=0.9, uncertainty_scale=0.018
        )
        
        return configs
        
    def generate_bathymetry(self, seafloor_type: SeafloorType, 
                          custom_config: Optional[BathymetryConfig] = None,
                          survey_location: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate bathymetric depth and uncertainty data with realistic coordinates and noise."""
        
        # Get realistic survey location first
        if survey_location is None:
            survey_location = SurveyLocation.get_location_by_seafloor_type(seafloor_type)
        
        # Use custom config or seafloor-specific config
        if custom_config:
            config = custom_config
        else:
            # Start with seafloor-specific config but override with location depths
            config = BathymetryConfig(
                **{k: v for k, v in self.seafloor_configs.get(seafloor_type, self.config).__dict__.items()}
            )
        
        # ALWAYS override with location-specific coordinates and depths
        config.origin_x = survey_location['utm_x']
        config.origin_y = survey_location['utm_y'] 
        config.projection = survey_location['projection']
        
        # CRITICAL: Use location depth ranges, not seafloor type ranges
        location_depth_range = survey_location['depth_range']
        config.base_depth = (location_depth_range[0] + location_depth_range[1]) / 2
        config.depth_range = abs(location_depth_range[1] - location_depth_range[0])
            
        print(f"Generating {seafloor_type.value} bathymetry ({config.width}x{config.height})")
        print(f"Location: {survey_location['name']}")
        print(f"Coordinates: UTM {config.origin_x:.0f}E, {config.origin_y:.0f}N ({config.projection})")
        print(f"Target depth range: {location_depth_range[0]}m to {location_depth_range[1]}m")
        print(f"Base depth: {config.base_depth:.1f}m, Range: ±{config.depth_range/2:.1f}m")
        
        # Print enabled noise patterns
        noise_types = []
        if config.noise_config.enable_sidelobe:
            noise_types.append("sidelobe")
        if config.noise_config.enable_refraction:
            noise_types.append("refraction")
        if config.noise_config.enable_bubble_sweep:
            noise_types.append("bubble-sweep")
        
        if noise_types:
            print(f"Enabled noise patterns: {', '.join(noise_types)}")
        else:
            print("No acoustic noise patterns enabled")
        
        # Create coordinate grids
        x = np.arange(0, config.width * config.resolution, config.resolution)
        y = np.arange(0, config.height * config.resolution, config.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Generate fractal surface using spectral method
        depth_surface = self._generate_fractal_surface(config)
        
        # Add bathymetric features based on seafloor type
        depth_surface = self._add_bathymetric_features(depth_surface, X, Y, seafloor_type, config)
        
        # Apply base depth and scaling - ensure we stay within location bounds
        depth_data = config.base_depth + depth_surface * (config.depth_range / 2)
        
        # Store the original range before noise
        original_range = (np.min(depth_data), np.max(depth_data))
        print(f"Base bathymetry range: {original_range[0]:.1f}m to {original_range[1]:.1f}m")
        
        # Clamp to location depth range to ensure accuracy
        depth_data = np.clip(depth_data, location_depth_range[0], location_depth_range[1])
        clipped_range = (np.min(depth_data), np.max(depth_data))
        print(f"After initial clipping: {clipped_range[0]:.1f}m to {clipped_range[1]:.1f}m")
        
        # CRITICAL DEBUG: Let's track exactly what happens to the noise
        print(f"\n🔍 DEBUGGING NOISE APPLICATION:")
        print(f"1. Base bathymetry range: {np.min(depth_data):.3f} to {np.max(depth_data):.3f}m")
        
        # Store original data for comparison
        original_depth_data = depth_data.copy()
        
        # Add realistic acoustic noise patterns  
        depth_data = self._apply_acoustic_noise(depth_data, X, Y, config)
        
        # Check if noise was actually applied
        depth_change = depth_data - original_depth_data
        max_change = np.max(np.abs(depth_change))
        print(f"2. Maximum depth change from noise: {max_change:.6f}m")
        
        if max_change < 0.001:
            print("   ⚠️  WARNING: Noise application had minimal effect!")
            print("   🔧 Attempting to force apply noise with higher amplitude...")
            
            # Force apply noise with higher amplitude for testing
            if config.noise_config.enable_refraction:
                print("   🔧 Forcing refraction noise...")
                test_refraction = np.zeros_like(depth_data)
                for i in range(depth_data.shape[1]):
                    normalized_x = (i - depth_data.shape[1]/2) / (depth_data.shape[1]/2)
                    test_refraction[:, i] = 5.0 * (normalized_x ** 2)  # Very obvious 5m smile
                depth_data += test_refraction
                print(f"      Applied test refraction: {np.min(test_refraction):.3f} to {np.max(test_refraction):.3f}m")
            
            if config.noise_config.enable_sidelobe:
                print("   🔧 Forcing sidelobe noise...")
                test_sidelobe = np.zeros_like(depth_data)
                for i in range(depth_data.shape[1]):
                    test_sidelobe[:, i] = 2.0 * np.sin(2 * np.pi * i / 20)  # Very obvious 2m stripes
                depth_data += test_sidelobe  
                print(f"      Applied test sidelobe: {np.min(test_sidelobe):.3f} to {np.max(test_sidelobe):.3f}m")
        
        print(f"3. After noise application: {np.min(depth_data):.3f} to {np.max(depth_data):.3f}m")
        
        # Check range after noise
        noisy_range = (np.min(depth_data), np.max(depth_data))
        print(f"After adding noise: {noisy_range[0]:.1f}m to {noisy_range[1]:.1f}m")
        
        # Add random noise
        if config.noise_level > 0:
            # Scale noise to be smaller to avoid going outside depth range
            noise_scale = min(config.noise_level * config.depth_range, config.depth_range * 0.1)
            noise = np.random.normal(0, noise_scale, depth_data.shape)
            depth_data += noise
            random_noise_range = (np.min(depth_data), np.max(depth_data))
            print(f"After random noise: {random_noise_range[0]:.1f}m to {random_noise_range[1]:.1f}m")
            
        # IMPORTANT: Only clamp if we're going significantly outside bounds
        # Allow some overshoot to preserve noise patterns
        buffer_margin = max(2.0, config.depth_range * 0.1)  # 2m or 10% of range, whichever is larger
        extended_min = location_depth_range[0] - buffer_margin
        extended_max = location_depth_range[1] + buffer_margin
        
        original_extent = (np.min(depth_data), np.max(depth_data))
        
        # Only clip if we're way outside the expected range
        if np.min(depth_data) < extended_min or np.max(depth_data) > extended_max:
            print(f"⚠️  Depth data extends beyond buffer margin ({extended_min:.1f} to {extended_max:.1f}m)")
            print(f"    Actual range: {original_extent[0]:.1f} to {original_extent[1]:.1f}m")
            print(f"    Applying loose clipping...")
            
            # Use broadcasting-safe clipping
            try:
                depth_data_clipped = np.clip(depth_data, extended_min, extended_max)
                depth_data = depth_data_clipped.astype(np.float32)  # Ensure consistent type
                final_clipped_range = (np.min(depth_data), np.max(depth_data))
                print(f"    After loose clipping: {final_clipped_range[0]:.1f}m to {final_clipped_range[1]:.1f}m")
            except Exception as e:
                print(f"    Error in clipping: {e}")
                print(f"    Skipping clipping to avoid broadcasting issues...")
        else:
            print(f"✅ Depth data within acceptable range - preserving noise patterns")
            
        # Generate uncertainty data (enhanced with noise effects)
        print(f"Generating uncertainty data...")
        try:
            uncertainty_data = self._generate_uncertainty(depth_data, config)
            print(f"✅ Uncertainty generated successfully: {uncertainty_data.shape}")
        except Exception as e:
            print(f"❌ Error generating uncertainty: {e}")
            print(f"   Creating fallback uncertainty...")
            uncertainty_data = np.full_like(depth_data, 1.0, dtype=np.float32)
        
        # Ensure proper data types
        print(f"Ensuring proper data types...")
        try:
            depth_data = depth_data.astype(np.float32)
            uncertainty_data = uncertainty_data.astype(np.float32)
            print(f"✅ Data types converted successfully")
            print(f"   Depth shape: {depth_data.shape}, dtype: {depth_data.dtype}")
            print(f"   Uncertainty shape: {uncertainty_data.shape}, dtype: {uncertainty_data.dtype}")
        except Exception as e:
            print(f"❌ Error converting data types: {e}")
            raise
        
        # Verify final depth range
        actual_min = np.min(depth_data)
        actual_max = np.max(depth_data)
        print(f"Actual depth range: {actual_min:.1f}m to {actual_max:.1f}m")
        
        return depth_data, uncertainty_data
    
    def _apply_acoustic_noise(self, depth_data: np.ndarray, X: np.ndarray, Y: np.ndarray, 
                            config: BathymetryConfig) -> np.ndarray:
        """Apply realistic acoustic noise patterns to bathymetric data."""
        
        noisy_depth = depth_data.copy()
        height, width = depth_data.shape
        
        # Calculate approximate water depth for scaling noise effects
        mean_depth = abs(np.mean(depth_data))
        
        print(f"  Starting noise application:")
        print(f"    Input depth range: {np.min(depth_data):.3f} to {np.max(depth_data):.3f}m")
        print(f"    Mean depth: {mean_depth:.1f}m")
        
        # Apply sidelobe noise
        if config.noise_config.enable_sidelobe:
            print("  Adding sidelobe noise pattern...")
            sidelobe_noise = self._generate_sidelobe_noise(width, height, config.noise_config, mean_depth)
            print(f"    Sidelobe noise range: {np.min(sidelobe_noise):.3f} to {np.max(sidelobe_noise):.3f}m")
            
            # Show before/after for a sample line
            sample_line = height // 2
            before_sample = noisy_depth[sample_line, width//4:3*width//4]
            noisy_depth += sidelobe_noise
            after_sample = noisy_depth[sample_line, width//4:3*width//4]
            print(f"    Sample line before: {np.mean(before_sample):.3f}±{np.std(before_sample):.3f}m")
            print(f"    Sample line after:  {np.mean(after_sample):.3f}±{np.std(after_sample):.3f}m")
            
        # Apply sound speed refraction noise
        if config.noise_config.enable_refraction:
            print("  Adding sound speed refraction noise...")
            refraction_noise = self._generate_refraction_noise(width, height, config.noise_config, mean_depth)
            print(f"    Refraction noise range: {np.min(refraction_noise):.3f} to {np.max(refraction_noise):.3f}m")
            
            # Show cross-track pattern
            center_line = refraction_noise[height//2, :]
            print(f"    Cross-track pattern: left={center_line[0]:.3f}, center={center_line[width//2]:.3f}, right={center_line[-1]:.3f}")
            
            before_range = (np.min(noisy_depth), np.max(noisy_depth))
            noisy_depth += refraction_noise
            after_range = (np.min(noisy_depth), np.max(noisy_depth))
            print(f"    Depth before refraction: {before_range[0]:.3f} to {before_range[1]:.3f}m")
            print(f"    Depth after refraction:  {after_range[0]:.3f} to {after_range[1]:.3f}m")
            
        # Apply bubble sweep down noise
        if config.noise_config.enable_bubble_sweep:
            print("  Adding bubble sweep down noise...")
            bubble_noise = self._generate_bubble_sweep_noise(X, Y, config.noise_config, mean_depth)
            print(f"    Bubble noise range: {np.min(bubble_noise):.3f} to {np.max(bubble_noise):.3f}m")
            
            before_range = (np.min(noisy_depth), np.max(noisy_depth))
            noisy_depth += bubble_noise
            after_range = (np.min(noisy_depth), np.max(noisy_depth))
            print(f"    Depth before bubbles: {before_range[0]:.3f} to {before_range[1]:.3f}m")
            print(f"    Depth after bubbles:  {after_range[0]:.3f} to {after_range[1]:.3f}m")
            
        print(f"  Final noisy depth range: {np.min(noisy_depth):.3f} to {np.max(noisy_depth):.3f}m")
        
        return noisy_depth
    
    def _generate_sidelobe_noise(self, width: int, height: int, noise_config: NoiseConfig, 
                               mean_depth: float) -> np.ndarray:
        """Generate sidelobe noise pattern - regular striping perpendicular to survey tracks."""
        
        print(f"    🔧 Generating sidelobe noise: {width}x{height}")
        
        # SIMPLIFIED: Just create obvious stripes
        sidelobe_pattern = np.zeros((height, width))
        
        for x in range(width):
            # Simple sinusoidal pattern across the swath
            stripe_value = noise_config.sidelobe_amplitude * np.sin(2 * np.pi * noise_config.sidelobe_frequency * x / width)
            
            # Add outer beam enhancement (stronger at edges)
            distance_from_center = abs(x - width/2) / (width/2)
            enhancement = 1 + (noise_config.sidelobe_outer_beam_factor - 1) * distance_from_center
            stripe_value *= enhancement
            
            # Apply to entire column (all y values for this x)
            sidelobe_pattern[:, x] = stripe_value
        
        print(f"    Sidelobe pattern range: {np.min(sidelobe_pattern):.6f} to {np.max(sidelobe_pattern):.6f}m")
        
        if np.max(np.abs(sidelobe_pattern)) < 0.001:
            print("    ⚠️  Sidelobe pattern is too small! Using fallback pattern...")
            # Create obvious fallback pattern
            for x in range(width):
                stripe_value = 1.0 * np.sin(2 * np.pi * x / 20)  # 1m amplitude, 20 pixel period
                sidelobe_pattern[:, x] = stripe_value
            print(f"    Fallback pattern range: {np.min(sidelobe_pattern):.6f} to {np.max(sidelobe_pattern):.6f}m")
        
        return sidelobe_pattern
    
    def _generate_refraction_noise(self, width: int, height: int, noise_config: NoiseConfig,
                                 mean_depth: float) -> np.ndarray:
        """Generate sound speed refraction noise - smile/frown patterns and asymmetric bias."""
        
        print(f"    🔧 Generating refraction noise: {width}x{height}")
        
        # SIMPLIFIED: Just create obvious smile pattern
        refraction_noise = np.zeros((height, width))
        
        for x in range(width):
            # Create simple U-shape (smile) across swath
            normalized_x = (x - width/2) / (width/2)  # -1 to 1 across swath
            smile_value = noise_config.refraction_smile_amplitude * (normalized_x ** 2)
            
            # Add simple asymmetry (tilt)
            if noise_config.refraction_asymmetry > 0:
                asymmetric_value = noise_config.refraction_asymmetry * noise_config.refraction_smile_amplitude * normalized_x
                smile_value += asymmetric_value
            
            # Apply to entire column (all y values for this x)
            refraction_noise[:, x] = smile_value
        
        print(f"    Base smile amplitude: {noise_config.refraction_smile_amplitude:.6f}m")
        print(f"    Refraction pattern range: {np.min(refraction_noise):.6f} to {np.max(refraction_noise):.6f}m")
        
        # Check cross-track pattern
        center_line = refraction_noise[height//2, :]
        print(f"    Cross-track test: left={center_line[0]:.6f}, center={center_line[width//2]:.6f}, right={center_line[-1]:.6f}")
        
        if np.max(np.abs(refraction_noise)) < 0.001:
            print("    ⚠️  Refraction pattern is too small! Using fallback pattern...")
            # Create obvious fallback pattern
            for x in range(width):
                normalized_x = (x - width/2) / (width/2)
                refraction_noise[:, x] = 3.0 * (normalized_x ** 2)  # 3m amplitude smile
            print(f"    Fallback pattern range: {np.min(refraction_noise):.6f} to {np.max(refraction_noise):.6f}m")
        
        return refraction_noise
    
    def _generate_bubble_sweep_noise(self, X: np.ndarray, Y: np.ndarray, noise_config: NoiseConfig,
                                   mean_depth: float) -> np.ndarray:
        """Generate bubble sweep down noise - completely safe implementation with no broadcasting."""
        
        height, width = X.shape
        
        # Create output array with explicit dtype to prevent any type issues
        bubble_noise = np.zeros((height, width), dtype=np.float32)
        
        print(f"    🔧 Generating safe bubble sweep noise: {width}x{height}")
        
        # Very conservative approach - absolutely no vectorized operations
        # that could cause broadcasting issues
        
        # Limit to very few survey lines for safety
        max_lines = min(3, max(1, width // 500))  # Very conservative
        
        print(f"    Ultra-safe configuration: {max_lines} survey lines")
        
        try:
            # Process each survey line individually
            for line_idx in range(max_lines):
                line_x = int((line_idx + 0.5) * width / max_lines)
                
                print(f"    Processing line {line_idx + 1}/{max_lines} at x={line_x}")
                
                # Very simple wake parameters
                wake_width = min(10, width // 100)  # Very narrow
                wake_length = min(20, height // 50)  # Very short
                
                # Process one pixel at a time - completely safe
                for y in range(0, height, 5):  # Skip every 5th row for speed
                    for wake_y_offset in range(0, min(wake_length, height - y), 2):  # Skip every 2nd for speed
                        wake_y = y + wake_y_offset
                        
                        if wake_y >= height:
                            break
                            
                        # Calculate wake strength (simple exponential decay)
                        if wake_length > 0:
                            wake_strength = np.exp(-wake_y_offset / max(1, wake_length / 3))
                        else:
                            wake_strength = 1.0
                            
                        # Add lateral spread around the survey line
                        for dx in range(-wake_width, wake_width + 1, 2):  # Skip every 2nd for speed
                            x = line_x + dx
                            
                            if 0 <= x < width:
                                # Simple Gaussian lateral spread
                                if wake_width > 0:
                                    lateral_strength = np.exp(-(dx * dx) / max(1, (wake_width / 2) ** 2))
                                else:
                                    lateral_strength = 1.0 if dx == 0 else 0.0
                                
                                # Calculate final bubble effect
                                bubble_effect = float(noise_config.bubble_sweep_amplitude * 
                                                    wake_strength * lateral_strength * 0.3)
                                
                                # Add small random variation
                                bubble_effect *= (0.8 + 0.4 * np.random.random())
                                
                                # Safely assign to array (no broadcasting here!)
                                current_value = float(bubble_noise[wake_y, x])
                                bubble_noise[wake_y, x] = np.float32(current_value + bubble_effect)
            
            # Apply depth scaling using scalar operations only
            depth_factor = float(max(0.2, 50.0 / max(10.0, mean_depth)))
            
            # Scale the entire array safely
            bubble_noise = bubble_noise * np.float32(depth_factor)
            
            # Add one simple patch to represent random bubble effects
            if width > 50 and height > 50:
                patch_x = width // 2
                patch_y = height // 2
                patch_radius = min(8, width // 200)
                patch_intensity = float(noise_config.bubble_sweep_amplitude * 0.1)
                
                for dy in range(-patch_radius, patch_radius + 1):
                    for dx in range(-patch_radius, patch_radius + 1):
                        py = patch_y + dy
                        px = patch_x + dx
                        
                        if 0 <= py < height and 0 <= px < width:
                            distance = np.sqrt(float(dx * dx + dy * dy))
                            if distance <= patch_radius and distance > 0:
                                patch_strength = np.exp(-distance * distance / max(1, patch_radius * patch_radius))
                                patch_effect = float(patch_intensity * patch_strength)
                                
                                current_value = float(bubble_noise[py, px])
                                bubble_noise[py, px] = np.float32(current_value + patch_effect)
            
            print(f"    Safe bubble generation complete: {np.min(bubble_noise):.3f} to {np.max(bubble_noise):.3f}m")
            
        except Exception as e:
            print(f"    ⚠️  Error in bubble generation: {e}")
            print(f"    Using fallback: simple uniform pattern")
            
            # Ultimate fallback - just create a simple pattern
            try:
                for i in range(0, width, 100):  # Every 100 pixels
                    for j in range(0, height, 50):  # Every 50 pixels
                        if i < width and j < height:
                            bubble_noise[j, i] = np.float32(noise_config.bubble_sweep_amplitude * 0.1)
            except:
                # If even this fails, return zeros
                bubble_noise = np.zeros((height, width), dtype=np.float32)
                print(f"    Using zero fallback")
        
        return bubble_noise
        
    def _generate_fractal_surface(self, config: BathymetryConfig) -> np.ndarray:
        """Generate fractal surface using spectral synthesis (red noise)."""
        
        # Create frequency grids
        fx = np.fft.fftfreq(config.width, d=config.resolution)
        fy = np.fft.fftfreq(config.height, d=config.resolution)
        FX, FY = np.meshgrid(fx, fy)
        
        # Calculate radial frequency
        FR = np.sqrt(FX**2 + FY**2)
        
        # Avoid division by zero
        FR[FR == 0] = 1e-10
        
        # Generate amplitude spectrum with specified slope
        # Power law: S(f) = f^spectral_slope
        amplitude = np.power(FR, config.spectral_slope/2.0)
        
        # Scale amplitude
        amplitude *= config.roughness_scale * (config.width * config.resolution)
        
        # Generate random phases
        phases = np.random.uniform(0, 2*np.pi, amplitude.shape)
        
        # Create complex spectrum
        spectrum = amplitude * np.exp(1j * phases)
        
        # Ensure Hermitian symmetry for real output
        spectrum = self._ensure_hermitian_symmetry(spectrum)
        
        # Inverse FFT to get spatial domain
        surface = np.real(np.fft.ifft2(spectrum))
        
        # Normalize to unit variance
        surface = (surface - np.mean(surface)) / (np.std(surface) + 1e-10)
        
        return surface
        
    def _ensure_hermitian_symmetry(self, spectrum: np.ndarray) -> np.ndarray:
        """Ensure Hermitian symmetry for real-valued inverse FFT."""
        height, width = spectrum.shape
        
        # Handle DC component
        spectrum[0, 0] = np.real(spectrum[0, 0])
        
        # Handle Nyquist frequencies
        if height % 2 == 0:
            spectrum[height//2, 0] = np.real(spectrum[height//2, 0])
        if width % 2 == 0:
            spectrum[0, width//2] = np.real(spectrum[0, width//2])
        if height % 2 == 0 and width % 2 == 0:
            spectrum[height//2, width//2] = np.real(spectrum[height//2, width//2])
            
        # Enforce conjugate symmetry
        for i in range(1, height):
            for j in range(1, width):
                spectrum[i, j] = np.conj(spectrum[height-i, width-j])
                
        return spectrum
        
    def _add_bathymetric_features(self, base_surface: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                seafloor_type: SeafloorType, config: BathymetryConfig) -> np.ndarray:
        """Add specific bathymetric features based on seafloor type."""
        
        surface = base_surface.copy()
        
        if seafloor_type == SeafloorType.SEAMOUNT:
            # Add volcanic cone
            center_x, center_y = X.shape[1]//2, X.shape[0]//2
            radius = min(X.shape) // 4
            cone = self._create_seamount(X, Y, center_x, center_y, radius, config.feature_amplitude)
            surface += cone
            
        elif seafloor_type == SeafloorType.CANYON:
            # Add canyon feature
            canyon = self._create_canyon(X, Y, config.feature_amplitude)
            surface += canyon
            
        elif seafloor_type == SeafloorType.RIDGE:
            # Add ridge feature
            ridge = self._create_ridge(X, Y, config.feature_amplitude)
            surface += ridge
            
        elif seafloor_type == SeafloorType.SHALLOW_COASTAL:
            # Add channels and sandbars
            for i in range(config.num_features):
                feature = self._create_coastal_feature(X, Y, i, config.feature_amplitude)
                surface += feature
                
        elif seafloor_type in [SeafloorType.CONTINENTAL_SHELF, SeafloorType.DEEP_OCEAN]:
            # Add random features
            for i in range(config.num_features):
                feature = self._create_random_feature(X, Y, i, config.feature_amplitude)
                surface += feature
                
        return surface
        
    def _create_seamount(self, X: np.ndarray, Y: np.ndarray, 
                        center_x: float, center_y: float, radius: float, amplitude: float) -> np.ndarray:
        """Create a seamount (volcanic cone) feature."""
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        cone = amplitude * np.exp(-(distance / radius)**2) * np.cos(np.pi * distance / (2 * radius))
        cone[distance > radius] = 0
        return cone
        
    def _create_canyon(self, X: np.ndarray, Y: np.ndarray, amplitude: float) -> np.ndarray:
        """Create a canyon feature."""
        center_y = Y.shape[0] // 2
        width = Y.shape[0] // 8
        canyon = -amplitude * np.exp(-((Y - center_y) / width)**2)
        
        # Add meandering
        meander = 10 * np.sin(X / 50)
        canyon_shifted = np.zeros_like(canyon)
        
        for i in range(X.shape[1]):
            shift = int(meander[0, i])
            canyon_shifted[:, i] = np.roll(canyon[:, i], shift)
            
        return canyon_shifted
        
    def _create_ridge(self, X: np.ndarray, Y: np.ndarray, amplitude: float) -> np.ndarray:
        """Create a ridge feature."""
        center_y = Y.shape[0] // 2
        width = Y.shape[0] // 6
        ridge = amplitude * np.exp(-((Y - center_y) / width)**2)
        
        # Add variations along ridge
        variations = 0.3 * amplitude * np.sin(X / 30)
        ridge += variations
        
        return ridge
        
    def _create_coastal_feature(self, X: np.ndarray, Y: np.ndarray, 
                              feature_index: int, amplitude: float) -> np.ndarray:
        """Create coastal features like channels or sandbars."""
        feature = np.zeros_like(X)
        
        if feature_index % 2 == 0:
            # Channel
            y_pos = (feature_index + 1) * Y.shape[0] // (6)
            width = 20
            depth = -amplitude * 0.5
            feature[max(0, y_pos-width):min(Y.shape[0], y_pos+width), :] = depth
        else:
            # Sandbar
            y_pos = (feature_index + 1) * Y.shape[0] // (6)
            width = 15
            height = amplitude * 0.3
            feature[max(0, y_pos-width):min(Y.shape[0], y_pos+width), :] = height
            
        return feature
        
    def _create_random_feature(self, X: np.ndarray, Y: np.ndarray,
                             feature_index: int, amplitude: float) -> np.ndarray:
        """Create random bathymetric features."""
        # Random center position
        center_x = np.random.uniform(0.2, 0.8) * X.shape[1]
        center_y = np.random.uniform(0.2, 0.8) * X.shape[0]
        
        # Random size
        radius = np.random.uniform(20, 60)
        
        # Random amplitude (can be positive or negative)
        feat_amplitude = amplitude * np.random.uniform(-1, 1)
        
        # Create feature
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        feature = feat_amplitude * np.exp(-(distance / radius)**2)
        
        return feature
        
    def _generate_uncertainty(self, depth_data: np.ndarray, config: BathymetryConfig) -> np.ndarray:
        """Generate realistic uncertainty data with noise-dependent adjustments."""
        
        print(f"  Generating uncertainty data for {depth_data.shape} grid...")
        
        # Ensure we're working with the right shape and type
        height, width = depth_data.shape
        
        # Base uncertainty - create with explicit shape and type
        uncertainty = np.full((height, width), config.uncertainty_base, dtype=np.float32)
        
        # Depth-dependent uncertainty - use broadcasting-safe operations
        depth_abs = np.abs(depth_data)
        depth_uncertainty = depth_abs * config.uncertainty_scale
        uncertainty = uncertainty + depth_uncertainty.astype(np.float32)
        
        # Random variations - create array with same shape
        random_factors = np.random.uniform(0.5, 1.5, size=(height, width)).astype(np.float32)
        random_uncertainty = uncertainty * random_factors
        
        # Slope-dependent uncertainty - use broadcasting-safe gradient calculation
        try:
            gy, gx = np.gradient(depth_data)
            slope = np.sqrt(gx**2 + gy**2)
            slope_uncertainty = slope * 0.1
            slope_uncertainty = slope_uncertainty.astype(np.float32)
        except Exception as e:
            print(f"    Warning: Gradient calculation failed: {e}")
            slope_uncertainty = np.zeros((height, width), dtype=np.float32)
        
        # Noise-dependent uncertainty adjustments - handle each type separately
        noise_uncertainty_adjustment = np.zeros((height, width), dtype=np.float32)
        
        try:
            if config.noise_config.enable_sidelobe:
                # Sidelobe noise increases uncertainty in outer beams
                x_coords = np.arange(width, dtype=np.float32)
                center = float(width // 2)
                beam_distance = np.abs(x_coords - center) / center
                
                # Create 2D array by broadcasting safely
                sidelobe_uncertainty = np.zeros((height, width), dtype=np.float32)
                sidelobe_amplitude = float(config.noise_config.sidelobe_amplitude * 0.5)
                
                for i in range(width):
                    beam_dist = float(beam_distance[i])
                    uncertainty_value = sidelobe_amplitude * beam_dist
                    sidelobe_uncertainty[:, i] = uncertainty_value
                
                noise_uncertainty_adjustment += sidelobe_uncertainty
                
        except Exception as e:
            print(f"    Warning: Sidelobe uncertainty calculation failed: {e}")
        
        try:
            if config.noise_config.enable_refraction:
                # Refraction errors increase uncertainty across swath
                refraction_uncertainty_value = float(config.noise_config.refraction_smile_amplitude * 0.3)
                refraction_uncertainty = np.full((height, width), refraction_uncertainty_value, dtype=np.float32)
                noise_uncertainty_adjustment += refraction_uncertainty
                
        except Exception as e:
            print(f"    Warning: Refraction uncertainty calculation failed: {e}")
        
        try:
            if config.noise_config.enable_bubble_sweep:
                # Bubble noise creates patchy uncertainty increases
                bubble_base = float(config.noise_config.bubble_sweep_amplitude * 0.4)
                bubble_random = np.random.uniform(0.5, 1.5, size=(height, width)).astype(np.float32)
                bubble_uncertainty = bubble_base * bubble_random
                noise_uncertainty_adjustment += bubble_uncertainty
                
        except Exception as e:
            print(f"    Warning: Bubble uncertainty calculation failed: {e}")
        
        # Combine all uncertainty components safely
        try:
            total_uncertainty = (uncertainty + random_uncertainty + 
                               slope_uncertainty + noise_uncertainty_adjustment)
            
            # Ensure reasonable bounds
            total_uncertainty = np.clip(total_uncertainty, 0.1, 10.0)
            
            # Ensure output is float32
            total_uncertainty = total_uncertainty.astype(np.float32)
            
        except Exception as e:
            print(f"    Error in uncertainty combination: {e}")
            print(f"    Using fallback uncertainty...")
            total_uncertainty = np.full((height, width), 1.0, dtype=np.float32)
        
        print(f"  Uncertainty range: {np.min(total_uncertainty):.3f} to {np.max(total_uncertainty):.3f}m")
        
        return total_uncertainty
        
    def save_as_bag(self, depth_data: np.ndarray, uncertainty_data: np.ndarray,
                   output_path: Path, config: BathymetryConfig,
                   metadata: Optional[Dict] = None, survey_location: Optional[Dict] = None) -> bool:
        """Save data as BAG file format with realistic coordinates."""
        
        print(f"\n💾 DEBUGGING BAG SAVE PROCESS:")
        print(f"   Input depth range: {np.min(depth_data):.6f} to {np.max(depth_data):.6f}m")
        print(f"   Input data type: {depth_data.dtype}")
        print(f"   Input shape: {depth_data.shape}")
        
        # Check for obvious patterns before saving
        height, width = depth_data.shape
        if width > 10:
            center_row = depth_data[height//2, :]
            cross_track_std = np.std(center_row)
            print(f"   Cross-track variation: {cross_track_std:.6f}m")
            
            # Sample a few specific points
            left = center_row[0]
            center = center_row[width//2] 
            right = center_row[-1]
            print(f"   Sample points: left={left:.6f}, center={center:.6f}, right={right:.6f}")
        
        try:
            driver = gdal.GetDriverByName('BAG')
            if driver is None:
                print("Warning: BAG driver not available")
                return False
                
            height, width = depth_data.shape
            
            # CRITICAL: Ensure data is proper float32
            depth_data_f32 = depth_data.astype(np.float32)
            uncertainty_data_f32 = uncertainty_data.astype(np.float32)
            
            print(f"   Converting to float32...")
            print(f"   New depth range: {np.min(depth_data_f32):.6f} to {np.max(depth_data_f32):.6f}m")
            
            dataset = driver.Create(str(output_path), width, height, 2, gdal.GDT_Float32)
            
            # Set realistic geospatial information
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
            
            # Add comprehensive metadata including location info and noise patterns
            if metadata:
                for key, value in metadata.items():
                    dataset.SetMetadataItem(key, str(value))
            
            if survey_location:
                location_metadata = {
                    'SURVEY_LOCATION': survey_location['name'],
                    'SURVEY_DESCRIPTION': survey_location['description'],
                    'UTM_ZONE': config.projection,
                    'ORIGIN_UTM_X': f"{config.origin_x:.0f}",
                    'ORIGIN_UTM_Y': f"{config.origin_y:.0f}",
                    'COVERAGE_EXTENT_M': f"{width * config.resolution:.0f} x {height * config.resolution:.0f}",
                    'COORDINATE_SYSTEM': 'UTM (Universal Transverse Mercator)'
                }
                for key, value in location_metadata.items():
                    dataset.SetMetadataItem(key, value, 'LOCATION')
            
            # Add noise pattern metadata
            noise_metadata = {
                'SIDELOBE_NOISE': 'TRUE' if config.noise_config.enable_sidelobe else 'FALSE',
                'REFRACTION_NOISE': 'TRUE' if config.noise_config.enable_refraction else 'FALSE',
                'BUBBLE_SWEEP_NOISE': 'TRUE' if config.noise_config.enable_bubble_sweep else 'FALSE'
            }
            
            if config.noise_config.enable_sidelobe:
                noise_metadata.update({
                    'SIDELOBE_AMPLITUDE_M': f"{config.noise_config.sidelobe_amplitude:.3f}",
                    'SIDELOBE_FREQUENCY': f"{config.noise_config.sidelobe_frequency:.1f}",
                    'SIDELOBE_OUTER_BEAM_FACTOR': f"{config.noise_config.sidelobe_outer_beam_factor:.1f}"
                })
                
            if config.noise_config.enable_refraction:
                noise_metadata.update({
                    'REFRACTION_SMILE_AMPLITUDE_M': f"{config.noise_config.refraction_smile_amplitude:.3f}",
                    'REFRACTION_ASYMMETRY': f"{config.noise_config.refraction_asymmetry:.3f}",
                    'REFRACTION_TEMPORAL_DRIFT_M': f"{config.noise_config.refraction_temporal_drift:.3f}"
                })
                
            if config.noise_config.enable_bubble_sweep:
                noise_metadata.update({
                    'BUBBLE_SWEEP_AMPLITUDE_M': f"{config.noise_config.bubble_sweep_amplitude:.3f}",
                    'BUBBLE_PLUME_WIDTH_FACTOR': f"{config.noise_config.bubble_plume_width:.3f}",
                    'BUBBLE_WAKE_LENGTH_M': f"{config.noise_config.bubble_wake_length:.1f}"
                })
            
            for key, value in noise_metadata.items():
                dataset.SetMetadataItem(key, value, 'ACOUSTIC_NOISE')
            
            # Force everything to disk
            dataset.FlushCache()
            
            # Close dataset
            dataset = None
            
            print(f"   ✅ BAG file written: {output_path}")
            
            # CRITICAL: Immediately verify by reading back
            print(f"   🔍 Verifying written file...")
            verify_dataset = gdal.Open(str(output_path), gdal.GA_ReadOnly)
            if verify_dataset is None:
                print(f"   ❌ Cannot read back written file!")
                return False
            
            verify_band = verify_dataset.GetRasterBand(1)
            verify_data = verify_band.ReadAsArray()
            verify_dataset = None
            
            print(f"   Verified depth range: {np.min(verify_data):.6f} to {np.max(verify_data):.6f}m")
            print(f"   Data preservation: {np.array_equal(depth_data_f32, verify_data)}")
            
            if not np.allclose(depth_data_f32, verify_data, rtol=1e-6):
                print(f"   ⚠️  Warning: Written data differs from input!")
                max_diff = np.max(np.abs(depth_data_f32 - verify_data))
                print(f"   Maximum difference: {max_diff:.6f}m")
            else:
                print(f"   ✅ Data preserved perfectly!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving BAG file: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def save_as_geotiff(self, depth_data: np.ndarray, output_path: Path,
                       config: BathymetryConfig, metadata: Optional[Dict] = None) -> bool:
        """Save depth data as GeoTIFF format."""
        try:
            driver = gdal.GetDriverByName('GTiff')
            height, width = depth_data.shape
            dataset = driver.Create(str(output_path), width, height, 1, gdal.GDT_Float32)
            
            # Set geospatial information
            geotransform = [
                config.origin_x, config.resolution, 0,
                config.origin_y + (height * config.resolution), 0, -config.resolution
            ]
            dataset.SetGeoTransform(geotransform)
            dataset.SetProjection(config.projection)
            
            # Write depth data
            band = dataset.GetRasterBand(1)
            band.WriteArray(depth_data)
            band.SetDescription('Depth')
            band.SetNoDataValue(-9999)
            
            # Add metadata
            if metadata:
                dataset.SetMetadata(metadata)
                
            # Close dataset
            dataset = None
            print(f"✅ GeoTIFF file saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving GeoTIFF: {e}")
            return False
            
    def save_as_xyz(self, depth_data: np.ndarray, output_path: Path,
                   config: BathymetryConfig) -> bool:
        """Save data as XYZ text format."""
        try:
            height, width = depth_data.shape
            
            # Create coordinate arrays
            x = np.arange(config.origin_x, config.origin_x + width * config.resolution, config.resolution)
            y = np.arange(config.origin_y, config.origin_y + height * config.resolution, config.resolution)
            X, Y = np.meshgrid(x, y)
            
            # Flatten arrays
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = depth_data.flatten()
            
            # Save to file
            data = np.column_stack((x_flat, y_flat, z_flat))
            np.savetxt(output_path, data, fmt='%.3f %.3f %.3f', 
                      header='X Y Z (Easting Northing Depth)')
            
            print(f"✅ XYZ file saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving XYZ file: {e}")
            return False
            
    def create_visualization(self, depth_data: np.ndarray, uncertainty_data: np.ndarray,
                           output_path: Path, seafloor_type: SeafloorType, 
                           config: BathymetryConfig) -> bool:
        """Create comprehensive visualization plots of the generated data including noise analysis."""
        try:
            # Determine number of subplots based on enabled noise patterns
            num_noise_patterns = sum([
                config.noise_config.enable_sidelobe,
                config.noise_config.enable_refraction,
                config.noise_config.enable_bubble_sweep
            ])
            
            if num_noise_patterns > 0:
                fig, axes = plt.subplots(3, 3, figsize=(18, 15))
                fig.suptitle(f'Synthetic Bathymetric Data with Acoustic Noise - {seafloor_type.value.title()}', fontsize=16)
            else:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'Synthetic Bathymetric Data - {seafloor_type.value.title()}', fontsize=14)
            
            # Always show these basic plots
            if num_noise_patterns > 0:
                ax_depth = axes[0, 0]
                ax_uncertainty = axes[0, 1]
                ax_hist = axes[1, 0]
                ax_scatter = axes[1, 1]
            else:
                ax_depth = axes[0, 0]
                ax_uncertainty = axes[0, 1]
                ax_hist = axes[1, 0]
                ax_scatter = axes[1, 1]
            
            # Depth data
            im1 = ax_depth.imshow(depth_data, cmap='viridis', origin='lower')
            ax_depth.set_title('Depth Data (m)')
            ax_depth.set_xlabel('Easting')
            ax_depth.set_ylabel('Northing')
            plt.colorbar(im1, ax=ax_depth)
            
            # Uncertainty data
            im2 = ax_uncertainty.imshow(uncertainty_data, cmap='plasma', origin='lower')
            ax_uncertainty.set_title('Uncertainty Data (m)')
            ax_uncertainty.set_xlabel('Easting')
            ax_uncertainty.set_ylabel('Northing')
            plt.colorbar(im2, ax=ax_uncertainty)
            
            # Depth histogram
            ax_hist.hist(depth_data.flatten(), bins=50, alpha=0.7, color='blue')
            ax_hist.set_title('Depth Distribution')
            ax_hist.set_xlabel('Depth (m)')
            ax_hist.set_ylabel('Frequency')
            ax_hist.grid(True, alpha=0.3)
            
            # Uncertainty vs Depth
            sample_indices = np.random.choice(depth_data.size, 5000, replace=False)
            depth_sample = depth_data.flatten()[sample_indices]
            uncertainty_sample = uncertainty_data.flatten()[sample_indices]
            
            ax_scatter.scatter(depth_sample, uncertainty_sample, alpha=0.5, s=1)
            ax_scatter.set_title('Uncertainty vs Depth')
            ax_scatter.set_xlabel('Depth (m)')
            ax_scatter.set_ylabel('Uncertainty (m)')
            ax_scatter.grid(True, alpha=0.3)
            
            # Add noise-specific visualizations if patterns are enabled
            if num_noise_patterns > 0:
                noise_col = 2
                
                # Cross-track depth profile to show noise patterns
                center_line = depth_data.shape[0] // 2
                cross_track_profile = depth_data[center_line, :]
                axes[0, noise_col].plot(cross_track_profile)
                axes[0, noise_col].set_title('Cross-Track Depth Profile\n(Center Line)')
                axes[0, noise_col].set_xlabel('Across-Track Distance')
                axes[0, noise_col].set_ylabel('Depth (m)')
                axes[0, noise_col].grid(True, alpha=0.3)
                
                # Noise pattern analysis
                noise_row = 2
                if config.noise_config.enable_sidelobe:
                    # Show sidelobe striping pattern
                    sidelobe_demo = self._generate_sidelobe_noise(
                        depth_data.shape[1], depth_data.shape[0], 
                        config.noise_config, abs(np.mean(depth_data))
                    )
                    im_sidelobe = axes[noise_row, 0].imshow(sidelobe_demo, cmap='RdBu', origin='lower')
                    axes[noise_row, 0].set_title('Sidelobe Noise Pattern')
                    plt.colorbar(im_sidelobe, ax=axes[noise_row, 0])
                    
                if config.noise_config.enable_refraction:
                    # Show refraction smile/frown pattern
                    refraction_demo = self._generate_refraction_noise(
                        depth_data.shape[1], depth_data.shape[0],
                        config.noise_config, abs(np.mean(depth_data))
                    )
                    im_refraction = axes[noise_row, 1].imshow(refraction_demo, cmap='RdBu', origin='lower')
                    axes[noise_row, 1].set_title('Refraction Noise Pattern')
                    plt.colorbar(im_refraction, ax=axes[noise_row, 1])
                    
                if config.noise_config.enable_bubble_sweep:
                    # Show bubble sweep pattern
                    height, width = depth_data.shape
                    x = np.arange(0, width * config.resolution, config.resolution)
                    y = np.arange(0, height * config.resolution, config.resolution)
                    X, Y = np.meshgrid(x, y)
                    
                    bubble_demo = self._generate_bubble_sweep_noise(
                        X, Y, config.noise_config, abs(np.mean(depth_data))
                    )
                    im_bubble = axes[noise_row, 2].imshow(bubble_demo, cmap='RdBu', origin='lower')
                    axes[noise_row, 2].set_title('Bubble Sweep Noise Pattern')
                    plt.colorbar(im_bubble, ax=axes[noise_row, 2])
                
                # Fill any unused subplot with noise summary
                remaining_axes = []
                if not config.noise_config.enable_sidelobe:
                    remaining_axes.append(axes[noise_row, 0])
                if not config.noise_config.enable_refraction:
                    remaining_axes.append(axes[noise_row, 1])
                if not config.noise_config.enable_bubble_sweep:
                    remaining_axes.append(axes[noise_row, 2])
                    
                for ax in remaining_axes:
                    ax.text(0.5, 0.5, 'Noise Pattern\nNot Enabled', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Visualization saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return False

def calculate_grid_size_for_nodes(target_nodes: int) -> Tuple[int, int]:
    """Calculate grid dimensions to achieve target number of nodes."""
    # For square grids: nodes = width * height
    side_length = int(np.sqrt(target_nodes))
    
    # Common grid sizes that are powers of 2 or multiples of 64
    common_sizes = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    
    # Find the closest common size that gives us at least target_nodes
    for size in common_sizes:
        if size * size >= target_nodes:
            actual_nodes = size * size
            print(f"Target nodes: {target_nodes:,}")
            print(f"Grid size: {size} x {size}")
            print(f"Actual nodes: {actual_nodes:,}")
            return size, size
    
    # If no common size is large enough, use the calculated size
    actual_nodes = side_length * side_length
    print(f"Target nodes: {target_nodes:,}")
    print(f"Grid size: {side_length} x {side_length}")
    print(f"Actual nodes: {actual_nodes:,}")
    return side_length, side_length

def create_noise_preset_configs() -> Dict[str, NoiseConfig]:
    """Create predefined noise configuration presets for common scenarios."""
    
    presets = {}
    
    # Clean data (no noise)
    presets['clean'] = NoiseConfig()
    
    # Shallow water survey with all noise types
    presets['shallow_water'] = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=0.2,
        sidelobe_frequency=10.0,
        enable_refraction=True,
        refraction_smile_amplitude=0.5,
        refraction_asymmetry=0.3,
        enable_bubble_sweep=True,
        bubble_sweep_amplitude=0.6,
        bubble_plume_width=0.2
    )
    
    # Deep water survey (mainly refraction issues)
    presets['deep_water'] = NoiseConfig(
        enable_refraction=True,
        refraction_smile_amplitude=3.0,  # Increased for visibility
        refraction_asymmetry=0.7,
        refraction_temporal_drift=0.8,   # Increased for visibility
        enable_sidelobe=True,
        sidelobe_amplitude=0.5,
        sidelobe_outer_beam_factor=3.0
    )
    
    # High-speed survey (bubble issues)
    presets['high_speed'] = NoiseConfig(
        enable_bubble_sweep=True,
        bubble_sweep_amplitude=1.2,
        bubble_plume_width=0.3,
        bubble_wake_length=80.0,
        bubble_random_factor=0.5,
        enable_sidelobe=True,
        sidelobe_amplitude=0.4
    )
    
    # Poor conditions (all noise, high levels) - but make bubble noise optional
    presets['poor_conditions'] = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=0.8,
        sidelobe_frequency=12.0,
        sidelobe_outer_beam_factor=4.0,
        enable_refraction=True,
        refraction_smile_amplitude=2.5,  # Increased for visibility
        refraction_asymmetry=0.8,
        refraction_temporal_drift=1.0,   # Increased for visibility
        enable_bubble_sweep=True,  # Can be disabled if problematic
        bubble_sweep_amplitude=1.0,
        bubble_plume_width=0.25,
        bubble_wake_length=100.0,
        bubble_random_factor=0.7
    )
    
    # Poor conditions without bubble noise (for large grids)
    presets['poor_conditions_no_bubble'] = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=1.0,
        sidelobe_frequency=12.0,
        sidelobe_outer_beam_factor=4.0,
        enable_refraction=True,
        refraction_smile_amplitude=3.0,
        refraction_asymmetry=0.8,
        refraction_temporal_drift=1.2,
        enable_bubble_sweep=False,  # Disabled for performance
    )
    
    # Sidelobe only (for testing sidelobe removal algorithms)
    presets['sidelobe_only'] = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=0.6,
        sidelobe_frequency=8.0,
        sidelobe_outer_beam_factor=3.0
    )
    
    # Refraction only (for testing sound speed correction)
    presets['refraction_only'] = NoiseConfig(
        enable_refraction=True,
        refraction_smile_amplitude=2.0,  # Increased for visibility
        refraction_asymmetry=0.5,
        refraction_temporal_drift=0.6    # Increased for visibility
    )
    
    # Bubble sweep only (for testing bubble noise removal)
    presets['bubble_only'] = NoiseConfig(
        enable_bubble_sweep=True,
        bubble_sweep_amplitude=0.9,
        bubble_plume_width=0.2,
        bubble_wake_length=75.0,
        bubble_random_factor=0.4
    )
    
    return presets

def create_large_bag_dataset(output_dir: Path, target_nodes: int = 50000, 
                           num_files: int = 5, resolution: float = 1.0,
                           location_name: str = None, noise_preset: str = 'clean') -> List[Path]:
    """Create BAG files with at least the target number of nodes, realistic coordinates, and noise patterns."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get noise configuration
    noise_presets = create_noise_preset_configs()
    if noise_preset not in noise_presets:
        print(f"Warning: Unknown noise preset '{noise_preset}'. Available: {list(noise_presets.keys())}")
        noise_preset = 'clean'
    
    noise_config = noise_presets[noise_preset]
    
    # Calculate required grid size
    width, height = calculate_grid_size_for_nodes(target_nodes)
    actual_nodes = width * height
    
    print(f"\n🌊 Creating large BAG dataset with {actual_nodes:,} nodes per file")
    print(f"📐 Grid dimensions: {width} x {height}")
    print(f"📏 Resolution: {resolution}m per pixel")
    print(f"📊 Coverage area: {width * resolution:.1f}m x {height * resolution:.1f}m")
    print(f"🔊 Noise preset: {noise_preset}")
    
    # Print enabled noise patterns
    noise_types = []
    if noise_config.enable_sidelobe:
        noise_types.append("sidelobe")
    if noise_config.enable_refraction:
        noise_types.append("refraction")
    if noise_config.enable_bubble_sweep:
        noise_types.append("bubble-sweep")
    
    if noise_types:
        print(f"🔊 Active noise patterns: {', '.join(noise_types)}")
    else:
        print("🔇 No acoustic noise patterns (clean data)")
    
    # Get available survey locations
    available_locations = SurveyLocation.get_all_locations()
    
    if location_name and location_name.lower() in available_locations:
        survey_locations = [available_locations[location_name.lower()]]
        print(f"🗺️  Using specified location: {survey_locations[0]['name']}")
    else:
        # Use a variety of locations
        survey_locations = list(available_locations.values())
        print(f"🗺️  Using {len(survey_locations)} different survey locations")
    
    # Estimate file sizes and memory requirements
    estimated_size_mb = (actual_nodes * 4 * 2) / (1024 * 1024)  # 4 bytes per float32, 2 bands
    total_size_mb = estimated_size_mb * num_files
    memory_req_mb = estimated_size_mb * 3  # Rough estimate for processing
    
    print(f"💾 Estimated file size: {estimated_size_mb:.1f} MB each")
    print(f"💾 Total dataset size: {total_size_mb:.1f} MB")
    print(f"🧠 Memory requirement: ~{memory_req_mb:.1f} MB per file processing")
    
    if memory_req_mb > 8000:  # > 8GB
        print("⚠️  Warning: Large memory requirement. Consider processing files individually.")
    
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
                'PURPOSE': 'Large-scale testing of Enhanced Bathymetric CAE with noise patterns',
                'NOISE_PRESET': noise_preset,
                'NOISE_PATTERNS': ', '.join(noise_types) if noise_types else 'None'
            }
            
            # Save BAG file with location info
            location_code = survey_location['name'].split(',')[0].replace(' ', '_').lower()
            noise_suffix = f"_{noise_preset}" if noise_preset != 'clean' else ""
            filename = f"large_{location_code}_{seafloor_type.value}_{actual_nodes//1000}k_nodes{noise_suffix}_{i:03d}.bag"
            file_path = output_dir / filename
            
            print(f"   Saving BAG file: {filename}")
            if generator.save_as_bag(depth_data, uncertainty_data, file_path, 
                                   large_config, metadata, survey_location):
                created_files.append(file_path)
                
                # Verify the created file
                try:
                    from osgeo import gdal
                    ds = gdal.Open(str(file_path))
                    if ds:
                        actual_width = ds.RasterXSize
                        actual_height = ds.RasterYSize
                        actual_bands = ds.RasterCount
                        geotransform = ds.GetGeoTransform()
                        projection = ds.GetProjection()
                        
                        # Check for noise metadata
                        noise_metadata = ds.GetMetadata('ACOUSTIC_NOISE')
                        ds = None
                        
                        print(f"   ✅ Verified: {actual_width}x{actual_height}, {actual_bands} bands")
                        print(f"   📊 Total nodes: {actual_width * actual_height:,}")
                        print(f"   🗺️  Origin: UTM {geotransform[0]:.0f}E, {geotransform[3]:.0f}N")
                        
                        if noise_metadata:
                            enabled_noise = [k.split('_')[0] for k, v in noise_metadata.items() 
                                           if k.endswith('_NOISE') and v == 'TRUE']
                            if enabled_noise:
                                print(f"   🔊 Noise patterns: {', '.join(enabled_noise)}")
                    else:
                        print(f"   ⚠️  Could not verify file")
                except Exception as e:
                    print(f"   ⚠️  Verification error: {e}")
            else:
                print(f"   ❌ Failed to save {filename}")
                
        except Exception as e:
            print(f"   ❌ Error creating file {i+1}: {e}")
            continue
    
    if created_files:
        # Create dataset summary with location and noise information
        summary = {
            'dataset_info': {
                'description': f'Large synthetic BAG dataset with {actual_nodes:,} nodes per file and realistic acoustic noise',
                'target_nodes': target_nodes,
                'actual_nodes': actual_nodes,
                'grid_dimensions': f"{width}x{height}",
                'resolution': f"{resolution}m",
                'num_files': len(created_files),
                'total_size_estimate': f"{total_size_mb:.1f} MB",
                'seafloor_types': [st.value for st in seafloor_types[:num_files]],
                'survey_locations': [loc['name'] for loc in survey_locations[:num_files]],
                'coordinate_systems': list(set([loc['projection'] for loc in survey_locations[:num_files]])),
                'noise_preset': noise_preset,
                'noise_patterns': noise_types,
                'creation_date': '2024-01-01T00:00:00Z',
                'use_case': 'Large-scale bathymetric CAE training and testing with acoustic noise simulation'
            },
            'noise_configuration': {
                'preset_name': noise_preset,
                'sidelobe_enabled': noise_config.enable_sidelobe,
                'refraction_enabled': noise_config.enable_refraction,
                'bubble_sweep_enabled': noise_config.enable_bubble_sweep,
                'sidelobe_amplitude_m': noise_config.sidelobe_amplitude if noise_config.enable_sidelobe else 0,
                'refraction_amplitude_m': noise_config.refraction_smile_amplitude if noise_config.enable_refraction else 0,
                'bubble_amplitude_m': noise_config.bubble_sweep_amplitude if noise_config.enable_bubble_sweep else 0
            },
            'files': [
                {
                    'filename': f.name,
                    'path': str(f),
                    'nodes': actual_nodes,
                    'seafloor_type': seafloor_types[i % len(seafloor_types)].value,
                    'location': survey_locations[i % len(survey_locations)]['name'],
                    'utm_zone': survey_locations[i % len(survey_locations)]['projection'],
                    'noise_preset': noise_preset
                } for i, f in enumerate(created_files)
            ],
            'processing_notes': {
                'memory_requirement': f"~{memory_req_mb:.1f} MB per file",
                'recommended_batch_size': "1-2 for memory efficiency",
                'grid_size_compatible': f"{min(512, width)} (for pipeline compatibility)",
                'coordinate_systems': 'Files use appropriate UTM zones for their geographic location',
                'noise_testing': 'Files include realistic acoustic noise patterns for algorithm testing'
            }
        }
        
        summary_path = output_dir / f'large_dataset_summary_{noise_preset}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Large BAG dataset creation complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Files created: {len(created_files)}")
        print(f"🎯 Nodes per file: {actual_nodes:,}")
        print(f"🗺️  Survey locations: {len(set([loc['name'] for loc in survey_locations[:num_files]]))}")
        print(f"🔊 Noise preset: {noise_preset}")
        print(f"📋 Summary saved: {summary_path}")
        
        # Memory and processing recommendations
        print(f"\n💡 Processing Recommendations:")
        print(f"   • Use --batch-size 1 for memory efficiency")
        print(f"   • Consider --grid-size {min(512, width)} for pipeline compatibility")
        print(f"   • Process files individually if memory is limited")
        print(f"   • Expected processing time: {len(created_files) * 2:.0f}-{len(created_files) * 5:.0f} minutes")
        print(f"   • Files use realistic UTM coordinates for their locations")
        print(f"   • Files include realistic acoustic noise for algorithm testing")
        
    else:
        print(f"\n❌ No files were successfully created")
    
    return created_files

def create_dataset(output_dir: Path, num_files: int = 5, formats: List[str] = None,
                  noise_preset: str = 'clean') -> List[Path]:
    """Create a complete synthetic dataset with multiple seafloor types and noise patterns."""
    
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
            'NOISE_PATTERNS': ', '.join(noise_types) if noise_types else 'None'
        }
        
        # Save in requested formats
        for format_type in formats:
            if format_type.lower() in ['bag', 'BAG']:
                noise_suffix = f"_{noise_preset}" if noise_preset != 'clean' else ""
                filename = f"synthetic_{seafloor_type.value}{noise_suffix}_{i:03d}.bag"
                file_path = output_dir / filename
                if generator.save_as_bag(depth_data, uncertainty_data, file_path, 
                                       generator.config, metadata):
                    created_files.append(file_path)
                    
            elif format_type.lower() in ['tiff', 'tif', 'geotiff']:
                noise_suffix = f"_{noise_preset}" if noise_preset != 'clean' else ""
                filename = f"synthetic_{seafloor_type.value}{noise_suffix}_{i:03d}.tif"
                file_path = output_dir / filename
                if generator.save_as_geotiff(depth_data, file_path, 
                                           generator.config, metadata):
                    created_files.append(file_path)
                    
            elif format_type.lower() in ['xyz', 'XYZ']:
                noise_suffix = f"_{noise_preset}" if noise_preset != 'clean' else ""
                filename = f"synthetic_{seafloor_type.value}{noise_suffix}_{i:03d}.xyz"
                file_path = output_dir / filename
                if generator.save_as_xyz(depth_data, file_path, generator.config):
                    created_files.append(file_path)
        
        # Create visualization
        noise_suffix = f"_{noise_preset}" if noise_preset != 'clean' else ""
        vis_path = output_dir / f"visualization_{seafloor_type.value}{noise_suffix}_{i:03d}.png"
        generator.create_visualization(depth_data, uncertainty_data, vis_path, seafloor_type, generator.config)
    
    # Create dataset summary
    summary = {
        'dataset_info': {
            'description': 'Synthetic bathymetric dataset for Enhanced Bathymetric CAE with acoustic noise patterns',
            'num_files': len(created_files),
            'seafloor_types': [st.value for st in seafloor_types[:num_files]],
            'formats': formats,
            'noise_preset': noise_preset,
            'noise_patterns': noise_types,
            'creation_date': '2024-01-01T00:00:00Z'
        },
        'noise_configuration': {
            'preset_name': noise_preset,
            'sidelobe_enabled': noise_config.enable_sidelobe,
            'refraction_enabled': noise_config.enable_refraction,
            'bubble_sweep_enabled': noise_config.enable_bubble_sweep
        },
        'files': [f.name for f in created_files]
    }
    
    summary_path = output_dir / f'dataset_summary_{noise_preset}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Dataset creation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Files created: {len(created_files)}")
    print(f"🔊 Noise preset: {noise_preset}")
    print(f"📋 Summary saved: {summary_path}")
    
    return created_files

def test_refraction_noise_generation():
    """Test function to verify refraction noise generation is working."""
    print("🧪 Testing refraction noise generation...")
    
    # Create test configuration with more visible refraction
    config = NoiseConfig(
        enable_refraction=True,
        refraction_smile_amplitude=2.0,  # Make it bigger for testing
        refraction_asymmetry=0.5,
        refraction_temporal_drift=0.3
    )
    
    generator = SyntheticBathymetryGenerator(BathymetryConfig())
    
    # Generate refraction noise directly
    refraction_noise = generator._generate_refraction_noise(256, 256, config, 100.0)
    
    print(f"✅ Refraction noise generated successfully!")
    print(f"   Shape: {refraction_noise.shape}")
    print(f"   Range: {np.min(refraction_noise):.3f} to {np.max(refraction_noise):.3f}m")
    print(f"   Mean absolute value: {np.mean(np.abs(refraction_noise)):.3f}m")
    
    # Check for smile pattern by looking at center row
    center_row = refraction_noise[refraction_noise.shape[0]//2, :]
    left_edge = center_row[0]
    center_val = center_row[len(center_row)//2]
    right_edge = center_row[-1]
    
    print(f"   Cross-track profile (center row): left={left_edge:.3f}, center={center_val:.3f}, right={right_edge:.3f}")
    
    if abs(left_edge - center_val) > 0.1 or abs(right_edge - center_val) > 0.1:
        print(f"   ✅ Smile/frown pattern detected!")
    else:
        print(f"   ⚠️  Smile pattern may be too weak")
    
    return refraction_noise

def test_sidelobe_noise_generation():
    """Test function to verify sidelobe noise generation is working."""
    print("🧪 Testing sidelobe noise generation...")
    
    # Create test configuration
    config = NoiseConfig(
        enable_sidelobe=True,
        sidelobe_amplitude=0.5,
        sidelobe_frequency=8.0,
        sidelobe_outer_beam_factor=2.0
    )
    
    generator = SyntheticBathymetryGenerator(BathymetryConfig())
    
    # Generate sidelobe noise directly
    sidelobe_noise = generator._generate_sidelobe_noise(256, 256, config, 100.0)
    
    print(f"✅ Sidelobe noise generated successfully!")
    print(f"   Shape: {sidelobe_noise.shape}")
    print(f"   Range: {np.min(sidelobe_noise):.3f} to {np.max(sidelobe_noise):.3f}m")
    print(f"   Mean absolute value: {np.mean(np.abs(sidelobe_noise)):.3f}m")
    
    return sidelobe_noise

def create_noise_debug_visualization(output_path: Path):
    """Create a detailed visualization showing each noise pattern separately."""
    print("🔍 Creating noise pattern debug visualization...")
    
    # Create test configuration with visible noise
    config = BathymetryConfig(
        width=256,
        height=256,
        base_depth=-100.0,
        depth_range=20.0,
        noise_config=NoiseConfig(
            enable_sidelobe=True,
            sidelobe_amplitude=1.0,
            sidelobe_frequency=6.0,
            enable_refraction=True,
            refraction_smile_amplitude=3.0,
            refraction_asymmetry=0.7,
            enable_bubble_sweep=True,
            bubble_sweep_amplitude=1.5
        )
    )
    
    generator = SyntheticBathymetryGenerator(config)
    
    # Create coordinate grids
    x = np.arange(0, config.width * config.resolution, config.resolution)
    y = np.arange(0, config.height * config.resolution, config.resolution)
    X, Y = np.meshgrid(x, y)
    
    # Generate base bathymetry
    base_surface = generator._generate_fractal_surface(config)
    base_depth = config.base_depth + base_surface * (config.depth_range / 2)
    
    # Generate each noise pattern separately
    sidelobe_noise = generator._generate_sidelobe_noise(config.width, config.height, config.noise_config, 100.0)
    refraction_noise = generator._generate_refraction_noise(config.width, config.height, config.noise_config, 100.0)
    bubble_noise = generator._generate_bubble_sweep_noise(X, Y, config.noise_config, 100.0)
    
    # Create combined noise
    combined_noise = sidelobe_noise + refraction_noise + bubble_noise
    noisy_bathymetry = base_depth + combined_noise
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Acoustic Noise Pattern Debug Visualization', fontsize=16)
    
    # Row 1: Individual noise patterns
    im1 = axes[0, 0].imshow(sidelobe_noise, cmap='RdBu_r', origin='lower')
    axes[0, 0].set_title(f'Sidelobe Noise\nRange: {np.min(sidelobe_noise):.3f} to {np.max(sidelobe_noise):.3f}m')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    im2 = axes[0, 1].imshow(refraction_noise, cmap='RdBu_r', origin='lower')
    axes[0, 1].set_title(f'Refraction Noise\nRange: {np.min(refraction_noise):.3f} to {np.max(refraction_noise):.3f}m')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    im3 = axes[0, 2].imshow(bubble_noise, cmap='RdBu_r', origin='lower')
    axes[0, 2].set_title(f'Bubble Noise\nRange: {np.min(bubble_noise):.3f} to {np.max(bubble_noise):.3f}m')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # Row 2: Combined patterns
    im4 = axes[1, 0].imshow(combined_noise, cmap='RdBu_r', origin='lower')
    axes[1, 0].set_title(f'Combined Noise\nRange: {np.min(combined_noise):.3f} to {np.max(combined_noise):.3f}m')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    
    im5 = axes[1, 1].imshow(base_depth, cmap='viridis', origin='lower')
    axes[1, 1].set_title(f'Base Bathymetry\nRange: {np.min(base_depth):.3f} to {np.max(base_depth):.3f}m')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    im6 = axes[1, 2].imshow(noisy_bathymetry, cmap='viridis', origin='lower')
    axes[1, 2].set_title(f'Final Noisy Bathymetry\nRange: {np.min(noisy_bathymetry):.3f} to {np.max(noisy_bathymetry):.3f}m')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    # Row 3: Cross-track profiles
    center_line = config.height // 2
    
    axes[2, 0].plot(sidelobe_noise[center_line, :], 'r-', label='Sidelobe')
    axes[2, 0].plot(refraction_noise[center_line, :], 'b-', label='Refraction')  
    axes[2, 0].plot(bubble_noise[center_line, :], 'g-', label='Bubble')
    axes[2, 0].set_title('Cross-track Noise Profiles (Center Line)')
    axes[2, 0].set_xlabel('Across-track Distance')
    axes[2, 0].set_ylabel('Noise Amplitude (m)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(base_depth[center_line, :], 'k-', label='Base')
    axes[2, 1].plot(noisy_bathymetry[center_line, :], 'r-', label='With Noise')
    axes[2, 1].set_title('Cross-track Bathymetry Comparison')
    axes[2, 1].set_xlabel('Across-track Distance')
    axes[2, 1].set_ylabel('Depth (m)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Difference plot
    depth_difference = noisy_bathymetry - base_depth
    im7 = axes[2, 2].imshow(depth_difference, cmap='RdBu_r', origin='lower')
    axes[2, 2].set_title(f'Depth Difference (Noise Only)\nRange: {np.min(depth_difference):.3f} to {np.max(depth_difference):.3f}m')
    plt.colorbar(im7, ax=axes[2, 2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Debug visualization saved: {output_path}")
    print(f"📊 Noise Statistics:")
    print(f"   Sidelobe: {np.mean(np.abs(sidelobe_noise)):.3f}m RMS")
    print(f"   Refraction: {np.mean(np.abs(refraction_noise)):.3f}m RMS")
    print(f"   Bubble: {np.mean(np.abs(bubble_noise)):.3f}m RMS")
    print(f"   Combined: {np.mean(np.abs(combined_noise)):.3f}m RMS")
    print(f"   Signal-to-Noise Ratio: {np.std(base_depth)/np.std(combined_noise):.2f}")

def test_all_noise_patterns():
    """Test all noise patterns to verify they're working."""
    print("🧪 Testing all noise patterns...")
    
    # Test each pattern individually
    bubble_noise = test_bubble_noise_generation()
    print()
    refraction_noise = test_refraction_noise_generation()
    print()
    sidelobe_noise = test_sidelobe_noise_generation()
    print()
    
    # Test combined generation
    print("🧪 Testing combined noise generation...")
    config = BathymetryConfig(
        width=256,
        height=256,
        noise_config=NoiseConfig(
            enable_sidelobe=True,
            sidelobe_amplitude=0.4,
            enable_refraction=True,
            refraction_smile_amplitude=1.0,
            enable_bubble_sweep=True,
            bubble_sweep_amplitude=0.8
        )
    )
    
    generator = SyntheticBathymetryGenerator(config)
    depth_data, uncertainty_data = generator.generate_bathymetry(SeafloorType.CONTINENTAL_SHELF, config)
    
    print(f"✅ Combined noise test completed!")
    print(f"   Final depth range: {np.min(depth_data):.3f} to {np.max(depth_data):.3f}m")
    print(f"   Uncertainty range: {np.min(uncertainty_data):.3f} to {np.max(uncertainty_data):.3f}m")

def test_bubble_noise_generation():
    """Test function to verify bubble noise generation is working."""
    print("🧪 Testing bubble noise generation...")
    
    # Create simple test configuration
    config = BathymetryConfig(
        width=256,
        height=256,
        noise_config=NoiseConfig(
            enable_bubble_sweep=True,
            bubble_sweep_amplitude=1.0,
            bubble_plume_width=0.2,
            bubble_wake_length=50.0
        )
    )
    
    generator = SyntheticBathymetryGenerator(config)
    
    # Create simple coordinate grids
    x = np.arange(0, config.width * config.resolution, config.resolution)
    y = np.arange(0, config.height * config.resolution, config.resolution)
    X, Y = np.meshgrid(x, y)
    
    # Generate bubble noise directly
    bubble_noise = generator._generate_bubble_sweep_noise(X, Y, config.noise_config, 50.0)
    
    print(f"✅ Bubble noise generated successfully!")
    print(f"   Shape: {bubble_noise.shape}")
    print(f"   Range: {np.min(bubble_noise):.3f} to {np.max(bubble_noise):.3f}")
    print(f"   Non-zero values: {np.count_nonzero(bubble_noise)} / {bubble_noise.size}")
    print(f"   Mean amplitude: {np.mean(bubble_noise[bubble_noise > 0]):.3f}")
    
    return bubble_noise

def main():
    """Main function with command line interface for enhanced noise-capable generator."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic bathymetric data with realistic acoustic noise patterns for Enhanced Bathymetric CAE testing'
    )
    
    parser.add_argument('--output', '-o', type=Path, default='synthetic_bathymetry',
                       help='Output directory (default: synthetic_bathymetry)')
    parser.add_argument('--num-files', '-n', type=int, default=5,
                       help='Number of files to generate (default: 5)')
    parser.add_argument('--formats', '-f', nargs='+', default=['bag', 'tiff'],
                       choices=['bag', 'tiff', 'xyz'],
                       help='Output formats (default: bag tiff)')
    parser.add_argument('--seafloor-type', '-s', type=str, 
                       choices=[st.value for st in SeafloorType],
                       help='Generate single seafloor type')
    parser.add_argument('--width', type=int, default=512,
                       help='Grid width in pixels (default: 512)')
    parser.add_argument('--height', type=int, default=512,
                       help='Grid height in pixels (default: 512)')
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Spatial resolution in meters (default: 1.0)')
    parser.add_argument('--target-nodes', type=int, default=50000,
                       help='Target number of nodes for large BAG files (default: 50000)')
    parser.add_argument('--large-bags', action='store_true',
                       help='Generate large BAG files with specified node count')
    parser.add_argument('--location', '-l', type=str,
                       help='Specific survey location (e.g., monterey_bay, chesapeake_bay)')
    parser.add_argument('--list-locations', action='store_true',
                       help='List all available survey locations')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only estimate grid size and memory requirements')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    # Noise pattern arguments
    parser.add_argument('--noise-preset', type=str, default='clean',
                       help='Noise preset configuration (default: clean)')
    parser.add_argument('--list-noise-presets', action='store_true',
                       help='List all available noise presets')
    parser.add_argument('--enable-sidelobe', action='store_true',
                       help='Enable sidelobe noise pattern')
    parser.add_argument('--enable-refraction', action='store_true',
                       help='Enable sound speed refraction noise')
    parser.add_argument('--enable-bubble-sweep', action='store_true',
                       help='Enable bubble sweep down noise')
    parser.add_argument('--sidelobe-amplitude', type=float, default=0.3,
                       help='Sidelobe noise amplitude in meters (default: 0.3)')
    parser.add_argument('--refraction-amplitude', type=float, default=1.0,
                       help='Refraction smile amplitude in meters (default: 1.0)')
    parser.add_argument('--bubble-amplitude', type=float, default=0.8,
                       help='Bubble sweep amplitude in meters (default: 0.8)')
    parser.add_argument('--test-bubble-noise', action='store_true',
                       help='Test bubble noise generation and exit')
    parser.add_argument('--test-refraction-noise', action='store_true',
                       help='Test refraction noise generation and exit')
    parser.add_argument('--test-sidelobe-noise', action='store_true',
                       help='Test sidelobe noise generation and exit')
    parser.add_argument('--test-all-noise', action='store_true',
                       help='Test all noise patterns and exit')
    parser.add_argument('--debug-noise-visualization', action='store_true',
                       help='Create detailed noise pattern debug visualization and exit')
    
    args = parser.parse_args()
    
    print("🌊 Enhanced Bathymetric CAE - Synthetic Data Generator with Acoustic Noise")
    print("=" * 80)
    
    # Handle noise testing
    if args.test_bubble_noise:
        test_bubble_noise_generation()
        return
    
    if args.test_refraction_noise:
        test_refraction_noise_generation()
        return
        
    if args.test_sidelobe_noise:
        test_sidelobe_noise_generation()
        return
        
    if args.test_all_noise:
        test_all_noise_patterns()
        return
        
    if args.debug_noise_visualization:
        output_path = args.output / 'noise_debug_visualization.png'
        args.output.mkdir(parents=True, exist_ok=True)
        create_noise_debug_visualization(output_path)
        return
    
    # Handle noise preset listing
    if args.list_noise_presets:
        print("\n🔊 Available Noise Presets:")
        print("=" * 40)
        presets = create_noise_preset_configs()
        
        for preset_name, config in presets.items():
            print(f"📢 {preset_name.upper()}")
            
            noise_types = []
            if config.enable_sidelobe:
                noise_types.append(f"sidelobe ({config.sidelobe_amplitude:.1f}m)")
            if config.enable_refraction:
                noise_types.append(f"refraction ({config.refraction_smile_amplitude:.1f}m)")
            if config.enable_bubble_sweep:
                noise_types.append(f"bubble-sweep ({config.bubble_sweep_amplitude:.1f}m)")
            
            if noise_types:
                print(f"   Patterns: {', '.join(noise_types)}")
            else:
                print(f"   Patterns: No noise (clean data)")
            
            # Add description
            descriptions = {
                'clean': 'Perfect data with no acoustic noise artifacts',
                'shallow_water': 'Typical shallow water survey with moderate noise levels',
                'deep_water': 'Deep water survey with strong refraction effects',
                'high_speed': 'High-speed survey with significant bubble issues',
                'poor_conditions': 'Worst-case scenario with all noise types at high levels',
                'sidelobe_only': 'Pure sidelobe noise for testing cleanup algorithms',
                'refraction_only': 'Pure refraction noise for sound speed testing',
                'bubble_only': 'Pure bubble noise for bubble detection testing'
            }
            
            if preset_name in descriptions:
                print(f"   Use case: {descriptions[preset_name]}")
            print()
        
        print("Usage examples:")
        print("  --noise-preset shallow_water")
        print("  --noise-preset poor_conditions")
        print("  --noise-preset sidelobe_only")
        return
    
    # Handle location listing
    if args.list_locations:
        print("\n🗺️  Available Survey Locations:")
        print("=" * 40)
        locations = SurveyLocation.get_all_locations()
        for key, loc in locations.items():
            utm_zone = loc['projection'].split(':')[1]
            print(f"📍 {key.upper()}")
            print(f"   Name: {loc['name']}")
            print(f"   Description: {loc['description']}")
            print(f"   UTM Zone: {utm_zone}")
            print(f"   Depth Range: {loc['depth_range'][0]}m to {loc['depth_range'][1]}m")
            print()
        
        print("Usage examples:")
        print("  --location monterey_bay")
        print("  --location chesapeake_bay")
        print("  --location gulf_of_mexico")
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
            print(f"   • Sidelobe: {custom_noise_config.sidelobe_amplitude:.1f}m amplitude")
        if custom_noise_config.enable_refraction:
            print(f"   • Refraction: {custom_noise_config.refraction_smile_amplitude:.1f}m amplitude")
        if custom_noise_config.enable_bubble_sweep:
            print(f"   • Bubble sweep: {custom_noise_config.bubble_sweep_amplitude:.1f}m amplitude")
        if not any([custom_noise_config.enable_sidelobe, custom_noise_config.enable_refraction, custom_noise_config.enable_bubble_sweep]):
            print(f"   • No noise patterns enabled")
    else:
        # Use preset
        noise_preset_name = args.noise_preset
        custom_noise_config = None
    
    # Handle large BAG generation
    if args.large_bags:
        print(f"\n🎯 Generating large BAG files with {args.target_nodes:,}+ nodes")
        created_files = create_large_bag_dataset(
            args.output, 
            args.target_nodes, 
            args.num_files, 
            args.resolution,
            args.location,
            noise_preset_name
        )
        return
    
    # Create custom configuration if specified
    if any([args.width != 512, args.height != 512, args.resolution != 1.0]) or custom_noise_config:
        config = BathymetryConfig(
            width=args.width,
            height=args.height, 
            resolution=args.resolution
        )
        
        if custom_noise_config:
            config.noise_config = custom_noise_config
    else:
        config = None
    
    if args.seafloor_type:
        # Generate single seafloor type
        seafloor_type = SeafloorType(args.seafloor_type)
        
        # Apply custom noise if specified
        if custom_noise_config:
            if config is None:
                config = BathymetryConfig(noise_config=custom_noise_config)
        
        generator = SyntheticBathymetryGenerator(config or BathymetryConfig())
        
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        depth_data, uncertainty_data = generator.generate_bathymetry(seafloor_type, config)
        
        # Determine noise types for metadata
        active_noise = config.noise_config if config else NoiseConfig()
        noise_types = []
        if active_noise.enable_sidelobe:
            noise_types.append("sidelobe")
        if active_noise.enable_refraction:
            noise_types.append("refraction")
        if active_noise.enable_bubble_sweep:
            noise_types.append("bubble-sweep")
        
        # Create metadata
        metadata = {
            'SEAFLOOR_TYPE': seafloor_type.value,
            'CREATION_DATE': '2024-01-01T00:00:00Z',
            'GENERATOR': 'Enhanced Bathymetric CAE Synthetic Generator with Acoustic Noise',
            'DEPTH_RANGE': f"{np.min(depth_data):.2f} to {np.max(depth_data):.2f} m",
            'RESOLUTION': f"{generator.config.resolution} m",
            'NOISE_PRESET': noise_preset_name,
            'NOISE_PATTERNS': ', '.join(noise_types) if noise_types else 'None'
        }
        
        # Save in requested formats
        created_files = []
        for format_type in args.formats:
            noise_suffix = f"_{noise_preset_name}" if noise_preset_name != 'clean' else ""
            
            if format_type.lower() in ['bag']:
                filename = f"synthetic_{seafloor_type.value}_single{noise_suffix}.bag"
                file_path = args.output / filename
                if generator.save_as_bag(depth_data, uncertainty_data, file_path, 
                                       generator.config, metadata):
                    created_files.append(file_path)
                    
            elif format_type.lower() in ['tiff', 'tif']:
                filename = f"synthetic_{seafloor_type.value}_single{noise_suffix}.tif"
                file_path = args.output / filename
                if generator.save_as_geotiff(depth_data, file_path, 
                                           generator.config, metadata):
                    created_files.append(file_path)
                    
            elif format_type.lower() in ['xyz']:
                filename = f"synthetic_{seafloor_type.value}_single{noise_suffix}.xyz"
                file_path = args.output / filename
                if generator.save_as_xyz(depth_data, file_path, generator.config):
                    created_files.append(file_path)
        
        # Create visualization if requested
        if args.visualize:
            noise_suffix = f"_{noise_preset_name}" if noise_preset_name != 'clean' else ""
            vis_path = args.output / f"visualization_{seafloor_type.value}_single{noise_suffix}.png"
            generator.create_visualization(depth_data, uncertainty_data, vis_path, seafloor_type, generator.config)
            
        print(f"\n✅ Single file generation complete!")
        print(f"📁 Output directory: {args.output}")
        print(f"📊 Files created: {len(created_files)}")
        print(f"🔊 Noise configuration: {noise_preset_name}")
        
    else:
        # Generate complete dataset
        created_files = create_dataset(args.output, args.num_files, args.formats, noise_preset_name)
        
        # Create visualizations if requested
        if args.visualize:
            print("\n🎨 Creating additional visualizations...")
            
            # Get noise config for visualization
            if custom_noise_config:
                vis_config = BathymetryConfig(noise_config=custom_noise_config)
            else:
                noise_presets = create_noise_preset_configs()
                preset_noise_config = noise_presets.get(noise_preset_name, NoiseConfig())
                vis_config = BathymetryConfig(noise_config=preset_noise_config)
            
            generator = SyntheticBathymetryGenerator(vis_config)
            
            # Create comparison plot of all seafloor types
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'Synthetic Bathymetric Data - All Seafloor Types (Noise: {noise_preset_name})', fontsize=16)
            
            seafloor_types = list(SeafloorType)
            for i, seafloor_type in enumerate(seafloor_types):
                if i >= 8:  # Limit to 8 for visualization
                    break
                    
                row = i // 4
                col = i % 4
                
                depth_data, _ = generator.generate_bathymetry(seafloor_type, vis_config)
                
                im = axes[row, col].imshow(depth_data, cmap='viridis', origin='lower')
                axes[row, col].set_title(f'{seafloor_type.value.replace("_", " ").title()}')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                
                # Add colorbar
                plt.colorbar(im, ax=axes[row, col], shrink=0.8)
            
            plt.tight_layout()
            noise_suffix = f"_{noise_preset_name}" if noise_preset_name != 'clean' else ""
            comparison_path = args.output / f'seafloor_types_comparison{noise_suffix}.png'
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Comparison plot saved: {comparison_path}")
    
    print("\n🎉 Synthetic bathymetric data generation completed successfully!")
    print("\n💡 Usage tips:")
    print("  • Use BAG files for testing uncertainty handling")
    print("  • Use GeoTIFF files for basic depth processing")
    print("  • Different seafloor types test various processing scenarios")
    print("  • Different noise presets test acoustic artifact removal algorithms")
    print("  • Check dataset_summary.json for file details")
    print("  • Noise patterns are stored in BAG metadata for reference")
    
    print("\n🔊 Available noise testing scenarios:")
    print("  • 'clean': Test baseline performance without artifacts")
    print("  • 'sidelobe_only': Test sidelobe noise removal algorithms")
    print("  • 'refraction_only': Test sound speed correction algorithms")
    print("  • 'bubble_only': Test bubble noise detection and removal")
    print("  • 'poor_conditions': Test robustness under worst-case conditions")
    print("  • 'shallow_water'/'deep_water': Test depth-specific scenarios")

if __name__ == "__main__":
    main()