# tests/factories/data_factory.py

import numpy as np
import factory
from pathlib import Path
from typing import Tuple, Optional
from core.enums import SeafloorType


class BathymetricDataFactory(factory.Factory):
    """Factory for generating realistic bathymetric test data."""
    
    class Meta:
        model = dict
    
    # Basic parameters
    width = 64
    height = 64
    seafloor_type = SeafloorType.CONTINENTAL_SHELF
    noise_level = 0.1
    seed = 42
    
    @factory.lazy_attribute
    def depth_data(self):
        """Generate depth data based on seafloor type."""
        np.random.seed(self.seed)
        
        if self.seafloor_type == SeafloorType.SHALLOW_COASTAL:
            return self._generate_shallow_coastal_data()
        elif self.seafloor_type == SeafloorType.DEEP_OCEAN:
            return self._generate_deep_ocean_data()
        elif self.seafloor_type == SeafloorType.SEAMOUNT:
            return self._generate_seamount_data()
        elif self.seafloor_type == SeafloorType.ABYSSAL_PLAIN:
            return self._generate_abyssal_plain_data()
        else:  # CONTINENTAL_SHELF or default
            return self._generate_continental_shelf_data()
    
    @factory.lazy_attribute
    def uncertainty_data(self):
        """Generate uncertainty data based on depth."""
        # IHO S-44 standards: uncertainty increases with depth
        base_uncertainty = np.abs(self.depth_data * 0.01)  # 1% of depth
        noise = np.random.normal(0, 0.1, self.depth_data.shape)
        return np.clip(base_uncertainty + noise, 0.1, 5.0)
    
    def _generate_shallow_coastal_data(self) -> np.ndarray:
        """Generate shallow coastal bathymetry with high variability."""
        x = np.linspace(0, 10, self.width)
        y = np.linspace(0, 10, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Base bathymetry with channels and sandbars
        depth = -5 - 10 * np.sin(2 * X) - 5 * np.cos(3 * Y)
        
        # Add features like channels
        channel_mask = (X > 4) & (X < 6) & (Y > 3) & (Y < 7)
        depth[channel_mask] -= 5
        
        # Add noise
        depth += np.random.normal(0, self.noise_level * 2, depth.shape)
        
        return depth.astype(np.float32)
    
    def _generate_deep_ocean_data(self) -> np.ndarray:
        """Generate deep ocean bathymetry with steep slopes."""
        x = np.linspace(0, 10, self.width)
        y = np.linspace(0, 10, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Deep ocean with continental slope
        depth = -1000 - 200 * X - 100 * Y**2
        
        # Add canyons
        canyon_mask = np.abs(Y - 5) < 0.5
        depth[canyon_mask] -= 500
        
        # Add noise
        depth += np.random.normal(0, self.noise_level * 20, depth.shape)
        
        return depth.astype(np.float32)
    
    def _generate_seamount_data(self) -> np.ndarray:
        """Generate seamount bathymetry with high relief."""
        x = np.linspace(-5, 5, self.width)
        y = np.linspace(-5, 5, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Base deep ocean floor
        depth = np.full((self.height, self.width), -3000.0)
        
        # Add seamount (Gaussian peak)
        distance = np.sqrt(X**2 + Y**2)
        seamount_height = 2500 * np.exp(-distance**2 / 4)
        depth += seamount_height
        
        # Add smaller peaks
        for i in range(3):
            offset_x = np.random.uniform(-2, 2)
            offset_y = np.random.uniform(-2, 2)
            peak_distance = np.sqrt((X - offset_x)**2 + (Y - offset_y)**2)
            peak_height = 500 * np.exp(-peak_distance**2 / 1)
            depth += peak_height
        
        # Add noise
        depth += np.random.normal(0, self.noise_level * 30, depth.shape)
        
        return depth.astype(np.float32)
    
    def _generate_continental_shelf_data(self) -> np.ndarray:
        """Generate continental shelf bathymetry."""
        x = np.linspace(0, 10, self.width)
        y = np.linspace(0, 10, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Gradual slope from shelf to slope break
        depth = -50 - 15 * X - 5 * np.sin(Y)
        
        # Add shelf break
        shelf_break = X > 7
        depth[shelf_break] = -200 - 100 * (X[shelf_break] - 7)
        
        # Add noise
        depth += np.random.normal(0, self.noise_level * 5, depth.shape)
        
        return depth.astype(np.float32)
    
    def _generate_abyssal_plain_data(self) -> np.ndarray:
        """Generate abyssal plain bathymetry (low relief)."""
        # Very flat with minimal relief
        base_depth = -5000
        
        # Small undulations
        x = np.linspace(0, 2*np.pi, self.width)
        y = np.linspace(0, 2*np.pi, self.height)
        X, Y = np.meshgrid(x, y)
        
        undulations = 10 * np.sin(X) * np.cos(Y)
        depth = np.full((self.height, self.width), base_depth) + undulations
        
        # Minimal noise
        depth += np.random.normal(0, self.noise_level, depth.shape)
        
        return depth.astype(np.float32)


class ConfigurationFactory(factory.Factory):
    """Factory for generating test configurations."""
    
    class Meta:
        model = dict
    
    # Basic training parameters
    epochs = 5
    batch_size = 2
    learning_rate = 0.001
    validation_split = 0.2
    
    # Model architecture
    grid_size = 64
    base_filters = 16
    depth = 3
    dropout_rate = 0.1
    ensemble_size = 2
    
    # Feature flags
    enable_adaptive_processing = True
    enable_expert_review = True
    enable_constitutional_constraints = True
    
    # Quality weights
    ssim_weight = 0.25
    roughness_weight = 0.25
    feature_preservation_weight = 0.25
    consistency_weight = 0.25
    
    # Paths
    input_folder = "test_input"
    output_folder = "test_output"
    model_path = "test_model.h5"


class QualityMetricsFactory(factory.Factory):
    """Factory for generating test quality metrics."""
    
    class Meta:
        model = dict
    
    ssim = factory.Faker('pyfloat', min_value=0.6, max_value=1.0)
    roughness = factory.Faker('pyfloat', min_value=0.0, max_value=0.5)
    feature_preservation = factory.Faker('pyfloat', min_value=0.5, max_value=1.0)
    consistency = factory.Faker('pyfloat', min_value=0.6, max_value=1.0)
    hydrographic_compliance = factory.Faker('pyfloat', min_value=0.5, max_value=1.0)
    
    @factory.lazy_attribute
    def composite_quality(self):
        """Calculate composite quality from individual metrics."""
        return (
            0.25 * self.ssim +
            0.25 * (1.0 - min(self.roughness, 1.0)) +
            0.25 * self.feature_preservation +
            0.25 * self.consistency
        )


class TestFileFactory:
    """Factory for creating test files with realistic data."""
    
    @staticmethod
    def create_test_bag_file(output_path: Path, seafloor_type: SeafloorType = SeafloorType.CONTINENTAL_SHELF,
                           width: int = 64, height: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Create a test BAG file with depth and uncertainty data."""
        try:
            from osgeo import gdal
            
            # Generate test data
            data_factory = BathymetricDataFactory(
                width=width, 
                height=height, 
                seafloor_type=seafloor_type
            )
            test_data = data_factory.build()
            
            depth_data = test_data['depth_data']
            uncertainty_data = test_data['uncertainty_data']
            
            # Create BAG file
            driver = gdal.GetDriverByName('BAG')
            if driver is None:
                raise RuntimeError("BAG driver not available")
            
            dataset = driver.Create(
                str(output_path), width, height, 2, gdal.GDT_Float32
            )
            
            # Set geospatial information
            dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
            dataset.SetProjection('EPSG:4326')
            
            # Write depth data
            depth_band = dataset.GetRasterBand(1)
            depth_band.WriteArray(depth_data)
            depth_band.SetDescription('Depth')
            
            # Write uncertainty data
            uncertainty_band = dataset.GetRasterBand(2)
            uncertainty_band.WriteArray(uncertainty_data)
            uncertainty_band.SetDescription('Uncertainty')
            
            # Add metadata
            metadata = {
                'BAG_DATETIME': '2024-01-01T00:00:00',
                'BAG_VERSION': '1.6.0',
                'SEAFLOOR_TYPE': seafloor_type.value
            }
            dataset.SetMetadata(metadata)
            
            # Close dataset
            dataset = None
            
            return depth_data, uncertainty_data
            
        except Exception as e:
            print(f"Warning: Could not create BAG file - {e}")
            return None, None
    
    @staticmethod
    def create_test_geotiff(output_path: Path, seafloor_type: SeafloorType = SeafloorType.CONTINENTAL_SHELF,
                           width: int = 64, height: int = 64) -> Optional[np.ndarray]:
        """Create a test GeoTIFF file with bathymetry data."""
        try:
            from osgeo import gdal
            
            # Generate test data
            data_factory = BathymetricDataFactory(
                width=width,
                height=height,
                seafloor_type=seafloor_type
            )
            test_data = data_factory.build()
            depth_data = test_data['depth_data']
            
            # Create GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                raise RuntimeError("GTiff driver not available")
            
            dataset = driver.Create(
                str(output_path), width, height, 1, gdal.GDT_Float32
            )
            
            # Set geospatial information
            dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
            dataset.SetProjection('EPSG:4326')
            
            # Write data
            band = dataset.GetRasterBand(1)
            band.WriteArray(depth_data)
            band.SetNoDataValue(-9999)
            
            # Add metadata
            metadata = {
                'AREA_OR_POINT': 'Point',
                'SEAFLOOR_TYPE': seafloor_type.value,
                'UNITS': 'meters'
            }
            dataset.SetMetadata(metadata)
            
            # Close dataset
            dataset = None
            
            return depth_data
            
        except Exception as e:
            print(f"Warning: Could not create GeoTIFF - {e}")
            return None
    
    @staticmethod
    def create_test_dataset_collection(output_dir: Path, num_files: int = 5) -> dict:
        """Create a collection of test files with various characteristics."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        seafloor_types = list(SeafloorType)
        created_files = {
            'bag_files': [],
            'tiff_files': [],
            'metadata': []
        }
        
        for i in range(num_files):
            seafloor_type = seafloor_types[i % len(seafloor_types)]
            
            # Alternate between BAG and GeoTIFF
            if i % 2 == 0:
                # Create BAG file
                bag_path = output_dir / f"test_{seafloor_type.value}_{i}.bag"
                depth, uncertainty = TestFileFactory.create_test_bag_file(
                    bag_path, seafloor_type
                )
                if depth is not None:
                    created_files['bag_files'].append(bag_path)
                    created_files['metadata'].append({
                        'file': str(bag_path),
                        'type': 'BAG',
                        'seafloor_type': seafloor_type.value,
                        'has_uncertainty': True,
                        'depth_range': [float(depth.min()), float(depth.max())]
                    })
            else:
                # Create GeoTIFF file
                tiff_path = output_dir / f"test_{seafloor_type.value}_{i}.tif"
                depth = TestFileFactory.create_test_geotiff(
                    tiff_path, seafloor_type
                )
                if depth is not None:
                    created_files['tiff_files'].append(tiff_path)
                    created_files['metadata'].append({
                        'file': str(tiff_path),
                        'type': 'GeoTIFF',
                        'seafloor_type': seafloor_type.value,
                        'has_uncertainty': False,
                        'depth_range': [float(depth.min()), float(depth.max())]
                    })
        
        return created_files