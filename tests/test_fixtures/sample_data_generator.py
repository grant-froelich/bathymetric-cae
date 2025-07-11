# tests/test_fixtures/sample_data_generator.py
"""
Generate sample test data for comprehensive testing.
"""

import numpy as np
import tempfile
from pathlib import Path
from osgeo import gdal
import json


class TestDataGenerator:
    """Generate sample bathymetric data for testing."""
    
    @staticmethod
    def create_sample_bag_file(output_path: Path, width: int = 64, height: int = 64):
        """Create sample BAG file with depth and uncertainty data."""
        try:
            # Create depth data (seafloor bathymetry)
            x = np.linspace(0, 10, width)
            y = np.linspace(0, 10, height)
            X, Y = np.meshgrid(x, y)
            
            # Create realistic bathymetry with features
            depth_data = -50 - 20 * np.sin(X) - 10 * np.cos(Y) + np.random.normal(0, 2, (height, width))
            
            # Create uncertainty data
            uncertainty_data = np.abs(depth_data * 0.02) + np.random.uniform(0.1, 1.0, (height, width))
            
            # Create BAG file
            driver = gdal.GetDriverByName('BAG')
            dataset = driver.Create(
                str(output_path), width, height, 2, gdal.GDT_Float32
            )
            
            # Set geotransform and projection
            dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
            dataset.SetProjection('EPSG:4326')
            
            # Write data
            depth_band = dataset.GetRasterBand(1)
            depth_band.WriteArray(depth_data.astype(np.float32))
            depth_band.SetDescription('Depth')
            
            uncertainty_band = dataset.GetRasterBand(2)
            uncertainty_band.WriteArray(uncertainty_data.astype(np.float32))
            uncertainty_band.SetDescription('Uncertainty')
            
            # Close dataset
            dataset = None
            
            return depth_data, uncertainty_data
            
        except Exception as e:
            print(f"Warning: Could not create BAG file - {e}")
            return None, None
    
    @staticmethod
    def create_sample_geotiff(output_path: Path, width: int = 64, height: int = 64):
        """Create sample GeoTIFF file with bathymetry data."""
        try:
            # Create bathymetry data
            depth_data = TestDataGenerator._generate_bathymetry_data(width, height)
            
            # Create GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(
                str(output_path), width, height, 1, gdal.GDT_Float32
            )
            
            # Set geotransform and projection
            dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
            dataset.SetProjection('EPSG:4326')
            
            # Write data
            band = dataset.GetRasterBand(1)
            band.WriteArray(depth_data.astype(np.float32))
            band.SetNoDataValue(-9999)
            
            # Close dataset
            dataset = None
            
            return depth_data
            
        except Exception as e:
            print(f"Warning: Could not create GeoTIFF - {e}")
            return None
    
    @staticmethod
    def _generate_bathymetry_data(width: int, height: int, seafloor_type: str = "mixed"):
        """Generate realistic bathymetry data for different seafloor types."""
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        
        if seafloor_type == "shallow_coastal":
            # Shallow coastal with high variability
            depth_data = -10 - 15 * np.sin(2*X) - 10 * np.cos(2*Y) + np.random.normal(0, 3, (height, width))
        elif seafloor_type == "deep_ocean":
            # Deep ocean with steep slopes
            depth_data = -2000 - 500 * X - 200 * np.sin(Y) + np.random.normal(0, 50, (height, width))
        elif seafloor_type == "seamount":
            # Seamount with peak
            center_x, center_y = width//2, height//2
            distance = np.sqrt((np.arange(width)[None, :] - center_x)**2 + 
                             (np.arange(height)[:, None] - center_y)**2)
            depth_data = -2000 + 1500 * np.exp(-distance/10) + np.random.normal(0, 20, (height, width))
        else:  # mixed
            # Mixed terrain
            depth_data = -100 - 50 * np.sin(X) - 30 * np.cos(Y) + np.random.normal(0, 5, (height, width))
        
        return depth_data
    
    @staticmethod
    def create_test_dataset(output_dir: Path, num_files: int = 5):
        """Create a complete test dataset with various file types."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        # Create different types of files
        seafloor_types = ["shallow_coastal", "deep_ocean", "seamount", "mixed"]
        
        for i in range(num_files):
            seafloor_type = seafloor_types[i % len(seafloor_types)]
            
            # Create BAG file
            if i % 2 == 0:
                bag_path = output_dir / f"test_bathymetry_{seafloor_type}_{i}.bag"
                depth, uncertainty = TestDataGenerator.create_sample_bag_file(bag_path)
                if depth is not None:
                    created_files.append(bag_path)
            
            # Create GeoTIFF file
            else:
                tiff_path = output_dir / f"test_bathymetry_{seafloor_type}_{i}.tif"
                depth = TestDataGenerator.create_sample_geotiff(tiff_path)
                if depth is not None:
                    created_files.append(tiff_path)
        
        # Create metadata file
        metadata = {
            "dataset_info": {
                "description": "Test dataset for Enhanced Bathymetric CAE",
                "num_files": len(created_files),
                "file_types": ["BAG", "GeoTIFF"],
                "seafloor_types": seafloor_types,
                "grid_size": "64x64",
                "coordinate_system": "EPSG:4326"
            },
            "files": [str(f.name) for f in created_files]
        }
        
        metadata_path = output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return created_files, metadata_path