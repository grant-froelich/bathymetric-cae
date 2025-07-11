# tests/utils/mock_gdal.py

from unittest.mock import Mock


class MockGDALDataset:
    """Mock GDAL dataset for testing without file dependencies."""
    
    def __init__(self, data, geotransform=None, projection=None, metadata=None):
        self.data = data if isinstance(data, list) else [data]
        self.geotransform = geotransform or (0, 1, 0, 0, 0, -1)
        self.projection = projection or 'EPSG:4326'
        self.metadata = metadata or {}
        self.RasterCount = len(self.data)
    
    def GetRasterBand(self, band_num):
        band_data = self.data[band_num - 1]  # 1-indexed
        
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = band_data
        mock_band.SetDescription = Mock()
        mock_band.WriteArray = Mock()
        mock_band.SetNoDataValue = Mock()
        
        return mock_band
    
    def GetGeoTransform(self):
        return self.geotransform
    
    def GetProjection(self):
        return self.projection
    
    def GetMetadata(self):
        return self.metadata
    
    def SetGeoTransform(self, transform):
        self.geotransform = transform
    
    def SetProjection(self, projection):
        self.projection = projection
    
    def SetMetadata(self, metadata, domain=''):
        if domain:
            self.metadata[domain] = metadata
        else:
            self.metadata.update(metadata)
    
    def FlushCache(self):
        pass


class MockGDALDriver:
    """Mock GDAL driver for creating datasets."""
    
    def __init__(self, driver_name="GTiff"):
        self.driver_name = driver_name
    
    def Create(self, filename, xsize, ysize, bands, datatype):
        """Create a mock dataset."""
        import numpy as np
        
        # Create mock data for each band
        data = [np.zeros((ysize, xsize), dtype=np.float32) for _ in range(bands)]
        
        dataset = MockGDALDataset(data)
        dataset.filename = filename
        dataset.xsize = xsize
        dataset.ysize = ysize
        dataset.bands = bands
        dataset.datatype = datatype
        
        return dataset


def mock_gdal_open(filename):
    """Mock gdal.Open function."""
    import numpy as np
    
    # Generate some test data based on filename
    if "shallow" in str(filename).lower():
        data = np.full((64, 64), -50.0, dtype=np.float32)
    elif "deep" in str(filename).lower():
        data = np.full((64, 64), -3000.0, dtype=np.float32)
    elif "seamount" in str(filename).lower():
        # Create seamount-like data
        x = np.linspace(-1, 1, 64)
        y = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, y)
        data = -2000 + 1500 * np.exp(-(X**2 + Y**2) * 5)
    else:
        # Default bathymetry
        data = np.random.uniform(-200, -10, (64, 64)).astype(np.float32)
    
    # Create uncertainty data for BAG files
    if str(filename).endswith('.bag'):
        uncertainty = np.abs(data * 0.02) + np.random.uniform(0.1, 1.0, (64, 64))
        return MockGDALDataset([data, uncertainty])
    else:
        return MockGDALDataset(data)


def mock_gdal_get_driver_by_name(driver_name):
    """Mock gdal.GetDriverByName function."""
    return MockGDALDriver(driver_name)