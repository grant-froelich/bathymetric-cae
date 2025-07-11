"""
Bathymetric data processing and preprocessing utilities.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Union, Tuple, Optional

from config.config import Config
from utils.memory_utils import memory_monitor

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    try:
        import gdal
        GDAL_AVAILABLE = True
    except ImportError:
        GDAL_AVAILABLE = False
        logging.warning("GDAL not available - some file formats will not be supported")


class BathymetricProcessor:
    """Enhanced bathymetric data processor."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required but not available")

    def preprocess_bathymetric_grid(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int], Optional[dict]]:
        """Enhanced preprocessing with comprehensive error handling."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
    
        if ext not in self.config.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}")
    
        dataset = None  # ✅ Initialize for proper cleanup
        try:
            with memory_monitor(f"Loading {file_path.name}"):
                dataset = gdal.Open(str(file_path))
                if dataset is None:
                    raise ValueError(f"Cannot open file: {file_path}")
            
                # Get geospatial information
                geotransform = dataset.GetGeoTransform()
                projection = dataset.GetProjection()
                metadata = dataset.GetMetadata()
            
                # Process based on file type
                if ext == ".bag":
                    depth_data, uncertainty_data = self._process_bag_file(dataset)
                else:
                    depth_data = self._process_standard_file(dataset)
                    uncertainty_data = None
            
                # ✅ IMPORTANT: Close dataset immediately after reading
                dataset = None  # This closes the GDAL dataset
            
                # Validate and clean data
                depth_data = self._validate_and_clean_data(depth_data, file_path)
            
                # Prepare input data
                if uncertainty_data is not None:
                    uncertainty_data = self._validate_and_clean_data(uncertainty_data, file_path, is_uncertainty=True)
                    input_data = self._prepare_multi_channel_input(depth_data, uncertainty_data)
                else:
                    input_data = self._prepare_single_channel_input(depth_data)
            
                # Store metadata
                geo_metadata = {
                    'geotransform': geotransform,
                    'projection': projection,
                    'metadata': metadata
                }
            
                return input_data, depth_data.shape, geo_metadata
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            raise
        finally:
            # ✅ Ensure dataset is always closed
            if dataset is not None:
                dataset = None
            import gc
            gc.collect()
    
    def _process_bag_file(self, dataset) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process BAG file with proper band handling."""
        if dataset.RasterCount < 2:
            self.logger.warning("BAG file missing uncertainty band")
            depth_data = dataset.GetRasterBand(1).ReadAsArray(
                buf_xsize=self.config.grid_size,
                buf_ysize=self.config.grid_size
            ).astype(np.float32)
            return depth_data, None
        else:
            depth_data = dataset.GetRasterBand(1).ReadAsArray(
                buf_xsize=self.config.grid_size,
                buf_ysize=self.config.grid_size
            ).astype(np.float32)
            uncertainty_data = dataset.GetRasterBand(2).ReadAsArray(
                buf_xsize=self.config.grid_size,
                buf_ysize=self.config.grid_size
            ).astype(np.float32)
            return depth_data, uncertainty_data
    
    def _process_standard_file(self, dataset) -> np.ndarray:
        """Process standard raster file."""
        return dataset.GetRasterBand(1).ReadAsArray(
            buf_xsize=self.config.grid_size,
            buf_ysize=self.config.grid_size
        ).astype(np.float32)
    
    def _validate_and_clean_data(self, data: np.ndarray, file_path: Path, is_uncertainty: bool = False) -> np.ndarray:
        """Validate and clean data with enhanced error handling."""
        if data is None or data.size == 0:
            raise ValueError(f"Empty data in {file_path}")
        
        # Check for all invalid values
        if np.all(~np.isfinite(data)):
            raise ValueError(f"All values are invalid in {file_path}")
        
        # Handle invalid values
        invalid_mask = ~np.isfinite(data)
        if np.any(invalid_mask):
            valid_data = data[~invalid_mask]
            if len(valid_data) > 0:
                if is_uncertainty:
                    # For uncertainty, use median
                    fill_value = np.median(valid_data)
                else:
                    # For depth, use mean
                    fill_value = np.mean(valid_data)
                data[invalid_mask] = fill_value
                self.logger.warning(f"Replaced {np.sum(invalid_mask)} invalid values in {file_path.name}")
            else:
                data.fill(0)
        
        return data
    
    def _prepare_multi_channel_input(self, depth_data: np.ndarray, uncertainty_data: np.ndarray) -> np.ndarray:
        """Prepare multi-channel input with proper normalization."""
        # Normalize depth data
        depth_normalized = self._robust_normalize(depth_data)
        
        # Normalize uncertainty data
        uncertainty_normalized = self._robust_normalize(uncertainty_data)
        
        return np.stack([depth_normalized, uncertainty_normalized], axis=-1)
    
    def _prepare_single_channel_input(self, depth_data: np.ndarray) -> np.ndarray:
        """Prepare single-channel input."""
        depth_normalized = self._robust_normalize(depth_data)
        return np.expand_dims(depth_normalized, axis=-1)
    
    def _robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """Robust normalization using percentiles to handle outliers."""
        p1, p99 = np.percentile(data, [1, 99])
        
        if p99 == p1:
            self.logger.warning("Constant values detected, using standard normalization")
            data_min, data_max = np.min(data), np.max(data)
            if data_max == data_min:
                return np.full_like(data, 0.5)
            else:
                return (data - data_min) / (data_max - data_min)
        
        # Clip outliers and normalize
        data_clipped = np.clip(data, p1, p99)
        return (data_clipped - p1) / (p99 - p1)