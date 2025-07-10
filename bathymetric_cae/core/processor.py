"""
Bathymetric Data Processor Module

This module handles preprocessing of bathymetric data files, including
loading, validation, normalization, and preparation for machine learning.

Author: Bathymetric CAE Team
License: MIT
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from osgeo import gdal

from ..utils.memory_utils import memory_monitor, force_garbage_collection
from ..utils.logging_utils import log_function_call


class BathymetricProcessor:
    """
    Enhanced bathymetric data processor with comprehensive error handling.
    
    This class handles the preprocessing of various bathymetric file formats,
    including data validation, cleaning, normalization, and preparation for
    machine learning models.
    
    Attributes:
        config: Configuration object containing processing parameters
        logger: Logger instance for this processor
    """
    
    def __init__(self, config):
        """
        Initialize the bathymetric processor.
        
        Args:
            config: Configuration object with processing parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate supported formats
        if not hasattr(config, 'supported_formats') or not config.supported_formats:
            self.config.supported_formats = ['.bag', '.tif', '.tiff', '.asc', '.xyz']
        
        self.logger.info(f"Initialized processor with formats: {self.config.supported_formats}")
    
    @log_function_call
    def preprocess_bathymetric_grid(
        self, 
        file_path: Union[str, Path]
    ) -> Tuple[np.ndarray, Tuple[int, int], Optional[dict]]:
        """
        Enhanced preprocessing with comprehensive error handling.
        
        Args:
            file_path: Path to the bathymetric file
            
        Returns:
            Tuple containing:
                - Preprocessed input data as numpy array
                - Original shape of the data
                - Geospatial metadata dictionary
                
        Raises:
            ValueError: If file format is unsupported or data is invalid
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        if ext not in self.config.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}")
        
        try:
            with memory_monitor(f"Loading {file_path.name}", self.logger):
                dataset = gdal.Open(str(file_path))
                if dataset is None:
                    raise ValueError(f"Cannot open file with GDAL: {file_path}")
                
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
                
                # Validate and clean data
                depth_data = self._validate_and_clean_data(
                    depth_data, file_path, is_uncertainty=False
                )
                
                # Prepare input data
                if uncertainty_data is not None:
                    uncertainty_data = self._validate_and_clean_data(
                        uncertainty_data, file_path, is_uncertainty=True
                    )
                    input_data = self._prepare_multi_channel_input(depth_data, uncertainty_data)
                else:
                    input_data = self._prepare_single_channel_input(depth_data)
                
                # Store metadata
                geo_metadata = {
                    'geotransform': geotransform,
                    'projection': projection,
                    'metadata': metadata,
                    'file_path': str(file_path),
                    'file_format': ext
                }
                
                self.logger.info(
                    f"Processed {file_path.name}: shape={input_data.shape}, "
                    f"channels={input_data.shape[-1]}"
                )
                
                return input_data, depth_data.shape, geo_metadata
                
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            raise
        finally:
            # Clean up GDAL dataset
            if 'dataset' in locals():
                dataset = None
            force_garbage_collection()
    
    def _process_bag_file(self, dataset) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process BAG file with proper band handling.
        
        BAG files typically contain depth in band 1 and uncertainty in band 2.
        
        Args:
            dataset: GDAL dataset object
            
        Returns:
            Tuple of (depth_data, uncertainty_data)
        """
        if dataset.RasterCount < 1:
            raise ValueError("BAG file has no raster bands")
        
        # Read depth data (band 1)
        depth_data = dataset.GetRasterBand(1).ReadAsArray(
            buf_xsize=self.config.grid_size,
            buf_ysize=self.config.grid_size
        ).astype(np.float32)
        
        # Read uncertainty data (band 2) if available
        uncertainty_data = None
        if dataset.RasterCount >= 2:
            try:
                uncertainty_data = dataset.GetRasterBand(2).ReadAsArray(
                    buf_xsize=self.config.grid_size,
                    buf_ysize=self.config.grid_size
                ).astype(np.float32)
            except Exception as e:
                self.logger.warning(f"Could not read uncertainty band: {e}")
        else:
            self.logger.warning("BAG file missing uncertainty band")
        
        return depth_data, uncertainty_data
    
    def _process_standard_file(self, dataset) -> np.ndarray:
        """
        Process standard raster file (GeoTIFF, ASCII Grid, etc.).
        
        Args:
            dataset: GDAL dataset object
            
        Returns:
            numpy.ndarray: Processed depth data
        """
        if dataset.RasterCount < 1:
            raise ValueError("File has no raster bands")
        
        return dataset.GetRasterBand(1).ReadAsArray(
            buf_xsize=self.config.grid_size,
            buf_ysize=self.config.grid_size
        ).astype(np.float32)
    
    def _validate_and_clean_data(
        self, 
        data: np.ndarray, 
        file_path: Path, 
        is_uncertainty: bool = False
    ) -> np.ndarray:
        """
        Validate and clean data with enhanced error handling.
        
        Args:
            data: Input data array
            file_path: Path to source file (for logging)
            is_uncertainty: Whether this is uncertainty data
            
        Returns:
            numpy.ndarray: Cleaned data
            
        Raises:
            ValueError: If data is completely invalid
        """
        if data is None or data.size == 0:
            raise ValueError(f"Empty data in {file_path}")
        
        # Check for all invalid values
        invalid_mask = ~np.isfinite(data)
        invalid_count = np.sum(invalid_mask)
        total_count = data.size
        invalid_percent = (invalid_count / total_count) * 100
        
        if invalid_count == total_count:
            raise ValueError(f"All values are invalid in {file_path}")
        
        # Log data quality
        self.logger.debug(
            f"{file_path.name}: {invalid_percent:.1f}% invalid values "
            f"({'uncertainty' if is_uncertainty else 'depth'})"
        )
        
        # Handle invalid values
        if invalid_count > 0:
            valid_data = data[~invalid_mask]
            
            if len(valid_data) > 0:
                if is_uncertainty:
                    # For uncertainty, use median (more robust)
                    fill_value = np.median(valid_data)
                else:
                    # For depth, use mean
                    fill_value = np.mean(valid_data)
                
                data[invalid_mask] = fill_value
                self.logger.debug(
                    f"Replaced {invalid_count} invalid values with {fill_value:.3f}"
                )
            else:
                data.fill(0)
                self.logger.warning(f"No valid values found, filled with zeros")
        
        # Additional validation
        if not is_uncertainty:
            # Check for reasonable depth values (basic sanity check)
            depth_range = np.max(data) - np.min(data)
            if depth_range == 0:
                self.logger.warning("Constant depth values detected")
        
        return data
    
    def _prepare_multi_channel_input(
        self, 
        depth_data: np.ndarray, 
        uncertainty_data: np.ndarray
    ) -> np.ndarray:
        """
        Prepare multi-channel input with proper normalization.
        
        Args:
            depth_data: Depth data array
            uncertainty_data: Uncertainty data array
            
        Returns:
            numpy.ndarray: Multi-channel input array
        """
        # Normalize depth data
        depth_normalized = self._robust_normalize(depth_data)
        
        # Normalize uncertainty data
        uncertainty_normalized = self._robust_normalize(uncertainty_data)
        
        return np.stack([depth_normalized, uncertainty_normalized], axis=-1)
    
    def _prepare_single_channel_input(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Prepare single-channel input.
        
        Args:
            depth_data: Depth data array
            
        Returns:
            numpy.ndarray: Single-channel input array
        """
        depth_normalized = self._robust_normalize(depth_data)
        return np.expand_dims(depth_normalized, axis=-1)
    
    def _robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Robust normalization using percentiles to handle outliers.
        
        Args:
            data: Input data array
            
        Returns:
            numpy.ndarray: Normalized data array
        """
        # Use percentiles to handle outliers
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
    
    def batch_preprocess_files(
        self, 
        file_paths: List[Union[str, Path]]
    ) -> List[Tuple[np.ndarray, Tuple[int, int], dict]]:
        """
        Preprocess multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of preprocessed data tuples
        """
        results = []
        successful = 0
        failed = 0
        
        self.logger.info(f"Batch processing {len(file_paths)} files...")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                result = self.preprocess_bathymetric_grid(file_path)
                results.append(result)
                successful += 1
                
                if i % 10 == 0:
                    self.logger.info(f"Processed {i}/{len(file_paths)} files")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                failed += 1
                continue
        
        self.logger.info(
            f"Batch processing complete: {successful} successful, {failed} failed"
        )
        
        return results
    
    def validate_file_format(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if file format is supported.
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if format is supported
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        return ext in self.config.supported_formats
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict:
        """
        Get basic information about a bathymetric file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict: File information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': 'File not found'}
        
        try:
            dataset = gdal.Open(str(file_path))
            if dataset is None:
                return {'error': 'Cannot open with GDAL'}
            
            info = {
                'filename': file_path.name,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'format': file_path.suffix.lower(),
                'width': dataset.RasterXSize,
                'height': dataset.RasterYSize,
                'bands': dataset.RasterCount,
                'projection': dataset.GetProjection(),
                'geotransform': dataset.GetGeoTransform()
            }
            
            # Get data type of first band
            if dataset.RasterCount > 0:
                band = dataset.GetRasterBand(1)
                info['data_type'] = gdal.GetDataTypeName(band.DataType)
                info['nodata_value'] = band.GetNoDataValue()
            
            dataset = None
            return info
            
        except Exception as e:
            return {'error': str(e)}


def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats.
    
    Returns:
        List[str]: Supported file extensions
    """
    return ['.bag', '.tif', '.tiff', '.asc', '.xyz']


def validate_gdal_installation() -> Dict[str, Any]:
    """
    Validate GDAL installation and capabilities.
    
    Returns:
        Dict[str, Any]: GDAL validation results
    """
    try:
        from osgeo import gdal
        
        # Get GDAL version
        gdal_version = gdal.__version__
        
        # Get available drivers
        driver_count = gdal.GetDriverCount()
        drivers = []
        for i in range(driver_count):
            driver = gdal.GetDriver(i)
            drivers.append(driver.GetDescription())
        
        # Check for specific drivers we need
        required_drivers = ['GTiff', 'BAG', 'AAIGrid']
        available_drivers = {driver: driver in drivers for driver in required_drivers}
        
        return {
            'gdal_available': True,
            'version': gdal_version,
            'driver_count': driver_count,
            'required_drivers': available_drivers,
            'all_drivers_available': all(available_drivers.values())
        }
        
    except ImportError:
        return {
            'gdal_available': False,
            'error': 'GDAL not installed'
        }
    except Exception as e:
        return {
            'gdal_available': False,
            'error': str(e)
        }