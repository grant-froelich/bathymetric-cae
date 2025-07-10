"""
Data processing and I/O operations for Enhanced Bathymetric CAE Processing.

This module handles loading, preprocessing, and saving of bathymetric data
in various formats including BAG, GeoTIFF, ASCII Grid, and XYZ.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Union, Any
import datetime

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    logging.warning("GDAL not available - some file formats may not be supported")

from ..config import Config
from ..core.enums import DataFormat, DEFAULT_NODATA_VALUE


class BathymetricProcessor:
    """Enhanced bathymetric data processor with comprehensive format support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not GDAL_AVAILABLE:
            self.logger.warning("GDAL not available - functionality will be limited")
    
    def preprocess_bathymetric_grid(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int], Optional[Dict]]:
        """Enhanced preprocessing with comprehensive error handling."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        
        if ext not in self.config.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}")
        
        try:
            self.logger.info(f"Loading {file_path.name} ({ext})")
            
            if not GDAL_AVAILABLE:
                return self._fallback_processing(file_path)
            
            # Open with GDAL
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
            
            # Get original shape before resizing
            original_shape = depth_data.shape
            
            # Validate and clean data
            depth_data = self._validate_and_clean_data(depth_data, file_path)
            
            # Resize to configured grid size
            depth_data = self._resize_to_grid(depth_data)
            
            # Prepare input data
            if uncertainty_data is not None:
                uncertainty_data = self._validate_and_clean_data(uncertainty_data, file_path, is_uncertainty=True)
                uncertainty_data = self._resize_to_grid(uncertainty_data)
                input_data = self._prepare_multi_channel_input(depth_data, uncertainty_data)
            else:
                input_data = self._prepare_single_channel_input(depth_data)
            
            # Store metadata
            geo_metadata = {
                'geotransform': geotransform,
                'projection': projection,
                'metadata': metadata,
                'original_shape': original_shape,
                'file_format': ext
            }
            
            self.logger.info(f"Successfully loaded {file_path.name}: {original_shape} -> {input_data.shape}")
            
            return input_data, original_shape, geo_metadata
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            raise
        finally:
            # Cleanup GDAL dataset
            if 'dataset' in locals() and dataset is not None:
                del dataset
    
    def _process_bag_file(self, dataset) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process BAG file with proper band handling."""
        try:
            if dataset.RasterCount < 1:
                raise ValueError("BAG file has no raster bands")
            
            # Read depth data (band 1)
            depth_band = dataset.GetRasterBand(1)
            depth_data = depth_band.ReadAsArray(
                buf_xsize=self.config.grid_size,
                buf_ysize=self.config.grid_size
            )
            
            if depth_data is None:
                raise ValueError("Failed to read depth data from BAG file")
            
            depth_data = depth_data.astype(np.float32)
            
            # Read uncertainty data (band 2) if available
            uncertainty_data = None
            if dataset.RasterCount >= 2:
                uncertainty_band = dataset.GetRasterBand(2)
                uncertainty_data = uncertainty_band.ReadAsArray(
                    buf_xsize=self.config.grid_size,
                    buf_ysize=self.config.grid_size
                )
                
                if uncertainty_data is not None:
                    uncertainty_data = uncertainty_data.astype(np.float32)
                else:
                    self.logger.warning("Failed to read uncertainty data from BAG file")
            else:
                self.logger.warning("BAG file missing uncertainty band")
            
            return depth_data, uncertainty_data
            
        except Exception as e:
            self.logger.error(f"Error processing BAG file: {e}")
            raise
    
    def _process_standard_file(self, dataset) -> np.ndarray:
        """Process standard raster file (GeoTIFF, ASCII Grid)."""
        try:
            if dataset.RasterCount < 1:
                raise ValueError("File has no raster bands")
            
            # Read first band
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray(
                buf_xsize=self.config.grid_size,
                buf_ysize=self.config.grid_size
            )
            
            if data is None:
                raise ValueError("Failed to read data from file")
            
            return data.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error processing standard file: {e}")
            raise
    
    def _fallback_processing(self, file_path: Path) -> Tuple[np.ndarray, Tuple[int, int], Optional[Dict]]:
        """Fallback processing when GDAL is not available."""
        self.logger.warning("Using fallback processing - limited functionality")
        
        ext = file_path.suffix.lower()
        
        if ext == '.xyz':
            return self._process_xyz_file(file_path)
        elif ext == '.asc':
            return self._process_asc_file(file_path)
        else:
            raise RuntimeError(f"Cannot process {ext} files without GDAL")
    
    def _process_xyz_file(self, file_path: Path) -> Tuple[np.ndarray, Tuple[int, int], Optional[Dict]]:
        """Process XYZ format file."""
        try:
            # Read XYZ data
            data = np.loadtxt(file_path, delimiter=None)
            
            if data.shape[1] < 3:
                raise ValueError("XYZ file must have at least 3 columns (X, Y, Z)")
            
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            
            # Create regular grid
            x_unique = np.unique(x)
            y_unique = np.unique(y)
            
            if len(x_unique) * len(y_unique) != len(data):
                self.logger.warning("Irregular XYZ grid detected - interpolating")
                # Simple gridding
                xi = np.linspace(x.min(), x.max(), self.config.grid_size)
                yi = np.linspace(y.min(), y.max(), self.config.grid_size)
                from scipy.interpolate import griddata
                Xi, Yi = np.meshgrid(xi, yi)
                depth_data = griddata((x, y), z, (Xi, Yi), method='linear')
            else:
                # Regular grid - reshape
                ny, nx = len(y_unique), len(x_unique)
                depth_data = z.reshape(ny, nx)
                depth_data = self._resize_to_grid(depth_data)
            
            original_shape = depth_data.shape
            input_data = self._prepare_single_channel_input(depth_data)
            
            geo_metadata = {
                'bounds': (x.min(), y.min(), x.max(), y.max()),
                'file_format': '.xyz'
            }
            
            return input_data, original_shape, geo_metadata
            
        except Exception as e:
            self.logger.error(f"Error processing XYZ file: {e}")
            raise
    
    def _process_asc_file(self, file_path: Path) -> Tuple[np.ndarray, Tuple[int, int], Optional[Dict]]:
        """Process ASCII Grid file."""
        try:
            with open(file_path, 'r') as f:
                # Read header
                header = {}
                for _ in range(6):
                    line = f.readline().strip().split()
                    header[line[0].lower()] = float(line[1])
                
                # Read data
                data = []
                for line in f:
                    data.extend([float(x) for x in line.split()])
            
            # Reshape data
            ncols = int(header['ncols'])
            nrows = int(header['nrows'])
            depth_data = np.array(data).reshape(nrows, ncols).astype(np.float32)
            
            # Handle NODATA values
            nodata = header.get('nodata_value', DEFAULT_NODATA_VALUE)
            depth_data[depth_data == nodata] = np.nan
            
            original_shape = depth_data.shape
            depth_data = self._resize_to_grid(depth_data)
            input_data = self._prepare_single_channel_input(depth_data)
            
            geo_metadata = {
                'header': header,
                'file_format': '.asc'
            }
            
            return input_data, original_shape, geo_metadata
            
        except Exception as e:
            self.logger.error(f"Error processing ASCII file: {e}")
            raise
    
    def _resize_to_grid(self, data: np.ndarray) -> np.ndarray:
        """Resize data to configured grid size."""
        try:
            if data.shape == (self.config.grid_size, self.config.grid_size):
                return data
            
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_y = self.config.grid_size / data.shape[0]
            zoom_x = self.config.grid_size / data.shape[1]
            
            # Resize using scipy zoom
            resized = zoom(data, (zoom_y, zoom_x), order=1, prefilter=False)
            
            # Ensure exact size
            if resized.shape != (self.config.grid_size, self.config.grid_size):
                # Crop or pad to exact size
                resized = self._crop_or_pad_to_size(resized, self.config.grid_size)
            
            return resized.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error resizing data: {e}")
            # Fallback to simple interpolation
            return self._simple_resize(data)
    
    def _crop_or_pad_to_size(self, data: np.ndarray, target_size: int) -> np.ndarray:
        """Crop or pad data to exact target size."""
        current_h, current_w = data.shape
        
        if current_h > target_size:
            # Crop height
            start = (current_h - target_size) // 2
            data = data[start:start + target_size, :]
        elif current_h < target_size:
            # Pad height
            pad_h = target_size - current_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            data = np.pad(data, ((pad_top, pad_bottom), (0, 0)), mode='edge')
        
        current_h, current_w = data.shape
        
        if current_w > target_size:
            # Crop width
            start = (current_w - target_size) // 2
            data = data[:, start:start + target_size]
        elif current_w < target_size:
            # Pad width
            pad_w = target_size - current_w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            data = np.pad(data, ((0, 0), (pad_left, pad_right)), mode='edge')
        
        return data
    
    def _simple_resize(self, data: np.ndarray) -> np.ndarray:
        """Simple resize fallback using linear interpolation."""
        try:
            from scipy.interpolate import RectBivariateSpline
            
            h, w = data.shape
            x = np.arange(w)
            y = np.arange(h)
            
            # Create spline interpolator
            spline = RectBivariateSpline(y, x, data, kx=1, ky=1)
            
            # New coordinates
            new_x = np.linspace(0, w-1, self.config.grid_size)
            new_y = np.linspace(0, h-1, self.config.grid_size)
            
            # Interpolate
            resized = spline(new_y, new_x)
            
            return resized.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in simple resize: {e}")
            # Ultimate fallback - just repeat/truncate
            return np.resize(data, (self.config.grid_size, self.config.grid_size))
    
    def _validate_and_clean_data(self, data: np.ndarray, file_path: Path, is_uncertainty: bool = False) -> np.ndarray:
        """Validate and clean data with enhanced error handling."""
        if data is None or data.size == 0:
            raise ValueError(f"Empty data in {file_path}")
        
        # Check for all invalid values
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            raise ValueError(f"All values are invalid in {file_path}")
        
        # Handle invalid values
        if not np.all(valid_mask):
            valid_data = data[valid_mask]
            
            if len(valid_data) > 0:
                if is_uncertainty:
                    # For uncertainty, use median
                    fill_value = np.median(valid_data)
                else:
                    # For depth, use mean
                    fill_value = np.mean(valid_data)
                
                data = data.copy()
                data[~valid_mask] = fill_value
                
                invalid_count = np.sum(~valid_mask)
                invalid_percent = (invalid_count / data.size) * 100
                
                self.logger.warning(f"Replaced {invalid_count} invalid values ({invalid_percent:.1f}%) in {file_path.name}")
            else:
                data = np.zeros_like(data)
        
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
        finite_data = data[np.isfinite(data)]
        
        if len(finite_data) == 0:
            self.logger.warning("No finite values for normalization")
            return np.full_like(data, 0.5)
        
        p1, p99 = np.percentile(finite_data, [1, 99])
        
        if p99 == p1:
            self.logger.warning("Constant values detected, using standard normalization")
            data_min, data_max = np.min(finite_data), np.max(finite_data)
            if data_max == data_min:
                return np.full_like(data, 0.5)
            else:
                return (data - data_min) / (data_max - data_min)
        
        # Clip outliers and normalize
        data_clipped = np.clip(data, p1, p99)
        normalized = (data_clipped - p1) / (p99 - p1)
        
        return normalized
    
    def save_enhanced_results(self, data: np.ndarray, output_path: Path, 
                            original_shape: Tuple[int, int], geo_metadata: Dict,
                            quality_metrics: Dict, adaptive_params: Dict):
        """Save enhanced results with comprehensive metadata."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Resize data back to original shape if needed
            if data.shape != original_shape:
                data = self._resize_data_to_shape(data, original_shape)
            
            # Determine output format
            ext = output_path.suffix.lower()
            
            if not GDAL_AVAILABLE:
                self._save_without_gdal(data, output_path, quality_metrics, adaptive_params)
                return
            
            # Use GDAL for saving
            if ext == '.bag':
                driver_name = 'BAG'
            elif ext in ['.tif', '.tiff']:
                driver_name = 'GTiff'
            elif ext == '.asc':
                driver_name = 'AAIGrid'
            else:
                output_path = output_path.with_suffix('.tif')
                driver_name = 'GTiff'
            
            # Create output dataset
            driver = gdal.GetDriverByName(driver_name)
            if driver is None:
                raise RuntimeError(f"Driver {driver_name} not available")
            
            dataset = driver.Create(
                str(output_path),
                original_shape[1], original_shape[0], 1,
                gdal.GDT_Float32
            )
            
            if dataset is None:
                raise RuntimeError(f"Failed to create dataset: {output_path}")
            
            # Set geospatial information
            if 'geotransform' in geo_metadata:
                dataset.SetGeoTransform(geo_metadata['geotransform'])
            if 'projection' in geo_metadata:
                dataset.SetProjection(geo_metadata['projection'])
            
            # Write data
            band = dataset.GetRasterBand(1)
            band.WriteArray(data.astype(np.float32))
            band.SetNoDataValue(DEFAULT_NODATA_VALUE)
            
            # Set enhanced metadata
            self._set_enhanced_metadata(dataset, quality_metrics, adaptive_params)
            
            # Flush and close
            dataset.FlushCache()
            dataset = None
            
            self.logger.info(f"Successfully saved enhanced results: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving {output_path}: {e}")
            raise
    
    def _resize_data_to_shape(self, data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize data back to original shape."""
        if data.shape == target_shape:
            return data
        
        try:
            from scipy.ndimage import zoom
            
            zoom_y = target_shape[0] / data.shape[0]
            zoom_x = target_shape[1] / data.shape[1]
            
            resized = zoom(data, (zoom_y, zoom_x), order=1, prefilter=False)
            
            # Ensure exact size
            if resized.shape != target_shape:
                resized = self._crop_or_pad_to_target_shape(resized, target_shape)
            
            return resized
            
        except Exception as e:
            self.logger.error(f"Error resizing to original shape: {e}")
            return data
    
    def _crop_or_pad_to_target_shape(self, data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Crop or pad to exact target shape."""
        target_h, target_w = target_shape
        current_h, current_w = data.shape
        
        # Handle height
        if current_h > target_h:
            start = (current_h - target_h) // 2
            data = data[start:start + target_h, :]
        elif current_h < target_h:
            pad_h = target_h - current_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            data = np.pad(data, ((pad_top, pad_bottom), (0, 0)), mode='edge')
        
        # Handle width
        current_h, current_w = data.shape
        if current_w > target_w:
            start = (current_w - target_w) // 2
            data = data[:, start:start + target_w]
        elif current_w < target_w:
            pad_w = target_w - current_w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            data = np.pad(data, ((0, 0), (pad_left, pad_right)), mode='edge')
        
        return data
    
    def _set_enhanced_metadata(self, dataset, quality_metrics: Dict, adaptive_params: Dict):
        """Set enhanced metadata for the dataset."""
        processing_metadata = {
            'PROCESSING_DATE': datetime.datetime.now().isoformat(),
            'PROCESSING_SOFTWARE': 'Enhanced Bathymetric CAE v2.0',
            'MODEL_TYPE': 'Ensemble Convolutional Autoencoder',
            'ENSEMBLE_SIZE': str(self.config.ensemble_size),
            'GRID_SIZE': str(self.config.grid_size),
            'COMPOSITE_QUALITY': f"{quality_metrics.get('composite_quality', 0):.4f}",
            'SSIM_SCORE': f"{quality_metrics.get('ssim', 0):.4f}",
            'FEATURE_PRESERVATION': f"{quality_metrics.get('feature_preservation', 0):.4f}",
            'HYDROGRAPHIC_COMPLIANCE': f"{quality_metrics.get('hydrographic_compliance', 0):.4f}",
            'SEAFLOOR_TYPE': adaptive_params.get('seafloor_type', 'unknown'),
            'ADAPTIVE_PROCESSING': 'TRUE' if self.config.enable_adaptive_processing else 'FALSE',
            'CONSTITUTIONAL_CONSTRAINTS': 'TRUE' if self.config.enable_constitutional_constraints else 'FALSE'
        }
        dataset.SetMetadata(processing_metadata, 'PROCESSING')
        
        # Add quality metrics as metadata
        quality_metadata = {f"QUALITY_{k.upper()}": str(v) for k, v in quality_metrics.items()}
        dataset.SetMetadata(quality_metadata, 'QUALITY')
        
        # Add adaptive parameters as metadata
        adaptive_metadata = {f"ADAPTIVE_{k.upper()}": str(v) for k, v in adaptive_params.items()}
        dataset.SetMetadata(adaptive_metadata, 'ADAPTIVE')
    
    def _save_without_gdal(self, data: np.ndarray, output_path: Path, 
                          quality_metrics: Dict, adaptive_params: Dict):
        """Save data without GDAL (limited functionality)."""
        ext = output_path.suffix.lower()
        
        if ext == '.asc':
            self._save_asc_file(data, output_path, quality_metrics)
        elif ext == '.xyz':
            self._save_xyz_file(data, output_path, quality_metrics)
        else:
            # Save as numpy array
            np.save(output_path.with_suffix('.npy'), data)
            self.logger.warning(f"Saved as numpy array: {output_path.with_suffix('.npy')}")
    
    def _save_asc_file(self, data: np.ndarray, output_path: Path, quality_metrics: Dict):
        """Save as ASCII Grid file."""
        try:
            nrows, ncols = data.shape
            
            with open(output_path, 'w') as f:
                # Write header
                f.write(f"ncols {ncols}\n")
                f.write(f"nrows {nrows}\n")
                f.write(f"xllcorner 0\n")
                f.write(f"yllcorner 0\n")
                f.write(f"cellsize 1\n")
                f.write(f"NODATA_value {DEFAULT_NODATA_VALUE}\n")
                
                # Write data
                for row in data:
                    f.write(' '.join(f"{val:.6f}" for val in row) + '\n')
            
            self.logger.info(f"Saved ASCII grid: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving ASCII file: {e}")
            raise
    
    def _save_xyz_file(self, data: np.ndarray, output_path: Path, quality_metrics: Dict):
        """Save as XYZ file."""
        try:
            nrows, ncols = data.shape
            
            with open(output_path, 'w') as f:
                for i in range(nrows):
                    for j in range(ncols):
                        f.write(f"{j} {i} {data[i, j]:.6f}\n")
            
            self.logger.info(f"Saved XYZ file: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving XYZ file: {e}")
            raise
    
    def get_supported_formats(self) -> Dict[str, bool]:
        """Get list of supported formats and their availability."""
        formats = {}
        
        for fmt in self.config.supported_formats:
            if fmt in ['.xyz', '.asc']:
                formats[fmt] = True  # Always supported
            else:
                formats[fmt] = GDAL_AVAILABLE
        
        return formats
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a bathymetric file."""
        file_path = Path(file_path)
        
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Basic file checks
            if not file_path.exists():
                validation_result['errors'].append(f"File does not exist: {file_path}")
                return validation_result
            
            if file_path.stat().st_size == 0:
                validation_result['errors'].append("File is empty")
                return validation_result
            
            ext = file_path.suffix.lower()
            if ext not in self.config.supported_formats:
                validation_result['errors'].append(f"Unsupported format: {ext}")
                return validation_result
            
            # Try to load and validate data
            input_data, original_shape, geo_metadata = self.preprocess_bathymetric_grid(file_path)
            
            validation_result['info'] = {
                'format': ext,
                'original_shape': original_shape,
                'processed_shape': input_data.shape,
                'channels': input_data.shape[-1],
                'has_uncertainty': input_data.shape[-1] > 1,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
            validation_result['valid'] = True
            
        except Exception as e:
            validation_result['errors'].append(str(e))
        
        return validation_result