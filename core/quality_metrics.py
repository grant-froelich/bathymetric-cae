"""
Quality metrics for bathymetric data processing.
FIXED: Added missing calculate_ssim method and improved error handling.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Union
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim


class BathymetricQualityMetrics:
    """Comprehensive quality metrics for bathymetric data processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_ssim(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Structural Similarity Index (SSIM) between original and processed data.
        
        FIXED: Added missing calculate_ssim method that was being called by the pipeline.
        """
        return self.calculate_ssim_safe(original, processed)
    
    @staticmethod
    def calculate_ssim_safe(original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate SSIM with comprehensive error handling."""
        try:
            if original.shape != cleaned.shape:
                logging.warning("Shape mismatch in SSIM calculation")
                return 0.0
            
            # Ensure data is float64 and finite
            original = original.astype(np.float64)
            cleaned = cleaned.astype(np.float64)
            
            # Handle non-finite values
            if not (np.isfinite(original).all() and np.isfinite(cleaned).all()):
                logging.warning("Non-finite values in SSIM calculation")
                # Mask out non-finite values
                mask = np.isfinite(original) & np.isfinite(cleaned)
                if np.sum(mask) < original.size * 0.1:  # Less than 10% valid data
                    return 0.0
                
                # Use only finite values
                original = original[mask]
                cleaned = cleaned[mask]
                
                # Reshape to 2D if needed for SSIM
                if original.size < 49:  # Minimum size for 7x7 window
                    return 0.0
                
                # For 1D data, reshape to approximate square
                side_length = int(np.sqrt(original.size))
                original = original[:side_length*side_length].reshape(side_length, side_length)
                cleaned = cleaned[:side_length*side_length].reshape(side_length, side_length)
            
            # Calculate data range
            data_range = float(max(cleaned.max() - cleaned.min(), 1e-8))
            
            # Determine appropriate window size
            min_dim = min(original.shape[-2:])
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            win_size = max(3, win_size)  # Minimum window size of 3
            
            # Calculate SSIM
            ssim_value = ssim(
                original,
                cleaned,
                data_range=data_range,
                gaussian_weights=True,
                win_size=win_size
            )
            
            return float(ssim_value)
            
        except Exception as e:
            logging.error(f"Error calculating SSIM: {e}")
            return 0.0
    
    @staticmethod
    def calculate_roughness(data: np.ndarray) -> float:
        """Calculate surface roughness metric."""
        try:
            data = BathymetricQualityMetrics._ensure_compatible_array(data)
            
            if data.size == 0:
                return 0.0
            
            # Calculate gradient magnitude
            grad_y, grad_x = np.gradient(data)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Handle non-finite gradients
            finite_gradients = gradient_magnitude[np.isfinite(gradient_magnitude)]
            if len(finite_gradients) == 0:
                return 0.0
            
            # Roughness as normalized standard deviation of gradients
            roughness = np.std(finite_gradients) / (np.mean(np.abs(finite_gradients)) + 1e-8)
            
            return float(roughness)
            
        except Exception as e:
            logging.error(f"Error calculating roughness: {e}")
            return 0.0
    
    @staticmethod
    def calculate_feature_preservation(original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate feature preservation score."""
        try:
            original = BathymetricQualityMetrics._ensure_compatible_array(original)
            processed = BathymetricQualityMetrics._ensure_compatible_array(processed)
            
            if original.shape != processed.shape:
                logging.warning("Shape mismatch in feature preservation calculation")
                return 0.0
            
            if original.size == 0:
                return 0.0
            
            # Calculate edge maps using Sobel filter
            from scipy import ndimage
            
            original_edges = ndimage.sobel(original)
            processed_edges = ndimage.sobel(processed)
            
            # Handle non-finite values
            orig_finite = np.isfinite(original_edges)
            proc_finite = np.isfinite(processed_edges)
            valid_mask = orig_finite & proc_finite
            
            if np.sum(valid_mask) == 0:
                return 0.0
            
            # Calculate correlation between edge maps
            orig_edges_valid = original_edges[valid_mask]
            proc_edges_valid = processed_edges[valid_mask]
            
            if len(orig_edges_valid) < 2:
                return 0.0
            
            # Normalize edge values
            orig_edges_norm = (orig_edges_valid - np.mean(orig_edges_valid)) / (np.std(orig_edges_valid) + 1e-8)
            proc_edges_norm = (proc_edges_valid - np.mean(proc_edges_valid)) / (np.std(proc_edges_valid) + 1e-8)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(orig_edges_norm, proc_edges_norm)[0, 1]
            
            if np.isfinite(correlation):
                return float(max(0.0, correlation))  # Ensure non-negative
            else:
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating feature preservation: {e}")
            return 0.0
    
    @staticmethod
    def calculate_depth_consistency(data: np.ndarray) -> float:
        """Calculate depth consistency metric."""
        try:
            data = BathymetricQualityMetrics._ensure_compatible_array(data)
            
            if data.size == 0:
                return 0.0
            
            # Use 3x3 neighborhood for consistency check
            kernel = np.ones((3, 3)) / 9
            local_mean = ndimage.convolve(data, kernel, mode='constant', cval=np.nan)
            
            # Calculate local deviation
            deviation = np.abs(data - local_mean)
            
            # Handle non-finite values
            finite_mask = np.isfinite(deviation)
            if np.sum(finite_mask) == 0:
                return 0.0
            
            finite_deviation = deviation[finite_mask]
            finite_data = data[finite_mask]
            
            # Consistency as inverse of normalized deviation
            mean_depth = np.mean(np.abs(finite_data))
            mean_deviation = np.mean(finite_deviation)
            
            if mean_depth == 0:
                return 1.0 if mean_deviation == 0 else 0.0
            
            consistency = 1.0 / (1.0 + mean_deviation / mean_depth)
            
            return float(consistency)
            
        except Exception as e:
            logging.error(f"Error calculating depth consistency: {e}")
            return 0.0
    
    @staticmethod
    def calculate_hydrographic_standards_compliance(data: np.ndarray, 
                                                   uncertainty: Optional[np.ndarray] = None) -> float:
        """Calculate compliance with IHO S-44 hydrographic standards."""
        try:
            data = BathymetricQualityMetrics._ensure_compatible_array(data)
            
            if uncertainty is None:
                # Simple uncertainty estimation based on depth
                uncertainty = np.abs(data * 0.02) + 0.1
            else:
                uncertainty = BathymetricQualityMetrics._ensure_compatible_array(uncertainty)
            
            # IHO S-44 standards (simplified) - Order 1a
            # Total vertical uncertainty = √(a² + (b×d)²)
            # where a = 0.5m, b = 0.013, d = depth
            a = 0.5  # Fixed component (meters)
            b = 0.013  # Depth-dependent component (ratio)
            
            depth_dependent_uncertainty = np.sqrt(a**2 + (b * np.abs(data))**2)
            
            # Check compliance
            compliance_mask = uncertainty <= depth_dependent_uncertainty
            finite_mask = np.isfinite(uncertainty) & np.isfinite(depth_dependent_uncertainty)
            
            if np.sum(finite_mask) == 0:
                return 0.0
            
            total_valid = np.sum(finite_mask)
            compliant_valid = np.sum(compliance_mask & finite_mask)
            
            compliance_ratio = compliant_valid / total_valid
            
            return float(compliance_ratio)
                
        except Exception as e:
            logging.error(f"Error calculating hydrographic standards compliance: {e}")
            return 0.0
    
    def calculate_composite_quality(self, original: np.ndarray, processed: np.ndarray,
                                  uncertainty: Optional[np.ndarray] = None,
                                  weights: Optional[dict] = None) -> dict:
        """Calculate comprehensive quality assessment with all metrics.
        
        Args:
            original: Original bathymetric data
            processed: Processed bathymetric data  
            uncertainty: Uncertainty data (optional)
            weights: Metric weights (optional)
            
        Returns:
            Dictionary with all quality metrics and composite score
        """
        if weights is None:
            weights = {
                'ssim_weight': 0.25,
                'roughness_weight': 0.25,
                'feature_preservation_weight': 0.25,
                'consistency_weight': 0.25
            }
        
        try:
            # Calculate all individual metrics
            metrics = {
                'ssim': self.calculate_ssim_safe(original, processed),
                'roughness': self.calculate_roughness(processed),
                'feature_preservation': self.calculate_feature_preservation(original, processed),
                'consistency': self.calculate_depth_consistency(processed),
                'hydrographic_compliance': self.calculate_hydrographic_standards_compliance(
                    processed, uncertainty
                )
            }
            
            # Calculate composite quality score
            metrics['composite_quality'] = (
                weights['ssim_weight'] * metrics['ssim'] +
                weights['roughness_weight'] * max(0, 1.0 - metrics['roughness']) +
                weights['feature_preservation_weight'] * metrics['feature_preservation'] +
                weights['consistency_weight'] * metrics['consistency']
            )
            
            # Add additional derived metrics
            metrics['mae'] = float(np.mean(np.abs(original - processed)))
            metrics['rmse'] = float(np.sqrt(np.mean((original - processed)**2)))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating composite quality: {e}")
            return {
                'ssim': 0.0,
                'roughness': 1.0,
                'feature_preservation': 0.0,
                'consistency': 0.0,
                'hydrographic_compliance': 0.0,
                'composite_quality': 0.0,
                'mae': float('inf'),
                'rmse': float('inf')
            }
    
    @staticmethod
    def _ensure_compatible_array(data: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Ensure data is a compatible numpy array."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.size == 0:
            return data
        
        # Ensure at least 2D
        if data.ndim == 1:
            side = int(np.sqrt(data.size))
            if side * side == data.size:
                data = data.reshape(side, side)
            else:
                # Pad to make square
                target_size = side + 1
                padded = np.zeros(target_size * target_size)
                padded[:data.size] = data
                data = padded.reshape(target_size, target_size)
        
        return data.astype(np.float64)
