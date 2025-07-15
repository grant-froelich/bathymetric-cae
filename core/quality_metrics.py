"""
Quality metrics for bathymetric data assessment - COMPLETELY OpenCV-Free Version.
Fixed to eliminate all OpenCV dependencies and compatibility issues.
"""

import numpy as np
import logging
from typing import Optional
from scipy import ndimage


class BathymetricQualityMetrics:
    """Domain-specific quality metrics for bathymetric data - OpenCV-free version."""
    
    @staticmethod
    def _ensure_compatible_array(data: np.ndarray) -> np.ndarray:
        """Ensure array is compatible with processing operations."""
        # Convert to float64 for better numerical stability
        if data.dtype != np.float64:
            data = data.astype(np.float64)
        
        # Handle NaN and infinite values
        if np.any(~np.isfinite(data)):
            data = np.where(np.isfinite(data), data, 0)
        
        # Ensure data is 2D
        if data.ndim != 2:
            if data.ndim == 3 and data.shape[-1] == 1:
                data = data.squeeze(-1)
            else:
                raise ValueError(f"Expected 2D array, got shape {data.shape}")
        
        return data
    
    @staticmethod
    def calculate_roughness(data: np.ndarray) -> float:
        """Calculate seafloor roughness using standard deviation of slopes."""
        try:
            data = BathymetricQualityMetrics._ensure_compatible_array(data)
            
            # Use NumPy gradient instead of OpenCV
            grad_y, grad_x = np.gradient(data)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            
            finite_slopes = slope[np.isfinite(slope)]
            if len(finite_slopes) > 0:
                return float(np.std(finite_slopes))
            else:
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating roughness: {e}")
            return 0.0
    
    @staticmethod
    def calculate_feature_preservation(original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate how well bathymetric features are preserved."""
        try:
            original = BathymetricQualityMetrics._ensure_compatible_array(original)
            cleaned = BathymetricQualityMetrics._ensure_compatible_array(cleaned)
            
            if original.shape != cleaned.shape:
                logging.warning("Shape mismatch in feature preservation calculation")
                return 0.0
            
            # Use pure scipy.ndimage Laplacian - NO OpenCV
            laplacian_orig = BathymetricQualityMetrics._scipy_laplacian(original)
            laplacian_clean = BathymetricQualityMetrics._scipy_laplacian(cleaned)
            
            # Calculate correlation
            orig_flat = laplacian_orig.flatten()
            clean_flat = laplacian_clean.flatten()
            
            finite_mask = np.isfinite(orig_flat) & np.isfinite(clean_flat)
            if np.sum(finite_mask) < 2:
                return 0.0
            
            orig_finite = orig_flat[finite_mask]
            clean_finite = clean_flat[finite_mask]
            
            if np.std(orig_finite) == 0 or np.std(clean_finite) == 0:
                return 1.0 if np.array_equal(orig_finite, clean_finite) else 0.0
            
            correlation_matrix = np.corrcoef(orig_finite, clean_finite)
            correlation = correlation_matrix[0, 1]
            
            return float(correlation) if np.isfinite(correlation) else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating feature preservation: {e}")
            return 0.0
    
    @staticmethod
    def _scipy_laplacian(data: np.ndarray) -> np.ndarray:
        """Calculate Laplacian using ONLY scipy.ndimage - NO OpenCV."""
        try:
            # Ensure data is float64 for numerical stability
            data = data.astype(np.float64)
            
            # Use scipy's Laplacian filter
            return ndimage.laplace(data, mode='nearest')
            
        except Exception as e:
            logging.error(f"Error in scipy Laplacian calculation: {e}")
            # Fallback to manual convolution
            return BathymetricQualityMetrics._manual_laplacian(data)
    
    @staticmethod
    def _manual_laplacian(data: np.ndarray) -> np.ndarray:
        """Manual Laplacian calculation as fallback."""
        try:
            # Ensure float64
            data = data.astype(np.float64)
            
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
            
            # Use scipy.ndimage.convolve instead of manual convolution
            return ndimage.convolve(data, kernel, mode='constant', cval=0.0)
            
        except Exception as e:
            logging.error(f"Error in manual Laplacian: {e}")
            # Ultimate fallback - return zeros
            return np.zeros_like(data, dtype=np.float64)
    
    @staticmethod
    def calculate_depth_consistency(data: np.ndarray) -> float:
        """Calculate consistency of depth measurements using ONLY scipy."""
        try:
            data = BathymetricQualityMetrics._ensure_compatible_array(data)
            
            # Use scipy.ndimage for local variance calculation
            # Create a uniform filter for local mean
            local_mean = ndimage.uniform_filter(data, size=3, mode='nearest')
            
            # Calculate local variance
            data_squared = data * data
            local_mean_squared = ndimage.uniform_filter(data_squared, size=3, mode='nearest')
            local_variance = local_mean_squared - local_mean * local_mean
            
            finite_variance = local_variance[np.isfinite(local_variance)]
            if len(finite_variance) > 0:
                consistency = 1.0 / (1.0 + np.mean(finite_variance))
                return float(consistency)
            else:
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating depth consistency: {e}")
            return 0.0
    
    @staticmethod
    def calculate_hydrographic_standards_compliance(data: np.ndarray, 
                                                  uncertainty: Optional[np.ndarray] = None) -> float:
        """Calculate compliance with IHO hydrographic standards."""
        try:
            data = BathymetricQualityMetrics._ensure_compatible_array(data)
            
            if uncertainty is None:
                # Simple uncertainty estimation
                uncertainty = np.abs(data * 0.02) + 0.1
            else:
                uncertainty = BathymetricQualityMetrics._ensure_compatible_array(uncertainty)
            
            # IHO S-44 standards (simplified)
            depth_dependent_uncertainty = 0.5 + 0.013 * np.abs(data)
            
            compliance_mask = uncertainty <= depth_dependent_uncertainty
            finite_mask = np.isfinite(uncertainty) & np.isfinite(depth_dependent_uncertainty)
            
            if np.sum(finite_mask) > 0:
                compliance = np.sum(compliance_mask & finite_mask) / np.sum(finite_mask)
                return float(compliance)
            else:
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating hydrographic standards compliance: {e}")
            return 0.0

    @staticmethod 
    def calculate_ssim_safe(original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate SSIM with comprehensive error handling."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            if original.shape != cleaned.shape:
                logging.warning("Shape mismatch in SSIM calculation")
                return 0.0
            
            # Ensure data is float64 and finite
            original = original.astype(np.float64)
            cleaned = cleaned.astype(np.float64)
            
            if not (np.isfinite(original).all() and np.isfinite(cleaned).all()):
                logging.warning("Non-finite values in SSIM calculation")
                return 0.0
            
            data_range = float(max(cleaned.max() - cleaned.min(), 1e-8))
            
            return ssim(
                original,
                cleaned,
                data_range=data_range,
                gaussian_weights=True,
                win_size=min(7, min(original.shape[-2:]))
            )
        except Exception as e:
            logging.error(f"Error calculating SSIM: {e}")
            return 0.0
