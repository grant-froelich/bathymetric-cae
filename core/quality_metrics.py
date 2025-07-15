"""
Quality metrics for bathymetric data assessment.
Fixed version with OpenCV compatibility.
"""

import numpy as np
import logging
from typing import Optional


class BathymetricQualityMetrics:
    """Domain-specific quality metrics for bathymetric data with OpenCV compatibility fixes."""
    
    @staticmethod
    def _ensure_compatible_array(data: np.ndarray) -> np.ndarray:
        """Ensure array is compatible with processing operations."""
        # Convert to float32 and ensure it's a valid format
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
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
            
            # Use NumPy gradient instead of OpenCV to avoid format issues
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
            
            # Use NumPy-based Laplacian instead of OpenCV
            laplacian_orig = BathymetricQualityMetrics._numpy_laplacian(original)
            laplacian_clean = BathymetricQualityMetrics._numpy_laplacian(cleaned)
            
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
    def _numpy_laplacian(data: np.ndarray) -> np.ndarray:
        """Calculate Laplacian using NumPy to avoid OpenCV issues."""
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        
        # Manual convolution
        padded = np.pad(data, 1, mode='edge')
        result = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                patch = padded[i:i+3, j:j+3]
                result[i, j] = np.sum(patch * kernel)
        
        return result
    
    @staticmethod
    def calculate_depth_consistency(data: np.ndarray) -> float:
        """Calculate consistency of depth measurements."""
        try:
            data = BathymetricQualityMetrics._ensure_compatible_array(data)
            
            # Use simple local variance calculation
            kernel_size = 3
            padded = np.pad(data, kernel_size//2, mode='edge')
            local_variance = np.zeros_like(data)
            
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    patch = padded[i:i+kernel_size, j:j+kernel_size]
                    local_variance[i, j] = np.var(patch)
            
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
