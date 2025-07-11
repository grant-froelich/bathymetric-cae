"""
Quality metrics for bathymetric data assessment.
"""

import numpy as np
import cv2
from typing import Optional


class BathymetricQualityMetrics:
    """Domain-specific quality metrics for bathymetric data."""
    
    @staticmethod
    def calculate_roughness(data: np.ndarray) -> float:
        """Calculate seafloor roughness using standard deviation of slopes."""
        try:
            gradient = np.gradient(data)
            slope = np.sqrt(gradient[0]**2 + gradient[1]**2)
            return float(np.std(slope[np.isfinite(slope)]))
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_feature_preservation(original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate how well bathymetric features are preserved."""
        try:
            # Use Laplacian to detect features
            laplacian_orig = cv2.Laplacian(original.astype(np.float32), cv2.CV_64F)
            laplacian_clean = cv2.Laplacian(cleaned.astype(np.float32), cv2.CV_64F)
            
            # Calculate correlation between feature maps
            correlation = np.corrcoef(laplacian_orig.flatten(), laplacian_clean.flatten())[0, 1]
            return float(correlation) if np.isfinite(correlation) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_depth_consistency(data: np.ndarray) -> float:
        """Calculate consistency of depth measurements."""
        try:
            # Use local variance as a measure of consistency
            kernel = np.ones((3, 3)) / 9
            local_mean = cv2.filter2D(data.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((data - local_mean)**2, -1, kernel)
            consistency = 1.0 / (1.0 + np.mean(local_variance[np.isfinite(local_variance)]))
            return float(consistency)
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_hydrographic_standards_compliance(data: np.ndarray, 
                                                  uncertainty: Optional[np.ndarray] = None) -> float:
        """Calculate compliance with IHO hydrographic standards."""
        try:
            if uncertainty is None:
                # Estimate uncertainty from local variation
                kernel = np.ones((3, 3)) / 9
                local_std = cv2.filter2D((data - cv2.filter2D(data, -1, kernel))**2, -1, kernel)
                uncertainty = np.sqrt(local_std)
            
            # IHO S-44 standards (simplified)
            depth_dependent_uncertainty = 0.5 + 0.013 * np.abs(data)
            compliance = np.mean(uncertainty <= depth_dependent_uncertainty)
            return float(compliance)
        except Exception:
            return 0.0
