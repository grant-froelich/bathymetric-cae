"""
Constitutional AI constraints for bathymetric data processing.
Fixed to remove OpenCV dependency and use scipy.ndimage instead.
"""

import numpy as np
from scipy import ndimage


class BathymetricConstraints:
    """Constitutional AI rules for bathymetric data cleaning - OpenCV-free version."""
    
    @staticmethod
    def validate_depth_continuity(data: np.ndarray, max_gradient: float = 0.1) -> np.ndarray:
        """Ensure depth changes are physically plausible."""
        gradient = np.gradient(data)
        gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
        violation_mask = gradient_magnitude > max_gradient
        return violation_mask
    
    @staticmethod
    def preserve_depth_features(original: np.ndarray, cleaned: np.ndarray, 
                               feature_threshold: float = 0.05) -> np.ndarray:
        """Ensure important bathymetric features are preserved."""
        # Use scipy.ndimage Laplacian instead of OpenCV
        laplacian_orig = ndimage.laplace(original.astype(np.float64), mode='nearest')
        laplacian_clean = ndimage.laplace(cleaned.astype(np.float64), mode='nearest')
        
        feature_loss = np.abs(laplacian_orig - laplacian_clean)
        return feature_loss > feature_threshold
    
    @staticmethod
    def enforce_monotonicity(data: np.ndarray, direction: str = 'increasing') -> np.ndarray:
        """Enforce monotonic depth changes where appropriate."""
        if direction == 'increasing':
            violations = np.diff(data, axis=0) < 0
        else:
            violations = np.diff(data, axis=0) > 0
        return violations