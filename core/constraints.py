"""
Constitutional AI constraints for bathymetric data processing.
"""

import numpy as np
import cv2


class BathymetricConstraints:
    """Constitutional AI rules for bathymetric data cleaning."""
    
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
        # Detect significant depth features using Laplacian
        laplacian_orig = cv2.Laplacian(original.astype(np.float32), cv2.CV_64F)
        laplacian_clean = cv2.Laplacian(cleaned.astype(np.float32), cv2.CV_64F)
        
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
