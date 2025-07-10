"""
Constitutional AI constraints for bathymetric data processing.

This module implements domain-specific constraints to ensure processed bathymetric
data maintains physical plausibility and scientific integrity.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

from .enums import SeafloorType


class ConstraintViolation:
    """Represents a constraint violation."""
    
    def __init__(self, constraint_type: str, severity: float, 
                 location: Optional[Tuple[int, int]] = None,
                 description: str = ""):
        self.constraint_type = constraint_type
        self.severity = severity  # 0.0 to 1.0
        self.location = location
        self.description = description
        self.timestamp = np.datetime64('now')
    
    def __repr__(self):
        return (f"ConstraintViolation(type={self.constraint_type}, "
                f"severity={self.severity:.3f}, location={self.location})")


class BathymetricConstraint(ABC):
    """Abstract base class for bathymetric constraints."""
    
    @abstractmethod
    def validate(self, data: np.ndarray, **kwargs) -> Tuple[bool, Optional[ConstraintViolation]]:
        """Validate data against this constraint."""
        pass
    
    @abstractmethod
    def apply_correction(self, original: np.ndarray, processed: np.ndarray, 
                        **kwargs) -> np.ndarray:
        """Apply correction to enforce this constraint."""
        pass


class DepthContinuityConstraint(BathymetricConstraint):
    """Ensures depth changes are physically plausible."""
    
    def __init__(self, max_gradient: float = 0.1, seafloor_type: Optional[SeafloorType] = None):
        self.max_gradient = max_gradient
        self.seafloor_type = seafloor_type
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Adjust gradient based on seafloor type
        if seafloor_type:
            self._adjust_gradient_for_seafloor()
    
    def _adjust_gradient_for_seafloor(self):
        """Adjust gradient threshold based on seafloor type."""
        adjustments = {
            SeafloorType.SHALLOW_COASTAL: 0.05,  # Stricter for shallow areas
            SeafloorType.CONTINENTAL_SHELF: 0.08,
            SeafloorType.DEEP_OCEAN: 0.1,
            SeafloorType.SEAMOUNT: 0.15,  # Allow steeper gradients
            SeafloorType.ABYSSAL_PLAIN: 0.05,  # Very flat areas
            SeafloorType.UNKNOWN: 0.1
        }
        self.max_gradient = adjustments.get(self.seafloor_type, 0.1)
    
    def validate(self, data: np.ndarray, **kwargs) -> Tuple[bool, Optional[ConstraintViolation]]:
        """Validate depth continuity."""
        try:
            violation_mask = self._calculate_violations(data)
            
            if np.any(violation_mask):
                severity = np.sum(violation_mask) / violation_mask.size
                violation = ConstraintViolation(
                    constraint_type="depth_continuity",
                    severity=float(severity),
                    description=f"Gradient violations: {np.sum(violation_mask)} pixels"
                )
                return False, violation
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error validating depth continuity: {e}")
            return False, ConstraintViolation(
                constraint_type="depth_continuity",
                severity=1.0,
                description=f"Validation error: {e}"
            )
    
    def _calculate_violations(self, data: np.ndarray) -> np.ndarray:
        """Calculate gradient violations."""
        if data.size == 0:
            return np.array([])
        
        # Calculate gradients
        gradient = np.gradient(data)
        gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
        
        # Find violations
        violation_mask = gradient_magnitude > self.max_gradient
        
        return violation_mask
    
    def apply_correction(self, original: np.ndarray, processed: np.ndarray, 
                        **kwargs) -> np.ndarray:
        """Apply correction to enforce depth continuity."""
        try:
            violation_mask = self._calculate_violations(processed)
            
            if not np.any(violation_mask):
                return processed
            
            corrected = processed.copy()
            
            # Apply smoothing to violation areas
            blend_factor = kwargs.get('blend_factor', 0.5)
            
            # Use Gaussian smoothing on violation areas
            smoothed = cv2.GaussianBlur(processed.astype(np.float32), (5, 5), 1.0)
            
            # Blend original, smoothed, and processed data
            corrected[violation_mask] = (
                blend_factor * original[violation_mask] +
                (1 - blend_factor) * 0.5 * (processed[violation_mask] + smoothed[violation_mask])
            )
            
            self.logger.info(f"Applied depth continuity correction to {np.sum(violation_mask)} pixels")
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"Error applying depth continuity correction: {e}")
            return processed


class FeaturePreservationConstraint(BathymetricConstraint):
    """Ensures important bathymetric features are preserved."""
    
    def __init__(self, feature_threshold: float = 0.05):
        self.feature_threshold = feature_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, data: np.ndarray, original: np.ndarray = None, **kwargs) -> Tuple[bool, Optional[ConstraintViolation]]:
        """Validate feature preservation."""
        if original is None:
            return True, None
        
        try:
            violation_mask = self._detect_feature_loss(original, data)
            
            if np.any(violation_mask):
                severity = np.sum(violation_mask) / violation_mask.size
                violation = ConstraintViolation(
                    constraint_type="feature_preservation",
                    severity=float(severity),
                    description=f"Feature loss detected: {np.sum(violation_mask)} pixels"
                )
                return False, violation
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error validating feature preservation: {e}")
            return False, ConstraintViolation(
                constraint_type="feature_preservation",
                severity=1.0,
                description=f"Validation error: {e}"
            )
    
    def _detect_feature_loss(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Detect significant feature loss using Laplacian."""
        try:
            # Calculate Laplacian for feature detection
            laplacian_orig = cv2.Laplacian(original.astype(np.float32), cv2.CV_64F)
            laplacian_proc = cv2.Laplacian(processed.astype(np.float32), cv2.CV_64F)
            
            # Calculate feature loss
            feature_loss = np.abs(laplacian_orig - laplacian_proc)
            
            # Normalize by original feature strength
            orig_strength = np.abs(laplacian_orig)
            normalized_loss = np.divide(
                feature_loss, 
                orig_strength + 1e-8,  # Avoid division by zero
                out=np.zeros_like(feature_loss),
                where=(orig_strength > 1e-8)
            )
            
            # Identify significant feature loss
            violation_mask = normalized_loss > self.feature_threshold
            
            return violation_mask
            
        except Exception as e:
            self.logger.error(f"Error detecting feature loss: {e}")
            return np.zeros_like(original, dtype=bool)
    
    def apply_correction(self, original: np.ndarray, processed: np.ndarray, 
                        **kwargs) -> np.ndarray:
        """Apply correction to preserve features."""
        try:
            violation_mask = self._detect_feature_loss(original, processed)
            
            if not np.any(violation_mask):
                return processed
            
            corrected = processed.copy()
            
            # Restore features by blending with original
            feature_weight = kwargs.get('feature_weight', 0.7)
            corrected[violation_mask] = (
                feature_weight * original[violation_mask] +
                (1 - feature_weight) * processed[violation_mask]
            )
            
            self.logger.info(f"Applied feature preservation correction to {np.sum(violation_mask)} pixels")
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"Error applying feature preservation correction: {e}")
            return processed


class MonotonicityConstraint(BathymetricConstraint):
    """Enforces monotonic depth changes where appropriate."""
    
    def __init__(self, direction: str = 'increasing', tolerance: float = 0.01):
        self.direction = direction
        self.tolerance = tolerance
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, data: np.ndarray, **kwargs) -> Tuple[bool, Optional[ConstraintViolation]]:
        """Validate monotonicity."""
        try:
            violations = self._find_monotonicity_violations(data)
            
            if np.any(violations):
                severity = np.sum(violations) / max(violations.size, 1)
                violation = ConstraintViolation(
                    constraint_type="monotonicity",
                    severity=float(severity),
                    description=f"Monotonicity violations: {np.sum(violations)} locations"
                )
                return False, violation
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error validating monotonicity: {e}")
            return False, ConstraintViolation(
                constraint_type="monotonicity",
                severity=1.0,
                description=f"Validation error: {e}"
            )
    
    def _find_monotonicity_violations(self, data: np.ndarray) -> np.ndarray:
        """Find violations of monotonicity constraint."""
        if data.size == 0:
            return np.array([])
        
        # Calculate differences along both axes
        diff_x = np.diff(data, axis=1)
        diff_y = np.diff(data, axis=0)
        
        if self.direction == 'increasing':
            violations_x = diff_x < -self.tolerance
            violations_y = diff_y < -self.tolerance
        else:  # decreasing
            violations_x = diff_x > self.tolerance
            violations_y = diff_y > self.tolerance
        
        # Pad to match original shape
        violations_x = np.pad(violations_x, ((0, 0), (0, 1)), mode='constant', constant_values=False)
        violations_y = np.pad(violations_y, ((0, 1), (0, 0)), mode='constant', constant_values=False)
        
        return violations_x | violations_y
    
    def apply_correction(self, original: np.ndarray, processed: np.ndarray, 
                        **kwargs) -> np.ndarray:
        """Apply monotonicity correction."""
        try:
            violations = self._find_monotonicity_violations(processed)
            
            if not np.any(violations):
                return processed
            
            corrected = processed.copy()
            
            # Apply smoothing to violation areas
            kernel_size = kwargs.get('kernel_size', 3)
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            smoothed = cv2.filter2D(processed.astype(np.float32), -1, kernel)
            
            corrected[violations] = smoothed[violations]
            
            self.logger.info(f"Applied monotonicity correction to {np.sum(violations)} pixels")
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"Error applying monotonicity correction: {e}")
            return processed


class PhysicalPlausibilityConstraint(BathymetricConstraint):
    """Ensures depth values are physically plausible."""
    
    def __init__(self, min_depth: float = -11000, max_depth: float = 9000):
        self.min_depth = min_depth  # Challenger Deep is ~-11000m
        self.max_depth = max_depth   # Allow for some land areas
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, data: np.ndarray, **kwargs) -> Tuple[bool, Optional[ConstraintViolation]]:
        """Validate physical plausibility."""
        try:
            violations = self._find_implausible_values(data)
            
            if np.any(violations):
                severity = np.sum(violations) / violations.size
                violation = ConstraintViolation(
                    constraint_type="physical_plausibility",
                    severity=float(severity),
                    description=f"Implausible values: {np.sum(violations)} pixels"
                )
                return False, violation
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error validating physical plausibility: {e}")
            return False, ConstraintViolation(
                constraint_type="physical_plausibility",
                severity=1.0,
                description=f"Validation error: {e}"
            )
    
    def _find_implausible_values(self, data: np.ndarray) -> np.ndarray:
        """Find physically implausible depth values."""
        return (data < self.min_depth) | (data > self.max_depth)
    
    def apply_correction(self, original: np.ndarray, processed: np.ndarray, 
                        **kwargs) -> np.ndarray:
        """Apply physical plausibility correction."""
        try:
            violations = self._find_implausible_values(processed)
            
            if not np.any(violations):
                return processed
            
            corrected = processed.copy()
            
            # Clamp to valid range or use original values
            use_original = kwargs.get('use_original_for_implausible', True)
            
            if use_original:
                corrected[violations] = original[violations]
            else:
                corrected = np.clip(corrected, self.min_depth, self.max_depth)
            
            self.logger.info(f"Applied physical plausibility correction to {np.sum(violations)} pixels")
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"Error applying physical plausibility correction: {e}")
            return processed


class BathymetricConstraints:
    """Constitutional AI rules for bathymetric data cleaning."""
    
    def __init__(self, seafloor_type: Optional[SeafloorType] = None):
        self.seafloor_type = seafloor_type
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize constraints
        self.constraints = {
            'depth_continuity': DepthContinuityConstraint(seafloor_type=seafloor_type),
            'feature_preservation': FeaturePreservationConstraint(),
            'monotonicity': MonotonicityConstraint(),
            'physical_plausibility': PhysicalPlausibilityConstraint()
        }
        
        self.violation_history = []
    
    def validate_all(self, data: np.ndarray, original: np.ndarray = None, 
                    **kwargs) -> Dict[str, ConstraintViolation]:
        """Validate data against all constraints."""
        violations = {}
        
        for name, constraint in self.constraints.items():
            try:
                if name == 'feature_preservation' and original is not None:
                    is_valid, violation = constraint.validate(data, original=original, **kwargs)
                else:
                    is_valid, violation = constraint.validate(data, **kwargs)
                
                if not is_valid and violation:
                    violations[name] = violation
                    
            except Exception as e:
                self.logger.error(f"Error validating constraint {name}: {e}")
                violations[name] = ConstraintViolation(
                    constraint_type=name,
                    severity=1.0,
                    description=f"Validation error: {e}"
                )
        
        # Store violations in history
        if violations:
            self.violation_history.extend(violations.values())
        
        return violations
    
    def apply_corrections(self, original: np.ndarray, processed: np.ndarray, 
                         **kwargs) -> np.ndarray:
        """Apply all constraint corrections."""
        corrected = processed.copy()
        
        # Apply corrections in order of importance
        correction_order = [
            'physical_plausibility',
            'depth_continuity',
            'feature_preservation',
            'monotonicity'
        ]
        
        for constraint_name in correction_order:
            if constraint_name in self.constraints:
                try:
                    constraint = self.constraints[constraint_name]
                    corrected = constraint.apply_correction(original, corrected, **kwargs)
                except Exception as e:
                    self.logger.error(f"Error applying {constraint_name} correction: {e}")
        
        return corrected
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations."""
        if not self.violation_history:
            return {'total_violations': 0}
        
        summary = {
            'total_violations': len(self.violation_history),
            'by_type': {},
            'average_severity': 0.0,
            'max_severity': 0.0
        }
        
        # Count by type
        for violation in self.violation_history:
            constraint_type = violation.constraint_type
            if constraint_type not in summary['by_type']:
                summary['by_type'][constraint_type] = 0
            summary['by_type'][constraint_type] += 1
        
        # Calculate severity statistics
        severities = [v.severity for v in self.violation_history]
        summary['average_severity'] = float(np.mean(severities))
        summary['max_severity'] = float(np.max(severities))
        
        return summary
    
    def clear_violation_history(self):
        """Clear the violation history."""
        self.violation_history.clear()
    
    @staticmethod
    def validate_depth_continuity(data: np.ndarray, max_gradient: float = 0.1) -> np.ndarray:
        """Static method for backward compatibility."""
        constraint = DepthContinuityConstraint(max_gradient=max_gradient)
        violations = constraint._calculate_violations(data)
        return violations
    
    @staticmethod
    def preserve_depth_features(original: np.ndarray, cleaned: np.ndarray, 
                               feature_threshold: float = 0.05) -> np.ndarray:
        """Static method for backward compatibility."""
        constraint = FeaturePreservationConstraint(feature_threshold=feature_threshold)
        violations = constraint._detect_feature_loss(original, cleaned)
        return violations
    
    @staticmethod
    def enforce_monotonicity(data: np.ndarray, direction: str = 'increasing') -> np.ndarray:
        """Static method for backward compatibility."""
        constraint = MonotonicityConstraint(direction=direction)
        violations = constraint._find_monotonicity_violations(data)
        return violations