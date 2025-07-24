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
            original = BathymetricQualityMetrics._ensure_compatible_array(original)
            cleaned = BathymetricQualityMetrics._ensure_compatible_array(cleaned)
            
            if original.shape != cleaned.shape:
                logging.warning("Shape mismatch in SSIM calculation")
                return 0.0
            
            if original.size == 0:
                return 0.0
            
            # Handle non-finite values
            orig_finite = np.isfinite(original)
            clean_finite = np.isfinite(cleaned)
            valid_mask = orig_finite & clean_finite
            
            if np.sum(valid_mask) < 10:  # Need minimum pixels for meaningful SSIM
                return 0.0
            
            # Extract valid regions
            orig_valid = original[valid_mask]
            clean_valid = cleaned[valid_mask]
            
            if len(orig_valid) < 2:
                return 0.0
            
            # Use skimage SSIM with error handling
            try:
                ssim_value = ssim(original, cleaned, data_range=cleaned.max() - cleaned.min())
                return float(max(0.0, ssim_value))
            except Exception:
                # Fallback to correlation-based similarity
                if np.std(orig_valid) == 0 or np.std(clean_valid) == 0:
                    return 1.0 if np.allclose(orig_valid, clean_valid) else 0.0
                
                correlation = np.corrcoef(orig_valid, clean_valid)[0, 1]
                return float(max(0.0, correlation)) if np.isfinite(correlation) else 0.0
                
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
            
            # Handle non-finite values
            finite_data = data[np.isfinite(data)]
            if len(finite_data) == 0:
                return 0.0
            
            # Calculate local variance using a moving window
            from scipy import ndimage
            
            # Use a 3x3 kernel for local variance calculation
            kernel = np.ones((3, 3)) / 9
            local_mean = ndimage.convolve(data, kernel, mode='constant', cval=np.nan)
            local_variance = ndimage.convolve((data - local_mean)**2, kernel, mode='constant', cval=np.nan)
            
            # Calculate consistency as inverse of normalized variance
            finite_variance = local_variance[np.isfinite(local_variance)]
            if len(finite_variance) == 0:
                return 1.0  # Perfectly consistent if no variance can be calculated
            
            mean_variance = np.mean(finite_variance)
            max_variance = np.max(finite_variance)
            
            if max_variance == 0:
                return 1.0  # Perfectly consistent
            
            consistency = 1.0 - (mean_variance / max_variance)
            return float(max(0.0, min(1.0, consistency)))
            
        except Exception as e:
            logging.error(f"Error calculating depth consistency: {e}")
            return 0.0
    
    @staticmethod
    def calculate_hydrographic_standards_compliance(depth_data: np.ndarray, uncertainty_data: Optional[np.ndarray] = None) -> float:
        """Calculate compliance with hydrographic standards."""
        try:
            depth_data = BathymetricQualityMetrics._ensure_compatible_array(depth_data)
            
            if depth_data.size == 0:
                return 0.0
            
            # Handle non-finite values
            finite_depth = depth_data[np.isfinite(depth_data)]
            if len(finite_depth) == 0:
                return 0.0
            
            # IHO S-44 standards (simplified version)
            # Different accuracy requirements based on depth ranges
            compliance_scores = []
            
            # Shallow water (0-20m): ±0.25m + 0.0075 * depth
            shallow_mask = (finite_depth >= 0) & (finite_depth <= 20)
            if np.any(shallow_mask):
                shallow_depths = finite_depth[shallow_mask]
                required_accuracy = 0.25 + 0.0075 * shallow_depths
                if uncertainty_data is not None:
                    uncertainty_data = BathymetricQualityMetrics._ensure_compatible_array(uncertainty_data)
                    if uncertainty_data.shape == depth_data.shape:
                        shallow_uncertainty = uncertainty_data.flatten()[shallow_mask]
                        compliance = np.mean(shallow_uncertainty <= required_accuracy)
                        compliance_scores.append(compliance)
            
            # Medium depth (20-100m): ±0.5m + 0.01 * depth  
            medium_mask = (finite_depth > 20) & (finite_depth <= 100)
            if np.any(medium_mask):
                medium_depths = finite_depth[medium_mask]
                required_accuracy = 0.5 + 0.01 * medium_depths
                if uncertainty_data is not None:
                    uncertainty_data = BathymetricQualityMetrics._ensure_compatible_array(uncertainty_data)
                    if uncertainty_data.shape == depth_data.shape:
                        medium_uncertainty = uncertainty_data.flatten()[medium_mask]
                        compliance = np.mean(medium_uncertainty <= required_accuracy)
                        compliance_scores.append(compliance)
            
            # Deep water (>100m): ±1.0m + 0.02 * depth
            deep_mask = finite_depth > 100
            if np.any(deep_mask):
                deep_depths = finite_depth[deep_mask]
                required_accuracy = 1.0 + 0.02 * deep_depths
                if uncertainty_data is not None:
                    uncertainty_data = BathymetricQualityMetrics._ensure_compatible_array(uncertainty_data)
                    if uncertainty_data.shape == depth_data.shape:
                        deep_uncertainty = uncertainty_data.flatten()[deep_mask]
                        compliance = np.mean(deep_uncertainty <= required_accuracy)
                        compliance_scores.append(compliance)
            
            if compliance_scores:
                return float(np.mean(compliance_scores))
            else:
                # If no uncertainty data, use depth gradient consistency as proxy
                grad_y, grad_x = np.gradient(depth_data)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                finite_gradients = gradient_magnitude[np.isfinite(gradient_magnitude)]
                
                if len(finite_gradients) == 0:
                    return 0.5  # Neutral score
                
                # Lower gradients indicate smoother, more compliant data
                mean_gradient = np.mean(finite_gradients)
                max_gradient = np.max(finite_gradients)
                
                if max_gradient == 0:
                    return 1.0
                
                smoothness = 1.0 - (mean_gradient / max_gradient)
                return float(max(0.0, min(1.0, smoothness)))
                
        except Exception as e:
            logging.error(f"Error calculating hydrographic standards compliance: {e}")
            return 0.0

    @staticmethod
    def calculate_anthropogenic_preservation(original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate preservation score for man-made features."""
        try:
            from scipy import ndimage
            
            # Detect high-contrast features (likely anthropogenic)
            orig_edges = ndimage.sobel(original)
            proc_edges = ndimage.sobel(processed)
            
            # Focus on very strong edges
            strong_edge_threshold = np.percentile(np.abs(orig_edges), 95)
            orig_strong = np.abs(orig_edges) > strong_edge_threshold
            proc_strong = np.abs(proc_edges) > strong_edge_threshold
            
            if not np.any(orig_strong):
                return 1.0  # No anthropogenic features to preserve
            
            # Calculate preservation ratio
            preserved_ratio = np.sum(orig_strong & proc_strong) / np.sum(orig_strong)
            return float(preserved_ratio)
            
        except Exception as e:
            logging.error(f"Error calculating anthropogenic preservation: {e}")
            return 0.0

    @staticmethod
    def calculate_geometric_structure_preservation(original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate preservation of geometric structures (docks, piers, etc.)."""
        try:
            # Detect regular geometric patterns
            from scipy import ndimage
            
            # Use multiple structure detection kernels
            structure_kernels = [
                np.ones((3, 7)) / 21,  # Horizontal structures
                np.ones((7, 3)) / 21,  # Vertical structures
                np.ones((5, 5)) / 25   # Square structures
            ]
            
            correlations = []
            for kernel in structure_kernels:
                orig_conv = ndimage.convolve(original, kernel)
                proc_conv = ndimage.convolve(processed, kernel)
                
                if np.std(orig_conv) > 0 and np.std(proc_conv) > 0:
                    corr = np.corrcoef(orig_conv.flatten(), proc_conv.flatten())[0, 1]
                    if np.isfinite(corr):
                        correlations.append(max(0.0, corr))
            
            return float(np.mean(correlations)) if correlations else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating geometric structure preservation: {e}")
            return 0.0
    
    def calculate_composite_quality(self, original: np.ndarray, processed: np.ndarray, 
                                   uncertainty: Optional[np.ndarray] = None, 
                                   weights: Optional[dict] = None) -> dict:
        """Calculate comprehensive quality metrics including composite score.
        
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
        
        # Convert to float32 for consistency
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        return data