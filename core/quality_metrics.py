"""
Quality metrics for bathymetric data processing.

This module implements domain-specific quality metrics aligned with 
hydrographic standards and best practices.
"""

import numpy as np
import cv2
import logging
from typing import Dict, Optional, Tuple, Any
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from scipy.stats import pearsonr

from .enums import MetricType, QualityLevel, IHO_STANDARDS


class MetricCalculationError(Exception):
    """Exception raised when metric calculation fails."""
    pass


class BaseQualityMetric:
    """Base class for quality metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray, 
                 **kwargs) -> float:
        """Calculate the metric. Should be implemented by subclasses."""
        raise NotImplementedError
    
    def _validate_inputs(self, original: np.ndarray, processed: np.ndarray):
        """Validate input arrays."""
        if original is None or processed is None:
            raise MetricCalculationError("Input arrays cannot be None")
        
        if original.shape != processed.shape:
            raise MetricCalculationError(f"Shape mismatch: {original.shape} vs {processed.shape}")
        
        if original.size == 0:
            raise MetricCalculationError("Input arrays cannot be empty")


class SSIMMetric(BaseQualityMetric):
    """Structural Similarity Index Metric."""
    
    def __init__(self):
        super().__init__("SSIM")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray, 
                 **kwargs) -> float:
        """Calculate SSIM with comprehensive error handling."""
        try:
            self._validate_inputs(original, processed)
            
            # Ensure we have finite values
            orig_finite = original[np.isfinite(original)]
            proc_finite = processed[np.isfinite(processed)]
            
            if len(orig_finite) == 0 or len(proc_finite) == 0:
                return 0.0
            
            # Calculate data range
            data_range = max(
                float(np.ptp(orig_finite)),
                float(np.ptp(proc_finite)),
                1e-8
            )
            
            # Determine window size
            min_dim = min(original.shape[-2:])
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            win_size = max(3, win_size)  # Ensure minimum size
            
            # Calculate SSIM
            ssim_value = ssim(
                original.astype(np.float64),
                processed.astype(np.float64),
                data_range=data_range,
                gaussian_weights=True,
                win_size=win_size,
                K1=0.01,
                K2=0.03
            )
            
            return float(ssim_value) if np.isfinite(ssim_value) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating SSIM: {e}")
            return 0.0


class RoughnessMetric(BaseQualityMetric):
    """Seafloor roughness metric using standard deviation of slopes."""
    
    def __init__(self):
        super().__init__("Roughness")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray, 
                 **kwargs) -> float:
        """Calculate seafloor roughness."""
        try:
            self._validate_inputs(original, processed)
            
            # Use processed data for roughness calculation
            data = processed if kwargs.get('use_processed', True) else original
            
            # Calculate gradients
            gradient = np.gradient(data)
            slope = np.sqrt(gradient[0]**2 + gradient[1]**2)
            
            # Calculate roughness as standard deviation of slopes
            finite_slopes = slope[np.isfinite(slope)]
            
            if len(finite_slopes) == 0:
                return 0.0
            
            roughness = float(np.std(finite_slopes))
            return roughness
            
        except Exception as e:
            self.logger.error(f"Error calculating roughness: {e}")
            return 0.0


class FeaturePreservationMetric(BaseQualityMetric):
    """Feature preservation metric using Laplacian correlation."""
    
    def __init__(self):
        super().__init__("Feature Preservation")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray, 
                 **kwargs) -> float:
        """Calculate how well bathymetric features are preserved."""
        try:
            self._validate_inputs(original, processed)
            
            # Calculate Laplacians for feature detection
            laplacian_orig = cv2.Laplacian(original.astype(np.float32), cv2.CV_64F)
            laplacian_proc = cv2.Laplacian(processed.astype(np.float32), cv2.CV_64F)
            
            # Flatten and remove invalid values
            orig_flat = laplacian_orig.flatten()
            proc_flat = laplacian_proc.flatten()
            
            # Keep only finite values from both arrays
            valid_mask = np.isfinite(orig_flat) & np.isfinite(proc_flat)
            
            if np.sum(valid_mask) < 10:  # Need at least 10 points for correlation
                return 0.0
            
            orig_valid = orig_flat[valid_mask]
            proc_valid = proc_flat[valid_mask]
            
            # Calculate correlation
            if np.std(orig_valid) < 1e-10 or np.std(proc_valid) < 1e-10:
                # If either has no variation, check if they're similar
                return 1.0 if np.allclose(orig_valid, proc_valid, atol=1e-6) else 0.0
            
            correlation, _ = pearsonr(orig_valid, proc_valid)
            
            return float(correlation) if np.isfinite(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating feature preservation: {e}")
            return 0.0


class ConsistencyMetric(BaseQualityMetric):
    """Depth consistency metric using local variance."""
    
    def __init__(self):
        super().__init__("Consistency")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray, 
                 **kwargs) -> float:
        """Calculate consistency of depth measurements."""
        try:
            self._validate_inputs(original, processed)
            
            # Use processed data for consistency calculation
            data = processed if kwargs.get('use_processed', True) else original
            
            # Calculate local variance using a sliding window
            kernel_size = kwargs.get('kernel_size', 3)
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            
            # Calculate local mean and variance
            local_mean = cv2.filter2D(data.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((data - local_mean)**2, -1, kernel)
            
            # Calculate consistency metric
            finite_variance = local_variance[np.isfinite(local_variance)]
            
            if len(finite_variance) == 0:
                return 0.0
            
            mean_variance = np.mean(finite_variance)
            consistency = 1.0 / (1.0 + mean_variance)
            
            return float(consistency)
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency: {e}")
            return 0.0


class HydrographicComplianceMetric(BaseQualityMetric):
    """IHO hydrographic standards compliance metric."""
    
    def __init__(self, standard: str = "order_1a"):
        super().__init__("Hydrographic Compliance")
        self.standard = standard
        
        if standard not in IHO_STANDARDS:
            self.logger.warning(f"Unknown standard {standard}, using order_1a")
            self.standard = "order_1a"
    
    def calculate(self, original: np.ndarray, processed: np.ndarray, 
                 uncertainty: Optional[np.ndarray] = None, **kwargs) -> float:
        """Calculate compliance with IHO hydrographic standards."""
        try:
            self._validate_inputs(original, processed)
            
            # Use processed data for compliance calculation
            data = processed if kwargs.get('use_processed', True) else original
            
            if uncertainty is None:
                # Estimate uncertainty from local variation
                uncertainty = self._estimate_uncertainty(data)
            
            # Get standard requirements
            standard_info = IHO_STANDARDS[self.standard]
            
            # Calculate depth-dependent uncertainty threshold
            depth_accuracy_formula = standard_info["depth_accuracy"]
            
            # Parse the formula (e.g., "0.5m + 0.013*depth")
            if "+" in depth_accuracy_formula:
                parts = depth_accuracy_formula.split("+")
                base_error = float(parts[0].replace("m", "").strip())
                depth_factor = float(parts[1].replace("*depth", "").strip())
            else:
                base_error = float(depth_accuracy_formula.replace("m", "").strip())
                depth_factor = 0.0
            
            # Calculate required accuracy for each depth
            depths = np.abs(data)
            required_accuracy = base_error + depth_factor * depths
            
            # Check compliance
            compliance_mask = uncertainty <= required_accuracy
            finite_mask = np.isfinite(uncertainty) & np.isfinite(required_accuracy)
            
            if np.sum(finite_mask) == 0:
                return 0.0
            
            compliance_rate = np.sum(compliance_mask & finite_mask) / np.sum(finite_mask)
            
            return float(compliance_rate)
            
        except Exception as e:
            self.logger.error(f"Error calculating hydrographic compliance: {e}")
            return 0.0
    
    def _estimate_uncertainty(self, data: np.ndarray) -> np.ndarray:
        """Estimate uncertainty from local variation."""
        try:
            kernel = np.ones((3, 3)) / 9
            local_mean = cv2.filter2D(data.astype(np.float32), -1, kernel)
            local_std = cv2.filter2D((data - local_mean)**2, -1, kernel)
            uncertainty = np.sqrt(local_std)
            return uncertainty
        except Exception as e:
            self.logger.error(f"Error estimating uncertainty: {e}")
            return np.full_like(data, 1.0)


class NoiseReductionMetric(BaseQualityMetric):
    """Metric to assess noise reduction effectiveness."""
    
    def __init__(self):
        super().__init__("Noise Reduction")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray, 
                 **kwargs) -> float:
        """Calculate noise reduction effectiveness."""
        try:
            self._validate_inputs(original, processed)
            
            # Estimate noise levels in both images
            original_noise = self._estimate_noise_level(original)
            processed_noise = self._estimate_noise_level(processed)
            
            if original_noise <= 0:
                return 1.0  # No noise to reduce
            
            # Calculate noise reduction ratio
            noise_reduction = max(0, (original_noise - processed_noise) / original_noise)
            
            return float(noise_reduction)
            
        except Exception as e:
            self.logger.error(f"Error calculating noise reduction: {e}")
            return 0.0
    
    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level using high-frequency content."""
        try:
            # Use Laplacian as high-frequency filter
            laplacian = cv2.Laplacian(data.astype(np.float32), cv2.CV_64F)
            finite_laplacian = laplacian[np.isfinite(laplacian)]
            
            if len(finite_laplacian) == 0:
                return 0.0
            
            # Use median absolute deviation as robust noise estimate
            median_val = np.median(finite_laplacian)
            mad = np.median(np.abs(finite_laplacian - median_val))
            
            # Convert MAD to standard deviation equivalent
            noise_level = 1.4826 * mad
            
            return float(noise_level)
            
        except Exception as e:
            self.logger.error(f"Error estimating noise level: {e}")
            return 0.0


class EdgePreservationMetric(BaseQualityMetric):
    """Metric to assess edge preservation quality."""
    
    def __init__(self):
        super().__init__("Edge Preservation")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray, 
                 **kwargs) -> float:
        """Calculate edge preservation quality."""
        try:
            self._validate_inputs(original, processed)
            
            # Detect edges in both images
            original_edges = self._detect_edges(original)
            processed_edges = self._detect_edges(processed)
            
            # Calculate edge preservation ratio
            original_edge_strength = np.sum(original_edges)
            processed_edge_strength = np.sum(processed_edges)
            
            if original_edge_strength == 0:
                return 1.0  # No edges to preserve
            
            preservation_ratio = min(1.0, processed_edge_strength / original_edge_strength)
            
            return float(preservation_ratio)
            
        except Exception as e:
            self.logger.error(f"Error calculating edge preservation: {e}")
            return 0.0
    
    def _detect_edges(self, data: np.ndarray) -> np.ndarray:
        """Detect edges using Sobel operator."""
        try:
            # Apply Sobel edge detection
            sobel_x = cv2.Sobel(data.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(data.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate edge magnitude
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            return edge_magnitude
            
        except Exception as e:
            self.logger.error(f"Error detecting edges: {e}")
            return np.zeros_like(data)


class BathymetricQualityMetrics:
    """Domain-specific quality metrics for bathymetric data."""
    
    def __init__(self, iho_standard: str = "order_1a"):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize individual metrics
        self.metrics = {
            MetricType.SSIM: SSIMMetric(),
            MetricType.ROUGHNESS: RoughnessMetric(),
            MetricType.FEATURE_PRESERVATION: FeaturePreservationMetric(),
            MetricType.CONSISTENCY: ConsistencyMetric(),
            MetricType.HYDROGRAPHIC_COMPLIANCE: HydrographicComplianceMetric(iho_standard)
        }
        
        # Additional metrics
        self.additional_metrics = {
            'noise_reduction': NoiseReductionMetric(),
            'edge_preservation': EdgePreservationMetric()
        }
    
    def calculate_all_metrics(self, original: np.ndarray, processed: np.ndarray,
                            uncertainty: Optional[np.ndarray] = None,
                            weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate all quality metrics."""
        results = {}
        
        # Calculate primary metrics
        for metric_type, metric in self.metrics.items():
            try:
                if metric_type == MetricType.HYDROGRAPHIC_COMPLIANCE:
                    value = metric.calculate(original, processed, uncertainty=uncertainty)
                else:
                    value = metric.calculate(original, processed)
                
                results[metric_type.value] = value
                
            except Exception as e:
                self.logger.error(f"Error calculating {metric_type.value}: {e}")
                results[metric_type.value] = 0.0
        
        # Calculate additional metrics
        for name, metric in self.additional_metrics.items():
            try:
                results[name] = metric.calculate(original, processed)
            except Exception as e:
                self.logger.error(f"Error calculating {name}: {e}")
                results[name] = 0.0
        
        # Calculate basic error metrics
        try:
            results.update(self._calculate_basic_metrics(original, processed))
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
        
        # Calculate composite quality score
        if weights is None:
            weights = {
                'ssim': 0.3,
                'roughness': 0.2,
                'feature_preservation': 0.3,
                'consistency': 0.2
            }
        
        results['composite_quality'] = self._calculate_composite_quality(results, weights)
        
        return results
    
    def _calculate_basic_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Calculate basic error metrics."""
        try:
            # Ensure finite values
            orig_finite = original[np.isfinite(original) & np.isfinite(processed)]
            proc_finite = processed[np.isfinite(original) & np.isfinite(processed)]
            
            if len(orig_finite) == 0:
                return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
            
            # Mean Absolute Error
            mae = float(np.mean(np.abs(orig_finite - proc_finite)))
            
            # Root Mean Square Error
            rmse = float(np.sqrt(np.mean((orig_finite - proc_finite)**2)))
            
            # Mean Absolute Percentage Error (avoid division by zero)
            non_zero_mask = np.abs(orig_finite) > 1e-10
            if np.any(non_zero_mask):
                mape = float(np.mean(np.abs((orig_finite[non_zero_mask] - proc_finite[non_zero_mask]) / 
                                          orig_finite[non_zero_mask])) * 100)
            else:
                mape = 0.0
            
            return {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
            return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
    
    def _calculate_composite_quality(self, metrics: Dict[str, float], 
                                   weights: Dict[str, float]) -> float:
        """Calculate weighted composite quality score."""
        try:
            total_weight = 0.0
            weighted_sum = 0.0
            
            for metric_name, weight in weights.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    
                    # Normalize roughness (lower is better)
                    if metric_name == 'roughness':
                        value = 1.0 / (1.0 + value)
                    
                    weighted_sum += weight * value
                    total_weight += weight
            
            if total_weight > 0:
                return float(weighted_sum / total_weight)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating composite quality: {e}")
            return 0.0
    
    def assess_quality_level(self, composite_score: float) -> QualityLevel:
        """Assess quality level from composite score."""
        return QualityLevel.from_score(composite_score)
    
    def calculate_uncertainty_metrics(self, uncertainty_data: np.ndarray, 
                                    processed_data: np.ndarray) -> Dict[str, float]:
        """Calculate uncertainty-specific metrics."""
        try:
            if uncertainty_data is None or processed_data is None:
                return {}
            
            finite_uncertainty = uncertainty_data[np.isfinite(uncertainty_data)]
            finite_processed = processed_data[np.isfinite(processed_data)]
            
            if len(finite_uncertainty) == 0:
                return {}
            
            metrics = {
                'mean_uncertainty': float(np.mean(finite_uncertainty)),
                'std_uncertainty': float(np.std(finite_uncertainty)),
                'max_uncertainty': float(np.max(finite_uncertainty)),
                'min_uncertainty': float(np.min(finite_uncertainty)),
                'uncertainty_range': float(np.ptp(finite_uncertainty))
            }
            
            # Calculate uncertainty-depth correlation if both arrays have same valid locations
            valid_mask = np.isfinite(uncertainty_data) & np.isfinite(processed_data)
            if np.sum(valid_mask) > 10:
                uncertainty_valid = uncertainty_data[valid_mask]
                processed_valid = processed_data[valid_mask]
                
                if np.std(uncertainty_valid) > 1e-10 and np.std(processed_valid) > 1e-10:
                    correlation, _ = pearsonr(uncertainty_valid, np.abs(processed_valid))
                    metrics['uncertainty_depth_correlation'] = float(correlation) if np.isfinite(correlation) else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating uncertainty metrics: {e}")
            return {}
    
    def generate_quality_report(self, original: np.ndarray, processed: np.ndarray,
                              uncertainty: Optional[np.ndarray] = None,
                              weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        try:
            # Calculate all metrics
            quality_metrics = self.calculate_all_metrics(original, processed, uncertainty, weights)
            
            # Assess quality level
            composite_score = quality_metrics.get('composite_quality', 0.0)
            quality_level = self.assess_quality_level(composite_score)
            
            # Calculate uncertainty metrics if available
            uncertainty_metrics = {}
            if uncertainty is not None:
                uncertainty_metrics = self.calculate_uncertainty_metrics(uncertainty, processed)
            
            # Create report
            report = {
                'quality_metrics': quality_metrics,
                'uncertainty_metrics': uncertainty_metrics,
                'quality_level': quality_level.value,
                'quality_score': composite_score,
                'assessment': {
                    'is_acceptable': quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD, QualityLevel.ACCEPTABLE],
                    'recommendations': self._generate_recommendations(quality_metrics, quality_level)
                },
                'data_statistics': {
                    'original_shape': original.shape,
                    'processed_shape': processed.shape,
                    'valid_pixels_original': int(np.sum(np.isfinite(original))),
                    'valid_pixels_processed': int(np.sum(np.isfinite(processed))),
                    'completeness': float(np.sum(np.isfinite(processed)) / processed.size)
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, metrics: Dict[str, float], 
                                quality_level: QualityLevel) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        try:
            if quality_level == QualityLevel.UNACCEPTABLE:
                recommendations.append("Data quality is unacceptable - consider reprocessing with different parameters")
            
            # Specific metric-based recommendations
            if metrics.get('ssim', 0) < 0.7:
                recommendations.append("Low structural similarity - consider reducing smoothing or improving noise reduction")
            
            if metrics.get('feature_preservation', 0) < 0.6:
                recommendations.append("Poor feature preservation - reduce smoothing factor or increase edge preservation")
            
            if metrics.get('consistency', 0) < 0.6:
                recommendations.append("Low consistency - consider additional noise reduction or smoothing")
            
            if metrics.get('hydrographic_compliance', 0) < 0.8:
                recommendations.append("Does not meet hydrographic standards - review processing parameters")
            
            if metrics.get('roughness', 0) > 0.1:
                recommendations.append("High roughness detected - consider increasing smoothing")
            
            # Positive feedback
            if quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]:
                recommendations.append("Good quality results - parameters are well-tuned")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error")
        
        return recommendations
    
    # Static methods for backward compatibility
    @staticmethod
    def calculate_roughness(data: np.ndarray) -> float:
        """Static method for calculating roughness."""
        metric = RoughnessMetric()
        return metric.calculate(data, data, use_processed=True)
    
    @staticmethod
    def calculate_feature_preservation(original: np.ndarray, cleaned: np.ndarray) -> float:
        """Static method for calculating feature preservation."""
        metric = FeaturePreservationMetric()
        return metric.calculate(original, cleaned)
    
    @staticmethod
    def calculate_depth_consistency(data: np.ndarray) -> float:
        """Static method for calculating depth consistency."""
        metric = ConsistencyMetric()
        return metric.calculate(data, data, use_processed=True)
    
    @staticmethod
    def calculate_hydrographic_standards_compliance(data: np.ndarray, 
                                                  uncertainty: Optional[np.ndarray] = None) -> float:
        """Static method for calculating hydrographic compliance."""
        metric = HydrographicComplianceMetric()
        return metric.calculate(data, data, uncertainty=uncertainty, use_processed=True)


def create_quality_metrics_calculator(iho_standard: str = "order_1a") -> BathymetricQualityMetrics:
    """Factory function to create quality metrics calculator."""
    return BathymetricQualityMetrics(iho_standard=iho_standard)
    