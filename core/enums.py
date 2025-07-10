"""
Enumerations and constants for Enhanced Bathymetric CAE Processing.

This module contains all enumeration types used throughout the application.
"""

from enum import Enum
from typing import Dict, Tuple


class SeafloorType(Enum):
    """Seafloor environment types for adaptive processing."""
    SHALLOW_COASTAL = "shallow_coastal"
    DEEP_OCEAN = "deep_ocean"
    CONTINENTAL_SHELF = "continental_shelf"
    SEAMOUNT = "seamount"
    ABYSSAL_PLAIN = "abyssal_plain"
    UNKNOWN = "unknown"
    
    @property
    def depth_range(self) -> Tuple[float, float]:
        """Get the typical depth range for this seafloor type."""
        ranges = {
            self.SHALLOW_COASTAL: (0, 200),
            self.CONTINENTAL_SHELF: (200, 2000),
            self.DEEP_OCEAN: (2000, 6000),
            self.ABYSSAL_PLAIN: (6000, 11000),
            self.SEAMOUNT: (0, 6000),  # Special case based on topology
            self.UNKNOWN: (0, 11000)
        }
        return ranges[self]
    
    @property
    def description(self) -> str:
        """Get a human-readable description of the seafloor type."""
        descriptions = {
            self.SHALLOW_COASTAL: "Shallow coastal waters with high variability",
            self.CONTINENTAL_SHELF: "Continental shelf with moderate slopes",
            self.DEEP_OCEAN: "Deep ocean basins with variable topography",
            self.ABYSSAL_PLAIN: "Deep, relatively flat abyssal plains",
            self.SEAMOUNT: "Underwater mountains with steep slopes",
            self.UNKNOWN: "Unclassified or mixed seafloor type"
        }
        return descriptions[self]


class ProcessingStrategy(Enum):
    """Processing strategy types."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"
    
    @property
    def threshold(self) -> float:
        """Get the quality threshold for this level."""
        thresholds = {
            self.EXCELLENT: 0.9,
            self.GOOD: 0.8,
            self.ACCEPTABLE: 0.7,
            self.POOR: 0.5,
            self.UNACCEPTABLE: 0.0
        }
        return thresholds[self]
    
    @classmethod
    def from_score(cls, score: float) -> 'QualityLevel':
        """Determine quality level from score."""
        if score >= cls.EXCELLENT.threshold:
            return cls.EXCELLENT
        elif score >= cls.GOOD.threshold:
            return cls.GOOD
        elif score >= cls.ACCEPTABLE.threshold:
            return cls.ACCEPTABLE
        elif score >= cls.POOR.threshold:
            return cls.POOR
        else:
            return cls.UNACCEPTABLE


class ExpertReviewFlag(Enum):
    """Types of expert review flags."""
    LOW_QUALITY = "low_quality"
    FEATURE_LOSS = "feature_loss"
    STANDARDS_VIOLATION = "standards_violation"
    POOR_SIMILARITY = "poor_similarity"
    HIGH_UNCERTAINTY = "high_uncertainty"
    PROCESSING_ERROR = "processing_error"
    MANUAL_REQUEST = "manual_request"
    
    @property
    def priority(self) -> int:
        """Get the priority level for this flag type (1=highest, 5=lowest)."""
        priorities = {
            self.STANDARDS_VIOLATION: 1,
            self.PROCESSING_ERROR: 1,
            self.FEATURE_LOSS: 2,
            self.HIGH_UNCERTAINTY: 2,
            self.LOW_QUALITY: 3,
            self.POOR_SIMILARITY: 3,
            self.MANUAL_REQUEST: 4
        }
        return priorities[self]
    
    @property
    def description(self) -> str:
        """Get a description of this flag type."""
        descriptions = {
            self.LOW_QUALITY: "Overall quality score below threshold",
            self.FEATURE_LOSS: "Significant bathymetric features were lost",
            self.STANDARDS_VIOLATION: "Violates hydrographic standards",
            self.POOR_SIMILARITY: "Low structural similarity to original",
            self.HIGH_UNCERTAINTY: "High uncertainty in processed data",
            self.PROCESSING_ERROR: "Error occurred during processing",
            self.MANUAL_REQUEST: "Manually requested for review"
        }
        return descriptions[self]


class DataFormat(Enum):
    """Supported data formats."""
    BAG = ".bag"
    GEOTIFF = ".tif"
    GEOTIFF_ALT = ".tiff"
    ASCII_GRID = ".asc"
    XYZ = ".xyz"
    
    @property
    def gdal_driver(self) -> str:
        """Get the GDAL driver name for this format."""
        drivers = {
            self.BAG: "BAG",
            self.GEOTIFF: "GTiff",
            self.GEOTIFF_ALT: "GTiff",
            self.ASCII_GRID: "AAIGrid",
            self.XYZ: None  # Custom handling required
        }
        return drivers[self]
    
    @property
    def supports_uncertainty(self) -> bool:
        """Check if format supports uncertainty bands."""
        return self in [self.BAG]
    
    @classmethod
    def from_extension(cls, extension: str) -> 'DataFormat':
        """Get format from file extension."""
        ext_lower = extension.lower()
        for fmt in cls:
            if fmt.value == ext_lower:
                return fmt
        raise ValueError(f"Unsupported format: {extension}")


class ModelType(Enum):
    """Model architecture types."""
    BASIC_CAE = "basic_cae"
    ADVANCED_CAE = "advanced_cae"
    UNCERTAINTY_CAE = "uncertainty_cae"
    ENSEMBLE_CAE = "ensemble_cae"


class MetricType(Enum):
    """Quality metric types."""
    SSIM = "ssim"
    ROUGHNESS = "roughness"
    FEATURE_PRESERVATION = "feature_preservation"
    CONSISTENCY = "consistency"
    HYDROGRAPHIC_COMPLIANCE = "hydrographic_compliance"
    COMPOSITE_QUALITY = "composite_quality"
    
    @property
    def is_higher_better(self) -> bool:
        """Check if higher values indicate better quality for this metric."""
        higher_better = {
            self.SSIM: True,
            self.ROUGHNESS: False,  # Lower roughness is generally better
            self.FEATURE_PRESERVATION: True,
            self.CONSISTENCY: True,
            self.HYDROGRAPHIC_COMPLIANCE: True,
            self.COMPOSITE_QUALITY: True
        }
        return higher_better[self]


# Constants
DEFAULT_DEPTH_RANGES = {
    SeafloorType.SHALLOW_COASTAL: (0, 200),
    SeafloorType.CONTINENTAL_SHELF: (200, 2000),
    SeafloorType.DEEP_OCEAN: (2000, 6000),
    SeafloorType.ABYSSAL_PLAIN: (6000, 11000),
    SeafloorType.SEAMOUNT: (0, 6000)
}

SUPPORTED_EXTENSIONS = [fmt.value for fmt in DataFormat]

# IHO S-44 Standards Constants
IHO_STANDARDS = {
    "special_order": {"horizontal_accuracy": 2.0, "depth_accuracy": "0.25m + 0.0075*depth"},
    "order_1a": {"horizontal_accuracy": 5.0, "depth_accuracy": "0.5m + 0.013*depth"},
    "order_1b": {"horizontal_accuracy": 5.0, "depth_accuracy": "0.5m + 0.013*depth"},
    "order_2": {"horizontal_accuracy": 20.0, "depth_accuracy": "1.0m + 0.023*depth"}
}

# Processing Constants
MEMORY_CLEANUP_THRESHOLD = 1000  # MB
MIN_VALID_PIXELS = 100
MAX_OUTLIER_PERCENTAGE = 5.0
DEFAULT_NODATA_VALUE = -9999.0
