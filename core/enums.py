"""
Enumerations for Enhanced Bathymetric CAE Processing.
"""

from enum import Enum


class SeafloorType(Enum):
    """Seafloor environment types for adaptive processing."""
    SHALLOW_COASTAL = "shallow_coastal"
    DEEP_OCEAN = "deep_ocean"
    CONTINENTAL_SHELF = "continental_shelf"
    SEAMOUNT = "seamount"
    ABYSSAL_PLAIN = "abyssal_plain"
    UNKNOWN = "unknown"
