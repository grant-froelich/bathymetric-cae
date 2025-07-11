# tests/test_core_enums.py
"""
Test core enumerations.
"""

import pytest
from core.enums import SeafloorType


class TestSeafloorType:
    """Test SeafloorType enumeration."""
    
    def test_seafloor_type_values(self):
        """Test SeafloorType enum values."""
        assert SeafloorType.SHALLOW_COASTAL.value == "shallow_coastal"
        assert SeafloorType.DEEP_OCEAN.value == "deep_ocean"
        assert SeafloorType.CONTINENTAL_SHELF.value == "continental_shelf"
        assert SeafloorType.SEAMOUNT.value == "seamount"
        assert SeafloorType.ABYSSAL_PLAIN.value == "abyssal_plain"
        assert SeafloorType.UNKNOWN.value == "unknown"
    
    def test_seafloor_type_completeness(self):
        """Test that all expected seafloor types are defined."""
        expected_types = {
            "shallow_coastal", "deep_ocean", "continental_shelf",
            "seamount", "abyssal_plain", "unknown"
        }
        actual_types = {st.value for st in SeafloorType}
        assert actual_types == expected_types
