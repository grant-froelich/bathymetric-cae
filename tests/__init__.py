# tests/__init__.py

"""
Enhanced Bathymetric CAE Processing Test Suite v2.0

Comprehensive test suite including unit tests, integration tests,
performance tests, and quality assurance tools.
"""

# Test configuration
TEST_SUITE_VERSION = "2.0.0"
MINIMUM_COVERAGE = 85.0

# Test markers
UNIT_MARKER = "unit"
INTEGRATION_MARKER = "integration"
PERFORMANCE_MARKER = "performance"
SLOW_MARKER = "slow"
GPU_MARKER = "gpu"

# Import key test utilities for easy access
try:
    from .fixtures import (
        ModelTestFixture,
        DataProcessingFixture,
        PipelineTestFixture,
        PerformanceTestFixture
    )
except ImportError:
    # Graceful fallback if fixtures not available
    ModelTestFixture = None
    DataProcessingFixture = None
    PipelineTestFixture = None
    PerformanceTestFixture = None

try:
    from .factories import (
        BathymetricDataFactory,
        ConfigurationFactory,
        TestFileFactory
    )
except ImportError:
    # Graceful fallback if factories not available
    BathymetricDataFactory = None
    ConfigurationFactory = None
    TestFileFactory = None

try:
    from .utils import (
        TestHelpers,
        PerformanceMonitor,
        MockGDALDataset
    )
except ImportError:
    # Graceful fallback if utils not available
    TestHelpers = None
    PerformanceMonitor = None
    MockGDALDataset = None

__all__ = [
    # Test fixtures
    'ModelTestFixture',
    'DataProcessingFixture',
    'PipelineTestFixture', 
    'PerformanceTestFixture',
    
    # Data factories
    'BathymetricDataFactory',
    'ConfigurationFactory',
    'TestFileFactory',
    
    # Utilities
    'TestHelpers',
    'PerformanceMonitor',
    'MockGDALDataset',
    
    # Constants
    'TEST_SUITE_VERSION',
    'MINIMUM_COVERAGE',
    'UNIT_MARKER',
    'INTEGRATION_MARKER',
    'PERFORMANCE_MARKER',
    'SLOW_MARKER',
    'GPU_MARKER'
]

# Test suite information
def get_test_suite_info():
    """Get information about the test suite."""
    return {
        'version': TEST_SUITE_VERSION,
        'minimum_coverage': MINIMUM_COVERAGE,
        'available_markers': [
            UNIT_MARKER,
            INTEGRATION_MARKER, 
            PERFORMANCE_MARKER,
            SLOW_MARKER,
            GPU_MARKER
        ],
        'available_fixtures': [
            name for name in [
                'ModelTestFixture',
                'DataProcessingFixture',
                'PipelineTestFixture',
                'PerformanceTestFixture'
            ] if globals()[name] is not None
        ],
        'available_factories': [
            name for name in [
                'BathymetricDataFactory',
                'ConfigurationFactory', 
                'TestFileFactory'
            ] if globals()[name] is not None
        ],
        'available_utilities': [
            name for name in [
                'TestHelpers',
                'PerformanceMonitor',
                'MockGDALDataset'
            ] if globals()[name] is not None
        ]
    }