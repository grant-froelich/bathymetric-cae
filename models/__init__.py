# models/__init__.py
"""Machine learning models for Enhanced Bathymetric CAE Processing."""

from .ensemble import (
    BathymetricEnsemble,
    ModelVariant,
    create_bathymetric_ensemble
)

from .base_models import (
    BaseCAE,
    LightweightCAE,
    StandardCAE,
    RobustCAE,
    DeepCAE,
    WideCAE,
    UncertaintyCAE,
    create_model_by_type,
    get_model_variants,
    compile_model
)

__all__ = [
    # Ensemble
    'BathymetricEnsemble',
    'ModelVariant',
    'create_bathymetric_ensemble',
    
    # Base Models
    'BaseCAE',
    'LightweightCAE',
    'StandardCAE', 
    'RobustCAE',
    'DeepCAE',
    'WideCAE',
    'UncertaintyCAE',
    'create_model_by_type',
    'get_model_variants',
    'compile_model'
]
