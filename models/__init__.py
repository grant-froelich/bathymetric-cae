# models/__init__.py
from .ensemble import BathymetricEnsemble
from .architectures import AdvancedCAE, UncertaintyCAE, LightweightCAE, create_model_variant, get_model_variants_for_ensemble