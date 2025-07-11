# models/__init__.py
from models.ensemble import BathymetricEnsemble
from models.architectures import AdvancedCAE, UncertaintyCAE, LightweightCAE, create_model_variant, get_model_variants_for_ensemble