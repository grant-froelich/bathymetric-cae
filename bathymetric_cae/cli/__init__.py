# cli/__init__.py
"""Command line interface for bathymetric CAE package."""

from .main import main, create_argument_parser, validate_requirements_command

__all__ = ['main', 'create_argument_parser', 'validate_requirements_command']

# ================================================================================