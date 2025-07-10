"""
Main module entry point for bathymetric_cae package.

This allows the package to be executed as a module:
python -m bathymetric_cae [arguments]

Author: Bathymetric CAE Team
License: MIT
"""

import sys
from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())
