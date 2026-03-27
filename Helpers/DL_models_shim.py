"""
Backward-compatible shim for Helpers.DL_models.
Imports everything from the new modular `models` package.

Usage (old code still works):
    from Helpers.DL_models_shim import setup_model, MBA_tsm_with_padding

For new code, prefer:
    from models import setup_model, MBA_tsm_with_padding
"""
import sys
import os

# Add project root to path so 'models' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import *
from models.setup import setup_model
