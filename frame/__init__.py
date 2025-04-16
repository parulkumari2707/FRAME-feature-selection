"""
FRAME Feature Selector Library Initialization
---------------------------------------------

This package provides the FRAMESelector class for feature selection using a hybrid approach
of Recursive Feature Elimination (RFE) and Forward Feature Selection with XGBoost.

Usage:
    from frame_selector import FRAMESelector
"""

from .frame_selector import FRAMESelector
from .version import __version__

__all__ = ["FRAMESelector", "__version__"]
