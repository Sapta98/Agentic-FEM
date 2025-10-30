"""
Frontend Components
==================

Reusable UI components for the FEM simulation frontend.
"""

from .terminal_interface import TerminalInterface
from .config_panels import ConfigPanels
from .input_fields import InputFields
from .result_display import ResultDisplay

__all__ = [
'TerminalInterface',
'ConfigPanels',
'InputFields',
'ResultDisplay'
]
