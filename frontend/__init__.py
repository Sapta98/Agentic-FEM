"""
Frontend Module
===============

Comprehensive frontend system for the FEM simulation application.
Includes terminal interface, mesh visualization, configuration panels,
and interactive components.

Components:
- Terminal Interface: Main web-based terminal
- Mesh Visualizer: 3D mesh and field visualization
- Configuration Panels: Interactive parameter editing
- UI Components: Reusable interface elements
- Static Assets: CSS, JavaScript, and resources
"""

from .frontend_manager import FrontendManager
from .components.terminal_interface import TerminalInterface
from .visualizers.mesh_visualizer import MeshVisualizer
from .components.config_panels import ConfigPanels

__all__ = [
'FrontendManager',
'TerminalInterface',
'MeshVisualizer',
'ConfigPanels'
]

__version__ = "1.0.0"
