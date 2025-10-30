"""
FEniCS Backend Module
Consolidated FEniCS solver and visualization components
"""

import logging

# Import the main components
try:
	from .local_fenics_solver import fenics_solver
	from .local_field_visualizer import field_visualizer
	FENICS_AVAILABLE = True
except ImportError as e:
	logging.warning(f"FEniCS backend components not available: {e}")
	fenics_solver = None
	field_visualizer = None
	FENICS_AVAILABLE = False

	__all__ = ['fenics_solver', 'field_visualizer', 'FENICS_AVAILABLE']
