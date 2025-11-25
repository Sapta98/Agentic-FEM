"""
FEniCS Backend Module
Consolidated FEniCS solver and visualization components
"""

import logging

# Import the main components
try:
	from .solver_core import FEniCSSolver
	# Create a global instance for backward compatibility
	fenics_solver = FEniCSSolver()
	from .local_field_visualizer import FieldVisualizer, field_visualizer
	FENICS_AVAILABLE = True
except ImportError as e:
	logging.warning(f"FEniCS backend components not available: {e}")
	FEniCSSolver = None
	fenics_solver = None
	FieldVisualizer = None
	field_visualizer = None
	FENICS_AVAILABLE = False

__all__ = ['fenics_solver', 'field_visualizer', 'FENICS_AVAILABLE', 'FEniCSSolver', 'FieldVisualizer']
