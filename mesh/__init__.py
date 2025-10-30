"""
Mesh Generation System
======================

A comprehensive mesh generation system that creates appropriate mesh types
(1D, 2D, 3D) based on geometry and physics simulation requirements.

Usage:
from mesh import MeshGenerator

mesh_gen = MeshGenerator()
mesh_data = mesh_gen.generate_mesh(geometry_type, dimensions, physics_type)
"""

from .mesh_generator import MeshGenerator
from .utils.mesh_detector import detect_mesh_dimensions
from .config.mesh_config import MeshConfig

__all__ = ['MeshGenerator', 'detect_mesh_dimensions', 'MeshConfig']
