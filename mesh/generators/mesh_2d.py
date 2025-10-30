"""
2D Mesh Generator
=================

Generates 2D meshes for plate, membrane, disc, and rectangle geometries using GMSH.
"""

import numpy as np
from typing import Dict, Any, List
from .base_generator import BaseMeshGenerator

class MeshGenerator2D(BaseMeshGenerator):
	"""Generator for 2D meshes (plates, membranes, discs) using GMSH exclusively"""

	def __init__(self):
		super().__init__(mesh_dimension=2)
		# Import GMSH generator
		from ..utils.gmsh_generator import GMSHGenerator
		self.gmsh_generator = GMSHGenerator()

	def generate_mesh(self, geometry_type: str, dimensions: Dict[str, float], mesh_parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate 2D mesh using GMSH exclusively"""
		if not self.validate_dimensions(geometry_type, dimensions):
			return self._create_error_mesh("Invalid dimensions")

		# Use GMSH exclusively
		if not self.gmsh_generator.gmsh_available:
			return self._create_error_mesh("GMSH not available - GMSH is required for mesh generation")

		return self.gmsh_generator.generate_mesh(geometry_type, dimensions, mesh_parameters)

	def get_required_dimensions(self, geometry_type: str) -> List[str]:
		"""Get required dimensions for 2D geometries"""
		if geometry_type in ['plate', 'membrane', 'rectangle']:
			return ['length', 'width']
		elif geometry_type == 'disc':
			return ['radius']
		else:
			return []

	def _create_error_mesh(self, error_message: str) -> Dict[str, Any]:
		"""Create error mesh"""
		self.logger.error(error_message)
		return {
			'vertices': [],
			'faces': [],
			'cells': {},
			'type': 'error',
			'bounds': {'min': [0, 0, 0], 'max': [0, 0, 0]},
			'mesh_stats': {'vertices': 0, 'faces': 0, 'cells': 0, 'cell_types': []},
			'success': False,
			'error': error_message
		}

	def validate_2d_geometry(self, geometry_type: str) -> bool:
		"""Validate that the geometry type is supported for 2D meshing"""
		supported_geometries = ['plate', 'membrane', 'disc', 'rectangle']
		if geometry_type not in supported_geometries:
			self.logger.error(f"Unsupported 2D geometry type: {geometry_type}. Supported types: {supported_geometries}")
			return False
		return True

	def get_supported_geometries(self) -> List[str]:
		"""Get list of supported 2D geometry types"""
		return ['plate', 'membrane', 'disc', 'rectangle']

# Manual mesh creation functions removed - use GMSH exclusively

	def get_mesh_info(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Get information about the generated 2D mesh"""
		if not mesh_data.get('success', False):
			return {'error': mesh_data.get('error', 'Unknown error')}

		stats = mesh_data.get('mesh_stats', {})
		bounds = mesh_data.get('bounds', {})
		dimensions = mesh_data.get('dimensions', {})
		
		return {
			'type': '2D Mesh',
			'geometry': mesh_data.get('geometry_type', 'unknown'),
			'vertices': stats.get('vertices', 0),
			'faces': stats.get('faces', 0),
			'cells': stats.get('cells', 0),
			'dimensions': dimensions,
			'bounds': bounds,
			'quality': stats.get('mesh_quality', {}),
			'success': True
		}

	def calculate_area(self, mesh_data: Dict[str, Any]) -> float:
		"""Calculate the area of the 2D mesh"""
		if not mesh_data.get('success', False):
			return 0.0

		geometry_type = mesh_data.get('geometry_type', '')
		dimensions = mesh_data.get('dimensions', {})
		
		if geometry_type in ['plate', 'membrane', 'rectangle']:
			length = dimensions.get('length', 0)
			width = dimensions.get('width', 0)
			return length * width
		elif geometry_type == 'disc':
			radius = dimensions.get('radius', 0)
			return np.pi * radius * radius
		
		return 0.0