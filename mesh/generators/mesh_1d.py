"""
1D Mesh Generator
=================

Generates 1D meshes for line geometry using GMSH exclusively.
Only supports line geometry as requested.
"""

import numpy as np
from typing import Dict, Any, List
from .base_generator import BaseMeshGenerator

class MeshGenerator1D(BaseMeshGenerator):
	"""Generator for 1D meshes (line only) using GMSH exclusively"""

	def __init__(self):
		super().__init__(mesh_dimension=1)
		# Import GMSH generator
		from ..utils.gmsh_generator import GMSHGenerator
		self.gmsh_generator = GMSHGenerator()

	def generate_mesh(self, geometry_type: str, dimensions: Dict[str, float], mesh_parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate 1D mesh using GMSH exclusively"""
		if not self.validate_dimensions(geometry_type, dimensions):
			return self._create_error_mesh("Invalid dimensions")

		if geometry_type not in ['line', 'rod', 'bar']:
			return self._create_error_mesh(f"Only 'line', 'rod', 'bar' geometry supported for 1D. Got: {geometry_type}")

		# Use GMSH exclusively
		if not self.gmsh_generator.gmsh_available:
			return self._create_error_mesh("GMSH not available - GMSH is required for mesh generation")

		return self.gmsh_generator.generate_mesh(geometry_type, dimensions, mesh_parameters)

	def get_required_dimensions(self, geometry_type: str) -> List[str]:
		"""Get required dimensions for 1D geometries"""
		if geometry_type in ['line', 'rod', 'bar']:
			return ['length']
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

	def validate_1d_geometry(self, geometry_type: str) -> bool:
		"""Validate that the geometry type is supported for 1D meshing"""
		supported_geometries = ['line', 'rod', 'bar']
		if geometry_type not in supported_geometries:
			self.logger.error(f"Unsupported 1D geometry type: {geometry_type}. Supported types: {supported_geometries}")
			return False
		return True

	def get_supported_geometries(self) -> List[str]:
		"""Get list of supported 1D geometry types"""
		return ['line', 'rod', 'bar']

	def create_simple_1d_mesh(self, length: float, num_elements: int = 10) -> Dict[str, Any]:
		"""Create a simple 1D mesh without GMSH (fallback)"""
		try:
			# Create vertices along the line
			vertices = []
			for i in range(num_elements + 1):
				x = (i / num_elements) * length
				vertices.append([x, 0.0, 0.0])

			# Create faces (line segments)
			faces = []
			for i in range(num_elements):
				faces.append([i, i + 1])

			# Create cells (1D elements)
			cells = {
				'line': [[i, i + 1] for i in range(num_elements)]
			}

			# Calculate statistics
			stats = self.calculate_mesh_statistics(vertices, faces, cells)

			# Create bounds
			bounds = self.get_mesh_bounds(vertices)

			return {
				'vertices': vertices,
				'faces': faces,
				'cells': cells,
				'type': '1d_mesh',
				'bounds': bounds,
				'mesh_stats': stats,
				'success': True,
				'geometry_type': 'line',
				'dimensions': {'length': length},
				'num_elements': num_elements
			}

		except Exception as e:
			self.logger.error(f"Error creating simple 1D mesh: {e}")
			return self._create_error_mesh(f"Failed to create simple 1D mesh: {e}")

	def get_mesh_info(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Get information about the generated 1D mesh"""
		if not mesh_data.get('success', False):
			return {'error': mesh_data.get('error', 'Unknown error')}

		stats = mesh_data.get('mesh_stats', {})
		bounds = mesh_data.get('bounds', {})
		
		return {
			'type': '1D Mesh',
			'geometry': mesh_data.get('geometry_type', 'unknown'),
			'vertices': stats.get('vertices', 0),
			'faces': stats.get('faces', 0),
			'cells': stats.get('cells', 0),
			'length': mesh_data.get('dimensions', {}).get('length', 0),
			'bounds': bounds,
			'quality': stats.get('mesh_quality', {}),
			'success': True
		}