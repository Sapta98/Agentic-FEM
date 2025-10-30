"""
3D Mesh Generator
=================

Generates 3D meshes for cube, box, beam, cylinder, sphere, and solid geometries using GMSH.
"""

import numpy as np
from typing import Dict, Any, List
from .base_generator import BaseMeshGenerator

class MeshGenerator3D(BaseMeshGenerator):
	"""Generator for 3D meshes (cubes, beams, cylinders, spheres) using GMSH exclusively"""

	def __init__(self):
		super().__init__(mesh_dimension=3)
		# Import GMSH generator
		from ..utils.gmsh_generator import GMSHGenerator
		self.gmsh_generator = GMSHGenerator()

	def generate_mesh(self, geometry_type: str, dimensions: Dict[str, float], mesh_parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate 3D mesh using GMSH exclusively"""
		if not self.validate_dimensions(geometry_type, dimensions):
			return self._create_error_mesh("Invalid dimensions")

		# Use GMSH exclusively
		if not self.gmsh_generator.gmsh_available:
			return self._create_error_mesh("GMSH not available - GMSH is required for mesh generation")

		return self.gmsh_generator.generate_mesh(geometry_type, dimensions, mesh_parameters)

	def get_required_dimensions(self, geometry_type: str) -> List[str]:
		"""Get required dimensions for 3D geometries"""
		if geometry_type == 'cube':
			return ['length']  # Cube only needs one dimension (all sides equal)
		elif geometry_type in ['box', 'solid', 'rectangular']:
			return ['length', 'width', 'height']  # Box needs all three dimensions
		elif geometry_type == 'beam':
			return ['length', 'width', 'height']
		elif geometry_type == 'cylinder':
			return ['radius', 'height']  # height is preferred, but length is accepted as alias
		elif geometry_type == 'sphere':
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

	def validate_3d_geometry(self, geometry_type: str) -> bool:
		"""Validate that the geometry type is supported for 3D meshing"""
		supported_geometries = ['cube', 'box', 'beam', 'cylinder', 'sphere', 'solid', 'rectangular']
		if geometry_type not in supported_geometries:
			self.logger.error(f"Unsupported 3D geometry type: {geometry_type}. Supported types: {supported_geometries}")
			return False
		return True

	def get_supported_geometries(self) -> List[str]:
		"""Get list of supported 3D geometry types"""
		return ['cube', 'box', 'beam', 'cylinder', 'sphere', 'solid', 'rectangular']

	def get_mesh_info(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Get information about the generated 3D mesh"""
		if not mesh_data.get('success', False):
			return {'error': mesh_data.get('error', 'Unknown error')}

		stats = mesh_data.get('mesh_stats', {})
		bounds = mesh_data.get('bounds', {})
		dimensions = mesh_data.get('dimensions', {})
		
		return {
			'type': '3D Mesh',
			'geometry': mesh_data.get('geometry_type', 'unknown'),
			'vertices': stats.get('vertices', 0),
			'faces': stats.get('faces', 0),
			'cells': stats.get('cells', 0),
			'dimensions': dimensions,
			'bounds': bounds,
			'quality': stats.get('mesh_quality', {}),
			'success': True
		}

	def calculate_volume(self, mesh_data: Dict[str, Any]) -> float:
		"""Calculate the volume of the 3D mesh"""
		if not mesh_data.get('success', False):
			return 0.0

		geometry_type = mesh_data.get('geometry_type', '')
		dimensions = mesh_data.get('dimensions', {})
		
		if geometry_type == 'cube':
			length = dimensions.get('length', 0)
			return length * length * length
		elif geometry_type in ['box', 'solid', 'rectangular']:
			length = dimensions.get('length', 0)
			width = dimensions.get('width', 0)
			height = dimensions.get('height', 0)
			return length * width * height
		elif geometry_type == 'beam':
			length = dimensions.get('length', 0)
			width = dimensions.get('width', 0)
			height = dimensions.get('height', 0)
			return length * width * height
		elif geometry_type == 'cylinder':
			radius = dimensions.get('radius', 0)
			height = dimensions.get('height', 0)
			return np.pi * radius * radius * height
		elif geometry_type == 'sphere':
			radius = dimensions.get('radius', 0)
			return (4.0 / 3.0) * np.pi * radius * radius * radius
		
		return 0.0

	def calculate_surface_area(self, mesh_data: Dict[str, Any]) -> float:
		"""Calculate the surface area of the 3D mesh"""
		if not mesh_data.get('success', False):
			return 0.0

		geometry_type = mesh_data.get('geometry_type', '')
		dimensions = mesh_data.get('dimensions', {})
		
		if geometry_type == 'cube':
			length = dimensions.get('length', 0)
			return 6 * length * length
		elif geometry_type in ['box', 'solid', 'rectangular']:
			length = dimensions.get('length', 0)
			width = dimensions.get('width', 0)
			height = dimensions.get('height', 0)
			return 2 * (length * width + length * height + width * height)
		elif geometry_type == 'beam':
			length = dimensions.get('length', 0)
			width = dimensions.get('width', 0)
			height = dimensions.get('height', 0)
			return 2 * (length * width + length * height + width * height)
		elif geometry_type == 'cylinder':
			radius = dimensions.get('radius', 0)
			height = dimensions.get('height', 0)
			return 2 * np.pi * radius * (radius + height)
		elif geometry_type == 'sphere':
			radius = dimensions.get('radius', 0)
			return 4 * np.pi * radius * radius
		
		return 0.0

	def get_geometry_center(self, mesh_data: Dict[str, Any]) -> List[float]:
		"""Get the center point of the 3D geometry"""
		if not mesh_data.get('success', False):
			return [0.0, 0.0, 0.0]

		geometry_type = mesh_data.get('geometry_type', '')
		dimensions = mesh_data.get('dimensions', {})
		
		if geometry_type == 'cube':
			length = dimensions.get('length', 0)
			return [length / 2, length / 2, length / 2]
		elif geometry_type in ['box', 'solid', 'rectangular', 'beam']:
			length = dimensions.get('length', 0)
			width = dimensions.get('width', 0)
			height = dimensions.get('height', 0)
			return [length / 2, width / 2, height / 2]
		elif geometry_type == 'cylinder':
			radius = dimensions.get('radius', 0)
			height = dimensions.get('height', 0)
			return [0.0, 0.0, height / 2]  # Centered at origin in x,y, centered in z
		elif geometry_type == 'sphere':
			# Sphere is centered at origin
			return [0.0, 0.0, 0.0]
		
		return [0.0, 0.0, 0.0]

	def validate_mesh_quality(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate the quality of the generated 3D mesh"""
		if not mesh_data.get('success', False):
			return {'valid': False, 'error': 'Mesh generation failed'}

		quality_info = {
			'valid': True,
			'warnings': [],
			'errors': []
		}

		stats = mesh_data.get('mesh_stats', {})
		vertices = stats.get('vertices', 0)
		cells = stats.get('cells', 0)

		# Check for minimum mesh requirements
		if vertices < 4:  # Minimum for a tetrahedron
			quality_info['errors'].append('Insufficient vertices for 3D mesh')
			quality_info['valid'] = False

		if cells < 1:
			quality_info['errors'].append('No cells generated')
			quality_info['valid'] = False

		# Check mesh quality metrics if available
		mesh_quality = stats.get('mesh_quality', {})
		if mesh_quality:
			aspect_ratio = mesh_quality.get('aspect_ratio', 1.0)
			if aspect_ratio > 10.0:
				quality_info['warnings'].append(f'High aspect ratio detected: {aspect_ratio:.2f}')

		return quality_info