"""
Mesh Configuration
==================

Configuration for mesh generation based on geometry only.
Uses professional mesh generators like GMSH for high-quality mesh creation.
"""

from typing import Dict, List, Tuple, Any
import numpy as np

class MeshConfig:
	"""Configuration for mesh generation system - geometry only"""

	def __init__(self):
		self.geometry_mesh_mapping = self._get_geometry_mesh_mapping()
		self.mesh_quality_settings = self._get_mesh_quality_settings()
		self.mesh_generator_settings = self._get_mesh_generator_settings()

	def _get_geometry_mesh_mapping(self) -> Dict[str, Dict[str, Any]]:
		"""Map geometry types to their natural mesh dimensions"""
		return {
			# 1D Geometries
			'line': {
				'natural_dim': 1,
				'description': '1D line segment',
				'gmsh_type': 'line',
				'required_dimensions': ['length']
			},
			'rod': {
				'natural_dim': 1,
				'description': '1D rod/bar',
				'gmsh_type': 'line',
				'required_dimensions': ['length']
			},
			'bar': {
				'natural_dim': 1,
				'description': '1D bar',
				'gmsh_type': 'line',
				'required_dimensions': ['length']
			},

			# 2D Geometries
			'plate': {
				'natural_dim': 2,
				'description': '2D plate/shell',
				'gmsh_type': 'surface',
				'required_dimensions': ['length', 'width']
			},
			'membrane': {
				'natural_dim': 2,
				'description': '2D membrane',
				'gmsh_type': 'surface',
				'required_dimensions': ['length', 'width']
			},
			'disc': {
				'natural_dim': 2,
				'description': '2D circular disc',
				'gmsh_type': 'surface',
				'required_dimensions': ['radius']
			},
			'rectangle': {
				'natural_dim': 2,
				'description': '2D rectangle',
				'gmsh_type': 'surface',
				'required_dimensions': ['length', 'width']
			},
			'square': {
				'natural_dim': 2,
				'description': '2D square',
				'gmsh_type': 'surface',
				'required_dimensions': ['length']
			},

			# 3D Geometries
			'cube': {
				'natural_dim': 3,
				'description': '3D cube',
				'gmsh_type': 'volume',
				'required_dimensions': ['length']
			},
			'box': {
				'natural_dim': 3,
				'description': '3D rectangular box',
				'gmsh_type': 'volume',
				'required_dimensions': ['length', 'width', 'height']
			},
			'beam': {
				'natural_dim': 3,
				'description': '3D beam',
				'gmsh_type': 'volume',
				'required_dimensions': ['length', 'width', 'height']
			},
			'cylinder': {
				'natural_dim': 3,
				'description': '3D cylinder',
				'gmsh_type': 'volume',
				'required_dimensions': ['radius', 'height'],
				'alternative_dimensions': {'height': ['length']}  # Accept 'length' as alias for 'height'
			},
			'sphere': {
				'natural_dim': 3,
				'description': '3D sphere',
				'gmsh_type': 'volume',
				'required_dimensions': ['radius']
			},
			'solid': {
				'natural_dim': 3,
				'description': '3D solid body',
				'gmsh_type': 'volume',
				'required_dimensions': ['length', 'width', 'height']
			},
			'rectangular': {
				'natural_dim': 3,
				'description': '3D rectangular solid',
				'gmsh_type': 'volume',
				'required_dimensions': ['length', 'width', 'height']
			},
		}

	def _get_mesh_generator_settings(self) -> Dict[str, Any]:
		"""Settings for professional mesh generators"""
		return {
			'gmsh': {
				'available': True,
				'default_algorithm': 'Delaunay',  # For 2D: Delaunay, Frontal-Delaunay
				'default_algorithm_3d': 'Delaunay',  # For 3D: Delaunay, Frontal-Delaunay
				'element_order': 1,  # Linear elements
				'optimize': True,
				'smoothing': 5,  # Number of smoothing iterations
				'quality_measure': 'gamma',  # gamma, eta, alpha
				'min_quality': 0.1
			},
			'meshio': {
				'available': True,
				'supported_formats': ['vtk', 'xdmf', 'msh', 'stl'],
				'default_format': 'vtk'
			}
		}

	def _get_mesh_quality_settings(self) -> Dict[str, Any]:
		"""Mesh quality and resolution settings"""
		return {
			'coarse': {
				'resolution': 20,
				'mesh_size': 0.5,
				'element_quality': 0.3,
				'gmsh_mesh_size': 0.5
			},
			'medium': {
				'resolution': 50,
				'mesh_size': 0.2,
				'element_quality': 0.2,
				'gmsh_mesh_size': 0.2
			},
			'fine': {
				'resolution': 100,
				'mesh_size': 0.1,
				'element_quality': 0.1,
				'gmsh_mesh_size': 0.1
			},
			'very_fine': {
				'resolution': 200,
				'mesh_size': 0.05,
				'element_quality': 0.05,
				'gmsh_mesh_size': 0.05
			},

			# Default settings
			'default_resolution': 50,
			'default_quality': 0.2,
			'default_mesh_size': 0.2,

			# Dimension-specific settings
			'1d': {'min_elements': 10, 'max_elements': 1000, 'default_mesh_size': 0.1},
			'2d': {'min_elements': 50, 'max_elements': 10000, 'default_mesh_size': 0.2},
			'3d': {'min_elements': 200, 'max_elements': 50000, 'default_mesh_size': 0.3}
		}

	def get_mesh_dimension(self, geometry_type: str) -> int:
		"""Get the mesh dimension for a geometry type"""
		geo_config = self.geometry_mesh_mapping.get(geometry_type, {})
		return geo_config.get('natural_dim', 3)

	def get_mesh_quality(self, quality_level: str = 'medium') -> Dict[str, Any]:
		"""Get mesh quality settings"""
		return self.mesh_quality_settings.get(quality_level, self.mesh_quality_settings['medium'])

	def get_geometry_config(self, geometry_type: str) -> Dict[str, Any]:
		"""Get configuration for a geometry type"""
		return self.geometry_mesh_mapping.get(geometry_type, {})

	def get_gmsh_settings(self) -> Dict[str, Any]:
		"""Get GMSH generator settings"""
		return self.mesh_generator_settings.get('gmsh', {})

	def get_required_dimensions(self, geometry_type: str) -> List[str]:
		"""Get required dimensions for a geometry type"""
		geo_config = self.get_geometry_config(geometry_type)
		return geo_config.get('required_dimensions', [])

	def is_gmsh_available(self) -> bool:
		"""Check if GMSH is available"""
		return self.mesh_generator_settings.get('gmsh', {}).get('available', False)

	def get_supported_geometries(self, dimension: int = None) -> List[str]:
		"""Get list of supported geometries, optionally filtered by dimension"""
		geometries = []
		for geo_type, config in self.geometry_mesh_mapping.items():
			if dimension is None or config.get('natural_dim') == dimension:
				geometries.append(geo_type)
		return geometries

	def get_geometry_description(self, geometry_type: str) -> str:
		"""Get description for a geometry type"""
		geo_config = self.get_geometry_config(geometry_type)
		return geo_config.get('description', 'Unknown geometry')

	def get_gmsh_type(self, geometry_type: str) -> str:
		"""Get GMSH type for a geometry"""
		geo_config = self.get_geometry_config(geometry_type)
		return geo_config.get('gmsh_type', 'volume')

	def validate_geometry(self, geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
		"""Validate geometry configuration and dimensions"""
		result = {
			'valid': True,
			'errors': [],
			'warnings': []
		}

		# Check if geometry type is supported
		if geometry_type not in self.geometry_mesh_mapping:
			result['valid'] = False
			result['errors'].append(f"Unsupported geometry type: {geometry_type}")
			return result

		# Check required dimensions (convert strings to numbers first)
		required_dims = self.get_required_dimensions(geometry_type)
		missing_dims = []
		for dim in required_dims:
			if dim not in dimensions:
				missing_dims.append(dim)
			else:
				value = dimensions[dim]
				# Convert string to number if needed
				try:
					if isinstance(value, str):
						value = float(value.strip()) if value.strip() else None
					elif not isinstance(value, (int, float)):
						value = float(value) if value is not None else None
					
					if value is None:
						missing_dims.append(dim)
					elif value <= 0:
						missing_dims.append(dim)
					# Update the dimensions dict with converted value
					dimensions[dim] = value
				except (ValueError, TypeError):
					# If conversion fails, treat as missing
					missing_dims.append(dim)

		if missing_dims:
			result['valid'] = False
			result['errors'].append(f"Missing or invalid dimensions for {geometry_type}: {missing_dims}")

		# Check dimension values (ensure they're numbers, not strings)
		# Convert all dimensions to numbers and update the dict in place
		for dim_name, value in list(dimensions.items()):
			if value is None:
				result['warnings'].append(f"Dimension {dim_name} is None")
			else:
				# Convert string values to float if needed
				try:
					original_value = value
					if isinstance(value, str):
						value = float(value.strip()) if value.strip() else None
					elif not isinstance(value, (int, float)):
						value = float(value)
					
					# Update the dimensions dict with converted value
					if value is not None:
						dimensions[dim_name] = value
					
					if value is None:
						result['warnings'].append(f"Dimension {dim_name} is None or empty")
					elif value <= 0:
						result['warnings'].append(f"Dimension {dim_name} should be positive, got {value}")
				except (ValueError, TypeError) as e:
					result['warnings'].append(f"Dimension {dim_name} has invalid value '{original_value}' (type: {type(original_value).__name__}): {e}")

		return result

	def get_mesh_parameters(self, geometry_type: str, quality_level: str = 'medium') -> Dict[str, Any]:
		"""Get mesh parameters for a geometry type and quality level"""
		quality_settings = self.get_mesh_quality(quality_level)
		gmsh_settings = self.get_gmsh_settings()
		
		mesh_params = {
			'resolution': quality_settings.get('resolution', 50),
			'mesh_size': quality_settings.get('gmsh_mesh_size', 0.2),
			'element_quality': quality_settings.get('element_quality', 0.2),
			'algorithm': gmsh_settings.get('default_algorithm', 'Delaunay'),
			'element_order': gmsh_settings.get('element_order', 1),
			'optimize': gmsh_settings.get('optimize', True),
			'smoothing': gmsh_settings.get('smoothing', 5),
			'quality_measure': gmsh_settings.get('quality_measure', 'gamma'),
			'min_quality': gmsh_settings.get('min_quality', 0.1)
		}

		# Add dimension-specific settings
		mesh_dim = self.get_mesh_dimension(geometry_type)
		dim_key = f'{mesh_dim}d'
		dim_settings = self.mesh_quality_settings.get(dim_key, {})
		
		if 'default_mesh_size' in dim_settings:
			mesh_params['mesh_size'] = dim_settings['default_mesh_size']

		return mesh_params

	def get_available_quality_levels(self) -> List[str]:
		"""Get list of available mesh quality levels"""
		return ['coarse', 'medium', 'fine', 'very_fine']

	def get_config_summary(self) -> Dict[str, Any]:
		"""Get a summary of the mesh configuration"""
		return {
			'total_geometries': len(self.geometry_mesh_mapping),
			'1d_geometries': len(self.get_supported_geometries(1)),
			'2d_geometries': len(self.get_supported_geometries(2)),
			'3d_geometries': len(self.get_supported_geometries(3)),
			'quality_levels': self.get_available_quality_levels(),
			'gmsh_available': self.is_gmsh_available(),
			'default_quality': 'medium'
		}

# Global configuration instance
mesh_config = MeshConfig()