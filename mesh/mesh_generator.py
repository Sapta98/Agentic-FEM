"""
Main Mesh Generator
===================

Main mesh generator that coordinates mesh generation using GMSH directly.
Determines the appropriate mesh type based on geometry.
"""

import logging
from typing import Dict, Any, Optional
from .utils.mesh_detector import detect_mesh_dimensions, validate_geometry_dimensions
from .config.mesh_config import mesh_config
from .utils.gmsh_generator import GMSHGenerator

logger = logging.getLogger(__name__)

class MeshGenerator:
	"""Main mesh generator that creates appropriate meshes based on geometry"""

	def __init__(self):
		self.logger = logger
		self.gmsh_generator = GMSHGenerator()

	def generate_mesh(self, geometry_type: str, dimensions: Dict[str, float],
			mesh_quality: str = 'medium',
			custom_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""
		Generate mesh based on geometry requirements only

		Args:
			geometry_type: Type of geometry (cube, beam, plate, etc.)
			dimensions: Dictionary with geometry dimensions
			mesh_quality: Quality level (coarse, medium, fine, very_fine)
			custom_parameters: Custom mesh parameters

		Returns:
			Dictionary with mesh data and metadata
		"""
		try:
			self.logger.info(f"Generating mesh for {geometry_type}")
			self.logger.info("Loading visualization... Please wait while mesh is generated...")

			# Validate geometry first
			validation = validate_geometry_dimensions(geometry_type, dimensions)
			if not validation['valid']:
				return self._create_error_mesh(f"Geometry validation failed: {validation['errors']}")

			# Detect mesh requirements (geometry only)
			mesh_config_info = detect_mesh_dimensions(geometry_type, dimensions)
			mesh_dim = mesh_config_info['mesh_dimension']

			# Get mesh quality settings
			quality_settings = mesh_config.get_mesh_quality(mesh_quality)

			# Prepare mesh parameters
			mesh_parameters = {
				'resolution': quality_settings['resolution'],
				'element_quality': quality_settings['element_quality'],
				'mesh_type': mesh_config_info['mesh_type'],
				'special_requirements': mesh_config_info['mesh_parameters']['special_requirements'],
				'gmsh_type': mesh_config_info['mesh_parameters'].get('gmsh_type', 'volume'),
				'mesh_size': mesh_config_info['mesh_parameters'].get('mesh_size', quality_settings['gmsh_mesh_size'])
			}

			# Add custom parameters if provided
			if custom_parameters:
				mesh_parameters.update(custom_parameters)

			# Validate mesh dimension
			if mesh_dim not in [1, 2, 3]:
				return self._create_error_mesh(f"Unsupported mesh dimension: {mesh_dim}")

			# Check GMSH availability
			if not self.gmsh_generator.gmsh_available:
				return self._create_error_mesh("GMSH not available - GMSH is required for mesh generation")

			# Generate mesh directly using GMSHGenerator
			mesh_data = self.gmsh_generator.generate_mesh(geometry_type, dimensions, mesh_parameters)

			# Add metadata
			mesh_data.update({
				'mesh_dimension': mesh_dim,
				'geometry_type': geometry_type,
				'mesh_quality': mesh_quality,
				'mesh_config': mesh_config_info,
				'generator_used': 'GMSHGenerator'
			})

			nodes_count = mesh_data.get('mesh_stats', {}).get('num_vertices', len(mesh_data.get('vertices', [])))
			self.logger.info(f"Successfully generated {mesh_dim}D mesh with {nodes_count} vertices")

			return mesh_data

		except Exception as e:
			self.logger.error(f"Error generating mesh: {e}")
			return self._create_error_mesh(str(e))

	def get_supported_geometries(self) -> Dict[int, list]:
		"""Get list of supported geometries by dimension"""
		supported = {}
		
		# Get geometries from mesh_config grouped by dimension
		for dim in [1, 2, 3]:
			geometries = mesh_config.get_supported_geometries(dimension=dim)
			supported[dim] = geometries

		return supported

	def validate_geometry(self, geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
		"""Validate geometry parameters"""
		# Use the mesh detector validation
		validation = validate_geometry_dimensions(geometry_type, dimensions)
		
		if not validation['valid']:
			return validation

		# Get mesh dimension
		mesh_dim = mesh_config.get_mesh_dimension(geometry_type)

		if mesh_dim not in [1, 2, 3]:
			return {
				'valid': False,
				'error': f'Unsupported geometry type: {geometry_type}'
			}

		# Get required dimensions from mesh_config
		required_dims = mesh_config.get_required_dimensions(geometry_type)

		return {
			'valid': True,
			'mesh_dimension': mesh_dim,
			'required_dimensions': required_dims,
			'warnings': validation.get('warnings', []),
			'suggestions': validation.get('suggestions', [])
		}

	def get_mesh_info(self, geometry_type: str) -> Dict[str, Any]:
		"""Get information about mesh generation for given geometry"""
		try:
			# Get mesh dimension
			mesh_dim = mesh_config.get_mesh_dimension(geometry_type)

			if mesh_dim not in [1, 2, 3]:
				return {'error': f'Unsupported geometry type: {geometry_type}'}

			# Get geometry config
			geometry_config = mesh_config.get_geometry_config(geometry_type)

			# Get mesh detection info
			mesh_config_info = detect_mesh_dimensions(geometry_type, {})

			return {
				'geometry_type': geometry_type,
				'mesh_dimension': mesh_dim,
				'required_dimensions': mesh_config.get_required_dimensions(geometry_type),
				'geometry_config': geometry_config,
				'mesh_type': mesh_config_info['mesh_type'],
				'gmsh_available': mesh_config.is_gmsh_available(),
				'generator_class': 'GMSHGenerator',
				'description': geometry_config.get('description', 'Unknown geometry')
			}

		except Exception as e:
			return {'error': str(e)}

	def get_available_quality_levels(self) -> list:
		"""Get available mesh quality levels"""
		return mesh_config.get_available_quality_levels()

	def get_quality_settings(self, quality_level: str = 'medium') -> Dict[str, Any]:
		"""Get mesh quality settings for a given level"""
		return mesh_config.get_mesh_quality(quality_level)

	def estimate_mesh_complexity(self, geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
		"""Estimate mesh generation complexity"""
		try:
			from .utils.mesh_detector import get_mesh_generation_strategy
			
			strategy = get_mesh_generation_strategy(geometry_type, dimensions)
			
			return {
				'complexity_score': strategy['complexity_score'],
				'expected_elements': strategy['expected_elements'],
				'optimal_mesh_size': strategy['optimal_mesh_size'],
				'special_handling': strategy['special_handling'],
				'quality_optimization': strategy['quality_optimization'],
				'recommended_algorithm': strategy['recommended_algorithm']
			}
		except Exception as e:
			return {'error': str(e)}

	def get_mesh_statistics(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Get statistics about a generated mesh"""
		if not mesh_data.get('success', False):
			return {'error': 'Invalid mesh data'}

		stats = mesh_data.get('mesh_stats', {})
		physical_groups = mesh_data.get('physical_groups', {})
		
		return {
			'vertices': stats.get('num_vertices', 0),
			'faces': stats.get('num_faces', 0),
			'cells': stats.get('num_cells', 0),
			'cell_types': stats.get('cell_types', []),
			'physical_groups': len(physical_groups),
			'mesh_dimension': mesh_data.get('mesh_dimension', 0),
			'geometry_type': mesh_data.get('geometry_type', 'unknown'),
			'quality': mesh_data.get('mesh_quality', 'unknown'),
			'generator': mesh_data.get('generator_used', 'unknown')
		}

	def compare_mesh_qualities(self, geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
		"""Compare different mesh quality levels for a geometry"""
		qualities = ['coarse', 'medium', 'fine', 'very_fine']
		comparison = {}

		for quality in qualities:
			try:
				settings = self.get_quality_settings(quality)
				mesh_data = self.generate_mesh(geometry_type, dimensions, quality)
				
				if mesh_data.get('success', False):
					stats = self.get_mesh_statistics(mesh_data)
					comparison[quality] = {
						'vertices': stats['vertices'],
						'faces': stats['faces'],
						'cells': stats['cells'],
						'mesh_size': settings.get('gmsh_mesh_size', 0.2),
						'resolution': settings.get('resolution', 50),
						'element_quality': settings.get('element_quality', 0.2)
					}
				else:
					comparison[quality] = {'error': mesh_data.get('error', 'Unknown error')}
			except Exception as e:
				comparison[quality] = {'error': str(e)}

		return comparison

	def _create_error_mesh(self, error_message: str) -> Dict[str, Any]:
		"""Create error mesh response"""
		self.logger.error(error_message)
		return {
			'vertices': [],
			'faces': [],
			'cells': {},
			'physical_groups': {},
			'type': 'error',
			'bounds': {'min': [0, 0, 0], 'max': [0, 0, 0]},
			'mesh_stats': {'vertices': 0, 'faces': 0, 'cells': 0, 'cell_types': []},
			'success': False,
			'error': error_message,
			'mesh_dimension': 0
		}

	def get_generator_status(self) -> Dict[str, Any]:
		"""Get status of mesh generator"""
		supported = self.get_supported_geometries()
		status = {
			'total_generators': 1,
			'generators': {
				'GMSH': {
					'class': 'GMSHGenerator',
					'available': self.gmsh_generator.gmsh_available,
					'supported_geometries': supported
				}
			},
			'gmsh_available': mesh_config.is_gmsh_available(),
			'supported_geometries': supported
		}

		return status

# Global mesh generator instance
# Wrap in try-except to handle GMSH import failures gracefully
try:
    mesh_generator = MeshGenerator()
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to initialize global MeshGenerator: {e}")
    logger.error("Mesh generation will not be available. Please check GMSH installation.")
    # Create a None instance - callers should check for None
    mesh_generator = None