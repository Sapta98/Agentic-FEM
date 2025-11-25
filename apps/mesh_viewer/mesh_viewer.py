"""
Mesh Viewer - Geometry-only mesh preview
Handles mesh generation and visualization without materials or boundary conditions
Uses the new comprehensive mesh generation system
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MeshViewer:
	"""Handles geometry-only mesh preview operations"""

	def __init__(self):
		# Add paths for mesh generation
		project_root = Path(__file__).parent.parent.parent
		paths_to_add = [
			project_root / "main_app",
			project_root
		]
		
		for path in paths_to_add:
			if str(path) not in sys.path:
				sys.path.insert(0, str(path))

		# Use new mesh generation system
		from mesh import MeshGenerator
		from frontend.visualizers.mesh_visualizer import MeshVisualizer

		self.mesh_generator = MeshGenerator()
		self.mesh_visualizer = MeshVisualizer("frontend/static")
		logger.debug("Mesh viewer initialized with new mesh system and VTK.js")

	def _make_json_safe(self, obj):
		"""Recursively convert objects to JSON-safe format"""
		import numpy as np
		from types import SimpleNamespace
		
		if isinstance(obj, dict):
			result = {}
			for key, value in obj.items():
				# Skip GMSH-specific fields that can't be JSON serialized
				if key in ['gmsh_model', 'gmsh_initialized', 'gmsh_available']:
					if key == 'gmsh_initialized':
						result['gmsh_model_was_available'] = value
					continue
				
				# CRITICAL: Check if value is a PhysicalGroupWrapper FIRST (before GMSH check)
				# PhysicalGroupWrapper might be from a module with 'gmsh' in the name, but it's JSON-serializable
				if key == 'physical_groups' and isinstance(value, dict):
					# Special handling for physical_groups dictionary
					result[key] = {}
					for pg_key, pg_value in value.items():
						if hasattr(pg_value, '_data') and hasattr(pg_value, '__dict__') and hasattr(pg_value, 'dim') and hasattr(pg_value, 'tag'):
							# It's a PhysicalGroupWrapper, convert to dict directly
							# CRITICAL: Use _data dict directly to ensure node_tags is included
							if isinstance(pg_value._data, dict):
								result[key][pg_key] = self._make_json_safe(pg_value._data)
							else:
								# Fallback: manually construct dict
								result[key][pg_key] = {
									'dim': pg_value.dim,
									'tag': pg_value.tag,
									'entities': self._make_json_safe(getattr(pg_value, 'entities', [])),
									'name': getattr(pg_value, 'name', None),
									'dimension': getattr(pg_value, 'dim', None),
									'entity_coordinates': self._make_json_safe(getattr(pg_value, 'entity_coordinates', [])),
									'node_tags': self._make_json_safe(getattr(pg_value, 'node_tags', []))  # CRITICAL: Include node_tags
								}
						else:
							# Already a dict or other type, recurse
							result[key][pg_key] = self._make_json_safe(pg_value)
				elif hasattr(value, '_data') and hasattr(value, '__dict__') and hasattr(value, 'dim') and hasattr(value, 'tag'):
					# It's a PhysicalGroupWrapper, convert to dict directly
					# CRITICAL: Use _data dict directly to ensure node_tags is included
					if isinstance(value._data, dict):
						result[key] = self._make_json_safe(value._data)
					else:
						# Fallback: manually construct dict
						result[key] = {
							'dim': value.dim,
							'tag': value.tag,
							'entities': self._make_json_safe(getattr(value, 'entities', [])),
							'name': getattr(value, 'name', None),
							'dimension': getattr(value, 'dim', None),
							'entity_coordinates': self._make_json_safe(getattr(value, 'entity_coordinates', [])),
							'node_tags': self._make_json_safe(getattr(value, 'node_tags', []))  # CRITICAL: Include node_tags
						}
				elif hasattr(value, '__module__') and 'gmsh' in str(value.__module__):
					# Skip any GMSH-related objects (but not PhysicalGroupWrapper which we already handled)
					continue
				else:
					result[key] = self._make_json_safe(value)
			return result
		elif isinstance(obj, list):
			return [self._make_json_safe(item) for item in obj]
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, (np.integer, np.floating)):
			return obj.item()
		elif isinstance(obj, SimpleNamespace):
			# Convert SimpleNamespace to dict
			return self._make_json_safe(obj.__dict__)
		elif hasattr(obj, '_data') and hasattr(obj, '__dict__'):
			# Handle PhysicalGroupWrapper or similar objects with _data attribute
			if hasattr(obj, 'dim') and hasattr(obj, 'tag'):
				# It's a physical group wrapper, convert to dict
				return {
					'dim': obj.dim,
					'tag': obj.tag,
					'entities': self._make_json_safe(getattr(obj, 'entities', [])),
					'name': getattr(obj, 'name', None),
					'dimension': getattr(obj, 'dim', None)
				}
		elif hasattr(obj, '__dict__') and not isinstance(obj, type):
			# Convert objects with __dict__ to dict
			return self._make_json_safe(obj.__dict__)
		elif hasattr(obj, '__module__') and 'gmsh' in str(obj.__module__):
			# Skip GMSH objects
			return None
		else:
			return obj


	def create_mesh_visualization(self, mesh_data: Dict[str, Any], field_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Create mesh visualization from existing mesh data (visualization only)"""
		try:
			logger.debug("Creating mesh visualization from existing mesh data")
			
			# Create VTK.js 3D visualization
			logger.debug(f"Mesh visualizer available: {self.mesh_visualizer is not None}")
			visualization_file = self.mesh_visualizer.create_mesh_visualization(mesh_data, field_data)
			logger.debug(f"Visualization file result: {visualization_file}")

			if visualization_file:
				return {
					"success": True,
					"visualization_url": visualization_file
				}
			else:
				return {
					"success": False,
					"error": "Failed to create visualization file"
				}
				
		except Exception as e:
			logger.error(f"Error creating mesh visualization: {e}")
			return {
				"success": False,
				"error": str(e)
			}

	def generate_mesh_preview(self, geometry_type: str, dimensions: Dict[str, float], field_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""
		Generate mesh preview using geometry data only

		Args:
			geometry_type: Type of geometry (beam, cylinder, cube, etc.)
			dimensions: Geometry dimensions
			field_data: Optional field data for visualization

		Returns:
			Dict with success status and mesh visualization URL
		"""
		logger.debug(f"Generating mesh preview for {geometry_type}")
		logger.debug(f"Dimensions: {dimensions}")

		# Generate mesh data using new mesh system (geometry only)
		mesh_data = self.mesh_generator.generate_mesh(
			geometry_type=geometry_type,
			dimensions=dimensions,
			mesh_quality='medium'
		)

		if not mesh_data.get('success', False):
			return {
				"success": False,
				"error": mesh_data.get('error', 'Failed to generate mesh data')
			}

		
		# Create visualization config (geometry only)
		config = {
			"geometry_type": geometry_type,
			"dimensions": dimensions
		}

		# Create VTK.js 3D visualization
		logger.debug(f"Mesh visualizer available: {self.mesh_visualizer is not None}")
		try:
			logger.debug("Creating mesh visualization...")
			visualization_file = self.mesh_visualizer.create_mesh_visualization(mesh_data, field_data)
			logger.debug(f"Visualization file result: {visualization_file}")

			if visualization_file:
				# Use the URL returned by the visualizer (includes full URL with port)
				visualization_url = visualization_file
				logger.debug(f"Returning visualization URL: {visualization_url}")
				
				# Log physical_groups before making JSON-safe (debug only)
				if 'physical_groups' in mesh_data:
					pg_count = len(mesh_data['physical_groups']) if isinstance(mesh_data['physical_groups'], dict) else 0
					pg_keys = list(mesh_data['physical_groups'].keys())[:10] if isinstance(mesh_data['physical_groups'], dict) else []
					logger.debug(f"Before _make_json_safe: mesh_data has {pg_count} physical_groups: {pg_keys}")
				else:
					logger.warning("Before _make_json_safe: mesh_data does NOT contain 'physical_groups'")
				
				# Make mesh data JSON-safe for API response
				json_safe_mesh_data = self._make_json_safe(mesh_data)
				
				# Log physical_groups after making JSON-safe (debug only)
				if 'physical_groups' in json_safe_mesh_data:
					pg_count = len(json_safe_mesh_data['physical_groups']) if isinstance(json_safe_mesh_data['physical_groups'], dict) else 0
					pg_keys = list(json_safe_mesh_data['physical_groups'].keys())[:10] if isinstance(json_safe_mesh_data['physical_groups'], dict) else []
					logger.debug(f"After _make_json_safe: json_safe_mesh_data has {pg_count} physical_groups: {pg_keys}")
				else:
					logger.warning("After _make_json_safe: json_safe_mesh_data does NOT contain 'physical_groups'")
				
				return {
					"success": True,
					"mesh_visualization_url": visualization_url,
					"geometry_type": geometry_type,
					"dimensions": dimensions,
					"mesh_dimension": mesh_data.get('mesh_dimension', 3),
					"mesh_stats": mesh_data.get('mesh_stats', {}),
					"mesh_type": mesh_data.get('type', 'unknown'),
					"generator": mesh_data.get('generator_used', 'unknown'),
					"mesh_data": json_safe_mesh_data  # JSON-safe mesh data for API
				}
			else:
				logger.error("Visualization file creation returned None")
				json_safe_mesh_data = self._make_json_safe(mesh_data)
				return {
					"success": False,
					"error": "Failed to create visualization file",
					"mesh_data": json_safe_mesh_data,  # JSON-safe mesh data
					"geometry_type": geometry_type,
					"dimensions": dimensions,
					"mesh_dimension": mesh_data.get('mesh_dimension', 3),
					"mesh_stats": mesh_data.get('mesh_stats', {}),
					"mesh_type": mesh_data.get('type', 'unknown'),
					"generator": mesh_data.get('generator', 'unknown')
				}
		except Exception as viz_error:
			logger.error(f"Error creating VTK visualization: {viz_error}")
			import traceback
			logger.error(f"Traceback: {traceback.format_exc()}")
			json_safe_mesh_data = self._make_json_safe(mesh_data)
			return {
				"success": False,
				"error": f"Failed to create VTK visualization: {str(viz_error)}",
				"mesh_data": json_safe_mesh_data,  # JSON-safe mesh data
				"geometry_type": geometry_type,
				"dimensions": dimensions,
				"mesh_dimension": mesh_data.get('mesh_dimension', 3),
				"mesh_stats": mesh_data.get('mesh_stats', {}),
				"mesh_type": mesh_data.get('type', 'unknown'),
				"generator": mesh_data.get('generator', 'unknown')
			}

	def get_supported_geometries(self) -> Dict[str, Dict[str, Any]]:
		"""Get list of supported geometry types and their required dimensions"""
		# Use the new mesh system's supported geometries
		supported_by_dim = self.mesh_generator.get_supported_geometries()

		# Import mesh_config to get required dimensions
		from mesh.config.mesh_config import mesh_config

		# Flatten and format for API
		geometries = {}
		for dim, geo_list in supported_by_dim.items():
			for geo_type in geo_list:
				# Get required dimensions from mesh_config
				required_dims = mesh_config.get_required_dimensions(geo_type)

				# Create example dimensions
				example = {}
				for dim_name in required_dims:
					if dim_name == 'radius':
						example[dim_name] = 0.5
					elif dim_name == 'length':
						example[dim_name] = 1.0
					elif dim_name == 'width':
						example[dim_name] = 0.5
					elif dim_name == 'height':
						example[dim_name] = 0.3
					elif dim_name == 'thickness':
						example[dim_name] = 0.02

				geometries[geo_type] = {
					"description": f"{dim}D {geo_type} geometry",
					"required_dimensions": required_dims,
					"mesh_dimension": dim,
					"example": example
				}

		return geometries

	def validate_geometry(self, geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
		"""Validate geometry type and dimensions"""
		# Use the new mesh system's validation
		return self.mesh_generator.validate_geometry(geometry_type, dimensions)