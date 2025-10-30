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
		logger.info("Mesh viewer initialized with new mesh system and VTK.js")

	def _make_json_safe(self, obj):
		"""Recursively convert objects to JSON-safe format"""
		import numpy as np
		
		if isinstance(obj, dict):
			result = {}
			for key, value in obj.items():
				# Skip GMSH-specific fields that can't be JSON serialized
				if key in ['gmsh_model', 'gmsh_initialized', 'gmsh_available']:
					if key == 'gmsh_initialized':
						result['gmsh_model_was_available'] = value
					continue
				elif hasattr(value, '__module__') and 'gmsh' in str(value.__module__):
					# Skip any GMSH-related objects
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
		elif hasattr(obj, '__module__') and 'gmsh' in str(obj.__module__):
			# Skip GMSH objects
			return None
		else:
			return obj


	def create_mesh_visualization(self, mesh_data: Dict[str, Any], field_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Create mesh visualization from existing mesh data (visualization only)"""
		try:
			logger.info("Creating mesh visualization from existing mesh data")
			
			# Create VTK.js 3D visualization
			logger.info(f"Mesh visualizer available: {self.mesh_visualizer is not None}")
			visualization_file = self.mesh_visualizer.create_mesh_visualization(mesh_data, field_data)
			logger.info(f"Visualization file result: {visualization_file}")

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
		logger.info(f"Generating mesh preview for {geometry_type}")
		logger.info(f"Dimensions: {dimensions}")

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
		logger.info(f"Mesh visualizer available: {self.mesh_visualizer is not None}")
		try:
			logger.info("Creating mesh visualization...")
			visualization_file = self.mesh_visualizer.create_mesh_visualization(mesh_data, field_data)
			logger.info(f"Visualization file result: {visualization_file}")

			if visualization_file:
				# Use the URL returned by the visualizer (includes full URL with port)
				visualization_url = visualization_file
				logger.info(f"Returning visualization URL: {visualization_url}")
				# Make mesh data JSON-safe for API response
				json_safe_mesh_data = self._make_json_safe(mesh_data)
				
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

		# Flatten and format for API
		geometries = {}
		for dim, geo_list in supported_by_dim.items():
			for geo_type in geo_list:
				# Get required dimensions from mesh generator
				required_dims = self.mesh_generator.generators[dim].get_required_dimensions(geo_type)

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