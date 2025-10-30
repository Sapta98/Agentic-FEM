"""
Base Mesh Generator
===================

Base class for all mesh generators providing common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseMeshGenerator(ABC):
	"""Base class for mesh generators"""

	def __init__(self, mesh_dimension: int):
		self.mesh_dimension = mesh_dimension
		self.logger = logger

	@abstractmethod
	def generate_mesh(self, geometry_type: str, dimensions: Dict[str, float], mesh_parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate mesh for given geometry and parameters"""
		pass

	@abstractmethod
	def get_required_dimensions(self, geometry_type: str) -> List[str]:
		"""Get list of required dimensions for geometry type"""
		pass

	def validate_dimensions(self, geometry_type: str, dimensions: Dict[str, float]) -> bool:
		"""Validate that required dimensions are provided"""
		from ..config.mesh_config import mesh_config
		
		required_dims = self.get_required_dimensions(geometry_type)
		
		# Get geometry config to check for alternative dimensions
		geometry_config = mesh_config.get_geometry_config(geometry_type)
		alternative_dims = geometry_config.get('alternative_dimensions', {})
		
		# Check for missing dimensions, considering alternatives
		missing_dims = []
		for dim in required_dims:
			if dim not in dimensions or dimensions.get(dim, 0) <= 0:
				# Check if there's an alternative dimension name
				alternatives = alternative_dims.get(dim, [])
				has_alternative = any(alt in dimensions and dimensions.get(alt, 0) > 0 for alt in alternatives)
				if not has_alternative:
					missing_dims.append(dim)

		if missing_dims:
			self.logger.error(f"Missing or invalid dimensions for {geometry_type}: {missing_dims}")
			return False

		return True

	def calculate_mesh_statistics(self, vertices: List[List[float]], faces: List[List[int]], cells: Dict[str, List[List[int]]]) -> Dict[str, Any]:
		"""Calculate mesh statistics"""
		stats = {
			'vertices': len(vertices),
			'faces': len(faces),
			'cells': sum(len(cell_list) for cell_list in cells.values()),
			'cell_types': list(cells.keys()),
			'mesh_dimension': self.mesh_dimension
		}

		# Calculate mesh quality metrics if possible
		if vertices and faces:
			stats['mesh_quality'] = self._calculate_mesh_quality(vertices, faces)

		return stats

	def _calculate_mesh_quality(self, vertices: List[List[float]], faces: List[List[int]]) -> Dict[str, float]:
		"""Calculate basic mesh quality metrics"""
		if not vertices or not faces:
			return {}

		try:
			# Calculate average edge length
			edge_lengths = []
			for face in faces:
				if len(face) >= 2:
					for i in range(len(face)):
						v1 = np.array(vertices[face[i]])
						v2 = np.array(vertices[face[(i+1) % len(face)]])
						edge_lengths.append(np.linalg.norm(v2 - v1))

			if edge_lengths:
				avg_edge_length = np.mean(edge_lengths)
				min_edge_length = np.min(edge_lengths)
				max_edge_length = np.max(edge_lengths)

				return {
					'average_edge_length': float(avg_edge_length),
					'min_edge_length': float(min_edge_length),
					'max_edge_length': float(max_edge_length),
					'aspect_ratio': float(max_edge_length / min_edge_length) if min_edge_length > 0 else 1.0
				}

		except Exception as e:
			self.logger.warning(f"Could not calculate mesh quality: {e}")

		return {}

	def create_bounds(self, vertices: List[List[float]]) -> Dict[str, List[float]]:
		"""Calculate bounding box for vertices"""
		if not vertices:
			return {'min': [0, 0, 0], 'max': [0, 0, 0]}

		vertices_array = np.array(vertices)
		return {
			'min': vertices_array.min(axis=0).tolist(),
			'max': vertices_array.max(axis=0).tolist()
		}

	def log_mesh_info(self, geometry_type: str, stats: Dict[str, Any]):
		"""Log mesh generation information"""
		self.logger.info(f"Generated {self.mesh_dimension}D mesh for {geometry_type}")
		self.logger.info(f"  Vertices: {stats['vertices']}")
		self.logger.info(f"  Faces: {stats['faces']}")
		self.logger.info(f"  Cells: {stats['cells']}")
		self.logger.info(f"  Cell types: {stats['cell_types']}")

	def get_mesh_bounds(self, vertices: List[List[float]]) -> Dict[str, Any]:
		"""Get comprehensive mesh bounds information"""
		if not vertices:
			return {}

		vertices_array = np.array(vertices)
		bounds = {
			'min': vertices_array.min(axis=0).tolist(),
			'max': vertices_array.max(axis=0).tolist(),
			'center': vertices_array.mean(axis=0).tolist(),
			'size': (vertices_array.max(axis=0) - vertices_array.min(axis=0)).tolist()
		}

		return bounds

	def validate_mesh_data(self, mesh_data: Dict[str, Any]) -> bool:
		"""Validate mesh data structure"""
		required_keys = ['vertices', 'faces', 'cells']
		
		for key in required_keys:
			if key not in mesh_data:
				self.logger.error(f"Missing required mesh data key: {key}")
				return False

		vertices = mesh_data.get('vertices', [])
		faces = mesh_data.get('faces', [])
		cells = mesh_data.get('cells', {})

		if not vertices:
			self.logger.error("No vertices found in mesh data")
			return False

		if not isinstance(vertices, list):
			self.logger.error("Vertices must be a list")
			return False

		if not isinstance(faces, list):
			self.logger.error("Faces must be a list")
			return False

		if not isinstance(cells, dict):
			self.logger.error("Cells must be a dictionary")
			return False

		return True

	def get_mesh_summary(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Get a summary of mesh data"""
		if not self.validate_mesh_data(mesh_data):
			return {}

		vertices = mesh_data['vertices']
		faces = mesh_data['faces']
		cells = mesh_data['cells']

		summary = {
			'num_vertices': len(vertices),
			'num_faces': len(faces),
			'num_cells': sum(len(cell_list) for cell_list in cells.values()),
			'cell_types': list(cells.keys()),
			'bounds': self.get_mesh_bounds(vertices),
			'mesh_dimension': self.mesh_dimension
		}

		# Add mesh quality if available
		if vertices and faces:
			summary['mesh_quality'] = self._calculate_mesh_quality(vertices, faces)

		return summary