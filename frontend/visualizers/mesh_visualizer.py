"""
Mesh Visualizer Component
=========================

3D mesh visualization component using VTK.js and web-based rendering.
Handles mesh display, interaction, and field visualization.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class MeshVisualizer:
	"""3D mesh visualization component"""

	def __init__(self, static_dir: str = "frontend/static"):
		"""
		Initialize mesh visualizer

		Args:
		static_dir: Directory for static assets
		"""
		self.static_dir = Path(static_dir)
		self.static_dir.mkdir(parents=True, exist_ok=True)

		# Visualization settings
		self.settings = {
			'camera_position': [0, 0, 3],
			'background_color': '#1a1a1a',
			'mesh_color': '#00ff00',
			'wireframe_color': '#ffffff',
			'show_wireframe': True,
			'show_axes': True,
			'auto_rotate': False
		}

		# Current mesh data
		self.current_mesh = None
		self.current_field = None

	def create_mesh_visualization(self, mesh_data: Dict[str, Any], field_data: Optional[Dict[str, Any]] = None) -> str:
		"""
		Create 3D mesh visualization HTML using VTK.js

		Args:
		mesh_data: Mesh geometry data
		field_data: Optional field data for visualization

		Returns:
		HTML content for mesh visualization
		"""
		# Prepare mesh data for VTK
		validated_mesh_data = self._prepare_vtk_mesh_data(mesh_data)

		self.current_mesh = validated_mesh_data
		self.current_field = field_data

		# Generate unique filename
		import time
		timestamp = int(time.time() * 1000)
		filename = f"mesh_visualization_{timestamp}.html"
		filepath = self.static_dir / filename

		# Create visualization HTML
		html_content = self._generate_visualization_html(validated_mesh_data, field_data)

		# Save to static directory
		with open(filepath, 'w', encoding='utf-8') as f:
			f.write(html_content)

		logger.info(f"Created mesh visualization: {filepath}")
		# Return the URL path for serving, not the file path
		filename = filepath.name
		return f"/static/{filename}"

	def _prepare_vtk_mesh_data(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Prepare and validate mesh data for VTK.js compatibility

		Args:
		mesh_data: Raw mesh data

		Returns:
		VTK-compatible mesh data
		"""
		# Validate required fields
		if 'vertices' not in mesh_data or not mesh_data['vertices']:
			raise ValueError("Mesh data must contain vertices")

		vertices = mesh_data['vertices']

		# Ensure vertices are in correct format [[x,y,z], [x,y,z], ...]
		if isinstance(vertices, list) and len(vertices) > 0:
			if isinstance(vertices[0], (int, float)):
				# Flat array - convert to nested format
				if len(vertices) % 3 != 0:
					raise ValueError(f"Flat vertex array length {len(vertices)} must be divisible by 3")
				vertices = [vertices[i:i+3] for i in range(0, len(vertices), 3)]
				logger.info(f"Converted flat vertex array to {len(vertices)} vertices")
			elif isinstance(vertices[0], (list, tuple)):
				# Already in correct format - validate each vertex
				for i, vertex in enumerate(vertices):
					if len(vertex) != 3:
						raise ValueError(f"Vertex {i} must have 3 coordinates (x, y, z), got {len(vertex)}")
					# Ensure all coordinates are numbers
					vertices[i] = [float(coord) for coord in vertex]
				logger.info(f"Validated {len(vertices)} vertices in nested format")
			else:
				raise ValueError(f"Invalid vertex format: expected list of numbers or list of coordinate arrays")
		else:
			raise ValueError("Vertices must be a non-empty list")

		# Prepare faces data
		faces = []
		if 'faces' in mesh_data and mesh_data['faces']:
			faces = mesh_data['faces']
			# Ensure all face indices are integers and validate face structure
			for i, face in enumerate(faces):
				if isinstance(face, (list, tuple)):
					if len(face) < 3:
						logger.warning(f"Face {i} has insufficient vertices: {len(face)} (minimum 3)")
						continue
					faces[i] = [int(idx) for idx in face]
					# Validate that face indices are within bounds
					max_vertex_idx = len(vertices) - 1
					for idx in faces[i]:
						if idx < 0 or idx > max_vertex_idx:
							logger.warning(f"Face {i} has invalid vertex index {idx} (max: {max_vertex_idx})")
				else:
					faces[i] = int(face)
			logger.info(f"Processed {len(faces)} faces")

		# Extract faces from elements if not already present
		if not faces and 'elements' in mesh_data and mesh_data['elements']:
			elements = mesh_data['elements']
			if 'triangles' in elements and elements['triangles']:
				faces = elements['triangles']
				# Ensure all face indices are integers
				for i, face in enumerate(faces):
					if isinstance(face, (list, tuple)):
						faces[i] = [int(idx) for idx in face]
					else:
						faces[i] = int(face)

		# Prepare cells data (VTK format)
		cells = {}
		if 'cells' in mesh_data and mesh_data['cells']:
			cells = mesh_data['cells']
			# Ensure cell indices are integers
			for cell_type, cell_data in cells.items():
				if isinstance(cell_data, list):
					cells[cell_type] = [[int(idx) for idx in cell] for cell in cell_data]

		# Return validated mesh data
		return {
			'vertices': vertices,
			'faces': faces,
			'cells': cells,
			'num_vertices': len(vertices),
			'num_faces': len(faces),
			'num_cells': sum(len(cell_data) for cell_data in cells.values()) if cells else 0,
			'mesh_dimension': mesh_data.get('mesh_dimension', 3)  # Preserve mesh dimension
		}

	def _generate_visualization_html(self, mesh_data: Dict[str, Any], field_data: Optional[Dict[str, Any]] = None) -> str:
		"""
		Generate HTML content for mesh visualization using VTK.js

		Args:
		mesh_data: Validated mesh data
		field_data: Optional field data

		Returns:
		HTML content string
		"""
		# Prepare mesh data for JavaScript
		mesh_data_js = json.dumps(mesh_data)

		# Prepare field data for JavaScript
		# No field data for mesh visualization

		# Settings
		settings_js = json.dumps(self.settings)

		# Load template file
		template_path = Path(__file__).parent.parent / "templates" / "mesh_visualization.html"
		
		with open(template_path, 'r', encoding='utf-8') as f:
			template_content = f.read()

		# Extract individual components for template
		# Flatten vertices from nested arrays to flat array for frontend
		vertices = mesh_data.get('vertices', [])
		if vertices and isinstance(vertices[0], list):
			vertices = [coord for vertex in vertices for coord in vertex]
		vertices_js = json.dumps(vertices)
		faces_js = json.dumps(mesh_data.get('faces', []))
		cells_js = json.dumps(mesh_data.get('cells', {}))
		mesh_dimension_js = json.dumps(mesh_data.get('mesh_dimension', 3))
		
		# Pass mesh data to template (no field data for mesh visualization)
		mesh_data_js = json.dumps(mesh_data)

		# Replace placeholders in template using string replacement
		html_content = template_content
		html_content = html_content.replace('{{mesh_dimension}}', mesh_dimension_js)
		# Create proper field data structure for mesh visualization
		field_data = {
			"values": [],
			"field_name": "Mesh",
			"field_units": "",
			"min_value": 0,
			"max_value": 1,
			"physics_type": "mesh_only",
			"available_fields": []
		}
		field_data_js = json.dumps(field_data)
		html_content = html_content.replace('{{field_data}}', field_data_js)
		html_content = html_content.replace('{{vertices}}', vertices_js)
		html_content = html_content.replace('{{faces}}', faces_js)
		html_content = html_content.replace('{{cells}}', cells_js)
		html_content = html_content.replace('{{mesh_data}}', mesh_data_js)
		html_content = html_content.replace('{{settings}}', settings_js)

		return html_content

	def update_settings(self, new_settings: Dict[str, Any]) -> None:
		"""
		Update visualization settings

		Args:
		new_settings: New settings to apply
		"""
		self.settings.update(new_settings)
		logger.debug(f"Updated mesh visualizer settings: {new_settings}")

	def cleanup_old_visualizations(self, max_age_hours: int = 24) -> int:
		"""
		Clean up old visualization files

		Args:
		max_age_hours: Maximum age of files in hours

		Returns:
		Number of files cleaned up
		"""
		import time

		current_time = time.time()
		max_age_seconds = max_age_hours * 3600
		cleaned_count = 0

		for file_path in self.static_dir.glob("mesh_visualization_*.html"):
			file_age = current_time - file_path.stat().st_mtime
			if file_age > max_age_seconds:
				file_path.unlink()
				cleaned_count += 1
				logger.info(f"Cleaned up old mesh visualization: {file_path}")

		return cleaned_count

	def get_current_mesh_info(self) -> Optional[Dict[str, Any]]:
		"""
		Get information about the current mesh

		Returns:
		Dictionary with mesh information or None if no mesh loaded
		"""
		if self.current_mesh is None:
			return None

		return {
			'num_vertices': self.current_mesh.get('num_vertices', 0),
			'num_faces': self.current_mesh.get('num_faces', 0),
			'num_cells': self.current_mesh.get('num_cells', 0),
			'has_field': self.current_field is not None,
			'field_name': self.current_field.get('name', 'Unknown') if self.current_field else None
		}