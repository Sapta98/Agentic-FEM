"""
Field Visualizer Component
===========================

Field data visualization component for displaying solution fields
on meshes (temperature, displacement, stress, etc.).
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class FieldVisualizer:
	"""Field data visualization component"""

	def __init__(self, static_dir: str = "frontend/static"):
		"""
		Initialize field visualizer

		Args:
		static_dir: Directory for static assets
		"""
		self.static_dir = Path(static_dir)
		self.static_dir.mkdir(parents=True, exist_ok=True)

		# Visualization settings
		self.settings = {
			'colormap': 'viridis',
			'field_range': 'auto',
		'min_value': 0,
		'max_value': 1,
		'show_contours': True,
		'contour_levels': 10,
		'show_colorbar': True,
		'opacity': 0.8
		}

		# Available colormaps
		self.colormaps = {
		'viridis': ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f',
		'#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'],
		'plasma': ['#0c0786', '#6a00a8', '#b02a8f', '#e16462', '#fca636', '#f0f921'],
		'inferno': ['#000004', '#1b0c42', '#4a0c6b', '#781c6d', '#a52c60',
		'#d14e53', '#f37651', '#feb078', '#f0f921'],
		'magma': ['#000004', '#1c1044', '#4c1a6b', '#7c1d70', '#ad2268',
		'#dd4968', '#f66e5c', '#fca636', '#f0f921'],
		'jet': ['#000080', '#0000ff', '#0080ff', '#00ffff', '#80ff00',
		'#ffff00', '#ff8000', '#ff0000', '#800000']
		}

	def create_field_visualization(self, mesh_data: Dict[str, Any], field_data: Dict[str, Any]) -> str:
		"""
		Create field visualization HTML

		Args:
		mesh_data: Mesh geometry data
		field_data: Field solution data

		Returns:
		HTML content for field visualization
		"""
		# Generate unique filename
		import time
		timestamp = int(time.time() * 1000)
		filename = f"field_visualization_{timestamp}.html"
		filepath = self.static_dir / filename

		# Create visualization HTML
		html_content = self._generate_field_visualization_html(mesh_data, field_data)

		# Save to static directory
		with open(filepath, 'w', encoding='utf-8') as f:
			f.write(html_content)

			logger.info(f"Created field visualization: {filepath}")
			return str(filepath)

	def _generate_field_visualization_html(self, mesh_data: Dict[str, Any], field_data: Dict[str, Any]) -> str:
		"""Generate HTML content for field visualization"""

		# Extract mesh and field information
		# Handle case where solution_data is passed as first arg (has 'coordinates', 'values', 'field_name', 'faces')
		# This is the common case when called from solver_agent
		solution_data = None
		if 'coordinates' in mesh_data and 'values' in mesh_data:
			# First arg is actually solution_data, which contains both mesh and field data
			solution_data = mesh_data
			# Use deformed_coordinates if available (for solid mechanics deflection visualization)
			# Otherwise use original coordinates
			if 'deformed_coordinates' in mesh_data and mesh_data['deformed_coordinates']:
				vertices = mesh_data['deformed_coordinates']
				logger.info(f"Using deformed_coordinates for visualization: {len(vertices)} vertices")
			else:
				vertices = mesh_data.get('coordinates', [])
				logger.debug(f"Using original coordinates (no deformed_coordinates available): {len(vertices)} vertices")
			faces = mesh_data.get('faces', [])
			field_values = mesh_data.get('values', [])
		elif 'vertices' in mesh_data:
			# Standard case: mesh_data has vertices/faces, field_data has field info
			solution_data = field_data
			# Use deformed_coordinates if available (for solid mechanics deflection visualization)
			vertices = field_data.get('deformed_coordinates', mesh_data.get('vertices', []))
			faces = mesh_data.get('faces', [])
			field_values = field_data.get('values', [])
		else:
			# Fallback: try to extract from field_data (may contain both mesh and field data)
			solution_data = field_data
			# Use deformed_coordinates if available (for solid mechanics deflection visualization)
			vertices = field_data.get('deformed_coordinates', field_data.get('coordinates', field_data.get('vertices', [])))
			faces = field_data.get('faces', [])
			field_values = field_data.get('values', [])
		
		# Get field information from solution structure
		# Priority: field_info structure > top-level field_name > fallback
		field_info = solution_data.get('field_info', {}) if solution_data else {}
		physics_type = solution_data.get('physics_type', 'unknown') if solution_data else field_data.get('physics_type', 'unknown')
		is_transient = bool(solution_data.get('is_transient')) if solution_data else False
		time_steps = solution_data.get('time_steps', []) if solution_data else []
		time_series = solution_data.get('time_series') if solution_data else None
		if time_series is None and solution_data:
			time_series = solution_data.get('solutions', [])
		time_stepping = solution_data.get('time_stepping', {}) if solution_data else {}
		initial_conditions = solution_data.get('initial_conditions', {}) if solution_data else {}
		
		# Try to get field name from field_info structure first
		if field_info:
			# Get visualization_field or primary_field from field_info
			visualization_field = field_info.get('visualization_field') or field_info.get('primary_field')
			if visualization_field and 'fields' in field_info:
				field_data_info = field_info['fields'].get(visualization_field, {})
				field_name = field_data_info.get('name', solution_data.get('field_name') if solution_data else 'Field')
				field_units = field_data_info.get('units', solution_data.get('field_units') if solution_data else '')
			else:
				# Fallback to top-level values
				field_name = solution_data.get('field_name') or solution_data.get('name', 'Field') if solution_data else 'Field'
				field_units = solution_data.get('field_units') or solution_data.get('units', '') if solution_data else ''
		else:
			# No field_info, use top-level values
			field_name = solution_data.get('field_name') or solution_data.get('name', 'Field') if solution_data else 'Field'
			field_units = solution_data.get('field_units') or solution_data.get('units', '') if solution_data else ''
		
		# Create display name: "{field_name} Field" (e.g., "Temperature Field", "Deflection Field")
		field_display_name = f"{field_name} Field" if field_name else "Field Visualization"

		# Prepare mesh data for JavaScript
		mesh_data_js = json.dumps({
			'vertices': vertices,
			'faces': faces
		})

		# Prepare field data for JavaScript
		# Include both 'name' and 'field_name' for backward compatibility
		field_data_payload = {
			'name': field_name,
			'field_name': field_name,  # Also include as 'field_name' for consistency
			'values': field_values,
			'units': field_units,
			'field_units': field_units,  # Also include as 'field_units' for consistency
			'physics_type': physics_type,  # Include physics type
			'field_info': field_info,  # Include field_info for available fields
			'is_transient': is_transient,
			'time_steps': time_steps,
			'time_series': time_series or [],
			'time_stepping': time_stepping,
			'initial_conditions': initial_conditions
		}
		field_data_js = json.dumps(field_data_payload)

		# Settings
		settings_js = json.dumps(self.settings)
		colormaps_js = json.dumps(self.colormaps)

		# Load template file
		template_path = Path(__file__).parent.parent / "templates" / "field_visualization.html"
		
		with open(template_path, 'r', encoding='utf-8') as f:
			template_content = f.read()

		# Replace placeholders in template using .replace() to avoid issues with curly braces in JSON
		html_content = template_content
		html_content = html_content.replace('{field_name}', field_name)
		html_content = html_content.replace('{field_units}', field_units)
		html_content = html_content.replace('{field_display_name}', field_display_name)  # e.g., "Temperature Field", "Displacement Field"
		html_content = html_content.replace('{mesh_data_js}', mesh_data_js)
		html_content = html_content.replace('{field_data_js}', field_data_js)
		html_content = html_content.replace('{settings_js}', settings_js)
		html_content = html_content.replace('{colormaps_js}', colormaps_js)

		return html_content

	def update_settings(self, new_settings: Dict[str, Any]) -> None:
		"""
		Update visualization settings

		Args:
		new_settings: New settings to apply
		"""
		self.settings.update(new_settings)
		logger.debug(f"Updated field visualizer settings: {new_settings}")

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

		for file_path in self.static_dir.glob("field_visualization_*.html"):
			file_age = current_time - file_path.stat().st_mtime
			if file_age > max_age_seconds:
				file_path.unlink()
				cleaned_count += 1
				logger.info(f"Cleaned up old field visualization: {file_path}")

		return cleaned_count