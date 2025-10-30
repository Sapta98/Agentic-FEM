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

		# Extract mesh information
		vertices = mesh_data.get('vertices', [])
		faces = mesh_data.get('faces', [])

		# Extract field information
		field_name = field_data.get('name', 'Field')
		field_values = field_data.get('values', [])
		field_units = field_data.get('units', '')

		# Prepare mesh data for JavaScript
		mesh_data_js = json.dumps({
			'vertices': vertices,
			'faces': faces
		})

		# Prepare field data for JavaScript
		field_data_js = json.dumps({
			'name': field_name,
			'values': field_values,
			'units': field_units
		})

		# Settings
		settings_js = json.dumps(self.settings)
		colormaps_js = json.dumps(self.colormaps)

		# Load template file
		template_path = Path(__file__).parent.parent / "templates" / "field_visualization.html"
		
		with open(template_path, 'r', encoding='utf-8') as f:
			template_content = f.read()

		# Replace placeholders in template
		html_content = template_content.format(
			field_name=field_name,
			field_units=field_units,
			mesh_data_js=mesh_data_js,
			field_data_js=field_data_js,
			settings_js=settings_js,
			colormaps_js=colormaps_js
		)

		return html_content

	def update_settings(self, new_settings: Dict[str, Any]) -> None:
		"""
		Update visualization settings

		Args:
		new_settings: New settings to apply
		"""
		self.settings.update(new_settings)
		logger.info(f"Updated field visualizer settings: {new_settings}")

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