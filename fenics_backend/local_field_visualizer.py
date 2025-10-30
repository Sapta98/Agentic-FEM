"""
Field Visualization Module for Agentic FEM System
Visualizes solution fields (temperature, displacement, stress) on 3D meshes
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import time

logger = logging.getLogger(__name__)

class FieldVisualizer:
	"""Visualizes FEM solution fields on 3D meshes"""

	def __init__(self):
		self.solution_data = None
		self.mesh_data = None
		self.field_type = None

	def create_field_visualization(self, solution_result: Dict[str, Any], mesh_data: Dict[str, Any]) -> str:
		"""
		Create HTML visualization of solution fields on mesh

		Args:
		solution_result: Result from FEniCS solver
		mesh_data: Original mesh data (fallback if GMSH mesh not available)

		Returns:
		Path to generated HTML visualization file
		"""
		logger.debug("Creating field visualization")

		# The solution data is now at the top level of the FEniCS result
		self.solution_data = solution_result
		
		# Log structure at debug level
		logger.debug(f"Solution result keys: {list(solution_result.keys())}")
		logger.debug(f"Solution data keys: {list(self.solution_data.keys())}")
		logger.debug(f"Coordinates length: {len(self.solution_data.get('coordinates', []))}")
		logger.debug(f"Values length: {len(self.solution_data.get('values', []))}")
		logger.debug(f"Cells length: {len(self.solution_data.get('cells', []))}")
		
		# Debug: Log the actual values
		if self.solution_data.get('values'):
			logger.debug(f"First 5 values: {self.solution_data.get('values')[:5]}")
			logger.debug(f"Last 5 values: {self.solution_data.get('values')[-5:]}")
		else:
			logger.warning("No values found in solution_data!")
			logger.info(f"Solution data content: {self.solution_data}")
		
		# Use GMSH mesh data directly (now provided by FEniCS solver)
		logger.debug("Using GMSH mesh data from FEniCS solution")
		self.mesh_data = {
			'faces': solution_result.get('faces', []),
			'cells': solution_result.get('cells', {}),
			'vertices': solution_result.get('coordinates', []),  # GMSH vertices
			'mesh_dimension': solution_result.get('mesh_info', {}).get('dimension', 3)
		}

		self.field_type = solution_result.get('physics_type', 'unknown')

		# Generate visualization HTML
		html_content = self._generate_field_html(solution_result)

		# Save to file
		viz_file = self._save_visualization_file(html_content)

		logger.info(f"Field visualization created: {viz_file}")
		return viz_file

	def _generate_field_html(self, solution_result: Dict[str, Any]) -> str:
		"""Generate HTML content for field visualization using VTK.js template"""

		# Get GMSH mesh data directly from solution_result (no reordering needed)
		coordinates = solution_result.get('coordinates', [])  # GMSH vertices
		values = solution_result.get('values', [])  # Solution mapped to GMSH vertices
		cells = solution_result.get('cells', {})  # GMSH cells (dict format)
		faces = solution_result.get('faces', [])  # GMSH faces
		
		logger.debug(f"Using GMSH data: {len(coordinates)} vertices, {len(values)} values, {len(cells)} cell types, {len(faces)} faces")
		
		# Handle mesh-only visualization (no solution data)
		if not coordinates or not values:
			logger.debug(f"Empty solution data: coordinates={len(coordinates)}, values={len(values)}")
			logger.debug("Creating mesh-only visualization")
			return self._generate_mesh_only_html()

		# Get field information
		field_name = solution_result.get('field_name', 'Unknown Field')
		field_units = solution_result.get('field_units', '')
		field_type = solution_result.get('field_type', 'scalar')
		
		# Calculate min/max values from actual field data
		min_val = float(np.min(values))
		max_val = float(np.max(values))
		logger.debug(f"Field value range: {min_val:.3f} to {max_val:.3f}")

		# Use GMSH mesh data directly (no complex extraction needed)
		mesh_data_js = {
			"vertices": coordinates,  # GMSH vertices
			"faces": faces,           # GMSH faces
			"cells": cells            # GMSH cells (dict format)
		}
		
		logger.debug(f"Using GMSH mesh data: {len(coordinates)} vertices, {len(faces)} faces, {len(cells)} cell types")

		# Create lookup table in Python
		lut_data = self._create_lookup_table(min_val, max_val, colormap="viridis")
		
		# Create field data structure for JavaScript
		field_data_js = {
			"values": values,  # Solution values mapped to GMSH vertices
			"field_name": field_name,
			"field_units": field_units,
			"field_type": field_type,
			"min_value": min_val,
			"max_value": max_val,
			"lookup_table": lut_data  # Add lookup table data
		}

		# Create settings for field visualization
		settings_js = {
			"colormap": "viridis",
			"opacity": 0.8,
			"show_colorbar": True,
			"background_color": [0.1, 0.1, 0.1]
		}
		
		# Convert to JSON strings
		mesh_data_js_str = json.dumps(mesh_data_js)
		field_data_js_str = json.dumps(field_data_js)
		settings_js_str = json.dumps(settings_js)

		# Load template file
		template_path = Path(__file__).parent.parent / "frontend" / "templates" / "field_visualization.html"
		
		with open(template_path, 'r', encoding='utf-8') as f:
			template_content = f.read()

		# Replace placeholders in template
		html_content = template_content
		html_content = html_content.replace('{field_name}', field_name)
		html_content = html_content.replace('{field_units}', field_units)
		html_content = html_content.replace('{field_type}', field_type)
		html_content = html_content.replace('{min_val}', str(min_val)) if 'min_val' in template_content else html_content
		html_content = html_content.replace('{max_val}', str(max_val)) if 'max_val' in template_content else html_content

		# FORCE REPLACEMENT - Debug what we're doing
		logger.debug("BEFORE REPLACEMENT:")
		logger.debug(f"  Template contains {{mesh_data_js}}: {'{mesh_data_js}' in template_content}")
		logger.debug(f"  Template contains {{field_data_js}}: {'{field_data_js}' in template_content}")
		logger.debug(f"  mesh_data_js_str: {mesh_data_js_str[:100]}...")
		logger.debug(f"  field_data_js_str: {field_data_js_str[:100]}...")
		
		# Replace the placeholders
		html_content = html_content.replace('{mesh_data_js}', mesh_data_js_str)
		html_content = html_content.replace('{field_data_js}', field_data_js_str)
		html_content = html_content.replace('{settings_js}', settings_js_str)
		
		# Debug after replacement
		logger.debug("AFTER REPLACEMENT:")
		logger.debug(f"  HTML contains meshData: {'meshData' in html_content}")
		logger.debug(f"  HTML contains fieldData: {'fieldData' in html_content}")
		logger.debug(f"  HTML contains cells: {'cells' in html_content}")
		logger.debug(f"  HTML contains line: {'line' in html_content}")

		return html_content

	def _save_visualization_file(self, html_content: str) -> str:
		"""Save HTML content to file and return URL path"""
		try:
			# Get frontend directory
			frontend_dir = Path(__file__).parent.parent / "frontend" / "static"
			logger.debug(f"Frontend directory: {frontend_dir}")
			logger.debug(f"Frontend directory exists: {frontend_dir.exists()}")
			
			# Create directory if it doesn't exist
			frontend_dir.mkdir(parents=True, exist_ok=True)
			logger.debug(f"Directory created/verified: {frontend_dir.exists()}")

			# Generate unique filename
			timestamp = int(time.time() * 1000)
			filename = f"field_visualization_{timestamp}.html"
			file_path = frontend_dir / filename
			logger.debug(f"File path: {file_path}")

			# Write HTML content
			logger.debug(f"Writing HTML content ({len(html_content)} characters) to {file_path}")
			with open(file_path, 'w', encoding='utf-8') as f:
				f.write(html_content)
			
			# Verify file was created
			if file_path.exists():
				logger.debug(f"File created successfully: {file_path}")
				logger.debug(f"File size: {file_path.stat().st_size} bytes")
			else:
				logger.error(f"File was not created: {file_path}")

			return f"/static/{filename}"
			
		except Exception as e:
			logger.error(f"Error saving visualization file: {e}")
			logger.error(f"Exception type: {type(e)}")
			import traceback
			logger.error(f"Traceback: {traceback.format_exc()}")
			raise

	def _generate_mesh_only_html(self) -> str:
		"""Generate HTML content for mesh-only visualization"""
		logger.info("Generating mesh-only visualization")
		
		# Use mesh data from the mesh_data parameter
		coordinates = self.mesh_data.get('vertices', [])
		faces = self.mesh_data.get('faces', [])
		cells = self.mesh_data.get('cells', {})
		
		if not coordinates:
			raise ValueError("No mesh data available for visualization")
		
		# Create mesh data structure for JavaScript
		mesh_data_js = {
			"vertices": coordinates,
			"faces": faces,
			"cells": cells
		}
		
		# Create empty field data for mesh-only visualization
		field_data_js = {
			"values": [],
			"field_name": "Mesh",
			"field_units": "",
			"min_value": 0,
			"max_value": 1
		}
		
		# Create settings for mesh-only visualization
		settings_js = {
			"colormap": "viridis",
			"opacity": 0.8,
			"show_colorbar": False,
			"background_color": [0.1, 0.1, 0.1]
		}
		
		# Convert to JSON strings
		mesh_data_js_str = json.dumps(mesh_data_js)
		field_data_js_str = json.dumps(field_data_js)
		settings_js_str = json.dumps(settings_js)
		
		# Load template file
		template_path = Path(__file__).parent.parent / "frontend" / "templates" / "field_visualization.html"
		with open(template_path, 'r') as f:
			template_content = f.read()
		
		# Replace placeholders
		html_content = template_content.replace('{mesh_data_js}', mesh_data_js_str)
		html_content = html_content.replace('{field_data_js}', field_data_js_str)
		html_content = html_content.replace('{settings_js}', settings_js_str)
		html_content = html_content.replace('{field_name}', 'Mesh Structure')
		html_content = html_content.replace('{field_units}', '')
		html_content = html_content.replace('{min_val}', '0')
		html_content = html_content.replace('{max_val}', '1')
		
		return html_content

	def _create_lookup_table(self, min_val: float, max_val: float, colormap: str = "viridis") -> Dict[str, Any]:
		"""Create lookup table data for VTK.js"""
		import matplotlib.pyplot as plt
		import matplotlib.cm as cm
		
		# Get matplotlib colormap
		cmap = cm.get_cmap(colormap)
		
		# Create 256 color entries
		num_colors = 256
		colors = []
		
		for i in range(num_colors):
			# Normalize i to [0, 1] range
			t = i / (num_colors - 1)
			# Get color from matplotlib colormap
			color = cmap(t)
			# Convert to RGB (0-1 range)
			colors.append([float(color[0]), float(color[1]), float(color[2]), 1.0])
		
		lut_data = {
			"number_of_colors": num_colors,
			"range": [min_val, max_val],
			"colors": colors,
			"colormap_name": colormap
		}
		
		logger.info(f"Created lookup table with {num_colors} colors, range {min_val:.3f} to {max_val:.3f}")
		return lut_data

# Global visualizer instance
field_visualizer = FieldVisualizer()
