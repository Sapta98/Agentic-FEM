"""
Configuration Panels Component
==============================

Interactive configuration panels for editing simulation parameters.
Handles geometry, material, physics, and boundary condition inputs.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigPanels:
	"""Configuration panels component"""

	def __init__(self, config_manager=None):
		"""
		Initialize configuration panels

		Args:
		config_manager: Configuration manager instance
		"""
		self.config_manager = config_manager
		self.current_context = {}

		# Panel settings
		self.settings = {
			'show_geometry_panel': True,
			'show_material_panel': True,
			'show_physics_panel': True,
			'show_boundary_conditions_panel': True,
			'auto_update_mesh': True,
			'validation_enabled': True
		}

	def generate_config_html(self, context: Dict[str, Any]) -> str:
		"""
		Generate HTML for configuration panels

		Args:
		context: Current simulation context

		Returns:
		HTML content for configuration panels
		"""
		self.current_context = context

		html_parts = []

		# Geometry panel
		if self.settings['show_geometry_panel']:
			html_parts.append(self._generate_geometry_panel())

		# Material panel
		if self.settings['show_material_panel']:
			html_parts.append(self._generate_material_panel())

		# Physics panel
		if self.settings['show_physics_panel']:
			html_parts.append(self._generate_physics_panel())

		# Boundary conditions panel
		if self.settings['show_boundary_conditions_panel']:
			html_parts.append(self._generate_boundary_conditions_panel())

		return '\n'.join(html_parts)

	def _generate_geometry_panel(self) -> str:
		"""Generate geometry configuration panel HTML"""
		geometry_type = self.current_context.get('geometry_type', '')
		dimensions = self.current_context.get('geometry_dimensions', {})

		# Get dimension fields for current geometry type
		dimension_fields = self._get_dimension_fields_for_geometry(geometry_type)

		html = f"""
		<div class="config-section geometry-section">
		<h3>Geometry Configuration</h3>
		<div class="config-group">
		<label for="geometry-type">Geometry Type:</label>
		<select id="geometry-type" onchange="updateConfig('geometry_type', this.value)">
		<option value="">Select geometry...</option>
		<option value="line" {'selected' if geometry_type == 'line' else ''}>Line (1D)</option>
		<option value="plate" {'selected' if geometry_type == 'plate' else ''}>Plate (2D)</option>
		<option value="rectangle" {'selected' if geometry_type == 'rectangle' else ''}>Rectangle (2D)</option>
		<option value="cube" {'selected' if geometry_type == 'cube' else ''}>Cube (3D)</option>
		<option value="box" {'selected' if geometry_type == 'box' else ''}>Box (3D)</option>
		<option value="cylinder" {'selected' if geometry_type == 'cylinder' else ''}>Cylinder (3D)</option>
		<option value="sphere" {'selected' if geometry_type == 'sphere' else ''}>Sphere (3D)</option>
		</select>
		</div>
		"""

		# Add dimension fields
		for field in dimension_fields:
			field_name = field['name']
			field_label = field['label']
			field_unit = field['unit']
			field_value = dimensions.get(field_name, '')

			html += f"""
			<div class="config-group">
			<label for="dim-{field_name}">{field_label} ({field_unit}):</label>
			<input type="number"
			id="dim-{field_name}"
			step="any"
			placeholder="Enter {field_label.lower()}"
			value="{field_value}"
			onchange="updateConfig('geometry_dimensions.{field_name}', this.value)">
			</div>
			"""

		html += "</div>"
		return html

	def _generate_material_panel(self) -> str:
		"""Generate material configuration panel HTML"""
		material_type = self.current_context.get('material_type', '')
		material_properties = self.current_context.get('material_properties', {})
		physics_type = self.current_context.get('physics_type', '')

		# Get material fields for current material and physics type
		material_fields = self._get_material_fields_for_material(material_type, physics_type)

		html = f"""
		<div class="config-section material-section">
		<h3>Material Configuration</h3>
		<div class="config-group">
		<label for="material-type">Material Type:</label>
		<select id="material-type" onchange="updateConfig('material_type', this.value)">
		<option value="">Select material...</option>
		<option value="aluminum" {'selected' if material_type == 'aluminum' else ''}>Aluminum</option>
		<option value="steel" {'selected' if material_type == 'steel' else ''}>Steel</option>
		<option value="copper" {'selected' if material_type == 'copper' else ''}>Copper</option>
		<option value="concrete" {'selected' if material_type == 'concrete' else ''}>Concrete</option>
		<option value="custom" {'selected' if material_type == 'custom' else ''}>Custom</option>
		</select>
		</div>
		"""

		# Add material property fields
		for field in material_fields:
			field_name = field['name']
			field_label = field['label']
			field_unit = field['unit']
			field_value = material_properties.get(field_name, field.get('default', ''))

			html += f"""
			<div class="config-group">
			<label for="mat-{field_name}">{field_label} ({field_unit}):</label>
			<input type="number"
			id="mat-{field_name}"
			step="any"
			placeholder="Enter {field_label.lower()}"
			value="{field_value}"
			onchange="updateConfig('material_properties.{field_name}', this.value)">
			</div>
			"""

		html += "</div>"
		return html

	def _generate_physics_panel(self) -> str:
		"""Generate physics configuration panel HTML"""
		physics_type = self.current_context.get('physics_type', '')

		html = f"""
		<div class="config-section physics-section">
		<h3>Physics Configuration</h3>
		<div class="config-group">
		<label for="physics-type">Physics Type:</label>
		<select id="physics-type" onchange="updateConfig('physics_type', this.value)">
		<option value="">Select physics...</option>
		<option value="heat_transfer" {'selected' if physics_type == 'heat_transfer' else ''}>Heat Transfer</option>
		<option value="solid_mechanics" {'selected' if physics_type == 'solid_mechanics' else ''}>Solid Mechanics</option>
		</select>
		</div>
		"""

		# Add physics-specific settings
		if physics_type == 'heat_transfer':
			html += """
			<div class="config-group">
			<label for="heat-analysis-type">Analysis Type:</label>
			<select id="heat-analysis-type" onchange="updateConfig('physics.analysis_type', this.value)">
			<option value="steady_state">Steady State</option>
			<option value="transient">Transient</option>
			</select>
			</div>
			"""

		elif physics_type == 'solid_mechanics':
			html += """
			<div class="config-group">
			<label for="mech-analysis-type">Analysis Type:</label>
			<select id="mech-analysis-type" onchange="updateConfig('physics.analysis_type', this.value)">
			<option value="static">Static</option>
			<option value="dynamic">Dynamic</option>
			</select>
			</div>
			"""

		html += "</div>"
		return html

	def _generate_boundary_conditions_panel(self) -> str:
		"""Generate boundary conditions configuration panel HTML"""
		boundary_conditions = self.current_context.get('boundary_conditions', [])

		html = """
		<div class="config-section boundary-conditions-section">
		<h3>Boundary Conditions</h3>
		<div class="boundary-conditions-container">
		"""

		# Add existing boundary conditions
		for i, bc in enumerate(boundary_conditions):
			bc_type = bc.get('type', '')
			bc_value = bc.get('value', '')
			bc_location = bc.get('location', '')

			html += f"""
			<div class="boundary-condition-item" data-index="{i}">
			<div class="config-group">
			<label>Location:</label>
			<input type="text" value="{bc_location}" onchange="updateBoundaryCondition({i}, 'location', this.value)">
			</div>
			<div class="config-group">
			<label>Type:</label>
			<select onchange="updateBoundaryCondition({i}, 'type', this.value)">
			<option value="dirichlet" {'selected' if bc_type == 'dirichlet' else ''}>Dirichlet</option>
			<option value="neumann" {'selected' if bc_type == 'neumann' else ''}>Neumann</option>
			</select>
			</div>
			<div class="config-group">
			<label>Value:</label>
			<input type="number" step="any" value="{bc_value}" onchange="updateBoundaryCondition({i}, 'value', this.value)">
			</div>
			<button type="button" onclick="removeBoundaryCondition({i})" class="btn btn-danger">Remove</button>
			</div>
			"""

		# Add button to add new boundary condition
		html += """
		<button type="button" onclick="addBoundaryCondition()" class="btn btn-primary">Add Boundary Condition</button>
		</div>
		</div>
		"""

		return html

	def _get_dimension_fields_for_geometry(self, geometry_type: str) -> List[Dict[str, Any]]:
		"""Get dimension fields for specific geometry type"""
		dimension_fields = {
			'line': [
				{'name': 'length', 'label': 'Length', 'unit': 'm'}
			],
			'plate': [
				{'name': 'length', 'label': 'Length', 'unit': 'm'},
				{'name': 'width', 'label': 'Width', 'unit': 'm'}
			],
			'rectangle': [
				{'name': 'length', 'label': 'Length', 'unit': 'm'},
				{'name': 'width', 'label': 'Width', 'unit': 'm'}
			],
			'cube': [
				{'name': 'length', 'label': 'Length', 'unit': 'm'},
				{'name': 'width', 'label': 'Width', 'unit': 'm'},
				{'name': 'height', 'label': 'Height', 'unit': 'm'}
			],
			'box': [
				{'name': 'length', 'label': 'Length', 'unit': 'm'},
				{'name': 'width', 'label': 'Width', 'unit': 'm'},
				{'name': 'height', 'label': 'Height', 'unit': 'm'}
			],
			'cylinder': [
				{'name': 'radius', 'label': 'Radius', 'unit': 'm'},
				{'name': 'height', 'label': 'Height', 'unit': 'm'}
			],
			'sphere': [
				{'name': 'radius', 'label': 'Radius', 'unit': 'm'}
			]
		}

		return dimension_fields.get(geometry_type, [])

	def _get_material_fields_for_material(self, material_type: str, physics_type: str) -> List[Dict[str, Any]]:
		"""Get material fields for specific material and physics type"""
		# Common material properties
		common_fields = [
			{'name': 'density', 'label': 'Density', 'unit': 'kg/m³', 'default': '7850'}
		]

		# Heat transfer specific properties
		heat_fields = [
			{'name': 'thermal_conductivity', 'label': 'Thermal Conductivity', 'unit': 'W/(m·K)', 'default': '50'},
			{'name': 'specific_heat', 'label': 'Specific Heat', 'unit': 'J/(kg·K)', 'default': '460'},
			{'name': 'thermal_expansion', 'label': 'Thermal Expansion', 'unit': '1/K', 'default': '12e-6'}
		]

		# Solid mechanics specific properties
		mechanics_fields = [
			{'name': 'young_modulus', 'label': 'Young\'s Modulus', 'unit': 'Pa', 'default': '200e9'},
			{'name': 'poisson_ratio', 'label': 'Poisson\'s Ratio', 'unit': '', 'default': '0.3'},
			{'name': 'yield_strength', 'label': 'Yield Strength', 'unit': 'Pa', 'default': '250e6'}
		]

		# Combine fields based on physics type
		if physics_type == 'heat_transfer':
			return common_fields + heat_fields
		elif physics_type == 'solid_mechanics':
			return common_fields + mechanics_fields
		else:
			return common_fields

	def update_settings(self, new_settings: Dict[str, Any]) -> None:
		"""
		Update panel settings

		Args:
		new_settings: New settings to apply
		"""
		self.settings.update(new_settings)
		logger.info(f"Updated config panels settings: {new_settings}")

	def get_panel_config(self, panel_name: str) -> Dict[str, Any]:
		"""
		Get configuration for specific panel

		Args:
		panel_name: Name of the panel

		Returns:
		Panel configuration dictionary
		"""
		panel_configs = {
			'geometry': {
				'available_types': ['line', 'plate', 'rectangle', 'cube', 'box', 'cylinder', 'sphere'],
				'default_type': 'line'
			},
			'material': {
				'available_types': ['aluminum', 'steel', 'copper', 'concrete', 'custom'],
				'default_type': 'steel'
			},
			'physics': {
				'available_types': ['heat_transfer', 'solid_mechanics'],
				'default_type': 'heat_transfer'
			},
			'boundary_conditions': {
				'available_types': ['dirichlet', 'neumann'],
				'max_conditions': 10
			}
		}

		return panel_configs.get(panel_name, {})