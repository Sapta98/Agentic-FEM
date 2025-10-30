#!/usr/bin/env python3
"""
Template Manager for Physics Simulation Templates
"""

import json
import os
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class TemplateManager:
	"""Manages simulation templates for different physics types"""

	def __init__(self, templates_dir: Optional[Path] = None):
		"""
		Initialize template manager

		Args:
			templates_dir: Directory containing template files
		"""
		if templates_dir is None:
			# Default to templates directory relative to this file
			templates_dir = Path(__file__).parent.parent / "templates"

		self.templates_dir = templates_dir
		self._templates_cache = {}
		self._load_templates()

	def _load_templates(self):
		"""Load all available templates"""
		try:
			template_files = {
				"heat_transfer": "heat_transfer_template.json",
				"solid_mechanics": "solid_mechanics_template.json"
			}

			for physics_type, filename in template_files.items():
				template_path = self.templates_dir / filename
				if template_path.exists():
					with open(template_path, 'r') as f:
						self._templates_cache[physics_type] = json.load(f)
						logger.info(f"Loaded template for {physics_type}")
				else:
					logger.warning(f"Template file not found: {template_path}")

		except Exception as e:
			logger.error(f"Error loading templates: {e}")

	def get_template(self, physics_type: str) -> Optional[Dict[str, Any]]:
		"""
		Get template for specific physics type

		Args:
			physics_type: Type of physics simulation

		Returns:
			Template dictionary or None if not found
		"""
		return self._templates_cache.get(physics_type)

	def get_boundary_condition_options(self, physics_type: str) -> List[Dict[str, Any]]:
		"""
		Get available boundary condition options for physics type

		Args:
			physics_type: Type of physics simulation

		Returns:
			List of boundary condition options
		"""
		template = self.get_template(physics_type)
		if template:
			return template.get("boundary_condition_options", [])
		return []

	def get_initial_condition_options(self, physics_type: str) -> List[Dict[str, Any]]:
		"""
		Get available initial condition options for physics type

		Args:
			physics_type: Type of physics simulation

		Returns:
			List of initial condition options
		"""
		template = self.get_template(physics_type)
		if template:
			return template.get("initial_condition_options", [])
		return []

	def get_geometry_dimension_requirements(self, physics_type: str, geometry_type: str) -> List[str]:
		"""
		Get required dimensions for specific geometry type

		Args:
			physics_type: Type of physics simulation
			geometry_type: Type of geometry (bar, beam, etc.)

		Returns:
			List of required dimension names
		"""
		template = self.get_template(physics_type)
		if template:
			requirements = template.get("geometry_dimension_requirements", {})
			return requirements.get(geometry_type, [])
		return []

	def get_material_property_info(self, physics_type: str) -> Dict[str, Dict[str, Any]]:
		"""
		Get material property information for physics type

		Args:
			physics_type: Type of physics simulation

		Returns:
			Dictionary of material properties with units and descriptions
		"""
		template = self.get_template(physics_type)
		if template:
			return template.get("material_property_sources", {})
		return {}

	def create_simulation_config(self, physics_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Create simulation configuration based on template and context

		Args:
			physics_type: Type of physics simulation
			context: Current context with provided information

		Returns:
			Complete simulation configuration
		"""
		template = self.get_template(physics_type)
		if not template:
			raise ValueError(f"No template found for physics type: {physics_type}")

		# Deep copy the template
		config = copy.deepcopy(template)

		# Update with context information
		if "material_type" in context:
			config["required_components"]["material_type"] = context["material_type"]

		if "geometry_type" in context:
			config["required_components"]["geometry_type"] = context["geometry_type"]

		if "geometry_dimensions" in context:
			config["required_components"]["geometry_dimensions"] = context["geometry_dimensions"]
			# Update mesh dimensions - replace with geometry-specific dimensions
			if isinstance(context["geometry_dimensions"], dict):
				# Clear existing dimensions and set only the relevant ones for this geometry
				config["pde_config"]["mesh_parameters"]["dimensions"] = context["geometry_dimensions"].copy()

		if "boundary_conditions" in context:
			config["required_components"]["boundary_conditions"] = context["boundary_conditions"]
			config["pde_config"]["boundary_conditions"] = context["boundary_conditions"]

		if "external_loads" in context:
			config["required_components"]["external_loads"] = context["external_loads"]
			config["pde_config"]["external_loads"] = context["external_loads"]

		if "initial_conditions" in context:
			config["required_components"]["initial_conditions"] = context["initial_conditions"]
			config["pde_config"]["initial_conditions"] = context["initial_conditions"]

		# Update material properties if provided
		if "material_properties" in context:
			logger.info(f"Template manager updating material properties: {context['material_properties']}")
			config["pde_config"]["material_properties"].update(context["material_properties"])
			logger.info(f"Template manager final material properties: {config['pde_config']['material_properties']}")

		return config

	def check_completeness(self, physics_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Check what information is still missing from the simulation

		Args:
			physics_type: Type of physics simulation
			context: Current context with provided information

		Returns:
			Dictionary with completeness information
		"""
		template = self.get_template(physics_type)
		if not template:
			return {"complete": False, "missing": ["template_not_found"]}

		required = template["required_components"]
		missing = []

		# Check material type
		if not context.get("material_type"):
			missing.append("material_type")

		# Check geometry type
		if not context.get("geometry_type"):
			missing.append("geometry_type")

		# Check geometry dimensions
		geometry_type = context.get("geometry_type")
		if geometry_type:
			required_dims = self.get_geometry_dimension_requirements(physics_type, geometry_type)
			provided_dims = context.get("geometry_dimensions", {})
			for dim in required_dims:
				if dim not in provided_dims or provided_dims[dim] is None:
					missing.append(f"geometry_dimension_{dim}")

		# Check boundary conditions
		if not context.get("boundary_conditions"):
			missing.append("boundary_conditions")

		# Material properties are auto-generated when material type is provided
		# Only check if user provided specific properties that need validation
		material_type = context.get("material_type")
		if material_type and not context.get("material_properties"):
			# Material properties will be auto-generated, don't mark as missing
			pass

		return {
			"complete": len(missing) == 0,
			"missing": missing,
			"missing_count": len(missing)
		}

	def get_available_physics_types(self) -> List[str]:
		"""Get list of available physics types"""
		return list(self._templates_cache.keys())

	def get_template_info(self, physics_type: str) -> Dict[str, Any]:
		"""Get information about a template"""
		template = self.get_template(physics_type)
		if not template:
			return {"error": f"No template found for {physics_type}"}

		return {
			"physics_type": physics_type,
			"description": template.get("description", "No description available"),
			"required_components": list(template.get("required_components", {}).keys()),
			"boundary_condition_options": len(self.get_boundary_condition_options(physics_type)),
			"initial_condition_options": len(self.get_initial_condition_options(physics_type)),
			"material_properties": len(self.get_material_property_info(physics_type))
		}

	def validate_context(self, physics_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate context against template requirements"""
		template = self.get_template(physics_type)
		if not template:
			return {"valid": False, "error": f"No template found for {physics_type}"}

		validation_result = {
			"valid": True,
			"errors": [],
			"warnings": [],
			"suggestions": []
		}

		# Validate material type
		if "material_type" in context:
			available_materials = template.get("material_property_sources", {}).keys()
			if context["material_type"] not in available_materials:
				validation_result["warnings"].append(f"Material type '{context['material_type']}' not in template")

		# Validate geometry type
		if "geometry_type" in context:
			required_dims = self.get_geometry_dimension_requirements(physics_type, context["geometry_type"])
			provided_dims = context.get("geometry_dimensions", {})
			
			for dim in required_dims:
				if dim not in provided_dims:
					validation_result["errors"].append(f"Missing required dimension: {dim}")
				elif provided_dims[dim] is None or provided_dims[dim] <= 0:
					validation_result["errors"].append(f"Invalid dimension value for {dim}")

		# Validate boundary conditions
		if "boundary_conditions" in context:
			bc_options = self.get_boundary_condition_options(physics_type)
			bc_names = [option.get("name", "") for option in bc_options]
			
			for bc in context["boundary_conditions"]:
				if bc.get("type") not in bc_names:
					validation_result["warnings"].append(f"Boundary condition type '{bc.get('type')}' not in template options")

		validation_result["valid"] = len(validation_result["errors"]) == 0
		return validation_result

	def get_simulation_summary(self, physics_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Get a summary of the simulation configuration"""
		template = self.get_template(physics_type)
		if not template:
			return {"error": f"No template found for {physics_type}"}

		completeness = self.check_completeness(physics_type, context)
		validation = self.validate_context(physics_type, context)

		return {
			"physics_type": physics_type,
			"completeness": completeness,
			"validation": validation,
			"geometry_type": context.get("geometry_type", "Not specified"),
			"material_type": context.get("material_type", "Not specified"),
			"has_boundary_conditions": bool(context.get("boundary_conditions")),
			"has_initial_conditions": bool(context.get("initial_conditions")),
			"has_external_loads": bool(context.get("external_loads")),
			"ready_for_simulation": completeness["complete"] and validation["valid"]
		}

	def update_template(self, physics_type: str, updates: Dict[str, Any]) -> bool:
		"""Update a template with new information"""
		try:
			if physics_type not in self._templates_cache:
				return False

			# Update the cached template
			self._templates_cache[physics_type].update(updates)

			# Save to file
			template_path = self.templates_dir / f"{physics_type}_template.json"
			with open(template_path, 'w') as f:
				json.dump(self._templates_cache[physics_type], f, indent=2)

			logger.info(f"Updated template for {physics_type}")
			return True

		except Exception as e:
			logger.error(f"Error updating template: {e}")
			return False

	def reload_templates(self) -> bool:
		"""Reload all templates from files"""
		try:
			self._templates_cache.clear()
			self._load_templates()
			logger.info("Templates reloaded successfully")
			return True
		except Exception as e:
			logger.error(f"Error reloading templates: {e}")
			return False