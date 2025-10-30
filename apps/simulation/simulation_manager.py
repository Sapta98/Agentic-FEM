"""
Simulation Manager
Handles complete simulation workflow including materials, boundary conditions, and PDE solving
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SimulationManager:
	"""Manages complete simulation workflow"""

	def __init__(self, mesh_viewer=None):
		# Add paths for imports
		project_root = Path(__file__).parent.parent.parent
		paths_to_add = [
			project_root / "main_app",
			project_root / "nlp_parser" / "src",
			project_root / 'fenics_backend',
			project_root / 'apps',
			project_root / 'frontend'
		]
		
		for path in paths_to_add:
			if str(path) not in sys.path:
				sys.path.insert(0, str(path))

		from prompt_analyzer import SimulationPromptParser
		from local_fenics_solver import FEniCSSolver
		from local_field_visualizer import FieldVisualizer
		from mesh_viewer.mesh_viewer import MeshViewer
		from visualizers.mesh_visualizer import MeshVisualizer

		self.parser = SimulationPromptParser()
		self.fenics_solver = FEniCSSolver()
		self.field_visualizer = FieldVisualizer()
		# Use provided mesh_viewer or create new one
		self.mesh_viewer = mesh_viewer if mesh_viewer is not None else MeshViewer()
		self.mesh_visualizer = MeshVisualizer()
		
		# Central storage for simulation data
		self.current_context = {}
		self.current_mesh_data = None
		self.current_gmsh_model = None  # Store GMSH model separately
		self.current_simulation_config = {}

	def clear_all_context(self) -> Dict[str, Any]:
		"""Clear all simulation context including GMSH models and cached mesh data"""
		cleanup_info = {
			'context_cleared': False,
			'gmsh_models_cleared': 0,
			'mesh_data_cleared': False,
			'visualizer_cache_cleared': False
		}
		
		try:
			# Clear basic context
			self.current_context = {}
			self.current_mesh_data = None
			self.current_gmsh_model = None
			self.current_simulation_config = {}
			cleanup_info['context_cleared'] = True
			
			# Clear GMSH models from mesh generators
			if hasattr(self.mesh_viewer, 'mesh_generator') and hasattr(self.mesh_viewer.mesh_generator, 'generators'):
				for mesh_dim, generator in self.mesh_viewer.mesh_generator.generators.items():
					if hasattr(generator, 'gmsh_generator'):
						# Force clear GMSH model regardless of initialization state
						generator.gmsh_generator.force_clear_gmsh()
						if generator.gmsh_generator.is_gmsh_initialized():
							generator.gmsh_generator.cleanup_gmsh()
						cleanup_info['gmsh_models_cleared'] += 1
					logger.debug(f"Cleared GMSH model for {mesh_dim}D generator")
			
			# Clear cached mesh data in visualizers
			if hasattr(self.mesh_viewer, 'mesh_visualizer'):
				self.mesh_viewer.mesh_visualizer.current_mesh = None
				self.mesh_viewer.mesh_visualizer.current_field = None
				cleanup_info['visualizer_cache_cleared'] = True
				logger.debug("Cleared cached mesh data in mesh visualizer")
			
			# Clear field visualizer cache
			if hasattr(self.field_visualizer, 'current_mesh'):
				self.field_visualizer.current_mesh = None
			if hasattr(self.field_visualizer, 'current_field'):
				self.field_visualizer.current_field = None
			
			logger.debug(f"Cleared all simulation context: {cleanup_info}")
			return cleanup_info
			
		except Exception as e:
			logger.error(f"Error clearing simulation context: {e}")
			cleanup_info['error'] = str(e)
			return cleanup_info

	def store_mesh_data(self, mesh_data: Dict[str, Any]) -> None:
		"""Store mesh data in simulation manager, getting GMSH model from mesh generator"""
		import numpy as np
		
		# Get GMSH model from mesh generator if available
		try:
			if hasattr(self.mesh_viewer, 'mesh_generator') and hasattr(self.mesh_viewer.mesh_generator, 'generators'):
				# Try to get GMSH model from the appropriate generator
				mesh_dim = mesh_data.get('mesh_dimension', 3)
				if mesh_dim in self.mesh_viewer.mesh_generator.generators:
					generator = self.mesh_viewer.mesh_generator.generators[mesh_dim]
					if hasattr(generator, 'gmsh_generator') and hasattr(generator.gmsh_generator, 'gmsh_model'):
						self.current_gmsh_model = generator.gmsh_generator.gmsh_model
						logger.debug(f"Stored GMSH model from generator: {type(self.current_gmsh_model)}")
					else:
						self.current_gmsh_model = None
						logger.warning("GMSH model not available from generator")
				else:
					self.current_gmsh_model = None
					logger.warning(f"No generator available for mesh dimension {mesh_dim}")
			else:
				self.current_gmsh_model = None
				logger.warning("Mesh generator not available")
		except Exception as e:
			logger.warning(f"Failed to get GMSH model from generator: {e}")
			self.current_gmsh_model = None
		
		# Add flag to indicate GMSH model availability
		mesh_data['gmsh_model_was_available'] = self.current_gmsh_model is not None
		
		# Convert numpy arrays to lists for JSON serialization
		mesh_data = self._make_json_safe(mesh_data)
		
		self.current_mesh_data = mesh_data
		logger.debug(f"Stored mesh data with keys: {list(mesh_data.keys())}")

	def _make_json_safe(self, obj):
		"""Recursively convert objects to JSON-safe format"""
		import numpy as np
		
		if isinstance(obj, dict):
			return {key: self._make_json_safe(value) for key, value in obj.items()}
		elif isinstance(obj, list):
			return [self._make_json_safe(item) for item in obj]
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, (np.integer, np.floating)):
			return obj.item()
		else:
			return obj

	def get_current_context(self) -> Dict[str, Any]:
		"""Get current simulation context"""
		return self.current_context

	def get_current_mesh_data(self) -> Optional[Dict[str, Any]]:
		"""Get current mesh data"""
		return self.current_mesh_data

	def get_current_gmsh_model(self):
		"""Get current GMSH model"""
		return self.current_gmsh_model


	def get_current_simulation_config(self) -> Dict[str, Any]:
		"""Get current simulation config"""
		return self.current_simulation_config

	def update_pde_config_from_context(self, context: Optional[Dict[str, Any]] = None) -> bool:
		"""
		Update pde_config based on current context changes
		
		Args:
			context: Updated context (if None, uses current_context)
			
		Returns:
			True if update was successful
		"""
		try:
			# Use provided context or current context
			update_context = context or self.current_context
			
			if not update_context:
				logger.warning("No context available for pde_config update")
				return False
			
			# Ensure simulation_config exists
			if "simulation_config" not in self.current_simulation_config:
				self.current_simulation_config["simulation_config"] = {}
			
			simulation_config = self.current_simulation_config["simulation_config"]
			
			# Ensure pde_config exists
			if "pde_config" not in simulation_config:
				simulation_config["pde_config"] = {}
			
			pde_config = simulation_config["pde_config"]
			
			# Update physics type if available
			if update_context.get("physics_type"):
				pde_config["physics_type"] = update_context["physics_type"]
				logger.info(f"Updated pde_config physics_type: {update_context['physics_type']}")
			
			# Update material properties if available
			if update_context.get("material_properties"):
				pde_config["material_properties"] = update_context["material_properties"]
				logger.info(f"Updated pde_config material_properties")
			
			# Update boundary conditions if available
			if update_context.get("boundary_conditions"):
				# Map boundary locations first
				mapped_bcs = self._map_boundary_locations(
					update_context["boundary_conditions"], 
					update_context.get("geometry_type", "")
				)
				
				# Add default boundary conditions
				processed_bcs = self._add_default_boundary_conditions(
					mapped_bcs, 
					update_context.get("geometry_type", ""),
					update_context.get("physics_type", "")
				)
				
				pde_config["boundary_conditions"] = processed_bcs
				logger.info(f"Updated pde_config boundary_conditions: {len(processed_bcs)} conditions")
			
			# Update geometry dimensions if available
			if update_context.get("geometry_dimensions"):
				if "mesh_parameters" not in pde_config:
					pde_config["mesh_parameters"] = {}
				pde_config["mesh_parameters"]["dimensions"] = update_context["geometry_dimensions"]
				logger.info(f"Updated pde_config mesh_parameters dimensions")
			
			logger.info("Successfully updated pde_config from context")
			return True
			
		except Exception as e:
			logger.error(f"Error updating pde_config from context: {e}")
			return False

	def parse_simulation_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Parse simulation prompt using NLP parser"""
		try:
			result = self.parser.parse(prompt, context or {})
			
			# Store the parsed context in simulation manager
			if result.get("updated_context"):
				self.current_context = result["updated_context"]
				logger.debug(f"Stored context with keys: {list(self.current_context.keys())}")
				
				# Apply boundary location mapping to the context immediately
				if self.current_context.get("boundary_conditions"):
					logger.debug("Applying boundary location mapping to context")
					mapped_bcs = self._map_boundary_locations(
						self.current_context["boundary_conditions"], 
						self.current_context.get("geometry_type", "")
					)
					self.current_context["boundary_conditions"] = mapped_bcs
					logger.debug(f"Applied boundary mapping: {len(mapped_bcs)} boundary conditions")
			
			if result.get("simulation_config"):
				self.current_simulation_config = result["simulation_config"]
				logger.debug(f"Stored simulation config with keys: {list(self.current_simulation_config.keys())}")
			
			return {
				"success": True,
				"result": {
					**result,
					"updated_context": self.current_context  # Include the mapped boundary conditions
				}
			}
		except Exception as e:
			logger.error(f"Error parsing simulation prompt: {e}")
			return {
				"success": False,
				"error": str(e)
			}

	def solve_pde(self, config: Dict[str, Any], mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Solve PDE using FEniCS solver with GMSH model"""
		try:
			logger.info("Solving PDE")
			
			# Pass GMSH model to FEniCS solver if available
			if self.current_gmsh_model is not None:
				logger.debug("Passing GMSH model to FEniCS solver")
				# Set GMSH model in the solver using the dedicated method
				self.fenics_solver.set_gmsh_model(self.current_gmsh_model)
			else:
				logger.warning("No GMSH model available for FEniCS solver")
				self.fenics_solver.set_gmsh_model(None)
			
			solution_result = self.fenics_solver.solve_simulation(config, mesh_data)

			if solution_result.get("success"):
				return solution_result  # Return the flat structure directly
			else:
				return {
					"success": False,
					"error": solution_result.get("error", "PDE solving failed")
				}
		except Exception as e:
			logger.error(f"Error solving PDE: {e}")
			return {
				"success": False,
				"error": str(e)
			}

	def create_field_visualization(self, solution_data: Dict[str, Any], mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Create field visualization from solution data"""
		try:
			logger.debug("Creating field visualization")
			field_visualization_url = self.field_visualizer.create_field_visualization(solution_data, mesh_data)

			return {
				"success": True,
				"field_visualization_url": field_visualization_url
			}
		except Exception as e:
			logger.error(f"Error creating field visualization: {e}")
			return {
				"success": False,
				"error": str(e)
			}

	def run_complete_simulation(self, prompt: str = None, context: Optional[Dict[str, Any]] = None, existing_mesh_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Run complete simulation workflow"""
		try:
			# Step 1: Use stored context or parse if needed
			if self.current_context and self._is_context_complete(self.current_context):
				logger.debug("Using stored complete context")
				result = {
					"action": "simulation_ready",
					"simulation_config": self.current_simulation_config,
					"context": self.current_context,
					"message": "Using stored complete context"
				}
			elif context and self._is_context_complete(context):
				logger.debug("Using provided complete context")
				result = {
					"action": "simulation_ready",
					"simulation_config": context.get("simulation_config", {}),
					"context": context,
					"message": "Using provided complete context"
				}
			else:
				logger.debug("Parsing prompt for new simulation")
				parse_result = self.parse_simulation_prompt(prompt, context)
				if not parse_result["success"]:
					return parse_result
				result = parse_result["result"]

				if result.get("action") != "simulation_ready":
					return {
						"success": True,
						"action": result.get("action", "unknown"),
						"message": result.get("message"),
						"simulation_config": result.get("simulation_config"),
						"context": result.get("updated_context"),
						"guidance": result.get("guidance")
					}

			# Step 2: Use stored mesh data or provided mesh data
			if self.current_mesh_data:
				logger.debug("Using stored mesh data for PDE solving")
				mesh_data = self.current_mesh_data
			elif existing_mesh_data:
				logger.debug("Using provided mesh data for PDE solving")
				mesh_data = existing_mesh_data
			else:
				logger.warning("No mesh data available for simulation")
				return {
					"success": False,
					"error": "No mesh data available. Please generate a mesh preview first, then try solving the PDE."
				}
			
			logger.debug(f"Mesh data keys: {list(mesh_data.keys())}")
			logger.debug(f"Mesh dimension: {mesh_data.get('mesh_dimension', 'unknown')}")
			logger.debug(f"Mesh vertices count: {len(mesh_data.get('vertices', []))}")
			
			mesh_result = {
				"success": True,
				"mesh_data": mesh_data,
				"mesh_visualization_url": None,  # Will be created by field visualization
				"geometry_type": mesh_data.get("geometry_type", "unknown"),
				"dimensions": mesh_data.get("dimensions", {})
			}

			# Step 3: Solve PDE
			# Merge context data (material properties, boundary conditions) into simulation config
			simulation_config = result["simulation_config"].copy()
			context = result.get("context", {})

			# Add material properties from context to PDE config
			if "pde_config" not in simulation_config:
				simulation_config["pde_config"] = {}
			if context.get("material_properties"):
				logger.debug(f"Adding material properties to PDE config: {context['material_properties']}")
				simulation_config["pde_config"]["material_properties"] = context["material_properties"]
			else:
				logger.warning("No material properties found in context")
			if context.get("boundary_conditions"):
				# Debug: Log original boundary conditions before mapping
				# reduce boundary condition logs to debug
				for i, bc in enumerate(context["boundary_conditions"]):
					logger.debug(f"Original BC {i+1}: {bc}")
				
				# First map vague locations to specific boundaries
				mapped_bcs = self._map_boundary_locations(
					context["boundary_conditions"], 
					context.get("geometry_type", "")
				)
				
				# Debug: Log mapped boundary conditions
				for i, bc in enumerate(mapped_bcs):
					logger.debug(f"Mapped BC {i+1}: {bc}")
				# Then add defaults for unspecified boundaries
				processed_bcs = self._add_default_boundary_conditions(
					mapped_bcs, 
					context.get("geometry_type", ""),
					context.get("physics_type", "")
				)
				# Note: Do not coerce heat-transfer BCs here; rely on parser output
				# Note: Do not coerce solid mechanics BCs here; rely on parser output
				simulation_config["pde_config"]["boundary_conditions"] = processed_bcs
			if context.get("physics_type"):
				simulation_config["pde_config"]["physics_type"] = context["physics_type"]

			# Log the complete PDE configuration for debugging
			# compress final PDE log output
			pde_config = simulation_config.get("pde_config", {})
			
			# Physics type
			physics_type = pde_config.get("physics_type", "NOT SET")
			logger.debug(f"Physics Type: {physics_type}")
			
			# Material properties
			material_props = pde_config.get("material_properties", {})
			if material_props:
				for prop, value in material_props.items():
					logger.debug(f"Material property {prop}={value}")
			else:
				logger.info("Material Properties: NOT SET")
			
			# Boundary conditions
			boundary_conditions = pde_config.get("boundary_conditions", [])
			if boundary_conditions:
				for i, bc in enumerate(boundary_conditions):
					logger.debug(f"BC {i+1}: {bc}")
			else:
				logger.info("Boundary Conditions: NOT SET")
			
			# Additional PDE config fields
			other_fields = {k: v for k, v in pde_config.items() if k not in ["physics_type", "material_properties", "boundary_conditions"]}
			if other_fields:
				for field, value in other_fields.items():
					logger.debug(f"PDE field {field}={value}")
			
			# end reduced PDE logs

			pde_result = self.solve_pde(simulation_config, mesh_result["mesh_data"])
			if not pde_result["success"]:
				return {
					"success": True,
					"action": "simulation_ready",
					"message": f"Mesh generated but PDE solving failed: {pde_result['error']}",
					"simulation_config": result["simulation_config"],
					"context": result.get("updated_context"),
					"mesh_visualization_url": mesh_result["mesh_visualization_url"]
				}

			# Step 4: Create field visualization using GMSH mesh data from FEniCS solver
			logger.debug("Creating field visualization with GMSH mesh data")
			solution_data = pde_result  # Solution data is now at the top level with GMSH structure
			
			# Use GMSH mesh data directly from FEniCS solver (no complex extraction needed)
			mesh_data_for_visualization = mesh_result["mesh_data"]  # Original GMSH mesh data
			
			logger.debug(f"Using GMSH mesh data: {len(mesh_data_for_visualization.get('vertices', []))} vertices")
			logger.debug(f"Solution data: {len(solution_data.get('coordinates', []))} coords, {len(solution_data.get('values', []))} values")
			
			# Create field visualization (includes mesh + field data with toggle option)
			logger.debug("Creating field visualization...")
			field_visualization_url = self.field_visualizer.create_field_visualization(
				solution_data,  # GMSH-structured solution data
				mesh_data_for_visualization  # Original GMSH mesh data
			)
			logger.info(f"Field visualization ready: {field_visualization_url}")
			
			# Create mesh visualization (mesh preview only)
			logger.debug("Creating mesh visualization...")
			mesh_visualization_url = self.mesh_visualizer.create_mesh_visualization(
				mesh_data_for_visualization
			)
			logger.info(f"Mesh visualization ready: {mesh_visualization_url}")

			return {
				"success": True,
				"action": "pde_solved",
				"message": "Simulation completed successfully",
				"simulation_config": result["simulation_config"],
				"context": result.get("updated_context"),
				"mesh_visualization_url": mesh_visualization_url,  # Mesh preview only
				"field_visualization_url": field_visualization_url,  # Field visualization with mesh + field
				"solution_data": pde_result  # GMSH-structured solution data
			}

		except Exception as e:
			logger.error(f"Complete simulation workflow failed: {e}")
			return {
				"success": False,
				"error": str(e)
			}

	def _is_context_complete(self, context: Dict[str, Any]) -> bool:
		"""Check if context has all required information for simulation"""
		required_fields = [
			'physics_type',
			'material_type', 
			'geometry_type',
			'geometry_dimensions',
			'boundary_conditions'
		]
		
		logger.debug(f"Checking context completeness. Available keys: {list(context.keys())}")
		
		for field in required_fields:
			value = context.get(field)
			logger.debug(f"Checking field '{field}': present={field in context}, truthy={bool(value)}")
			if not value:
				logger.info(f"Context incomplete - missing or falsy: {field}")
				return False
		
		# Check if boundary conditions is not empty
		boundary_conditions = context.get('boundary_conditions', [])
		logger.debug(f"Final boundary_conditions length={len(boundary_conditions) if isinstance(boundary_conditions, (list, dict)) else 'N/A'}")
		if not boundary_conditions or len(boundary_conditions) == 0:
			logger.info("Context incomplete - boundary_conditions is empty")
			return False
			
		logger.debug("Context is complete")
		return True

	def _map_boundary_locations(self, bcs: list, geometry_type: str) -> list:
		"""Map vague boundary locations to specific boundaries"""
		if not bcs:
			return []
		
		# Get available boundaries for this geometry
		available_boundaries = self._get_available_boundaries(geometry_type)
		
		# Geometry-specific location mappings
		geometry_specific_mappings = {
			'cylinder': {
				'one end': 'top', 'other end': 'bottom', 'one side': 'top', 'other side': 'bottom',
				'left': 'top', 'right': 'bottom', 'start': 'top', 'end': 'bottom',
				'top': 'top', 'bottom': 'bottom', 'upper': 'top', 'lower': 'bottom',
				'circumference': 'curved surface', 'surface': 'curved surface'
			},
			'disc': {
				'circumference': 'circumference', 'edge': 'circumference', 'perimeter': 'circumference',
				'center': 'center', 'middle': 'center', 'origin': 'center',
				'one side': 'circumference', 'other side': 'center',
				'outer': 'circumference', 'inner': 'center'
			},
			'rod': {
				'one end': 'left', 'other end': 'right', 'one side': 'left', 'other side': 'right',
				'left': 'left', 'right': 'right', 'start': 'left', 'end': 'right',
				'top': 'left', 'bottom': 'right', 'upper': 'left', 'lower': 'right'
			},
			'line': {
				'one end': 'left', 'other end': 'right', 'one side': 'left', 'other side': 'right',
				'left': 'left', 'right': 'right', 'start': 'left', 'end': 'right',
				'top': 'left', 'bottom': 'right', 'upper': 'left', 'lower': 'right'
			}
		}
		
		# Generic location mappings (fallback)
		location_mappings = {
			'one side': 'left', 'left side': 'left', 'start': 'left',
			'opposite side': 'right', 'other side': 'right', 'opposite end': 'right',
			'right side': 'right', 'end': 'right',
			'top': 'top', 'upper': 'top', 'top surface': 'top',
			'bottom': 'bottom', 'lower': 'bottom', 'bottom surface': 'bottom',
			'front': 'front', 'back': 'back', 'side': 'side',
			'circumference': 'circumference', 'surface': 'surface',
			'curved surface': 'curved surface', 'all': 'all', 'all_boundary': 'all_boundary'
		}
		
		mapped_bcs = []
		for bc in bcs:
			mapped_bc = bc.copy()
			location = bc.get('location', '').lower().strip()
			
			# Try geometry-specific mapping first
			mapped_location = None
			if geometry_type in geometry_specific_mappings:
				if location in geometry_specific_mappings[geometry_type]:
					mapped_location = geometry_specific_mappings[geometry_type][location]
				logger.debug(f"Geometry-specific mapping for {geometry_type}: '{location}' → '{mapped_location}'")
			
			# Fallback to generic mapping
			if not mapped_location and location in location_mappings:
				mapped_location = location_mappings[location]
				logger.debug(f"Generic mapping: '{location}' → '{mapped_location}'")
			
			# Validate mapped location is available for this geometry
			if mapped_location and (mapped_location in available_boundaries or mapped_location in ['all', 'all_boundary']):
				mapped_bc['location'] = mapped_location
				mapped_bc['mapped_location'] = mapped_location
			else:
				# Try confidence-based prediction based on temperature value
				value = bc.get('value', 0)
				if isinstance(value, (int, float)):
					if value > 50:  # High temperature
						if 'left' in available_boundaries:
							mapped_bc['location'] = 'left'
							logger.debug(f"High temperature mapping: '{location}' (value: {value}) → 'left'")
						elif 'top' in available_boundaries:
							mapped_bc['location'] = 'top'
							logger.debug(f"High temperature mapping: '{location}' (value: {value}) → 'top'")
					elif value < 50:  # Low temperature
						if 'right' in available_boundaries:
							mapped_bc['location'] = 'right'
							logger.debug(f"Low temperature mapping: '{location}' (value: {value}) → 'right'")
						elif 'bottom' in available_boundaries:
							mapped_bc['location'] = 'bottom'
							logger.debug(f"Low temperature mapping: '{location}' (value: {value}) → 'bottom'")
				else:
					# Fallback to keyword-based mapping
					if 'hot' in location or 'high' in location:
						if 'left' in available_boundaries:
							mapped_bc['location'] = 'left'
							logger.debug(f"Keyword-based mapping: '{location}' → 'left'")
						elif 'top' in available_boundaries:
							mapped_bc['location'] = 'top'
							logger.debug(f"Keyword-based mapping: '{location}' → 'top'")
					elif 'cold' in location or 'low' in location:
						if 'right' in available_boundaries:
							mapped_bc['location'] = 'right'
							logger.debug(f"Keyword-based mapping: '{location}' → 'right'")
						elif 'bottom' in available_boundaries:
							mapped_bc['location'] = 'bottom'
							logger.debug(f"Keyword-based mapping: '{location}' → 'bottom'")
					else:
						# Special handling for disc geometry
						if geometry_type == 'disc' and len(bcs) == 2:
							# For disc with 2 BCs, use value-based mapping
							value = bc.get('value', 0)
							if isinstance(value, (int, float)):
								if value > 50:  # High temperature/force
									mapped_bc['location'] = 'circumference'
									logger.debug(f"Disc high value mapping: '{location}' (value: {value}) → 'circumference'")
								else:  # Low temperature/force
									mapped_bc['location'] = 'center'
									logger.debug(f"Disc low value mapping: '{location}' (value: {value}) → 'center'")
							else:
								# Default for disc: alternate between circumference and center
								circumference_count = sum(1 for bc in bcs if bc.get('location') == 'circumference')
								if circumference_count == 0:
									mapped_bc['location'] = 'circumference'
									logger.debug(f"Disc default mapping: '{location}' → 'circumference'")
								else:
									mapped_bc['location'] = 'center'
									logger.debug(f"Disc default mapping: '{location}' → 'center'")
						else:
							# Default: use first available boundary
							mapped_bc['location'] = available_boundaries[0] if available_boundaries else 'left'
							logger.debug(f"Default mapping: '{location}' → '{mapped_bc['location']}'")
			
			mapped_bcs.append(mapped_bc)
		
		return mapped_bcs

	def _add_default_boundary_conditions(self, bcs: list, geometry_type: str, physics_type: str) -> list:
		"""Add default boundary conditions for unspecified boundaries based on geometry"""
		if not bcs:
			return []
		
		# Get available boundaries for this geometry
		available_boundaries = self._get_available_boundaries(geometry_type)
		
		# Get list of specified locations
		specified_locations = {bc.get('location') for bc in bcs}
		
		# Skip if "all_boundary" is already specified
		if 'all_boundary' in specified_locations or 'all' in specified_locations:
			logger.debug("All boundaries already specified, skipping defaults")
			return bcs
		
		# Get default BC from configuration
		default_type = 'insulated' if physics_type == 'heat_transfer' else 'free'
		default_value = 0
		default_bc_type = 'neumann'
		
		# Add default BCs for unspecified boundaries
		result = bcs.copy()
		for boundary in available_boundaries:
			if boundary not in specified_locations:
				result.append({
					'type': default_type,
					'location': boundary,
					'value': default_value,
					'bc_type': default_bc_type,
					'confidence': 0.5
				})
				logger.debug(f"Added default {default_type} BC on {boundary}")
		
		return result

	def _get_available_boundaries(self, geometry_type: str) -> list:
		"""Get available boundaries for a geometry type"""
		# Load geometry boundaries from config
		import json
		from pathlib import Path
		
		try:
			# Use project-root config path
			config_path = Path(__file__).parents[2] / "config" / "geometry_boundaries.json"
			with open(config_path, 'r') as f:
				geometry_boundaries = json.load(f)
			
			# Search in all dimension categories
			for dim_group in ['1d', '2d', '3d']:
				if dim_group in geometry_boundaries.get('geometries', {}):
					if geometry_type in geometry_boundaries['geometries'][dim_group]:
						return geometry_boundaries['geometries'][dim_group][geometry_type].get('available_boundaries', [])
		except Exception as e:
			logger.debug(f"Could not load geometry boundaries config: {e}")
		
		# Fallback boundaries
		defaults = {
			'line': ['left', 'right'], 'rod': ['left', 'right'], 'bar': ['left', 'right'],
			'plate': ['left', 'right', 'top', 'bottom'], 'membrane': ['left', 'right', 'top', 'bottom'],
			'rectangle': ['left', 'right', 'top', 'bottom'], 'square': ['left', 'right', 'top', 'bottom'],
			'disc': ['circumference', 'center'],
			'cube': ['left', 'right', 'top', 'bottom', 'front', 'back'],
			'box': ['left', 'right', 'top', 'bottom', 'front', 'back'],
			'beam': ['left', 'right', 'top', 'bottom', 'front', 'back'],
			'cylinder': ['top', 'bottom', 'curved surface'],
			'sphere': ['surface', 'center']
		}
		return defaults.get(geometry_type, ['left', 'right', 'top', 'bottom'])

	def _extract_geometry_type(self, config: Dict[str, Any]) -> str:
		"""Extract geometry type from config"""
		# Try multiple locations
		if 'required_components' in config and 'geometry_type' in config['required_components']:
			return config['required_components']['geometry_type']
		elif 'geometry_type' in config:
			return config['geometry_type']
		elif 'geometry' in config and 'type' in config['geometry']:
			return config['geometry']['type']
		else:
			return 'beam'  # default

	def _extract_dimensions(self, config: Dict[str, Any]) -> Dict[str, float]:
		"""Extract dimensions from config"""
		dimensions = {}

		# Try multiple locations
		if 'pde_config' in config and 'mesh_parameters' in config['pde_config'] and 'dimensions' in config['pde_config']['mesh_parameters']:
			dimensions = config['pde_config']['mesh_parameters']['dimensions']
		elif 'required_components' in config and 'geometry_dimensions' in config['required_components']:
			dimensions = config['required_components']['geometry_dimensions']
		elif 'mesh_parameters' in config and 'dimensions' in config['mesh_parameters']:
			dimensions = config['mesh_parameters']['dimensions']
		elif 'geometry_dimensions' in config:
			dimensions = config['geometry_dimensions']
		elif 'geometry' in config and 'dimensions' in config['geometry']:
			dimensions = config['geometry']['dimensions']

		# Convert to float
		converted_dimensions = {}
		for key, value in dimensions.items():
			try:
				converted_dimensions[key] = float(str(value))
			except (ValueError, TypeError):
				converted_dimensions[key] = 1.0  # default

		return converted_dimensions