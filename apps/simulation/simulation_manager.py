"""
Simulation Manager
Handles complete simulation workflow including materials, boundary conditions, and PDE solving
Now integrated with Master Agent system
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SimulationManager:
	"""Manages complete simulation workflow with agent integration"""

	def __init__(self, mesh_viewer=None, master_agent=None):
		"""Initialize SimulationManager with master agent (required for agentic workflow)"""
		if master_agent is None:
			raise ValueError("master_agent is required for agentic workflow")
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
		from fenics_backend import FEniCSSolver
		from local_field_visualizer import FieldVisualizer
		from mesh_viewer.mesh_viewer import MeshViewer
		from visualizers.mesh_visualizer import MeshVisualizer

		# Store components for internal use (parser accessed through master agent)
		self.fenics_solver = FEniCSSolver()
		self.field_visualizer = FieldVisualizer()
		# Use provided mesh_viewer or create new one
		self.mesh_viewer = mesh_viewer if mesh_viewer is not None else MeshViewer()
		self.mesh_visualizer = MeshVisualizer()
		
		# Master agent for orchestration (required)
		self.master_agent = master_agent
		
		# Parser is accessed through master agent, but keep reference for health checks
		self.parser = master_agent.parser if hasattr(master_agent, 'parser') else None
		
		# Central storage for simulation data
		self.current_context = {}
		self.current_mesh_data = None
		self.current_msh_file = None  # Path to exported .msh file
		self.current_simulation_config = {}

	def clear_all_context(self) -> Dict[str, Any]:
		"""Clear all simulation context including cached mesh data"""
		cleanup_info = {
			'context_cleared': False,
			'msh_files_removed': 0,
			'mesh_data_cleared': False,
			'visualizer_cache_cleared': False
		}
		
		try:
			# Clear parser's internal context through master agent
			if self.master_agent and hasattr(self.master_agent, 'parser') and self.master_agent.parser:
				self.master_agent.parser.clear_context()
				logger.debug("Cleared parser's internal context via master agent")
			
			# Clear basic context
			self.current_context = {}
			self.current_mesh_data = None
			
			# Remove any previously exported .msh file
			if self.current_msh_file and os.path.exists(self.current_msh_file):
				try:
					os.remove(self.current_msh_file)
					cleanup_info['msh_files_removed'] = 1
					logger.debug(f"Removed cached mesh file: {self.current_msh_file}")
				except Exception as remove_err:
					logger.warning(f"Could not remove cached mesh file {self.current_msh_file}: {remove_err}")
			self.current_msh_file = None
			
			self.current_simulation_config = {}
			cleanup_info['context_cleared'] = True
			
			# Clear physics_type from config_manager's pde_config
			# Import here to avoid circular imports
			from config.config_manager import config_manager
			pde_config = config_manager.get_pde_config()
			# Clear physics_type and boundary conditions to ensure fresh start
			if "physics_type" in pde_config:
				pde_config.pop("physics_type", None)  # Remove physics_type entirely
			if "boundary_conditions" in pde_config:
				pde_config["boundary_conditions"] = []  # Clear boundary conditions
			config_manager.save_config()
			logger.debug("Cleared physics_type and boundary_conditions from config_manager")
			
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
		"""Store mesh data in simulation manager"""
		import numpy as np
		
		new_msh_file = mesh_data.get('msh_file')
		if new_msh_file and self.current_msh_file and new_msh_file != self.current_msh_file:
			if os.path.exists(self.current_msh_file):
				try:
					os.remove(self.current_msh_file)
					logger.debug(f"Removed previous mesh file: {self.current_msh_file}")
				except Exception as remove_err:
					logger.warning(f"Could not remove previous mesh file {self.current_msh_file}: {remove_err}")
		self.current_msh_file = new_msh_file
		if self.current_msh_file:
			logger.info(f"Stored mesh file for solver: {self.current_msh_file}")
		else:
			logger.error("Mesh data missing 'msh_file'; solver cannot load mesh without it")
		
		# Convert numpy arrays to lists for JSON serialization
		mesh_data = self._make_json_safe(mesh_data)
		
		if 'physical_groups' not in mesh_data:
			logger.warning("mesh_data does NOT contain 'physical_groups' - this will cause issues with boundary condition resolution!")
		
		self.current_mesh_data = mesh_data
		logger.debug(f"Stored mesh data with keys: {list(mesh_data.keys())}")

	def _make_json_safe(self, obj):
		"""Recursively convert objects to JSON-safe format"""
		import numpy as np
		
		if isinstance(obj, dict):
			# Special handling: if this dict looks like a physical group (has 'dim' and 'tag' but missing 'node_tags'),
			# and we're processing physical_groups, ensure node_tags is preserved
			result = {}
			for key, value in obj.items():
				result[key] = self._make_json_safe(value)
			# If this is a physical group dict that's missing node_tags, check if we can find it
			if 'dim' in result and 'tag' in result and 'node_tags' not in result:
				# This might be a physical group that lost node_tags - log it
				import logging
				logger_debug = logging.getLogger(__name__)
				logger_debug.warning(f"Physical group dict missing node_tags: name={result.get('name', 'unknown')}, keys={list(result.keys())}")
			return result
		elif isinstance(obj, list):
			return [self._make_json_safe(item) for item in obj]
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, (np.integer, np.floating)):
			return obj.item()
		elif hasattr(obj, '_data') and hasattr(obj, '__dict__'):
			# Handle PhysicalGroupWrapper or similar objects with _data attribute
			if hasattr(obj, 'dim') and hasattr(obj, 'tag'):
				# It's a physical group wrapper, convert to dict
				# CRITICAL: Use _data dict directly to ensure node_tags is included
				# _data is the source of truth and contains all fields including node_tags
				if isinstance(obj._data, dict):
					# DEBUG: Log what's in _data before serialization
					import logging
					logger_debug = logging.getLogger(__name__)
					logger_debug.info(f"Serializing PhysicalGroupWrapper: name={getattr(obj, 'name', 'unknown')}, _data keys={list(obj._data.keys())}, node_tags in _data={'node_tags' in obj._data}, node_tags length={len(obj._data.get('node_tags', []))}")
					result = self._make_json_safe(obj._data)
					# DEBUG: Log what's in result after serialization
					if isinstance(result, dict):
						logger_debug.info(f"After serialization: keys={list(result.keys())}, node_tags in result={'node_tags' in result}, node_tags length={len(result.get('node_tags', []))}")
					return result
				else:
					# Fallback: manually construct dict
					return {
						'dim': obj.dim,
						'tag': obj.tag,
						'entities': self._make_json_safe(getattr(obj, 'entities', [])),
						'name': getattr(obj, 'name', None),
						'dimension': getattr(obj, 'dim', None),
							'entity_coordinates': self._make_json_safe(getattr(obj, 'entity_coordinates', [])),
							'node_tags': self._make_json_safe(getattr(obj, 'node_tags', []))  # CRITICAL: Include node_tags
					}
		elif hasattr(obj, '__dict__') and not isinstance(obj, type):
			# Convert objects with __dict__ to dict
			return self._make_json_safe(obj.__dict__)
		else:
			return obj

	def get_current_context(self) -> Dict[str, Any]:
		"""Get current simulation context"""
		return self.current_context

	def get_current_mesh_data(self) -> Optional[Dict[str, Any]]:
		"""Get current mesh data"""
		return self.current_mesh_data

	def get_current_msh_file(self) -> Optional[str]:
		"""Get path to the current exported .msh file"""
		return self.current_msh_file


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
			boundary_conditions = update_context.get("boundary_conditions")
			geometry_type = update_context.get("geometry_type")
			physics_type = update_context.get("physics_type")
			
			# Only process boundary conditions if we have geometry_type (needed for mapping)
			if boundary_conditions and geometry_type:
				try:
					# Check if BCs already have placeholders (from parser)
					has_placeholders = any(bc.get('source') == 'placeholder' or bc.get('is_placeholder') for bc in boundary_conditions if isinstance(bc, dict))
					
					if has_placeholders:
						# BCs already have placeholders, just map locations if needed
						mapped_bcs = self._map_boundary_locations(
							boundary_conditions, 
							geometry_type
						)
						pde_config["boundary_conditions"] = mapped_bcs
					else:
						# Map boundary locations first
						mapped_bcs = self._map_boundary_locations(
							boundary_conditions, 
							geometry_type
						)
						
						# Add default boundary conditions only if parser didn't add placeholders
						processed_bcs = self._add_default_boundary_conditions(
							mapped_bcs, 
							geometry_type,
							physics_type
						)
						
						pde_config["boundary_conditions"] = processed_bcs
				except Exception as e:
					logger.error(f"Error processing boundary conditions: {e}", exc_info=True)
					# Still store boundary conditions even if mapping fails
					pde_config["boundary_conditions"] = boundary_conditions
			elif boundary_conditions:
				# Store boundary conditions even without geometry_type (will be processed later)
				pde_config["boundary_conditions"] = boundary_conditions
			
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
		"""Parse simulation prompt using master agent (agentic workflow)"""
		try:
			result = self.master_agent.execute_task("parse_simulation", {
				'prompt': prompt,
				'context': context or self.current_context
			})
			
			# Store the parsed context in simulation manager
			# Handle both "updated_context" and "context" fields
			context_to_store = result.get("updated_context") or result.get("context") or {}
			if context_to_store:
				self.current_context = context_to_store
				logger.debug(f"Stored context with keys: {list(self.current_context.keys())}")
				
				# IMPORTANT: Update PDE config from context to ensure boundary conditions are stored
				# This transfers boundary conditions from context to pde_config in simulation_config
				update_success = self.update_pde_config_from_context(context_to_store)
				if not update_success:
					logger.warning("Failed to update PDE config from context")
				
				# Ensure result has updated_context for consistency
				if "updated_context" not in result:
					result["updated_context"] = context_to_store
			
			if result.get("simulation_config"):
				self.current_simulation_config = result["simulation_config"]
				logger.debug(f"Stored simulation config with keys: {list(self.current_simulation_config.keys())}")
				
				# Ensure boundary conditions from context are also in simulation_config
				if context_to_store.get("boundary_conditions"):
					if "simulation_config" not in self.current_simulation_config:
						self.current_simulation_config["simulation_config"] = {}
					if "pde_config" not in self.current_simulation_config["simulation_config"]:
						self.current_simulation_config["simulation_config"]["pde_config"] = {}
			
			# Return in the expected format
			return {
				"success": True,
				"result": result
			}
		except Exception as e:
			logger.error(f"Error parsing simulation prompt with master agent: {e}")
			return {
				"success": False,
				"error": str(e)
			}

	def solve_pde(self, config: Dict[str, Any], mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Solve PDE using FEniCS solver with GMSH model"""
		try:
			logger.info("Solving PDE")
			
			# CRITICAL: Ensure mesh_data includes physical_groups from current_mesh_data if missing
			# The mesh generator creates physical_groups with correct names, but they might be lost during serialization
			if self.current_mesh_data and 'physical_groups' in self.current_mesh_data:
				if 'physical_groups' not in mesh_data or not mesh_data.get('physical_groups'):
					logger.info(f"Adding physical_groups from current_mesh_data to mesh_data ({len(self.current_mesh_data.get('physical_groups', {}))} groups)")
					mesh_data['physical_groups'] = self.current_mesh_data['physical_groups']
				else:
					logger.debug(f"mesh_data already has physical_groups ({len(mesh_data.get('physical_groups', {}))} groups)")
			else:
				logger.warning("current_mesh_data does not have physical_groups - cannot supplement mesh_data")
			
			# Provide mesh file path to FEniCS solver
			msh_file = mesh_data.get('msh_file') or self.current_msh_file
			if not msh_file:
				logger.error("Mesh data missing 'msh_file'; cannot solve PDE")
				return {
					"success": False,
					"error": "Mesh file not available for solver"
				}
			if not os.path.exists(msh_file):
				logger.error(f"Mesh file does not exist: {msh_file}")
				return {
					"success": False,
					"error": f"Mesh file not found: {msh_file}"
				}
			self.fenics_solver.set_mesh_file(msh_file)
			
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
		"""Run complete simulation workflow using master agent (agentic workflow)"""
		try:
			result = self.master_agent.execute_task("run_complete_simulation", {
				'prompt': prompt,
				'context': context or self.current_context,
				'mesh_data': existing_mesh_data or self.current_mesh_data
			})
			
			# Update internal state from result
			if result.get("context"):
				self.current_context = result["context"]
			if result.get("simulation_config"):
				self.current_simulation_config = result["simulation_config"]
			if result.get("solution_data"):
				# Store solution data
				if "simulation_config" not in self.current_simulation_config:
					self.current_simulation_config["simulation_config"] = {}
				self.current_simulation_config["solution_data"] = result["solution_data"]
			# Store mesh data if available
			if result.get("mesh_data"):
				self.current_mesh_data = result["mesh_data"]
			
			return result
		except Exception as e:
			logger.error(f"Error running complete simulation with master agent: {e}")
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
		if not boundary_conditions or len(boundary_conditions) == 0:
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
			location = bc.get('location', '').strip()  # Keep original case for validation
			location_lower = location.lower().strip()
			
			# Check if location is already valid for this geometry
			if location in available_boundaries:
				# Location is already valid, no mapping needed
				logger.debug(f"Location '{location}' is already valid for {geometry_type}, skipping mapping")
				mapped_bcs.append(mapped_bc)
				continue
			
			# Location needs mapping - try geometry-specific mapping first
			mapped_location = None
			if geometry_type in geometry_specific_mappings:
				if location_lower in geometry_specific_mappings[geometry_type]:
					mapped_location = geometry_specific_mappings[geometry_type][location_lower]
					logger.debug(f"Geometry-specific mapping for {geometry_type}: '{location}' → '{mapped_location}'")
			
			# Fallback to generic mapping
			if not mapped_location and location_lower in location_mappings:
				mapped_location = location_mappings[location_lower]
				logger.debug(f"Generic mapping: '{location}' → '{mapped_location}'")
			
			# Validate mapped location is available for this geometry
			if mapped_location and (mapped_location in available_boundaries or mapped_location in ['all', 'all_boundary']):
				mapped_bc['location'] = mapped_location
				logger.debug(f"Mapped location '{location}' to '{mapped_location}' for {geometry_type}")
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
								else:  # Low temperature/force
									mapped_bc['location'] = 'center'
							else:
								# Default for disc: alternate between circumference and center
								circumference_count = sum(1 for bc in bcs if bc.get('location') == 'circumference')
								if circumference_count == 0:
									mapped_bc['location'] = 'circumference'
								else:
									mapped_bc['location'] = 'center'
						else:
							# Default: use first available boundary
							mapped_bc['location'] = available_boundaries[0] if available_boundaries else 'left'
			
			mapped_bcs.append(mapped_bc)
		
		return mapped_bcs

	def _add_default_boundary_conditions(self, bcs: list, geometry_type: str, physics_type: str) -> list:
		"""Add default boundary conditions for unspecified boundaries based on geometry"""
		if not bcs:
			logger.warning(f"_add_default_boundary_conditions called with empty bcs list for {geometry_type}")
			return []
		
		# Get available boundaries for this geometry
		available_boundaries = self._get_available_boundaries(geometry_type)
		if not available_boundaries:
			return bcs
		
		# Get list of specified locations
		specified_locations = {bc.get('location') for bc in bcs}
		
		# Skip if "all_boundary" is already specified
		if 'all_boundary' in specified_locations or 'all' in specified_locations:
			return bcs
		
		# Get default BC from configuration
		default_type = 'insulated' if physics_type == 'heat_transfer' else 'free'
		default_value = 0
		default_bc_type = 'neumann'
		
		# Add default BCs for unspecified boundaries
		result = bcs.copy()
		added_count = 0
		for boundary in available_boundaries:
			if boundary not in specified_locations:
				result.append({
					'type': default_type,
					'location': boundary,
					'value': default_value,
					'bc_type': default_bc_type,
					'confidence': 0.5,
					'source': 'default'
				})
				added_count += 1
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