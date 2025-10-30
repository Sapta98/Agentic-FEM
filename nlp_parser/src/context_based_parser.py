#!/usr/bin/env python3
"""
Context-Based Natural Language Parser for Physics Simulations
"""

import logging
from typing import Dict, Any, Optional
from openai import OpenAI

try:
	from .prompt_templates import PromptManager
	from .template_manager import TemplateManager
except ImportError:
	from prompt_templates import PromptManager
	from template_manager import TemplateManager

logger = logging.getLogger(__name__)

class ContextBasedParser:
	"""Context-based parser that continuously builds simulation context"""

	def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4"):
		"""
		Initialize the context-based parser

		Args:
			openai_api_key: OpenAI API key (if None, will use environment variable)
			model: OpenAI model to use (default: gpt-4)
		"""
		if openai_api_key is None:
			import os
			openai_api_key = os.getenv("OPENAI_API_KEY")

			if not openai_api_key:
				raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")

		self.client = OpenAI(api_key=openai_api_key)
		self.model = model
		self.prompt_manager = PromptManager(self.client, model)
		self.template_manager = TemplateManager()

		# Current simulation context
		self.context = {}

		# Statistics tracking
		self.parse_count = 0
		self.success_count = 0
		self.error_count = 0
		self.avg_response_time = 0.0

		# Load geometry boundaries configuration
		self.geometry_boundaries = self._load_geometry_boundaries()

	def parse_prompt(self, prompt: str) -> dict:
		"""
		Parse prompt and update context with any new information found

		Args:
			prompt: User's natural language prompt

		Returns:
			Dictionary with parsing results and next steps
		"""
		try:
			self.parse_count += 1
			start_time = time.time()

			# Check if user explicitly wants to clear context
			if "clear context" in prompt.lower():
				self.clear_context()
				return {
					"action": "context_cleared",
					"message": "Context cleared. Ready for new simulation.",
					"context": self.context
				}

			# Step 1: Check if this is a new simulation or continuation
			is_new_simulation = self._is_new_simulation(prompt)

			if is_new_simulation or not self.context.get('physics_type'):
				# Check if this is a physics simulation
				physics_result = self._check_physics_simulation(prompt)
				if not physics_result['is_physics']:
					return {
						"action": "request_info",
						"message": physics_result['message'],
						"context": self.context
					}

				# Update physics type
				self.context['physics_type'] = physics_result['physics_type']
				self.context['confidence_scores'] = physics_result['confidence_scores']
				logger.info(f"Physics type established: {self.context['physics_type']}")
			else:
				logger.info(f"Continuing with existing physics type: {self.context['physics_type']}")

			# Step 2: Check for material
			material_result = self._check_material(prompt)
			if material_result['found']:
				old_material = self.context.get('material_type', 'none')
				self.context['material_type'] = material_result['material_type']
				self._fetch_material_properties()
				if old_material != 'none' and old_material != material_result['material_type']:
					logger.info(f"Material updated: {old_material} -> {self.context['material_type']}")
				else:
					logger.info(f"Material established: {self.context['material_type']}")
			elif not self.context.get('material_type'):
				return self._request_material_info()

			# Step 3: Check for geometry
			geometry_result = self._check_geometry(prompt)
			if geometry_result['found']:
				old_geometry = self.context.get('geometry_type', 'none')
				self.context['geometry_type'] = geometry_result['geometry_type']
				if old_geometry != 'none' and old_geometry != geometry_result['geometry_type']:
					logger.info(f"Geometry updated: {old_geometry} -> {self.context['geometry_type']}")
				else:
					logger.info(f"Geometry established: {self.context['geometry_type']}")
			elif not self.context.get('geometry_type'):
				return self._request_geometry_info()

			# Step 4: Check for dimensions
			dimensions_result = self._check_dimensions(prompt)
			if dimensions_result['found']:
				self.context['geometry_dimensions'] = dimensions_result['dimensions']
				logger.info(f"Dimensions updated: {dimensions_result['dimensions']}")
			elif not self.context.get('geometry_dimensions'):
				return self._request_dimensions_info()

			# Step 5: Check for boundary conditions
			bc_result = self._check_boundary_conditions(prompt)
			if bc_result['found']:
				# Add default BCs for unspecified boundaries
				processed_bcs = self._add_default_boundary_conditions(bc_result['boundary_conditions'])
				
				self.context['boundary_conditions'] = processed_bcs
				
				# Enhanced logging for boundary conditions
				bc_summary = []
				for bc in processed_bcs:
					bc_type = bc.get('type', 'unknown')
					location = bc.get('location', 'unspecified')
					value = bc.get('value')
					bc_classification = bc.get('bc_type', 'unknown')
					confidence = bc.get('confidence', 0.0)
					
					value_str = f" = {value}" if value is not None else ""
					bc_summary.append(f"{bc_type} at {location}{value_str} ({bc_classification} BC, confidence: {confidence:.1f})")
				
				logger.info(f"Boundary conditions detected: {', '.join(bc_summary)}")
				logger.info(f"Full boundary conditions data: {processed_bcs}")
			elif not self.context.get('boundary_conditions'):
				return self._request_boundary_conditions_info()

			# Step 6: Check for external loads
			loads_result = self._check_external_loads(prompt)
			if loads_result['found']:
				self.context['external_loads'] = loads_result['external_loads']
				# Enhanced logging for external loads
				loads_list = loads_result['external_loads']
				loads_summary = []
				for load in loads_list:
					load_type = load.get('type', 'unknown')
					magnitude = load.get('magnitude', 'unspecified')
					direction = load.get('direction', 'unspecified')
					location = load.get('location', 'unspecified')
					confidence = load.get('confidence', 0.0)
					
					loads_summary.append(f"{load_type} {magnitude} in {direction} at {location} (confidence: {confidence:.1f})")
				
				logger.info(f"External loads detected: {', '.join(loads_summary)}")
				logger.info(f"Full external loads data: {loads_result['external_loads']}")

			# Step 7: Complete data structure
			completion_result = self._complete_data_structure(prompt)
			if completion_result['success']:
				self.context.update(completion_result['data'])
				logger.info("Data structure completed")

			# Update statistics
			response_time = time.time() - start_time
			self.avg_response_time = (self.avg_response_time * (self.parse_count - 1) + response_time) / self.parse_count
			self.success_count += 1

			# Check if simulation is ready
			completeness = self.template_manager.check_completeness(
				self.context['physics_type'], self.context
			)

			if completeness['complete']:
				return {
					"action": "simulation_ready",
					"message": "Simulation is ready to run!",
					"context": self.context,
					"completeness": completeness,
					"simulation_config": self._create_simulation_config()
				}
			else:
				return {
					"action": "continue",
					"message": f"Simulation progress: {len(completeness['missing'])} items missing",
					"context": self.context,
					"completeness": completeness,
					"next_steps": self._get_next_steps(completeness['missing'])
				}

		except Exception as e:
			self.error_count += 1
			logger.error(f"Error parsing prompt: {e}")
			return {
				"action": "error",
				"message": f"Error parsing prompt: {str(e)}",
				"context": self.context
			}

	def _is_new_simulation(self, prompt: str) -> bool:
		"""Check if this is a new simulation request"""
		new_simulation_keywords = [
			"new simulation", "start over", "begin", "create", "simulate",
			"analyze", "model", "design", "calculate"
		]
		return any(keyword in prompt.lower() for keyword in new_simulation_keywords)

	def _check_physics_simulation(self, prompt: str) -> dict:
		"""Check if prompt is about physics simulation"""
		physics_result = self.prompt_manager.identify_physics_type(prompt)
		
		if physics_result.get('error'):
			return {
				'is_physics': False,
				'message': 'Unable to determine if this is a physics simulation. Please clarify.',
				'physics_type': None,
				'confidence_scores': {}
			}

		if not physics_result.get('is_physics', False):
			return {
				'is_physics': False,
				'message': 'This does not appear to be a physics simulation. Please provide a physics-related prompt.',
				'physics_type': None,
				'confidence_scores': physics_result.get('confidence_scores', {})
			}

		# Determine physics type from confidence scores with heuristic override
		confidence_scores = physics_result.get('confidence_scores', {})
		heat_transfer_score = confidence_scores.get('heat_transfer', 0)
		solid_mechanics_score = confidence_scores.get('solid_mechanics', 0)
		other_physics_score = confidence_scores.get('other_physics', 0)

		# Heuristic keywords indicating solid mechanics
		text = (prompt or "").lower()
		sm_keywords = [
			"stress", "strain", "load", "force", "pressure load", "traction",
			"fixed", "clamped", "cantilever", "displacement", "deflection",
			"beam", "bar", "rod", "young's modulus", "poisson"
		]
		if any(k in text for k in sm_keywords):
			physics_type = 'solid_mechanics'
			# Boost score so downstream prompts reflect the override
			confidence_scores['solid_mechanics'] = max(solid_mechanics_score, 0.9)
			logger.info("Heuristic override: detected solid mechanics by keywords in prompt")
		else:
			if heat_transfer_score > solid_mechanics_score and heat_transfer_score > other_physics_score:
				physics_type = 'heat_transfer'
			elif solid_mechanics_score > other_physics_score:
				physics_type = 'solid_mechanics'
			else:
					physics_type = 'heat_transfer'  # Default to heat transfer

		return {
			'is_physics': True,
			'message': f'Physics simulation detected: {physics_type}',
			'physics_type': physics_type,
			'confidence_scores': confidence_scores
		}

	def _check_material(self, prompt: str) -> dict:
		"""Check for material information in prompt"""
		material_result = self.prompt_manager.identify_material_type(
			prompt, self.context.get('physics_type', ''), self._get_context_summary()
		)

		if material_result.get('error'):
			return {'found': False, 'material_type': None}

		if material_result.get('has_material_type', False):
			return {
				'found': True,
				'material_type': material_result.get('material_type')
			}

		return {'found': False, 'material_type': None}

	def _check_geometry(self, prompt: str) -> dict:
		"""Check for geometry information in prompt using both AI and geometry classifier"""
		# First try AI-based detection
		geometry_result = self.prompt_manager.identify_geometry_type(
			prompt, self.context.get('physics_type', ''), self._get_context_summary()
		)

		if geometry_result.get('error'):
			return {'found': False, 'geometry_type': None}

		if geometry_result.get('has_geometry_type', False):
			ai_geometry = geometry_result.get('geometry_type')
			ai_confidence = geometry_result.get('confidence', 0.5)
			
			# Use geometry classifier for additional validation and ranking
			try:
				from .geometry_classifier import classify_geometry
				from pathlib import Path
				
				dimensions_path = Path(__file__).parent.parent.parent / "config" / "dimensions.json"
				classifier_results = classify_geometry(prompt, dimensions_path)
				
				if classifier_results:
					best_classifier = classifier_results[0]
					classifier_geometry = best_classifier['geometry_type']
					classifier_confidence = best_classifier['confidence']
					
					# Use classifier result if it has higher confidence or if AI failed
					if classifier_confidence > ai_confidence or ai_confidence < 0.3:
						logger.info(f"Using geometry classifier result: {classifier_geometry} (confidence: {classifier_confidence:.2f})")
						return {
							'found': True,
							'geometry_type': classifier_geometry,
							'confidence': classifier_confidence,
							'source': 'classifier'
						}
					else:
						logger.info(f"Using AI result: {ai_geometry} (confidence: {ai_confidence:.2f})")
						return {
							'found': True,
							'geometry_type': ai_geometry,
							'confidence': ai_confidence,
							'source': 'ai'
						}
			except Exception as e:
				logger.warning(f"Geometry classifier failed: {e}, using AI result")
			
			return {
				'found': True,
				'geometry_type': ai_geometry,
				'confidence': ai_confidence,
				'source': 'ai'
			}

		return {'found': False, 'geometry_type': None}

	def _check_dimensions(self, prompt: str) -> dict:
		"""Check for dimension information in prompt"""
		geometry_type = self.context.get('geometry_type')
		if not geometry_type:
			return {'found': False, 'dimensions': None}

		# Get required dimensions for this geometry type
		required_dims = self.template_manager.get_geometry_dimension_requirements(
			self.context.get('physics_type', ''), geometry_type
		)

		if not required_dims:
			return {'found': False, 'dimensions': None}

		dimensions_result = self.prompt_manager.parse_dimensions(prompt, geometry_type, required_dims)

		if dimensions_result.get('error'):
			return {'found': False, 'dimensions': None}

		dimensions = dimensions_result.get('dimensions', {})
		units = dimensions_result.get('units', {})
		
		if dimensions:
			# Convert units to meters (base unit)
			converted_dimensions = self._convert_dimension_units(dimensions, units)
			return {'found': True, 'dimensions': converted_dimensions}

	def _convert_dimension_units(self, dimensions: dict, units: dict) -> dict:
		"""Convert dimension units to meters (base unit)"""
		converted = {}
		
		for dim_name, value in dimensions.items():
			if value is None:
				converted[dim_name] = None
				continue
				
			try:
				num_value = float(value)
				unit = units.get(dim_name, 'm').lower()
				
				# Convert to meters
				if unit in ['mm', 'millimeter', 'millimeters']:
					num_value = num_value / 1000
				elif unit in ['cm', 'centimeter', 'centimeters']:
					num_value = num_value / 100
				elif unit in ['km', 'kilometer', 'kilometers']:
					num_value = num_value * 1000
				elif unit in ['inch', 'inches', 'in']:
					num_value = num_value * 0.0254
				elif unit in ['ft', 'feet', 'foot']:
					num_value = num_value * 0.3048
				elif unit in ['m', 'meter', 'meters']:
					# Already in meters
					pass
				else:
					# Default to meters if unit not recognized
					logger.warning(f"Unknown unit '{unit}' for dimension '{dim_name}', assuming meters")
				
				converted[dim_name] = num_value
				logger.info(f"Converted {dim_name}: {value} {unit} -> {num_value} m")
				
			except (ValueError, TypeError) as e:
				logger.warning(f"Could not convert dimension '{dim_name}' value '{value}': {e}")
				converted[dim_name] = value
		
		return converted

	def _check_boundary_conditions(self, prompt: str) -> dict:
		geometry_type = self.context.get('geometry_type', '').lower()
		available_boundaries = self._get_available_boundaries(geometry_type)
		
		# STEP 1: Use boundary location classifier for initial mapping
		logger.info(f"Step 1: Using boundary location classifier for geometry '{geometry_type}'")
		classifier_locations = []
		
		try:
			from .boundary_location_classifier import classify_boundary_locations
			from pathlib import Path
			
			geometry_boundaries_path = Path(__file__).parent.parent.parent / "config" / "geometry_boundaries.json"
			classifier_results = classify_boundary_locations(prompt, geometry_type, geometry_boundaries_path)
			
			if classifier_results:
				# Convert classifier results to boundary_locations format
				for result in classifier_results:
					if result['confidence'] > 0.3:  # Only use high-confidence results
						classifier_locations.append({
							'vague_location': 'detected by classifier',
							'specific_boundary': result['boundary'],
							'value': None,
							'confidence': result['confidence']
						})
				
				logger.info(f"Boundary location classifier results: {classifier_locations}")
		except Exception as e:
			logger.warning(f"Boundary location classifier failed: {e}")
		
		# STEP 2: Use AI for additional boundary location detection
		logger.info("Step 2: Using AI for boundary location mapping")
		locations_result = self.prompt_manager.analyze_boundary_locations(
			prompt,
			geometry_type,
			available_boundaries
		)
		
		if locations_result.get('error'):
			logger.error(f"Error in AI boundary location mapping: {locations_result.get('error')}")
			ai_locations = []
		else:
			ai_locations = locations_result.get('boundary_locations', [])
		
		# STEP 3: Combine classifier and AI results
		all_locations = classifier_locations + ai_locations
		if not all_locations:
			logger.info("No boundary locations found")
			return {'found': False, 'boundary_conditions': None}

		logger.info(f"Combined boundary locations: {all_locations}")
		
		# STEP 4: Determine boundary condition types for each location
		logger.info("Step 4: Determining boundary condition types")
		bc_types_result = self.prompt_manager.analyze_boundary_condition_types(
			prompt,
			self.context.get('physics_type', ''),
			all_locations
		)
		
		if bc_types_result.get('error'):
			logger.error(f"Error in boundary condition type analysis: {bc_types_result.get('error')}")
			return {'found': False, 'boundary_conditions': None}

		boundary_conditions = bc_types_result.get('boundary_conditions', [])
		# Normalize BC synonyms and fix heat-transfer flux/temperature labeling based on units in prompt
		if boundary_conditions:
			try:
				pt = self.context.get('physics_type', '').lower()
				text = (prompt or '').lower()
				import re
				wperm2 = re.search(r"\b([-+]?\d*\.?\d+)\s*(w\s*/\s*m\s*\^?2|w\s*/\s*m2)\b", text, re.IGNORECASE)
				for bc in boundary_conditions:
					bct = (bc.get('type') or bc.get('bc_type') or '').strip().lower()
					if pt == 'heat_transfer':
						if bct in ('neumann', 'flux', 'heat flux', 'heat_flux'):
							bc['type'] = 'heat_flux'
							bc['bc_type'] = 'neumann'
						elif bct in ('dirichlet', 'temperature', 'fixed'):
							bc['type'] = 'temperature'
							bc['bc_type'] = 'dirichlet'
					elif pt == 'solid_mechanics':
						if bct in ('neumann', 'traction'):
							bc['type'] = 'traction'
							bc['bc_type'] = 'neumann'
						elif bct in ('dirichlet', 'fixed', 'displacement'):
							bc['type'] = 'fixed'
							bc['bc_type'] = 'dirichlet'
				# If heat transfer prompt includes W/m^2 but no flux BC was detected, assign one
				if self.context.get('physics_type','').lower() == 'heat_transfer' and wperm2:
					has_flux = any((bc.get('type') or '').lower() in ('heat_flux','flux') for bc in boundary_conditions)
					if not has_flux and boundary_conditions:
						boundary_conditions[0]['type'] = 'heat_flux'
						boundary_conditions[0]['bc_type'] = 'neumann'
						try:
							boundary_conditions[0]['value'] = float(wperm2.group(1))
						except Exception:
							pass
			except Exception:
				pass

		if boundary_conditions:
			logger.info(f"Final boundary conditions: {boundary_conditions}")
			return {
				'found': True,
				'boundary_conditions': boundary_conditions
			}

		return {'found': False, 'boundary_conditions': None}

	def _load_geometry_boundaries(self) -> dict:
		"""Load geometry boundaries configuration from JSON file"""
		import json
		import os
		from pathlib import Path
		
		# Try to find config file relative to project root
		current_dir = Path(__file__).parent.parent.parent.parent
		config_path = current_dir / "config" / "geometry_boundaries.json"
		
		if config_path.exists():
			with open(config_path, 'r') as f:
				return json.load(f)
		else:
			logger.warning(f"Geometry boundaries config not found at {config_path}, using defaults")
			return {}

	def _add_default_boundary_conditions(self, bcs: list) -> list:
		"""Add default boundary conditions for unspecified boundaries based on geometry"""
		if not bcs:
			return []
		
		geometry_type = self.context.get('geometry_type', '').lower()
		physics_type = self.context.get('physics_type', '').lower()
		
		# Get available boundaries from configuration
		available_boundaries = self._get_available_boundaries(geometry_type)
		
		# Get list of specified locations
		specified_locations = {bc.get('location') for bc in bcs}
		
		# Skip if "all_boundary" is already specified
		if 'all_boundary' in specified_locations or 'all' in specified_locations:
			logger.info("All boundaries already specified, skipping defaults")
			return bcs
		
		# Get default BC from configuration
		if self.geometry_boundaries and 'default_boundary_conditions' in self.geometry_boundaries:
			defaults = self.geometry_boundaries['default_boundary_conditions'].get(physics_type, 
				self.geometry_boundaries['default_boundary_conditions'].get('heat_transfer', {}))
			default_type = defaults.get('type', 'insulated')
			default_value = defaults.get('value', 0)
			default_bc_type = defaults.get('bc_type', 'neumann')
		else:
			# Fallback defaults
			if physics_type == 'heat_transfer':
				default_type = 'insulated'
				default_value = 0
				default_bc_type = 'neumann'
			elif physics_type == 'solid_mechanics':
				default_type = 'free'
				default_value = 0
				default_bc_type = 'neumann'
			else:
				default_type = 'insulated'
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
				logger.info(f"Added default {default_type} BC on {boundary}")
		
		return result

	def _get_available_boundaries(self, geometry_type: str) -> list:
		"""Get available boundaries for a geometry type from configuration"""
		if not self.geometry_boundaries:
			# Fallback if config not loaded
			defaults = {
				'line': ['left', 'right'], 'rod': ['left', 'right'], 'bar': ['left', 'right'],
				'plate': ['left', 'right', 'top', 'bottom'], 'membrane': ['left', 'right', 'top', 'bottom'],
				'rectangle': ['left', 'right', 'top', 'bottom'], 'square': ['left', 'right', 'top', 'bottom'],
				'disc': ['circumference'],
				'cube': ['left', 'right', 'top', 'bottom', 'front', 'back'],
				'box': ['left', 'right', 'top', 'bottom', 'front', 'back'],
				'beam': ['left', 'right', 'top', 'bottom', 'front', 'back'],
				'cylinder': ['top', 'bottom', 'curved surface'],
				'sphere': ['surface']
			}
			return defaults.get(geometry_type, ['left', 'right', 'top', 'bottom'])
		
		# Search in all dimension categories
		for dim_group in ['1d', '2d', '3d']:
			if dim_group in self.geometry_boundaries.get('geometries', {}):
				if geometry_type in self.geometry_boundaries['geometries'][dim_group]:
					return self.geometry_boundaries['geometries'][dim_group][geometry_type].get('available_boundaries', [])
		
		return ['left', 'right', 'top', 'bottom']  # default fallback

	def _check_external_loads(self, prompt: str) -> dict:
		"""Check for external loads in prompt"""
		loads_result = self.prompt_manager.analyze_external_loads(
			prompt, self.context.get('physics_type', ''), self._get_context_summary()
		)

		if loads_result.get('error'):
			return {'found': False, 'external_loads': None}

		if loads_result.get('has_external_loads', False):
			return {
				'found': True,
				'external_loads': loads_result.get('external_loads', [])
			}

		return {'found': False, 'external_loads': None}

	def _complete_data_structure(self, prompt: str) -> dict:
		"""Complete the data structure with missing information"""
		try:
			# Preserve existing material properties before calling AI completion
			existing_material_properties = self.context.get('material_properties', {})
			
			completion_result = self.prompt_manager.complete_data_structure(
				prompt,
				self.context.get('physics_type', ''),
				self.context.get('material_type', ''),
				self.context.get('geometry_type', ''),
				self.context.get('boundary_conditions', {}),
				self._get_context_summary()
			)

			if completion_result.get('error'):
				return {'success': False, 'data': {}}

			# If we had real material properties, preserve them instead of AI placeholders
			if existing_material_properties and any(isinstance(v, (int, float)) for v in existing_material_properties.values()):
				logger.info("Preserving existing material properties from JSON file")
				completion_result['material_properties'] = existing_material_properties

			return {
				'success': True,
				'data': completion_result
			}
		except Exception as e:
			logger.error(f"Error completing data structure: {e}")
			return {'success': False, 'data': {}}

	def _fetch_material_properties(self):
		"""Fetch material properties for the current material type"""
		try:
			material_type = self.context.get('material_type')
			physics_type = self.context.get('physics_type', '')

			if material_type and physics_type:
				# Try to load from JSON file first
				json_properties = self._load_material_properties_from_json(material_type)
				if json_properties:
					self.context['material_properties'] = json_properties
					logger.info(f"Loaded material properties for {material_type} from JSON file")
					
					# Log the properties
					prop_summary = []
					for prop_name, prop_value in json_properties.items():
						prop_summary.append(f"{prop_name}: {prop_value}")
					logger.info(f"Material properties for {material_type}: {', '.join(prop_summary)}")
					return
				
				# Fallback to AI if JSON not found
				logger.info(f"Material {material_type} not found in JSON file, using AI fallback")
				properties_result = self.prompt_manager.get_material_properties(material_type, physics_type)
				if not properties_result.get('error'):
					raw_properties = properties_result.get('properties', {})
					# Parse material properties to extract numeric values
					parsed_properties = self._parse_material_properties(raw_properties, material_type, physics_type)
					self.context['material_properties'] = parsed_properties
					
					# Enhanced logging for material properties
					prop_summary = []
					for prop_name, prop_value in parsed_properties.items():
						if isinstance(prop_value, (int, float)):
							prop_summary.append(f"{prop_name}: {prop_value}")
						else:
							prop_summary.append(f"{prop_name}: {prop_value}")
					
					logger.info(f"Material properties fetched for {material_type}: {', '.join(prop_summary)}")
					logger.info(f"Full material properties data: {parsed_properties}")
		except Exception as e:
			logger.error(f"Error fetching material properties: {e}")

	def _load_material_properties_from_json(self, material_type: str) -> dict:
		"""Load material properties from JSON file"""
		try:
			import json
			import os
			
			# Get the path to the config directory
			current_dir = os.path.dirname(os.path.abspath(__file__))
			config_path = os.path.join(current_dir, '..', '..', 'config', 'material_properties.json')
			config_path = os.path.normpath(config_path)
			
			logger.info(f"Looking for material properties file at: {config_path}")
			logger.info(f"Current directory: {current_dir}")
			
			if not os.path.exists(config_path):
				logger.warning(f"Material properties file not found at {config_path}")
				return {}
			
			with open(config_path, 'r') as f:
				materials_data = json.load(f)
			
			logger.info(f"Loaded materials from JSON: {list(materials_data.keys())}")
			
			# Try exact match first
			if material_type.lower() in materials_data:
				logger.info(f"Found exact match for {material_type}")
				return materials_data[material_type.lower()]
			
			# Try partial matches
			material_lower = material_type.lower()
			for key, properties in materials_data.items():
				if material_lower in key or key in material_lower:
					logger.info(f"Found partial match: {key} for {material_type}")
					return properties
			
			logger.warning(f"No material properties found for {material_type}")
			return {}
			
		except Exception as e:
			logger.error(f"Error loading material properties from JSON: {e}")
			return {}

	def _parse_material_properties(self, raw_properties: dict, material_type: str, physics_type: str) -> dict:
		"""Parse material properties to extract numeric values from strings with units"""
		parsed = {}
		
		# Default material properties for common materials
		material_defaults = {
			'steel': {
				'density': 7850.0,
				'thermal_conductivity': 50.0,
				'specific_heat': 460.0,
				'youngs_modulus': 200e9,
				'poisson_ratio': 0.3,
				'yield_strength': 250e6,
				'thermal_expansion': 12e-6
			},
			'aluminum': {
				'density': 2700.0,
				'thermal_conductivity': 205.0,
				'specific_heat': 900.0,
				'youngs_modulus': 70e9,
				'poisson_ratio': 0.33,
				'yield_strength': 95e6,
				'thermal_expansion': 23e-6
			},
			'copper': {
				'density': 8960.0,
				'thermal_conductivity': 400.0,
				'specific_heat': 385.0,
				'youngs_modulus': 110e9,
				'poisson_ratio': 0.34,
				'yield_strength': 70e6,
				'thermal_expansion': 17e-6
			},
			'concrete': {
				'density': 2300.0,
				'thermal_conductivity': 1.7,
				'specific_heat': 880.0,
				'youngs_modulus': 30e9,
				'poisson_ratio': 0.2,
				'compressive_strength': 30e6,
				'thermal_expansion': 10e-6
			},
			'wood': {
				'density': 600.0,
				'thermal_conductivity': 0.12,
				'specific_heat': 1700.0,
				'youngs_modulus': 12e9,
				'poisson_ratio': 0.4,
				'compressive_strength': 40e6,
				'thermal_expansion': 5e-6
			}
		}
		
		# Use defaults for the material type if available
		defaults = material_defaults.get(material_type.lower(), {})
		
		for prop_name, prop_value in raw_properties.items():
			if isinstance(prop_value, (int, float)):
				# Already a numeric value
				parsed[prop_name] = prop_value
			elif isinstance(prop_value, str):
				# Try to extract numeric value from string
				import re
				# Look for numbers in the string
				numbers = re.findall(r'[\d.]+(?:[eE][+-]?\d+)?', prop_value)
				if numbers:
					try:
						# Use the first number found
						parsed[prop_name] = float(numbers[0])
					except ValueError:
						# If parsing fails, use default if available
						parsed[prop_name] = defaults.get(prop_name, 1.0)
				else:
					# No numbers found, use default if available
					parsed[prop_name] = defaults.get(prop_name, 1.0)
			else:
				# Unknown type, use default if available
				parsed[prop_name] = defaults.get(prop_name, 1.0)
		
		# Ensure we have all required properties for the physics type
		if physics_type == 'solid_mechanics':
			required_props = ['youngs_modulus', 'poisson_ratio', 'density']
			for prop in required_props:
				if prop not in parsed:
					parsed[prop] = defaults.get(prop, 1.0)
		elif physics_type == 'heat_transfer':
			required_props = ['thermal_conductivity', 'specific_heat', 'density']
			for prop in required_props:
				if prop not in parsed:
					parsed[prop] = defaults.get(prop, 1.0)
		
		return parsed

	def _get_context_summary(self) -> str:
		"""Get a summary of the current context"""
		summary_parts = []
		
		if self.context.get('physics_type'):
			summary_parts.append(f"Physics: {self.context['physics_type']}")
		
		if self.context.get('material_type'):
			summary_parts.append(f"Material: {self.context['material_type']}")
		
		if self.context.get('geometry_type'):
			summary_parts.append(f"Geometry: {self.context['geometry_type']}")
		
		if self.context.get('geometry_dimensions'):
			dims = self.context['geometry_dimensions']
			summary_parts.append(f"Dimensions: {dims}")
		
		return ", ".join(summary_parts)

	def _request_material_info(self) -> dict:
		"""Request material information from user"""
		return {
			"action": "request_info",
			"message": "What material is this simulation for? (e.g., steel, aluminum, copper)",
			"context": self.context,
			"missing": "material_type"
		}

	def _request_geometry_info(self) -> dict:
		"""Request geometry information from user"""
		return {
			"action": "request_info",
			"message": "What geometry are you simulating? (e.g., beam, plate, cylinder, cube)",
			"context": self.context,
			"missing": "geometry_type"
		}

	def _request_dimensions_info(self) -> dict:
		"""Request dimension information from user"""
		geometry_type = self.context.get('geometry_type', 'object')
		return {
			"action": "request_info",
			"message": f"What are the dimensions of the {geometry_type}? (e.g., length, width, height, radius)",
			"context": self.context,
			"missing": "geometry_dimensions"
		}

	def _request_boundary_conditions_info(self) -> dict:
		"""Request boundary conditions from user"""
		return {
			"action": "request_info",
			"message": "What are the boundary conditions? (e.g., fixed, free, temperature, force)",
			"context": self.context,
			"missing": "boundary_conditions"
		}

	def _get_next_steps(self, missing_items: list) -> list:
		"""Get next steps based on missing items"""
		next_steps = []
		
		for item in missing_items:
			if item == "material_type":
				next_steps.append("Specify the material type")
			elif item == "geometry_type":
				next_steps.append("Specify the geometry type")
			elif item.startswith("geometry_dimension_"):
				next_steps.append(f"Provide the {item.replace('geometry_dimension_', '')} dimension")
			elif item == "boundary_conditions":
				next_steps.append("Specify boundary conditions")
		
		return next_steps

	def _create_simulation_config(self) -> dict:
		"""Create complete simulation configuration"""
		try:
			return self.template_manager.create_simulation_config(
				self.context['physics_type'], self.context
			)
		except Exception as e:
			logger.error(f"Error creating simulation config: {e}")
			return {}

	def clear_context(self):
		"""Clear the simulation context"""
		self.context = {}
		logger.info("Context cleared")

	def get_context(self) -> dict:
		"""Get current simulation context"""
		return self.context.copy()

	def update_context(self, updates: dict):
		"""Update simulation context with new information"""
		self.context.update(updates)
		logger.info(f"Context updated: {list(updates.keys())}")

	def validate_context(self) -> dict:
		"""Validate current simulation context"""
		if not self.context.get('physics_type'):
			return {'valid': False, 'missing': ['physics_type']}

		completeness = self.template_manager.check_completeness(
			self.context['physics_type'], self.context
		)

		return {
			'valid': completeness['complete'],
			'missing': completeness['missing'],
			'completeness': completeness
		}

	def get_missing_information(self) -> list:
		"""Get list of missing information for simulation"""
		completeness = self.template_manager.check_completeness(
			self.context.get('physics_type', ''), self.context
		)
		return completeness.get('missing', [])

	def is_simulation_ready(self) -> bool:
		"""Check if simulation is ready to run"""
		completeness = self.template_manager.check_completeness(
			self.context.get('physics_type', ''), self.context
		)
		return completeness.get('complete', False)

	def get_simulation_summary(self) -> dict:
		"""Get summary of current simulation configuration"""
		return self.template_manager.get_simulation_summary(
			self.context.get('physics_type', ''), self.context
		)

	def set_model(self, model: str):
		"""Set the OpenAI model to use"""
		self.model = model
		self.prompt_manager.model = model
		logger.info(f"Model changed to: {model}")

# Import time for response time tracking
import time