#!/usr/bin/env python3
"""
Context-Based Natural Language Parser for Physics Simulations
"""

import logging
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from math import isfinite
from config.config_manager import config_manager

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
		
		# Load geometry keywords from config (for keyword-based detection)
		self.geometry_keywords = self._load_geometry_keywords_from_config()

	def check_completeness(self, physics_type: str, context: dict) -> dict:
		"""Check completeness of simulation context (used by ParserAgent)"""
		return self.template_manager.check_completeness(physics_type, context)

	# DEPRECATED: parse_prompt() removed - replaced by ParserAgent._parse_prompt()
	# All parsing is now done through the agentic workflow via ParserAgent

	# Removed unused methods: _is_new_simulation, _check_physics_simulation, _check_material, 
	# _check_geometry, _check_dimensions, _check_boundary_conditions, _detect_bc_for_boundary,
	# _check_external_loads, _complete_data_structure, _fetch_material_properties, 
	# _get_context_summary, _request_material_info, _request_geometry_info, 
	# _request_dimensions_info, _request_boundary_conditions_info, _get_next_steps
	# These were only used by the deprecated parse_prompt() method

	def _get_default_dimensions(self, geometry_type: str) -> dict:
		"""Get default dimensions for a geometry type"""
		default_dimensions = {
			# 1D geometries
			'line': {'length': 1.0},
			'rod': {'length': 1.0},
			'bar': {'length': 1.0},
			
			# 2D geometries
			'plate': {'length': 1.0, 'width': 0.8, 'thickness': 0.02},
			'membrane': {'length': 1.0, 'width': 0.8},
			'disc': {'radius': 0.5},
			'rectangle': {'length': 1.0, 'width': 0.8},
			'square': {'length': 1.0},
			
			# 3D geometries
			'cube': {'length': 1.0, 'width': 1.0, 'height': 1.0},  # 1x1x1 cube (all sides equal)
			'box': {'length': 1.0, 'width': 2.0, 'height': 1.0},  # 1x2x1 cuboid
			'beam': {'length': 1.0, 'width': 0.1, 'height': 0.1},
			'cylinder': {'radius': 0.5, 'length': 1.0},  # 1m length, 0.5m radius
			'sphere': {'radius': 0.5},
			'solid': {'length': 1.0, 'width': 1.0, 'height': 1.0},
			'rectangular': {'length': 1.0, 'width': 2.0, 'height': 1.0},
		}
		
		# Normalize geometry type to lowercase
		geometry_type = geometry_type.lower() if geometry_type else ''
		
		return default_dimensions.get(geometry_type, {})
	

	def _convert_dimension_units(self, dimensions: dict, units: dict) -> dict:
		"""Convert dimension units to meters (base unit) and apply alias mappings."""
		converted = {}
		
		for dim_name, value in dimensions.items():
			if value is None:
				converted[dim_name] = None
				continue
				
			try:
				num_value = float(value)
				unit = (units.get(dim_name) or 'm').lower()
				
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
					pass  # already meters
				else:
					logger.warning(f"Unknown unit '{unit}' for dimension '{dim_name}', assuming meters")
				
				converted[dim_name] = num_value
				logger.debug(f"Converted {dim_name}: {value} {unit} -> {num_value} m")
				
			except (ValueError, TypeError) as e:
				logger.warning(f"Could not convert dimension '{dim_name}' value '{value}': {e}")
				converted[dim_name] = value
		
		return self._apply_dimension_aliases(converted)

	def _apply_dimension_aliases(self, dimensions: dict) -> dict:
		"""Apply alias mappings from configuration (e.g., diameter → radius)."""
		alias_config = self._get_dimension_aliases()
		if not alias_config:
			return dimensions
		
		resolved = dict(dimensions)
		
		# First pass: resolve aliases for missing canonical dimensions
		for canonical, aliases in alias_config.items():
			current_value = resolved.get(canonical)
			# Only resolve if canonical is missing or zero
			if current_value not in (None, '', 0):
				continue
			
			for alias in aliases:
				value = self._evaluate_dimension_alias(alias, resolved)
				if value is not None:
					resolved[canonical] = value
					logger.debug(f"Resolved alias '{alias}' to '{canonical}' with value {value}")
					break
		
		# Second pass: if we have both diameter and radius, prefer radius (remove diameter)
		# This handles cases where both were extracted but we only need radius
		if 'diameter' in resolved and 'radius' in resolved:
			# If radius was resolved from diameter, remove diameter to avoid confusion
			if resolved.get('radius') is not None and resolved.get('diameter') is not None:
				# Check if radius equals diameter/2 (within tolerance)
				diameter_val = self._to_float(resolved.get('diameter'))
				radius_val = self._to_float(resolved.get('radius'))
				if diameter_val is not None and radius_val is not None:
					expected_radius = diameter_val / 2.0
					if abs(radius_val - expected_radius) < 1e-6:
						logger.debug(f"Removing 'diameter' from dimensions (radius={radius_val} already computed from diameter={diameter_val})")
						del resolved['diameter']
		
		return resolved

	def _evaluate_dimension_alias(self, alias: str, values: dict) -> Optional[float]:
		"""Evaluate a single alias expression against available dimension values."""
		if not alias:
			return None
		
		alias_clean = alias.strip()
		
		# Direct lookup
		if alias_clean in values:
			return self._to_float(values.get(alias_clean))
		
		# Expressions like "diameter/2"
		import re
		expr_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*/\s*([0-9.]+)$', alias_clean)
		if expr_match:
			base_key = expr_match.group(1)
			denominator = float(expr_match.group(2))
			base_val = self._to_float(values.get(base_key))
			if base_val is not None and isfinite(base_val) and denominator != 0:
				return base_val / denominator
			return None
		
		return None

	def _to_float(self, value: Any) -> Optional[float]:
		"""Safely convert a value to float."""
		if value in (None, '', []):
			return None
		try:
			return float(value)
		except (TypeError, ValueError):
			try:
				return float(str(value).strip())
			except (TypeError, ValueError):
				return None

	def _get_dimension_aliases(self) -> Dict[str, Any]:
		"""Fetch dimension alias mapping from configuration."""
		if not hasattr(self, "_dimension_aliases_cache"):
			dim_config = config_manager.get_dimensions_config()
			self._dimension_aliases_cache = dim_config.get('dimension_aliases', {})
		return self._dimension_aliases_cache

	def _extract_and_remove_parsed_sections(
		self, 
		prompt: str, 
		geometry_type: Optional[str] = None,
		physics_type: Optional[str] = None,
		material_type: Optional[str] = None,
		dimensions: Optional[dict] = None
	) -> str:
		"""
		Extract and remove parsed sections (geometry, physics, material) from prompt.
		This leaves only boundary condition information in the cleaned prompt.
		
		Args:
			prompt: Original prompt text
			geometry_type: Detected geometry type
			physics_type: Detected physics type
			material_type: Detected material type
			dimensions: Detected dimensions dictionary
		
		Returns:
			Cleaned prompt with geometry/physics/material sections removed
		"""
		import re
		
		cleaned = prompt
		
		# Remove geometry-related text
		if geometry_type:
			# Remove geometry type mentions
			geometry_pattern = re.compile(r'\b' + re.escape(geometry_type) + r'\b', re.IGNORECASE)
			cleaned = geometry_pattern.sub('', cleaned)
			
			# Remove geometry keywords (wire, rod, bar, beam, plate, cylinder, cube, etc.)
			geometry_keywords = ['wire', 'rod', 'bar', 'beam', 'plate', 'cylinder', 'cube', 'box', 'sphere', 'block']
			for keyword in geometry_keywords:
				keyword_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
				cleaned = keyword_pattern.sub('', cleaned)
		
		# Remove dimension-related text
		if dimensions:
			# Remove dimension values and units (e.g., "1m", "60cm", "diameter 60cm")
			dimension_patterns = [
				r'\b\d+\.?\d*\s*(m|cm|mm|km|inch|inches|in)\b',  # Simple dimension values
				r'\b(length|width|height|thickness|radius|diameter|d|r|l|w|h)\s*[:=]?\s*\d+\.?\d*\s*(m|cm|mm|km|inch|inches|in)?\b',  # Named dimensions
				r'\b\d+\.?\d*\s*(m|cm|mm|km|inch|inches|in)\s*(long|wide|tall|thick|diameter|radius)\b',  # Dimension with descriptor
			]
			for pattern in dimension_patterns:
				dim_pattern = re.compile(pattern, re.IGNORECASE)
				cleaned = dim_pattern.sub('', cleaned)
		
		# Remove physics-related text (but be careful not to remove BC values)
		if physics_type:
			# Remove physics type mentions, but NOT temperature/force values
			physics_keywords = {
				'heat_transfer': ['heat transfer', 'thermal conduction', 'thermal convection'],  # Removed 'temperature' to preserve BC values
				'solid_mechanics': ['mechanics', 'stress analysis', 'strain analysis', 'deformation analysis']  # Removed 'force' and 'load' to preserve BC values
			}
			keywords = physics_keywords.get(physics_type, [])
			for keyword in keywords:
				keyword_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
				cleaned = keyword_pattern.sub('', cleaned)
		
		# Remove material-related text
		if material_type:
			# Remove material type mentions
			material_pattern = re.compile(r'\b' + re.escape(material_type) + r'\b', re.IGNORECASE)
			cleaned = material_pattern.sub('', cleaned)
			
			# Remove common material keywords
			material_keywords = ['copper', 'steel', 'aluminum', 'aluminium', 'iron', 'titanium', 'brass', 'bronze']
			for keyword in material_keywords:
				keyword_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
				cleaned = keyword_pattern.sub('', cleaned)
		
		# Clean up extra whitespace and punctuation
		cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
		cleaned = re.sub(r'\s*,\s*,', ',', cleaned)  # Remove double commas
		cleaned = re.sub(r'^\s*[,.\s]+|[,.\s]+\s*$', '', cleaned)  # Remove leading/trailing punctuation
		cleaned = cleaned.strip()
		
		return cleaned if cleaned else prompt  # Return original if everything was removed
	
	
	def _add_placeholder_boundary_conditions(
		self, detected_bcs: list, geometry_type: str, physics_type: str, available_boundaries: list
	) -> list:
		"""
		Add placeholder boundary conditions for unspecified boundaries.
		This ensures all boundaries for the geometry type have BCs defined.
		CRITICAL: Preserves user-modified boundary conditions - never overwrites them with placeholders.
		VALIDATES: Filters out BC locations that are not valid for the geometry type.
		
		Args:
			detected_bcs: List of detected boundary conditions (can be empty list or None)
			geometry_type: The geometry type
			physics_type: The physics type
			available_boundaries: List of available boundaries for the geometry type
		
		Returns:
			Complete list of boundary conditions with placeholders for all boundaries
		"""
		if not available_boundaries:
			logger.warning(f"No available boundaries for {geometry_type}, cannot add placeholders")
			return detected_bcs if detected_bcs else []
		
		# Ensure detected_bcs is a list (can be None or empty list)
		detected_bcs = detected_bcs or []
		
		# CRITICAL: Filter and match BC locations to geometry-specific boundaries using confidence scores
		# Only keep BCs with locations that are in available_boundaries or can be mapped to them
		# Use confidence scores from boundary location analysis to select best matches
		valid_detected_bcs = []
		available_boundaries_lower = [b.lower() for b in available_boundaries]
		
		# Track which available boundaries have been matched (to avoid duplicates)
		matched_boundaries = set()
		
		for bc in detected_bcs:
			if not bc or not bc.get('location'):
				continue
			
			location = bc.get('location', '').strip()
			location_lower = location.lower().strip()
			confidence = bc.get('confidence', 0.5)  # Get confidence score (default 0.5)
			
			# Check if location is directly valid (exact match)
			if location in available_boundaries:
				if location not in matched_boundaries:
					matched_boundaries.add(location)
					valid_detected_bcs.append(bc)
					logger.debug(f"BC location '{location}' matches exactly (confidence: {confidence})")
				else:
					logger.debug(f"BC location '{location}' already matched, skipping duplicate")
			elif location_lower in available_boundaries_lower:
				# Case-insensitive match - find the exact boundary name
				exact_match = None
				for avail in available_boundaries:
					if avail.lower() == location_lower:
						exact_match = avail
						break
				if exact_match and exact_match not in matched_boundaries:
					matched_boundaries.add(exact_match)
					bc_copy = bc.copy()
					bc_copy['location'] = exact_match
					valid_detected_bcs.append(bc_copy)
					logger.debug(f"BC location '{location}' matched case-insensitively to '{exact_match}' (confidence: {confidence})")
				else:
					logger.debug(f"BC location '{location}' already matched, skipping duplicate")
			else:
				# Try to map the location to a valid one using geometry-specific mappings
				mapped_location = self._map_boundary_location(location, geometry_type, available_boundaries)
				if mapped_location:
					# Use confidence score to decide if this mapping is acceptable
					# Lower confidence threshold (0.3) allows mappings but filters very uncertain ones
					if mapped_location not in matched_boundaries:
						matched_boundaries.add(mapped_location)
						bc_copy = bc.copy()
						bc_copy['location'] = mapped_location
						# Preserve original confidence or use mapping confidence
						bc_copy['confidence'] = confidence
						valid_detected_bcs.append(bc_copy)
						logger.info(f"Mapped BC location '{location}' to '{mapped_location}' for {geometry_type} (confidence: {confidence})")
					else:
						logger.debug(f"Mapped location '{mapped_location}' already matched, skipping duplicate for '{location}'")
				else:
					# Location cannot be mapped - only skip if confidence is very low
					if confidence < 0.2:
						logger.warning(f"BC location '{location}' is not valid for {geometry_type}, cannot be mapped, and confidence is low ({confidence}), skipping")
					else:
						# Even if unmappable, warn but could potentially add as-is if needed
						logger.warning(f"BC location '{location}' is not valid for {geometry_type} and cannot be mapped (confidence: {confidence}), skipping")
		
		logger.debug(f"Filtered {len(detected_bcs)} detected BCs to {len(valid_detected_bcs)} valid BCs for {geometry_type} using confidence scores")
		
		# Get list of boundaries that already have BCs
		# CRITICAL: Check for user-modified BCs (source='user' or is_user_modified=True)
		# These should NEVER be overwritten with placeholders
		specified_boundaries = {}
		user_modified_boundaries = set()
		
		for bc in valid_detected_bcs:
			if bc and bc.get('location'):
				location = bc.get('location')
				specified_boundaries[location] = bc
				# Check if this BC is user-modified
				if bc.get('source') == 'user' or bc.get('is_user_modified') or not bc.get('is_placeholder'):
					# If source is not 'placeholder' and not explicitly marked as placeholder, treat as user-modified
					if bc.get('source') != 'placeholder':
						user_modified_boundaries.add(location)
						logger.debug(f"Boundary {location} has user-modified BC, will preserve it")
		
		logger.debug(f"Adding placeholders for {geometry_type}: {len(valid_detected_bcs)} valid detected BC(s), {len(available_boundaries)} available boundaries, {len(specified_boundaries)} specified, {len(user_modified_boundaries)} user-modified")
		
		# Get default BC type from config for unspecified boundaries
		default_bc_config = None
		if self.geometry_boundaries:
			default_bc_config = self.geometry_boundaries.get('default_boundary_conditions', {}).get(physics_type, {})
		
		# Set default BC type and value based on physics type
		if default_bc_config:
			default_type = default_bc_config.get('type', 'free')
			default_value = default_bc_config.get('value', 0)
			default_bc_type = default_bc_config.get('bc_type', 'neumann')
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
				default_type = 'free'
				default_value = 0
				default_bc_type = 'neumann'
		
		logger.debug(f"Default BC for {physics_type}: {default_type} (value: {default_value}, bc_type: {default_bc_type})")
		
		# Create complete BC list starting with valid detected BCs (preserve all user-modified BCs)
		complete_bcs = valid_detected_bcs.copy() if valid_detected_bcs else []
		
		# Add placeholders ONLY for boundaries that don't have BCs (including user-modified ones)
		for boundary in available_boundaries:
			if boundary not in specified_boundaries:
				# Add placeholder BC for this boundary
				placeholder_bc = {
					'location': boundary,
					'type': default_type,
					'bc_type': default_bc_type,
					'value': default_value,
					'confidence': 0.5,
					'source': 'placeholder',
					'is_placeholder': True  # Flag to indicate this is a placeholder
				}
				complete_bcs.append(placeholder_bc)
				logger.debug(f"Added placeholder BC for {boundary}: {default_type} (value: {default_value})")
			elif boundary in user_modified_boundaries:
				logger.debug(f"Boundary {boundary} has user-modified BC, preserving it (not overwriting with placeholder)")
			else:
				logger.debug(f"Boundary {boundary} already has a BC, skipping placeholder")
		
		logger.info(f"Added {len(complete_bcs) - len(valid_detected_bcs)} placeholder BC(s) for {geometry_type}, total: {len(complete_bcs)} BC(s) ({len(user_modified_boundaries)} user-modified preserved, {len(detected_bcs) - len(valid_detected_bcs)} invalid locations filtered out)")
		return complete_bcs

	def _load_geometry_boundaries(self) -> dict:
		"""Load geometry boundaries configuration from JSON file"""
		import json
		import os
		from pathlib import Path
		
		# Try to find config file relative to project root
		# From nlp_parser/src/context_based_parser.py, go up 3 levels to project root
		current_dir = Path(__file__).parent.parent.parent
		config_path = current_dir / "config" / "geometry_boundaries.json"
		
		if config_path.exists():
			with open(config_path, 'r') as f:
				return json.load(f)
		else:
			logger.warning(f"Geometry boundaries config not found at {config_path}, using defaults")
			return {}
	
	def _load_geometry_keywords_from_config(self) -> dict:
		"""Load geometry keywords from config files instead of hardcoding"""
		import json
		from pathlib import Path
		
		try:
			current_dir = Path(__file__).parent.parent.parent
			
			# Load geometry types from dimensions.json (geometry_dimensions section)
			dimensions_path = current_dir / "config" / "dimensions.json"
			geometry_keywords = {}
			geometry_dimension_map = {}  # Map geometry type to dimension group
			
			if dimensions_path.exists():
				with open(dimensions_path, 'r') as f:
					dim_data = json.load(f)
				
				# First, get all geometry types from geometry_dimensions
				geom_dims = dim_data.get('geometry_dimensions', {})
				for dim_group in ['1D', '2D', '3D']:
					if dim_group in geom_dims:
						for geometry_type in geom_dims[dim_group].keys():
							geometry_keywords[geometry_type] = [geometry_type]
							geometry_dimension_map[geometry_type] = dim_group
				
				# Then, add aliases from geometry_type_aliases
				aliases = dim_data.get('geometry_type_aliases', {})
				for dim_group in ['1D', '2D', '3D']:
					if dim_group in aliases:
						for geometry_type, synonym_list in aliases[dim_group].items():
							# If geometry type exists, add synonyms to it
							if geometry_type in geometry_keywords:
								geometry_keywords[geometry_type].extend(synonym_list)
								# Ensure dimension is mapped
								if geometry_type not in geometry_dimension_map:
									geometry_dimension_map[geometry_type] = dim_group
							else:
								# If geometry type doesn't exist in geometry_dimensions, create entry
								# This handles cases where aliases reference types not in geometry_dimensions
								geometry_keywords[geometry_type] = [geometry_type] + list(synonym_list)
								geometry_dimension_map[geometry_type] = dim_group
			
			# Also load from geometry_boundaries.json to get all available geometries
			# This ensures we have all geometries even if they're missing from dimensions.json
			boundaries_path = current_dir / "config" / "geometry_boundaries.json"
			if boundaries_path.exists():
				with open(boundaries_path, 'r') as f:
					boundaries_data = json.load(f)
				
				geometries = boundaries_data.get('geometries', {})
				# Handle both lowercase ('1d', '2d', '3d') and uppercase ('1D', '2D', '3D') formats
				for dim_group_lower in ['1d', '2d', '3d']:
					# Normalize to uppercase for consistency
					dim_group = dim_group_lower.upper()
					if dim_group_lower in geometries:
						for geometry_type in geometries[dim_group_lower].keys():
							# Ensure geometry type exists in keywords
							if geometry_type not in geometry_keywords:
								geometry_keywords[geometry_type] = [geometry_type]
								logger.debug(f"Added geometry type '{geometry_type}' from geometry_boundaries.json")
							# Map dimension group (use uppercase for consistency)
							if geometry_type not in geometry_dimension_map:
								geometry_dimension_map[geometry_type] = dim_group
			
			# Add dimension prefix variants (e.g., "1d rod", "2d plate")
			for geometry_type, keywords in geometry_keywords.items():
				# Get dimension group from map (loaded from config files)
				dim_group = geometry_dimension_map.get(geometry_type)
				
				# If not found in map, try to determine from geometry type name
				if not dim_group:
					if geometry_type in ['line', 'rod', 'bar']:
						dim_group = '1D'
					elif geometry_type in ['plate', 'membrane', 'disc', 'rectangle', 'square']:
						dim_group = '2D'
					elif geometry_type in ['cube', 'box', 'beam', 'cylinder', 'sphere']:
						dim_group = '3D'
				
				# Add dimension prefix variants
				if dim_group:
					prefix_map = {'1D': ['1d', '1-d'], '2D': ['2d', '2-d'], '3D': ['3d', '3-d']}
					for prefix in prefix_map.get(dim_group, []):
						# Add prefixes to geometry type and first few keywords
						for keyword in [geometry_type] + keywords[:5]:  # Limit to avoid too many variations
							prefixed_keyword = f'{prefix} {keyword}'
							if prefixed_keyword not in keywords:
								keywords.append(prefixed_keyword)
					
					# Remove duplicates and sort by length (longest first)
					geometry_keywords[geometry_type] = sorted(list(set(keywords)), key=len, reverse=True)
			
			logger.debug(f"Loaded {len(geometry_keywords)} geometry types from config")
			return geometry_keywords
			
		except Exception as e:
			logger.warning(f"Failed to load geometry keywords from config: {e}, using defaults")
			return self._get_default_geometry_keywords()
	
	def _get_default_geometry_keywords(self) -> dict:
		"""Fallback default geometry keywords if config file is not available"""
		return {
			# 1D geometries
			'rod': ['1d rod', '1-d rod', 'rod'],
			'bar': ['1d bar', '1-d bar', 'bar'],
			'line': ['1d line', '1-d line', '1d', 'line'],
			# 2D geometries
			'plate': ['2d plate', '2-d plate', 'plate'],
			'rectangle': ['2d rectangle', '2-d rectangle', 'rectangle'],
			'square': ['2d square', '2-d square', 'square'],
			'disc': ['2d circle', '2-d circle', 'disc', 'disk', 'circle'],
			'membrane': ['2d membrane', '2-d membrane', 'membrane'],
			# 3D geometries
			'cube': ['3d cube', '3-d cube', 'cube'],
			'box': ['3d box', '3-d box', 'rectangular solid', 'box', 'rectangular'],
			'beam': ['3d beam', '3-d beam', 'beam'],
			'cylinder': ['3d cylinder', '3-d cylinder', 'cylinder', 'pipe', 'tube'],
			'sphere': ['3d sphere', '3-d sphere', 'sphere', 'ball'],
		}

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
			logger.debug("All boundaries already specified, skipping defaults")
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
				logger.debug(f"Added default {default_type} BC on {boundary}")
		
		return result

	def _get_available_boundaries(self, geometry_type: str) -> list:
		"""Get available boundaries for a geometry type from configuration"""
		if not geometry_type:
			return []
		
		# Normalize geometry type to lowercase for config lookup
		geometry_type_lower = geometry_type.lower()
		
		if not self.geometry_boundaries:
			# Fallback if config not loaded
			defaults = {
				'line': ['left', 'right'], 'rod': ['left', 'right'], 'bar': ['left', 'right'],
				'plate': ['left', 'right', 'top', 'bottom'], 'membrane': ['left', 'right', 'top', 'bottom'],
				'rectangle': ['left', 'right', 'top', 'bottom'], 'square': ['left', 'right', 'top', 'bottom'],
				'disc': ['circumference', 'center'],
				'cube': ['left', 'right', 'top', 'bottom', 'front', 'back'],
				'box': ['left', 'right', 'top', 'bottom', 'front', 'back'],
				'beam': ['left', 'right', 'top', 'bottom', 'front', 'back'],
				'cylinder': ['top', 'bottom', 'curved surface'],
				'sphere': ['center', 'surface']
			}
			return defaults.get(geometry_type_lower, ['left', 'right', 'top', 'bottom'])
		
		# Search in all dimension categories (config uses lowercase keys)
		for dim_group in ['1d', '2d', '3d']:
			if dim_group in self.geometry_boundaries.get('geometries', {}):
				# Config keys are lowercase, so use lowercase geometry type
				if geometry_type_lower in self.geometry_boundaries['geometries'][dim_group]:
					boundaries = self.geometry_boundaries['geometries'][dim_group][geometry_type_lower].get('available_boundaries', [])
					if boundaries:
						logger.debug(f"Found boundaries for {geometry_type} (normalized: {geometry_type_lower}) in {dim_group}: {boundaries}")
						return boundaries
		
		logger.warning(f"No boundaries found in config for geometry type: {geometry_type} (normalized: {geometry_type_lower})")
		return ['left', 'right', 'top', 'bottom']  # default fallback
	
	def _map_boundary_location(self, location: str, geometry_type: str, available_boundaries: list) -> Optional[str]:
		"""
		Map a vague boundary location to a specific boundary for the geometry type.
		Uses location mappings from geometry_boundaries.json config file.
		"""
		if not location or not available_boundaries:
			return None
		
		location_lower = location.lower().strip()
		geometry_type_lower = geometry_type.lower().strip() if geometry_type else ''
		
		# Geometry-specific mappings from configuration
		if self.geometry_boundaries:
			geo_specific = self.geometry_boundaries.get('geometry_specific_location_mappings', {})
			if geometry_type_lower in geo_specific:
				mapped = self._match_location_with_map(location_lower, geo_specific[geometry_type_lower], available_boundaries)
				if mapped:
					return mapped
		
		# Use location mappings from geometry_boundaries.json config file
		if self.geometry_boundaries:
			mappings = self.geometry_boundaries.get('location_mappings', {}).get('vague_to_specific', {})
			
			# Direct lookup
			if location_lower in mappings:
				mapped = mappings[location_lower]
				if mapped in available_boundaries:
					return mapped
			
			# Partial match: check if location contains any mapping key
			for vague_key, specific_boundary in mappings.items():
				if vague_key in location_lower:
					if specific_boundary in available_boundaries:
						return specific_boundary
			
			# Reverse: check if any mapping value matches location
			for vague_key, specific_boundary in mappings.items():
				if location_lower in vague_key or vague_key in location_lower:
					if specific_boundary in available_boundaries:
						return specific_boundary
		
		# Fallback: check if location already matches an available boundary (case-insensitive)
		for avail_bound in available_boundaries:
			if location_lower == avail_bound.lower() or location_lower in avail_bound.lower() or avail_bound.lower() in location_lower:
				return avail_bound
		
		return None

	def _match_location_with_map(self, location_lower: str, mapping: Dict[str, str], available_boundaries: list) -> Optional[str]:
		"""Match a boundary location against a provided mapping dictionary."""
		if not mapping:
			return None
		
		# Exact match
		if location_lower in mapping:
			mapped = mapping[location_lower]
			if mapped in available_boundaries:
				return mapped
		
		# Partial match (substring)
		for vague_key, target in mapping.items():
			if vague_key in location_lower or location_lower in vague_key:
				if target in available_boundaries:
					return target
		
		return None


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
		logger.debug("Context cleared")

	def get_context(self) -> dict:
		"""Get current simulation context"""
		return self.context.copy()

	def update_context(self, updates: dict):
		"""Update simulation context with new information"""
		self.context.update(updates)
		logger.debug(f"Context updated: {list(updates.keys())}")

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
		logger.debug(f"Model changed to: {model}")