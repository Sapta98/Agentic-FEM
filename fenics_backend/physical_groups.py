"""
Physical group resolution and boundary location utilities
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
import dolfinx

logger = logging.getLogger(__name__)


def load_geometry_boundaries_config() -> dict:
	"""Load geometry_boundaries.json config file"""
	try:
		# Find config file relative to project root
		current_dir = Path(__file__).parent.parent
		config_path = current_dir / "config" / "geometry_boundaries.json"
		
		if config_path.exists():
			with open(config_path, 'r') as f:
				config = json.load(f)
				logger.debug(f"Loaded geometry_boundaries config from {config_path}")
				return config
		else:
			logger.warning(f"Geometry boundaries config not found at {config_path}, using empty config")
			return {}
	except Exception as e:
		logger.warning(f"Error loading geometry_boundaries config: {e}, using empty config")
		return {}

def generate_name_variations(name: str) -> list:
	"""Generate all possible variations of a physical group name for fuzzy matching"""
	if not name:
		return []
	
	variations = [name]
	name_lower = name.lower()
	
	# Add/remove common suffixes
	if not name_lower.endswith("_boundary"):
		variations.append(f"{name}_boundary")
		variations.append(f"{name_lower}_boundary")
	if name_lower.endswith("_boundary"):
		base = name_lower.replace("_boundary", "")
		variations.append(base)
		variations.append(name.replace("_boundary", ""))
	
	# Remove underscores
	variations.append(name.replace("_", ""))
	variations.append(name_lower.replace("_", ""))
	
	# Add/remove "boundary" suffix (without underscore)
	if "boundary" not in name_lower:
		variations.append(f"{name}boundary")
		variations.append(f"{name_lower}boundary")
	if "boundary" in name_lower and not name_lower.endswith("_boundary"):
		base = name_lower.replace("boundary", "")
		variations.append(base)
	
	# Case variations
	variations.append(name_lower)
	variations.append(name.upper())
	variations.append(name.title())
	
	# Remove duplicates and empty strings
	variations = list(set([v for v in variations if v]))
	return variations


def normalize_physical_group_keys(physical_groups: dict) -> dict:
	"""Create a dict keyed by normalized name variations for robust lookup."""
	if not physical_groups:
		return {}
	normalized = {}
	for name, group in physical_groups.items():
		if not isinstance(name, str):
			continue
		for variant in generate_name_variations(name) + [name]:
			key = variant.lower().strip()
			if key:
				normalized[key] = group
	return normalized


def find_physical_group_by_location(location: str, physical_groups: dict, fdim: int = None) -> tuple:
	"""
	Find physical group that matches the given location using fuzzy matching.
	
	Returns: (physical_group, tag, confidence_score)
	"""
	if not location or not physical_groups:
		return (None, None, 0.0)
	
	location_lower = location.lower().strip()
	
	# DEBUG: Log what we're looking for
	logger.debug(f"find_physical_group_by_location: looking for '{location}' (lower: '{location_lower}') in physical_groups with keys: {list(physical_groups.keys())}")
	
	# Create a lowercase-keyed dict for case-insensitive lookup (like resolve_via_physical_groups)
	physical_groups_lower = {k.lower().strip(): (k, v) for k, v in physical_groups.items()}
	logger.debug(f"find_physical_group_by_location: physical_groups_lower keys: {list(physical_groups_lower.keys())}")
	
	# Helper to check dimension compatibility (more lenient)
	def dim_matches(group, expected_fdim):
		if expected_fdim is None:
			return True
		# Handle both dict and object types
		if isinstance(group, dict):
			group_dim = group.get('dim')
		else:
			if not hasattr(group, 'dim'):
				# If group doesn't have dim attribute, accept it (assume it's correct)
				return True
			group_dim = getattr(group, 'dim', None)
		if group_dim is None:
			# If dim is None, accept it
			return True
		# Exact match or accept if dimension is reasonable (2D facets for 3D mesh)
		return group_dim == expected_fdim
	
	# Step 1: Direct match (highest confidence) - case-insensitive
	if location_lower in physical_groups_lower:
		original_key, group = physical_groups_lower[location_lower]
		logger.debug(f"find_physical_group_by_location: Found match! original_key='{original_key}', group type={type(group)}")
		# For center and other point groups, don't enforce dimension check
		# For Neumann BCs on surfaces/curves, dimension should match fdim
		center_aliases = {"center", "middle", "origin", "core"}
		if location_lower in center_aliases or dim_matches(group, fdim):
			# Get tag - handle both dict and object
			if isinstance(group, dict):
				tag = group.get('tag')
				logger.debug(f"find_physical_group_by_location: group is dict, tag={tag}")
			else:
				tag = getattr(group, 'tag', None)
				# Also try _data dict if tag is None
				if tag is None and hasattr(group, '_data') and isinstance(group._data, dict):
					tag = group._data.get('tag')
				logger.debug(f"find_physical_group_by_location: group is object, tag={tag}")
			if tag is None:
				logger.error(f"find_physical_group_by_location: tag is None for '{location}'! group={group}, group type={type(group)}")
				# Try to get tag from _data if available
				if hasattr(group, '_data') and isinstance(group._data, dict):
					tag = group._data.get('tag')
					logger.debug(f"find_physical_group_by_location: Retrieved tag from _data: {tag}")
			if tag is not None:
				return (group, tag, 1.0)
			else:
				logger.error(f"find_physical_group_by_location: Could not extract tag from group for '{location}'")
	else:
		logger.error(f"find_physical_group_by_location: '{location_lower}' NOT FOUND in physical_groups_lower keys: {list(physical_groups_lower.keys())}")
	
	# Step 2: Try variations (medium confidence)
	variations = generate_name_variations(location)
	for variant in variations:
		variant_lower = variant.lower().strip()
		if variant_lower in physical_groups_lower:
			original_key, group = physical_groups_lower[variant_lower]
			if dim_matches(group, fdim):
				tag = group.get('tag') if isinstance(group, dict) else getattr(group, 'tag', None)
				return (group, tag, 0.8)
	
	# Step 3: Try all physical group names with variations
	for pg_name_lower, (original_key, group) in physical_groups_lower.items():
		pg_variations = generate_name_variations(original_key)
		if location_lower in [v.lower().strip() for v in pg_variations]:
			if dim_matches(group, fdim):
				tag = group.get('tag') if isinstance(group, dict) else getattr(group, 'tag', None)
				return (group, tag, 0.7)
	
	# Step 4: Fuzzy string matching (lower confidence) - simple substring match
	for pg_name_lower, (original_key, group) in physical_groups_lower.items():
		# Check if location is substring of pg_name or vice versa
		if location_lower in pg_name_lower or pg_name_lower in location_lower:
			if dim_matches(group, fdim):
				tag = group.get('tag') if isinstance(group, dict) else getattr(group, 'tag', None)
				return (group, tag, 0.6)
	
	return (None, None, 0.0)


def map_location_via_config(location: str, geometry_type: str) -> str:
	"""
	Map user location to canonical form using geometry_boundaries.json config.
	Returns the mapped location or original if no mapping found.
	"""
	if not location or not geometry_type:
		return location
	
	config = load_geometry_boundaries_config()
	if not config:
		return location
	
	location_lower = location.lower().strip()
	geometry_type_lower = geometry_type.lower()
	
	# Try geometry-specific mappings first
	geo_specific = config.get('geometry_specific_location_mappings', {})
	if geometry_type_lower in geo_specific:
		mappings = geo_specific[geometry_type_lower]
		if location_lower in mappings:
			mapped = mappings[location_lower]
			logger.debug(f"Mapped '{location}' -> '{mapped}' via geometry-specific config for {geometry_type}")
			return mapped
	
	# Try general location mappings
	general_mappings = config.get('location_mappings', {}).get('vague_to_specific', {})
	if location_lower in general_mappings:
		mapped = general_mappings[location_lower]
		logger.debug(f"Mapped '{location}' -> '{mapped}' via general config")
		return mapped
	
	return location


def find_closest_dof_to_point(V, point: np.ndarray) -> np.ndarray:
	"""Find closest DOF to a point (for point constraints like 'center')
	
	Uses mesh vertices directly to avoid MPI communication issues with tabulate_dof_coordinates().
	For P1 elements, DOF index = vertex index, so this is exact.
	For P2+ elements, this finds the closest vertex DOF (which is still a valid approximation).
	"""
	try:
		# Use mesh geometry directly to avoid MPI communication issues
		# This is safer and works for both P1 and P2 elements
		mesh = V.mesh
		gdim = point.shape[0]
		vertex_coords = mesh.geometry.x[:, :gdim]
		
		# Find closest vertex
		tree = cKDTree(vertex_coords)
		distances, vertex_indices = tree.query(point, k=1)
		
		# For P1 elements, DOF index = vertex index
		# For P2 elements, vertex DOFs are the first N DOFs where N = num_vertices
		# So vertex index = DOF index for vertex DOFs
		if np.isscalar(vertex_indices):
			return np.array([int(vertex_indices)], dtype=np.int32)
		else:
			return np.array([int(vertex_indices)], dtype=np.int32)
	except Exception as e:
		logger.error(f"Error finding closest DOF to point {point}: {e}")
		# Last resort: try tabulate_dof_coordinates with error handling
		try:
			from mpi4py import MPI
			# Check if function space communicator is safe
			mesh_comm = getattr(V.mesh, 'comm', None)
			if mesh_comm is not None:
				comm_size = mesh_comm.Get_size()
				if comm_size > 1:
					logger.warning(f"Function space mesh has {comm_size} processes, using vertex fallback")
					raise RuntimeError("Multi-process communicator detected")
			
			dof_coords = V.tabulate_dof_coordinates()
			if dof_coords.size == 0:
				raise RuntimeError("No DOF coordinates returned")
			
			gdim = point.shape[0]
			dof_coords = dof_coords.reshape((-1, gdim))
			tree = cKDTree(dof_coords)
			distances, indices = tree.query(point, k=1)
			return np.array([int(indices)], dtype=np.int32) if np.isscalar(indices) else np.array(indices, dtype=np.int32).flatten()
		except Exception as e2:
			logger.error(f"All methods failed for finding DOF to point: {e2}")
			return np.array([], dtype=np.int32)


def resolve_via_physical_groups(
	target_location: str,
	physical_groups: dict,
	V,
	mesh,
	fdim: int,
	gdim: int,
	method_label: str,
	geometry_type: str = None,
	original_mesh_data: dict = None,
) -> Optional[Tuple[np.ndarray, str, float]]:
	"""Shared helper to resolve DOFs using named physical groups.
	Since GMSH generates exact JSON names, try direct lookup first."""
	if not target_location or not physical_groups:
		return None
	
	location_key = target_location.lower().strip()
	center_aliases = {"center", "middle", "origin", "core"}
	dim_to_check = 0 if location_key in center_aliases else fdim
	
	# No special handling needed for "center" - GMSH now creates a physical group
	# for the closest mesh node to (0, 0, 0) after mesh generation
	
	# GMSH generates exact JSON names - simple direct lookup!
	# Try EXACT match first (case-sensitive)
	if target_location in physical_groups:
		pg_match = physical_groups[target_location]
	else:
		# Create a lowercase-keyed dict for case-insensitive lookup
		physical_groups_lower = {k.lower().strip(): v for k, v in physical_groups.items()}
		
		# Direct lookup - no fuzzy matching needed
		if location_key not in physical_groups_lower:
			logger.error(f"Boundary location '{location_key}' not found. Available: {list(physical_groups_lower.keys())}")
			return None
		
		pg_match = physical_groups_lower[location_key]
	
	if isinstance(pg_match, dict):
		tag = pg_match.get('tag')
	else:
		tag = getattr(pg_match, 'tag', None)
	
	# Simple dimension check
	if isinstance(pg_match, dict):
		group_dim = pg_match.get('dim')
	else:
		group_dim = getattr(pg_match, 'dim', None)
	logger.debug(f"Dimension check: location_key='{location_key}', dim_to_check={dim_to_check}, group_dim={group_dim}")
	if dim_to_check is not None and group_dim is not None and group_dim != dim_to_check:
		logger.error(f"Dimension mismatch: expected {dim_to_check}, got {group_dim}")
		return None
	
	# Handle point physical groups (dimension 0) - use node_tags just like surface groups
	# No special handling needed - use the same node_tags mapping approach
	
	if tag is None:
		logger.error(f"Tag is None for location_key='{location_key}'")
		return None
	
	# Use ONLY physical group node tags - NO facet_tags
	# Get node tags from physical group (stored from GMSH)
	node_tags_gmsh = None
	if isinstance(pg_match, dict):
		node_tags_gmsh = pg_match.get("node_tags", [])
		entities = pg_match.get("entities", [])
	else:
		node_tags_gmsh = getattr(pg_match, "node_tags", None) or []
		entities = getattr(pg_match, "entities", [])
		# Also try _data dict if it exists
		if hasattr(pg_match, '_data') and isinstance(pg_match._data, dict):
			node_tags_from_data = pg_match._data.get("node_tags", [])
			if node_tags_from_data and (not node_tags_gmsh or len(node_tags_gmsh) == 0):
				node_tags_gmsh = node_tags_from_data
	
	if not node_tags_gmsh or len(node_tags_gmsh) == 0:
		logger.error(f"Physical group '{target_location}' (tag={tag}): No node_tags stored. Check _make_json_safe in simulation_manager.py")
		return None
	
	# Map GMSH node tags to DOLFINx DOF indices
	# GMSH node tags are 1-based, and correspond to indices in original_mesh_data["vertices"]
	# We need to map these to DOLFINx vertex/DOF indices
	try:
		from dolfinx import fem
		
		# Get GMSH vertex coordinates from original_mesh_data
		if original_mesh_data is None or "vertices" not in original_mesh_data:
			logger.error(f"Physical group '{target_location}': original_mesh_data not available! Cannot map GMSH node tags to DOLFINx DOFs.")
			return None
		
		gmsh_vertices = np.array(original_mesh_data["vertices"])
		if len(gmsh_vertices) == 0:
			logger.error(f"Physical group '{target_location}': No vertices in original_mesh_data!")
			return None
		
		# GMSH node tags are 1-based, so node_tag N corresponds to vertex index N-1
		# Get coordinates of GMSH nodes for this physical group
		gmsh_node_coords = []
		for node_tag in node_tags_gmsh:
			# GMSH uses 1-based indexing, so node_tag 1 = vertex index 0
			vertex_idx = int(node_tag) - 1
			if 0 <= vertex_idx < len(gmsh_vertices):
				gmsh_node_coords.append(gmsh_vertices[vertex_idx][:gdim])
			else:
				logger.warning(f"  GMSH node tag {node_tag} out of range (max vertex index: {len(gmsh_vertices)-1})")
		
		if len(gmsh_node_coords) == 0:
			logger.error(f"Physical group '{target_location}': No valid GMSH node coordinates found!")
			return None

		gmsh_node_coords = np.array(gmsh_node_coords)
		
		# Get DOLFINx vertex coordinates
		dolfinx_vertices = mesh.geometry.x[:, :gdim]
		
		# Map GMSH node coordinates to DOLFINx vertices using nearest neighbor
		tree = cKDTree(dolfinx_vertices)
		distances, dolfinx_indices = tree.query(gmsh_node_coords, k=1)
		
		# Check mapping quality
		max_dist = np.max(distances)
		if max_dist > 1e-6:
			logger.warning(f"Large mapping distances for '{target_location}': max={max_dist:.2e}")
		
		# Get unique DOLFINx vertex indices
		dolfinx_vertex_indices = np.unique(dolfinx_indices)
		
		# Get DOFs from vertices
		# For P1 elements, DOF index = vertex index
		# For P2 elements, we need to get DOFs associated with these vertices
		def _needs_vertex_dofs(function_space) -> bool:
			try:
				element = function_space.ufl_element()
				value_shape = element.value_shape()
				if len(value_shape) > 0:
					return False
				try:
					degree = element.degree()
					return degree == 1
				except AttributeError:
					try:
						basix_element = function_space.basix_element
						degree = basix_element.degree
						return degree == 1
					except (AttributeError, TypeError):
						return False
			except Exception:
				return False

		# Check element degree before deciding on DOF location method
		is_p1 = _needs_vertex_dofs(V)
		
		# For both P1 and P2 Lagrange elements, vertex DOF index = vertex index
		dofs = dolfinx_vertex_indices.astype(np.int32)
		
		if dofs.size > 0:
			return (dofs, method_label, 1.0)  # Exact match = 1.0 confidence
		else:
			logger.error(f"Physical group '{target_location}': No DOFs found after mapping!")
			return None
			
	except Exception as exc:
		logger.error(f"Physical group resolution failed for '{target_location}': {exc}")
		import traceback
		logger.debug(f"Traceback: {traceback.format_exc()}")
	return None


def resolve_boundary_location(
	location: str,
	V,
	mesh,
	physical_groups: dict,
	geometry_type: str,
	pde_config: dict,
	fdim: int,
	original_mesh_data: dict = None,
) -> tuple:
	"""
	Unified boundary location resolution - ONLY uses physical groups.
	NO geometric fallback methods.
	
	Returns: (dofs: np.ndarray, method: str, confidence: float)
	Raises: RuntimeError if physical group not found
	"""
	if not location:
		raise RuntimeError(f"Empty boundary location provided")
	
	location_lower = location.lower().strip()
	gdim = int(mesh.geometry.dim)
	
	# Step 1: Try physical groups first (most reliable) - ONLY method, no facet_tags
	if physical_groups:
		result = resolve_via_physical_groups(location_lower, physical_groups, V, mesh, fdim, gdim, "physical_group", geometry_type, original_mesh_data)
		if result:
			return result
		else:
			logger.warning(f"Direct lookup failed for '{location}', trying config mapping...")
	
	# Step 2: Try config-driven mapping + physical groups
	if geometry_type:
		canonical_location = map_location_via_config(location_lower, geometry_type)
		logger.info(f"Config mapping: '{location_lower}' -> '{canonical_location}'")
		if canonical_location != location_lower and physical_groups:
			logger.info(f"Attempting to resolve boundary location '{location}' using config-mapped name '{canonical_location}'")
			result = resolve_via_physical_groups(
				canonical_location,
				physical_groups,
				V,
				mesh,
				fdim,
				gdim,
				"config_physical_group",
				geometry_type,
				original_mesh_data,
			)
			if result:
				logger.info(f"Successfully resolved boundary location '{location}' using config-mapped name '{canonical_location}'")
				return result
			else:
				logger.warning(f"Config-mapped lookup also failed for '{location}' -> '{canonical_location}'")
	
	# NO FALLBACK - Physical groups are required
	# GMSH generates exact JSON names, so show original keys
	available_groups = list(physical_groups.keys()) if physical_groups else []
	raise RuntimeError(
		f"Could not resolve boundary location '{location}' using physical groups. "
		f"Available physical groups: {available_groups}. "
		f"Physical groups MUST be provided by the mesh generator."
	)

