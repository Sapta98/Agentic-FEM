"""
FEniCS-based PDE solver for finite element simulations
"""

import os
# CRITICAL: Set environment variables BEFORE importing MPI/DOLFINx to force single-process execution
# This must be done before any MPI initialization
# Note: mpi4py is built with MPICH, so use MPICH environment variables, not OMPI
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Don't set OMPI variables if using MPICH - they cause warnings
# Instead, ensure MPI.COMM_WORLD has size 1 by not launching with mpirun/mpiexec

import logging
import os
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy.spatial import cKDTree
from petsc4py import PETSc
import dolfinx
import basix
from dolfinx import mesh
from dolfinx.cpp.mesh import create_mesh
from dolfinx.mesh import CellType
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

logger = logging.getLogger(__name__)
ScalarType = PETSc.ScalarType

class FEniCSSolver:
	"""FEniCS-based PDE solver for finite element simulations"""

	def __init__(self):
		self.mesh = None
		self.function_space = None
		self.solution = None
		self.mesh_data = None
		self.boundary_conditions = []
		self.cell_tags = None
		self.facet_tags = None
		self.msh_file_path = None  # Path to exported .msh mesh file
		self._reset_vertex_mapping_cache()
		
		# Ensure MPI is initialized for single-process execution
		# This prevents MPI rank errors when DOLFINx operations are performed
		if not MPI.Is_initialized():
			MPI.Init()
			logger.debug("MPI initialized in FEniCSSolver.__init__")
		else:
			logger.debug("MPI already initialized")
		
		# Log MPI communicator information
		comm_size = MPI.COMM_WORLD.Get_size()
		comm_rank = MPI.COMM_WORLD.Get_rank()
		logger.info(f"MPI.COMM_WORLD at initialization: size={comm_size}, rank={comm_rank}")
		
		# CRITICAL: If COMM_WORLD has size > 1, this will cause MPI errors
		# We need to ensure all operations use COMM_SELF instead
		if comm_size > 1:
			logger.warning(f"WARNING: MPI.COMM_WORLD has {comm_size} processes at initialization. All mesh operations will use COMM_SELF to avoid errors.")
		
		# Initialize PETSc for single-process execution
		try:
			import os
			# Set environment variables to force single-process execution
			os.environ.setdefault("PETSC_COMM_WORLD_SIZE", "1")
			os.environ.setdefault("OMP_NUM_THREADS", "1")
			logger.debug("Set environment variables for single-process PETSc execution")
		except Exception as e:
			logger.debug(f"Could not set environment variables: {e}")

	def set_mesh_file(self, msh_file: Optional[str]):
		"""Set path to exported .msh file for mesh creation"""
		self.msh_file_path = msh_file
		if msh_file:
			logger.debug(f"Mesh file set for FEniCS solver: {msh_file}")
		else:
			logger.warning("Mesh file cleared in FEniCS solver")

	def inspect_mesh_tags(self):
		"""Inspect and log information about cell tags and facet tags"""
		if self.cell_tags is not None:
			logger.debug("=== Cell Tags Information ===")
			logger.debug(f"Cell tags type: {type(self.cell_tags)}")
			logger.debug(f"Cell tags values shape: {self.cell_tags.values.shape}")
			logger.debug(f"Cell tags unique values: {np.unique(self.cell_tags.values)}")
			logger.debug(f"Cell tags indices shape: {self.cell_tags.indices.shape}")
			
			# Show tag distribution
			unique_tags, counts = np.unique(self.cell_tags.values, return_counts=True)
			for tag, count in zip(unique_tags, counts):
				logger.debug(f"  Tag {tag}: {count} cells")
		else:
			logger.debug("No cell tags available")
			
		if self.facet_tags is not None:
			logger.debug("=== Facet Tags Information ===")
			logger.debug(f"Facet tags type: {type(self.facet_tags)}")
			logger.debug(f"Facet tags values shape: {self.facet_tags.values.shape}")
			logger.debug(f"Facet tags unique values: {np.unique(self.facet_tags.values)}")
			logger.debug(f"Facet tags indices shape: {self.facet_tags.indices.shape}")
			
			# Show tag distribution
			unique_tags, counts = np.unique(self.facet_tags.values, return_counts=True)
			for tag, count in zip(unique_tags, counts):
				logger.debug(f"  Tag {tag}: {count} facets")
		else:
			logger.debug("No facet tags available")

	def _reset_vertex_mapping_cache(self):
		"""Reset cached vertex mapping data (call when mesh changes)."""
		self._gmsh_vertices_original = None
		self._gmsh_vertices_for_mapping = None
		self._dolfinx_to_gmsh_indices = None
	
	# ==================== Unified Boundary Location Resolution System ====================
	
	def _load_geometry_boundaries_config(self) -> dict:
		"""Load geometry_boundaries.json config file"""
		import json
		from pathlib import Path
		
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
	
	def _generate_name_variations(self, name: str) -> list:
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
	
	def _normalize_physical_group_keys(self, physical_groups: dict) -> dict:
		"""Create a dict keyed by normalized name variations for robust lookup."""
		if not physical_groups:
			return {}
		normalized = {}
		for name, group in physical_groups.items():
			if not isinstance(name, str):
				continue
			for variant in self._generate_name_variations(name) + [name]:
				key = variant.lower().strip()
				if key:
					normalized[key] = group
		return normalized

	def _extract_fe_metadata(self, pde_config: Dict[str, Any], default_family: str = "Lagrange", default_degree: int = 1) -> Tuple[str, int]:
		"""Return (family, degree) for FE spaces with sane fallbacks."""
		family = (pde_config.get("family", default_family) or default_family).strip()
		try:
			degree = int(pde_config.get("degree", default_degree))
		except (TypeError, ValueError):
			degree = default_degree
		return family, degree

	def _extract_material_value(
		self,
		pde_config: Dict[str, Any],
		key: str,
		default: float = 0.0,
		aliases: Optional[List[str]] = None,
	) -> float:
		"""Pull a scalar material value from common config locations with optional aliases."""
		sources = [
			pde_config.get("material_properties") or {},
			pde_config.get("material") or {},
			pde_config,
		]
		keys = [key] + (aliases or [])
		for candidate in keys:
			for src in sources:
				if candidate in src and src[candidate] is not None:
					try:
						return float(src[candidate])
					except (TypeError, ValueError):
						logger.debug(f"Material value for '{candidate}' is not numeric: {src[candidate]!r}")
		return float(default)

	def _ensure_single_process_comm(self, comm: Optional[MPI.Comm], context: str = "") -> None:
		"""Raise if communicator has more than one rank; helps catch MPI misuse early."""
		if comm is None or not hasattr(comm, "Get_size"):
			return
		try:
			size = comm.Get_size()
		except Exception:
			return
		if size > 1:
			label = f"{context} " if context else ""
			msg = f"{label}communicator has {size} processes but application is single-process."
			logger.error(f"CRITICAL: {msg}")
			raise RuntimeError(msg)

	def _prepare_petsc_solver_options(
		self,
		pde_config: Dict[str, Any],
		PETSc_module,
		default_prefix: str,
	) -> Tuple[Dict[str, Any], str]:
		"""Build PETSc option dict + prefix and ensure single-process settings."""
		prefix = str(pde_config.get("petsc_options_prefix", pde_config.get("options_prefix", default_prefix)))
		petsc_opts = {
			"ksp_type": pde_config.get("ksp_type", "preonly"),
			"pc_type": pde_config.get("pc_type", "lu"),
		}
		if "pc_factor_mat_solver_type" in pde_config:
			petsc_opts["pc_factor_mat_solver_type"] = pde_config["pc_factor_mat_solver_type"]
		petsc_opts.setdefault("mat_type", "aij")
		petsc_opts.setdefault("ksp_rtol", "1e-10")

		self._configure_petsc_single_process(petsc_opts, PETSc_module)
		return petsc_opts, prefix

	def _configure_petsc_single_process(self, petsc_opts: Dict[str, Any], PETSc_module) -> bool:
		"""Clear PETSc global options and set provided ones (best-effort)."""
		try:
			opts = PETSc_module.Options()
			opts.clear()
			for key, value in petsc_opts.items():
				opts.setValue(key, str(value))
			logger.debug("Configured PETSc for single-process execution")
			return True
		except Exception as exc:
			logger.warning(f"Could not configure PETSc options: {exc}")
			return False
	
	def _find_physical_group_by_location(self, location: str, physical_groups: dict, fdim: int = None) -> tuple:
		"""
		Find physical group that matches the given location using fuzzy matching.
		
		Returns: (physical_group, tag, confidence_score)
		"""
		if not location or not physical_groups:
			return (None, None, 0.0)
		
		location_lower = location.lower().strip()
		
		# Step 1: Direct match (highest confidence)
		if location_lower in physical_groups:
			group = physical_groups[location_lower]
			if fdim is None or (hasattr(group, 'dim') and group.dim == fdim):
				return (group, getattr(group, 'tag', None), 1.0)
		
		# Step 2: Try variations (medium confidence)
		variations = self._generate_name_variations(location)
		for variant in variations:
			if variant in physical_groups:
				group = physical_groups[variant]
				if fdim is None or (hasattr(group, 'dim') and group.dim == fdim):
					return (group, getattr(group, 'tag', None), 0.8)
		
		# Step 3: Try all physical group names with variations
		for pg_name, group in physical_groups.items():
			pg_variations = self._generate_name_variations(pg_name)
			if location_lower in pg_variations:
				if fdim is None or (hasattr(group, 'dim') and group.dim == fdim):
					return (group, getattr(group, 'tag', None), 0.7)
		
		# Step 4: Fuzzy string matching (lower confidence) - simple substring match
		for pg_name, group in physical_groups.items():
			pg_name_lower = pg_name.lower()
			# Check if location is substring of pg_name or vice versa
			if location_lower in pg_name_lower or pg_name_lower in location_lower:
				if fdim is None or (hasattr(group, 'dim') and group.dim == fdim):
					return (group, getattr(group, 'tag', None), 0.6)
		
		return (None, None, 0.0)
	
	def _map_location_via_config(self, location: str, geometry_type: str) -> str:
		"""
		Map user location to canonical form using geometry_boundaries.json config.
		Returns the mapped location or original if no mapping found.
		"""
		if not location or not geometry_type:
			return location
		
		config = self._load_geometry_boundaries_config()
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
	
	def _find_closest_dof_to_point(self, V, point: np.ndarray) -> np.ndarray:
		"""Find closest DOF to a point (for point constraints like 'center')"""
		from scipy.spatial import cKDTree
		
		dof_coords = V.tabulate_dof_coordinates().reshape((-1, point.shape[0]))
		tree = cKDTree(dof_coords)
		distances, indices = tree.query(point, k=1)
		return np.array([indices], dtype=np.int32)

	def _resolve_via_physical_groups(
		self,
		target_location: str,
		physical_groups: dict,
		V,
		mesh,
		fdim: int,
		gdim: int,
		facet_tags,
		method_label: str,
	) -> Optional[Tuple[np.ndarray, str, float]]:
		"""Shared helper to resolve DOFs using named physical groups."""
		if not target_location or not physical_groups:
			return None
		
		location_key = target_location.lower().strip()
		center_aliases = {"center", "middle", "origin", "core"}
		dim_to_check = 0 if location_key in center_aliases else fdim
		
		pg_match, tag, confidence = self._find_physical_group_by_location(location_key, physical_groups, dim_to_check)
		if pg_match is None:
			return None
		# Accept any confidence > 0 (physical groups are authoritative)
		if confidence <= 0.0:
			return None
		
		if dim_to_check == 0 and getattr(pg_match, "dim", None) == 0:
			try:
				point = np.zeros(gdim)
				entity_coords = getattr(pg_match, "entity_coordinates", None)
				if entity_coords:
					point = np.array(entity_coords[0][:gdim])
				dofs = self._find_closest_dof_to_point(V, point)
				if len(dofs) > 0:
					return (dofs, method_label + "_point", confidence)
			except Exception as exc:
				logger.debug(f"Point physical group resolution failed for '{target_location}': {exc}")
			return None
		
		if tag is None or facet_tags is None or not hasattr(facet_tags, "values"):
			return None
		
		try:
			from dolfinx import fem
			facet_indices = np.where(facet_tags.values == tag)[0]
			if facet_indices.size == 0:
				return None

			mesh.topology.create_connectivity(fdim, 0)

			def _needs_vertex_dofs(function_space) -> bool:
				try:
					element = function_space.ufl_element()
					value_shape = element.value_shape()
					is_scalar = value_shape == () or value_shape == (1,)
					return is_scalar and element.family().lower() in ("lagrange", "cg") and element.degree() == 1
				except Exception:
					return False

			if _needs_vertex_dofs(V):
				conn = mesh.topology.connectivity(fdim, 0)
				if conn is None:
					return None
				boundary_vertices = set()
				for facet_idx in facet_indices:
					for vertex in conn.links(int(facet_idx)):
						boundary_vertices.add(vertex)
				if not boundary_vertices:
					return None
				boundary_vertices = np.array(sorted(boundary_vertices), dtype=np.int32)
				dofs = fem.locate_dofs_topological(V, 0, boundary_vertices)
			else:
				dofs = fem.locate_dofs_topological(V, fdim, facet_indices.astype(np.int32))

			if len(dofs) > 0:
				return (dofs, method_label, confidence)
		except Exception as exc:
			logger.debug(f"Facet physical group resolution failed for '{target_location}': {exc}")
		
		return None

	def _build_scalar_function_space(self, pde_config: Dict[str, Any]):
		"""Create scalar CG function space with communicator safety checks."""
		from dolfinx import fem

		if not hasattr(self, "domain") or self.domain is None:
			raise RuntimeError("Mesh (self.domain) not initialized.")

		family, degree = self._extract_fe_metadata(pde_config, default_degree=1)
		mesh = self.domain
		self._ensure_single_process_comm(getattr(mesh, "comm", None), "scalar field mesh")
		V = fem.functionspace(mesh, (family, degree))
		self._ensure_single_process_comm(getattr(getattr(V, "mesh", None), "comm", None), "scalar field function space")
		return V, family, degree

	def _build_vector_function_space(self, pde_config: Dict[str, Any], gdim: int):
		"""Create vector CG function space with communicator safety checks."""
		from dolfinx import fem
		import basix

		if not hasattr(self, "domain") or self.domain is None:
			raise RuntimeError("Mesh (self.domain) not initialized.")

		family, degree = self._extract_fe_metadata(pde_config, default_degree=1)
		mesh = self.domain
		self._ensure_single_process_comm(getattr(mesh, "comm", None), "vector field mesh")
		element = basix.ufl.element(family, mesh.basix_cell(), degree, shape=(gdim,))
		V = fem.functionspace(mesh, element)
		self._ensure_single_process_comm(getattr(getattr(V, "mesh", None), "comm", None), "vector field function space")
		return V, family, degree
	
	def _resolve_boundary_location(self, location: str, V, mesh, physical_groups: dict, 
	                               geometry_type: str, pde_config: dict, fdim: int) -> tuple:
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
		facet_tags = getattr(self, "facet_tags", None)
		if facet_tags is None and hasattr(self, "mesh_data"):
			if isinstance(self.mesh_data, dict):
				facet_tags = self.mesh_data.get("facet_tags")
			elif hasattr(self.mesh_data, "facet_tags"):
				facet_tags = self.mesh_data.facet_tags
		
		# Step 1: Try physical groups first (most reliable)
		if physical_groups:
			result = self._resolve_via_physical_groups(location_lower, physical_groups, V, mesh, fdim, gdim, facet_tags, "physical_group")
			if result:
				return result
		
		# Step 2: Try config-driven mapping + physical groups
		if geometry_type:
			canonical_location = self._map_location_via_config(location_lower, geometry_type)
			if canonical_location != location_lower and physical_groups:
				result = self._resolve_via_physical_groups(
					canonical_location,
					physical_groups,
					V,
					mesh,
					fdim,
					gdim,
					facet_tags,
					"config_physical_group",
				)
				if result:
					return result
		
		# NO FALLBACK - Physical groups are required
		available_groups = list(physical_groups.keys()) if physical_groups else []
		raise RuntimeError(
			f"Could not resolve boundary location '{location}' using physical groups. "
			f"Available physical groups: {available_groups}. "
			f"Physical groups MUST be provided by the mesh generator."
		)

	def _ensure_scalar_vertex_mapping(self):
		"""Ensure mapping between DOLFINx vertices and original GMSH vertices is prepared."""
		if self._gmsh_vertices_original is not None and self._dolfinx_to_gmsh_indices is not None:
			return

		if not hasattr(self, "domain") or self.domain is None:
			raise RuntimeError("Mesh (self.domain) not initialized. Cannot build vertex mapping.")

		if not hasattr(self, "original_mesh_data") or self.original_mesh_data is None:
			raise RuntimeError("Original mesh data not available. Cannot build vertex mapping.")

		gmsh_vertices_original = np.array(self.original_mesh_data["vertices"])
		domain = self.domain
		gdim = int(domain.geometry.dim)
		gmsh_dim = gmsh_vertices_original.shape[1]
		gmsh_vertices_for_mapping = gmsh_vertices_original[:, :gdim] if gmsh_dim >= gdim else gmsh_vertices_original

		dolfinx_vertices = domain.geometry.x
		gmsh_tree = cKDTree(gmsh_vertices_for_mapping)
		distances, dolfinx_to_gmsh_indices = gmsh_tree.query(dolfinx_vertices, k=1)

		if np.max(distances) > 1e-6:
			logger.warning(
				f"Vertex mapping distances are large: max={np.max(distances):.2e}, mean={np.mean(distances):.2e}"
			)
		else:
			logger.debug(f"Vertex mapping successful: max distance={np.max(distances):.2e}")

		self._gmsh_vertices_original = gmsh_vertices_original
		self._gmsh_vertices_for_mapping = gmsh_vertices_for_mapping
		self._dolfinx_to_gmsh_indices = dolfinx_to_gmsh_indices

	def _map_scalar_dofs_to_gmsh(self, solution_values: np.ndarray) -> np.ndarray:
		"""Map scalar FEM solution DOFs to original GMSH vertex ordering."""
		self._ensure_scalar_vertex_mapping()

		num_vertices = self._gmsh_vertices_original.shape[0]
		gmsh_solution_values = np.full(num_vertices, np.nan, dtype=float)

		for dolfinx_idx, gmsh_idx in enumerate(self._dolfinx_to_gmsh_indices):
			if dolfinx_idx >= len(solution_values):
				break
			value = float(solution_values[dolfinx_idx])
			if np.isnan(gmsh_solution_values[gmsh_idx]):
				gmsh_solution_values[gmsh_idx] = value
			else:
				gmsh_solution_values[gmsh_idx] = 0.5 * (gmsh_solution_values[gmsh_idx] + value)

		unmapped = np.isnan(gmsh_solution_values)
		if np.any(unmapped):
			mapped_coords = self._gmsh_vertices_for_mapping[~unmapped]
			mapped_values = gmsh_solution_values[~unmapped]
			unmapped_coords = self._gmsh_vertices_for_mapping[unmapped]
			if len(mapped_coords) > 0 and len(unmapped_coords) > 0:
				tree = cKDTree(mapped_coords)
				_, nearest = tree.query(unmapped_coords, k=1)
				gmsh_solution_values[unmapped] = mapped_values[nearest]

		return gmsh_solution_values

	def _extract_initial_scalar_value(self, pde_config: Dict[str, Any], default: float = 0.0) -> float:
		"""Extract a scalar initial value (e.g., temperature) from PDE config."""
		initial_conditions = pde_config.get("initial_conditions") or []

		def _to_float(val):
			try:
				return float(val)
			except (TypeError, ValueError):
				return None

		for ic in initial_conditions:
			if not isinstance(ic, dict):
				continue
			value = ic.get("value")
			if isinstance(value, (int, float)):
				return float(value)
			if isinstance(value, dict):
				for key in ("temperature", "value", "initial_temperature"):
					if key in value:
						val = _to_float(value[key])
						if val is not None:
							return val
			for key in ("initial_temperature", "temperature", "value"):
				if key in ic:
					val = _to_float(ic[key])
					if val is not None:
						return val

		fallback = pde_config.get("initial_temperature")
		fallback_val = _to_float(fallback)
		if fallback_val is not None:
			return fallback_val

		return float(default)

	def extract_solution_data(self, physics_type: str = "heat_transfer") -> Dict[str, Any]:
		"""Extract solution data and map to GMSH vertices"""
		if getattr(self, "solution", None) is None:
			logger.error("No solution available.")
			return {"success": False, "status": "error", "message": "No solution available."}
		if not hasattr(self, "domain"):
			logger.error("No dolfinx mesh (self.domain) available.")
			return {"success": False, "status": "error", "message": "No dolfinx mesh available."}
		
		u = self.solution
		domain = self.domain
		gdim = int(domain.geometry.dim)
		
		# Get original GMSH mesh data (this is what we want to visualize)
		gmsh_vertices_original = np.array(self.original_mesh_data["vertices"])
		original_cells = self.original_mesh_data.get('cells', {})
		original_faces = self.original_mesh_data.get('faces', [])
		gmsh_dim = gmsh_vertices_original.shape[1]  # Original GMSH dimension (could be 2D or 3D)
		
		# Get field info
		field_info = self._get_field_info(physics_type)
		primary_field = field_info.get("primary_field", "temperature")
		visualization_field = field_info.get("visualization_field", primary_field)
		field_type = field_info["fields"][primary_field]["type"]
		
		# Get solution values from DOLFINx
		solution_values = u.x.array
		
		# Get DOLFINx mesh vertex coordinates
		# NOTE: These may be different from GMSH vertices (reordered, different coordinates)
		dolfinx_vertices = domain.geometry.x
		
		# Normalize GMSH vertices to match DOLFINx dimension for mapping (use first gdim components)
		gmsh_vertices_for_mapping = gmsh_vertices_original[:, :gdim] if gmsh_dim >= gdim else gmsh_vertices_original
		
		# Use KDTree to map DOLFINx vertices to GMSH vertices
		# This handles cases where DOLFINx reorders or slightly modifies vertices
		logger.debug(f"Mapping {len(dolfinx_vertices)} DOLFINx vertices to {len(gmsh_vertices_for_mapping)} GMSH vertices")
		gmsh_tree = cKDTree(gmsh_vertices_for_mapping)
		distances, dolfinx_to_gmsh_indices = gmsh_tree.query(dolfinx_vertices, k=1)
		
		# Log mapping quality
		if np.max(distances) > 1e-6:
			logger.warning(f"Large vertex mapping distances detected: max={np.max(distances):.2e}, mean={np.mean(distances):.2e}")
		else:
			logger.debug(f"Vertex mapping successful: max distance={np.max(distances):.2e}")
		
		if field_type == "scalar":
			# Scalar field: map solution from DOLFINx vertices to GMSH vertices
			gmsh_solution_values = self._map_scalar_dofs_to_gmsh(solution_values)
			coords_for_viz = self._gmsh_vertices_original.tolist()
			
			viz_field_name = field_info["fields"][visualization_field]["name"]
			viz_field_units = field_info["fields"][visualization_field]["units"]
			
			return {
				"success": True,
				"status": "success",
				"physics_type": physics_type,
				"field_type": "scalar",
				"coordinates": coords_for_viz,
				"values": gmsh_solution_values.tolist(),
				"faces": original_faces,
				"cells": original_cells,
				"field_name": viz_field_name,
				"field_units": viz_field_units,
				"min_value": float(np.nanmin(gmsh_solution_values)),
				"max_value": float(np.nanmax(gmsh_solution_values)),
				"field_info": field_info,
				"mesh_info": {
					"num_vertices": len(gmsh_vertices_original),
					"num_dofs": len(solution_values),
					"dimension": int(domain.topology.dim),
					"geometry_dim": int(domain.geometry.dim)
				}
			}
		
		elif field_type == "vector":
			# Vector field: map displacement from DOLFINx vertices to GMSH vertices
			# DOLFINx solution: [u0_x, u0_y, u0_z, u1_x, u1_y, u1_z, ...] per DOLFINx vertex
			
			# Initialize displacement arrays for GMSH vertices (gdim components)
			gmsh_displacement_gdim = np.zeros((len(gmsh_vertices_original), gdim))
			displacement_count = np.zeros(len(gmsh_vertices_original))  # Track mappings per GMSH vertex
			
			# Map displacement: DOLFINx vertex i -> GMSH vertex gmsh_idx
			for dolfinx_idx, gmsh_idx in enumerate(dolfinx_to_gmsh_indices):
				# Extract displacement vector for this DOLFINx vertex (gdim components)
				for comp in range(gdim):
					dof_idx = dolfinx_idx * gdim + comp
					if dof_idx < len(solution_values):
						gmsh_displacement_gdim[gmsh_idx, comp] += solution_values[dof_idx]
				displacement_count[gmsh_idx] += 1
			
			# Average displacement if multiple DOLFINx vertices mapped to same GMSH vertex
			for gmsh_idx in range(len(gmsh_vertices_original)):
				if displacement_count[gmsh_idx] > 1:
					gmsh_displacement_gdim[gmsh_idx] /= displacement_count[gmsh_idx]
			
			# Fill any unmapped GMSH vertices with nearest neighbor interpolation
			unmapped = displacement_count == 0
			if np.any(unmapped):
				mapped_coords = gmsh_vertices_for_mapping[~unmapped]
				mapped_displacement = gmsh_displacement_gdim[~unmapped]
				unmapped_coords = gmsh_vertices_for_mapping[unmapped]
				if len(mapped_coords) > 0 and len(unmapped_coords) > 0:
					tree = cKDTree(mapped_coords)
					_, nearest = tree.query(unmapped_coords, k=1)
					gmsh_displacement_gdim[unmapped] = mapped_displacement[nearest]
			
			# Pad displacement to match original GMSH dimension
			# If original GMSH is 3D but problem is 2D, pad z-component with zeros
			if gmsh_dim > gdim:
				# Pad with zeros for additional dimensions
				zero_padding = np.zeros((len(gmsh_vertices_original), gmsh_dim - gdim))
				gmsh_displacement = np.hstack([gmsh_displacement_gdim, zero_padding])
			elif gmsh_dim < gdim:
				# Truncate to GMSH dimension (shouldn't happen, but handle it)
				gmsh_displacement = gmsh_displacement_gdim[:, :gmsh_dim]
			else:
				# Dimensions match
				gmsh_displacement = gmsh_displacement_gdim
			
			# Compute displacement magnitude using gdim components
			displacement_magnitude = np.linalg.norm(gmsh_displacement_gdim, axis=1)
			
			# Compute deformed coordinates: original GMSH vertices + displacement
			# This preserves the original GMSH vertex dimension
			deformed_coordinates = gmsh_vertices_original + gmsh_displacement
			
			# Scale for visualization if displacement is very small
			max_displacement = np.max(displacement_magnitude) if len(displacement_magnitude) > 0 else 0.0
			mesh_size = np.max(gmsh_vertices_original.max(axis=0) - gmsh_vertices_original.min(axis=0))
			displacement_scale = 1.0
			if mesh_size > 0 and max_displacement > 0:
				relative_displacement = max_displacement / mesh_size
				if relative_displacement < 0.01:
					displacement_scale = 0.05 / relative_displacement
					logger.info(f"Auto-scaling displacement: {displacement_scale:.2f}x")
			
			# Apply scaling to displacement and compute scaled deformed coordinates
			scaled_displacement = gmsh_displacement * displacement_scale
			deformed_coordinates_scaled = gmsh_vertices_original + scaled_displacement
			
			viz_field_name = field_info["fields"][visualization_field]["name"]
			viz_field_units = field_info["fields"][visualization_field]["units"]
			
			return {
				"success": True,
				"status": "success",
				"physics_type": physics_type,
				"field_type": "vector",
				"primary_field": primary_field,
				"visualization_field": visualization_field,
				"available_fields": field_info.get("available_fields", [primary_field]),
				"coordinates": gmsh_vertices_original.tolist(),  # Original GMSH vertices (full dimension)
				"deformed_coordinates": deformed_coordinates_scaled.tolist(),  # Deformed coordinates (same dimension as original)
				"values": displacement_magnitude.tolist(),  # Displacement magnitude for scalar coloring
				"vector_values": gmsh_displacement.tolist(),  # Displacement vector (unscaled, full dimension)
				"displacement_scale": float(displacement_scale),
				"faces": original_faces,  # Original GMSH faces (correct connectivity)
				"cells": original_cells,   # Original GMSH cells (correct connectivity)
				"field_name": viz_field_name,
				"field_units": viz_field_units,
				"field_info": field_info,
				"mesh_info": {
					"num_vertices": len(gmsh_vertices_original),
					"num_dofs": len(solution_values),
					"dimension": int(domain.topology.dim),
					"geometry_dim": int(domain.geometry.dim)
				}
			}
		
		else:
			# Fallback
			logger.warning(f"Unsupported field type '{field_type}'")
			gmsh_solution_values = np.zeros(len(gmsh_vertices_original))
			for dolfinx_idx, gmsh_idx in enumerate(dolfinx_to_gmsh_indices):
				if dolfinx_idx < len(solution_values):
					gmsh_solution_values[gmsh_idx] = solution_values[dolfinx_idx]
			
			viz_field_name = field_info["fields"][visualization_field]["name"]
			viz_field_units = field_info["fields"][visualization_field]["units"]
			
			return {
				"success": True,
				"status": "success",
				"physics_type": physics_type,
				"field_type": "scalar",
				"coordinates": gmsh_vertices_original.tolist(),
				"values": gmsh_solution_values.tolist(),
				"faces": original_faces,
				"cells": original_cells,
				"field_name": viz_field_name,
				"field_units": viz_field_units,
				"field_info": field_info,
				"mesh_info": {
					"num_vertices": len(gmsh_vertices_original),
					"num_dofs": len(solution_values),
					"dimension": int(domain.topology.dim),
					"geometry_dim": int(domain.geometry.dim)
				}
			}

	def _canon_loc(self, loc: Optional[str], geometry_type: str = None) -> str:
		"""
		Canonicalize location name using config-driven mapping.
		Uses geometry_boundaries.json as the source of truth.
		"""
		if not loc:
			return loc
		
		loc_lower = loc.strip().lower()
		
		# Use config-driven mapping
		if geometry_type:
			mapped = self._map_location_via_config(loc_lower, geometry_type)
			if mapped != loc_lower:
				return mapped
		
		# Fallback: simple common mappings (for backward compatibility)
		simple_aliases = {
			"all": "all_boundary", "entire": "all_boundary", "complete": "all_boundary",
			"all_boundary": "all_boundary"
		}
		
		return simple_aliases.get(loc_lower, loc)

	def _vec(self, value, gdim: int) -> List[float]:
		"""
		Accepts scalar, list/tuple, or dict with *_x/_y/_z and returns a length-gdim list.
		Missing components are filled with 0.0.
		"""
		if value is None:
			return [0.0] * gdim
		# dict with components?
		if isinstance(value, dict):
			cand = []
			for key in ("x","y","z"):
				# accept 'ux', 'fx', etc., or plain 'x'
				v = (value.get(f"u{key}") if f"u{key}" in value else
					value.get(f"f{key}") if f"f{key}" in value else
					value.get(f"{key}", None))
				if v is None:
					cand.append(0.0)
				else:
					cand.append(float(v))
			return cand[:gdim] + [0.0] * max(0, gdim - len(cand))
		# numeric → along x by convention (or 0 vector? choose 0 vector is safer)
		try:
			# if you prefer "scalar means x-component", replace with [float(value)] + [0.0]*(gdim-1)
			_ = float(value)
			return [float(value)] + [0.0]*(gdim-1)
		except Exception:
			pass
		# sequence
		if isinstance(value, (list, tuple)):
			vec = [float(v) for v in value[:gdim]]
			return vec + [0.0] * (gdim - len(vec))
		# fallback
		return [0.0] * gdim

	def _convert_heat_flux_units(self, value: float, units: Optional[str] = None) -> float:
		"""
		Convert heat flux from various units to W/m² (SI base unit).
		
		Supported units:
		- W/m², W/m^2 (SI base unit) - no conversion
		- W/cm², W/cm^2 - multiply by 10000
		- cal/(cm²·s), cal/(cm^2·s) - multiply by 41840
		- BTU/(ft²·h), BTU/(ft^2·h) - multiply by 3.15459
		- kW/m², kW/m^2 - multiply by 1000
		- MW/m², MW/m^2 - multiply by 1000000
		"""
		if units is None:
			return value
		
		units = units.strip().lower().replace(" ", "").replace("·", "")
		
		# Conversion factors to W/m²
		conversions = {
			"w/m²": 1.0,
			"w/m^2": 1.0,
			"w/m2": 1.0,
			"w/cm²": 10000.0,
			"w/cm^2": 10000.0,
			"w/cm2": 10000.0,
			"cal/(cm²·s)": 41840.0,
			"cal/(cm^2·s)": 41840.0,
			"cal/(cm2·s)": 41840.0,
			"cal/(cm²s)": 41840.0,
			"cal/(cm^2s)": 41840.0,
			"cal/(cm2s)": 41840.0,
			"btu/(ft²·h)": 3.15459,
			"btu/(ft^2·h)": 3.15459,
			"btu/(ft2·h)": 3.15459,
			"btu/(ft²h)": 3.15459,
			"btu/(ft^2h)": 3.15459,
			"btu/(ft2h)": 3.15459,
			"kw/m²": 1000.0,
			"kw/m^2": 1000.0,
			"kw/m2": 1000.0,
			"mw/m²": 1000000.0,
			"mw/m^2": 1000000.0,
			"mw/m2": 1000000.0,
		}
		
		if units in conversions:
			converted_value = value * conversions[units]
			logger.debug(f"Converted heat flux: {value} {units} → {converted_value} W/m²")
			return converted_value
		else:
			logger.warning(f"Unknown heat flux unit: '{units}'. Assuming W/m². Supported units: {list(conversions.keys())}")
			return value

	
	def _convert_pressure_units(self, value: float, units: Optional[str] = None) -> float:
		"""
		Convert pressure/traction from various units to Pa (SI base unit).
		
		Supported units:
		- Pa, N/m², N/m^2 (SI base unit) - no conversion
		- kPa - multiply by 1000
		- MPa - multiply by 1000000
		- GPa - multiply by 1000000000
		- bar - multiply by 100000
		- atm - multiply by 101325
		- psi - multiply by 6894.76
		- ksi - multiply by 6894760
		- psf - multiply by 47.88
		"""
		if units is None:
			return value
		
		units = units.strip().lower().replace(" ", "")
		
		# Conversion factors to Pa
		conversions = {
			"pa": 1.0,
			"n/m²": 1.0,
			"n/m^2": 1.0,
			"n/m2": 1.0,
			"kpa": 1000.0,
			"mpa": 1000000.0,
			"gpa": 1000000000.0,
			"bar": 100000.0,
			"atm": 101325.0,
			"atmosphere": 101325.0,
			"psi": 6894.76,
			"ksi": 6894760.0,
			"psf": 47.88,
			"lbf/ft²": 47.88,
			"lbf/ft^2": 47.88,
			"lbf/ft2": 47.88,
		}
		
		if units in conversions:
			converted_value = value * conversions[units]
			logger.debug(f"Converted pressure/traction: {value} {units} → {converted_value} Pa")
			return converted_value
		else:
			logger.warning(f"Unknown pressure/traction unit: '{units}'. Assuming Pa. Supported units: {list(conversions.keys())}")
			return value
	
	def normalize_heat_bcs(self, raw_bcs: List[Dict[str, Any]], assume_celsius: bool=True) -> List[Dict[str, Any]]:
		"""
		Input examples (NLP):
		{"type":"temperature","location":"left","value":100,"bc_type":"dirichlet"}
		{"type":"insulated","location":"right","bc_type":"neumann"}
		{"type":"flux","location":"top","value":1000,"units":"W/cm²"}  # Unit conversion supported
		Output (solver-ready):
		{"type":"dirichlet","location":"x_min","value":373.15}
		{"type":"neumann","location":"x_max","value":0.0}
		{"type":"neumann","location":"top","value":10000000.0}  # Converted to W/m²
		"""
		def to_kelvin(x):
			try:
				x = float(x)
				return x + 273.15 if assume_celsius else x
			except Exception:
				return x

		out = []
		for bc in (raw_bcs or []):
			loc = self._canon_loc(bc.get("location"))
			t   = (bc.get("type") or "").strip().lower()
			hint= (bc.get("bc_type") or "").strip().lower()

			# classify
			if t in ("temperature","fixed") or hint == "dirichlet":
				btype = "dirichlet"
				val = bc.get("value", 0.0)
				val = to_kelvin(val) if val is not None else None
			elif t in ("insulated","free","neumann_zero"):
				btype = "neumann"   # treat insulated as neumann with q=0
				val = 0.0
			elif t in ("neumann","flux","heat_flux","heat flux") or hint == "neumann":
				btype = "neumann"
				val = float(bc.get("value", 0.0))
				# Convert units if specified
				units = bc.get("units") or bc.get("parameters", {}).get("units")
				if units:
					val = self._convert_heat_flux_units(val, units)
				# Special case: for discs and spheres, a "center" heat flux is not a boundary condition;
				# mark it as an internal source so the solver can add it to the RHS volume term.
				if (bc.get("location","center").strip().lower() == "center"):
					btype = "internal_source"
			elif t in ("convection","robin","radiation"):
				# Expect `value: {"h":..., "Tinf":...}` ; pass through as-is
				btype = "robin"
				val = bc.get("value", {})
			else:
				# unknown → skip safely
				continue

			out.append({"type": btype, "location": loc, "value": val})
		return out

	def normalize_solid_bcs(self, raw_bcs: List[Dict[str, Any]], gdim: int) -> List[Dict[str, Any]]:
		"""
		Input examples (NLP):
		{"type":"fixed","location":"left","value":0,"bc_type":"dirichlet"}
		{"type":"free","location":"top","bc_type":"neumann"}
		{"type":"displacement","location":"right","value":[0.001,0,0]}
		{"type":"force","location":"right","value":{"fx":1e5},"units":"psi"}  # Unit conversion supported
		{"type":"pressure","location":"right","value":100,"units":"MPa"}  # Unit conversion supported
		Output (solver-ready):
		{"type":"dirichlet","location":"x_min","value":[0,0,(0)]}
		{"type":"neumann",  "location":"y_max","value":[0,0,(0)]}  # zero traction
		{"type":"dirichlet","location":"x_max","value":[...]}
		{"type":"neumann",  "location":"x_max","value":[689476000.0,0,0]}  # Converted to Pa
		{"type":"pressure", "location":"x_max","value":100000000.0}  # Converted to Pa
		"""
		out = []
		for bc in (raw_bcs or []):
			loc = self._canon_loc(bc.get("location"))
			t   = (bc.get("type") or "").strip().lower()
			hint= (bc.get("bc_type") or "").strip().lower()

			# Dirichlet displacement
			if t in ("fixed","fixed support","dirichlet","displacement") or hint == "dirichlet":
				btype = "dirichlet"
				if t in ("fixed","fixed support") and bc.get("value") in (None, "", 0, 0.0):
					val = [0.0]*gdim
				else:
					val = self._vec(bc.get("value", 0.0), gdim)
				out.append({"type": btype, "location": loc, "value": val})
				continue

			# Roller support (constrain one component); expect {"constrained_direction":"x|y|z|normal"}
			if t in ("roller","roller support"):
				dir_key = (bc.get("constrained_direction") or "").strip().lower()
				idx = {"x":0, "y":1, "z":2}.get(dir_key, None)
				if idx is not None and idx < gdim:
					val = [None]*gdim  # None = unconstrained; 0.0 = constrained to zero
					val[idx] = 0.0
					# Represent as component-wise Dirichlet instruction for your solver layer:
					out.append({"type":"dirichlet", "location":loc, "value":val})
				else:
					# Fallback: no-op
					out.append({"type":"neumann", "location":loc, "value":[0.0]*gdim})
				continue

			# Zero-traction (free/symmetry)
			if t in ("free","symmetry","neumann_zero"):
				out.append({"type":"neumann", "location":loc, "value":[0.0]*gdim})
				continue

			# Traction / force vector (Neumann)
			if t in ("force","traction") or hint == "neumann":
				vec_raw = bc.get("value", 0.0)
				# Convert units if specified (applies to all components)
				units = bc.get("units") or bc.get("parameters", {}).get("units")
				if units:
					# Convert each component if vector, or single value if scalar
					if isinstance(vec_raw, (list, tuple)):
						# List/tuple: convert each component
						vec = [self._convert_pressure_units(float(v), units) for v in vec_raw]
					elif isinstance(vec_raw, dict):
						# Dictionary format: convert each numeric component, preserve structure
						vec_dict = {}
						for key, val in vec_raw.items():
							try:
								# Convert numeric values, preserve non-numeric as-is
								vec_dict[key] = self._convert_pressure_units(float(val), units)
							except (ValueError, TypeError):
								# Non-numeric value, preserve as-is
								vec_dict[key] = val
						# Convert dict to vector using _vec
						vec = self._vec(vec_dict, gdim)
					else:
						# Scalar: convert and let _vec handle it
						converted_val = self._convert_pressure_units(float(vec_raw), units)
						vec = self._vec(converted_val, gdim)
				else:
					vec = self._vec(vec_raw, gdim)
				out.append({"type":"neumann", "location":loc, "value": vec})
				continue

			# Pressure scalar
			if t == "pressure":
				p = float(bc.get("value", 0.0))
				# Convert units if specified
				units = bc.get("units") or bc.get("parameters", {}).get("units")
				if units:
					p = self._convert_pressure_units(p, units)
				out.append({"type":"pressure", "location":loc, "value": p})
				continue

			# Unknowns → skip safely
			continue

		return out

	def normalize_bcs_by_physics(self, physics_type: str, raw_bcs: List[Dict[str, Any]], gdim: int) -> List[Dict[str, Any]]:
		pt = (physics_type or "").strip().lower()
		if pt == "heat_transfer":
			return self.normalize_heat_bcs(raw_bcs, assume_celsius=True)
		if pt == "solid_mechanics":
			return self.normalize_solid_bcs(raw_bcs, gdim=gdim)
		# default
		return raw_bcs or []

	def solve_simulation(self, config: Dict[str, Any], mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Run the simulation selected by `config` on the mesh described by `mesh_data`.
		Returns a dict with solution/visualization payloads (from the solver).
		"""
		logger.debug("Starting simulation")

		# Store original mesh data for later use in solution extraction
		self.original_mesh_data = mesh_data
		self._reset_vertex_mapping_cache()

		# ---- mesh validation ----
		if not isinstance(mesh_data, dict) or "vertices" not in mesh_data:
			raise ValueError("mesh_data must include a 'vertices' array and cell blocks")
		if all(k not in mesh_data.get("cells", mesh_data) for k in ("tetra", "tetrahedron", "triangle", "triangle_2nd", "line", "line_2nd")):
			logger.warning("mesh_data has no recognized 'cells'; attempting to use top-level keys.")

		# CRITICAL: Extract physical groups from mesh_data FIRST (before creating DOLFINx mesh)
		# The mesh generator already extracted them with correct names like 'left_boundary', 'right_boundary', etc.
		physical_groups_from_mesh_data = mesh_data.get('physical_groups')
		logger.info(f"Checking for physical_groups in mesh_data: present={physical_groups_from_mesh_data is not None}, type={type(physical_groups_from_mesh_data)}, keys={list(mesh_data.keys())[:20] if isinstance(mesh_data, dict) else 'N/A'}")
		if physical_groups_from_mesh_data:
			logger.info(f"Found physical groups in mesh_data: {len(physical_groups_from_mesh_data)} groups, keys={list(physical_groups_from_mesh_data.keys())[:10] if isinstance(physical_groups_from_mesh_data, dict) else 'N/A'}")
			# Convert dict-based physical groups back to PhysicalGroupWrapper objects for compatibility
			# (They may have been serialized to dicts by _make_json_safe)
			converted_physical_groups = {}
			for name, group in physical_groups_from_mesh_data.items():
				if isinstance(group, dict):
					# Reconstruct PhysicalGroupWrapper from dict
					from types import SimpleNamespace
					wrapper = SimpleNamespace()
					wrapper.dim = group.get('dim')
					wrapper.tag = group.get('tag')
					wrapper.entities = group.get('entities', [])
					wrapper.name = group.get('name')
					wrapper.dimension = group.get('dimension', group.get('dim'))
					wrapper.entity_coordinates = group.get('entity_coordinates', [])
					converted_physical_groups[name] = wrapper
					logger.info(f"  Physical group from mesh_data: '{name}' -> tag {wrapper.tag}, dim={wrapper.dim}")
				elif hasattr(group, 'tag'):
					# Already a PhysicalGroupWrapper object
					converted_physical_groups[name] = group
					logger.info(f"  Physical group from mesh_data: '{name}' -> tag {group.tag}, dim={group.dim}")
				else:
					logger.warning(f"  Physical group from mesh_data: '{name}' -> unexpected type {type(group)}")
			physical_groups_from_mesh_data = converted_physical_groups

		# ---- create DOLFINx mesh from exported .msh file ----
		logger.debug("Creating DOLFINx mesh from exported .msh file...")
		msh_file = self.msh_file_path or mesh_data.get('msh_file')
		if not msh_file:
			raise RuntimeError("No .msh file path provided for mesh creation")
		if not os.path.exists(msh_file):
			raise RuntimeError(f".msh file not found: {msh_file}")
		
		mesh_comm = MPI.COMM_SELF
		logger.info(f"Reading mesh from {msh_file} with gmshio.read_from_msh (COMM_SELF)")
		try:
			mesh_tuple = gmshio.read_from_msh(msh_file, mesh_comm, rank=0)
			if not isinstance(mesh_tuple, (list, tuple)):
				raise RuntimeError("gmshio.read_from_msh returned unexpected type")
			if len(mesh_tuple) < 3:
				raise RuntimeError("gmshio.read_from_msh returned fewer than 3 values")
			domain, cell_tags, facet_tags = mesh_tuple[:3]
		except Exception as mesh_error:
			logger.error(f"Failed to read mesh from {msh_file}: {mesh_error}")
			raise
		
		self.domain = domain
		self.cell_tags = cell_tags
		self.facet_tags = facet_tags
		
		# Log mesh communicator information for debugging
		if hasattr(self.domain, 'comm'):
			mesh_comm = self.domain.comm
			comm_size = mesh_comm.Get_size()
			comm_rank = mesh_comm.Get_rank()
			logger.info(f"Mesh communicator after creation: size={comm_size}, rank={comm_rank}")
			if comm_size > 1:
				logger.error(f"CRITICAL: Mesh was created with {comm_size} processes! Expected size=1.")
				raise RuntimeError(f"Mesh communicator has {comm_size} processes but application is single-process. Mesh creation failed.")
		else:
			logger.warning("Mesh does not have a 'comm' attribute")
		
		# Use physical groups from mesh_data if available
		if physical_groups_from_mesh_data:
			physical_groups = self._normalize_physical_group_keys(physical_groups_from_mesh_data)
			logger.info(f"Using {len(physical_groups_from_mesh_data)} physical groups from mesh_data with correct names")
		else:
			logger.warning("Physical groups not found in mesh_data and no fallback is available without metadata")
			physical_groups = {}
		
		# Log information about cell tags and facet tags
		logger.debug(f"Cell tags: {self.cell_tags}")
		if self.cell_tags is not None:
			logger.debug(f"Cell tags values: {self.cell_tags.values}")
			logger.debug(f"Cell tags unique values: {np.unique(self.cell_tags.values)}")
		
		logger.debug(f"Facet tags: {self.facet_tags}")
		if self.facet_tags is not None:
			logger.debug(f"Facet tags values: {self.facet_tags.values}")
			logger.debug(f"Facet tags unique values: {np.unique(self.facet_tags.values)}")
		
		# Log physical groups
		if physical_groups:
			logger.info(f"Received {len(physical_groups_from_mesh_data)} physical groups from mesh_data")
			for name, group in physical_groups_from_mesh_data.items():
				logger.info(f"  Physical group: '{name}' -> tag {group.tag}, dim={group.dim}")
		else:
			logger.warning("No physical groups available for this mesh")
		
		# Inspect mesh tags for detailed information
		self.inspect_mesh_tags()
		logger.debug(f"Mesh ready: {self.domain.topology.dim}D, vertices={self.domain.geometry.x.shape[0]}")

		if self.domain is None:
			raise RuntimeError("Failed to create DOLFINx mesh")

		tdim = int(self.domain.topology.dim)
		gdim = int(self.domain.geometry.dim)
		nvtx = int(self.domain.geometry.x.shape[0])
		self.mesh_data = mesh_data  # keep original for inverse mapping & viz
		logger.debug(f"Mesh created: gdim={gdim}, tdim={tdim}, vertices={nvtx}, cells={self.domain.topology.index_map(tdim).size_global}")

		# ---- config parsing ----
		pde_config = (config or {}).get("pde_config", {}) or {}
		
		# Log the PDE config received by FEniCS solver (compact)
		logger.debug("FEniCS solver received PDE config")
		logger.debug(f"Physics Type: {pde_config.get('physics_type', 'NOT SET')}")
		logger.debug(f"Material Properties: {pde_config.get('material_properties', 'NOT SET')}")
		logger.debug(f"Raw Boundary Conditions: {pde_config.get('boundary_conditions', 'NOT SET')}")
		
		raw_bcs = pde_config.get("boundary_conditions", [])
		logger.info(f"Normalizing {len(raw_bcs)} boundary conditions for physics_type={pde_config.get('physics_type')}")
		for i, bc in enumerate(raw_bcs):
			logger.info(f"  Raw BC {i+1}: type='{bc.get('type', 'N/A')}', location='{bc.get('location', 'N/A')}', value={bc.get('value', 'N/A')}")
		
		normalized = self.normalize_bcs_by_physics(pde_config.get("physics_type"), raw_bcs, 3)
		pde_config["boundary_conditions"] = normalized
		
		# Log normalized boundary conditions
		logger.info(f"Normalized {len(normalized)} boundary conditions:")
		for i, bc in enumerate(normalized):
			logger.info(f"  Normalized BC {i+1}: type='{bc.get('type', 'N/A')}', location='{bc.get('location', 'N/A')}', value={bc.get('value', 'N/A')}")
		logger.debug("=" * 60)
		
		physics_type = (pde_config.get("physics_type", "heat_transfer") or "").strip().lower()
		family = (pde_config.get("family", "Lagrange") or "Lagrange").strip()  # "P" also fine
		degree = self.domain.geometry.cmap.degree

		# Precompute common connectivities (helps BCs, searches, etc.)
		self.domain.topology.create_connectivity(0, tdim)
		self.domain.topology.create_connectivity(tdim, 0)

		# ---- delegate to physics-specific solver ----
		# NOTE: let each solver build its own function space(s) as needed
		# based on (family, degree, gdim) to avoid accidental scalar/vector mismatch.
		if physics_type == "heat_transfer":
			# expect solver to create scalar FunctionSpace(family, degree)
			return self._solve_heat_transfer({**pde_config, "family": family, "degree": degree})
		elif physics_type == "solid_mechanics":
			# expect solver to create vector FunctionSpace(family, degree, (gdim,))
			return self._solve_solid_mechanics({**pde_config, "family": family, "degree": degree, "gdim": gdim})
		else:
			raise ValueError(f"Unknown physics type: {physics_type!r}")

	def _solve_heat_transfer(self, pde_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Solve steady heat conduction: -div(k ∇u) = 0 with boundary conditions using DOLFINx (robust)."""
		logger.debug("Solving heat transfer")
		try:
			from dolfinx import fem
			from dolfinx.fem.petsc import LinearProblem
			from petsc4py import PETSc
			import ufl

			ScalarType = PETSc.ScalarType

			if not hasattr(self, "domain"):
				raise RuntimeError("Mesh (self.domain) not initialized. Call solve_simulation first.")

			time_params = pde_config.get("time_stepping") or {}

			def _is_positive(value) -> bool:
				try:
					return float(value) > 0
				except (TypeError, ValueError):
					return False

			is_transient = False
			if isinstance(time_params, dict) and time_params:
				if time_params.get("enabled") is False:
					is_transient = False
				else:
					dt = time_params.get("time_step")
					total_time = time_params.get("total_time")
					is_transient = _is_positive(dt) and _is_positive(total_time)

			if is_transient:
				logger.info("Detected transient heat transfer configuration. Using transient solver.")
				return self._solve_heat_transfer_transient(pde_config)

			m = self.domain
			tdim = m.topology.dim
			fdim = tdim - 1
			m.topology.create_connectivity(fdim, tdim)
			m.topology.create_connectivity(tdim, fdim)
			
			# --- FE space (scalar) ---
			V, family, degree = self._build_scalar_function_space(pde_config)
			mesh_comm = getattr(m, 'comm', MPI.COMM_SELF)
			
			# --- trial/test ---
			u = ufl.TrialFunction(V)
			v = ufl.TestFunction(V)

			# --- material parameter ---
			kappa_val = self._extract_material_value(pde_config, "thermal_conductivity", default=1.0, aliases=["k"])
			kappa = fem.Constant(m, ScalarType(kappa_val))

			# --- boundary terms & measures ---
			bcs_dirichlet, a_extra, L_extra, dx, ds = self._prepare_boundary_forms(V, pde_config, u, v, "heat_transfer")

			# --- weak form ---
			a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
			L = fem.Constant(m, ScalarType(0.0)) * v * dx
			if a_extra is not None: a += a_extra
			if L_extra is not None: L += L_extra
				
			# --- linear solver options (configurable) ---
			petsc_opts, prefix = self._prepare_petsc_solver_options(pde_config, PETSc, default_prefix="ht_")
			
			# --- solve (handle API diffs across dolfinx versions) ---
			comm_size = mesh_comm.Get_size()
			comm_rank = mesh_comm.Get_rank()
			logger.info(f"Using mesh communicator for solve: size={comm_size}, rank={comm_rank} (mesh comm type: {type(mesh_comm)})")
			
			# Ensure PETSc uses the same communicator as the mesh
			try:
				# Some versions of LinearProblem accept a comm parameter
				problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts, petsc_options_prefix=prefix)
			except TypeError:
				# Some versions use options_prefix instead of petsc_options_prefix
				try:
					problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts, options_prefix=prefix)
				except TypeError:
					# Try without prefix parameter
					problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts)

			try:
				solution = problem.solve()
			except Exception as solve_error:
				# Check if it's an MPI rank error
				error_str = str(solve_error)
				if "Invalid rank" in error_str or "MPI" in error_str:
					logger.error(f"MPI communicator error during solve: {solve_error}")
					logger.error(f"Mesh communicator: size={comm_size}, rank={comm_rank}")
					logger.error(f"MPI.COMM_WORLD: size={MPI.COMM_WORLD.Get_size()}, rank={MPI.COMM_WORLD.Get_rank()}")
					# Try to get more information about the error
					import traceback
					logger.error(f"Full traceback: {traceback.format_exc()}")
					raise RuntimeError(f"MPI communicator mismatch during solve. Mesh comm size={comm_size}, MPI.COMM_WORLD size={MPI.COMM_WORLD.Get_size()}. Error: {solve_error}")
				raise

			self.solution = solution
			return self.extract_solution_data("heat_transfer")

		except Exception as e:
			# Make the underlying init error visible (so you don't only see __del__ noise)
			try:
				bc_types = [bc.get("type") for bc in (pde_config.get("boundary_conditions") or [])]
				logger.error(f"[heat] Failure details — "
							f"BC count={len(bc_types)} types={bc_types}, "
							f"family={family}, degree={degree}, kappa={kappa_val}")
			except Exception:
				pass
			logger.error(f"Heat transfer solver failed: {e}")
			raise

	def _apply_bcs_to_initial_condition(self, u_prev, V, m, pde_config, bcs_dirichlet, fdim):
		"""Apply Dirichlet boundary conditions to initial condition function."""
		if not bcs_dirichlet:
			return
		
		bc_list = pde_config.get("boundary_conditions", []) or []
		bc_applied_count = 0
		
		for bc_idx, bc_config in enumerate(bc_list):
			btype = (bc_config.get("type") or "").strip().lower()
			if btype not in ("temperature", "dirichlet", "fixed"):
				logger.debug(f"Skipping BC {bc_idx+1} in initial condition: type='{btype}' is not a Dirichlet BC")
				continue
			
			loc_raw = bc_config.get("location")
			if loc_raw is None or (isinstance(loc_raw, str) and not loc_raw.strip()):
				logger.warning(f"BC {bc_idx+1} has no location specified, skipping initial condition application")
				continue
			
			loc = str(loc_raw).strip().lower()
			val = bc_config.get("value")
			if val is None:
				logger.debug(f"BC {bc_idx+1} at '{loc}' has no value, skipping")
				continue
			
			try:
				val = float(val)  # Already in Kelvin from normalize_heat_bcs
			except (TypeError, ValueError):
				logger.debug(f"BC {bc_idx+1} at '{loc}' has invalid value '{val}', skipping")
				continue
			
			logger.debug(f"Processing BC {bc_idx+1} for initial condition: original_location='{loc_raw}', location='{loc}', value={val:.3f} K")
			
			# Get physical groups and geometry type for unified resolution
			physical_groups = {}
			if hasattr(self, 'mesh_data') and hasattr(self.mesh_data, 'physical_groups'):
				physical_groups = self.mesh_data.physical_groups
			
			geometry_type = (pde_config.get("mesh_parameters", {}).get("geometry_type") or
							pde_config.get("geometry_type") or "")
			
			# Use unified boundary location resolution
			dofs, method, confidence = self._resolve_boundary_location(loc, V, m, physical_groups, geometry_type, pde_config, fdim)
			
			# Apply the boundary value to initial condition
			if dofs is not None and (isinstance(dofs, np.ndarray) and dofs.size > 0 or (not isinstance(dofs, np.ndarray) and hasattr(dofs, '__len__') and len(dofs) > 0)):
				dofs_size = dofs.size if isinstance(dofs, np.ndarray) else len(dofs)
				u_prev.x.array[dofs] = val
				bc_applied_count += 1
				logger.debug(f"Applied BC value {val:.3f} K ({val - 273.15:.3f} C) to {dofs_size} DOFs at '{loc}' (method: {method}, confidence: {confidence:.2f})")
			else:
				logger.warning(f"Could not locate DOFs for BC at '{loc}' (no DOFs found), will be enforced during solve")
		
		logger.debug(f"Applied {bc_applied_count}/{len(bcs_dirichlet)} boundary conditions to initial condition")

	def _compute_steady_state_for_convergence(self, m, V, kappa, u, v, dx, a_extra, L_extra, bcs_dirichlet, petsc_opts, prefix, PETSc):
		"""Compute steady-state solution for convergence checking."""
		from dolfinx import fem
		from dolfinx.fem.petsc import LinearProblem
		import ufl
		
		ScalarType = PETSc.ScalarType
		
		logger.info("Computing steady-state solution for convergence checking...")
		try:
			# Build steady-state problem: -div(k*grad(u)) = 0
			a_steady = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
			L_steady = fem.Constant(m, ScalarType(0.0)) * v * dx
			if a_extra is not None:
				a_steady += a_extra
			if L_extra is not None:
				L_steady += L_extra
			
			logger.debug(f"Steady-state solve: applying {len(bcs_dirichlet)} Dirichlet boundary conditions")
			
			self._configure_petsc_single_process(petsc_opts, PETSc)
			
			problem_steady = LinearProblem(
				a_steady,
				L_steady,
				bcs=bcs_dirichlet,
				petsc_options=petsc_opts,
				petsc_options_prefix=prefix,
			)
			steady_state_solution = problem_steady.solve()
			logger.info("Steady-state solution computed successfully")
			return steady_state_solution
		except Exception as e:
			logger.warning(f"Could not compute steady-state solution for convergence: {e}. Using relative change method instead.")
			return None

	def _run_transient_time_stepping(self, problem, u_prev, final_solution, V, steady_state_solution, 
	                                 dt, max_steps, convergence_tol, check_convergence, initial_mean):
		"""Run transient time-stepping loop until convergence."""
		import time
		
		time_steps = [0.0]
		solutions_series = []
		
		# Record initial condition
		initial_values = self._map_scalar_dofs_to_gmsh(u_prev.x.array)
		initial_min = float(np.min(initial_values))
		initial_max = float(np.max(initial_values))
		initial_mean_val = float(np.mean(initial_values))
		solutions_series.append({
			"time": 0.0,
			"values": initial_values.tolist(),
			"min_value": initial_min,
			"max_value": initial_max,
			"mean_value": initial_mean_val,
		})
		logger.info(f"Initial temperature field at t=0.0: min={initial_min:.3f}, max={initial_max:.3f}, mean={initial_mean_val:.3f}")
		
		max_wall_time = 30.0  # seconds
		start_wall_time = time.time()
		prev_mean = initial_mean_val
		converged = False
		actual_steps = 0
		wall_time_exceeded = False
		
		for step in range(1, max_steps + 1):
			current_time = step * dt
			
			# Check wall time limit
			elapsed_wall_time = time.time() - start_wall_time
			if elapsed_wall_time >= max_wall_time:
				wall_time_exceeded = True
				logger.warning(f"Wall time limit of {max_wall_time}s reached at step {step} (t={current_time:.4f}, elapsed: {elapsed_wall_time:.2f}s). Stopping and saving data.")
				break
			
			time_steps.append(current_time)
			
			u_step = problem.solve()
			final_solution.x.array[:] = u_step.x.array
			u_prev.x.array[:] = u_step.x.array
			
			gmsh_values = self._map_scalar_dofs_to_gmsh(u_step.x.array)
			step_min = float(np.min(gmsh_values))
			step_max = float(np.max(gmsh_values))
			step_mean = float(np.mean(gmsh_values))
			solutions_series.append({
				"time": current_time,
				"values": gmsh_values.tolist(),
				"min_value": step_min,
				"max_value": step_max,
				"mean_value": step_mean,
			})
			
			# Check convergence
			if check_convergence and step > 1:
				if steady_state_solution is not None:
					diff = u_step.x.array - steady_state_solution.x.array
					norm_diff = np.linalg.norm(diff)
					norm_steady = np.linalg.norm(steady_state_solution.x.array)
					relative_norm_diff = norm_diff / max(norm_steady, 1.0)
					
					if relative_norm_diff < convergence_tol:
						converged = True
						actual_steps = step
						logger.info(f"Converged at step {step} (t={current_time:.4f}): relative norm difference {relative_norm_diff:.2e} < tolerance {convergence_tol:.2e}")
						logger.info(f"  Absolute norm difference: {norm_diff:.6e}, Steady-state norm: {norm_steady:.6e}")
						break
				else:
					mean_change = abs(step_mean - prev_mean)
					relative_change = mean_change / max(abs(prev_mean), 1.0)
					if relative_change < convergence_tol:
						converged = True
						actual_steps = step
						logger.info(f"Converged at step {step} (t={current_time:.4f}): relative change {relative_change:.2e} < tolerance {convergence_tol:.2e}")
						break
			
			prev_mean = step_mean
			actual_steps = step
			
			# Log every 10 steps or at the last step
			if step % 10 == 0 or step == max_steps:
				logger.info(f"Step {step}/{max_steps} (t={current_time:.4f}, dt={dt:.6f}): temperature min={step_min:.3f}, max={step_max:.3f}, mean={step_mean:.3f}")
		
		elapsed_wall_time = time.time() - start_wall_time
		if wall_time_exceeded:
			logger.warning(f"Simulation stopped due to wall time limit ({max_wall_time}s). Completed {actual_steps} steps in {elapsed_wall_time:.2f}s. Saving current state.")
		elif not converged and actual_steps >= max_steps:
			logger.warning(f"Did not converge after {actual_steps} steps (elapsed: {elapsed_wall_time:.2f}s). Consider increasing max_steps.")
		elif converged:
			logger.info(f"Simulation converged in {actual_steps} steps (elapsed: {elapsed_wall_time:.2f}s).")
		
		return time_steps, solutions_series, actual_steps, converged, wall_time_exceeded, elapsed_wall_time

	def _assemble_transient_result(self, base_result, final_solution, steady_state_solution, V, 
	                               time_steps, solutions_series, actual_steps, converged, wall_time_exceeded,
	                               elapsed_wall_time, dt, total_time, method, heat_capacity, initial_value):
		"""Assemble final result dictionary for transient simulation."""
		from dolfinx import fem
		
		actual_final_time = time_steps[-1] if time_steps else total_time
		
		self.solution = fem.Function(V, name="temperature")
		self.solution.x.array[:] = final_solution.x.array
		
		final_values = self._map_scalar_dofs_to_gmsh(final_solution.x.array)
		final_min = float(np.min(final_values))
		final_max = float(np.max(final_values))
		final_mean = float(np.mean(final_values))
		logger.info(f"Final temperature field at t={actual_final_time:.4f}: min={final_min:.3f}, max={final_max:.3f}, mean={final_mean:.3f}")
		
		# Add steady-state solution if available
		steady_state_data = None
		if steady_state_solution is not None:
			steady_state_values = self._map_scalar_dofs_to_gmsh(steady_state_solution.x.array)
			steady_state_min = float(np.min(steady_state_values))
			steady_state_max = float(np.max(steady_state_values))
			steady_state_mean = float(np.mean(steady_state_values))
			steady_state_data = {
				"time": float('inf'),
				"values": steady_state_values.tolist(),
				"min_value": steady_state_min,
				"max_value": steady_state_max,
				"mean_value": steady_state_mean,
			}
			logger.info(f"Steady-state solution: min={steady_state_min:.3f}, max={steady_state_max:.3f}, mean={steady_state_mean:.3f}")
		
		base_result["is_transient"] = True
		base_result["time_stepping"] = {
			"time_step": dt,
			"requested_total_time": total_time if total_time > 0 else None,
			"actual_total_time": time_steps[-1] if time_steps else 0.0,
			"actual_steps": actual_steps,
			"converged": converged,
			"wall_time_exceeded": wall_time_exceeded,
			"elapsed_wall_time": elapsed_wall_time,
			"method": method,
			"heat_capacity": heat_capacity,
			"steady_state_solution": steady_state_data,
		}
		base_result["time_steps"] = time_steps
		base_result["solutions"] = solutions_series
		base_result["time_series"] = solutions_series
		base_result["initial_conditions"] = {
			"temperature": initial_value,
			"description": "Uniform initial temperature value applied to domain",
		}
		
		return base_result

	def _solve_heat_transfer_transient(self, pde_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Solve transient heat conduction with implicit time-stepping."""
		logger.info("Solving transient heat transfer equation")
		try:
			from dolfinx import fem
			from dolfinx.fem.petsc import LinearProblem
			from petsc4py import PETSc
			import ufl

			ScalarType = PETSc.ScalarType

			if not hasattr(self, "domain"):
				raise RuntimeError("Mesh (self.domain) not initialized. Call solve_simulation first.")

			m = self.domain
			tdim = m.topology.dim
			fdim = tdim - 1
			m.topology.create_connectivity(fdim, tdim)
			m.topology.create_connectivity(tdim, fdim)

			V, family, degree = self._build_scalar_function_space(pde_config)

			u = ufl.TrialFunction(V)
			v = ufl.TestFunction(V)

			time_params = pde_config.get("time_stepping") or {}
			requested_dt = float(time_params.get("time_step", 0.0) or 0.0)
			total_time = float(time_params.get("total_time", 0.0) or 0.0)
			method = (time_params.get("method") or "backward_euler").strip().lower()

			if total_time <= 0.0:
				logger.warning("Invalid time-stepping parameters. Falling back to steady-state solver.")
				return self._solve_heat_transfer({**pde_config, "time_stepping": {}})

			supported_methods = {"backward_euler", "implicit_euler", "implicit"}
			if method not in supported_methods:
				logger.warning(f"Time-stepping method '{method}' not supported. Falling back to backward_euler.")
				method = "backward_euler"

			kappa_val = self._extract_material_value(pde_config, "thermal_conductivity", default=1.0, aliases=["k"])
			kappa = fem.Constant(m, ScalarType(kappa_val))

			density = self._extract_material_value(pde_config, "density", default=1.0)
			specific_heat = self._extract_material_value(pde_config, "specific_heat", default=1.0, aliases=["cp"])
			heat_capacity = density * specific_heat
			if heat_capacity <= 0.0 or not np.isfinite(heat_capacity):
				logger.debug(f"Invalid heat capacity ({heat_capacity}); defaulting to 1.0")
				heat_capacity = 1.0
			
			# Calculate thermal diffusivity: alpha = k / (rho * cp)
			thermal_diffusivity = kappa_val / (density * specific_heat) if (density * specific_heat) > 0 else 1.0
			
			# Calculate minimum element size for stability check
			# Estimate h_min from mesh geometry
			coords = m.geometry.x
			if coords.shape[0] > 1:
				# Compute minimum distance between nodes (approximate minimum element size)
				# Use a sample-based approach to avoid computing all pairwise distances
				n_samples = min(100, coords.shape[0])  # Sample up to 100 nodes
				sample_indices = np.linspace(0, coords.shape[0] - 1, n_samples, dtype=int)
				sample_coords = coords[sample_indices]
				
				# Compute distances from sample to all nodes
				min_distances = []
				for i, sample_coord in enumerate(sample_coords):
					# Distance from this sample to all other nodes
					distances = np.linalg.norm(coords - sample_coord, axis=1)
					# Remove zero distance (self)
					distances = distances[distances > 1e-10]
					if len(distances) > 0:
						min_distances.append(np.min(distances))
				
				if min_distances:
					h_min = float(np.mean(min_distances))
				else:
					# Fallback: estimate from mesh bounding box
					bbox_min = np.min(coords, axis=0)
					bbox_max = np.max(coords, axis=0)
					bbox_size = np.linalg.norm(bbox_max - bbox_min)
					h_min = bbox_size / 100.0
			else:
				# Fallback: estimate from mesh bounding box
				bbox_min = np.min(coords, axis=0)
				bbox_max = np.max(coords, axis=0)
				bbox_size = np.linalg.norm(bbox_max - bbox_min)
				# Estimate h_min as a fraction of bounding box (rough estimate)
				h_min = bbox_size / 100.0  # Conservative estimate
			
			# Calculate maximum stable time step based on stability criterion
			# For implicit Euler, method is unconditionally stable, but we want accuracy
			# Conservative estimate: dt_max = h_min^2 / (2 * alpha) for explicit-like accuracy
			# For implicit, we can use larger steps, but use conservative estimate
			if thermal_diffusivity > 0:
				dt_max_stable = (h_min ** 2) / (2.0 * thermal_diffusivity)
			else:
				dt_max_stable = total_time / 100.0  # Fallback: use 100 steps
			
			# Determine time step: use max stable dt unless a meaningful requested_dt is provided
			# A requested_dt is "meaningful" if it's at least 10% of max stable (not too small)
			meaningful_requested_dt = requested_dt > 0.0 and requested_dt >= 0.1 * dt_max_stable
			
			if meaningful_requested_dt:
				# Use requested dt, but cap at max stable for safety
				dt = min(requested_dt, dt_max_stable)
				if dt < requested_dt:
					logger.warning(f"Requested time_step {requested_dt} exceeds maximum stable time_step {dt_max_stable:.6f}. Using {dt:.6f}")
				else:
					logger.info(f"Using requested time_step: {dt:.6f} (max stable: {dt_max_stable:.6f})")
			else:
				# No meaningful requested_dt, use max stable dt
				dt = dt_max_stable
				if requested_dt > 0.0:
					logger.info(f"Requested time_step {requested_dt:.6f} is too small compared to max stable {dt_max_stable:.6f}. Using max stable dt: {dt:.6f}")
				else:
					logger.info(f"Auto-determined time_step: {dt:.6f} (max stable: {dt_max_stable:.6f}, h_min: {h_min:.6f}, alpha: {thermal_diffusivity:.6f})")
			
			# Running until convergence - no fixed total_time or num_steps limit
			logger.info(f"Transient heat transfer: time_step={dt:.6f}, method={method}, running until convergence")
			logger.info(f"  Stability: h_min={h_min:.6f}, thermal_diffusivity={thermal_diffusivity:.6f}, dt_max_stable={dt_max_stable:.6f}")
			logger.info(f"  Using time_step={dt:.6f} (requested: {requested_dt if requested_dt > 0.0 else 'auto'})")

			u_prev = fem.Function(V, name="u_prev")
			# Extract initial value - if in Celsius, convert to Kelvin (BCs are already in Kelvin)
			initial_value_celsius = self._extract_initial_scalar_value(pde_config, default=0.0)
			# Convert to Kelvin: assume initial value is in Celsius
			initial_value = initial_value_celsius + 273.15
			logger.info(f"Initial temperature: {initial_value_celsius} C = {initial_value:.3f} K")
			
			# Set initial condition uniformly
			u_prev.x.array[:] = initial_value
			
			# Prepare boundary conditions
			bcs_dirichlet, a_extra, L_extra, dx, ds = self._prepare_boundary_forms(V, pde_config, u, v, "heat_transfer")
			logger.info(f"Transient heat transfer: prepared {len(bcs_dirichlet)} Dirichlet boundary conditions")
			
			# Apply boundary conditions to initial condition
			self._apply_bcs_to_initial_condition(u_prev, V, m, pde_config, bcs_dirichlet, fdim)
			logger.info(f"Initial temperature field: body at {initial_value:.3f} K ({initial_value_celsius} C), boundaries at specified BC values")

			mass_coeff = heat_capacity / dt
			a_form = (mass_coeff * ufl.inner(u, v) + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * dx
			if a_extra is not None:
				a_form += a_extra

			L_form = (mass_coeff * ufl.inner(u_prev, v)) * dx
			if L_extra is not None:
				L_form += L_extra
				logger.info(f"Transient heat transfer: Added L_extra (Neumann BCs/heat flux terms) to weak form")
			else:
				logger.warning(f"Transient heat transfer: L_extra is None - no Neumann BCs/heat flux terms found!")

			time_petsc_cfg = dict(pde_config)
			if time_params.get("petsc_options_prefix"):
				time_petsc_cfg["petsc_options_prefix"] = time_params["petsc_options_prefix"]
			petsc_opts, prefix = self._prepare_petsc_solver_options(time_petsc_cfg, PETSc, default_prefix="ht_ts_")

			logger.info(f"Transient solve: applying {len(bcs_dirichlet)} Dirichlet boundary conditions")
			for i, bc in enumerate(bcs_dirichlet):
				try:
					dof_indices = bc.dof_indices()
					logger.info(f"  Dirichlet BC {i+1}: {len(dof_indices)} DOFs constrained")
				except Exception:
					logger.debug(f"  Dirichlet BC {i+1}: (could not extract DOF count)")
			
			problem = LinearProblem(
				a_form,
				L_form,
				bcs=bcs_dirichlet,
				petsc_options=petsc_opts,
				petsc_options_prefix=prefix,
			)

			final_solution = fem.Function(V, name="u_final")

			# Convergence parameters
			convergence_tol = float(time_params.get("convergence_tolerance", 1e-4))
			max_steps = int(time_params.get("max_steps", 10000))
			check_convergence = time_params.get("check_convergence", True)
			
			# Compute steady-state solution for convergence checking
			steady_state_solution = None
			if check_convergence:
				steady_state_solution = self._compute_steady_state_for_convergence(
					m, V, kappa, u, v, dx, a_extra, L_extra, bcs_dirichlet, petsc_opts, prefix, PETSc
				)
			
			if steady_state_solution is not None:
				logger.info(f"Convergence checking enabled (steady-state comparison): tolerance={convergence_tol}, max_steps={max_steps} (safety limit only)")
			else:
				logger.info(f"Convergence checking enabled (relative change): tolerance={convergence_tol}, max_steps={max_steps} (safety limit only)")
			
			logger.info(f"Starting time-stepping: running until convergence (max_steps={max_steps} as safety limit, max_wall_time=30.0s)")
			
			# Get initial mean for convergence checking
			initial_values = self._map_scalar_dofs_to_gmsh(u_prev.x.array)
			initial_mean = float(np.mean(initial_values))
			
			# Run time-stepping loop
			time_steps, solutions_series, actual_steps, converged, wall_time_exceeded, elapsed_wall_time = \
				self._run_transient_time_stepping(
					problem, u_prev, final_solution, V, steady_state_solution,
					dt, max_steps, convergence_tol, check_convergence, initial_mean
				)

			# Assemble and return result
			base_result = self.extract_solution_data("heat_transfer")
			if not base_result.get("success"):
				return base_result

			return self._assemble_transient_result(
				base_result, final_solution, steady_state_solution, V,
				time_steps, solutions_series, actual_steps, converged, wall_time_exceeded,
				elapsed_wall_time, dt, total_time, method, heat_capacity, initial_value
			)

		except Exception as e:
			logger.error(f"Transient heat transfer solver failed: {e}", exc_info=True)
			raise

	def _solve_solid_mechanics(self, pde_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Small-strain linear elasticity in 1D/2D/3D with Dirichlet/Neumann/pressure BCs."""
		logger.debug("Solving solid mechanics with DOLFINx")
		try:
			from dolfinx import fem, mesh as _mesh
			from dolfinx.fem.petsc import LinearProblem
			import ufl
			import numpy as np
			from petsc4py import PETSc

			ScalarType = PETSc.ScalarType

			if not hasattr(self, "domain"):
				raise RuntimeError("Mesh (self.domain) not initialized. Call solve_simulation first.")

			m = self.domain
			tdim = int(m.topology.dim)
			gdim = int(m.geometry.dim)
			V, family, degree = self._build_vector_function_space(pde_config, gdim)
			
			logger.debug(f"Created function space: family={family}, degree={degree}, gdim={gdim}")
			logger.debug(f"Function space V: {V}")

			# --- trial/test ---
			u = ufl.TrialFunction(V)
			v = ufl.TestFunction(V)

			# --- material parameters ---
			# Get material properties from the correct location
			material_props = pde_config.get("material_properties", {}) or {}
			mat = pde_config.get("material", {}) or {}
			
			# Use material_properties if available, otherwise fall back to material
			if material_props:
				E = float(material_props.get("youngs_modulus", material_props.get("E", 1.0)))
				nu = float(material_props.get("poisson_ratio", material_props.get("nu", 0.3)))
				logger.debug(f"Using material properties: E={E:.2e} Pa, nu={nu:.3f}")
			else:
				E = float(mat.get("E", 1.0))
				nu = float(mat.get("nu", 0.3))
				logger.debug(f"Using fallback material properties: E={E:.2e} Pa, nu={nu:.3f}")
			
			plane = (mat.get("plane", "strain") or "strain").lower() if tdim == 2 else None
			mu = E / (2.0 * (1.0 + nu))
			if tdim == 2 and plane == "stress":
				# plane stress: λ = 2μν/(1-ν)
				lmbda = 2.0 * mu * nu / (1.0 - nu)
			else:
				# 3D or plane strain
				lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

			def eps(w):
				return ufl.sym(ufl.grad(w))  # small-strain tensor

			def sigma(w):
				return lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)

			# --- measures & normals ---
			dx = ufl.Measure("dx", domain=m)
			ds = ufl.Measure("ds", domain=m)
			n = ufl.FacetNormal(m)

			# --- body force ---
			bf = pde_config.get("body_force", mat.get("body_force"))
			if bf is None:
				f_vec = np.zeros(gdim, dtype=np.float64)
			else:
				arr = np.array(bf, dtype=np.float64).ravel()
				if arr.size == 1:
					f_vec = np.full(gdim, arr[0], dtype=np.float64)
				else:
					f_vec = np.zeros(gdim, dtype=np.float64)
					f_vec[:min(gdim, arr.size)] = arr[:min(gdim, arr.size)]
			f = fem.Constant(m, f_vec)

			# --- boundary terms & measures ---
			bcs_dirichlet, a_extra, L_extra, dx, ds = self._prepare_boundary_forms(V, pde_config, u, v, "solid_mechanics")

			# --- bilinear & linear forms ---
			a = ufl.inner(sigma(u), eps(v)) * dx
			L = ufl.dot(f, v) * dx
			if a_extra is not None: a += a_extra
			if L_extra is not None: L += L_extra

			# --- solve ---
			petsc_opts, prefix = self._prepare_petsc_solver_options(pde_config, PETSc, default_prefix="sm_")
			problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts, petsc_options_prefix=prefix)
			solution = problem.solve()
			self.solution = solution  # used by extract_solution_data

			return self.extract_solution_data("solid_mechanics")

		except Exception as e:
			logger.error(f"Solid mechanics solver failed: {e}")
			raise e

	def _get_field_info(self, physics_type: str) -> Dict[str, Any]:
		"""Get field information based on physics type"""
		field_info = {
			"heat_transfer": {
				"primary_field": "temperature",
				"available_fields": ["temperature"],
				"fields": {
					"temperature": {
						"name": "Temperature",
						"units": "°C",
						"type": "scalar"
					}
				}
			},
			"solid_mechanics": {
				"primary_field": "displacement",  # For solver (vector field)
				"visualization_field": "deflection",  # For visualization (scalar field)
				"available_fields": ["displacement", "deflection", "stress", "strain"],
				"fields": {
					"displacement": {
						"name": "Displacement",
						"units": "m",
						"type": "vector"
					},
					"deflection": {
						"name": "Deflection",
						"units": "m",
						"type": "scalar"
					},
					"stress": {
						"name": "Stress",
						"units": "Pa",
						"type": "tensor"
					},
					"strain": {
						"name": "Strain",
						"units": "",
						"type": "tensor"
					}
				}
			}
		}
		
		# Handle unknown physics types
		if physics_type not in field_info:
			logger.warning(f"Unknown physics type: {physics_type}. Using generic fallback.")
			return {
				"primary_field": "generic_field",
				"available_fields": ["generic_field"],
				"fields": {
					"generic_field": {
						"name": f"{physics_type.title()} Field",
						"units": "",
						"type": "unknown"
					}
				}
			}
		
		return field_info[physics_type]

	def _get_mesh_cells(self) -> List[List[int]]:
		"""Get mesh cells for visualization"""
		cells = []
		for i in range(self.mesh.num_cells()):
			cell = self.mesh.cells()[i]
			cells.append(cell.tolist())
		return cells

	def _get_physical_groups_from_mesh_data(self):
		"""Get physical groups from mesh_data - ONLY source for boundary detection."""
		physical_groups = {}
		if hasattr(self, "mesh_data") and self.mesh_data:
			if isinstance(self.mesh_data, dict):
				pg_dict = self.mesh_data.get("physical_groups")
			elif hasattr(self.mesh_data, "physical_groups"):
				pg_dict = self.mesh_data.physical_groups
			else:
				pg_dict = None
			
			if pg_dict:
				physical_groups = self._normalize_physical_group_keys(pg_dict)
		return physical_groups

	def _get_facet_tags_from_mesh_data(self):
		"""Get facet_tags from mesh_data - ONLY source for facet tags."""
		facet_tags = None
		if hasattr(self, 'facet_tags') and self.facet_tags is not None:
			facet_tags = self.facet_tags
		elif hasattr(self, 'mesh_data') and self.mesh_data:
			if isinstance(self.mesh_data, dict):
				facet_tags = self.mesh_data.get("facet_tags")
			elif hasattr(self.mesh_data, 'facet_tags'):
				facet_tags = self.mesh_data.facet_tags
		return facet_tags

	def _apply_dirichlet_bc(self, bc, loc, V, m, physical_groups, geometry_type, pde_config, fdim, gdim, physics_type):
		"""Apply a single Dirichlet BC using physical groups."""
		import numpy as np
		from dolfinx import fem
		from petsc4py import PETSc
		ScalarType = PETSc.ScalarType
		
		dofs, method, confidence = self._resolve_boundary_location(loc, V, m, physical_groups, geometry_type, pde_config, fdim)
		
		if dofs is None or (isinstance(dofs, np.ndarray) and dofs.size == 0):
			logger.warning(f"[{physics_type}] Dirichlet: no DOFs at '{loc}', skipping.")
			return None
		
		if physics_type == "heat_transfer":
			val = float(bc.get("value", 0.0))
			logger.info(f"[{physics_type}] Dirichlet BC '{loc}': Located {dofs.size} DOFs using method '{method}' (confidence: {confidence:.2f}), applying value {val} K")
			return fem.dirichletbc(ScalarType(val), dofs, V)
		
		elif physics_type == "solid_mechanics":
			val = bc.get("value", 0.0)
			vec = np.array(val, dtype=np.float64).ravel()
			bcs = []
			if vec.size == 1:
				for i in range(gdim):
					V_i, _ = V.sub(i).collapse()
					dofs_i, _, _ = self._resolve_boundary_location(loc, V_i, m, physical_groups, geometry_type, pde_config, fdim)
					if dofs_i is not None and (isinstance(dofs_i, np.ndarray) and dofs_i.size > 0):
						bcs.append(fem.dirichletbc(ScalarType(vec[0]), dofs_i, V.sub(i)))
			else:
				for i in range(min(vec.size, gdim)):
					V_i, _ = V.sub(i).collapse()
					dofs_i, _, _ = self._resolve_boundary_location(loc, V_i, m, physical_groups, geometry_type, pde_config, fdim)
					if dofs_i is not None and (isinstance(dofs_i, np.ndarray) and dofs_i.size > 0):
						bcs.append(fem.dirichletbc(ScalarType(vec[i] if i < vec.size else 0.0), dofs_i, V.sub(i)))
			return bcs
		
		return None

	def _apply_neumann_bc(self, bc, loc, physical_groups, fdim, ds, physics_type, m, v, dx, u, n):
		"""Apply a single Neumann BC using physical groups."""
		from dolfinx import fem
		from petsc4py import PETSc
		import ufl
		import numpy as np
		
		ScalarType = PETSc.ScalarType
		btype = (bc.get("type") or "").strip().lower()
		
		# Find physical group for this location
		pg_match, tag, confidence = self._find_physical_group_by_location(loc, physical_groups, fdim)
		if tag is None or confidence <= 0.0:
			available_groups = list(physical_groups.keys())
			raise RuntimeError(
				f"[{physics_type}] Could not find physical group for location '{loc}' (Neumann BC). "
				f"Available physical groups: {available_groups}. "
				f"Physical groups MUST be provided by the mesh generator."
			)
		
		ds_sel = ds(tag)
		logger.info(f"[{physics_type}] ✓ Found physical group '{pg_match.name if pg_match else 'unknown'}' (tag {tag}, confidence: {confidence:.2f}) for location '{loc}'")
		
		if physics_type == "heat_transfer":
			if btype in ("flux", "neumann", "heat_flux", "heat flux"):
				q = ScalarType(float(bc.get("value", 0.0)))
				logger.info(f"[{physics_type}] Adding Neumann BC (heat flux) at '{loc}': q={q:.6f} W/m²")
				return q * v * ds_sel
			elif btype == "internal_source":
				q0 = ScalarType(float(bc.get("value", 0.0)))
				X = np.asarray(m.geometry.x)
				if X.ndim == 1:
					X = X.reshape(-1, 1)
				center = X.mean(axis=0)
				gmin, gmax = X.min(axis=0), X.max(axis=0)
				span = np.maximum(gmax - gmin, 1.0)
				r0 = 0.05 * float(np.max(span))
				xc = [float(center[i] if i < X.shape[1] else 0.0) for i in range(m.geometry.dim)]
				d2 = sum((ufl.SpatialCoordinate(m)[i] - xc[i])**2 for i in range(m.geometry.dim))
				chi = ufl.exp(-d2/(r0*r0))
				return q0 * chi * v * dx
			elif btype in ("robin", "convection", "radiation"):
				raw = bc.get("value", {})
				if isinstance(raw, dict):
					h = raw.get("h", raw.get("h_coeff", raw.get("h_rad", 0.0)))
					Tinf = raw.get("Tinf", raw.get("T_inf", raw.get("ambient", 0.0)))
				else:
					h = bc.get("h", bc.get("h_coeff", bc.get("h_rad", 0.0)))
					Tinf = bc.get("Tinf", bc.get("T_inf", bc.get("ambient", 0.0)))
				h = float(h or 0.0)
				Tinf = float(Tinf or 0.0)
				if h == 0.0:
					return None, None
				hc = ScalarType(h)
				Tc = ScalarType(Tinf)
				return hc * Tc * v * ds_sel, hc * u * v * ds_sel  # L_term, a_term
		
		elif physics_type == "solid_mechanics":
			if btype in ("traction", "neumann"):
				tval = bc.get("value", 0.0)
				arr = np.array(tval, dtype=np.float64).ravel()
				tvec = np.zeros(m.geometry.dim, dtype=np.float64)
				tvec[:min(m.geometry.dim, arr.size)] = arr[:min(m.geometry.dim, arr.size)]
				if np.allclose(tvec, 0.0):
					return None
				t = fem.Constant(m, tvec)
				return ufl.dot(t, v) * ds_sel
			elif btype == "pressure":
				pval = float(bc.get("value", 0.0))
				p = fem.Constant(m, ScalarType(pval))
				return (-p) * ufl.dot(n, v) * ds_sel
		
		return None

	def _prepare_boundary_forms(self, V, pde_config, u, v, physics_type="heat_transfer"):
		"""
		Build Dirichlet BCs and boundary RHS/stiffness contributions using ONLY physical groups.
		
		Returns: (bcs_dirichlet, a_extra, L_extra, dx, ds)
		"""
		import numpy as np
		import ufl
		from dolfinx import fem
		from petsc4py import PETSc

		ScalarType = PETSc.ScalarType
		m = V.mesh
		tdim = int(m.topology.dim)
		fdim = tdim - 1
		gdim = int(m.geometry.dim)

		dx = ufl.Measure("dx", domain=m)
		n = ufl.FacetNormal(m)

		bc_list = pde_config.get("boundary_conditions", []) or []
		logger.info(f"[{physics_type}] Processing {len(bc_list)} boundary conditions using physical groups")
		
		# Get physical groups and facet_tags from mesh_data (ONLY source)
		physical_groups = self._get_physical_groups_from_mesh_data()
		facet_tags = self._get_facet_tags_from_mesh_data()
		
		if not physical_groups:
			logger.warning(f"[{physics_type}] No physical groups found in mesh_data - boundary conditions may fail")
		
		# Create ds measure with facet_tags if available
		if facet_tags is not None:
			ds = ufl.Measure("ds", domain=m, subdomain_data=facet_tags)
			logger.info(f"[{physics_type}] Using facet_tags from mesh_data for ds measure")
		else:
			ds = ufl.Measure("ds", domain=m)
			logger.warning(f"[{physics_type}] No facet_tags from mesh_data - Neumann BCs will fail")
		
		geometry_type = pde_config.get("mesh_parameters", {}).get("geometry_type", "") or pde_config.get("geometry_type", "")
		
		bcs_dirichlet = []
		a_extra = None
		L_extra = None

		# Process all boundary conditions
		for bc in bc_list:
			btype = (bc.get("type") or "").strip().lower()
			loc = bc.get("location", "all_boundary")
			
			logger.info(f"[{physics_type}] Processing BC: type='{btype}', location='{loc}', value={bc.get('value', 'N/A')}")

			# Dirichlet BCs
			if physics_type == "heat_transfer" and btype in ("temperature", "dirichlet", "fixed"):
				bc_result = self._apply_dirichlet_bc(bc, loc, V, m, physical_groups, geometry_type, pde_config, fdim, gdim, physics_type)
				if bc_result:
					if isinstance(bc_result, list):
						bcs_dirichlet.extend(bc_result)
					else:
						bcs_dirichlet.append(bc_result)
			
			elif physics_type == "solid_mechanics" and btype in ("displacement", "dirichlet", "fixed"):
				bc_result = self._apply_dirichlet_bc(bc, loc, V, m, physical_groups, geometry_type, pde_config, fdim, gdim, physics_type)
				if bc_result:
					if isinstance(bc_result, list):
						bcs_dirichlet.extend(bc_result)
					else:
						bcs_dirichlet.append(bc_result)
			
			# Neumann BCs
			else:
				result = self._apply_neumann_bc(bc, loc, physical_groups, fdim, ds, physics_type, m, v, dx, u, n)
				if result is not None:
					if isinstance(result, tuple):  # Robin BC returns (L_term, a_term)
						L_term, a_term = result
						L_extra = (L_extra + L_term) if L_extra is not None else L_term
						a_extra = (a_extra + a_term) if a_extra is not None else a_term
					else:  # Single term for L_extra
						L_extra = (L_extra + result) if L_extra is not None else result

		return bcs_dirichlet, a_extra, L_extra, dx, ds

# Create a global instance
fenics_solver = FEniCSSolver()
