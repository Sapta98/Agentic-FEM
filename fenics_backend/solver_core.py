"""
FEniCS Solver Core - Main solver class that ties all modules together
"""

import os
# CRITICAL: Set environment variables BEFORE importing MPI/DOLFINx to force single-process execution
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import numpy as np
from typing import Dict, Any, Optional
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

from .mesh_management import reset_vertex_mapping_cache, create_dolfinx_mesh
from .physical_groups import resolve_boundary_location
from .bc_normalization import normalize_bcs_by_physics
from .solution_extraction import extract_solution_data
from .solvers.heat_transfer import solve_heat_transfer
from .solvers.solid_mechanics import solve_solid_mechanics

logger = logging.getLogger(__name__)


class FEniCSSolver:
	"""FEniCS-based PDE solver for finite element simulations"""

	def __init__(self):
		self.mesh = None
		self.function_space = None
		self.solution = None
		self.mesh_data = None
		self.domain = None
		self.original_mesh_data = None
		self.boundary_conditions = []
		self.cell_tags = None
		self.facet_tags = None
		self.msh_file_path = None  # Path to exported .msh mesh file
		self._cache_dict = reset_vertex_mapping_cache()
		
		# Ensure MPI is initialized for single-process execution
		if not MPI.Is_initialized():
			MPI.Init()
			logger.debug("MPI initialized in FEniCSSolver.__init__")
		else:
			logger.debug("MPI already initialized")
		
		# Log MPI communicator information
		comm_size = MPI.COMM_WORLD.Get_size()
		comm_rank = MPI.COMM_WORLD.Get_rank()
		logger.info(f"MPI.COMM_WORLD at initialization: size={comm_size}, rank={comm_rank}")
		
		if comm_size > 1:
			logger.warning(f"WARNING: MPI.COMM_WORLD has {comm_size} processes at initialization. All mesh operations will use COMM_SELF to avoid errors.")
		
		# Initialize PETSc for single-process execution
		try:
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
			logger.debug(f"Cell tags values shape: {self.cell_tags.values.shape}")
			logger.debug(f"Cell tags unique values: {np.unique(self.cell_tags.values)}")
			logger.debug(f"Cell tags indices shape: {self.cell_tags.indices.shape}")
			unique_tags, counts = np.unique(self.cell_tags.values, return_counts=True)
			for tag, count in zip(unique_tags, counts):
				logger.debug(f"  Tag {tag}: {count} cells")
		else:
			logger.debug("No cell tags available")
			
		if self.facet_tags is not None:
			logger.debug(f"Facet tags values shape: {self.facet_tags.values.shape}")
			logger.debug(f"Facet tags unique values: {np.unique(self.facet_tags.values)}")
			logger.debug(f"Facet tags indices shape: {self.facet_tags.indices.shape}")
			unique_tags, counts = np.unique(self.facet_tags.values, return_counts=True)
			for tag, count in zip(unique_tags, counts):
				logger.debug(f"  Tag {tag}: {count} facets")
		else:
			logger.debug("No facet tags available")

	def solve_simulation(self, config: Dict[str, Any], mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Run the simulation selected by `config` on the mesh described by `mesh_data`.
		Returns a dict with solution/visualization payloads (from the solver).
		"""
		logger.debug("Starting simulation")

		# Store original mesh data for later use in solution extraction
		self.original_mesh_data = mesh_data
		self._cache_dict = reset_vertex_mapping_cache()

		# ---- mesh validation ----
		if not isinstance(mesh_data, dict) or "vertices" not in mesh_data:
			raise ValueError("mesh_data must include a 'vertices' array and cell blocks")
		if all(k not in mesh_data.get("cells", mesh_data) for k in ("tetra", "tetrahedron", "triangle", "triangle_2nd", "line", "line_2nd")):
			logger.warning("mesh_data has no recognized 'cells'; attempting to use top-level keys.")

		# CRITICAL: Extract physical groups from mesh_data FIRST (before creating DOLFINx mesh)
		physical_groups_from_mesh_data = mesh_data.get('physical_groups')
		logger.info(f"Checking for physical_groups in mesh_data: present={physical_groups_from_mesh_data is not None}, type={type(physical_groups_from_mesh_data)}, keys={list(mesh_data.keys())[:20] if isinstance(mesh_data, dict) else 'N/A'}")
		if physical_groups_from_mesh_data:
			logger.info(f"Found physical groups in mesh_data: {len(physical_groups_from_mesh_data)} groups, keys={list(physical_groups_from_mesh_data.keys())[:10] if isinstance(physical_groups_from_mesh_data, dict) else 'N/A'}")
			# Convert dict-based physical groups back to PhysicalGroupWrapper objects for compatibility
			converted_physical_groups = {}
			for name, group in physical_groups_from_mesh_data.items():
				if isinstance(group, dict):
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
		
		# CRITICAL: Use COMM_SELF to ensure single-process execution
		# DOLFINx may internally try to use MPI operations, so we must ensure
		# the communicator is explicitly set to single-process
		mesh_comm = MPI.COMM_SELF
		
		try:
			mesh_tuple = gmshio.read_from_msh(msh_file, mesh_comm, rank=0)
			if not isinstance(mesh_tuple, (list, tuple)):
				raise RuntimeError("gmshio.read_from_msh returned unexpected type")
			if len(mesh_tuple) < 3:
				raise RuntimeError("gmshio.read_from_msh returned fewer than 3 values")
			domain, cell_tags, facet_tags = mesh_tuple[:3]
			
			# Check domain communicator
			if hasattr(domain, 'comm'):
				domain_comm = domain.comm
				comm_size = domain_comm.Get_size()
				if comm_size > 1:
					logger.error(f"CRITICAL: Domain has {comm_size} processes! This will cause MPI errors.")
		except Exception as mesh_error:
			logger.error(f"Failed to read mesh from {msh_file}: {mesh_error}")
			import traceback
			logger.error(f"Traceback: {traceback.format_exc()}")
			raise
		
		self.domain = domain
		self.cell_tags = cell_tags
		self.facet_tags = facet_tags
		
		# Check mesh communicator
		if hasattr(self.domain, 'comm'):
			mesh_comm = self.domain.comm
			comm_size = mesh_comm.Get_size()
			if comm_size > 1:
				logger.error(f"CRITICAL: Mesh was created with {comm_size} processes! Expected size=1.")
				raise RuntimeError(f"Mesh communicator has {comm_size} processes but application is single-process. Mesh creation failed.")
		
		# Use physical groups from mesh_data if available
		# GMSH generates exact JSON names, so no normalization needed
		if physical_groups_from_mesh_data:
			physical_groups = physical_groups_from_mesh_data
			logger.info(f"Using {len(physical_groups_from_mesh_data)} physical groups from mesh_data with exact JSON names")
		else:
			logger.warning("Physical groups not found in mesh_data and no fallback is available without metadata")
			physical_groups = {}
		
		# Log information about cell tags and facet tags
		logger.debug(f"Cell tags: {self.cell_tags}")
		if self.cell_tags is not None:
			logger.debug(f"Cell tags values: {self.cell_tags.values}")
			logger.debug(f"Cell tags unique values: {np.unique(self.cell_tags.values)}")
		
		if self.facet_tags is not None and physical_groups_from_mesh_data:
			unique_facet_tags, facet_counts = np.unique(self.facet_tags.values, return_counts=True)
			for name, group in physical_groups_from_mesh_data.items():
				pg_tag = group.tag if hasattr(group, 'tag') else group.get('tag')
				if pg_tag not in unique_facet_tags:
					logger.warning(f"Physical group '{name}' (tag={pg_tag}): NO facets found in facet_tags!")
		
		self.inspect_mesh_tags()

		if self.domain is None:
			raise RuntimeError("Failed to create DOLFINx mesh")

		tdim = int(self.domain.topology.dim)
		gdim = int(self.domain.geometry.dim)
		nvtx = int(self.domain.geometry.x.shape[0])
		
		self.mesh_data = mesh_data  # keep original for inverse mapping & viz
		
		cell_count = self.domain.topology.index_map(tdim).size_global

		# ---- config parsing ----
		pde_config = (config or {}).get("pde_config", {}) or {}
		
		# Normalize BCs
		raw_bcs = pde_config.get("boundary_conditions", [])
		from .physical_groups import map_location_via_config
		geometry_type = pde_config.get("mesh_parameters", {}).get("geometry_type") or pde_config.get("geometry_type", "")
		normalized = normalize_bcs_by_physics(
			pde_config.get("physics_type"), 
			raw_bcs, 
			3, 
			canon_loc_func=lambda loc, gt: map_location_via_config(loc, gt or geometry_type),
			geometry_type=geometry_type
		)
		pde_config["boundary_conditions"] = normalized
		
		physics_type = (pde_config.get("physics_type", "heat_transfer") or "").strip().lower()
		family = (pde_config.get("family", "Lagrange") or "Lagrange").strip()
		degree = self.domain.geometry.cmap.degree

		# Precompute common connectivities (helps BCs, searches, etc.)
		self.domain.topology.create_connectivity(0, tdim)
		self.domain.topology.create_connectivity(tdim, 0)

		# Create resolve function that uses physical groups
		def resolve_boundary_func(loc, V, m, physical_groups, geometry_type, pde_config, fdim):
			return resolve_boundary_location(loc, V, m, physical_groups, geometry_type, pde_config, fdim, self.mesh_data)
		
		# Create extract function
		def extract_solution_func(solution, domain, mesh_data, cache_dict, physics_type):
			self.solution = solution
			return extract_solution_data(solution, domain, mesh_data, cache_dict, physics_type)
		
		# ---- delegate to physics-specific solver ----
		pde_config_with_params = {**pde_config, "family": family, "degree": degree}
		if physics_type == "heat_transfer":
			return solve_heat_transfer(
				self.domain, 
				pde_config_with_params, 
				self.mesh_data, 
				self.facet_tags,
				resolve_boundary_func,
				extract_solution_func,
				self._cache_dict
			)
		elif physics_type == "solid_mechanics":
			pde_config_with_params["gdim"] = gdim
			return solve_solid_mechanics(
				self.domain,
				pde_config_with_params,
				self.mesh_data,
				self.facet_tags,
				resolve_boundary_func,
				extract_solution_func,
				self._cache_dict
			)
		else:
			raise ValueError(f"Unknown physics type: {physics_type!r}")

	def extract_solution_data(self, physics_type: str = "heat_transfer") -> Dict[str, Any]:
		"""Extract solution data and map to GMSH vertices"""
		if self.solution is None:
			logger.error("No solution available.")
			return {"success": False, "status": "error", "message": "No solution available."}
		return extract_solution_data(self.solution, self.domain, self.original_mesh_data, self._cache_dict, physics_type)

