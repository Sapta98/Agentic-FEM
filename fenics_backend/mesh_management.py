"""
Mesh management and vertex mapping utilities
"""

import logging
import numpy as np
from typing import Optional, List
from scipy.spatial import cKDTree
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

logger = logging.getLogger(__name__)


def reset_vertex_mapping_cache():
	"""Reset cached vertex mapping data (call when mesh changes)."""
	return {
		"_gmsh_vertices_original": None,
		"_gmsh_vertices_for_mapping": None,
		"_dolfinx_to_gmsh_indices": None,
	}


def ensure_scalar_vertex_mapping(domain, original_mesh_data, cache_dict):
	"""Ensure mapping between DOLFINx vertices and original GMSH vertices is prepared."""
	if cache_dict.get("_gmsh_vertices_original") is not None and cache_dict.get("_dolfinx_to_gmsh_indices") is not None:
		return

	if domain is None:
		raise RuntimeError("Mesh (domain) not initialized. Cannot build vertex mapping.")

	if original_mesh_data is None:
		raise RuntimeError("Original mesh data not available. Cannot build vertex mapping.")

	gmsh_vertices_original = np.array(original_mesh_data["vertices"])
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

	cache_dict["_gmsh_vertices_original"] = gmsh_vertices_original
	cache_dict["_gmsh_vertices_for_mapping"] = gmsh_vertices_for_mapping
	cache_dict["_dolfinx_to_gmsh_indices"] = dolfinx_to_gmsh_indices


def map_scalar_dofs_to_gmsh(solution_values: np.ndarray, domain, original_mesh_data, cache_dict, solution_function=None) -> np.ndarray:
	"""
	Map scalar FEM solution DOFs to original GMSH vertex ordering.
	
	Args:
		solution_values: DOF values from solution.x.array
		domain: DOLFINx mesh
		original_mesh_data: Original GMSH mesh data
		cache_dict: Cache for vertex mappings
		solution_function: Optional solution function to evaluate at vertices (more accurate for higher-order elements)
	
	Returns:
		Solution values mapped to GMSH vertex ordering
	"""
	ensure_scalar_vertex_mapping(domain, original_mesh_data, cache_dict)

	num_vertices = cache_dict["_gmsh_vertices_original"].shape[0]
	gmsh_solution_values = np.full(num_vertices, np.nan, dtype=float)

	# If solution function is provided, evaluate at GMSH vertex coordinates (more accurate)
	if solution_function is not None:
		try:
			from dolfinx import fem
			gmsh_vertices = cache_dict["_gmsh_vertices_original"]
			gdim = int(domain.geometry.dim)
			# Evaluate solution at each GMSH vertex
			for gmsh_idx in range(num_vertices):
				vertex_coords = gmsh_vertices[gmsh_idx, :gdim]
				try:
					value = solution_function.eval(vertex_coords, domain)
					if value.size > 0:
						gmsh_solution_values[gmsh_idx] = float(value[0])
				except Exception:
					pass  # Will fall back to DOF mapping below
		except Exception as e:
			logger.debug(f"Could not evaluate solution function at vertices: {e}, falling back to DOF mapping")

	# Fallback: Map DOFs to vertices (works correctly for P1 elements)
	# For P1 elements, DOF index = vertex index
	logger.debug(f"Mapping {len(solution_values)} DOF values to {num_vertices} GMSH vertices")
	mapped_count = 0
	for dolfinx_idx, gmsh_idx in enumerate(cache_dict["_dolfinx_to_gmsh_indices"]):
		if dolfinx_idx >= len(solution_values):
			break
		value = float(solution_values[dolfinx_idx])
		if np.isnan(gmsh_solution_values[gmsh_idx]):
			gmsh_solution_values[gmsh_idx] = value
			mapped_count += 1
		else:
			# If already set (from function evaluation), average with DOF value
			gmsh_solution_values[gmsh_idx] = 0.5 * (gmsh_solution_values[gmsh_idx] + value)
	logger.debug(f"Mapped {mapped_count} DOF values directly to GMSH vertices")

	# Fill any unmapped vertices using nearest neighbor interpolation
	unmapped = np.isnan(gmsh_solution_values)
	if np.any(unmapped):
		mapped_coords = cache_dict["_gmsh_vertices_for_mapping"][~unmapped]
		mapped_values = gmsh_solution_values[~unmapped]
		unmapped_coords = cache_dict["_gmsh_vertices_for_mapping"][unmapped]
		if len(mapped_coords) > 0 and len(unmapped_coords) > 0:
			tree = cKDTree(mapped_coords)
			_, nearest = tree.query(unmapped_coords, k=1)
			gmsh_solution_values[unmapped] = mapped_values[nearest]

	return gmsh_solution_values


def create_dolfinx_mesh(msh_file: Optional[str], comm=MPI.COMM_SELF):
	"""
	Create DOLFINx mesh from exported .msh file.
	
	Returns: (mesh, cell_tags, facet_tags)
	"""
	if not msh_file:
		raise ValueError("msh_file path is required")
	
	logger.debug(f"Loading mesh from {msh_file}")
	
	try:
		mesh_tuple = gmshio.read_from_msh(msh_file, comm, gdim=0)
		mesh, cell_tags, facet_tags = mesh_tuple[0], mesh_tuple[1], mesh_tuple[2]
		logger.debug(f"Successfully loaded mesh: {len(mesh.geometry.x)} vertices")
		return mesh, cell_tags, facet_tags
	except Exception as e:
		logger.error(f"Failed to load mesh from {msh_file}: {e}")
		raise


def get_mesh_cells(mesh):
	"""Extract cell connectivity from mesh."""
	try:
		cells = mesh.topology.connectivity(mesh.topology.dim, 0)
		if cells is None:
			return []
		cell_list = []
		for i in range(mesh.topology.index_map(mesh.topology.dim).size_local):
			cell_vertices = cells.links(i)
			cell_list.append(cell_vertices.tolist())
		return cell_list
	except Exception as e:
		logger.warning(f"Could not extract mesh cells: {e}")
		return []

