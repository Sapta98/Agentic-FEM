"""
Solution data extraction utilities
"""

import logging
import numpy as np
from typing import Dict, Any, List

from .mesh_management import map_scalar_dofs_to_gmsh, get_mesh_cells

logger = logging.getLogger(__name__)


def get_field_info(physics_type: str) -> Dict[str, Any]:
	"""Get field information for a given physics type."""
	field_info = {
		"heat_transfer": {
			"field_name": "temperature",
			"field_units": "K",
			"field_type": "scalar",
			"available_fields": ["temperature"]
		},
		"solid_mechanics": {
			"field_name": "displacement",
			"field_units": "m",
			"field_type": "vector",
			"available_fields": ["displacement", "stress", "strain"]
		}
	}
	return field_info.get(physics_type, {
		"field_name": "unknown",
		"field_units": "",
		"field_type": "scalar",
		"available_fields": []
	})


def extract_solution_data(solution, domain, original_mesh_data, cache_dict, physics_type: str = "heat_transfer") -> Dict[str, Any]:
	"""Extract solution data and map to GMSH vertices"""
	if solution is None:
		logger.error("No solution available.")
		return {"success": False, "status": "error", "message": "No solution available."}
	
	field_info = get_field_info(physics_type)
	field_name = field_info["field_name"]
	field_units = field_info["field_units"]
	field_type = field_info["field_type"]
	
	try:
		if field_type == "scalar":
			# Map scalar solution to GMSH vertices
			solution_values = solution.x.array
			# Pass solution function for accurate evaluation at vertices (works for all element types)
			gmsh_values = map_scalar_dofs_to_gmsh(solution_values, domain, original_mesh_data, cache_dict, solution_function=solution)
			
			# Get coordinates from original mesh data (GMSH mesh - same as preview)
			coordinates = np.array(original_mesh_data["vertices"]).tolist()
			
			# Get cells from original GMSH mesh data (NOT from DOLFINx - ensures same mesh as preview)
			cells = original_mesh_data.get("cells", {})
			
			# Get faces from original mesh data (needed for visualization)
			faces = original_mesh_data.get("faces", [])
			
			result = {
				"success": True,
				"status": "success",
				"coordinates": coordinates,
				"values": gmsh_values.tolist(),
				"cells": cells,
				"faces": faces,  # Include faces for mesh visualization
				"field_name": field_name,
				"field_units": field_units,
				"field_type": field_type,
				"physics_type": physics_type,
				"min_value": float(np.min(gmsh_values)),
				"max_value": float(np.max(gmsh_values)),
				"mean_value": float(np.mean(gmsh_values)),
			}
			
		elif field_type == "vector":
			# For vector fields (e.g., displacement), extract all components
			# Map each component to GMSH vertices
			gdim = int(domain.geometry.dim)
			components = {}
			gmsh_values_list = []
			
			# Get original GMSH vertex coordinates
			original_coords = np.array(original_mesh_data["vertices"])
			num_vertices = original_coords.shape[0]
			
			# Ensure vertex mapping is set up
			from .mesh_management import ensure_scalar_vertex_mapping
			ensure_scalar_vertex_mapping(domain, original_mesh_data, cache_dict)
			
			# Extract vector solution values
			# For vector function spaces, DOFs are stored as [u0_x, u0_y, u0_z, u1_x, u1_y, u1_z, ...]
			# where each vertex has gdim DOFs (for P1 elements)
			# For P2 elements, there are additional DOFs at edge midpoints
			solution_array = solution.x.array
			num_dofs = len(solution_array)
			
			# Get number of vertices in DOLFINx mesh
			num_dolfinx_vertices = domain.geometry.x.shape[0]
			
			# Extract each component
			for i in range(gdim):
				# For P1 elements: DOF index for component i of vertex j is j * gdim + i
				# Create array to hold component values at DOLFINx vertices
				component_values_dolfinx = np.zeros(num_dolfinx_vertices, dtype=float)
				
				# Extract component values from solution array
				# For P1 elements, we can directly index
				for vertex_idx in range(min(num_dolfinx_vertices, num_dofs // gdim)):
					dof_idx = vertex_idx * gdim + i
					if dof_idx < num_dofs:
						component_values_dolfinx[vertex_idx] = float(solution_array[dof_idx])
				
				# Map from DOLFINx vertices to GMSH vertices using the cached mapping
				dolfinx_to_gmsh = cache_dict["_dolfinx_to_gmsh_indices"]
				component_values_gmsh = np.zeros(num_vertices, dtype=float)
				
				for dolfinx_idx, gmsh_idx in enumerate(dolfinx_to_gmsh):
					if dolfinx_idx < len(component_values_dolfinx):
						component_values_gmsh[gmsh_idx] = component_values_dolfinx[dolfinx_idx]
				
				# Fill any unmapped vertices using nearest neighbor
				unmapped = component_values_gmsh == 0
				if np.any(unmapped):
					mapped_coords = original_coords[~unmapped]
					mapped_values = component_values_gmsh[~unmapped]
					unmapped_coords = original_coords[unmapped]
					if len(mapped_coords) > 0 and len(unmapped_coords) > 0:
						from scipy.spatial import cKDTree
						tree = cKDTree(mapped_coords)
						_, nearest = tree.query(unmapped_coords, k=1)
						component_values_gmsh[unmapped] = mapped_values[nearest]
				
				components[f"component_{i}"] = component_values_gmsh.tolist()
				gmsh_values_list.append(component_values_gmsh)
			
			# Compute magnitude for visualization
			magnitude = np.sqrt(sum(comp**2 for comp in gmsh_values_list))
			
			# Get coordinates (original + displacement for visualization)
			displacement = np.column_stack([gmsh_values_list[i] for i in range(gdim)])
			deformed_coordinates = (original_coords + displacement).tolist()
			
			# Get cells from original GMSH mesh data (NOT from DOLFINx - ensures same mesh as preview)
			cells = original_mesh_data.get("cells", {})
			
			# Get faces from original mesh data (needed for visualization)
			faces = original_mesh_data.get("faces", [])
			
			result = {
				"success": True,
				"status": "success",
				"coordinates": original_coords.tolist(),
				"deformed_coordinates": deformed_coordinates,
				"values": magnitude.tolist(),
				"components": components,
				"cells": cells,
				"faces": faces,  # Include faces for mesh visualization
				"field_name": field_name,
				"field_units": field_units,
				"field_type": field_type,
				"physics_type": physics_type,
				"min_value": float(np.min(magnitude)),
				"max_value": float(np.max(magnitude)),
				"mean_value": float(np.mean(magnitude)),
			}
		else:
			raise ValueError(f"Unknown field type: {field_type}")
		
		logger.debug(f"Extracted solution data: {field_name}, min={result['min_value']:.3f}, max={result['max_value']:.3f}")
		return result
		
	except Exception as e:
		logger.error(f"Error extracting solution data: {e}", exc_info=True)
		return {
			"success": False,
			"status": "error",
			"message": f"Error extracting solution data: {str(e)}"
		}

