"""
FEniCS-based PDE solver for finite element simulations
"""

import logging
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
		self.gmsh_model = None  # Store GMSH model separately

	def set_gmsh_model(self, gmsh_model):
		"""Set GMSH model from simulation manager"""
		self.gmsh_model = gmsh_model
		if gmsh_model is not None:
			logger.debug(f"GMSH model set in FEniCS solver: {type(gmsh_model)}")
		else:
			logger.warning("GMSH model set to None in FEniCS solver")

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

	def extract_solution_data(self, physics_type: str = "heat_transfer") -> Dict[str, Any]:
		"""Extract solution data using DOLFINx topology directly"""
		if getattr(self, "solution", None) is None:
			logger.error("No solution available.")
			return {"success": False, "status": "error", "message": "No solution available."}
		if not hasattr(self, "domain"):
			logger.error("No dolfinx mesh (self.domain) available.")
			return {"success": False, "status": "error", "message": "No dolfinx mesh available."}
		
		if True:
			u = self.solution
			domain = self.domain
			# Use DOLFINx mesh vertices for consistency with cell connectivity
			dolfinx_vertices = domain.geometry.x
			
			# Get GMSH mesh data directly
			gmsh_vertices = np.array(self.original_mesh_data["vertices"])
			original_cells = self.original_mesh_data.get('cells', {})
			original_faces = self.original_mesh_data.get('faces', [])
			
			logger.debug(f"Using GMSH mesh: {len(gmsh_vertices)} vertices, {len(original_cells)} cell types, {len(original_faces)} faces")
			
			# Get field info to determine field type
			field_info = self._get_field_info(physics_type)
			primary_field = field_info.get("primary_field", "temperature")
			field_type = field_info["fields"][primary_field]["type"]
			
			# Get solution values from FEniCS
			solution_values = u.x.array

			# Map FEniCS solution values to GMSH vertices
			# Create mapping from DOLFINx vertices to GMSH vertices
			gmsh_tree = cKDTree(gmsh_vertices)
			dolfinx_to_gmsh_indices = gmsh_tree.query(dolfinx_vertices, k=1)[1]
			
			if field_type == "scalar":
				# Scalar field (heat transfer)
				gmsh_solution_values = np.zeros(len(gmsh_vertices))
				for i, gmsh_idx in enumerate(dolfinx_to_gmsh_indices):
					gmsh_solution_values[gmsh_idx] = solution_values[i]
				
				# Calculate min and max values for proper colormap range
				min_val = float(np.min(gmsh_solution_values))
				max_val = float(np.max(gmsh_solution_values))
				logger.debug(f"Solution value range: {min_val:.3f} to {max_val:.3f}")
				
				solution_data = {
					"success": True,
					"status": "success",
					"physics_type": physics_type,
					"field_type": "scalar",
					"coordinates": gmsh_vertices.tolist(),  # Use GMSH vertices
					"values": gmsh_solution_values.tolist(),  # Mapped to GMSH vertices
					"faces": original_faces,  # Use original GMSH faces
					"cells": original_cells,   # Use original GMSH cells (dict format)
					"field_name": field_info["fields"][primary_field]["name"],
					"field_units": field_info["fields"][primary_field]["units"],
					"min_value": min_val,  # Add min value for colormap
					"max_value": max_val,  # Add max value for colormap
					"field_info": field_info,
					"mesh_info": {
						"num_vertices": len(gmsh_vertices),
						"num_dofs": len(solution_values),
						"dimension": int(domain.topology.dim),
						"geometry_dim": int(domain.geometry.dim)
					}
				}
				
			elif field_type == "vector":
				# Vector field (solid mechanics - displacement)
				gdim = int(domain.geometry.dim)
				gmsh_solution_values = np.zeros((len(gmsh_vertices), gdim))
				
				# For vector fields in FEniCS, the solution array is organized as:
				# [u0_x, u0_y, u0_z, u1_x, u1_y, u1_z, ...] for each node
				# We need to map this correctly to GMSH vertices
				
				# Map vector solution values to GMSH vertices
				for i, gmsh_idx in enumerate(dolfinx_to_gmsh_indices):
					# For each DOLFINx vertex, get all components
					for comp in range(gdim):
						dof_idx = i * gdim + comp
						if dof_idx < len(solution_values):
							gmsh_solution_values[gmsh_idx, comp] = solution_values[dof_idx]
				
				# For visualization, we'll use the magnitude of the displacement vector
				displacement_magnitude = np.linalg.norm(gmsh_solution_values, axis=1)
				
				solution_data = {
					"success": True,
					"status": "success",
					"physics_type": physics_type,
					"field_type": "vector",
					"coordinates": gmsh_vertices.tolist(),  # Use GMSH vertices
					"values": displacement_magnitude.tolist(),  # Magnitude for visualization
					"vector_values": gmsh_solution_values.tolist(),  # Full vector components
					"faces": original_faces,  # Use original GMSH faces
					"cells": original_cells,   # Use original GMSH cells (dict format)
					"field_name": field_info["fields"][primary_field]["name"],
					"field_units": field_info["fields"][primary_field]["units"],
					"field_info": field_info,
					"mesh_info": {
						"num_vertices": len(gmsh_vertices),
						"num_dofs": len(solution_values),
						"dimension": int(domain.topology.dim),
						"geometry_dim": int(domain.geometry.dim)
					}
				}
				
			else:
				# Tensor field or other - fallback to scalar magnitude
				logger.warning(f"Unsupported field type '{field_type}' for physics '{physics_type}'. Using scalar fallback.")
				gmsh_solution_values = np.zeros(len(gmsh_vertices))
				for i, gmsh_idx in enumerate(dolfinx_to_gmsh_indices):
					gmsh_solution_values[gmsh_idx] = solution_values[i]
				
				solution_data = {
					"success": True,
					"status": "success",
					"physics_type": physics_type,
					"field_type": "scalar",
					"coordinates": gmsh_vertices.tolist(),
					"values": gmsh_solution_values.tolist(),
					"faces": original_faces,
					"cells": original_cells,
					"field_name": field_info["fields"][primary_field]["name"],
					"field_units": field_info["fields"][primary_field]["units"],
					"field_info": field_info,
					"mesh_info": {
						"num_vertices": len(gmsh_vertices),
						"num_dofs": len(solution_values),
						"dimension": int(domain.topology.dim),
						"geometry_dim": int(domain.geometry.dim)
					}
				}
			
		logger.debug(f"GMSH-based solution extracted: {len(solution_values)} DOFs mapped to {len(gmsh_vertices)} GMSH vertices")
		return solution_data

	def _canon_loc(self, loc: Optional[str]) -> str:
		loc = (loc or "").strip().lower()
		aliases = {
			"left": "x_min", "right": "x_max",
			"top": "top", "bottom": "bottom",  # Keep as-is for geometry-specific handling
			"front": "z_min", "back": "z_max",
			"start": "x_min", "end": "x_max",
			"all": "all_boundary", "entire": "all_boundary", "complete": "all_boundary",
			"x_min":"x_min","x_max":"x_max","y_min":"y_min","y_max":"y_max","z_min":"z_min","z_max":"z_max",
			"all_boundary":"all_boundary",
			# Geometry-specific locations
			"curved_surface": "curved_surface", "curved surface": "curved_surface",
			"side": "curved_surface", "side_boundary": "curved_surface"
		}
		return aliases.get(loc, "all_boundary")

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

	def normalize_heat_bcs(self, raw_bcs: List[Dict[str, Any]], assume_celsius: bool=True) -> List[Dict[str, Any]]:
		"""
		Input examples (NLP):
		{"type":"temperature","location":"left","value":100,"bc_type":"dirichlet"}
		{"type":"insulated","location":"right","bc_type":"neumann"}
		Output (solver-ready):
		{"type":"dirichlet","location":"x_min","value":373.15}
		{"type":"neumann","location":"x_max","value":0.0}
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
				# Special case: for discs, a "center" heat flux is not a boundary condition;
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
		{"type":"force","location":"right","value":{"fx":1e5}}
		{"type":"pressure","location":"right","value":2e6}
		Output (solver-ready):
		{"type":"dirichlet","location":"x_min","value":[0,0,(0)]}
		{"type":"neumann",  "location":"y_max","value":[0,0,(0)]}  # zero traction
		{"type":"dirichlet","location":"x_max","value":[...]}
		{"type":"neumann",  "location":"x_max","value":[1e5,0,0]}
		{"type":"pressure", "location":"x_max","value":2e6}
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
				vec = self._vec(bc.get("value", 0.0), gdim)
				out.append({"type":"neumann", "location":loc, "value": vec})
				continue

			# Pressure scalar
			if t == "pressure":
				p = float(bc.get("value", 0.0))
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

		# ---- mesh validation ----
		if not isinstance(mesh_data, dict) or "vertices" not in mesh_data:
			raise ValueError("mesh_data must include a 'vertices' array and cell blocks")
		if all(k not in mesh_data.get("cells", mesh_data) for k in ("tetra", "tetrahedron", "triangle", "triangle_2nd", "line", "line_2nd")):
			logger.warning("mesh_data has no recognized 'cells'; attempting to use top-level keys.")

		# ---- create DOLFINx mesh using GMSH model from simulation manager ----
		logger.debug("Creating DOLFINx mesh from GMSH model...")
		
		if self.gmsh_model is not None:
			logger.debug(f"Using GMSH model from simulation manager: {type(self.gmsh_model)}")
			# Use GMSH model directly with dolfinx.io.gmsh
			logger.debug("Using GMSH model with dolfinx.io.gmsh.model_to_mesh")
			mesh_data = gmshio.model_to_mesh(
				self.gmsh_model, 
				MPI.COMM_SELF, 
				rank=0
			)
			# Extract domain and tags from mesh_data
			self.domain = mesh_data.mesh
			self.cell_tags = mesh_data.cell_tags
			self.facet_tags = mesh_data.facet_tags
			
			# Log information about cell tags and facet tags
			logger.debug(f"Cell tags: {self.cell_tags}")
			logger.debug(f"Cell tags values: {self.cell_tags.values}")
			logger.debug(f"Cell tags unique values: {np.unique(self.cell_tags.values)}")
			
			logger.debug(f"Facet tags: {self.facet_tags}")
			logger.debug(f"Facet tags values: {self.facet_tags.values}")
			logger.debug(f"Facet tags unique values: {np.unique(self.facet_tags.values)}")
			
			# Inspect mesh tags for detailed information
			self.inspect_mesh_tags()
			logger.debug(f"Mesh ready: {self.domain.topology.dim}D, vertices={self.domain.geometry.x.shape[0]}")
		else:
			logger.warning("No GMSH model available from simulation manager")

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
		normalized = self.normalize_bcs_by_physics(pde_config.get("physics_type"), raw_bcs, 3)
		pde_config["boundary_conditions"] = normalized
		
		# Log normalized boundary conditions
		logger.debug("Normalized boundary conditions:")
		for i, bc in enumerate(normalized):
			logger.debug(f"  BC {i+1}: {bc}")
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
			m = self.domain
			tdim = m.topology.dim
			fdim = tdim - 1
			m.topology.create_connectivity(fdim, tdim)
			m.topology.create_connectivity(tdim, fdim)
			
			# --- FE space (scalar) ---
			family = (pde_config.get("family", "Lagrange") or "Lagrange").strip()
			degree = int(pde_config.get("degree", 1))
				
			V = fem.functionspace(m, (family, degree))
				
			# --- trial/test ---
			u = ufl.TrialFunction(V)
			v = ufl.TestFunction(V)

			# --- material parameter ---
			# Prefer material_properties. Fall back to nested material or flat field.
			mat_props = pde_config.get("material_properties", {}) or {}
			mat_flat  = pde_config.get("material", {}) or {}
			kappa_val = (
				mat_props.get("thermal_conductivity")
				or mat_flat.get("thermal_conductivity")
				or pde_config.get("thermal_conductivity", 1.0)
			)
			kappa = fem.Constant(m, ScalarType(float(kappa_val)))

			# --- boundary terms & measures ---
			bcs_dirichlet, a_extra, L_extra, dx, ds = self._prepare_boundary_forms(V, pde_config, u, v, "heat_transfer")

			# --- weak form ---
			a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
			L = fem.Constant(m, ScalarType(0.0)) * v * dx
			if a_extra is not None: a += a_extra
			if L_extra is not None: L += L_extra
				
			# --- linear solver options (configurable) ---
			prefix = str(pde_config.get("petsc_options_prefix", pde_config.get("options_prefix", "ht_")))
			petsc_opts = {
				"ksp_type": pde_config.get("ksp_type", "preonly"),
				"pc_type":  pde_config.get("pc_type", "lu"),
			}
			if "pc_factor_mat_solver_type" in pde_config:
				petsc_opts["pc_factor_mat_solver_type"] = pde_config["pc_factor_mat_solver_type"]
			
			# --- solve (handle API diffs across dolfinx versions) ---
			try:
				problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts, petsc_options_prefix=prefix)
			except TypeError:
				# Some versions use options_prefix instead of petsc_options_prefix
				problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts, options_prefix=prefix)

			solution = problem.solve()

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

			# --- FE space (vector) ---
			family = (pde_config.get("family", "Lagrange") or "Lagrange").strip()
			degree = int(pde_config.get("degree", 1))
			
			# Create vector function space using correct DOLFINx syntax
			# Use basix element for vector spaces
			import basix
			element = basix.ufl.element(family, m.basix_cell(), degree, shape=(gdim,))
			V = fem.functionspace(m, element)
			
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
			petsc_opts = {
				"ksp_type": pde_config.get("ksp_type", "preonly"),
				"pc_type":  pde_config.get("pc_type", "lu"),
			}
			if "pc_factor_mat_solver_type" in pde_config:
				petsc_opts["pc_factor_mat_solver_type"] = pde_config["pc_factor_mat_solver_type"]

			prefix = str(pde_config.get("petsc_options_prefix", pde_config.get("options_prefix", "ht_")))
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
				"primary_field": "displacement",
				"available_fields": ["displacement", "stress", "strain"],
				"fields": {
					"displacement": {
						"name": "Displacement",
						"units": "mm",
						"type": "vector"
					},
					"stress": {
						"name": "Stress",
						"units": "MPa",
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

	def _prepare_boundary_forms(self, V, pde_config, u, v, physics_type="heat_transfer"):
		"""
		Build Dirichlet BCs and boundary RHS/stiffness contributions for heat transfer and solid mechanics:
		
		Heat Transfer:
		- 'dirichlet'/'temperature'/'fixed' -> T = const on boundary (DOF-based)
		- 'neumann'/'flux'/'heat_flux'      -> L += ∫ q v ds(tag)
		- 'robin'/'convection'/'radiation'  -> a += ∫ h u v ds(tag),  L += ∫ h Tinf v ds(tag)
		- 'insulated'/'symmetry'/'neumann_zero' -> no contribution
		
		Solid Mechanics:
		- 'dirichlet'/'displacement'/'fixed' -> u = const on boundary (vector DOF-based)
		- 'neumann'/'traction'               -> L += ∫ t·v ds(tag)
		- 'pressure'                        -> L += ∫ (-p) n·v ds(tag)
		- 'symmetry'/'roller'/'free'         -> no contribution
		
		Returns: (bcs_dirichlet, a_extra, L_extra, dx, ds)
		"""
		import numpy as np
		import ufl
		from dolfinx import fem, mesh as _mesh
		from petsc4py import PETSc

		ScalarType = PETSc.ScalarType

		m = V.mesh
		tdim = int(m.topology.dim)
		fdim = tdim - 1  # for 1D, fdim = 0 (vertices)
		gdim = int(m.geometry.dim)

		# Always provide measures; ds may be rebound below if we create facet_tags
		dx = ufl.Measure("dx", domain=m)
		ds = ufl.Measure("ds", domain=m)
		n = ufl.FacetNormal(m)

		bc_list = pde_config.get("boundary_conditions", []) or []
		
		# Determine if we have facet terms based on physics type
		if physics_type == "heat_transfer":
			has_facet_terms = any(
				(bc.get("type", "").lower() in ("flux", "neumann", "heat_flux", "robin", "convection", "radiation"))
				for bc in bc_list
			)
		elif physics_type == "solid_mechanics":
			has_facet_terms = any(
				(bc.get("type", "").lower() in ("traction", "neumann", "pressure"))
				for bc in bc_list
			)
		else:
			has_facet_terms = False

		# ----------------------------
		# 0) Try boundary_info meshtags
		# ----------------------------
		facet_tags = None
		name_to_tag = {}
		try:
			binfo = (self.mesh_data or {}).get("boundary_info", {}).get("boundaries", {})
			if binfo:
				# deterministic mapping (left=1, right=2 if present)
				names = ["left", "right"] + sorted([k for k in binfo.keys() if k not in ("left","right")])
				name_to_tag = {nm: i+1 for i, nm in enumerate([n for n in names if n in binfo])}

				# collect vertex indices for each named boundary (works for 1D)
				idxs, vals = [], []
				for nm, tag in name_to_tag.items():
					vids = list(map(int, binfo[nm].get("vertex_indices", [])))
					if vids:
						idxs.extend(vids)
						vals.extend([tag]*len(vids))

				if idxs:
					idxs = np.array(idxs, dtype=np.int32)
					vals = np.array(vals, dtype=np.int32)
					# sanity
					nv = m.geometry.x.shape[0]
					if idxs.max(initial=-1) >= nv or idxs.min(initial=0) < 0:
						raise ValueError(f"Boundary vertex index out of range w.r.t. mesh (0..{nv-1})")
					facet_tags = _mesh.meshtags(m, fdim, idxs, vals)
					ds = ufl.Measure("ds", domain=m, subdomain_data=facet_tags)
					
		except Exception as ex:
			# Non-fatal: we'll just fall back to geometric predicates
			try:
				logger.debug(f"[{physics_type}] boundary_info meshtags unavailable: {ex}")
			except Exception:
				pass

		bcs_dirichlet = []
		a_extra = None
		L_extra = None

		# If we have tags and only Dirichlet BCs, do a fast topological path
		if facet_tags is not None and not has_facet_terms:
			# First try to use physical groups for boundary mapping
			physical_group_mapping = {}
			if hasattr(self, 'mesh_data') and hasattr(self.mesh_data, 'physical_groups'):
				for name, group in self.mesh_data.physical_groups.items():
					if group.dim == fdim:  # Only facet groups
						physical_group_mapping[name] = group.tag
						logger.debug(f"Physical group mapping (fast path): '{name}' -> tag {group.tag}")
			
			for bc in bc_list:
				btype = (bc.get("type") or "").strip().lower()
				
				# Check if this is a Dirichlet BC for the current physics type
				if physics_type == "heat_transfer" and btype not in ("temperature", "dirichlet", "fixed"):
					continue
				elif physics_type == "solid_mechanics" and btype not in ("displacement", "dirichlet", "fixed"):
					continue

				loc = (bc.get("location") or "").strip().lower()
				
				# First try physical group mapping
				tag_id = None
				if loc in physical_group_mapping:
					tag_id = physical_group_mapping[loc]
					logger.debug(f"Using physical group mapping: '{loc}' -> tag {tag_id}")
				else:
					# Fall back to legacy mapping
					loc_alias = {
						"left":"left","x_min":"left",
						"right":"right","x_max":"right",
						"top":"top","y_max":"top",
						"bottom":"bottom","y_min":"bottom",
						"front":"front","z_min":"front",
						"back":"back","z_max":"back",
						"circumference":"curved_surface","curved_surface":"curved_surface",
						"center":"center",  # For internal sources
						"all":"all_boundary","all_boundary":"all_boundary"
					}
					loc_name = loc_alias.get(loc, loc)
					if loc_name in physical_group_mapping:
						tag_id = physical_group_mapping[loc_name]
						logger.debug(f"Using physical group mapping (alias): '{loc}' -> '{loc_name}' -> tag {tag_id}")
					elif loc_name in name_to_tag:
						tag_id = name_to_tag[loc_name]
						logger.debug(f"Using legacy mapping: '{loc}' -> '{loc_name}' -> tag {tag_id}")
				
				if tag_id is not None:
					facets = facet_tags.find(tag_id)
					if facets is not None and facets.size:
						if physics_type == "heat_transfer":
							# Scalar field
							dofs = fem.locate_dofs_topological(V, fdim, facets)
							if dofs.size:
								val = float(bc.get("value", 0.0))
								# Get coordinates of nodes where BC is applied
								bc_coords = m.geometry.x[dofs]
								logger.debug(f"[{physics_type}] Applying BC '{bc.get('type', 'unknown')}' at '{loc}' (tag {tag_id}) with value {val}")
								logger.debug(f"[{physics_type}] {dofs.size} DOFs constrained at coordinates:")
								for i, coord in enumerate(bc_coords):
									logger.debug(f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs[i]}: ({coord[0]:.6f})")
								bcs_dirichlet.append(fem.dirichletbc(ScalarType(val), dofs, V))
						elif physics_type == "solid_mechanics":
							# Vector field - collapse subspaces to avoid coordinate tabulation error
							val = bc.get("value", 0.0)
							vec = np.array(val, dtype=np.float64).ravel()
							if vec.size == 1:
								# Apply same value to all components
								for i in range(gdim):
									# Collapse the subspace to create a new function space with its own coordinate mapping
									V_i, _ = V.sub(i).collapse()
									dofs_i = fem.locate_dofs_topological(V_i, fdim, facets)
									if dofs_i.size:
										# Get coordinates of nodes where BC is applied
										bc_coords = m.geometry.x[dofs_i]
										logger.debug(f"[{physics_type}] Applying BC '{bc.get('type', 'unknown')}' at '{loc}' (tag {tag_id}) component {i} with value {vec[0]}")
										logger.debug(f"[{physics_type}] {dofs_i.size} DOFs constrained at coordinates:")
										for j, coord in enumerate(bc_coords):
											logger.debug(f"  DOF {dofs_i[j]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs_i[j]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs_i[j]}: ({coord[0]:.6f})")
										bcs_dirichlet.append(fem.dirichletbc(ScalarType(vec[0]), dofs_i, V.sub(i)))
							else:
								# Apply different values to each component
								for i in range(gdim):
									# Collapse the subspace to create a new function space with its own coordinate mapping
									V_i, _ = V.sub(i).collapse()
									dofs_i = fem.locate_dofs_topological(V_i, fdim, facets)
									if dofs_i.size:
										# Get coordinates of nodes where BC is applied
										bc_coords = m.geometry.x[dofs_i]
										logger.debug(f"[{physics_type}] Applying BC '{bc.get('type', 'unknown')}' at '{loc}' (tag {tag_id}) component {i} with value {vec[i] if i < vec.size else 0.0}")
										logger.debug(f"[{physics_type}] {dofs_i.size} DOFs constrained at coordinates:")
										for j, coord in enumerate(bc_coords):
											logger.debug(f"  DOF {dofs_i[j]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs_i[j]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs_i[j]}: ({coord[0]:.6f})")
										bcs_dirichlet.append(fem.dirichletbc(ScalarType(vec[i] if i < vec.size else 0.0), dofs_i, V.sub(i)))
				else:
					# If no matching tag name, silently skip (or fall back if you want)
					pass
			return bcs_dirichlet, a_extra, L_extra, dx, ds

		# -------------------------------------------
		# Fallback / General path
		#  - geometric predicates for locating facets
		#  - build facet_tags if needed (for ds(tag))
		# -------------------------------------------

		# geometry & predicates (safe across dims)
		X = np.asarray(m.geometry.x)
		if X.ndim == 1:
			X = X.reshape(-1, 1)
		if X.size == 0:
			raise RuntimeError(f"[{physics_type}] Mesh has no geometry points.")
		gdim = int(X.shape[1])

		gmin, gmax = X.min(axis=0), X.max(axis=0)
		span = np.maximum(gmax - gmin, 1.0)
		atol = 1e-10 * float(np.max(span)) + 1e-14

		def comp(x, i): return x[i] if i < x.shape[0] else np.zeros(x.shape[1], dtype=bool)
		def on_x_min(x): return np.isclose(comp(x, 0), gmin[0], atol=atol)
		def on_x_max(x): return np.isclose(comp(x, 0), gmax[0], atol=atol)
		def on_y_min(x): return np.isclose(comp(x, 1), gmin[1], atol=atol) if gdim >= 2 else np.zeros(x.shape[1], bool)
		def on_y_max(x): return np.isclose(comp(x, 1), gmax[1], atol=atol) if gdim >= 2 else np.zeros(x.shape[1], bool)
		def on_z_min(x): return np.isclose(comp(x, 2), gmin[2], atol=atol) if gdim >= 3 else np.zeros(x.shape[1], bool)
		def on_z_max(x): return np.isclose(comp(x, 2), gmax[2], atol=atol) if gdim >= 3 else np.zeros(x.shape[1], bool)
		def on_all(x):
			msk = on_x_min(x) | on_x_max(x)
			if gdim >= 2: msk |= on_y_min(x) | on_y_max(x)
			if gdim >= 3: msk |= on_z_min(x) | on_z_max(x)
			return msk

		def predicate_for(loc: str):
			loc = (loc or "").strip().lower()
			
			# Get geometry information from pde_config
			geometry_type = pde_config.get("mesh_parameters", {}).get("geometry_type", "")
			dimensions = pde_config.get("mesh_parameters", {}).get("dimensions", {})
			
			# Geometry-specific boundary mapping
			if geometry_type == "cylinder" and gdim >= 3:
				radius = dimensions.get("radius", 0.5)
				# Gmsh addCylinder(0,0,0, 0,0,height, radius) → cylinder is extruded along Z
				longitudinal_axis = 2
				transverse_axes = [0, 1]
				span_long = gmax[longitudinal_axis] - gmin[longitudinal_axis]
				# Use a tight tolerance near end faces to avoid overlap
				atol_face = max(1e-12, float(span_long) * 1e-12)
				logger.debug(f"[{physics_type}] Using cylinder Z-axis mapping for '{loc}', face_tol={atol_face:.3e}")
				
				# For cylinder: top/bottom are the end faces at min/max of the longitudinal axis
				if loc in ("top", "y_max", "z_max"):
					# Top face: nodes whose longitudinal coordinate is at the global max
					def on_top_face(x):
						return np.isclose(comp(x, longitudinal_axis), gmax[longitudinal_axis], atol=atol_face)
					return on_top_face
				elif loc in ("bottom", "y_min", "z_min"):
					# Bottom face: nodes whose longitudinal coordinate is at the global min
					def on_bottom_face(x):
						return np.isclose(comp(x, longitudinal_axis), gmin[longitudinal_axis], atol=atol_face)
					return on_bottom_face
				elif loc in ("curved_surface", "curved surface", "side", "side_boundary"):
					# Curved surface: NOT on the top/bottom faces (simple, robust)
					def on_curved_surface(x):
						coord = comp(x, longitudinal_axis)
						# Strictly between faces using tighter tolerance
						return (coord > gmin[longitudinal_axis] + atol_face) & (coord < gmax[longitudinal_axis] - atol_face)
					return on_curved_surface
			
			# Default geometric predicates for other geometries
			if loc in ("one_end", "x_min", "left"):   return on_x_min
			if loc in ("other_end", "x_max", "right"): return on_x_max
			if loc == "y_min": return on_y_min
			if loc == "y_max": return on_y_max
			if loc == "z_min": return on_z_min
			if loc == "z_max": return on_z_max
			return on_all

		# Fast path: Dirichlet-only, no facet tagging
		if not has_facet_terms:
			logger.debug(f"[{physics_type}] Processing {len(bc_list)} boundary conditions in fast path (Dirichlet-only)")
			
			# Try to use physical groups for boundary mapping in fast path too
			physical_group_mapping = {}
			if hasattr(self, 'mesh_data') and hasattr(self.mesh_data, 'physical_groups'):
				for name, group in self.mesh_data.physical_groups.items():
					if group.dim == fdim:  # Only facet groups
						physical_group_mapping[name] = group.tag
						logger.debug(f"Physical group mapping (fast path): '{name}' -> tag {group.tag}")
			
			for bc_idx, bc in enumerate(bc_list):
				btype = (bc.get("type") or "").strip().lower()
				loc = bc.get("location", "all_boundary")
				logger.debug(f"[{physics_type}] BC {bc_idx+1}: type='{btype}', location='{loc}', value={bc.get('value', 'N/A')}")
				
				# Check if this is a Dirichlet BC for the current physics type
				if physics_type == "heat_transfer" and btype not in ("temperature", "dirichlet", "fixed"):
					logger.debug(f"[{physics_type}] Skipping BC {bc_idx+1}: not a Dirichlet BC for heat_transfer")
					continue
				elif physics_type == "solid_mechanics" and btype not in ("displacement", "dirichlet", "fixed"):
					logger.debug(f"[{physics_type}] Skipping BC {bc_idx+1}: not a Dirichlet BC for solid_mechanics")
					continue
				
				# Try to use physical groups first, fall back to geometric predicates
				dofs = None
				if loc in physical_group_mapping:
					tag_id = physical_group_mapping[loc]
					logger.debug(f"[{physics_type}] Using physical group mapping: '{loc}' -> tag {tag_id}")
					if facet_tags is not None:
						facets = facet_tags.find(tag_id)
						if facets is not None and facets.size:
							dofs = fem.locate_dofs_topological(V, fdim, facets)
							logger.debug(f"[{physics_type}] Located {dofs.size} DOFs using physical group '{loc}'")
				
				if dofs is None or dofs.size == 0:
					# Fall back to geometric predicates
					pred = predicate_for(loc)
					logger.debug(f"[{physics_type}] Using geometric predicate for location '{loc}'")
					dofs = fem.locate_dofs_geometrical(V, pred)
					logger.debug(f"[{physics_type}] Located {dofs.size} DOFs using geometric predicate")
				
				if dofs.size == 0:
					logger.warning(f"[{physics_type}] Dirichlet: no DOFs at '{loc}', skipping BC {bc_idx+1}")
					continue
				
				if physics_type == "heat_transfer":
					# Scalar field
					val = float(bc.get("value", 0.0))
					# Get coordinates of nodes where BC is applied (use dof coordinates for correctness)
					dof_coords = V.tabulate_dof_coordinates().reshape((-1, gdim))
					bc_coords = dof_coords[dofs]
					logger.debug(f"[{physics_type}] ==========================================")
					logger.debug(f"[{physics_type}] Applying BC '{bc.get('type', 'unknown')}' at '{loc}' with value {val}")
					logger.debug(f"[{physics_type}] {dofs.size} DOFs constrained at coordinates:")
					# Print first 10 and last 10 coordinates to avoid too much output
					max_print = min(10, len(bc_coords))
					for i in range(max_print):
						coord = bc_coords[i]
						logger.debug(f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs[i]}: ({coord[0]:.6f})")
					if len(bc_coords) > 20:
						logger.debug(f"  ... ({len(bc_coords) - 20} more coordinates) ...")
						for i in range(max(0, len(bc_coords) - 10), len(bc_coords)):
							coord = bc_coords[i]
							logger.debug(f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs[i]}: ({coord[0]:.6f})")
					logger.debug(f"[{physics_type}] ==========================================")
					bcs_dirichlet.append(fem.dirichletbc(ScalarType(val), dofs, V))
				elif physics_type == "solid_mechanics":
					# Vector field
					val = bc.get("value", 0.0)
					vec = np.array(val, dtype=np.float64).ravel()
					if vec.size == 1:
						# Apply same value to all components
						for i in range(gdim):
							# Collapse the subspace to create a new function space with its own coordinate mapping
							V_i, _ = V.sub(i).collapse()
							dofs_i = fem.locate_dofs_geometrical(V_i, pred)
							if dofs_i.size:
								# Get coordinates of nodes where BC is applied (use dof coordinates for correctness)
								dof_coords = V_i.tabulate_dof_coordinates().reshape((-1, gdim))
								bc_coords = dof_coords[dofs_i]
								logger.debug(f"[{physics_type}] Applying BC '{bc.get('type', 'unknown')}' at '{loc}' component {i} with value {vec[0]}")
								logger.debug(f"[{physics_type}] {dofs_i.size} DOFs constrained at coordinates:")
								for j, coord in enumerate(bc_coords):
									logger.debug(f"  DOF {dofs_i[j]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs_i[j]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs_i[j]}: ({coord[0]:.6f})")
								bcs_dirichlet.append(fem.dirichletbc(ScalarType(vec[0]), dofs_i, V.sub(i)))
					else:
						# Apply different values to each component
						for i in range(gdim):
							# Collapse the subspace to create a new function space with its own coordinate mapping
							V_i, _ = V.sub(i).collapse()
							dofs_i = fem.locate_dofs_geometrical(V_i, pred)
							if dofs_i.size:
								# Get coordinates of nodes where BC is applied (use dof coordinates for correctness)
								dof_coords = V_i.tabulate_dof_coordinates().reshape((-1, gdim))
								bc_coords = dof_coords[dofs_i]
								logger.debug(f"[{physics_type}] Applying BC '{bc.get('type', 'unknown')}' at '{loc}' component {i} with value {vec[i] if i < vec.size else 0.0}")
								logger.debug(f"[{physics_type}] {dofs_i.size} DOFs constrained at coordinates:")
								for j, coord in enumerate(bc_coords):
									logger.debug(f"  DOF {dofs_i[j]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs_i[j]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs_i[j]}: ({coord[0]:.6f})")
								bcs_dirichlet.append(fem.dirichletbc(ScalarType(vec[i] if i < vec.size else 0.0), dofs_i, V.sub(i)))
			return bcs_dirichlet, a_extra, L_extra, dx, ds

		# General path: need facet tags
		logger.debug(f"[{physics_type}] Processing {len(bc_list)} boundary conditions in general path (with facet terms)")
		loc_to_facets = {}
		used_locations = []
		for bc in bc_list:
			loc = bc.get("location", "all_boundary")
			# Map common aliases used by NLP to actual physical names for discs
			# Keep original loc for tag_map keying
			if loc.strip().lower() == "circumference":
				loc = "circular_boundary"
			if loc in loc_to_facets:
				continue
			pred = predicate_for(loc)
			facets = _mesh.locate_entities_boundary(m, fdim, pred)
			loc_to_facets[loc] = facets.astype(np.int32)
			if facets.size > 0:
				used_locations.append(loc)

		facet_tags = None
		tag_map = {}
		if used_locations:
			all_facets, all_tags = [], []
			next_tag = 1
			for loc in used_locations:
				f = loc_to_facets[loc]
				if f.size == 0:
					continue
				tag_map[loc] = next_tag
				all_facets.append(f)
				all_tags.append(np.full(f.shape, next_tag, dtype=np.int32))
				next_tag += 1
			if all_facets:
				facets_concat = np.concatenate(all_facets).astype(np.int32)
				tags_concat   = np.concatenate(all_tags).astype(np.int32)
			if facets_concat.size > 0:
					facet_tags = _mesh.meshtags(m, fdim, facets_concat, tags_concat)

		# Prefer ds built from existing facet tags from gmsh/model_to_mesh when available
		pg_map = {}
		if hasattr(self, "mesh_data") and hasattr(self.mesh_data, "physical_groups"):
			for name, group in self.mesh_data.physical_groups.items():
				if group.dim == fdim:
					pg_map[name] = int(group.tag)
					try:
						logger.debug(f"[{physics_type}] Physical group available: '{name}' -> tag {group.tag}")
					except Exception:
						pass
		# Choose subdomain data priority: gmsh facet_tags (from model_to_mesh) > locally built meshtags > none
		preferred_facet_tags = self.facet_tags if getattr(self, "facet_tags", None) is not None else facet_tags
		ds = ufl.Measure("ds", domain=m, subdomain_data=preferred_facet_tags) if preferred_facet_tags is not None else ufl.Measure("ds", domain=m)

		# Track DOF sets per location to verify disjointness
		loc_to_dofs = {}
		for bc in bc_list:
			btype = (bc.get("type") or "").strip().lower()
			loc   = bc.get("location", "all_boundary")
			if loc.strip().lower() == "circumference":
				loc = "circular_boundary"
			pred  = predicate_for(loc)
			logger.debug(f"[{physics_type}] Processing BC: type='{btype}', location='{loc}', value={bc.get('value', 'N/A')}")

			# Dirichlet BCs
			if physics_type == "heat_transfer" and btype in ("temperature", "dirichlet", "fixed"):
				# Prefer topological dof selection using gmsh facet tags > local meshtags > geometrical
				dofs = None
				# Try gmsh physical group mapping
				if preferred_facet_tags is not None:
					# Normalize aliases
					alias = {
						"circumference": "curved_surface",
						"curved surface": "curved_surface"
					}
					loc_norm = alias.get(loc.strip().lower(), loc.strip().lower())
					if loc_norm in pg_map:
						facets_for_loc = preferred_facet_tags.find(pg_map[loc_norm])
						if facets_for_loc is not None and facets_for_loc.size:
							dofs = fem.locate_dofs_topological(V, fdim, facets_for_loc)
				# Fall back to locally created tag_map
				if dofs is None or dofs.size == 0:
					if facet_tags is not None and 'tag_map' in locals() and loc in tag_map:
						facets_for_loc = facet_tags.find(tag_map[loc])
						if facets_for_loc is not None and facets_for_loc.size:
							dofs = fem.locate_dofs_topological(V, fdim, facets_for_loc)
				# Finally, geometrical predicate
				if dofs is None or dofs.size == 0:
					dofs = fem.locate_dofs_geometrical(V, pred)
				logger.debug(f"[{physics_type}] Located {dofs.size} DOFs for location '{loc}'")
				if dofs.size == 0:
					logger.warning(f"[{physics_type}] Dirichlet: no DOFs at '{loc}', skipping.")
					continue
				# Track dofs for overlap diagnostics
				try:
					loc_to_dofs.setdefault(loc, set()).update(map(int, np.array(dofs, dtype=np.int64).ravel().tolist()))
				except Exception:
					pass
				val = float(bc.get("value", 0.0))
				# Get coordinates of nodes where BC is applied (use dof coordinates for correctness)
				dof_coords = V.tabulate_dof_coordinates().reshape((-1, gdim))
				bc_coords = dof_coords[dofs]
				logger.debug(f"[{physics_type}] ==========================================")
				logger.debug(f"[{physics_type}] Applying BC '{bc.get('type', 'unknown')}' at '{loc}' with value {val}")
				logger.debug(f"[{physics_type}] {dofs.size} DOFs constrained at coordinates:")
				# Print first 10 and last 10 coordinates to avoid too much output
				max_print = min(10, len(bc_coords))
				for i in range(max_print):
					coord = bc_coords[i]
					logger.debug(f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs[i]}: ({coord[0]:.6f})")
				if len(bc_coords) > 20:
					logger.debug(f"  ... ({len(bc_coords) - 20} more coordinates) ...")
					for i in range(max(0, len(bc_coords) - 10), len(bc_coords)):
						coord = bc_coords[i]
						logger.debug(f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})" if gdim >= 3 else f"  DOF {dofs[i]}: ({coord[0]:.6f}, {coord[1]:.6f})" if gdim >= 2 else f"  DOF {dofs[i]}: ({coord[0]:.6f})")
				logger.debug(f"[{physics_type}] ==========================================")
				bcs_dirichlet.append(fem.dirichletbc(ScalarType(val), dofs, V))
				continue
			elif physics_type == "solid_mechanics" and btype in ("displacement", "dirichlet", "fixed"):
				val = bc.get("value", 0.0)
				vec = np.array(val, dtype=np.float64).ravel()
				if vec.size == 1:
					# Apply same value to all components
					for i in range(gdim):
						# Collapse the subspace to create a new function space with its own coordinate mapping
						V_i, _ = V.sub(i).collapse()
						dofs_i = fem.locate_dofs_geometrical(V_i, pred)
						if dofs_i.size:
							bcs_dirichlet.append(fem.dirichletbc(ScalarType(vec[0]), dofs_i, V.sub(i)))
				else:
					# Apply different values to each component
					for i in range(gdim):
						# Collapse the subspace to create a new function space with its own coordinate mapping
						V_i, _ = V.sub(i).collapse()
						dofs_i = fem.locate_dofs_geometrical(V_i, pred)
						if dofs_i.size:
							bcs_dirichlet.append(fem.dirichletbc(ScalarType(vec[i] if i < vec.size else 0.0), dofs_i, V.sub(i)))
				continue

			# Select ds for this location (prefer gmsh physical group tags)
			alias = {"circumference": "curved_surface", "curved surface": "curved_surface"}
			loc_norm = alias.get(loc.strip().lower(), loc.strip().lower())
			if preferred_facet_tags is not None and loc_norm in pg_map:
				ds_sel = ds(pg_map[loc_norm])
			elif facet_tags is not None and loc in tag_map:
				ds_sel = ds(tag_map[loc])
			else:
				ds_sel = ds

			# Heat transfer boundary conditions
			if physics_type == "heat_transfer":
				if btype in ("flux", "neumann", "heat_flux", "heat flux"):
					q = ScalarType(float(bc.get("value", 0.0)))
					L_extra = (L_extra + q * v * ds_sel) if L_extra is not None else (q * v * ds_sel)
				elif btype == "internal_source":
					# Volume source term (e.g., heat flux specified at 'center' for a disc)
					q0 = ScalarType(float(bc.get("value", 0.0)))
					# Create a compactly-supported source around the centroid
					X = np.asarray(m.geometry.x)
					if X.ndim == 1:
						X = X.reshape(-1, 1)
					center = X.mean(axis=0)
					# radius ~ 5% of max span
					gmin, gmax = X.min(axis=0), X.max(axis=0)
					span = np.maximum(gmax - gmin, 1.0)
					r0 = 0.05 * float(np.max(span))
					# Define indicator via ufl using distance to center (projected to gdim available)
					xc = [float(center[i] if i < X.shape[1] else 0.0) for i in range(m.geometry.dim)]
					d2 = 0
					for i in range(m.geometry.dim):
						d2 = d2 + (ufl.SpatialCoordinate(m)[i] - xc[i])**2
					chi = ufl.exp(-d2/(r0*r0))  # smooth bump near center
					L_extra = (L_extra + q0 * chi * v * dx) if L_extra is not None else (q0 * chi * v * dx)

			elif btype in ("robin", "convection", "radiation"):
				raw = bc.get("value", {})
				if isinstance(raw, dict):
					h = raw.get("h", raw.get("h_coeff", raw.get("h_rad", 0.0)))
					Tinf = raw.get("Tinf", raw.get("T_inf", raw.get("ambient", 0.0)))
				else:
					h = bc.get("h", bc.get("h_coeff", bc.get("h_rad", 0.0)))
					Tinf = bc.get("Tinf", bc.get("T_inf", bc.get("ambient", 0.0)))
				h = float(h or 0.0); Tinf = float(Tinf or 0.0)
				if h == 0.0:
					continue
				hc = ScalarType(h); Tc = ScalarType(Tinf)
				a_term = hc * u * v * ds_sel
				L_term = hc * Tc * v * ds_sel
				a_extra = (a_extra + a_term) if a_extra is not None else a_term
				L_extra = (L_extra + L_term) if L_extra is not None else L_term

			# Solid mechanics boundary conditions
			elif physics_type == "solid_mechanics":
				if btype in ("traction", "neumann"):
					tval = bc.get("value", 0.0)
					arr = np.array(tval, dtype=np.float64).ravel()
					tvec = np.zeros(gdim, dtype=np.float64)
					tvec[:min(gdim, arr.size)] = arr[:min(gdim, arr.size)]
					t = fem.Constant(m, tvec)
					L_extra = (L_extra + ufl.dot(t, v) * ds_sel) if L_extra is not None else (ufl.dot(t, v) * ds_sel)

				elif btype == "pressure":
					pval = float(bc.get("value", 0.0))
					p = fem.Constant(m, ScalarType(pval))
					L_extra = (L_extra + (-p) * ufl.dot(n, v) * ds_sel) if L_extra is not None else ((-p) * ufl.dot(n, v) * ds_sel)

		# After processing, check for overlaps between locations
		try:
			locs = list(loc_to_dofs.keys())
			for i in range(len(locs)):
				for j in range(i+1, len(locs)):
					li, lj = locs[i], locs[j]
					over = loc_to_dofs[li].intersection(loc_to_dofs[lj])
					if over:
						logger.debug(f"[{physics_type}] DOF overlap detected between '{li}' and '{lj}': {len(over)} shared DOFs")
					else:
							logger.debug(f"[{physics_type}] DOF sets are disjoint for '{li}' and '{lj}'")
		except Exception:
			pass

		return bcs_dirichlet, a_extra, L_extra, dx, ds

# Create a global instance
fenics_solver = FEniCSSolver()