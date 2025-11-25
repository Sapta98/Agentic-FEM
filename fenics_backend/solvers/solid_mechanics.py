"""
Solid mechanics solver
"""

import logging
import numpy as np
from typing import Dict, Any
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl

from ..config_helpers import prepare_petsc_solver_options, build_vector_function_space
from ..boundary_conditions import prepare_boundary_forms

logger = logging.getLogger(__name__)
ScalarType = PETSc.ScalarType


def solve_solid_mechanics(domain, pde_config, mesh_data, facet_tags, resolve_func, extract_func, cache_dict):
	"""Small-strain linear elasticity in 1D/2D/3D with Dirichlet/Neumann/pressure BCs."""
	logger.debug("Solving solid mechanics with DOLFINx")
	try:
		m = domain
		tdim = int(m.topology.dim)
		gdim = int(m.geometry.dim)
		V = build_vector_function_space(pde_config, gdim, m)
		
		logger.debug(f"Created function space: gdim={gdim}")
		logger.debug(f"Function space V: {V}")

		# --- trial/test ---
		u = ufl.TrialFunction(V)
		v = ufl.TestFunction(V)

		# --- material parameters ---
		material_props = pde_config.get("material_properties", {}) or {}
		mat = pde_config.get("material", {}) or {}
		
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
		bcs_dirichlet, a_extra, L_extra, dx, ds = prepare_boundary_forms(V, pde_config, u, v, "solid_mechanics", mesh_data, facet_tags, resolve_func)

		# --- bilinear & linear forms ---
		a = ufl.inner(sigma(u), eps(v)) * dx
		L = ufl.dot(f, v) * dx
		if a_extra is not None: a += a_extra
		if L_extra is not None: L += L_extra

		# --- solve ---
		petsc_opts, prefix = prepare_petsc_solver_options(pde_config, PETSc, default_prefix="sm_")
		problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts, petsc_options_prefix=prefix)
		solution = problem.solve()

		return extract_func(solution, domain, mesh_data, cache_dict, "solid_mechanics")

	except Exception as e:
		logger.error(f"Solid mechanics solver failed: {e}")
		raise e

