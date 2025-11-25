"""
Boundary condition application and preparation
"""

import logging
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc
from typing import Dict, Any, List, Tuple, Optional

from .physical_groups import (
	resolve_boundary_location,
	find_physical_group_by_location,
)

logger = logging.getLogger(__name__)
ScalarType = PETSc.ScalarType


def get_physical_groups_from_mesh_data(mesh_data):
	"""Get physical groups from mesh_data - ONLY source for boundary detection."""
	physical_groups = {}
	if mesh_data:
		if isinstance(mesh_data, dict):
			pg_dict = mesh_data.get("physical_groups")
		elif hasattr(mesh_data, "physical_groups"):
			pg_dict = mesh_data.physical_groups
		else:
			pg_dict = None
		
		if pg_dict:
			# GMSH generates exact JSON names, so no normalization needed
			physical_groups = pg_dict
	return physical_groups


def get_facet_tags_from_mesh_data(mesh_data, facet_tags_attr=None):
	"""Get facet_tags from mesh_data - ONLY source for facet tags."""
	if facet_tags_attr is not None:
		return facet_tags_attr
	elif mesh_data:
		if isinstance(mesh_data, dict):
			return mesh_data.get("facet_tags")
		elif hasattr(mesh_data, 'facet_tags'):
			return mesh_data.facet_tags
	return None


def apply_dirichlet_bc(bc, loc, V, m, physical_groups, geometry_type, pde_config, fdim, gdim, physics_type, resolve_func):
	"""Apply a single Dirichlet BC using physical groups."""
	dofs, method, confidence = resolve_func(loc, V, m, physical_groups, geometry_type, pde_config, fdim)
	
	if dofs is None or (isinstance(dofs, np.ndarray) and dofs.size == 0):
		logger.warning(f"[{physics_type}] Dirichlet: no DOFs at '{loc}', skipping.")
		return None
	
	if physics_type == "heat_transfer":
		val = float(bc.get("value", 0.0))
		return fem.dirichletbc(ScalarType(val), dofs, V)
	
	elif physics_type == "solid_mechanics":
		val = bc.get("value", 0.0)
		vec = np.array(val, dtype=np.float64).ravel()
		bcs = []
		if vec.size == 1:
			for i in range(gdim):
				V_i, _ = V.sub(i).collapse()
				dofs_i, _, _ = resolve_func(loc, V_i, m, physical_groups, geometry_type, pde_config, fdim)
				if dofs_i is not None and (isinstance(dofs_i, np.ndarray) and dofs_i.size > 0):
					bcs.append(fem.dirichletbc(ScalarType(vec[0]), dofs_i, V.sub(i)))
		else:
			for i in range(min(vec.size, gdim)):
				V_i, _ = V.sub(i).collapse()
				dofs_i, _, _ = resolve_func(loc, V_i, m, physical_groups, geometry_type, pde_config, fdim)
				if dofs_i is not None and (isinstance(dofs_i, np.ndarray) and dofs_i.size > 0):
					bcs.append(fem.dirichletbc(ScalarType(vec[i] if i < vec.size else 0.0), dofs_i, V.sub(i)))
		return bcs
	
	return None


def apply_neumann_bc(bc, loc, physical_groups, fdim, ds, physics_type, m, v, dx, u, n):
	"""Apply a single Neumann BC using physical groups."""
	btype = (bc.get("type") or "").strip().lower()
	
	# Find physical group for this location
	pg_match, tag, confidence = find_physical_group_by_location(loc, physical_groups, fdim)
	if tag is None or confidence <= 0.0:
		available_groups = list(physical_groups.keys())
		raise RuntimeError(
			f"[{physics_type}] Could not find physical group for location '{loc}' (Neumann BC). "
			f"Available physical groups: {available_groups}. "
			f"Physical groups MUST be provided by the mesh generator."
		)
	
	ds_sel = ds(tag)
	# Get name - handle both dict and object types
	if pg_match:
		if isinstance(pg_match, dict):
			pg_name = pg_match.get('name', 'unknown')
		else:
			pg_name = getattr(pg_match, 'name', 'unknown')
	else:
		pg_name = 'unknown'
	
	if physics_type == "heat_transfer":
		if btype in ("flux", "neumann", "heat_flux", "heat flux"):
			q = ScalarType(float(bc.get("value", 0.0)))
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


def prepare_boundary_forms(V, pde_config, u, v, physics_type, mesh_data, facet_tags_attr, resolve_func):
	"""
	Build Dirichlet BCs and boundary RHS/stiffness contributions using ONLY physical groups.
	
	Returns: (bcs_dirichlet, a_extra, L_extra, dx, ds)
	"""
	m = V.mesh
	tdim = int(m.topology.dim)
	fdim = tdim - 1
	gdim = int(m.geometry.dim)

	dx = ufl.Measure("dx", domain=m)
	n = ufl.FacetNormal(m)

	bc_list = pde_config.get("boundary_conditions", []) or []
	
	# Get physical groups and facet_tags from mesh_data (ONLY source)
	physical_groups = get_physical_groups_from_mesh_data(mesh_data)
	facet_tags = get_facet_tags_from_mesh_data(mesh_data, facet_tags_attr)
	
	if not physical_groups:
		logger.warning(f"[{physics_type}] No physical groups found in mesh_data - boundary conditions may fail")
	
	# Create ds measure with facet_tags if available
	if facet_tags is not None:
		ds = ufl.Measure("ds", domain=m, subdomain_data=facet_tags)
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
		

		# Dirichlet BCs
		if physics_type == "heat_transfer" and btype in ("temperature", "dirichlet", "fixed"):
			bc_result = apply_dirichlet_bc(bc, loc, V, m, physical_groups, geometry_type, pde_config, fdim, gdim, physics_type, resolve_func)
			if bc_result:
				if isinstance(bc_result, list):
					bcs_dirichlet.extend(bc_result)
				else:
					bcs_dirichlet.append(bc_result)
		
		elif physics_type == "solid_mechanics" and btype in ("displacement", "dirichlet", "fixed"):
			bc_result = apply_dirichlet_bc(bc, loc, V, m, physical_groups, geometry_type, pde_config, fdim, gdim, physics_type, resolve_func)
			if bc_result:
				if isinstance(bc_result, list):
					bcs_dirichlet.extend(bc_result)
				else:
					bcs_dirichlet.append(bc_result)
		
		# Neumann BCs
		else:
			result = apply_neumann_bc(bc, loc, physical_groups, fdim, ds, physics_type, m, v, dx, u, n)
			if result is not None:
				if isinstance(result, tuple):  # Robin BC returns (L_term, a_term)
					L_term, a_term = result
					L_extra = (L_extra + L_term) if L_extra is not None else L_term
					a_extra = (a_extra + a_term) if a_extra is not None else a_term
				else:  # Single term for L_extra
					L_extra = (L_extra + result) if L_extra is not None else result

	return bcs_dirichlet, a_extra, L_extra, dx, ds

