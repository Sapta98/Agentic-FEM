"""
Boundary condition normalization utilities
"""

import logging
from typing import Dict, Any, List, Optional

from .unit_conversion import convert_heat_flux_units, convert_pressure_units

logger = logging.getLogger(__name__)


def canon_loc(loc: Optional[str], geometry_type: str = None, map_location_via_config=None) -> str:
	"""
	Canonicalize location name using config-driven mapping.
	Uses geometry_boundaries.json as the source of truth.
	"""
	if not loc:
		return loc
	
	loc_lower = loc.strip().lower()
	
	# Use config-driven mapping if available
	if geometry_type and map_location_via_config:
		mapped = map_location_via_config(loc_lower, geometry_type)
		if mapped != loc_lower:
			return mapped
	
	# Fallback: simple common mappings (for backward compatibility)
	simple_aliases = {
		"all": "all_boundary", "entire": "all_boundary", "complete": "all_boundary",
		"all_boundary": "all_boundary"
	}
	
	return simple_aliases.get(loc_lower, loc)


def vec(value, gdim: int) -> List[float]:
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
		vec_list = [float(v) for v in value[:gdim]]
		return vec_list + [0.0] * (gdim - len(vec_list))
	# fallback
	return [0.0] * gdim


def normalize_heat_bcs(raw_bcs: List[Dict[str, Any]], assume_celsius: bool=True, canon_loc_func=None, geometry_type: str = None) -> List[Dict[str, Any]]:
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
		loc = canon_loc_func(bc.get("location"), geometry_type) if canon_loc_func else canon_loc(bc.get("location"), geometry_type)
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
				val = convert_heat_flux_units(val, units)
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


def normalize_solid_bcs(raw_bcs: List[Dict[str, Any]], gdim: int, canon_loc_func=None, geometry_type: str = None) -> List[Dict[str, Any]]:
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
		loc = canon_loc_func(bc.get("location"), geometry_type) if canon_loc_func else canon_loc(bc.get("location"), geometry_type)
		t   = (bc.get("type") or "").strip().lower()
		hint= (bc.get("bc_type") or "").strip().lower()

		# Dirichlet displacement
		if t in ("fixed","fixed support","dirichlet","displacement") or hint == "dirichlet":
			btype = "dirichlet"
			if t in ("fixed","fixed support") and bc.get("value") in (None, "", 0, 0.0):
				val = [0.0]*gdim
			else:
				val = vec(bc.get("value", 0.0), gdim)
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
					vec_list = [convert_pressure_units(float(v), units) for v in vec_raw]
				elif isinstance(vec_raw, dict):
					# Dictionary format: convert each numeric component, preserve structure
					vec_dict = {}
					for key, val in vec_raw.items():
						try:
							# Convert numeric values, preserve non-numeric as-is
							vec_dict[key] = convert_pressure_units(float(val), units)
						except (ValueError, TypeError):
							# Non-numeric value, preserve as-is
							vec_dict[key] = val
					# Convert dict to vector using vec
					vec_list = vec(vec_dict, gdim)
				else:
					# Scalar: convert and let vec handle it
					converted_val = convert_pressure_units(float(vec_raw), units)
					vec_list = vec(converted_val, gdim)
			else:
				vec_list = vec(vec_raw, gdim)
			out.append({"type":"neumann", "location":loc, "value": vec_list})
			continue

		# Pressure scalar
		if t == "pressure":
			p = float(bc.get("value", 0.0))
			# Convert units if specified
			units = bc.get("units") or bc.get("parameters", {}).get("units")
			if units:
				p = convert_pressure_units(p, units)
			out.append({"type":"pressure", "location":loc, "value": p})
			continue

		# Unknowns → skip safely
		continue

	return out


def normalize_bcs_by_physics(physics_type: str, raw_bcs: List[Dict[str, Any]], gdim: int, canon_loc_func=None, geometry_type: str = None) -> List[Dict[str, Any]]:
	pt = (physics_type or "").strip().lower()
	if pt == "heat_transfer":
		return normalize_heat_bcs(raw_bcs, assume_celsius=True, canon_loc_func=canon_loc_func, geometry_type=geometry_type)
	if pt == "solid_mechanics":
		return normalize_solid_bcs(raw_bcs, gdim=gdim, canon_loc_func=canon_loc_func, geometry_type=geometry_type)
	# default
	return raw_bcs or []

