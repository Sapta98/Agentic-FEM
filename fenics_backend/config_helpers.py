"""
Configuration and material property extraction helpers
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from mpi4py import MPI
from petsc4py import PETSc
import basix
import dolfinx

logger = logging.getLogger(__name__)


def extract_fe_metadata(pde_config: Dict[str, Any], default_family: str = "Lagrange", default_degree: int = 1) -> Tuple[str, int]:
	"""Return (family, degree) for FE spaces with sane fallbacks."""
	family = (pde_config.get("family", default_family) or default_family).strip()
	try:
		degree = int(pde_config.get("degree", default_degree))
	except (TypeError, ValueError):
		degree = default_degree
	return family, degree


def extract_material_value(
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


def ensure_single_process_comm(comm: Optional[MPI.Comm], context: str = "") -> None:
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


def prepare_petsc_solver_options(
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

	configure_petsc_single_process(petsc_opts, PETSc_module)
	return petsc_opts, prefix


def configure_petsc_single_process(petsc_opts: Dict[str, Any], PETSc_module) -> bool:
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


def build_scalar_function_space(pde_config: Dict[str, Any], mesh):
	"""Build scalar function space from config."""
	family, degree = extract_fe_metadata(pde_config)
	element = basix.ufl.element(family, mesh.basix_cell(), degree)
	return dolfinx.fem.functionspace(mesh, element)


def build_vector_function_space(pde_config: Dict[str, Any], gdim: int, mesh):
	"""Build vector function space from config."""
	family, degree = extract_fe_metadata(pde_config)
	element = basix.ufl.element(family, mesh.basix_cell(), degree, shape=(gdim,))
	return dolfinx.fem.functionspace(mesh, element)


def extract_initial_scalar_value(pde_config: Dict[str, Any], default: float = 0.0) -> float:
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

