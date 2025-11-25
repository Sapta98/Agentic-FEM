"""
Heat transfer solver (steady and transient)
"""

import logging
import numpy as np
from typing import Dict, Any
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl
from mpi4py import MPI

from ..config_helpers import (
	extract_material_value,
	extract_initial_scalar_value,
	prepare_petsc_solver_options,
	build_scalar_function_space,
)
from ..boundary_conditions import prepare_boundary_forms
from ..solution_extraction import extract_solution_data
from .transient_helpers import (
	apply_bcs_to_initial_condition,
	compute_steady_state_for_convergence,
	run_transient_time_stepping,
	assemble_transient_result,
)

logger = logging.getLogger(__name__)
ScalarType = PETSc.ScalarType


def solve_heat_transfer(domain, pde_config, mesh_data, facet_tags, resolve_func, extract_func, cache_dict):
	"""Solve steady heat conduction: -div(k ∇u) = 0 with boundary conditions."""
	logger.debug("Solving heat transfer")
	try:
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
			return solve_heat_transfer_transient(domain, pde_config, mesh_data, facet_tags, resolve_func, extract_func, cache_dict)

		m = domain
		tdim = m.topology.dim
		fdim = tdim - 1
		m.topology.create_connectivity(fdim, tdim)
		m.topology.create_connectivity(tdim, fdim)
		
		# --- FE space (scalar) ---
		V = build_scalar_function_space(pde_config, m)
		mesh_comm = getattr(m, 'comm', MPI.COMM_SELF)
		
		# --- trial/test ---
		u = ufl.TrialFunction(V)
		v = ufl.TestFunction(V)

		# --- material parameter ---
		kappa_val = extract_material_value(pde_config, "thermal_conductivity", default=1.0, aliases=["k"])
		kappa = fem.Constant(m, ScalarType(kappa_val))

		# --- boundary terms & measures ---
		bcs_dirichlet, a_extra, L_extra, dx, ds = prepare_boundary_forms(V, pde_config, u, v, "heat_transfer", mesh_data, facet_tags, resolve_func)

		# --- weak form ---
		a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
		L = fem.Constant(m, ScalarType(0.0)) * v * dx
		if a_extra is not None: a += a_extra
		if L_extra is not None: L += L_extra
			
		# --- linear solver options (configurable) ---
		petsc_opts, prefix = prepare_petsc_solver_options(pde_config, PETSc, default_prefix="ht_")
		
		# --- solve (handle API diffs across dolfinx versions) ---
		comm_size = mesh_comm.Get_size()
		comm_rank = mesh_comm.Get_rank()
		logger.info(f"Using mesh communicator for solve: size={comm_size}, rank={comm_rank} (mesh comm type: {type(mesh_comm)})")
		
		# Ensure PETSc uses the same communicator as the mesh
		try:
			problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts, petsc_options_prefix=prefix)
		except TypeError:
			try:
				problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts, options_prefix=prefix)
			except TypeError:
				problem = LinearProblem(a, L, bcs=bcs_dirichlet, petsc_options=petsc_opts)

		try:
			solution = problem.solve()
		except Exception as solve_error:
			error_str = str(solve_error)
			if "Invalid rank" in error_str or "MPI" in error_str:
				logger.error(f"MPI communicator error during solve: {solve_error}")
				logger.error(f"Mesh communicator: size={comm_size}, rank={comm_rank}")
				logger.error(f"MPI.COMM_WORLD: size={MPI.COMM_WORLD.Get_size()}, rank={MPI.COMM_WORLD.Get_rank()}")
				import traceback
				logger.error(f"Full traceback: {traceback.format_exc()}")
				raise RuntimeError(f"MPI communicator mismatch during solve. Mesh comm size={comm_size}, MPI.COMM_WORLD size={MPI.COMM_WORLD.Get_size()}. Error: {solve_error}")
			raise

		return extract_func(solution, domain, mesh_data, cache_dict, "heat_transfer")

	except Exception as e:
		try:
			bc_types = [bc.get("type") for bc in (pde_config.get("boundary_conditions") or [])]
			logger.error(f"[heat] Failure details — BC count={len(bc_types)} types={bc_types}")
		except Exception:
			pass
		logger.error(f"Heat transfer solver failed: {e}")
		raise


def solve_heat_transfer_transient(domain, pde_config, mesh_data, facet_tags, resolve_func, extract_func, cache_dict):
	"""Solve transient heat conduction with implicit time-stepping."""
	logger.info("Solving transient heat transfer equation")
	try:
		m = domain
		tdim = m.topology.dim
		fdim = tdim - 1
		m.topology.create_connectivity(fdim, tdim)
		m.topology.create_connectivity(tdim, fdim)

		V = build_scalar_function_space(pde_config, m)

		u = ufl.TrialFunction(V)
		v = ufl.TestFunction(V)

		time_params = pde_config.get("time_stepping") or {}
		requested_dt = float(time_params.get("time_step", 0.0) or 0.0)
		total_time = float(time_params.get("total_time", 0.0) or 0.0)
		method = (time_params.get("method") or "backward_euler").strip().lower()

		if total_time <= 0.0:
			logger.warning("Invalid time-stepping parameters. Falling back to steady-state solver.")
			return solve_heat_transfer(domain, {**pde_config, "time_stepping": {}}, mesh_data, facet_tags, resolve_func, extract_func, cache_dict)

		supported_methods = {"backward_euler", "implicit_euler", "implicit"}
		if method not in supported_methods:
			logger.warning(f"Time-stepping method '{method}' not supported. Falling back to backward_euler.")
			method = "backward_euler"

		kappa_val = extract_material_value(pde_config, "thermal_conductivity", default=1.0, aliases=["k"])
		kappa = fem.Constant(m, ScalarType(kappa_val))

		density = extract_material_value(pde_config, "density", default=1.0)
		specific_heat = extract_material_value(pde_config, "specific_heat", default=1.0, aliases=["cp"])
		heat_capacity = density * specific_heat
		if heat_capacity <= 0.0 or not np.isfinite(heat_capacity):
			logger.debug(f"Invalid heat capacity ({heat_capacity}); defaulting to 1.0")
			heat_capacity = 1.0
		
		thermal_diffusivity = kappa_val / (density * specific_heat) if (density * specific_heat) > 0 else 1.0
		
		# Calculate minimum element size for stability check
		coords = m.geometry.x
		if coords.shape[0] > 1:
			n_samples = min(100, coords.shape[0])
			sample_indices = np.linspace(0, coords.shape[0] - 1, n_samples, dtype=int)
			sample_coords = coords[sample_indices]
			
			min_distances = []
			for sample_coord in sample_coords:
				distances = np.linalg.norm(coords - sample_coord, axis=1)
				distances = distances[distances > 1e-10]
				if len(distances) > 0:
					min_distances.append(np.min(distances))
			
			if min_distances:
				h_min = float(np.mean(min_distances))
			else:
				bbox_min = np.min(coords, axis=0)
				bbox_max = np.max(coords, axis=0)
				bbox_size = np.linalg.norm(bbox_max - bbox_min)
				h_min = bbox_size / 100.0
		else:
			bbox_min = np.min(coords, axis=0)
			bbox_max = np.max(coords, axis=0)
			bbox_size = np.linalg.norm(bbox_max - bbox_min)
			h_min = bbox_size / 100.0
		
		if thermal_diffusivity > 0:
			dt_max_stable = (h_min ** 2) / (2.0 * thermal_diffusivity)
		else:
			dt_max_stable = total_time / 100.0
		
		meaningful_requested_dt = requested_dt > 0.0 and requested_dt >= 0.1 * dt_max_stable
		
		if meaningful_requested_dt:
			dt = min(requested_dt, dt_max_stable)
			if dt < requested_dt:
				logger.warning(f"Requested time_step {requested_dt} exceeds maximum stable time_step {dt_max_stable:.6f}. Using {dt:.6f}")
			else:
				logger.info(f"Using requested time_step: {dt:.6f} (max stable: {dt_max_stable:.6f})")
		else:
			dt = dt_max_stable
			if requested_dt > 0.0:
				logger.info(f"Requested time_step {requested_dt:.6f} is too small compared to max stable {dt_max_stable:.6f}. Using max stable dt: {dt:.6f}")
			else:
				logger.info(f"Auto-determined time_step: {dt:.6f} (max stable: {dt_max_stable:.6f}, h_min: {h_min:.6f}, alpha: {thermal_diffusivity:.6f})")
		
		logger.info(f"Transient heat transfer: time_step={dt:.6f}, method={method}, running until convergence")

		u_prev = fem.Function(V, name="u_prev")
		initial_value_celsius = extract_initial_scalar_value(pde_config, default=0.0)
		initial_value = initial_value_celsius + 273.15
		logger.info(f"Initial temperature: {initial_value_celsius} C = {initial_value:.3f} K")
		
		u_prev.x.array[:] = initial_value
		
		bcs_dirichlet, a_extra, L_extra, dx, ds = prepare_boundary_forms(V, pde_config, u, v, "heat_transfer", mesh_data, facet_tags, resolve_func)
		logger.info(f"Transient heat transfer: prepared {len(bcs_dirichlet)} Dirichlet boundary conditions")
		
		apply_bcs_to_initial_condition(u_prev, V, m, pde_config, bcs_dirichlet, fdim, mesh_data, resolve_func)
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
		petsc_opts, prefix = prepare_petsc_solver_options(time_petsc_cfg, PETSc, default_prefix="ht_ts_")

		logger.info(f"Transient solve: applying {len(bcs_dirichlet)} Dirichlet boundary conditions")
		
		problem = LinearProblem(
			a_form,
			L_form,
			bcs=bcs_dirichlet,
			petsc_options=petsc_opts,
			petsc_options_prefix=prefix,
		)

		final_solution = fem.Function(V, name="u_final")

		convergence_tol = float(time_params.get("convergence_tolerance", 1e-4))
		max_steps = int(time_params.get("max_steps", 10000))
		check_convergence = time_params.get("check_convergence", True)
		
		steady_state_solution = None
		if check_convergence:
			steady_state_solution = compute_steady_state_for_convergence(
				m, V, kappa, u, v, dx, a_extra, L_extra, bcs_dirichlet, petsc_opts, prefix, PETSc
			)
		
		if steady_state_solution is not None:
			logger.info(f"Convergence checking enabled (steady-state comparison): tolerance={convergence_tol}, max_steps={max_steps}")
		else:
			logger.info(f"Convergence checking enabled (relative change): tolerance={convergence_tol}, max_steps={max_steps}")
		
		logger.info(f"Starting time-stepping: running until convergence (max_steps={max_steps} as safety limit, max_wall_time=30.0s)")
		
		from ..mesh_management import map_scalar_dofs_to_gmsh
		initial_values = map_scalar_dofs_to_gmsh(u_prev.x.array, domain, mesh_data, cache_dict)
		initial_mean = float(np.mean(initial_values))
		
		time_steps, solutions_series, actual_steps, converged, wall_time_exceeded, elapsed_wall_time = \
			run_transient_time_stepping(
				problem, u_prev, final_solution, V, steady_state_solution,
				dt, max_steps, convergence_tol, check_convergence, initial_mean,
				domain, mesh_data, cache_dict
			)

		base_result = extract_func(final_solution, domain, mesh_data, cache_dict, "heat_transfer")
		if not base_result.get("success"):
			return base_result

		result, solution = assemble_transient_result(
			base_result, final_solution, steady_state_solution, V,
			time_steps, solutions_series, actual_steps, converged, wall_time_exceeded,
			elapsed_wall_time, dt, total_time, method, heat_capacity, initial_value,
			domain, mesh_data, cache_dict
		)
		
		return result

	except Exception as e:
		logger.error(f"Transient heat transfer solver failed: {e}", exc_info=True)
		raise

