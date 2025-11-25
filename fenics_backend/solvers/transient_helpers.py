"""
Transient solver helper functions
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl

from ..mesh_management import map_scalar_dofs_to_gmsh
from ..physical_groups import resolve_boundary_location
from ..config_helpers import configure_petsc_single_process

logger = logging.getLogger(__name__)
ScalarType = PETSc.ScalarType


def apply_bcs_to_initial_condition(u_prev, V, m, pde_config, bcs_dirichlet, fdim, mesh_data, resolve_func):
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
		if mesh_data:
			if isinstance(mesh_data, dict):
				physical_groups = mesh_data.get("physical_groups", {})
			elif hasattr(mesh_data, 'physical_groups'):
				physical_groups = mesh_data.physical_groups
		
		geometry_type = (pde_config.get("mesh_parameters", {}).get("geometry_type") or
						pde_config.get("geometry_type") or "")
		
		# Use unified boundary location resolution
		dofs, method, confidence = resolve_func(loc, V, m, physical_groups, geometry_type, pde_config, fdim)
		
		# Apply the boundary value to initial condition
		if dofs is not None and (isinstance(dofs, np.ndarray) and dofs.size > 0 or (not isinstance(dofs, np.ndarray) and hasattr(dofs, '__len__') and len(dofs) > 0)):
			dofs_size = dofs.size if isinstance(dofs, np.ndarray) else len(dofs)
			u_prev.x.array[dofs] = val
			bc_applied_count += 1
			logger.debug(f"Applied BC value {val:.3f} K ({val - 273.15:.3f} C) to {dofs_size} DOFs at '{loc}' (method: {method}, confidence: {confidence:.2f})")
		else:
			logger.warning(f"Could not locate DOFs for BC at '{loc}' (no DOFs found), will be enforced during solve")
	
	logger.debug(f"Applied {bc_applied_count}/{len(bcs_dirichlet)} boundary conditions to initial condition")


def compute_steady_state_for_convergence(m, V, kappa, u, v, dx, a_extra, L_extra, bcs_dirichlet, petsc_opts, prefix, PETSc):
	"""Compute steady-state solution for convergence checking."""
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
		
		configure_petsc_single_process(petsc_opts, PETSc)
		
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


def run_transient_time_stepping(problem, u_prev, final_solution, V, steady_state_solution, 
                               dt, max_steps, convergence_tol, check_convergence, initial_mean,
                               domain, original_mesh_data, cache_dict):
	"""Run transient time-stepping loop until convergence."""
	time_steps = [0.0]
	solutions_series = []
	
	# Record initial condition
	initial_values = map_scalar_dofs_to_gmsh(u_prev.x.array, domain, original_mesh_data, cache_dict)
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
		
		gmsh_values = map_scalar_dofs_to_gmsh(u_step.x.array, domain, original_mesh_data, cache_dict)
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


def assemble_transient_result(base_result, final_solution, steady_state_solution, V, 
                               time_steps, solutions_series, actual_steps, converged, wall_time_exceeded,
                               elapsed_wall_time, dt, total_time, method, heat_capacity, initial_value,
                               domain, original_mesh_data, cache_dict):
	"""Assemble final result dictionary for transient simulation."""
	actual_final_time = time_steps[-1] if time_steps else total_time
	
	solution = fem.Function(V, name="temperature")
	solution.x.array[:] = final_solution.x.array
	
	final_values = map_scalar_dofs_to_gmsh(final_solution.x.array, domain, original_mesh_data, cache_dict)
	final_min = float(np.min(final_values))
	final_max = float(np.max(final_values))
	final_mean = float(np.mean(final_values))
	logger.info(f"Final temperature field at t={actual_final_time:.4f}: min={final_min:.3f}, max={final_max:.3f}, mean={final_mean:.3f}")
	
	# Add steady-state solution if available
	steady_state_data = None
	if steady_state_solution is not None:
		steady_state_values = map_scalar_dofs_to_gmsh(steady_state_solution.x.array, domain, original_mesh_data, cache_dict)
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
	
	return base_result, solution

