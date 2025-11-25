"""
Parser Agent
Coordinates specialized agents to build a complete PDE configuration from a prompt.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ParserAgent(BaseAgent):
	"""Agent that orchestrates the agentic parsing workflow."""

	def __init__(
		self,
		agent_bus=None,
		context_parser=None,
		template_manager=None,
		physics_agent=None,
		material_agent=None,
		geometry_agent=None,
		dimension_agent=None,
		boundary_agent=None,
	):
		super().__init__("parser_agent", agent_bus)
		self.context_parser = context_parser
		self.template_manager = template_manager or getattr(context_parser, "template_manager", None)

		self.physics_agent = physics_agent
		self.material_agent = material_agent
		self.geometry_agent = geometry_agent
		self.dimension_agent = dimension_agent
		self.boundary_agent = boundary_agent

	def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
		if task == "parse_prompt":
			return self._parse_prompt(context)
		if task == "clear_context":
			return self._clear_context()

		logger.warning(f"Unknown task for ParserAgent: {task}")
		return {"success": False, "error": f"Unknown task: {task}"}

	# ------------------------------------------------------------------ #
	# Primary parsing pipeline
	# ------------------------------------------------------------------ #
	def _parse_prompt(self, context: Dict[str, Any]) -> Dict[str, Any]:
		if not self.context_parser:
			return {"success": False, "error": "Context parser not initialized"}

		prompt = (context or {}).get("prompt", "")
		existing_context = (context or {}).get("context", {})

		if not prompt:
			return {"success": False, "error": "Prompt is required"}

		# Restore existing context into parser
		if existing_context:
			self.context_parser.context = existing_context.copy()

		current_context = self.context_parser.get_context()

		# ------------------------------------------------------------------
		# Stage A: identify physics, material, geometry
		# ------------------------------------------------------------------
		primary_results = self._run_primary_identification(prompt, current_context)
		if not primary_results.get("success"):
			return primary_results

		physics_type = primary_results.get("physics_type")
		material_type = primary_results.get("material_type")
		geometry_type = primary_results.get("geometry_type")
		partial_pde_config = primary_results.get("partial_pde_config", {})

		# Update internal context
		update_payload = {
			"physics_type": physics_type,
			"material_type": material_type,
			"material_properties": primary_results.get("material_properties", {}),
			"geometry_type": geometry_type,
		}
		self.context_parser.update_context(update_payload)
		current_context = self.context_parser.get_context()

		# ------------------------------------------------------------------
		# Stage B: build expectations based on templates
		# ------------------------------------------------------------------
		expectation_result = self._build_expectations(physics_type, geometry_type)
		required_dimensions = expectation_result.get("required_dimensions", [])
		available_boundaries = expectation_result.get("available_boundaries", [])

		# ------------------------------------------------------------------
		# Stage C: Clean prompt (remove already-identified sections)
		# ------------------------------------------------------------------
		cleaned_prompt = self.context_parser._extract_and_remove_parsed_sections(
			prompt,
			geometry_type,
			physics_type,
			material_type,
			current_context.get("geometry_dimensions", {}),
		)
		logger.info("Original prompt: '%s'", prompt)
		logger.info(
			"Cleaned prompt for BC detection (removed geometry/physics/material sections): '%s'",
			cleaned_prompt,
		)

		# ------------------------------------------------------------------
		# Stage D: Extract dimensions & boundary conditions in parallel
		# ------------------------------------------------------------------
		dimensions_result = self._run_dimension_agent(
			cleaned_prompt, geometry_type, required_dimensions
		)
		if not dimensions_result.get("success"):
			logger.debug("Dimension agent failed or returned no data, using defaults if needed")

		boundary_result = self._run_boundary_agent(
			cleaned_prompt, physics_type, geometry_type
		)

		# Apply dimensions (converted to meters + aliases)
		self._apply_dimension_result(dimensions_result, geometry_type, required_dimensions)

		# Apply boundary conditions (with placeholders)
		complete_bcs = self._apply_boundary_result(
			boundary_result, physics_type, geometry_type, available_boundaries
		)

		if complete_bcs:
			self.context_parser.update_context({"boundary_conditions": complete_bcs})

		# ------------------------------------------------------------------
		# Stage E: Determine completeness & build final response
		# ------------------------------------------------------------------
		final_context = self.context_parser.get_context().copy()
		completeness = self.context_parser.check_completeness(
			final_context.get("physics_type", ""), final_context
		)
		is_complete = completeness.get("complete", False)

		if is_complete:
			simulation_config = self.context_parser._create_simulation_config()
			action = "simulation_ready"
			message = "Simulation context complete"
		else:
			simulation_config = {}
			action = "request_info"
			missing = completeness.get("missing", [])
			message = (
				"Additional information required: "
				+ ", ".join(missing) if missing else "Additional information required"
			)

		result = {
			"success": True,
			"action": action,
			"message": message,
			"context": current_context,
			"updated_context": final_context,
			"simulation_config": simulation_config,
			"partial_pde_config": partial_pde_config,
			"completeness": completeness,
		}

		# Ensure boundary conditions are surfaced for UI
		if complete_bcs:
			result["updated_context"]["boundary_conditions"] = complete_bcs
			if simulation_config:
				simulation_config.setdefault("pde_config", {})["boundary_conditions"] = complete_bcs

		return result

	# ------------------------------------------------------------------ #
	# Helper stages
	# ------------------------------------------------------------------ #
	def _run_primary_identification(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
		if not all([self.physics_agent, self.material_agent, self.geometry_agent]):
			return {"success": False, "error": "Primary agents not initialized"}

		results = {}
		with ThreadPoolExecutor(max_workers=3) as executor:
			futures = {
				"physics": executor.submit(
					self.physics_agent.execute_task,
					"identify_physics_type",
					{"prompt": prompt},
				),
				"material": executor.submit(
					self.material_agent.execute_task,
					"identify_material",
					{"prompt": prompt, "physics_type": context.get("physics_type", "")},
				),
				"geometry": executor.submit(
					self.geometry_agent.execute_task,
					"identify_geometry",
					{"prompt": prompt, "physics_type": context.get("physics_type", "")},
				),
			}

			for key, future in futures.items():
				try:
					results[key] = future.result()
				except Exception as exc:
					logger.error("Agent %s failed: %s", key, exc)
					results[key] = {"success": False, "error": str(exc)}

		physics_result = results.get("physics", {})
		material_result = results.get("material", {})
		geometry_result = results.get("geometry", {})

		if (
			not physics_result.get("success")
			or not physics_result.get("is_physics", True)
			or not physics_result.get("physics_type")
		):
			return {
				"success": False,
				"action": "request_info",
				"message": physics_result.get(
					"message",
					physics_result.get("error", "Unable to determine physics type"),
				),
				"updated_context": self.context_parser.get_context(),
				"partial_pde_config": {},
			}

		physics_type = physics_result.get("physics_type")

		material_type = material_result.get("material_type") or context.get("material_type")
		geometry_type = geometry_result.get("geometry_type") or context.get("geometry_type")

		partial_pde_config = {
			"physics_type": physics_type,
			"material_type": material_type,
			"geometry_type": geometry_type,
		}

		return {
			"success": True,
			"physics_type": physics_type,
			"material_type": material_type,
			"material_properties": material_result.get("material_properties", {}),
			"geometry_type": geometry_type,
			"partial_pde_config": partial_pde_config,
		}

	def _build_expectations(self, physics_type: str, geometry_type: str) -> Dict[str, Any]:
		required_dimensions = []
		if self.template_manager and physics_type and geometry_type:
			try:
				required_dimensions = self.template_manager.get_geometry_dimension_requirements(
					physics_type, geometry_type
				)
			except Exception as exc:
				logger.debug(f"Could not get dimension requirements: {exc}")

		available_boundaries = self.context_parser._get_available_boundaries(geometry_type)

		return {
			"required_dimensions": required_dimensions or [],
			"available_boundaries": available_boundaries or [],
		}

	def _run_dimension_agent(self, prompt: str, geometry_type: str, required_dimensions: list) -> Dict[str, Any]:
		if not self.dimension_agent:
			return {"success": False, "error": "Dimension agent not initialized"}

		return self.dimension_agent.execute_task(
			"extract_dimensions",
			{
				"prompt": prompt,
				"geometry_type": geometry_type,
				"required_dimensions": required_dimensions,
			},
		)

	def _run_boundary_agent(self, prompt: str, physics_type: str, geometry_type: str) -> Dict[str, Any]:
		if not self.boundary_agent:
			return {"success": False, "error": "Boundary agent not initialized"}

		return self.boundary_agent.execute_task(
			"extract_boundary_conditions",
			{
				"prompt": prompt,
				"physics_type": physics_type,
				"geometry_type": geometry_type,
			},
		)

	def _apply_dimension_result(
		self,
		dimension_result: Dict[str, Any],
		geometry_type: str,
		required_dimensions: list,
	) -> None:
		raw_dimensions = dimension_result.get("dimensions") if dimension_result else {}

		if raw_dimensions:
			units = dimension_result.get("units", {})
			try:
				converted_dims = self.context_parser._convert_dimension_units(raw_dimensions, units)
				self.context_parser.update_context(
					{
						"geometry_dimensions": converted_dims,
						"dimension_source": "prompt",
					}
				)
				logger.info(f"Extracted geometry dimensions from prompt: {converted_dims}")
				return
			except Exception as exc:
				logger.warning(f"Could not convert dimensions: {exc}")

		# Fallback to defaults if prompt extraction failed
		default_dims = self.context_parser._get_default_dimensions(geometry_type)
		if default_dims:
			required_set = {dim for dim in required_dimensions if dim in default_dims}
			if required_set:
				filtered_defaults = {dim: default_dims[dim] for dim in required_set}
			else:
				filtered_defaults = default_dims

			self.context_parser.update_context(
				{
					"geometry_dimensions": filtered_defaults,
					"dimension_source": "default",
				}
			)
			logger.info(f"Using default dimensions for {geometry_type}: {filtered_defaults}")

	def _apply_boundary_result(
		self,
		boundary_result: Dict[str, Any],
		physics_type: str,
		geometry_type: str,
		available_boundaries: list,
	) -> list:
		detected_bcs = []
		if boundary_result and boundary_result.get("success"):
			detected_bcs = boundary_result.get("boundary_conditions", []) or []

		complete_bcs = self.context_parser._add_placeholder_boundary_conditions(
			detected_bcs,
			geometry_type,
			physics_type,
			available_boundaries,
		)
		logger.info(
			"Boundary conditions for %s: detected=%d, total=%d",
			geometry_type,
			len(detected_bcs),
			len(complete_bcs),
		)
		return complete_bcs

	# ------------------------------------------------------------------ #
	# Utility
	# ------------------------------------------------------------------ #
	def _clear_context(self) -> Dict[str, Any]:
		if self.context_parser:
			self.context_parser.clear_context()
			return {"success": True}
		return {"success": False, "error": "Context parser not initialized"}

