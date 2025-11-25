"""
Dimension Agent
Handles geometry dimension extraction and basic normalization
"""

import logging
from typing import Dict, Any, Optional

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DimensionAgent(BaseAgent):
	"""Agent responsible for extracting geometric dimensions from natural language prompts."""

	def __init__(self, agent_bus=None, prompt_manager: Optional[Any] = None):
		super().__init__("dimension_agent", agent_bus)
		self.prompt_manager = prompt_manager
		self.state["dimensions"] = {}
		self.state["units"] = {}

	def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Route supported tasks to handler methods."""
		if task == "extract_dimensions":
			return self._extract_dimensions(
				context.get("prompt", ""),
				context.get("geometry_type", ""),
				context.get("required_dimensions", []),
			)

		logger.warning(f"Unknown task for DimensionAgent: {task}")
		return {
			"success": False,
			"error": f"Unknown task: {task}",
		}

	def _extract_dimensions(
		self,
		prompt: str,
		geometry_type: str,
		required_dimensions: Optional[list],
	) -> Dict[str, Any]:
		"""Use PromptManager to extract raw dimension values and units from prompt text."""
		if not self.prompt_manager:
			return {
				"success": False,
				"error": "Prompt manager not available",
			}

		try:
			required_dimensions = required_dimensions or []
			result = self.prompt_manager.parse_dimensions(prompt, geometry_type, required_dimensions)

			if result.get("error"):
				return {
					"success": False,
					"error": result.get("error"),
				}

			dimensions = result.get("dimensions", {}) or {}
			units = result.get("units", {}) or {}

			self.state["dimensions"] = dimensions
			self.state["units"] = units

			self._send_state_update({
				"dimensions": dimensions,
				"units": units,
				"confidence": result.get("confidence"),
			})

			return {
				"success": True,
				"dimensions": dimensions,
				"units": units,
				"confidence": result.get("confidence"),
				"reasoning": result.get("reasoning"),
			}

		except Exception as exc:
			logger.error(f"Error extracting dimensions: {exc}")
			return {
				"success": False,
				"error": str(exc),
			}

