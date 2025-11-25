#!/usr/bin/env python3
"""
Prompt Analyzer - Main Interface for Natural Language Processing
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
	from .context_based_parser import ContextBasedParser
except ImportError:
	from context_based_parser import ContextBasedParser

try:
	from agents.specialized_agents import (
		PhysicsAgent,
		GeometryAgent,
		MaterialAgent,
		BoundaryConditionAgent,
		DimensionAgent,
		ParserAgent,
	)
except ImportError:
	from agents.specialized_agents import (
		PhysicsAgent,
		GeometryAgent,
		MaterialAgent,
		BoundaryConditionAgent,
		DimensionAgent,
		ParserAgent,
	)

logger = logging.getLogger(__name__)

class SimulationPromptParser:
	"""Natural Language Parser for Physics Simulations"""

	def __init__(self, model: str = None):
		"""
		Initialize the parser with OpenAI backend

		Args:
			model: OpenAI model to use. If None, will use OPENAI_MODEL env var or default to gpt-4
		"""
		# Load environment variables from .env file first
		self._load_env_variables()

		# Determine model to use
		if model is None:
			model = os.getenv("OPENAI_MODEL", "gpt-4")

		self.model = model

		# Load API key from environment
		api_key = self._load_api_key()

		# Initialize context-based parser core
		self.context_parser = ContextBasedParser(api_key, model)
		prompt_manager = getattr(self.context_parser, "prompt_manager", None)
		template_manager = getattr(self.context_parser, "template_manager", None)

		# Specialized agents used by parser agent
		self.physics_agent = PhysicsAgent(prompt_manager=prompt_manager)
		self.material_agent = MaterialAgent(prompt_manager=prompt_manager)
		self.geometry_agent = GeometryAgent(prompt_manager=prompt_manager)
		self.boundary_agent = BoundaryConditionAgent(prompt_manager=prompt_manager)
		self.dimension_agent = DimensionAgent(prompt_manager=prompt_manager)

		# Parser agent orchestrates specialized agents and updates context parser
		self.parser_agent = ParserAgent(
			context_parser=self.context_parser,
			template_manager=template_manager,
			physics_agent=self.physics_agent,
			material_agent=self.material_agent,
			geometry_agent=self.geometry_agent,
			dimension_agent=self.dimension_agent,
			boundary_agent=self.boundary_agent,
		)

		logger.debug(f"Prompt Analyzer initialized with model: {model}")
		logger.debug(f"Model configuration: OPENAI_MODEL={model}")

	def _load_env_variables(self):
		"""Load environment variables from .env file"""
		# Load from .env file
		env_file = Path(__file__).parent / ".env"
		if env_file.exists():
			try:
				with open(env_file, 'r') as f:
					content = f.read()

					# Parse environment variables from .env file
					for line in content.split('\n'):
						line = line.strip()
						if line and not line.startswith('#') and '=' in line:
							key, value = line.split('=', 1)
							key = key.strip()
							value = value.strip().strip('"\'')

							# Set environment variable if not already set
							if key == 'OPENAI_API_KEY' and value and not value.startswith('your-'):
								os.environ[key] = value
							elif key == 'OPENAI_MODEL' and value:
								os.environ[key] = value
								logger.info(f"Loaded model from .env file: {value}")

			except Exception as e:
				logger.warning(f"Error reading .env file: {e}")

	def _load_api_key(self) -> str:
		"""Load OpenAI API key from environment"""
		api_key = os.getenv('OPENAI_API_KEY')
		if api_key:
			return api_key

		raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or create a .env file with your API key.")

	def parse(self, prompt: str, context: dict = None) -> dict:
		"""
		Parse natural language prompt using context-based approach

		Args:
			prompt: User's natural language prompt
			context: Previous context from previous iterations

		Returns:
			Dictionary with parsing results
		"""
		try:
			# If context is provided, restore it to the parser
			if context:
				self.context_parser.context = context.copy()

			result = self.parser_agent.execute_task(
				"parse_prompt",
				{
					"prompt": prompt,
					"context": self.context_parser.get_context(),
				},
			)

			# Get the latest context from the parser (should include all processed information)
			latest_context = self.context_parser.get_context()
			
			# Log boundary conditions in latest context for debugging
			if latest_context.get('boundary_conditions'):
				bcs = latest_context.get('boundary_conditions', [])
				logger.info(f"Latest context contains boundary_conditions: {len(bcs) if isinstance(bcs, list) else 'N/A'} BC(s)")
				if isinstance(bcs, list):
					for i, bc in enumerate(bcs):
						logger.debug(f"  BC {i}: location={bc.get('location')}, type={bc.get('type')}, value={bc.get('value')}, source={bc.get('source')}")
			else:
				logger.warning(f"Latest context does NOT contain boundary_conditions!")
				logger.debug(f"Latest context keys: {list(latest_context.keys())}")
			
			# Add context to result for interactive demo
			# If result already has 'context', merge it with latest_context to ensure all fields are included
			if result.get('context'):
				# Merge result.context with latest_context (latest_context takes precedence)
				merged_context = {**result.get('context', {}), **latest_context}
				result['updated_context'] = merged_context
				logger.debug(f"Merged context from result.context and latest_context")
			else:
				result['updated_context'] = latest_context
			
			# Also ensure result.context is set if it's not already
			if 'context' not in result:
				result['context'] = latest_context
			else:
				# Update result.context with latest_context to ensure consistency
				result['context'].update(latest_context)
			
			# Verify boundary conditions are in the final result
			final_context = result.get('updated_context') or result.get('context') or {}
			if final_context.get('boundary_conditions'):
				bcs = final_context.get('boundary_conditions', [])
				logger.info(f"Final result contains boundary_conditions: {len(bcs) if isinstance(bcs, list) else 'N/A'} BC(s)")
			else:
				logger.warning(f"Final result does NOT contain boundary_conditions!")
				logger.debug(f"Final context keys: {list(final_context.keys())}")

			return result

		except Exception as e:
			logger.error(f"Context-based parsing failed: {e}")
			raise RuntimeError(f"Failed to parse prompt: {e}")

	def clear_context(self):
		"""Clear the simulation context"""
		self.parser_agent.execute_task("clear_context", {})


	def get_parser_info(self) -> dict:
		"""Get information about the parser"""
		return {
			"parser_type": "context_based_natural_language_parser",
			"model": self.model,
			"version": "3.0.0",
			"description": "Context-based NLP parser that continuously builds simulation context and only asks for missing information",
			"features": [
				"Continuous context building across prompts",
				"Automatic material property fetching",
				"Smart dimension parsing",
				"Template-driven simulation configuration",
				"Progressive information gathering"
			]
		}

	def get_context(self) -> Dict[str, Any]:
		"""Get current simulation context"""
		return self.context_parser.get_context()

	def update_context(self, updates: Dict[str, Any]) -> None:
		"""Update simulation context with new information"""
		self.context_parser.update_context(updates)

	def validate_context(self) -> Dict[str, Any]:
		"""Validate current simulation context"""
		return self.context_parser.validate_context()

	def get_missing_information(self) -> list:
		"""Get list of missing information for simulation"""
		return self.context_parser.get_missing_information()

	def is_simulation_ready(self) -> bool:
		"""Check if simulation is ready to run"""
		return self.context_parser.is_simulation_ready()

	def get_simulation_summary(self) -> Dict[str, Any]:
		"""Get summary of current simulation configuration"""
		return self.context_parser.get_simulation_summary()

	def reset_parser(self) -> None:
		"""Reset parser to initial state"""
		self.context_parser.clear_context()
		logger.info("Parser reset to initial state")

	def get_available_models(self) -> list:
		"""Get list of available OpenAI models"""
		return ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]

	def set_model(self, model: str) -> None:
		"""Set the OpenAI model to use"""
		if model not in self.get_available_models():
			raise ValueError(f"Unsupported model: {model}")
		
		self.model = model
		self.context_parser.set_model(model)
		logger.info(f"Model changed to: {model}")

	def get_parsing_statistics(self) -> Dict[str, Any]:
		"""Get statistics about parsing performance"""
		return {
			"total_parses": getattr(self.context_parser, 'parse_count', 0),
			"successful_parses": getattr(self.context_parser, 'success_count', 0),
			"failed_parses": getattr(self.context_parser, 'error_count', 0),
			"average_response_time": getattr(self.context_parser, 'avg_response_time', 0.0),
			"model": self.model,
			"context_size": len(self.get_context())
		}

	def export_context(self, filepath: str) -> bool:
		"""Export current context to file"""
		try:
			import json
			context = self.get_context()
			with open(filepath, 'w') as f:
				json.dump(context, f, indent=2)
			logger.info(f"Context exported to {filepath}")
			return True
		except Exception as e:
			logger.error(f"Failed to export context: {e}")
			return False

	def import_context(self, filepath: str) -> bool:
		"""Import context from file"""
		try:
			import json
			with open(filepath, 'r') as f:
				context = json.load(f)
			self.context_parser.context = context
			logger.info(f"Context imported from {filepath}")
			return True
		except Exception as e:
			logger.error(f"Failed to import context: {e}")
			return False

	def get_parser_status(self) -> Dict[str, Any]:
		"""Get current parser status"""
		return {
			"initialized": True,
			"model": self.model,
			"context_loaded": bool(self.get_context()),
			"simulation_ready": self.is_simulation_ready(),
			"missing_info_count": len(self.get_missing_information()),
			"parser_info": self.get_parser_info()
		}