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

		# Initialize context-based parser
		self.context_parser = ContextBasedParser(api_key, model)

		logger.info(f"Prompt Analyzer initialized with model: {model}")
		logger.info(f"Model configuration: OPENAI_MODEL={model}")

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

			result = self.context_parser.parse_prompt(prompt)

			# Add context to result for interactive demo
			result['updated_context'] = self.context_parser.get_context()

			return result

		except Exception as e:
			logger.error(f"Context-based parsing failed: {e}")
			raise RuntimeError(f"Failed to parse prompt: {e}")

	def clear_context(self):
		"""Clear the simulation context"""
		self.context_parser.clear_context()

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