"""
NLP Parser Module for Agentic Finite Element Simulations

This module provides OpenAI-powered natural language processing capabilities to interpret
physics simulation prompts and convert them into structured PDE configurations.
"""

from .context_based_parser import ContextBasedParser
from .prompt_analyzer import SimulationPromptParser
from .prompt_templates import PromptManager
from .template_manager import TemplateManager
from .geometry_classifier import classify_geometry
from .boundary_location_classifier import classify_boundary_locations
from .models.simulation_params import SimulationParameters
from .models.pde_config import PDEConfig

__version__ = "3.0.0"
__all__ = [
"ContextBasedParser",
"SimulationPromptParser",
"PromptManager",
"TemplateManager",
"classify_geometry",
"classify_boundary_locations",
"SimulationParameters",
"PDEConfig"
]
