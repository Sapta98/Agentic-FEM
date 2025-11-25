"""
Agent System for Agentic FEM
Provides master-agent orchestration and specialized agents
"""

from .base_agent import BaseAgent
from .master_agent import MasterAgent

__all__ = ['BaseAgent', 'MasterAgent']

