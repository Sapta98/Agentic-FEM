"""
Specialized Agents for FEM Simulation
"""

from .physics_agent import PhysicsAgent
from .geometry_agent import GeometryAgent
from .material_agent import MaterialAgent
from .boundary_condition_agent import BoundaryConditionAgent
from .dimension_agent import DimensionAgent
from .parser_agent import ParserAgent
from .mesh_agent import MeshAgent
from .solver_agent import SolverAgent

__all__ = [
    'PhysicsAgent',
    'GeometryAgent',
    'MaterialAgent',
    'BoundaryConditionAgent',
    'MeshAgent',
    'SolverAgent'
]

