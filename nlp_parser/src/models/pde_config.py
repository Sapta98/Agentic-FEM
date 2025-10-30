"""
PDE configuration data models
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from enum import Enum


class BoundaryConditionType(str, Enum):
	"""Types of boundary conditions"""
	DIRICHLET = "dirichlet"
	NEUMANN = "neumann"
	ROBIN = "robin"
	PERIODIC = "periodic"


class BoundaryCondition(BaseModel):
	"""Boundary condition specification"""
	type: BoundaryConditionType
	boundary_id: str  # e.g., "left", "right", "top", "bottom"
	value: Union[float, str, Dict[str, Any]]  # Can be constant, function, or expression
	description: Optional[str] = None


class InitialCondition(BaseModel):
	"""Initial condition specification"""
	field: str  # e.g., "temperature", "displacement", "velocity"
	value: Union[float, str, Dict[str, Any]]  # Initial value or function
	description: Optional[str] = None


class PDEConfig(BaseModel):
	"""Complete PDE configuration for finite element simulation"""
	pde_type: str  # e.g., "heat_equation", "elasticity", "navier_stokes"
	equations: List[str]  # PDE equations in mathematical form
	variables: List[str]  # Dependent variables (e.g., ["temperature"], ["u_x", "u_y"])
	boundary_conditions: List[BoundaryCondition]
	initial_conditions: List[InitialCondition]
	material_properties: Dict[str, float]
	mesh_parameters: Dict[str, Any]
	solver_parameters: Dict[str, Any] = Field(default_factory=dict)

	# FEniCS-specific configuration
	function_space_order: int = Field(default=1, ge=1, le=3)
	time_stepping: Optional[Dict[str, Any]] = None

class Config:
	use_enum_values = True
