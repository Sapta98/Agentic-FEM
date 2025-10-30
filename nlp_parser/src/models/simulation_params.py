"""
Simulation parameters data model
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class SimulationType(str, Enum):
	"""Types of physics simulations supported"""
	HEAT_TRANSFER = "heat_transfer"
	SOLID_MECHANICS = "solid_mechanics"
	FLUID_DYNAMICS = "fluid_dynamics"
	ELECTROMAGNETICS = "electromagnetics"


class GeometryType(str, Enum):
	"""Types of geometric configurations"""
	RECTANGULAR = "rectangular"
	CYLINDRICAL = "cylindrical"
	SPHERICAL = "spherical"
	COMPLEX = "complex"


class Material(BaseModel):
	"""Material properties for simulation"""
	name: str
	density: Optional[float] = None
	thermal_conductivity: Optional[float] = None
	specific_heat: Optional[float] = None
	youngs_modulus: Optional[float] = None
	poisson_ratio: Optional[float] = None
	thermal_expansion: Optional[float] = None


class Geometry(BaseModel):
	"""Geometric configuration"""
	type: GeometryType
	dimensions: Dict[str, float]  # e.g., {"length": 1.0, "width": 0.5, "height": 0.1}
	mesh_density: Optional[int] = Field(default=50, ge=10, le=1000)


class SimulationParameters(BaseModel):
	"""Complete simulation parameters extracted from natural language"""
	simulation_type: SimulationType
	geometry: Geometry
	materials: List[Material]
	boundary_conditions: Dict[str, Any]
	initial_conditions: Dict[str, Any]
	time_parameters: Optional[Dict[str, float]] = None
	solver_settings: Dict[str, Any] = Field(default_factory=dict)

class Config:
	use_enum_values = True
