"""
Unit conversion utilities for FEniCS solver
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def convert_heat_flux_units(value: float, units: Optional[str] = None) -> float:
	"""
	Convert heat flux from various units to W/m² (SI base unit).
	
	Supported units:
	- W/m², W/m^2 (SI base unit) - no conversion
	- W/cm², W/cm^2 - multiply by 10000
	- cal/(cm²·s), cal/(cm^2·s) - multiply by 41840
	- BTU/(ft²·h), BTU/(ft^2·h) - multiply by 3.15459
	- kW/m², kW/m^2 - multiply by 1000
	- MW/m², MW/m^2 - multiply by 1000000
	"""
	if units is None:
		return value
	
	units = units.strip().lower().replace(" ", "").replace("·", "")
	
	# Conversion factors to W/m²
	conversions = {
		"w/m²": 1.0,
		"w/m^2": 1.0,
		"w/m2": 1.0,
		"w/cm²": 10000.0,
		"w/cm^2": 10000.0,
		"w/cm2": 10000.0,
		"cal/(cm²·s)": 41840.0,
		"cal/(cm^2·s)": 41840.0,
		"cal/(cm2·s)": 41840.0,
		"cal/(cm²s)": 41840.0,
		"cal/(cm^2s)": 41840.0,
		"cal/(cm2s)": 41840.0,
		"btu/(ft²·h)": 3.15459,
		"btu/(ft^2·h)": 3.15459,
		"btu/(ft2·h)": 3.15459,
		"btu/(ft²h)": 3.15459,
		"btu/(ft^2h)": 3.15459,
		"btu/(ft2h)": 3.15459,
		"kw/m²": 1000.0,
		"kw/m^2": 1000.0,
		"kw/m2": 1000.0,
		"mw/m²": 1000000.0,
		"mw/m^2": 1000000.0,
		"mw/m2": 1000000.0,
	}
	
	if units in conversions:
		converted_value = value * conversions[units]
		logger.debug(f"Converted heat flux: {value} {units} → {converted_value} W/m²")
		return converted_value
	else:
		logger.warning(f"Unknown heat flux unit: '{units}'. Assuming W/m². Supported units: {list(conversions.keys())}")
		return value


def convert_pressure_units(value: float, units: Optional[str] = None) -> float:
	"""
	Convert pressure/traction from various units to Pa (SI base unit).
	
	Supported units:
	- Pa, N/m², N/m^2 (SI base unit) - no conversion
	- kPa - multiply by 1000
	- MPa - multiply by 1000000
	- GPa - multiply by 1000000000
	- bar - multiply by 100000
	- atm - multiply by 101325
	- psi - multiply by 6894.76
	- ksi - multiply by 6894760
	- psf - multiply by 47.88
	"""
	if units is None:
		return value
	
	units = units.strip().lower().replace(" ", "")
	
	# Conversion factors to Pa
	conversions = {
		"pa": 1.0,
		"n/m²": 1.0,
		"n/m^2": 1.0,
		"n/m2": 1.0,
		"kpa": 1000.0,
		"mpa": 1000000.0,
		"gpa": 1000000000.0,
		"bar": 100000.0,
		"atm": 101325.0,
		"atmosphere": 101325.0,
		"psi": 6894.76,
		"ksi": 6894760.0,
		"psf": 47.88,
		"lbf/ft²": 47.88,
		"lbf/ft^2": 47.88,
		"lbf/ft2": 47.88,
	}
	
	if units in conversions:
		converted_value = value * conversions[units]
		logger.debug(f"Converted pressure/traction: {value} {units} → {converted_value} Pa")
		return converted_value
	else:
		logger.warning(f"Unknown pressure/traction unit: '{units}'. Assuming Pa. Supported units: {list(conversions.keys())}")
		return value

