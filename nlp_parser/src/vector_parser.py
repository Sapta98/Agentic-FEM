"""
Vector Value Parser
Utility functions for parsing vector values from various formats
"""

import re
from typing import Union, List, Optional, Any


def parse_vector_value(value: Any) -> Union[float, List[float], Any]:
    """
    Parse a value that could be a scalar, vector, or other type.
    
    Supports multiple formats:
    - Scalar: 1000, 1e6, "1000"
    - Vector arrays: [1000, 2000, 0], (1000, 2000, 0), {1000, 2000, 0}
    - Vector strings: "[1000, 2000, 0]", "(1000, 2000, 0)", "1000, 2000, 0"
    - Already parsed: [1000, 2000, 0] (list)
    
    Args:
        value: Input value (scalar, list, string, etc.)
    
    Returns:
        - If vector detected: List[float]
        - If scalar: float
        - Otherwise: value as-is
    """
    # If already a list/array, validate and return
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except (ValueError, TypeError):
            return value
    
    # If not a string, try to convert to float (scalar)
    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    
    # String input - try to parse as vector first
    trimmed = value.strip()
    if not trimmed:
        return value
    
    # Try to match bracketed formats: [x,y,z], (x,y,z), {x,y,z}
    # Updated pattern to support scientific notation (1e6, 1.5e-3) and negative numbers
    bracket_pattern = r'^[\[\(\{]\s*([-\d\.eE\+\s,]+)\s*[\]\)\}]$'
    bracket_match = re.match(bracket_pattern, trimmed)
    
    if bracket_match:
        # Extract content inside brackets
        components = bracket_match.group(1)
    elif ',' in trimmed:
        # Comma-separated format: x, y, z
        components = trimmed
    else:
        # Not a vector format, try as scalar
        try:
            return float(trimmed)
        except (ValueError, TypeError):
            return value
    
    # Split by comma and parse each component
    parts = [p.strip() for p in components.split(',') if p.strip()]
    if not parts:
        return value
    
    try:
        numbers = [float(p) for p in parts]
        # If we have at least one valid number, return as vector
        if len(numbers) > 0:
            return numbers
    except (ValueError, TypeError):
        pass
    
    # Fallback: try as scalar
    try:
        return float(trimmed)
    except (ValueError, TypeError):
        return value


def normalize_boundary_condition_value(value: Any, bc_type: str, physics_type: str) -> Any:
    """
    Normalize boundary condition value, converting vectors when appropriate.
    
    Args:
        value: Raw value from parser/AI
        bc_type: Boundary condition type (e.g., "traction", "displacement", "flux")
        physics_type: Physics type ("heat_transfer" or "solid_mechanics")
    
    Returns:
        Normalized value (scalar or list)
    """
    parsed = parse_vector_value(value)
    
    # For solid mechanics vector BCs, ensure list format
    if physics_type == "solid_mechanics":
        if bc_type in ("traction", "force", "displacement", "neumann"):
            if isinstance(parsed, (int, float)):
                # Scalar -> convert to list (applied along x-axis)
                return [float(parsed), 0.0, 0.0]
            elif isinstance(parsed, list):
                # Already a list, pad to 3D if needed
                while len(parsed) < 3:
                    parsed.append(0.0)
                return parsed[:3]  # Ensure max 3 components
    
    # For heat transfer, vectors are not typically used (but allow if provided)
    if physics_type == "heat_transfer":
        if bc_type in ("flux", "heat_flux", "neumann"):
            # Heat flux is typically scalar, but allow vector if provided
            if isinstance(parsed, list):
                # Use first component as scalar flux
                return parsed[0] if len(parsed) > 0 else 0.0
            return float(parsed) if isinstance(parsed, (int, float)) else parsed
    
    return parsed

