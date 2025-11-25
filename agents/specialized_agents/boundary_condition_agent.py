"""
Boundary Condition Agent
Handles boundary condition parsing and validation
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'nlp_parser' / 'src'))

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class BoundaryConditionAgent(BaseAgent):
    """Agent responsible for boundary condition extraction and validation"""
    
    def __init__(self, agent_bus=None, prompt_manager: Optional[Any] = None):
        super().__init__("boundary_condition_agent", agent_bus)
        self.prompt_manager = prompt_manager
        self.state['boundary_conditions'] = []
        self.state['external_loads'] = []
    
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute boundary condition-related task"""
        if task == "extract_boundary_conditions":
            return self._extract_boundary_conditions(
                context.get('prompt', ''),
                context.get('physics_type', ''),
                context.get('geometry_type', '')
            )
        elif task == "analyze_boundary_locations":
            return self._analyze_boundary_locations(
                context.get('prompt', ''),
                context.get('geometry_type', ''),
                context.get('available_boundaries', [])
            )
        elif task == "validate_boundary_conditions":
            return self._validate_boundary_conditions(context)
        else:
            return {
                "success": False,
                "error": f"Unknown task: {task}"
            }
    
    def _extract_boundary_conditions(self, prompt: str, physics_type: str, geometry_type: str) -> Dict[str, Any]:
        """Extract boundary conditions from prompt"""
        if not self.prompt_manager:
            return {
                "success": False,
                "error": "Prompt manager not available"
            }
        
        try:
            # First analyze boundary locations
            available_boundaries = self._get_available_boundaries(geometry_type)
            location_result = self.prompt_manager.analyze_boundary_locations(
                prompt, geometry_type, available_boundaries
            )
            
            if location_result.get('error'):
                return {
                    "success": False,
                    "error": location_result.get('error')
                }
            
            boundary_locations = location_result.get('boundary_locations', [])
            
            # Then analyze boundary condition types
            bc_result = self.prompt_manager.analyze_boundary_condition_types(
                prompt, physics_type, boundary_locations
            )
            
            if bc_result.get('error'):
                return {
                    "success": False,
                    "error": bc_result.get('error')
                }
            
            boundary_conditions = bc_result.get('boundary_conditions', [])
            
            # Parse and normalize vector values in boundary conditions
            boundary_conditions = self._normalize_bc_values(boundary_conditions, physics_type)
            
            # Also check for external loads
            try:
                context_summary = self._get_context_summary()
                loads_result = self.prompt_manager.analyze_external_loads(
                    prompt, physics_type, context_summary
                )
                if not loads_result.get('error'):
                    external_loads = loads_result.get('external_loads', [])
                    self.state['external_loads'] = external_loads
            except Exception as e:
                logger.debug(f"Could not analyze external loads: {e}")
                external_loads = []
            
            self.state['boundary_conditions'] = boundary_conditions
            
            self._send_state_update({
                'boundary_conditions': boundary_conditions,
                'external_loads': self.state.get('external_loads', [])
            })
            
            return {
                "success": True,
                "boundary_conditions": boundary_conditions,
                "external_loads": self.state.get('external_loads', [])
            }
            
        except Exception as e:
            logger.error(f"Error extracting boundary conditions: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_boundary_locations(self, prompt: str, geometry_type: str, available_boundaries: List[str]) -> Dict[str, Any]:
        """Analyze boundary locations from prompt"""
        if not self.prompt_manager:
            return {
                "success": False,
                "error": "Prompt manager not available"
            }
        
        try:
            result = self.prompt_manager.analyze_boundary_locations(
                prompt, geometry_type, available_boundaries
            )
            return result
        except Exception as e:
            logger.error(f"Error analyzing boundary locations: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_boundary_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate boundary conditions in context"""
        boundary_conditions = context.get('boundary_conditions', [])
        physics_type = context.get('physics_type')
        geometry_type = context.get('geometry_type')
        
        if not boundary_conditions:
            return {
                "success": False,
                "error": "No boundary conditions in context"
            }
        
        # Validate each boundary condition
        valid = True
        errors = []
        
        for bc in boundary_conditions:
            if 'type' not in bc:
                valid = False
                errors.append("Boundary condition missing 'type'")
            if 'location' not in bc:
                valid = False
                errors.append("Boundary condition missing 'location'")
        
        return {
            "success": valid,
            "boundary_conditions": boundary_conditions,
            "valid": valid,
            "errors": errors if not valid else []
        }
    
    def _get_available_boundaries(self, geometry_type: str) -> List[str]:
        """Get available boundaries for a geometry type"""
        import json
        try:
            config_path = project_root / "config" / "geometry_boundaries.json"
            with open(config_path, 'r') as f:
                geometry_boundaries = json.load(f)
            
            for dim_group in ['1d', '2d', '3d']:
                if dim_group in geometry_boundaries.get('geometries', {}):
                    if geometry_type in geometry_boundaries['geometries'][dim_group]:
                        return geometry_boundaries['geometries'][dim_group][geometry_type].get('available_boundaries', [])
        except Exception as e:
            logger.debug(f"Could not load geometry boundaries: {e}")
        
        # Fallback
        defaults = {
            'line': ['left', 'right'],
            'rod': ['left', 'right'],
            'plate': ['left', 'right', 'top', 'bottom'],
            'disc': ['circumference', 'center'],
            'cube': ['left', 'right', 'top', 'bottom', 'front', 'back'],
            'cylinder': ['top', 'bottom', 'curved surface'],
            'sphere': ['center', 'surface']
        }
        return defaults.get(geometry_type, ['left', 'right', 'top', 'bottom'])
    
    def _get_context_summary(self) -> str:
        """Get summary of current context"""
        return f"Physics: {self.state.get('physics_type', 'unknown')}, Geometry: {self.state.get('geometry_type', 'unknown')}"
    
    def _normalize_bc_values(self, boundary_conditions: List[Dict[str, Any]], physics_type: str) -> List[Dict[str, Any]]:
        """
        Normalize boundary condition values, parsing vectors from various formats.
        
        Supports:
        - Scalar: 1000, "1000"
        - Vector arrays: [1000, 2000, 0]
        - Vector strings: "[1000, 2000, 0]", "(1000, 2000, 0)", "1000, 2000, 0"
        """
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / 'nlp_parser' / 'src'))
        
        try:
            from vector_parser import parse_vector_value, normalize_boundary_condition_value
        except ImportError:
            # Fallback: simple parsing
            def parse_vector_value(v):
                if isinstance(v, (list, tuple)):
                    return [float(x) for x in v]
                if isinstance(v, str):
                    # Try to parse bracketed formats (supports scientific notation)
                    import re
                    match = re.match(r'^[\[\(\{]\s*([-\d\.eE\+\s,]+)\s*[\]\)\}]$', v.strip())
                    if match:
                        parts = [float(p.strip()) for p in match.group(1).split(',') if p.strip()]
                        return parts if len(parts) >= 2 else (float(v) if parts else v)
                    elif ',' in v:
                        parts = [float(p.strip()) for p in v.split(',') if p.strip()]
                        return parts if len(parts) >= 2 else (float(v) if parts else v)
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return v
            
            def normalize_boundary_condition_value(val, bc_type, phys_type):
                parsed = parse_vector_value(val)
                if phys_type == "solid_mechanics" and bc_type in ("traction", "force", "displacement", "neumann"):
                    if isinstance(parsed, (int, float)):
                        return [float(parsed), 0.0, 0.0]
                    elif isinstance(parsed, list):
                        while len(parsed) < 3:
                            parsed.append(0.0)
                        return parsed[:3]
                return parsed
        
        normalized = []
        for bc in boundary_conditions:
            bc_copy = bc.copy()
            bc_type = bc.get('type', '').lower()
            value = bc.get('value')
            
            if value is not None:
                normalized_value = normalize_boundary_condition_value(value, bc_type, physics_type)
                bc_copy['value'] = normalized_value
                if isinstance(normalized_value, list):
                    logger.debug(f"Parsed vector value for {bc_type} at {bc.get('location')}: {value} -> {normalized_value}")
            
            normalized.append(bc_copy)
        
        return normalized

