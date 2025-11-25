"""
Mesh Agent
Handles mesh generation decisions and coordination
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MeshAgent(BaseAgent):
    """Agent responsible for mesh generation coordination"""
    
    def __init__(self, agent_bus=None, mesh_viewer=None):
        super().__init__("mesh_agent", agent_bus)
        self.mesh_viewer = mesh_viewer
        self.state['mesh_data'] = None
        self.state['mesh_generated'] = False
    
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mesh-related task"""
        if task == "generate_mesh":
            return self._generate_mesh(context)
        elif task == "validate_mesh":
            return self._validate_mesh(context)
        elif task == "get_mesh_data":
            return self._get_mesh_data()
        else:
            return {
                "success": False,
                "error": f"Unknown task: {task}"
            }
    
    def _generate_mesh(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mesh from context"""
        if not self.mesh_viewer:
            return {
                "success": False,
                "error": "Mesh viewer not available"
            }
        
        try:
            geometry_type = context.get('geometry_type')
            geometry_dimensions = context.get('geometry_dimensions', {})
            
            if not geometry_type:
                return {
                    "success": False,
                    "error": "No geometry type in context"
                }
            
            if not geometry_dimensions:
                return {
                    "success": False,
                    "error": "No geometry dimensions in context"
                }
            
            # Filter out None values and convert string dimensions to numbers
            # This prevents the '<=' comparison error with None and string types
            filtered_dimensions = {}
            none_dimensions = []
            for key, value in geometry_dimensions.items():
                if value is None:
                    none_dimensions.append(key)
                    logger.warning(f"Dimension '{key}' is None, will be treated as missing")
                else:
                    # Convert string values to float (dimensions from UI are often strings)
                    try:
                        if isinstance(value, str):
                            # Remove any whitespace and try to parse
                            value_str = value.strip()
                            if value_str:
                                filtered_dimensions[key] = float(value_str)
                            else:
                                none_dimensions.append(key)
                                logger.warning(f"Dimension '{key}' is empty string, will be treated as missing")
                        elif isinstance(value, (int, float)):
                            filtered_dimensions[key] = float(value)
                        else:
                            # Try to convert to float anyway
                            filtered_dimensions[key] = float(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Dimension '{key}' has invalid value '{value}' (type: {type(value).__name__}), will be treated as missing: {e}")
                        none_dimensions.append(key)
            
            if none_dimensions:
                logger.warning(f"Found None or invalid values for dimensions: {none_dimensions}")
            
            # Validate geometry with filtered dimensions
            validation = self.mesh_viewer.validate_geometry(geometry_type, filtered_dimensions)
            if not validation.get("valid"):
                error_msg = validation.get("errors", [])
                if isinstance(error_msg, list):
                    error_msg = "; ".join(error_msg)
                return {
                    "success": False,
                    "error": error_msg or "Invalid geometry parameters"
                }
            
            # Generate mesh using existing mesh_viewer (preserves old workflow)
            # Use filtered_dimensions to avoid passing None values
            result = self.mesh_viewer.generate_mesh_preview(geometry_type, filtered_dimensions)
            
            if result.get("success"):
                mesh_data = result.get("mesh_data")
                self.state['mesh_data'] = mesh_data
                self.state['mesh_generated'] = True
                
                self._send_state_update({
                    'mesh_generated': True,
                    'mesh_dimension': mesh_data.get('mesh_dimension') if mesh_data else None
                })
            
            # Return result in same format as old workflow
            return result
            
        except Exception as e:
            logger.error(f"Error generating mesh: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_mesh(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mesh in context"""
        mesh_data = self.state.get('mesh_data')
        
        if not mesh_data:
            return {
                "success": False,
                "error": "No mesh data available"
            }
        
        # Basic validation
        required_keys = ['vertices', 'cells', 'mesh_dimension']
        missing_keys = [key for key in required_keys if key not in mesh_data]
        
        if missing_keys:
            return {
                "success": False,
                "error": f"Mesh data missing keys: {missing_keys}",
                "valid": False
            }
        
        return {
            "success": True,
            "valid": True,
            "mesh_dimension": mesh_data.get('mesh_dimension'),
            "vertex_count": len(mesh_data.get('vertices', []))
        }
    
    def _get_mesh_data(self) -> Dict[str, Any]:
        """Get current mesh data"""
        mesh_data = self.state.get('mesh_data')
        
        if not mesh_data:
            return {
                "success": False,
                "error": "No mesh data available"
            }
        
        return {
            "success": True,
            "mesh_data": mesh_data
        }

