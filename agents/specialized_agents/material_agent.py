"""
Material Agent
Handles material property extraction and validation
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'nlp_parser' / 'src'))

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MaterialAgent(BaseAgent):
    """Agent responsible for material identification and property extraction"""
    
    def __init__(self, agent_bus=None, prompt_manager: Optional[Any] = None):
        super().__init__("material_agent", agent_bus)
        self.prompt_manager = prompt_manager
        self.state['material_type'] = None
        self.state['material_properties'] = {}
        self._load_material_properties()
    
    def _load_material_properties(self):
        """Load material properties from config"""
        try:
            material_props_path = project_root / "config" / "material_properties.json"
            if material_props_path.exists():
                with open(material_props_path, 'r') as f:
                    self.material_properties_db = json.load(f)
            else:
                self.material_properties_db = {}
        except Exception as e:
            logger.warning(f"Could not load material properties: {e}")
            self.material_properties_db = {}
    
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute material-related task"""
        if task == "identify_material":
            return self._identify_material(
                context.get('prompt', ''),
                context.get('physics_type', '')
            )
        elif task == "get_material_properties":
            return self._get_material_properties(
                context.get('material_type', ''),
                context.get('physics_type', '')
            )
        elif task == "validate_material":
            return self._validate_material(context)
        else:
            return {
                "success": False,
                "error": f"Unknown task: {task}"
            }
    
    def _identify_material(self, prompt: str, physics_type: str) -> Dict[str, Any]:
        """Identify material type from prompt"""
        if not self.prompt_manager:
            return {
                "success": False,
                "error": "Prompt manager not available"
            }
        
        try:
            context_summary = self._get_context_summary()
            material_result = self.prompt_manager.identify_material_type(
                prompt, physics_type, context_summary
            )
            
            if material_result.get('error'):
                return {
                    "success": False,
                    "error": material_result.get('error')
                }
            
            if not material_result.get('has_material_type', False):
                return {
                    "success": False,
                    "error": "No material type found in prompt"
                }
            
            material_type = material_result.get('material_type')
            self.state['material_type'] = material_type
            
            # Fetch material properties
            properties = self._fetch_material_properties(material_type, physics_type)
            self.state['material_properties'] = properties
            
            self._send_state_update({
                'material_type': material_type,
                'material_properties': properties
            })
            
            return {
                "success": True,
                "material_type": material_type,
                "material_properties": properties
            }
            
        except Exception as e:
            logger.error(f"Error identifying material: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_material_properties(self, material_type: str, physics_type: str) -> Dict[str, Any]:
        """Get material properties for a material type"""
        properties = self._fetch_material_properties(material_type, physics_type)
        
        self.state['material_properties'] = properties
        
        self._send_state_update({
            'material_properties': properties
        })
        
        return {
            "success": True,
            "material_type": material_type,
            "material_properties": properties
        }
    
    def _fetch_material_properties(self, material_type: str, physics_type: str) -> Dict[str, Any]:
        """Fetch material properties from database or AI"""
        # First try database
        if material_type in self.material_properties_db:
            props = self.material_properties_db[material_type].copy()
            # Filter by physics type if needed
            if physics_type == 'heat_transfer':
                return {
                    k: v for k, v in props.items()
                    if k in ['thermal_conductivity', 'density', 'specific_heat']
                }
            elif physics_type == 'solid_mechanics':
                return {
                    k: v for k, v in props.items()
                    if k in ['youngs_modulus', 'poissons_ratio', 'density']
                }
            return props
        
        # Fallback to AI if prompt_manager available
        if self.prompt_manager:
            try:
                ai_result = self.prompt_manager.get_material_properties(material_type, physics_type)
                if not ai_result.get('error'):
                    return ai_result.get('material_properties', {})
            except Exception as e:
                logger.warning(f"AI material property fetch failed: {e}")
        
        # Return empty dict if nothing found
        return {}
    
    def _validate_material(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate material in context"""
        material_type = context.get('material_type')
        material_properties = context.get('material_properties', {})
        
        if not material_type:
            return {
                "success": False,
                "error": "No material type in context"
            }
        
        if not material_properties:
            return {
                "success": False,
                "error": "No material properties in context"
            }
        
        return {
            "success": True,
            "material_type": material_type,
            "material_properties": material_properties,
            "valid": True
        }
    
    def _get_context_summary(self) -> str:
        """Get summary of current context"""
        return f"Physics: {self.state.get('physics_type', 'unknown')}, Material: {self.state.get('material_type', 'unknown')}"

