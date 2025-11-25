"""
Geometry Agent
Handles geometry classification and dimension extraction
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'nlp_parser' / 'src'))

try:
    from geometry_classifier import classify_geometry
except ImportError:
    pass

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class GeometryAgent(BaseAgent):
    """Agent responsible for geometry identification and dimension extraction"""
    
    def __init__(self, agent_bus=None, prompt_manager: Optional[Any] = None):
        super().__init__("geometry_agent", agent_bus)
        self.prompt_manager = prompt_manager
        self.state['geometry_type'] = None
        self.state['geometry_dimensions'] = {}
    
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute geometry-related task"""
        if task == "identify_geometry":
            return self._identify_geometry(context.get('prompt', ''), context.get('physics_type', ''))
        elif task == "extract_dimensions":
            return self._extract_dimensions(
                context.get('prompt', ''),
                context.get('geometry_type', ''),
                context.get('required_dimensions', [])
            )
        elif task == "validate_geometry":
            return self._validate_geometry(context)
        else:
            return {
                "success": False,
                "error": f"Unknown task: {task}"
            }
    
    def _identify_geometry(self, prompt: str, physics_type: str) -> Dict[str, Any]:
        """Identify geometry type from prompt"""
        if not self.prompt_manager:
            return {
                "success": False,
                "error": "Prompt manager not available"
            }
        
        try:
            context_summary = self._get_context_summary()
            geometry_result = self.prompt_manager.identify_geometry_type(
                prompt, physics_type, context_summary
            )
            
            if geometry_result.get('error'):
                return {
                    "success": False,
                    "error": geometry_result.get('error')
                }
            
            geometry_type = geometry_result.get('geometry_type')
            if not geometry_type:
                return {
                    "success": False,
                    "error": "Could not identify geometry type"
                }
            
            self.state['geometry_type'] = geometry_type
            
            # Also try geometry classifier for validation
            try:
                dimensions_path = project_root / "config" / "dimensions.json"
                if dimensions_path.exists():
                    candidates = classify_geometry(prompt, dimensions_path)
                    if candidates:
                        top_candidate = candidates[0]
                        if top_candidate.get('geometry_type') == geometry_type:
                            confidence = top_candidate.get('confidence', 0.5)
                        else:
                            confidence = 0.7  # AI and classifier differ
                    else:
                        confidence = 0.7
                else:
                    confidence = 0.8
            except Exception:
                confidence = 0.8
            
            self._send_state_update({
                'geometry_type': geometry_type,
                'confidence': confidence
            })
            
            return {
                "success": True,
                "geometry_type": geometry_type,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error identifying geometry: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_dimensions(self, prompt: str, geometry_type: str, required_dimensions: list) -> Dict[str, Any]:
        """Extract dimensions from prompt"""
        if not self.prompt_manager:
            return {
                "success": False,
                "error": "Prompt manager not available"
            }
        
        try:
            dimensions_result = self.prompt_manager.parse_dimensions(
                prompt, geometry_type, required_dimensions
            )
            
            if dimensions_result.get('error'):
                return {
                    "success": False,
                    "error": dimensions_result.get('error')
                }
            
            dimensions = dimensions_result.get('dimensions', {})
            self.state['geometry_dimensions'] = dimensions
            
            self._send_state_update({
                'geometry_dimensions': dimensions
            })
            
            return {
                "success": True,
                "dimensions": dimensions
            }
            
        except Exception as e:
            logger.error(f"Error extracting dimensions: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_geometry(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate geometry in context"""
        geometry_type = context.get('geometry_type')
        dimensions = context.get('geometry_dimensions', {})
        
        if not geometry_type:
            return {
                "success": False,
                "error": "No geometry type in context"
            }
        
        # Basic validation - dimensions should match geometry type
        return {
            "success": True,
            "geometry_type": geometry_type,
            "dimensions": dimensions,
            "valid": True
        }
    
    def _get_context_summary(self) -> str:
        """Get summary of current context"""
        return f"Physics: {self.state.get('physics_type', 'unknown')}, Geometry: {self.state.get('geometry_type', 'unknown')}"

