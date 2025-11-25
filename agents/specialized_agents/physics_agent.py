"""
Physics Agent
Handles physics type identification and validation
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'nlp_parser' / 'src'))

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PhysicsAgent(BaseAgent):
    """Agent responsible for physics type identification"""
    
    def __init__(self, agent_bus=None, prompt_manager: Optional[Any] = None):
        super().__init__("physics_agent", agent_bus)
        self.prompt_manager = prompt_manager
        self.state['physics_type'] = None
        self.state['confidence_scores'] = {}
    
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute physics-related task"""
        if task == "identify_physics_type":
            return self._identify_physics_type(context.get('prompt', ''))
        elif task == "validate_physics_type":
            return self._validate_physics_type(context)
        else:
            return {
                "success": False,
                "error": f"Unknown task: {task}"
            }
    
    def _identify_physics_type(self, prompt: str) -> Dict[str, Any]:
        """Identify physics type from prompt"""
        if not self.prompt_manager:
            return {
                "success": False,
                "error": "Prompt manager not available"
            }
        
        try:
            physics_result = self.prompt_manager.identify_physics_type(prompt)
            
            if physics_result.get('error'):
                return {
                    "success": False,
                    "error": physics_result.get('error'),
                    "physics_type": None
                }
            
            if not physics_result.get('is_physics', False):
                return {
                    "success": True,
                    "is_physics": False,
                    "message": "This does not appear to be a physics simulation",
                    "physics_type": None,
                    "confidence_scores": physics_result.get('confidence_scores', {})
                }
            
            # Determine physics type from confidence scores
            confidence_scores = physics_result.get('confidence_scores', {})
            heat_transfer_score = confidence_scores.get('heat_transfer', 0)
            solid_mechanics_score = confidence_scores.get('solid_mechanics', 0)
            
            # Heuristic keywords for solid mechanics
            text = (prompt or "").lower()
            sm_keywords = [
                "stress", "strain", "load", "force", "pressure load", "traction",
                "fixed", "clamped", "cantilever", "displacement", "deflection",
                "beam", "bar", "rod", "young's modulus", "poisson"
            ]
            
            if any(k in text for k in sm_keywords):
                physics_type = 'solid_mechanics'
                confidence_scores['solid_mechanics'] = max(solid_mechanics_score, 0.9)
            elif heat_transfer_score > solid_mechanics_score:
                physics_type = 'heat_transfer'
            else:
                physics_type = 'solid_mechanics' if solid_mechanics_score > 0 else 'heat_transfer'
            
            self.state['physics_type'] = physics_type
            self.state['confidence_scores'] = confidence_scores
            
            # Broadcast state update
            self._send_state_update({
                'physics_type': physics_type,
                'confidence_scores': confidence_scores
            })
            
            return {
                "success": True,
                "is_physics": True,
                "physics_type": physics_type,
                "confidence_scores": confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error identifying physics type: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_physics_type(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate physics type in context"""
        physics_type = context.get('physics_type')
        if not physics_type:
            return {
                "success": False,
                "error": "No physics type in context"
            }
        
        valid_types = ['heat_transfer', 'solid_mechanics']
        if physics_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid physics type: {physics_type}"
            }
        
        return {
            "success": True,
            "physics_type": physics_type,
            "valid": True
        }

