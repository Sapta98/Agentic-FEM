"""
Solver Agent
Handles PDE solving coordination
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'fenics_backend'))

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SolverAgent(BaseAgent):
    """Agent responsible for PDE solving coordination"""
    
    def __init__(self, agent_bus=None, fenics_solver=None, field_visualizer=None):
        super().__init__("solver_agent", agent_bus)
        self.fenics_solver = fenics_solver
        self.field_visualizer = field_visualizer
        self.state['solution_data'] = None
        self.state['solution_complete'] = False
    
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute solver-related task"""
        if task == "solve_pde":
            return self._solve_pde(context)
        elif task == "create_visualization":
            return self._create_visualization(context)
        elif task == "validate_solution":
            return self._validate_solution()
        else:
            return {
                "success": False,
                "error": f"Unknown task: {task}"
            }
    
    def _solve_pde(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Solve PDE using FEniCS"""
        if not self.fenics_solver:
            return {
                "success": False,
                "error": "FEniCS solver not available"
            }
        
        try:
            # Extract simulation config from context (handle nested structure)
            # Try unwrapped version first, then fall back to nested
            simulation_config = context.get('simulation_config_unwrapped') or context.get('simulation_config', {})
            mesh_data = context.get('mesh_data')
            
            if not simulation_config:
                return {
                    "success": False,
                    "error": "No simulation config in context"
                }
            
            if not mesh_data:
                return {
                    "success": False,
                    "error": "No mesh data in context"
                }
            
            # Handle nested simulation_config structure: {"simulation_config": {"pde_config": {...}}}
            # FEniCS solver expects: {"pde_config": {...}} at top level
            if "simulation_config" in simulation_config:
                # Extract pde_config from nested structure
                nested_sim_config = simulation_config.get("simulation_config", {})
                if "pde_config" in nested_sim_config:
                    # Extract pde_config to top level for FEniCS solver
                    config_for_solver = {"pde_config": nested_sim_config["pde_config"]}
                else:
                    # Use nested structure as-is
                    config_for_solver = nested_sim_config
            elif "pde_config" in simulation_config:
                # Already has pde_config at top level
                config_for_solver = simulation_config
            else:
                # Use as-is and hope FEniCS can handle it
                config_for_solver = simulation_config
            
            # Provide mesh file path if available
            msh_file = mesh_data.get('msh_file')
            if msh_file:
                self.fenics_solver.set_mesh_file(msh_file)
            
            # Solve using existing fenics_solver (preserves old workflow exactly)
            logger.debug(f"Solving PDE with config structure: {list(config_for_solver.keys())}")
            solution_result = self.fenics_solver.solve_simulation(config_for_solver, mesh_data)
            
            if solution_result.get("success"):
                self.state['solution_data'] = solution_result
                self.state['solution_complete'] = True
                
                self._send_state_update({
                    'solution_complete': True,
                    'has_solution': True
                })
            
            # Return in same format as old workflow
            return solution_result
            
        except Exception as e:
            logger.error(f"Error solving PDE: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_visualization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create field visualization"""
        if not self.field_visualizer:
            return {
                "success": False,
                "error": "Field visualizer not available"
            }
        
        try:
            # Get solution_data from context first (passed from master agent), then fall back to state
            solution_data = context.get('solution_data') or self.state.get('solution_data')
            mesh_data = context.get('mesh_data')
            
            if not solution_data:
                logger.error("No solution data available for visualization")
                return {
                    "success": False,
                    "error": "No solution data available"
                }
            
            if not mesh_data:
                logger.error("No mesh data available for visualization")
                return {
                    "success": False,
                    "error": "No mesh data available"
                }
            
            # Check if solution_data has the required fields
            if not solution_data.get('coordinates') or not solution_data.get('values'):
                logger.error(f"Solution data missing required fields: coordinates={bool(solution_data.get('coordinates'))}, values={bool(solution_data.get('values'))}")
                return {
                    "success": False,
                    "error": "Solution data missing coordinates or values"
                }
            
            logger.debug(f"Creating visualization with {len(solution_data.get('coordinates', []))} coordinates and {len(solution_data.get('values', []))} values")
            visualization_url = self.field_visualizer.create_field_visualization(
                solution_data, mesh_data
            )
            
            if visualization_url:
                logger.info(f"Field visualization created: {visualization_url}")
            
            return {
                "success": True,
                "visualization_url": visualization_url
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_solution(self) -> Dict[str, Any]:
        """Validate solution data"""
        solution_data = self.state.get('solution_data')
        
        if not solution_data:
            return {
                "success": False,
                "error": "No solution data available",
                "valid": False
            }
        
        # Basic validation
        required_keys = ['coordinates', 'values']
        missing_keys = [key for key in required_keys if key not in solution_data]
        
        if missing_keys:
            return {
                "success": False,
                "error": f"Solution data missing keys: {missing_keys}",
                "valid": False
            }
        
        return {
            "success": True,
            "valid": True,
            "coordinate_count": len(solution_data.get('coordinates', [])),
            "value_count": len(solution_data.get('values', []))
        }

