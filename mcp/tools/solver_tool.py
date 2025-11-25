"""
MCP Tool for PDE Solving
"""

from typing import Dict, Any
from ..mcp_server import MCPTool, MCPToolType


def create_solver_tool(fenics_solver) -> MCPTool:
    """Create PDE solver MCP tool"""
    
    def solve_pde(simulation_config: Dict[str, Any], mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve PDE using FEniCS"""
        try:
            result = fenics_solver.solve_simulation(simulation_config, mesh_data)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    return MCPTool(
        name="solve_pde",
        description="Solve a PDE using FEniCS solver",
        tool_type=MCPToolType.FUNCTION,
        input_schema={
            "type": "object",
            "properties": {
                "simulation_config": {
                    "type": "object",
                    "description": "Complete simulation configuration including PDE config"
                },
                "mesh_data": {
                    "type": "object",
                    "description": "Mesh data with vertices, cells, etc."
                }
            },
            "required": ["simulation_config", "mesh_data"]
        },
        handler=solve_pde
    )

