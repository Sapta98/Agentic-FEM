"""
MCP Tool for Visualization
"""

from typing import Dict, Any
from ..mcp_server import MCPTool, MCPToolType


def create_visualization_tool(field_visualizer, mesh_visualizer) -> MCPTool:
    """Create visualization MCP tool"""
    
    def create_field_visualization(solution_data: Dict[str, Any], 
                                  mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create field visualization"""
        try:
            visualization_url = field_visualizer.create_field_visualization(
                solution_data, mesh_data
            )
            return {
                "success": True,
                "visualization_url": visualization_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_mesh_visualization(mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create mesh visualization"""
        try:
            visualization_url = mesh_visualizer.create_mesh_visualization(mesh_data)
            return {
                "success": True,
                "visualization_url": visualization_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # Return a combined tool that handles both types
    def visualize(data_type: str, **kwargs) -> Dict[str, Any]:
        """Create visualization based on type"""
        if data_type == "field":
            return create_field_visualization(
                kwargs.get("solution_data"),
                kwargs.get("mesh_data")
            )
        elif data_type == "mesh":
            return create_mesh_visualization(kwargs.get("mesh_data"))
        else:
            return {
                "success": False,
                "error": f"Unknown visualization type: {data_type}"
            }
    
    return MCPTool(
        name="create_visualization",
        description="Create mesh or field visualization",
        tool_type=MCPToolType.FUNCTION,
        input_schema={
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["mesh", "field"],
                    "description": "Type of visualization to create"
                },
                "mesh_data": {
                    "type": "object",
                    "description": "Mesh data (required for both types)"
                },
                "solution_data": {
                    "type": "object",
                    "description": "Solution data (required for field visualization)"
                }
            },
            "required": ["data_type", "mesh_data"]
        },
        handler=visualize
    )

