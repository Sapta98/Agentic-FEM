"""
MCP Tool for Mesh Generation
"""

from typing import Dict, Any
from ..mcp_server import MCPTool, MCPToolType


def create_mesh_tool(mesh_viewer) -> MCPTool:
    """Create mesh generation MCP tool"""
    
    def generate_mesh(geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
        """Generate mesh for given geometry"""
        try:
            validation = mesh_viewer.validate_geometry(geometry_type, dimensions)
            if not validation.get("valid"):
                return {
                    "success": False,
                    "error": validation.get("error", "Invalid geometry parameters")
                }
            
            result = mesh_viewer.generate_mesh_preview(geometry_type, dimensions)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    return MCPTool(
        name="generate_mesh",
        description="Generate a mesh for a given geometry type and dimensions",
        tool_type=MCPToolType.FUNCTION,
        input_schema={
            "type": "object",
            "properties": {
                "geometry_type": {
                    "type": "string",
                    "description": "Type of geometry (line, rod, plate, disc, cube, cylinder, etc.)"
                },
                "dimensions": {
                    "type": "object",
                    "description": "Geometry dimensions as key-value pairs",
                    "additionalProperties": {"type": "number"}
                }
            },
            "required": ["geometry_type", "dimensions"]
        },
        handler=generate_mesh
    )

