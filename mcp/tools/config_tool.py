"""
MCP Tool for Configuration Management
"""

from typing import Dict, Any
from ..mcp_server import MCPTool, MCPToolType


def create_config_tool(config_manager) -> MCPTool:
    """Create configuration management MCP tool"""
    
    def get_config(key: str = None) -> Dict[str, Any]:
        """Get configuration value(s)"""
        try:
            if key:
                value = config_manager.get(key)
                return {
                    "success": True,
                    "key": key,
                    "value": value
                }
            else:
                all_config = config_manager.get_all()
                return {
                    "success": True,
                    "config": all_config
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_config(key: str, value: Any) -> Dict[str, Any]:
        """Update configuration value"""
        try:
            success = config_manager.set(key, value)
            if success:
                return {
                    "success": True,
                    "key": key,
                    "value": value
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to update {key}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def config_operation(operation: str, **kwargs) -> Dict[str, Any]:
        """Perform configuration operation"""
        if operation == "get":
            return get_config(kwargs.get("key"))
        elif operation == "update":
            return update_config(kwargs.get("key"), kwargs.get("value"))
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
    
    return MCPTool(
        name="manage_config",
        description="Get or update configuration values",
        tool_type=MCPToolType.FUNCTION,
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get", "update"],
                    "description": "Configuration operation to perform"
                },
                "key": {
                    "type": "string",
                    "description": "Configuration key (required for update, optional for get)"
                },
                "value": {
                    "description": "Configuration value (required for update)"
                }
            },
            "required": ["operation"]
        },
        handler=config_operation
    )

