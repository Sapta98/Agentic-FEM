"""
MCP Server Implementation
Exposes FEM simulation capabilities as MCP tools and resources
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MCPToolType(Enum):
    """Types of MCP tools"""
    FUNCTION = "function"
    PROCEDURE = "procedure"


@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    tool_type: MCPToolType = MCPToolType.FUNCTION
    input_schema: Optional[Dict[str, Any]] = None
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "inputSchema": self.input_schema or {}
        }


@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


class MCPServer:
    """MCP Server for exposing FEM simulation capabilities"""
    
    def __init__(self, name: str = "fem-simulation-server", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.server_info = {
            "name": name,
            "version": version,
            "protocol_version": "2024-11-05"
        }
        logger.debug(f"MCP Server {name} v{version} initialized")
    
    def register_tool(self, tool: MCPTool):
        """Register an MCP tool"""
        self.tools[tool.name] = tool
        logger.debug(f"Registered MCP tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource):
        """Register an MCP resource"""
        self.resources[resource.uri] = resource
        logger.debug(f"Registered MCP resource: {resource.uri}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available resources"""
        return [resource.to_dict() for resource in self.resources.values()]
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool"""
        if tool_name not in self.tools:
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        tool = self.tools[tool_name]
        
        if not tool.handler:
            return {
                "error": f"Tool '{tool_name}' has no handler"
            }
        
        try:
            result = tool.handler(**arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result) if not isinstance(result, str) else result
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "error": str(e),
                "isError": True
            }
    
    def get_resource(self, uri: str) -> Dict[str, Any]:
        """Get an MCP resource"""
        if uri not in self.resources:
            return {
                "error": f"Resource '{uri}' not found",
                "available_resources": list(self.resources.keys())
            }
        
        resource = self.resources[uri]
        
        if not resource.handler:
            return {
                "error": f"Resource '{uri}' has no handler"
            }
        
        try:
            content = resource.handler()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource.mime_type,
                        "text": json.dumps(content) if not isinstance(content, str) else content
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error getting resource {uri}: {e}")
            return {
                "error": str(e),
                "isError": True
            }
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return self.server_info
    
    def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol request"""
        if method == "initialize":
            return {
                "protocolVersion": self.server_info["protocol_version"],
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True}
                },
                "serverInfo": self.server_info
            }
        elif method == "tools/list":
            return {"tools": self.list_tools()}
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            return self.call_tool(tool_name, arguments)
        elif method == "resources/list":
            return {"resources": self.list_resources()}
        elif method == "resources/read":
            uri = params.get("uri")
            return self.get_resource(uri)
        else:
            return {
                "error": f"Unknown method: {method}",
                "available_methods": [
                    "initialize", "tools/list", "tools/call",
                    "resources/list", "resources/read"
                ]
            }

