"""
MCP Client Implementation
Connects to external MCP servers
"""

import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP Client for connecting to external MCP servers"""
    
    def __init__(self, server_url: str, server_name: Optional[str] = None):
        """
        Initialize MCP client
        
        Args:
            server_url: URL of the MCP server
            server_name: Optional name for the server
        """
        self.server_url = server_url.rstrip('/')
        self.server_name = server_name or server_url
        self.initialized = False
        self.server_info: Optional[Dict[str, Any]] = None
        logger.info(f"MCP Client initialized for {self.server_name} at {self.server_url}")
    
    def initialize(self) -> bool:
        """Initialize connection to MCP server"""
        try:
            response = self._send_request("initialize", {})
            if "error" not in response:
                self.server_info = response.get("serverInfo")
                self.initialized = True
                logger.info(f"Connected to MCP server: {self.server_name}")
                return True
            else:
                logger.error(f"Failed to initialize MCP client: {response.get('error')}")
                return False
        except Exception as e:
            logger.error(f"Error initializing MCP client: {e}")
            return False
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server"""
        try:
            response = self._send_request("tools/list", {})
            return response.get("tools", [])
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        try:
            response = self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })
            return response
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {"error": str(e)}
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from MCP server"""
        try:
            response = self._send_request("resources/list", {})
            return response.get("resources", [])
        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            return []
    
    def get_resource(self, uri: str) -> Dict[str, Any]:
        """Get a resource from the MCP server"""
        try:
            response = self._send_request("resources/read", {"uri": uri})
            return response
        except Exception as e:
            logger.error(f"Error getting resource {uri}: {e}")
            return {"error": str(e)}
    
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server"""
        try:
            response = requests.post(
                f"{self.server_url}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("result", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": str(e)}

