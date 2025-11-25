"""
Model Context Protocol (MCP) Integration
Provides MCP server and tools for external integration
"""

from .mcp_server import MCPServer
from .mcp_client import MCPClient

__all__ = ['MCPServer', 'MCPClient']

