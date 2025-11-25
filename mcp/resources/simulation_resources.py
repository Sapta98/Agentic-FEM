"""
MCP Resources for Simulation State
"""

from typing import Dict, Any, List
from ..mcp_server import MCPResource


def create_simulation_resources(simulation_manager, master_agent) -> List[MCPResource]:
    """Create MCP resources for simulation state"""
    
    def get_simulation_context() -> Dict[str, Any]:
        """Get current simulation context"""
        return simulation_manager.get_current_context()
    
    def get_mesh_data() -> Dict[str, Any]:
        """Get current mesh data"""
        mesh_data = simulation_manager.get_current_mesh_data()
        return mesh_data if mesh_data else {}
    
    def get_solution_data() -> Dict[str, Any]:
        """Get current solution data"""
        config = simulation_manager.get_current_simulation_config()
        # Solution data would be stored separately or in config
        return config.get("solution_data", {})
    
    def get_agent_status() -> Dict[str, Any]:
        """Get agent system status"""
        return master_agent.get_agent_status()
    
    def get_simulation_config() -> Dict[str, Any]:
        """Get complete simulation configuration"""
        return simulation_manager.get_current_simulation_config()
    
    resources = [
        MCPResource(
            uri="fem://simulation/context",
            name="Simulation Context",
            description="Current simulation context including physics type, material, geometry, and boundary conditions",
            handler=get_simulation_context
        ),
        MCPResource(
            uri="fem://simulation/mesh",
            name="Mesh Data",
            description="Current mesh data including vertices, cells, and mesh statistics",
            handler=get_mesh_data
        ),
        MCPResource(
            uri="fem://simulation/solution",
            name="Solution Data",
            description="PDE solution data including field values and coordinates",
            handler=get_solution_data
        ),
        MCPResource(
            uri="fem://agents/status",
            name="Agent Status",
            description="Status of all agents in the system",
            handler=get_agent_status
        ),
        MCPResource(
            uri="fem://simulation/config",
            name="Simulation Configuration",
            description="Complete simulation configuration including PDE config and required components",
            handler=get_simulation_config
        )
    ]
    
    return resources

