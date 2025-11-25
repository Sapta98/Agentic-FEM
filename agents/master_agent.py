"""
Master Agent
Orchestrates specialized agents for complete simulation workflow
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent
from .communication.agent_bus import AgentBus, MessageType
from .specialized_agents import (
    PhysicsAgent, GeometryAgent, MaterialAgent,
    BoundaryConditionAgent, MeshAgent, SolverAgent
)

# Import for template manager
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'nlp_parser' / 'src'))

logger = logging.getLogger(__name__)


class MasterAgent(BaseAgent):
    """Master agent that orchestrates specialized agents"""
    
    def __init__(self, agent_bus: Optional[AgentBus] = None, 
                 prompt_manager=None, mesh_viewer=None, 
                 fenics_solver=None, field_visualizer=None,
                 parser=None):
        super().__init__("master_agent", agent_bus)
        
        # Store the existing parser for fallback and context management
        self.parser = parser
        
        # Initialize specialized agents
        self.physics_agent = PhysicsAgent(self.agent_bus, prompt_manager)
        self.geometry_agent = GeometryAgent(self.agent_bus, prompt_manager)
        self.material_agent = MaterialAgent(self.agent_bus, prompt_manager)
        self.bc_agent = BoundaryConditionAgent(self.agent_bus, prompt_manager)
        self.mesh_agent = MeshAgent(self.agent_bus, mesh_viewer)
        self.solver_agent = SolverAgent(self.agent_bus, fenics_solver, field_visualizer)
        
        # Store references for easy access
        self.agents = {
            'physics': self.physics_agent,
            'geometry': self.geometry_agent,
            'material': self.material_agent,
            'boundary_condition': self.bc_agent,
            'mesh': self.mesh_agent,
            'solver': self.solver_agent
        }
        
        # Workflow state
        self.state['workflow_stage'] = 'initialized'
        self.state['current_task_id'] = None
        self.state['pending_tasks'] = []
        self.state['completed_tasks'] = []
        
        # Activate all agents
        for agent in self.agents.values():
            agent.activate()
        
        logger.debug("Master agent initialized with all specialized agents")
    
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task by orchestrating specialized agents"""
        if task == "parse_simulation":
            return self._parse_simulation(context)
        elif task == "run_complete_simulation":
            return self._run_complete_simulation(context)
        elif task == "generate_mesh":
            return self._generate_mesh(context)
        elif task == "solve_pde":
            return self._solve_pde(context)
        elif task == "parse_boundary_condition":
            return self._parse_boundary_condition(context)
        elif task == "update_boundary_condition":
            return self._update_boundary_condition(context)
        elif task == "clear_context":
            return self._clear_context()
        else:
            return {
                "success": False,
                "error": f"Unknown task: {task}"
            }
    
    def _parse_simulation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse simulation prompt using existing parser (agentic workflow)"""
        task_id = str(uuid.uuid4())
        self.state['current_task_id'] = task_id
        self.state['workflow_stage'] = 'parsing'
        
        prompt = context.get('prompt', '')
        existing_context = context.get('context', {})
        
        # Use existing parser (required for agentic workflow)
        if not self.parser:
            logger.error("Parser not available - required for agentic workflow")
            return {
                "success": False,
                "error": "Parser not initialized. Agentic workflow requires parser."
            }
        
        try:
            logger.info("Master: Using parser for context-based parsing")
            result = self.parser.parse(prompt, existing_context)
            
            # Ensure result has the expected format
            # Parser returns "context", but we need "updated_context"
            if result.get("updated_context"):
                result_context = result["updated_context"]
            elif result.get("context"):
                # Parser returns "context" field - use it
                result_context = result["context"]
                # Also set updated_context for consistency
                result["updated_context"] = result_context
            else:
                result_context = existing_context.copy()
            
            # Return in expected format
            if result.get("action") == "simulation_ready":
                return {
                    "success": True,
                    "action": "simulation_ready",
                    "message": result.get("message", "Simulation context complete"),
                    "updated_context": result_context,
                    "simulation_config": result.get("simulation_config", self._build_simulation_config(result_context))
                }
            else:
                return {
                    "success": True,
                    "action": result.get("action", "continue"),
                    "message": result.get("message", ""),
                    "updated_context": result_context,
                    "missing_info": result.get("completeness", {}).get("missing", [])
                }
        except Exception as e:
            logger.error(f"Error parsing simulation with parser: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_context": existing_context
            }
    
    def _generate_mesh(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mesh using mesh agent (which uses existing mesh_viewer)"""
        self.state['workflow_stage'] = 'mesh_generation'
        
        try:
            # Extract geometry info from context (handle both formats)
            current_context = context.get('context', context)
            geometry_type = current_context.get('geometry_type') or context.get('geometry_type')
            geometry_dimensions = current_context.get('geometry_dimensions') or context.get('geometry_dimensions', {})
            
            # Normalize dimensions: convert strings to numbers (UI often sends strings)
            # This ensures the mesh agent receives properly typed dimensions
            normalized_dimensions = {}
            for key, value in geometry_dimensions.items():
                if value is None:
                    continue  # Skip None values
                try:
                    if isinstance(value, str):
                        value_str = value.strip()
                        if value_str:
                            normalized_dimensions[key] = float(value_str)
                    elif isinstance(value, (int, float)):
                        normalized_dimensions[key] = float(value)
                    else:
                        # Try to convert anyway
                        normalized_dimensions[key] = float(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not normalize dimension '{key}' with value '{value}': {e}")
                    # Skip invalid dimensions - mesh agent will handle validation
            
            # Prepare context for mesh agent
            mesh_context = {
                'geometry_type': geometry_type,
                'geometry_dimensions': normalized_dimensions if normalized_dimensions else geometry_dimensions
            }
            
            # Use mesh agent which wraps the existing mesh_viewer
            result = self.mesh_agent.execute_task("generate_mesh", mesh_context)
            
            if result.get('success'):
                mesh_data = result.get('mesh_data')
                context['mesh_data'] = mesh_data
                self.state['workflow_stage'] = 'mesh_ready'
                # Also store mesh visualization URL if available
                if result.get('mesh_visualization_url'):
                    context['mesh_visualization_url'] = result.get('mesh_visualization_url')
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating mesh: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _solve_pde(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Solve PDE using solver agent (which uses existing fenics_solver)"""
        self.state['workflow_stage'] = 'solving'
        
        try:
            # Build simulation config from context
            current_context = context.get('context', context)
            simulation_config = self._build_simulation_config(current_context)
            # Store the nested structure for return value, but pass unwrapped to solver
            context['simulation_config'] = simulation_config
            # Also store unwrapped version for solver agent
            context['simulation_config_unwrapped'] = simulation_config.get('simulation_config', simulation_config)
            
            # Solve using solver agent (which wraps existing fenics_solver)
            solve_result = self.solver_agent.execute_task("solve_pde", context)
            
            if solve_result.get('success'):
                solution_data = solve_result
                context['solution_data'] = solution_data
                self.state['workflow_stage'] = 'solved'
            
            return solve_result
            
        except Exception as e:
            logger.error(f"Error solving PDE: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_complete_simulation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete simulation workflow"""
        self.state['workflow_stage'] = 'running'
        
        try:
            # Step 1: Parse if needed
            current_context = context.get('context', {})
            if not self._is_context_complete(current_context):
                parse_result = self._parse_simulation({
                    'prompt': context.get('prompt', ''),
                    'context': current_context
                })
                if not parse_result.get('success'):
                    return parse_result
                current_context = parse_result.get('updated_context', {})
                context['context'] = current_context
            
            # Step 2: Generate mesh - use existing mesh_viewer directly
            mesh_result = self._generate_mesh(context)
            if not mesh_result.get('success'):
                return {
                    "success": True,
                    "action": "simulation_ready",
                    "message": f"Mesh generation failed: {mesh_result.get('error')}",
                    "simulation_config": self._build_simulation_config(current_context),
                    "context": current_context
                }
            
            # Store mesh data in context
            mesh_data = mesh_result.get('mesh_data')
            context['mesh_data'] = mesh_data
            
            # Step 3: Solve PDE - use existing solver directly
            solve_result = self._solve_pde(context)
            
            # Step 4: Create visualizations if solver succeeded
            mesh_visualization_url = mesh_result.get('mesh_visualization_url')
            field_visualization_url = None
            
            if solve_result.get('success'):
                # Try to create field visualization
                try:
                    if self.solver_agent.field_visualizer and mesh_data:
                        viz_result = self.solver_agent.execute_task("create_visualization", {
                            'mesh_data': mesh_data,
                            'solution_data': solve_result
                        })
                        if viz_result.get('success'):
                            field_visualization_url = viz_result.get('visualization_url')
                except Exception as e:
                    logger.warning(f"Could not create field visualization: {e}")
            
            # Build final result matching old format
            return {
                "success": True,
                "action": "pde_solved",
                "message": "Simulation completed successfully",
                "simulation_config": self._build_simulation_config(current_context),
                "context": current_context,
                "mesh_visualization_url": mesh_visualization_url,
                "field_visualization_url": field_visualization_url,
                "solution_data": solve_result
            }
            
        except Exception as e:
            logger.error(f"Error in complete simulation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _is_context_complete(self, context: Dict[str, Any]) -> bool:
        """Check if context has all required information"""
        required_fields = [
            'physics_type', 'material_type', 'geometry_type',
            'geometry_dimensions', 'boundary_conditions'
        ]
        
        for field in required_fields:
            value = context.get(field)
            if not value:
                return False
        
        # Check boundary conditions is not empty
        boundary_conditions = context.get('boundary_conditions', [])
        if not boundary_conditions or len(boundary_conditions) == 0:
            return False
        
        return True
    
    def _build_simulation_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build simulation configuration from context"""
        from template_manager import TemplateManager
        
        physics_type = context.get('physics_type', 'heat_transfer')
        geometry_type = context.get('geometry_type', '')
        tm = TemplateManager()
        template = tm.get_template(physics_type) or {}
        
        pde_config = template.get('pde_config', {}).copy()
        pde_config['physics_type'] = physics_type
        pde_config['material_properties'] = context.get('material_properties', {})
        
        # Get boundary conditions from context and validate them for the current geometry
        boundary_conditions = context.get('boundary_conditions', [])
        logger.debug(f"_build_simulation_config: boundary_conditions in context: {len(boundary_conditions) if isinstance(boundary_conditions, list) else 'N/A'}, geometry_type: {geometry_type}")
        
        if boundary_conditions and geometry_type:
            # Validate and filter boundary conditions to match the current geometry
            # Get available boundaries for this geometry
            available_boundaries = self._get_available_boundaries_for_geometry(geometry_type)
            logger.debug(f"Available boundaries for {geometry_type}: {available_boundaries}")
            
            # Filter boundary conditions to only include those with valid locations for this geometry
            valid_bcs = []
            for i, bc in enumerate(boundary_conditions):
                location = bc.get('location', '').strip()  # Keep original case for comparison
                location_lower = location.lower().strip()
                logger.debug(f"  BC {i}: location='{location}' (lower: '{location_lower}')")
                
                # Check if location is valid for this geometry (case-insensitive)
                location_valid = location in available_boundaries or location_lower in [b.lower() for b in available_boundaries]
                if location_valid or location_lower in ['all', 'all_boundary']:
                    # Location is already valid, use it as-is
                    valid_bcs.append(bc)
                    logger.debug(f"    Location '{location}' is valid for {geometry_type}")
                else:
                    # Try to find a valid mapping
                    mapped_location = self._map_boundary_location_for_geometry(location, geometry_type, available_boundaries)
                    if mapped_location:
                        bc_copy = bc.copy()
                        bc_copy['location'] = mapped_location
                        valid_bcs.append(bc_copy)
                        logger.info(f"    Mapped boundary location '{location}' to '{mapped_location}' for geometry {geometry_type}")
                    else:
                        logger.warning(f"    Boundary condition with location '{location}' is not valid for geometry {geometry_type}, skipping")
            
            if valid_bcs:
                pde_config['boundary_conditions'] = valid_bcs
                logger.info(f"Validated {len(boundary_conditions)} boundary conditions to {len(valid_bcs)} valid conditions for geometry {geometry_type}")
                for i, bc in enumerate(valid_bcs):
                    logger.debug(f"  Valid BC {i}: location={bc.get('location')}, type={bc.get('type')}, value={bc.get('value')}")
            else:
                # No valid boundary conditions - use empty list (solver will handle defaults)
                pde_config['boundary_conditions'] = []
                logger.warning(f"No valid boundary conditions for geometry {geometry_type} after validation")
        elif boundary_conditions:
            # Boundary conditions exist but no geometry_type - store them anyway
            pde_config['boundary_conditions'] = boundary_conditions
            logger.info(f"Stored {len(boundary_conditions)} boundary conditions without geometry validation (geometry_type not available)")
        else:
            pde_config['boundary_conditions'] = boundary_conditions
        
        if context.get('geometry_dimensions'):
            if 'mesh_parameters' not in pde_config:
                pde_config['mesh_parameters'] = {}
            pde_config['mesh_parameters']['dimensions'] = context.get('geometry_dimensions')
        
        # Build simulation_config structure
        sim_config = {
            "simulation_config": {
                "pde_config": pde_config,
                "required_components": {
                    "physics_type": physics_type,
                    "material_type": context.get('material_type'),
                    "geometry_type": geometry_type,
                    "geometry_dimensions": context.get('geometry_dimensions')
                }
            }
        }
        
        # CRITICAL: Ensure boundary_conditions are in required_components if they exist in pde_config
        if pde_config.get('boundary_conditions'):
            sim_config["simulation_config"]["required_components"]["boundary_conditions"] = pde_config['boundary_conditions']
            logger.info(f"Added boundary_conditions to simulation_config.required_components: {len(pde_config['boundary_conditions'])} BC(s)")
        else:
            logger.warning(f"No boundary_conditions in pde_config for _build_simulation_config (context had {len(boundary_conditions) if boundary_conditions else 0} BC(s))")
        
        logger.info(f"_build_simulation_config result: pde_config.boundary_conditions = {len(pde_config.get('boundary_conditions', []))} BC(s)")
        
        return sim_config
    
    def _get_required_dimensions(self, geometry_type: str) -> List[str]:
        """Get required dimensions for a geometry type"""
        import json
        try:
            dimensions_path = Path(__file__).parent.parent.parent / "config" / "dimensions.json"
            with open(dimensions_path, 'r') as f:
                dimensions_config = json.load(f)
            
            for dim_group in ['1d', '2d', '3d']:
                if dim_group in dimensions_config.get('geometries', {}):
                    if geometry_type in dimensions_config['geometries'][dim_group]:
                        geom_config = dimensions_config['geometries'][dim_group][geometry_type]
                        return [d['name'] for d in geom_config.get('dimensions', [])]
        except Exception as e:
            logger.debug(f"Could not load dimensions config: {e}")
        
        # Fallback
        defaults = {
            'line': ['length'],
            'rod': ['length'],
            'plate': ['length', 'width'],
            'disc': ['radius'],
            'cube': ['length', 'width', 'height'],
            'cylinder': ['radius', 'height']
        }
        return defaults.get(geometry_type, ['length'])
    
    def _get_available_boundaries_for_geometry(self, geometry_type: str) -> List[str]:
        """Get available boundary locations for a geometry type"""
        # Get available boundaries from geometry boundaries config
        try:
            import json
            from pathlib import Path
            boundaries_path = Path(__file__).parent.parent.parent / "config" / "geometry_boundaries.json"
            if boundaries_path.exists():
                with open(boundaries_path, 'r') as f:
                    boundaries_config = json.load(f)
                
                    # Config structure: geometries -> 1d/2d/3d -> geometry_type -> available_boundaries
                    geometry_type_lower = geometry_type.lower()
                    for dim_group in ['1d', '2d', '3d']:
                        if dim_group in boundaries_config.get('geometries', {}):
                            if geometry_type_lower in boundaries_config['geometries'][dim_group]:
                                geom_config = boundaries_config['geometries'][dim_group][geometry_type_lower]
                                boundaries = geom_config.get('available_boundaries', [])
                                if boundaries:
                                    logger.debug(f"Found boundaries for {geometry_type} (normalized: {geometry_type_lower}) in {dim_group}: {boundaries}")
                                    return boundaries
        except Exception as e:
            logger.warning(f"Could not load geometry boundaries config: {e}")
        
        # Fallback boundaries
        defaults = {
            'line': ['left', 'right'],
            'rod': ['left', 'right'],
            'bar': ['left', 'right'],
            'plate': ['left', 'right', 'top', 'bottom'],
            'membrane': ['left', 'right', 'top', 'bottom'],
            'rectangle': ['left', 'right', 'top', 'bottom'],
            'square': ['left', 'right', 'top', 'bottom'],
            'disc': ['circumference', 'center'],
            'cube': ['left', 'right', 'top', 'bottom', 'front', 'back'],
            'box': ['left', 'right', 'top', 'bottom', 'front', 'back'],
            'beam': ['left', 'right', 'top', 'bottom', 'front', 'back'],
            'cylinder': ['top', 'bottom', 'curved surface'],
            'sphere': ['surface', 'center']
        }
        return defaults.get(geometry_type.lower(), ['left', 'right', 'top', 'bottom'])
    
    def _map_boundary_location_for_geometry(self, location: str, geometry_type: str, available_boundaries: List[str]) -> Optional[str]:
        """Map a boundary location to a valid location for the geometry"""
        location_lower = location.lower().strip()
        
        # Geometry-specific mappings
        geometry_mappings = {
            'cylinder': {
                'left': 'top', 'right': 'bottom', 'front': 'top', 'back': 'bottom',
                'start': 'top', 'end': 'bottom', 'upper': 'top', 'lower': 'bottom'
            },
            'disc': {
                'left': 'circumference', 'right': 'center', 'top': 'circumference', 'bottom': 'center',
                'edge': 'circumference', 'perimeter': 'circumference', 'outer': 'circumference', 'inner': 'center'
            },
            'sphere': {
                'left': 'surface', 'right': 'surface', 'top': 'surface', 'bottom': 'surface',
                'edge': 'surface', 'perimeter': 'surface', 'outer': 'surface'
            }
        }
        
        # Check geometry-specific mapping
        if geometry_type.lower() in geometry_mappings:
            if location_lower in geometry_mappings[geometry_type.lower()]:
                mapped = geometry_mappings[geometry_type.lower()][location_lower]
                if mapped in available_boundaries:
                    return mapped
        
        # Check if location is already valid
        if location_lower in available_boundaries:
            return location_lower
        
        # Try direct case-insensitive match
        for available in available_boundaries:
            if available.lower() == location_lower:
                return available
        
        # No valid mapping found
        return None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "master_agent": self.get_state(),
            "specialized_agents": {
                name: agent.get_state() for name, agent in self.agents.items()
            }
        }
    
    def reset_all(self):
        """Reset all agents"""
        for agent in self.agents.values():
            agent.reset()
        self.reset()
        logger.info("All agents reset")
    
    def _parse_boundary_condition(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse boundary condition change using parser"""
        prompt = context.get('prompt', '')
        boundary_condition = context.get('boundary_condition', {})
        existing_context = context.get('context', {})
        
        if not self.parser:
            return {
                "success": False,
                "error": "Parser not available"
            }
        
        try:
            result = self.parser.parse(prompt, existing_context)
            
            if result.get("updated_context", {}).get("boundary_conditions"):
                updated_bcs = result["updated_context"]["boundary_conditions"]
                location = boundary_condition.get("location") or boundary_condition.get("mapped_location")
                
                # Find matching boundary condition by location
                updated_bc = None
                for bc in updated_bcs:
                    if (bc.get("location") == location or 
                        bc.get("mapped_location") == location or
                        bc.get("location") == boundary_condition.get("location")):
                        updated_bc = bc
                        break
                
                if updated_bc:
                    return {
                        "success": True,
                        "boundary_condition": updated_bc,
                        "updated_context": result.get("updated_context")
                    }
            
            # If no specific update found, return the original boundary condition
            return {
                "success": True,
                "boundary_condition": boundary_condition,
                "updated_context": result.get("updated_context", existing_context)
            }
        except Exception as e:
            logger.error(f"Error parsing boundary condition: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_boundary_condition(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update boundary condition and sync with pde_config"""
        boundary_condition = context.get('boundary_condition', {})
        updated_context = context.get('context', {})
        
        if not updated_context:
            return {
                "success": False,
                "error": "Context is required"
            }
        
        try:
            # Update state
            self.state.update(updated_context)
            
            # Build simulation config to get updated pde_config
            simulation_config = self._build_simulation_config(updated_context)
            pde_config = simulation_config.get("simulation_config", {}).get("pde_config", {})
            
            return {
                "success": True,
                "message": "Boundary condition and pde_config updated successfully",
                "updated_pde_config": pde_config,
                "context": updated_context
            }
        except Exception as e:
            logger.error(f"Error updating boundary condition: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _clear_context(self) -> Dict[str, Any]:
        """Clear all simulation context"""
        try:
            # Clear parser context
            if self.parser:
                self.parser.clear_context()
            
            # Clear all agent states
            for agent in self.agents.values():
                agent.reset()
            
            # Clear master agent state
            self.state.clear()
            self.state['workflow_stage'] = 'initialized'
            self.state['current_task_id'] = None
            self.state['pending_tasks'] = []
            self.state['completed_tasks'] = []
            
            logger.info("All context cleared via master agent")
            return {
                "success": True,
                "message": "All context cleared successfully"
            }
        except Exception as e:
            logger.error(f"Error clearing context: {e}")
            return {
                "success": False,
                "error": str(e)
            }

