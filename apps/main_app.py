"""
Clean Main Application
Uses the new modular structure with central configuration
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our new modules
from config.config_manager import config_manager
from apps.mesh_viewer import MeshViewer
from apps.simulation import SimulationManager
from frontend.frontend_manager import FrontendManager
from nlp_parser.src.geometry_classifier import classify_geometry

# Import agent and MCP systems
from agents.master_agent import MasterAgent
from agents.communication.agent_bus import AgentBus
from mcp.mcp_server import MCPServer
from mcp.tools import create_mesh_tool, create_solver_tool, create_visualization_tool, create_config_tool
from mcp.resources import create_simulation_resources

# Add fenics_backend to path
sys.path.insert(0, str(project_root / 'fenics_backend'))

from config.logging_config import configure_logging, get_logger

# Setup logging once for the app
configure_logging()
logger = get_logger(__name__)

# Request/Response models
class SimulationRequest(BaseModel):
	prompt: str
	context: Optional[Dict[str, Any]] = None
	mesh_data: Optional[Dict[str, Any]] = None

class MeshPreviewRequest(BaseModel):
	geometry_type: str
	dimensions: Dict[str, float]

class ConfigUpdateRequest(BaseModel):
	key: str
	value: Any

class SimulationResponse(BaseModel):
	success: bool
	action: str
	message: Optional[str] = None
	simulation_config: Optional[Dict[str, Any]] = None
	context: Optional[Dict[str, Any]] = None
	guidance: Optional[Dict[str, Any]] = None
	mesh_visualization_url: Optional[str] = None
	field_visualization_url: Optional[str] = None
	solution_data: Optional[Dict[str, Any]] = None

# Initialize modules
mesh_viewer = MeshViewer()
frontend_manager = FrontendManager(config_manager)

# Initialize agent system
agent_bus = AgentBus()

# Get prompt manager for agents (will be initialized by master agent)
prompt_manager = None
try:
	from nlp_parser.src.prompt_templates import PromptManager
	from openai import OpenAI
	import os
	api_key = os.getenv("OPENAI_API_KEY")
	if api_key:
		client = OpenAI(api_key=api_key)
		model = os.getenv("OPENAI_MODEL", "gpt-4")
		prompt_manager = PromptManager(client, model)
except Exception as e:
	logger.warning(f"Could not initialize prompt manager for agents: {e}")

# Get FEniCS solver and field visualizer for agents
fenics_solver = None
field_visualizer = None
try:
	from fenics_backend import FEniCSSolver
	from local_field_visualizer import FieldVisualizer
	fenics_solver = FEniCSSolver()
	field_visualizer = FieldVisualizer()
except Exception as e:
	logger.warning(f"Could not initialize FEniCS components for agents: {e}")

# Get the existing parser for master agent (required for agentic workflow)
existing_parser = None
shared_template_manager = None
try:
	from prompt_analyzer import SimulationPromptParser
	existing_parser = SimulationPromptParser()
	# Reuse the template manager from the parser to avoid duplicate loads
	shared_template_manager = existing_parser.context_parser.template_manager
	logger.debug("Parser initialized for master agent")
except Exception as e:
	logger.error(f"Could not initialize parser: {e}")
	raise RuntimeError("Parser initialization failed - agentic workflow requires parser") from e

# Initialize master agent with existing parser (required)
master_agent = MasterAgent(
	agent_bus=agent_bus,
	prompt_manager=prompt_manager,
	mesh_viewer=mesh_viewer,
	fenics_solver=fenics_solver,
	field_visualizer=field_visualizer,
	parser=existing_parser
)

# Initialize simulation manager with master agent
simulation_manager = SimulationManager(mesh_viewer=mesh_viewer, master_agent=master_agent)

# Initialize MCP server
mcp_server = MCPServer(name="fem-simulation-server", version="2.2.0")

# Register MCP tools
if mesh_viewer:
	mcp_server.register_tool(create_mesh_tool(mesh_viewer))
if fenics_solver:
	mcp_server.register_tool(create_solver_tool(fenics_solver))
if field_visualizer:
	from frontend.visualizers.mesh_visualizer import MeshVisualizer
	mesh_visualizer = MeshVisualizer()
	mcp_server.register_tool(create_visualization_tool(field_visualizer, mesh_visualizer))
mcp_server.register_tool(create_config_tool(config_manager))

# Register MCP resources
for resource in create_simulation_resources(simulation_manager, master_agent):
	mcp_server.register_resource(resource)

# Create FastAPI application
app = FastAPI(
	title="Agentic FEM - Clean Architecture",
	description="Modular finite element simulation system with central configuration",
	version="2.1.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

# Add CORS middleware to handle preflight OPTIONS requests
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Allow all origins for local development
	allow_credentials=False,  # Must be False when allow_origins=["*"]
	allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
	allow_headers=["*"],  # Allow all headers
)

@app.on_event("startup")
async def startup_event():
	"""Initialize the application"""
	logger.info("Starting Agentic FEM - Clean Architecture v2.2.0")
	logger.info("Features: Modular Design + Master-Agent System + MCP Integration + Frontend Module")

	# Initialize frontend manager
	frontend_manager.initialize()
	logger.debug("Frontend manager initialized successfully")
	
	# Log agent system status
	logger.debug(f"Master agent initialized with {len(master_agent.agents)} specialized agents")
	logger.debug(f"MCP server initialized with {len(mcp_server.tools)} tools and {len(mcp_server.resources)} resources")

	logger.info("Application started successfully")

@app.get("/")
async def root():
	"""Serve the main interface"""
	# Use the frontend manager to get the main interface
	html_content = frontend_manager.get_main_interface()
	response = HTMLResponse(content=html_content)
	# Prevent caching to ensure users get the latest version
	response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
	response.headers["Pragma"] = "no-cache"
	response.headers["Expires"] = "0"
	return response

# Mount static files
app.mount("/static", StaticFiles(directory=str(project_root / "frontend" / "static")), name="static")

@app.get("/favicon.ico")
async def favicon():
	"""Serve favicon (prevents 404 errors)"""
	# Return 204 No Content to prevent browser from retrying
	return Response(status_code=204)

@app.get("/test")
async def test_endpoint():
	"""Test endpoint"""
	return {"message": "Test endpoint works"}

@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	return {
		"status": "healthy",
		"version": "2.2.0",
		"architecture": "modular",
		"components": {
			"config_manager": True,
			"mesh_viewer": mesh_viewer.mesh_generator is not None,
			"simulation_manager": simulation_manager.parser is not None,
			"fenics_solver": simulation_manager.fenics_solver is not None,
			"field_visualizer": simulation_manager.field_visualizer is not None,
			"frontend_manager": frontend_manager.is_initialized,
			"master_agent": master_agent is not None,
			"mcp_server": mcp_server is not None,
			"agent_count": len(master_agent.agents) if master_agent else 0,
			"mcp_tools": len(mcp_server.tools) if mcp_server else 0,
			"mcp_resources": len(mcp_server.resources) if mcp_server else 0
		}
	}

# Configuration endpoints
@app.get("/config")
async def get_config():
	"""Get current configuration"""
	return config_manager.get_all()

@app.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
	"""Update configuration value"""
	success = config_manager.set(request.key, request.value)
	if success:
		return {"success": True, "message": f"Updated {request.key}"}
	else:
		raise HTTPException(status_code=500, detail="Failed to update config")

@app.post("/config/reset")
async def reset_config():
	"""Reset configuration to defaults"""
	success = config_manager.reset_to_defaults()
	if success:
		return {"success": True, "message": "Configuration reset to defaults"}
	else:
		raise HTTPException(status_code=500, detail="Failed to reset config")

@app.get("/config/geometry")
async def get_geometry_config():
	"""Get geometry configuration for mesh preview"""
	return config_manager.get_geometry_for_mesh()

@app.get("/config/simulation")
async def get_simulation_config():
	"""Get full simulation configuration"""
	return config_manager.get_full_simulation_config()

@app.get("/config/boundary-condition-templates")
async def get_boundary_condition_templates(physics_type: str = None):
	"""Get boundary condition templates for physics types"""
	if physics_type:
		return config_manager.get_boundary_condition_templates(physics_type)
	else:
		return config_manager.get_boundary_condition_templates()

@app.get("/config/physics-types")
async def get_physics_types():
	"""Get available physics types"""
	return {
		"physics_types": config_manager.get_available_physics_types()
	}

@app.get("/config/dimensions-spec")
async def get_dimensions_spec():
	"""Serve the dimensions.json spec for frontend-driven dimension UI."""
	path = project_root / "config" / "dimensions.json"
	import json
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
		return JSONResponse(content=data)

@app.post("/nlp/geometry-candidates")
async def nlp_geometry_candidates(request: dict):
	"""Rank geometry candidates from prompt with confidence scores."""
	try:
		prompt = (request or {}).get("prompt", "")
		synonyms = (request or {}).get("synonyms") or None
		if not prompt:
			raise HTTPException(status_code=400, detail="prompt is required")

		dimensions_path = project_root / "config" / "dimensions.json"
		candidates = classify_geometry(prompt, dimensions_path, synonyms)
		return {"success": True, "candidates": candidates}
	except Exception as e:
		logger.error(f"Geometry NLP classification failed: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/add-boundary-condition")
async def add_boundary_condition_from_template(request: dict):
	"""Add boundary condition from template"""
	physics_type = request.get("physics_type")
	template_name = request.get("template_name")
	custom_params = request.get("custom_params", {})

	if not physics_type or not template_name:
		raise HTTPException(status_code=400, detail="physics_type and template_name are required")

	success = config_manager.add_boundary_condition_from_template(physics_type, template_name, custom_params)

	if success:
		return {
			"success": True,
			"message": f"Added {template_name} boundary condition",
			"boundary_conditions": config_manager.get("physics.boundary_conditions")
		}
	else:
		raise HTTPException(status_code=400, detail=f"Failed to add boundary condition template: {template_name}")

# Mesh preview endpoints (geometry only) - uses agentic workflow
@app.post("/mesh/preview")
async def generate_mesh_preview(request: MeshPreviewRequest):
	"""Generate mesh preview using master agent (agentic workflow)"""
	try:
		# Use master agent to generate mesh
		result = master_agent.execute_task("generate_mesh", {
			'geometry_type': request.geometry_type,
			'geometry_dimensions': request.dimensions
		})
		
		if not result.get("success"):
			raise HTTPException(status_code=500, detail=result.get("error", "Mesh generation failed"))
		
		# Update config with new geometry
		config_manager.update_geometry(request.geometry_type, request.dimensions)
		
		# Store mesh data in simulation manager for future simulation
		if "mesh_data" in result:
			simulation_manager.store_mesh_data(result["mesh_data"])
		
		return result
	except HTTPException:
		raise
	except Exception as e:
		logger.error(f"Error generating mesh preview: {e}")
		raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")

@app.get("/mesh/supported-geometries")
async def get_supported_geometries():
	"""Get supported geometry types"""
	return mesh_viewer.get_supported_geometries()

# Simulation endpoints (full workflow)
@app.post("/simulation/parse", response_model=SimulationResponse)
async def parse_simulation(request: SimulationRequest):
	"""Parse simulation prompt and update configuration"""
	try:
		# CRITICAL: Preserve user-modified boundary conditions from request.context
		# If request.context has boundary conditions with source='user' or is_user_modified=True,
		# we should merge them with parser results instead of overwriting them
		user_modified_bcs = []
		if request.context and request.context.get('boundary_conditions'):
			user_bcs = request.context.get('boundary_conditions', [])
			for bc in user_bcs:
				if isinstance(bc, dict):
					# Check if this BC is user-modified
					if bc.get('source') == 'user' or bc.get('is_user_modified') or not bc.get('is_placeholder'):
						if bc.get('source') != 'placeholder':
							user_modified_bcs.append(bc)
							logger.debug(f"Found user-modified BC: location={bc.get('location')}, type={bc.get('type')}, value={bc.get('value')}")
		
		result = simulation_manager.parse_simulation_prompt(request.prompt, request.context)

		if not result.get("success"):
			error_msg = result.get("error", "Unknown error occurred during parsing")
			logger.error(f"Parse simulation failed: {error_msg}")
			# Return error response instead of raising exception to prevent UI hang
			return SimulationResponse(
				success=False,
				action="error",
				message=f"Parse error: {error_msg}",
				simulation_config=None,
				context=request.context or {},
				guidance=None
			)

		parse_result = result.get("result", {})
		if not parse_result:
			logger.error("Parse result is empty")
			return SimulationResponse(
				success=False,
				action="error",
				message="Empty parse result",
				simulation_config=None,
				context=request.context or {},
				guidance=None
			)

		# Update config if simulation is ready
		if parse_result.get("action") == "simulation_ready" and parse_result.get("simulation_config"):
			sim_config = parse_result["simulation_config"]

			# Update geometry
			if "geometry_type" in sim_config and "geometry_dimensions" in sim_config:
				config_manager.update_geometry(
					sim_config["geometry_type"],
					sim_config["geometry_dimensions"]
				)

			# Update material
			if "material_type" in sim_config and "material_properties" in sim_config:
				config_manager.update_material(
					sim_config["material_type"],
					sim_config["material_properties"]
				)

			# Update physics
			if "physics_type" in sim_config:
				boundary_conditions = sim_config.get("boundary_conditions", [])
				external_loads = sim_config.get("external_loads", [])
				config_manager.update_physics(
					sim_config["physics_type"],
					boundary_conditions,
					external_loads
				)

		# Heuristic override: ensure physics_type matches BC semantics (avoids stale carryover)
		try:
			def _infer_physics_type(sim_cfg: dict, ctx: dict) -> str:
				bc_list = []
				if sim_cfg:
					bc_list = (sim_cfg.get("pde_config", {}) or {}).get("boundary_conditions", []) or []
				elif ctx:
					bc_list = ctx.get("boundary_conditions", []) or []
				# Normalize names
				heat_keys = {"temperature", "heat flux", "heat_flux", "convection", "robin", "radiation", "insulated"}
				solid_keys = {"displacement", "traction", "pressure", "fixed", "roller", "free"}
				for bc in bc_list:
					btype = (bc.get("type") or bc.get("bc_type") or "").strip().lower()
					if btype in solid_keys:
						return "solid_mechanics"
					if btype in heat_keys:
						return "heat_transfer"
				# Fall back to explicit markers
				if ctx and ctx.get("physics_type"):
					return ctx["physics_type"]
				if sim_cfg:
					return (sim_cfg.get("pde_config", {}) or {}).get("physics_type", "") or ""
				return ""

			# Apply override if needed
			sim_cfg = parse_result.get("simulation_config") or {}
			upd_ctx = parse_result.get("updated_context") or {}
			inferred = _infer_physics_type(sim_cfg, upd_ctx)
			if inferred:
				parse_result.setdefault("simulation_config", {}).setdefault("pde_config", {})["physics_type"] = inferred
				parse_result.setdefault("updated_context", {})["physics_type"] = inferred
		except Exception as ex:
			logger.debug(f"Error in physics type inference: {ex}")

		# Ensure boundary_condition_options match the active physics_type
		try:
			if shared_template_manager:
				tm = shared_template_manager
			else:
				from nlp_parser.src.template_manager import TemplateManager
				tm = TemplateManager()
			physics_type = None
			if parse_result.get("simulation_config"):
				physics_type = (
					parse_result["simulation_config"].get("pde_config", {}).get("physics_type") or
					parse_result.get("updated_context", {}).get("physics_type")
				)
			if physics_type:
				bco = tm.get_boundary_condition_options(physics_type)
				ico = tm.get_initial_condition_options(physics_type)
				parse_result.setdefault("simulation_config", {}).setdefault("required_components", {})
				parse_result["simulation_config"]["boundary_condition_options"] = bco
				parse_result["simulation_config"]["initial_condition_options"] = ico
				# Ensure core PDE descriptors (type/equations/variables) match physics template
				tpl = tm.get_template(physics_type) or {}
				pde_tpl = tpl.get("pde_config", {})
				if pde_tpl:
					parse_result["simulation_config"].setdefault("pde_config", {})
					for key in ("pde_type", "equations", "variables"):
						if key in pde_tpl:
							parse_result["simulation_config"]["pde_config"][key] = pde_tpl[key]
				
				# CRITICAL: Ensure boundary conditions from context are in simulation_config
				# Check both updated_context and context for boundary_conditions
				upd_ctx = parse_result.get("updated_context", {})
				ctx = parse_result.get("context", {})
				boundary_conditions = upd_ctx.get("boundary_conditions") or ctx.get("boundary_conditions")
				
				if boundary_conditions:
					logger.info(f"Found boundary_conditions in context: {len(boundary_conditions) if isinstance(boundary_conditions, list) else 'N/A'} BC(s)")
					# Ensure pde_config exists
					if "pde_config" not in parse_result["simulation_config"]:
						parse_result["simulation_config"]["pde_config"] = {}
					# Store boundary conditions in pde_config
					parse_result["simulation_config"]["pde_config"]["boundary_conditions"] = boundary_conditions
					parse_result["simulation_config"]["required_components"]["boundary_conditions"] = boundary_conditions
					# Also ensure they're in updated_context for UI
					parse_result.setdefault("updated_context", {})["boundary_conditions"] = boundary_conditions
					logger.info(f"Stored boundary_conditions in simulation_config.pde_config: {len(boundary_conditions) if isinstance(boundary_conditions, list) else 'N/A'} BC(s)")
				else:
					logger.warning("No boundary_conditions found in context to store in simulation_config")
					# Check if they exist in pde_config from simulation_manager
					pde_bcs = (parse_result.get("simulation_config", {}).get("pde_config", {}) or {}).get("boundary_conditions")
					if pde_bcs:
						logger.info(f"Found boundary_conditions in simulation_config.pde_config: {len(pde_bcs) if isinstance(pde_bcs, list) else 'N/A'} BC(s)")
						parse_result.setdefault("updated_context", {})["boundary_conditions"] = pde_bcs
					else:
						logger.warning("No boundary_conditions found in simulation_config.pde_config either")
		except Exception as ex:
			logger.error(f"Error in template normalization: {ex}", exc_info=True)

		# Always return a response to prevent UI from hanging
		# Get context from either updated_context or context field
		response_context = parse_result.get("updated_context") or parse_result.get("context") or {}
		
		# Log what we have in parse_result for debugging
		logger.debug(f"parse_result keys: {list(parse_result.keys())}")
		logger.debug(f"parse_result.updated_context keys: {list(parse_result.get('updated_context', {}).keys())}")
		logger.debug(f"parse_result.context keys: {list(parse_result.get('context', {}).keys())}")
		
		# Ensure geometry_dimensions are included if they exist in the context
		if not response_context.get("geometry_dimensions"):
			if parse_result.get("updated_context", {}).get("geometry_dimensions"):
				response_context["geometry_dimensions"] = parse_result["updated_context"]["geometry_dimensions"]
			elif parse_result.get("context", {}).get("geometry_dimensions"):
				response_context["geometry_dimensions"] = parse_result["context"]["geometry_dimensions"]
		
		# CRITICAL: Ensure boundary conditions are in response_context even if simulation_config is not ready
		# Check ALL possible sources for boundary_conditions
		bc_in_updated_ctx = parse_result.get("updated_context", {}).get("boundary_conditions")
		bc_in_ctx = parse_result.get("context", {}).get("boundary_conditions")
		bc_in_ctx_final = bc_in_updated_ctx or bc_in_ctx
		
		logger.debug(f"Boundary conditions check: updated_context={bc_in_updated_ctx is not None}, context={bc_in_ctx is not None}, final={bc_in_ctx_final is not None}")
		
		# CRITICAL: Merge user-modified boundary conditions with parser results
		# User-modified BCs take precedence over placeholder BCs
		if user_modified_bcs and bc_in_ctx_final:
			logger.info(f"Merging {len(user_modified_bcs)} user-modified BC(s) with parser results")
			# Create a map of user-modified BCs by location
			user_bc_map = {bc.get('location'): bc for bc in user_modified_bcs if bc.get('location')}
			
			# Merge: use user-modified BCs where they exist, otherwise use parser BCs
			merged_bcs = []
			if isinstance(bc_in_ctx_final, list):
				for bc in bc_in_ctx_final:
					location = bc.get('location')
					if location in user_bc_map:
						# Use user-modified BC instead of parser BC
						merged_bcs.append(user_bc_map[location])
						logger.debug(f"  Using user-modified BC for {location}: type={user_bc_map[location].get('type')}, value={user_bc_map[location].get('value')}")
					else:
						# Use parser BC (only if it's not a placeholder, or if no user BC exists)
						merged_bcs.append(bc)
						logger.debug(f"  Using parser BC for {location}: type={bc.get('type')}, value={bc.get('value')}, source={bc.get('source')}")
				
				# Also add any user-modified BCs that don't exist in parser results
				for location, user_bc in user_bc_map.items():
					if not any(bc.get('location') == location for bc in merged_bcs):
						merged_bcs.append(user_bc)
						logger.debug(f"  Added user-modified BC for {location} (not in parser results)")
				
				bc_in_ctx_final = merged_bcs
				logger.info(f"Merged boundary conditions: {len(merged_bcs)} BC(s) ({len(user_modified_bcs)} user-modified)")
		elif user_modified_bcs:
			# If we have user-modified BCs but no parser BCs, use user-modified BCs
			logger.info(f"Using {len(user_modified_bcs)} user-modified BC(s) (no parser BCs)")
			bc_in_ctx_final = user_modified_bcs
		elif bc_in_ctx_final:
			logger.info(f"Found boundary_conditions in parse_result: {len(bc_in_ctx_final) if isinstance(bc_in_ctx_final, list) else 'N/A'} BC(s)")
			if isinstance(bc_in_ctx_final, list):
				for i, bc in enumerate(bc_in_ctx_final):
					logger.debug(f"  BC {i}: location={bc.get('location')}, type={bc.get('type')}, value={bc.get('value')}, source={bc.get('source')}")
		
		# Always add boundary conditions to response_context if they exist
		if bc_in_ctx_final:
			if not response_context.get("boundary_conditions"):
				response_context["boundary_conditions"] = bc_in_ctx_final
				logger.info(f"Added boundary_conditions to response_context: {len(bc_in_ctx_final) if isinstance(bc_in_ctx_final, list) else 'N/A'} BC(s)")
			else:
				# Merge existing BCs with new BCs (user-modified take precedence)
				existing_bcs = response_context.get("boundary_conditions", [])
				if isinstance(existing_bcs, list) and isinstance(bc_in_ctx_final, list):
					# Create map of existing BCs by location
					existing_bc_map = {bc.get('location'): bc for bc in existing_bcs if bc.get('location')}
					# Update with new BCs (user-modified BCs will overwrite existing ones)
					for bc in bc_in_ctx_final:
						location = bc.get('location')
						if location:
							existing_bc_map[location] = bc
					response_context["boundary_conditions"] = list(existing_bc_map.values())
					logger.debug(f"Merged boundary conditions in response_context: {len(response_context['boundary_conditions'])} BC(s)")
				else:
					response_context["boundary_conditions"] = bc_in_ctx_final
					logger.debug(f"Replaced boundary conditions in response_context: {len(bc_in_ctx_final) if isinstance(bc_in_ctx_final, list) else 'N/A'} BC(s)")
		else:
			logger.warning(f"No boundary_conditions found in parse_result (checked updated_context and context)")
			logger.debug(f"  updated_context keys: {list(parse_result.get('updated_context', {}).keys())}")
			logger.debug(f"  context keys: {list(parse_result.get('context', {}).keys())}")
		
		# Ensure simulation_config includes boundary conditions if they exist in context
		sim_config = parse_result.get("simulation_config")
		if sim_config and bc_in_ctx_final:
			# Ensure pde_config exists in simulation_config
			if "pde_config" not in sim_config:
				sim_config["pde_config"] = {}
			# Store boundary conditions in pde_config if not already there
			if "boundary_conditions" not in sim_config["pde_config"]:
				sim_config["pde_config"]["boundary_conditions"] = bc_in_ctx_final
				logger.info(f"Added boundary_conditions to simulation_config.pde_config: {len(bc_in_ctx_final) if isinstance(bc_in_ctx_final, list) else 'N/A'} BC(s)")
			# Also ensure required_components has boundary_conditions
			if "required_components" not in sim_config:
				sim_config["required_components"] = {}
			if "boundary_conditions" not in sim_config["required_components"]:
				sim_config["required_components"]["boundary_conditions"] = bc_in_ctx_final
		
		# Final verification: Log what's being returned in response_context
		logger.debug(f"Final response_context keys: {list(response_context.keys())}")
		if response_context.get("boundary_conditions"):
			logger.info(f"Final response_context contains boundary_conditions: {len(response_context['boundary_conditions']) if isinstance(response_context['boundary_conditions'], list) else 'N/A'} BC(s)")
		else:
			logger.warning(f"Final response_context does NOT contain boundary_conditions!")
		
		return SimulationResponse(
			success=True,
			action=parse_result.get("action", "unknown"),
			message=parse_result.get("message"),
			simulation_config=sim_config,
			context=response_context,
			guidance=parse_result.get("guidance")
		)
		
	except Exception as e:
		logger.error(f"Exception in parse_simulation endpoint: {e}", exc_info=True)
		# Always return a response to prevent UI from hanging
		return SimulationResponse(
			success=False,
			action="error",
			message=f"Exception during parsing: {str(e)}",
			simulation_config=None,
			context=request.context or {},
			guidance=None
		)

@app.post("/simulation/solve", response_model=SimulationResponse)
async def solve_simulation(request: SimulationRequest):
	"""Run complete simulation workflow"""
	try:
		result = simulation_manager.run_complete_simulation(
			request.prompt, 
			request.context, 
			request.mesh_data
		)

		if not result.get("success"):
			error_msg = result.get("error", "Unknown error occurred during simulation")
			logger.error(f"Solve simulation failed: {error_msg}")
			# Return error response instead of raising exception to prevent UI hang
			return SimulationResponse(
				success=False,
				action="error",
				message=f"Solve error: {error_msg}",
				simulation_config=result.get("simulation_config"),
				context=request.context or {},
				guidance=None,
				mesh_visualization_url=result.get("mesh_visualization_url"),
				field_visualization_url=result.get("field_visualization_url"),
				solution_data=result.get("solution_data")
			)

		# Normalize boundary/initial condition options in response to match physics_type
		try:
			if shared_template_manager:
				tm = shared_template_manager
			else:
				from nlp_parser.src.template_manager import TemplateManager
				tm = TemplateManager()
			physics_type = None
			# Handle nested simulation_config structure: {"simulation_config": {"simulation_config": {"pde_config": {...}}}}
			sim_config = result.get("simulation_config", {})
			if sim_config:
				# Check if it's nested: {"simulation_config": {"pde_config": {...}}}
				if "simulation_config" in sim_config:
					pde_config = sim_config["simulation_config"].get("pde_config", {})
				else:
					pde_config = sim_config.get("pde_config", {})
				physics_type = pde_config.get("physics_type")
			
			if physics_type:
				bco = tm.get_boundary_condition_options(physics_type)
				ico = tm.get_initial_condition_options(physics_type)
				# Update the nested structure correctly
				if "simulation_config" in sim_config:
					sim_config["simulation_config"]["boundary_condition_options"] = bco
					sim_config["simulation_config"]["initial_condition_options"] = ico
					# Ensure PDE descriptors align with physics template
					tpl = tm.get_template(physics_type) or {}
					pde_tpl = tpl.get("pde_config", {})
					if pde_tpl:
						for key in ("pde_type", "equations", "variables"):
							if key in pde_tpl:
								sim_config["simulation_config"]["pde_config"][key] = pde_tpl[key]
				else:
					sim_config["boundary_condition_options"] = bco
					sim_config["initial_condition_options"] = ico
					# Ensure PDE descriptors align with physics template
					tpl = tm.get_template(physics_type) or {}
					pde_tpl = tpl.get("pde_config", {})
					if pde_tpl:
						for key in ("pde_type", "equations", "variables"):
							if key in pde_tpl:
								sim_config["pde_config"][key] = pde_tpl[key]
		except Exception as e:
			logger.debug(f"Error normalizing template options: {e}")

		# Ensure all required fields are present for SimulationResponse
		# Handle potential Pydantic validation errors
		try:
			response = SimulationResponse(
				success=result.get("success", True),
				action=result.get("action", "pde_solved"),
				message=result.get("message"),
				simulation_config=result.get("simulation_config"),
				context=result.get("context"),
				guidance=result.get("guidance"),
				mesh_visualization_url=result.get("mesh_visualization_url"),
				field_visualization_url=result.get("field_visualization_url"),
				solution_data=result.get("solution_data")
			)
			return response
		except Exception as validation_error:
			logger.error(f"Error creating SimulationResponse: {validation_error}", exc_info=True)
			# Return a valid response even if validation fails
			return SimulationResponse(
				success=result.get("success", False),
				action=result.get("action", "error"),
				message=result.get("message", "Response validation error"),
				simulation_config=result.get("simulation_config"),
				context=result.get("context", request.context or {}),
				guidance=result.get("guidance"),
				mesh_visualization_url=result.get("mesh_visualization_url"),
				field_visualization_url=result.get("field_visualization_url"),
				solution_data=result.get("solution_data")
			)
		
	except Exception as e:
		logger.error(f"Exception in solve_simulation endpoint: {e}", exc_info=True)
		# Always return a response to prevent UI from hanging
		try:
			return SimulationResponse(
				success=False,
				action="error",
				message=f"Exception during simulation: {str(e)}",
				simulation_config=None,
				context=request.context or {},
				guidance=None,
				mesh_visualization_url=None,
				field_visualization_url=None,
				solution_data=None
			)
		except Exception as response_error:
			logger.error(f"Failed to create error response: {response_error}")
			# Last resort: return JSON directly
			return JSONResponse(
				status_code=500,
				content={
					"success": False,
					"action": "error",
					"message": f"Exception during simulation: {str(e)}",
					"error": str(e)
				}
			)

@app.post("/parse_boundary_condition")
async def parse_boundary_condition(request: dict):
	"""Parse boundary condition change at specific location using master agent"""
	prompt = request.get('prompt', '')
	boundary_condition = request.get('boundary_condition', {})
	context = request.get('context', {})
	
	if not prompt or not boundary_condition:
		raise HTTPException(status_code=400, detail="prompt and boundary_condition are required")
	
	try:
		# Use master agent to parse boundary condition
		result = master_agent.execute_task("parse_boundary_condition", {
			'prompt': prompt,
			'boundary_condition': boundary_condition,
			'context': context
		})
		
		if not result.get("success"):
			return {
				"success": False,
				"error": result.get("error", "Failed to parse boundary condition")
			}
		
		# Update simulation manager context if updated
		if result.get("updated_context"):
			simulation_manager.current_context = result["updated_context"]
		
		return {
			"success": True,
			"boundary_condition": result.get("boundary_condition", boundary_condition)
		}
		
	except Exception as e:
		logger.error(f"Error parsing boundary condition: {e}")
		return {
			"success": False,
			"error": str(e)
		}

@app.post("/update_boundary_condition")
async def update_boundary_condition(request: dict):
	"""Update boundary condition and sync with pde_config using master agent"""
	boundary_condition = request.get('boundary_condition', {})
	context = request.get('context', {})
	
	# Context is always required, but boundary_condition is optional (for dimension updates, etc.)
	if not context:
		raise HTTPException(status_code=400, detail="context is required")
	
	try:
		# Use master agent to update boundary condition
		result = master_agent.execute_task("update_boundary_condition", {
			'boundary_condition': boundary_condition,
			'context': context
		})
		
		if not result.get("success"):
			return {
				"success": False,
				"error": result.get("error", "Failed to update boundary condition")
			}
		
		# Update simulation manager context
		if result.get("context"):
			simulation_manager.current_context = result["context"]
		
		# Update simulation config
		if result.get("updated_pde_config"):
			if "simulation_config" not in simulation_manager.current_simulation_config:
				simulation_manager.current_simulation_config["simulation_config"] = {}
			simulation_manager.current_simulation_config["simulation_config"]["pde_config"] = result["updated_pde_config"]
		
		message = result.get("message", "Boundary condition and pde_config updated successfully")
		logger.info(f"Successfully updated pde_config: {message}")
		
		return {
			"success": True,
			"message": message,
			"updated_pde_config": result.get("updated_pde_config", {})
		}
		
	except Exception as e:
		logger.error(f"Error updating boundary condition: {e}", exc_info=True)
		return {
			"success": False,
			"error": str(e)
		}

# Frontend-specific endpoints
@app.get("/frontend/status")
async def get_frontend_status():
	"""Get frontend system status"""
	return frontend_manager.get_system_status()

@app.post("/frontend/visualization/mesh")
async def create_mesh_visualization(request: dict):
	"""Create mesh visualization"""
	mesh_data = request.get('mesh_data', {})
	if not mesh_data:
		raise HTTPException(status_code=400, detail="mesh_data is required")

	visualization_url = frontend_manager.create_mesh_visualization(mesh_data)
	if not visualization_url:
		raise HTTPException(status_code=500, detail="Failed to create mesh visualization")

	return {
		"success": True,
		"visualization_url": visualization_url,
		"type": "mesh"
	}

@app.post("/frontend/visualization/field")
async def create_field_visualization(request: dict):
	"""Create field visualization"""
	mesh_data = request.get('mesh_data', {})
	field_data = request.get('field_data', {})

	if not mesh_data or not field_data:
		raise HTTPException(status_code=400, detail="mesh_data and field_data are required")

	visualization_url = frontend_manager.create_field_visualization(mesh_data, field_data)
	if not visualization_url:
		raise HTTPException(status_code=500, detail="Failed to create field visualization")

	return {
		"success": True,
		"visualization_url": visualization_url,
		"type": "field"
	}


@app.post("/frontend/report")
async def create_results_report(request: dict):
	"""Create results report"""
	results = request.get('results', {})
	if not results:
		raise HTTPException(status_code=400, detail="results data is required")

	report_url = frontend_manager.create_results_report(results)
	if not report_url:
		raise HTTPException(status_code=500, detail="Failed to create results report")

	return {
		"success": True,
		"report_url": report_url
	}

@app.post("/frontend/cleanup")
async def cleanup_frontend_files(request: dict = None):
	"""Clean up old frontend files"""
	max_age_hours = request.get('max_age_hours', 24) if request else 24
	cleanup_counts = frontend_manager.cleanup_old_files(max_age_hours)

	return {
		"success": True,
		"cleanup_counts": cleanup_counts,
		"total_cleaned": sum(cleanup_counts.values())
	}

@app.post("/clear-context")
async def clear_context():
	"""Clear all simulation context and reset to initial state using master agent"""
	try:
		# Use master agent to clear context
		agent_result = master_agent.execute_task("clear_context", {})
		
		if not agent_result.get("success"):
			logger.warning(f"Agent context clear had issues: {agent_result.get('error')}")
		
		# Also clear simulation manager state
		simulation_manager.current_context = {}
		simulation_manager.current_mesh_data = None
		simulation_manager.current_msh_file = None
		simulation_manager.current_simulation_config = {}
		
		# Clean up old static files (mesh and field visualizations)
		cleanup_counts = frontend_manager.cleanup_old_files(max_age_hours=0)  # Clean all files
		
		# Calculate total cleaned files
		total_cleaned = sum(cleanup_counts.values())
		
		logger.info("Cleared all simulation context and reset to initial state")
		logger.info(f"Cleaned up {total_cleaned} static files: {cleanup_counts}")
		
		return {
			"success": True,
			"message": "All context cleared successfully",
			"agent_cleanup": agent_result.get("message", "Agent context cleared"),
			"cleanup_counts": cleanup_counts,
			"total_cleaned": total_cleaned
		}
		
	except Exception as e:
		logger.error(f"Error clearing context: {e}")
		return {
			"success": False,
			"error": str(e)
		}

# MCP Server endpoints
@app.post("/mcp")
async def mcp_request(request: dict):
	"""Handle MCP protocol requests"""
	try:
		method = request.get("method", "")
		params = request.get("params", {})
		result = mcp_server.handle_request(method, params)
		return {
			"jsonrpc": "2.0",
			"id": request.get("id", 1),
			"result": result
		}
	except Exception as e:
		logger.error(f"Error handling MCP request: {e}")
		return {
			"jsonrpc": "2.0",
			"id": request.get("id", 1),
			"error": {
				"code": -32603,
				"message": str(e)
			}
		}

@app.get("/mcp/tools")
async def list_mcp_tools():
	"""List all available MCP tools"""
	return {"tools": mcp_server.list_tools()}

@app.get("/mcp/resources")
async def list_mcp_resources():
	"""List all available MCP resources"""
	return {"resources": mcp_server.list_resources()}

@app.post("/mcp/tools/call")
async def call_mcp_tool(request: dict):
	"""Call an MCP tool"""
	tool_name = request.get("name")
	arguments = request.get("arguments", {})
	result = mcp_server.call_tool(tool_name, arguments)
	return result

@app.get("/mcp/resources/{uri:path}")
async def get_mcp_resource(uri: str):
	"""Get an MCP resource"""
	result = mcp_server.get_resource(uri)
	return result

# Agent system endpoints
@app.get("/agents/status")
async def get_agent_status():
	"""Get status of all agents"""
	return master_agent.get_agent_status()

@app.post("/agents/reset")
async def reset_agents():
	"""Reset all agents"""
	master_agent.reset_all()
	return {"success": True, "message": "All agents reset"}

# No legacy endpoints needed - clean architecture only

if __name__ == "__main__":
	# Starting Agentic FEM - Clean Architecture v2.2.0
	# Modular Design: Master-Agent System + MCP Integration
	# Access at: http://localhost:8080
	# API docs at: http://localhost:8080/docs
	# MCP endpoint at: http://localhost:8080/mcp

	uvicorn.run(
		"apps.main_app:app",
		host="0.0.0.0",
		port=8080,
		reload=True,
		log_level="info"
	)