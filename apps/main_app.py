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
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
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

# Add fenics_backend to path
sys.path.insert(0, str(project_root / 'fenics_backend'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
simulation_manager = SimulationManager(mesh_viewer=mesh_viewer)
frontend_manager = FrontendManager(config_manager)

# Create FastAPI application
app = FastAPI(
	title="Agentic FEM - Clean Architecture",
	description="Modular finite element simulation system with central configuration",
	version="2.1.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

# No CORS middleware needed for local development

@app.on_event("startup")
async def startup_event():
	"""Initialize the application"""
	logger.info("Starting Agentic FEM - Clean Architecture v2.1.0")
	logger.info("Features: Modular Design + Central Config + Mesh Viewer + Simulation Manager + Frontend Module")

	# Initialize frontend manager
	frontend_manager.initialize()
	logger.info("Frontend manager initialized successfully")

	logger.info("Application started successfully")

@app.get("/")
async def root():
	"""Serve the main interface"""
	# Use the frontend manager to get the main interface
	html_content = frontend_manager.get_main_interface()
	return HTMLResponse(content=html_content)

# Mount static files
app.mount("/static", StaticFiles(directory=str(project_root / "frontend" / "static")), name="static")

@app.get("/test")
async def test_endpoint():
	"""Test endpoint"""
	return {"message": "Test endpoint works"}

@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	return {
		"status": "healthy",
		"version": "2.1.0",
		"architecture": "modular",
		"components": {
			"config_manager": True,
			"mesh_viewer": mesh_viewer.mesh_generator is not None,
			"simulation_manager": simulation_manager.parser is not None,
			"fenics_solver": simulation_manager.fenics_solver is not None,
			"field_visualizer": simulation_manager.field_visualizer is not None,
			"frontend_manager": frontend_manager.is_initialized
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

# Mesh preview endpoints (geometry only)
@app.post("/mesh/preview")
async def generate_mesh_preview(request: MeshPreviewRequest):
	"""Generate mesh preview using simulation manager (correct architecture)"""
	# Validate geometry
	validation = mesh_viewer.validate_geometry(request.geometry_type, request.dimensions)
	if not validation["valid"]:
		error_msg = validation.get("error", "Invalid geometry parameters")
		print(f"DEBUG: Validation failed for {request.geometry_type} with dimensions {request.dimensions}")
		print(f"DEBUG: Validation result: {validation}")
		raise HTTPException(status_code=400, detail=error_msg)

	# Generate mesh preview using mesh viewer (correct architecture)
	logger.info(f"Generating mesh preview for {request.geometry_type} with dimensions {request.dimensions}")
	try:
		result = mesh_viewer.generate_mesh_preview(request.geometry_type, request.dimensions)
	except Exception as e:
		logger.error(f"Error generating mesh preview: {e}")
		raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")

	if result["success"]:
		# Update config with new geometry
		config_manager.update_geometry(request.geometry_type, request.dimensions)
		
		# Store mesh data in simulation manager for future simulation
		# Note: GMSH model is handled internally by simulation manager
		if "mesh_data" in result:
			simulation_manager.store_mesh_data(result["mesh_data"])
		
		return result
	else:
		logger.error(f"Mesh preview failed: {result.get('error', 'Unknown error')}")
		raise HTTPException(status_code=500, detail=result["error"])

@app.get("/mesh/supported-geometries")
async def get_supported_geometries():
	"""Get supported geometry types"""
	return mesh_viewer.get_supported_geometries()

# Simulation endpoints (full workflow)
@app.post("/simulation/parse", response_model=SimulationResponse)
async def parse_simulation(request: SimulationRequest):
	"""Parse simulation prompt and update configuration"""
	result = simulation_manager.parse_simulation_prompt(request.prompt, request.context)

	if not result["success"]:
		raise HTTPException(status_code=500, detail=result["error"])

	parse_result = result["result"]

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
	except Exception:
		pass

	# Ensure boundary_condition_options match the active physics_type
	try:
		from template_manager import TemplateManager
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
			# If normalized BCs exist in pde_config, mirror them into context so UI renders the same
			pde_bcs = (parse_result.get("simulation_config", {}).get("pde_config", {}) or {}).get("boundary_conditions")
			if pde_bcs:
				parse_result.setdefault("updated_context", {})["boundary_conditions"] = pde_bcs
	except Exception as _ex:
		pass

	return SimulationResponse(
		success=True,
		action=parse_result.get("action", "unknown"),
		message=parse_result.get("message"),
		simulation_config=parse_result.get("simulation_config"),
		context=parse_result.get("updated_context"),
		guidance=parse_result.get("guidance")
	)

@app.post("/simulation/solve", response_model=SimulationResponse)
async def solve_simulation(request: SimulationRequest):
	"""Run complete simulation workflow"""
	result = simulation_manager.run_complete_simulation(
		request.prompt, 
		request.context, 
		request.mesh_data
	)

	if not result["success"]:
		raise HTTPException(status_code=500, detail=result["error"])

	# Normalize boundary/initial condition options in response to match physics_type
	try:
		from template_manager import TemplateManager
		tm = TemplateManager()
		physics_type = None
		if result.get("simulation_config"):
			physics_type = result["simulation_config"].get("pde_config", {}).get("physics_type")
		if physics_type:
			bco = tm.get_boundary_condition_options(physics_type)
			ico = tm.get_initial_condition_options(physics_type)
			result["simulation_config"]["boundary_condition_options"] = bco
			result["simulation_config"]["initial_condition_options"] = ico
			# Ensure PDE descriptors align with physics template
			tpl = tm.get_template(physics_type) or {}
			pde_tpl = tpl.get("pde_config", {})
			if pde_tpl:
				for key in ("pde_type", "equations", "variables"):
					if key in pde_tpl:
						result["simulation_config"]["pde_config"][key] = pde_tpl[key]
	except Exception:
		pass

	return SimulationResponse(**result)

@app.post("/parse_boundary_condition")
async def parse_boundary_condition(request: dict):
	"""Parse boundary condition change at specific location"""
	prompt = request.get('prompt', '')
	boundary_condition = request.get('boundary_condition', {})
	context = request.get('context', {})
	
	if not prompt or not boundary_condition:
		raise HTTPException(status_code=400, detail="prompt and boundary_condition are required")
	
	try:
		# Use the simulation manager's parser to parse the boundary condition change
		result = simulation_manager.parser.parse(prompt, context)
		
		if result.get("success") and result.get("updated_context", {}).get("boundary_conditions"):
			# Find the updated boundary condition that matches our location
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
					"boundary_condition": updated_bc
				}
		
		# If no specific update found, return the original boundary condition
		return {
			"success": True,
			"boundary_condition": boundary_condition
		}
		
	except Exception as e:
		logger.error(f"Error parsing boundary condition: {e}")
		return {
			"success": False,
			"error": str(e)
		}

@app.post("/update_boundary_condition")
async def update_boundary_condition(request: dict):
	"""Update boundary condition and sync with pde_config"""
	boundary_condition = request.get('boundary_condition', {})
	context = request.get('context', {})
	
	if not boundary_condition or not context:
		raise HTTPException(status_code=400, detail="boundary_condition and context are required")
	
	try:
		# Update the simulation manager's current context
		simulation_manager.current_context = context
		
		# Update the pde_config based on the new context
		update_success = simulation_manager.update_pde_config_from_context(context)
		
		if update_success:
			logger.info("Successfully updated pde_config from boundary condition change")
			return {
				"success": True,
				"message": "Boundary condition and pde_config updated successfully",
				"updated_pde_config": simulation_manager.get_current_simulation_config().get("simulation_config", {}).get("pde_config", {})
			}
		else:
			return {
				"success": False,
				"error": "Failed to update pde_config"
			}
		
	except Exception as e:
		logger.error(f"Error updating boundary condition: {e}")
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
	"""Clear all simulation context and reset to initial state"""
	try:
		# Use comprehensive cleanup method
		simulation_cleanup = simulation_manager.clear_all_context()
		
		# Clean up old static files (mesh and field visualizations)
		cleanup_counts = frontend_manager.cleanup_old_files(max_age_hours=0)  # Clean all files
		
		logger.info("Cleared all simulation context and reset to initial state")
		logger.info(f"Simulation cleanup: {simulation_cleanup}")
		logger.info(f"Cleaned up {cleanup_counts['total_cleaned']} static files: {cleanup_counts}")
		
		return {
			"success": True,
			"message": "All context cleared successfully",
			"simulation_cleanup": simulation_cleanup,
			"cleanup_counts": cleanup_counts
		}
		
	except Exception as e:
		logger.error(f"Error clearing context: {e}")
		return {
			"success": False,
			"error": str(e)
		}

# No legacy endpoints needed - clean architecture only

if __name__ == "__main__":
	# Starting Agentic FEM - Clean Architecture v2.1.0
	# Modular Design: mesh_viewer + simulation_manager + config_manager
	# Access at: http://localhost:8080
	# API docs at: http://localhost:8080/docs

	uvicorn.run(
		"apps.main_app:app",
		host="0.0.0.0",
		port=8080,
		reload=True,
		log_level="info"
	)