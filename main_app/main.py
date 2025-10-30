"""
Main application entry point for the Agentic FEM system
Integrated version with NLP parser, LaTeX rendering, and enhanced UI
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
# CORS middleware removed - not needed for local application
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add nlp_parser to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'nlp_parser' / 'src'))

try:
	from prompt_analyzer import SimulationPromptParser
except ImportError as e:
	logging.error(f"Error importing NLP parser: {e}")
	logging.error("Make sure the nlp_parser module is properly set up")
	sys.exit(1)

# Import FEniCS backend
try:
	import sys
	sys.path.insert(0, str(Path(__file__).parent.parent / 'fenics_backend'))
	from local_fenics_solver import FEniCSSolver, FENICS_AVAILABLE
	from local_field_visualizer import FieldVisualizer
	fenics_solver = FEniCSSolver()
	field_visualizer = FieldVisualizer()
	logging.info("FEniCS backend available")
except Exception as e:
	logging.warning(f"FEniCS backend not available: {e}")
	fenics_solver = None
	field_visualizer = None
	FENICS_AVAILABLE = False

from config.logging_config import configure_logging, get_logger

# Setup logging once for the app
configure_logging()
logger = get_logger(__name__)

# Request/Response models
class SimulationRequest(BaseModel):
	prompt: str
	context: Optional[Dict[str, Any]] = None
	mesh_data: Optional[Dict[str, Any]] = None

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

# Global parser instance
parser: Optional[SimulationPromptParser] = None

# Create FastAPI application
app = FastAPI(
	title="Agentic FEM Main Application",
	description="Main orchestrator for agentic finite element simulations with LaTeX rendering",
	version="2.0.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

# Mount static files for 3D visualizations
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# CORS middleware removed - not needed for local application

@app.on_event("startup")
async def startup_event():
	"""Initialize the application on startup"""
	global parser

	try:
		logger.info("Starting Agentic FEM Main Application v2.0.0...")
		logger.info("Features: NLP Parser + LaTeX Rendering + Enhanced UI")

		# Initialize the NLP parser
		# Model will be determined from OPENAI_MODEL environment variable or default to gpt-4
		parser = SimulationPromptParser()

		logger.info("Main application started successfully")
		logger.info("LaTeX rendering enabled via MathJax")
		logger.info("Enhanced simulation configuration display active")

	except Exception as e:
		logger.error(f"Failed to start application: {str(e)}")
		sys.exit(1)

@app.get("/")
async def root():
	"""Serve the terminal-style frontend HTML page"""
	return FileResponse('main_app/terminal_frontend.html')

@app.get("/api/")
async def api_root():
	"""API root endpoint"""
	return {
		"message": "Agentic FEM Main Application v2.0.0",
		"version": "2.0.0",
		"status": "running",
		"features": [
			"Natural Language Physics Parsing",
			"LaTeX Equation Rendering",
			"Enhanced Simulation Display",
			"External Load Detection",
			"Context-Based Parsing"
		],
		"components": {
			"nlp_parser": parser is not None,
			"latex_rendering": True,
			"enhanced_ui": True,
			"fenics_solver": FENICS_AVAILABLE,
			"paraview_web": PARAVIEW_AVAILABLE
		}
	}

@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	if not parser:
		raise HTTPException(status_code=503, detail="NLP parser not available")

	return {
		"status": "healthy",
		"version": "2.0.0",
		"components": {
			"nlp_parser": True,
			"latex_rendering": True,
			"enhanced_ui": True,
			"fenics_solver": FENICS_AVAILABLE,
			"paraview_web": PARAVIEW_AVAILABLE
		},
		"features": {
			"physics_detection": "Working",
			"material_properties": "Auto-fetching",
			"external_loads": "Detected",
			"latex_equations": "Rendered",
			"context_management": "Smart",
			"fenics_simulation": "Available" if FENICS_AVAILABLE else "Not Available",
			"mesh_visualization": "Available" if PARAVIEW_AVAILABLE else "Not Available"
		}
	}

@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
	"""Execute a simulation workflow using the NLP parser with enhanced display"""
	if not parser:
		raise HTTPException(status_code=503, detail="NLP parser not available")

	try:
		logger.info(f"Processing simulation request: {request.prompt}")

		# Parse the prompt using the NLP parser
		result = parser.parse(request.prompt, request.context)

		# Mesh visualization should be handled by your existing sophisticated meshing system
		mesh_visualization_url = None

		# Convert to response format
		response = SimulationResponse(
			success=True,
			action=result.get("action", "unknown"),
			message=result.get("message", ""),
			simulation_config=result.get("simulation_config"),
			context=result.get("context"),
			guidance=result.get("guidance"),
			mesh_visualization_url=mesh_visualization_url,
			field_visualization_url=result.get("field_visualization_url"),
			solution_data=result.get("solution_data")
		)

		return response

	except Exception as e:
		logger.error(f"Simulation error: {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve-pde")
async def solve_pde(request: dict):
	"""Solve PDE using FEniCS backend"""
	if not FENICS_AVAILABLE or not fenics_solver:
		raise HTTPException(status_code=503, detail="FEniCS solver not available")

	try:
		# Extract simulation parameters
		physics_type = request.get('physics_type', 'heat_transfer')
		geometry = request.get('geometry', {})
		material_properties = request.get('material_properties', {})
		boundary_conditions = request.get('boundary_conditions', [])

		# Solve the PDE
		result = fenics_solver.solve_simulation(
			physics_type=physics_type,
			geometry=geometry,
			material_properties=material_properties,
			boundary_conditions=boundary_conditions
		)

		return {
			"success": True,
			"solution_data": result.get('solution_data', {}),
			"mesh_data": result.get('mesh_data', {}),
			"field_visualization_url": result.get('field_visualization_url'),
			"message": "PDE solved successfully"
		}

	except Exception as e:
		logger.error(f"PDE solving error: {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-mesh-preview")
async def generate_mesh_preview():
	"""Generate a preview of the current mesh configuration"""
	# This endpoint should integrate with your existing sophisticated meshing system
	# For now, return a simple response indicating the endpoint is available
	return {
		"success": True,
		"message": "Mesh preview endpoint available - integrate with existing meshing system",
		"mesh_url": None
	}

# Mesh generation functions removed - use existing sophisticated meshing system
# The generate_and_serve_3d_mesh function should integrate with your existing mesh generation

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)