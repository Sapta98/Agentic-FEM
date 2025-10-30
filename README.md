# Agentic FEM - Finite Element Method Application

A modern web-based finite element method application with 3D mesh visualization and PDE solving capabilities.

## Features

- **3D Mesh Generation**: Create meshes for various geometries (cubes, cylinders, spheres, beams)
- **Interactive Visualization**: VTK.js-powered 3D mesh visualization with rotation, panning, and zooming
- **PDE Solving**: Solve heat transfer and solid mechanics problems using FEniCSx
- **Field Visualization**: Visualize solution fields (temperature, displacement, stress) on 3D meshes
- **Natural Language Interface**: Describe your problem in plain English
- **Material Properties Database**: Built-in material properties for common materials
- **Boundary Condition Management**: Interactive boundary condition configuration
- **Complete Context Management**: Clear context functionality with GMSH model cleanup

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js (installed via environment.yml or system) for VTK.js tooling

### Installation & Setup

**Option 1: Mamba Setup (Recommended)**
```bash
# Create mamba environment
mamba env create -f environment.yml

# Activate environment
mamba activate agentic-fem

# Start the application
./start.sh
```

**Option 2: Automated Setup**
```bash
# Run the complete setup script
./setup.sh

# Start the application
./start.sh
```

**Option 3: Manual Setup**
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the application (start.sh will install node modules if needed)
python -m uvicorn apps.main_app:app --host 0.0.0.0 --port 8080 --reload
```

### Access the Application

- **Main interface**: http://localhost:8080
- **API documentation**: http://localhost:8080/docs

## Architecture

### Core Components

- **`apps/`**: Main application with clean architecture
  - `main_app.py`: FastAPI application entry point
  - `mesh_viewer/`: 3D mesh visualization components
  - `simulation/`: PDE solving and simulation management

- **`mesh/`**: Mesh generation and processing
  - `generators/`: Geometry-specific mesh generators (1D, 2D, 3D)
  - `utils/`: GMSH integration and mesh utilities

- **`frontend/`**: Web interface components
  - `templates/`: HTML templates for visualization
  - `static/`: Static assets (CSS, JS, VTK.js)

- **`fenics_backend/`**: FEniCS integration for PDE solving
  - `local_fenics_solver.py`: FEniCS solver implementation
  - `local_field_visualizer.py`: Field visualization generator

- **`nlp_parser/`**: Natural language processing for problem description

## Usage

### Mesh Generation

1. Select geometry type (cube, cylinder, sphere, beam)
2. Set dimensions and parameters
3. Generate mesh preview
4. View interactive 3D mesh with VTK.js

### PDE Solving

1. Describe your problem in natural language
2. The system parses your description and extracts:
   - Physics type (heat transfer, solid mechanics, etc.)
   - Boundary conditions
   - Material properties
   - Geometry parameters
3. Solve the PDE using FEniCS or mock solver
4. Visualize field results on the mesh

### Example Problems

- **Heat Transfer**: "Solve heat transfer in a copper wire, 1m long, diameter 60cm, 100°C at one end, 10°C at other end"
- **Solid Mechanics**: "Analyze stress in a steel beam with fixed ends under distributed load"
- **Mixed Physics**: "Coupled heat transfer and solid mechanics in a composite material"

## API Endpoints

- `GET /`: Main web interface
- `POST /mesh/preview`: Generate mesh preview
- `POST /solve-pde`: Solve PDE problem
- `GET /docs`: API documentation

## Development

The application uses a clean architecture pattern with separated concerns:

- **Presentation Layer**: FastAPI web interface
- **Business Logic**: Simulation management and mesh processing
- **Data Layer**: Mesh generation and PDE solving
- **Integration**: GMSH for meshing, FEniCS for solving, VTK.js for visualization

### Startup behavior and frontend dependencies

- `start.sh` keeps console output minimal and will:
  - Ensure Python requirements are installed quietly
  - If `node_modules` is missing, run a quiet `npm ci` (or `npm install`) and copy VTK.js into `frontend/static/js/`
  - Start Uvicorn on port 8080

### Environment configuration

- Use a `.env` file (see any `*.env.example` in the repo) to configure secrets or runtime settings as needed.

## Notes

- VTK.js is downloaded and copied locally by the startup script; CDN loading is not enabled at this time.
- Set your OpenAI API key in `nlp_parser/src/.env` (e.g., `OPENAI_API_KEY=...`) before running the NLP-driven flows.

## License

MIT License - see LICENSE file for details.
