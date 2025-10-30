# Component Documentation

## Architecture Overview

The application follows a clean architecture pattern with the following main components:

### Recent Cleanup (v2.1.0)
- **Removed obsolete files**: Mixed analysis templates, duplicate CSS files, unused HTML templates
- **Cleaned up imports**: Consolidated sys.path management, removed duplicate imports
- **Streamlined codebase**: Removed redundant visualization wrappers, obsolete test files
- **Enhanced context management**: Complete GMSH model cleanup and static file management

## Core Components

### 1. Apps (`apps/`)
Main application layer with FastAPI backend and modular design.

**Files:**
- `main_app.py`: FastAPI application entry point
- `mesh_viewer/`: 3D mesh visualization components
- `simulation/`: PDE solving and simulation management

**Key Features:**
- RESTful API endpoints
- Web interface integration
- Modular component architecture

### 2. Mesh Generation (`mesh/`)
Handles mesh generation and processing using GMSH.

**Files:**
- `generators/`: Geometry-specific mesh generators
  - `mesh_1d.py`: 1D mesh generation
  - `mesh_2d.py`: 2D mesh generation  
  - `mesh_3d.py`: 3D mesh generation (cubes, cylinders, spheres)
- `utils/gmsh_generator.py`: GMSH integration utilities

**Key Features:**
- Multiple geometry types support
- Adaptive mesh sizing
- Volume and surface element generation

### 3. Frontend (`frontend/`)
Web interface components and templates.

**Files:**
- `templates/`: HTML templates
  - `mesh_visualization.html`: VTK.js mesh visualization
- `static/`: Static assets
  - `js/vtk.js`: VTK.js library
  - CSS files for styling

**Key Features:**
- VTK.js integration for 3D visualization
- Interactive mesh controls
- Responsive design

### 4. FEniCS Backend (`fenics_backend/`)
PDE solving and field visualization.

**Files:**
- `local_fenics_solver.py`: FEniCS solver implementation
- `local_field_visualizer.py`: Field visualization generator

**Key Features:**
- Heat transfer, solid mechanics
- Mock solver fallback when FEniCS unavailable
- VTK.js field visualization

### 5. NLP Parser (`nlp_parser/`)
Natural language processing for problem description.

**Files:**
- `src/prompt_analyzer.py`: Main NLP processing
- `src/models/`: Data models for simulation parameters
- `templates/`: Problem templates for different physics types (heat transfer, solid mechanics)
- `src/context_based_parser.py`: Context-aware parsing with material properties database

**Key Features:**
- Natural language problem parsing
- Material properties database integration
- Context-based boundary condition mapping
- Physics type detection (heat transfer, solid mechanics)
- Physics type classification
- Parameter extraction

## Data Flow

1. **User Input**: Natural language problem description
2. **NLP Processing**: Parse and extract simulation parameters
3. **Mesh Generation**: Create 3D mesh using GMSH
4. **Mesh Visualization**: Display interactive 3D mesh with VTK.js
5. **PDE Solving**: Solve using FEniCS or mock solver
6. **Field Visualization**: Display solution fields on mesh

## Key Technologies

- **FastAPI**: Web framework and API
- **GMSH**: Mesh generation
- **FEniCS**: PDE solving
- **VTK.js**: 3D visualization
- **Python**: Backend development
- **JavaScript**: Frontend visualization

## Configuration

- **Geometry Types**: cube, cylinder, sphere, beam
- **Physics Types**: heat_transfer, solid_mechanics
- **Materials**: copper, steel, aluminum, etc.
- **Boundary Conditions**: temperature, displacement, force, etc.
