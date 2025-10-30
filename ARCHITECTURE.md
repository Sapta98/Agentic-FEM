# FEM Simulation Application - Architecture & Information Flow

## Overview
This is a Finite Element Method (FEM) simulation application that uses natural language processing to parse physics simulation prompts and generate 3D mesh visualizations with PDE solving capabilities.

## Core Architecture

### 1. **SimulationManager** - Central Data Store
The `SimulationManager` is the **single source of truth** for all simulation data:

```python
class SimulationManager:
    def __init__(self):
        # Central storage for simulation data
        self.current_context = {}           # Parsed context (boundary conditions, materials, etc.)
        self.current_mesh_data = None      # Generated mesh data
        self.current_simulation_config = {} # Complete simulation configuration
```

**Responsibilities:**
- Store all parsed context (boundary conditions, material properties, geometry)
- Store generated mesh data
- Store simulation configuration
- Coordinate between parsing, mesh generation, and PDE solving

### 2. **NLP Parser** - Natural Language Processing
Located in `nlp_parser/src/`:

**Components:**
- `ContextBasedParser` - Main parser that extracts simulation parameters
- `PromptManager` - Handles OpenAI API calls for parsing
- `TemplateManager` - Manages physics templates and configurations

**Parses:**
- Physics type (heat_transfer, solid_mechanics, etc.)
- Material properties (copper, steel, etc.)
- Geometry (line, beam, cylinder, etc.)
- Dimensions (length, width, height, etc.)
- Boundary conditions (temperature, fixed, free, etc.)
- External loads and initial conditions

### 3. **Mesh Generation System**
Located in `mesh/`:

**Components:**
- `MeshGenerator` - Main mesh generation coordinator
- `GMSHGenerator` - Interfaces with GMSH for mesh creation
- `MeshVisualizer` - Creates 3D VTK.js visualizations

**Supports:**
- 1D elements (lines, rods)
- 2D elements (triangles, quads)
- 3D elements (tetrahedrons, hexahedrons)
- First-order and second-order elements

### 4. **FEniCS Backend** - PDE Solving
Located in `fenics_backend/`:

**Components:**
- `FEniCSSolver` - Solves PDEs using FEniCS
- `FieldVisualizer` - Creates field visualizations

**Capabilities:**
- Heat transfer simulations
- Solid mechanics simulations
- Mixed physics problems

### 5. **Frontend System**
Located in `frontend/`:

**Components:**
- `FrontendManager` - Main frontend coordinator
- `TerminalInterface` - Web-based terminal interface
- `MeshVisualizer` - 3D mesh visualization with VTK.js
- `FieldVisualizer` - Field solution visualization

## Information Flow

### Phase 1: Prompt Parsing
```
User Input → NLP Parser → SimulationManager
     ↓
1. User enters natural language prompt
2. NLP Parser extracts simulation parameters
3. SimulationManager stores parsed context
```

**Example Flow:**
```
Input: "Heat transfer in copper line, 1m long, 100°C at one end, 10°C at other end"

NLP Parser extracts:
- physics_type: "heat_transfer"
- material_type: "copper" 
- geometry_type: "line"
- dimensions: {"length": 1.0}
- boundary_conditions: [
    {"type": "temperature", "location": "one end", "value": 100},
    {"type": "temperature", "location": "other end", "value": 10}
  ]

SimulationManager stores:
- current_context = {physics_type, material_type, geometry_type, dimensions, boundary_conditions}
- current_simulation_config = {pde_config, required_components, etc.}
```

### Phase 2: Mesh Generation
```
SimulationManager → Mesh Generator → GMSH → Mesh Visualizer
     ↓
1. SimulationManager has complete context
2. Mesh generator creates geometry and mesh
3. GMSH generates mesh data (vertices, cells, elements)
4. Mesh visualizer creates 3D visualization
5. SimulationManager stores mesh data
```

**Data Flow:**
```
Context → MeshGenerator → GMSHGenerator → GMSH → Mesh Data
   ↓
Mesh Data → MeshVisualizer → VTK.js → 3D Visualization
   ↓
Mesh Data → SimulationManager.current_mesh_data
```

### Phase 3: PDE Solving
```
SimulationManager → FEniCS Solver → Field Visualizer
     ↓
1. SimulationManager has context + mesh data
2. FEniCS solver solves PDE using stored data
3. Field visualizer creates solution visualization
```

**Data Flow:**
```
Stored Context + Mesh Data → FEniCS Solver → Solution Data
   ↓
Solution Data → Field Visualizer → Field Visualization
```

## Key Design Principles

### 1. **Central Data Store**
- **SimulationManager** is the single source of truth
- All components read from/write to SimulationManager
- No data passing between frontend and backend

### 2. **No Double Parsing**
- Parse once, store in SimulationManager
- Subsequent operations use stored data
- Context completeness check prevents re-parsing

### 3. **Separation of Concerns**
- **NLP Parser**: Natural language understanding
- **Mesh System**: Geometry and mesh generation
- **FEniCS Backend**: PDE solving
- **Frontend**: User interface and visualization
- **SimulationManager**: Data coordination

### 4. **Progressive Information Gathering**
- Parse prompts incrementally
- Only ask for missing information
- Build complete context over multiple interactions

## API Endpoints

### Core Simulation Endpoints
- `POST /simulation/parse` - Parse natural language prompt
- `POST /simulation/solve` - Solve PDE using stored data
- `POST /mesh/preview` - Generate mesh preview

### Configuration Endpoints
- `GET /config` - Get current configuration
- `POST /config/update` - Update configuration
- `GET /config/geometry` - Get geometry configuration
- `GET /config/physics-types` - Get available physics types

### Frontend Endpoints
- `GET /` - Main application interface
- `POST /frontend/visualization/mesh` - Create mesh visualization
- `POST /frontend/visualization/field` - Create field visualization

## Data Structures

### Context Structure
```python
context = {
    "physics_type": "heat_transfer",
    "material_type": "copper",
    "material_properties": {
        "thermal_conductivity": 400,
        "density": 8960,
        "specific_heat": 385
    },
    "geometry_type": "line",
    "geometry_dimensions": {"length": 1.0},
    "boundary_conditions": [
        {
            "type": "temperature",
            "location": "one end",
            "value": 100,
            "confidence": 0.9
        }
    ],
    "external_loads": [...],
    "initial_conditions": [...]
}
```

### Mesh Data Structure
```python
mesh_data = {
    "vertices": [[x1,y1,z1], [x2,y2,z2], ...],
    "faces": [[v1,v2,v3], [v4,v5,v6], ...],
    "cells": {
        "line": [[v1,v2], [v3,v4], ...],
        "triangle": [[v1,v2,v3], [v4,v5,v6], ...],
        "tetrahedron": [[v1,v2,v3,v4], ...]
    },
    "mesh_dimension": 1,
    "mesh_stats": {
        "vertices": 11,
        "elements": 5,
        "mesh_quality": "medium"
    }
}
```

## Error Handling

### Context Completeness Check
```python
def _is_context_complete(self, context: Dict[str, Any]) -> bool:
    required_fields = [
        'physics_type', 'material_type', 'geometry_type',
        'geometry_dimensions', 'boundary_conditions'
    ]
    # Check all required fields are present and non-empty
```

### Graceful Degradation
- Missing information prompts user for clarification
- Partial context allows progressive simulation setup
- Clear error messages guide user to complete simulation

## Performance Considerations

### Caching
- SimulationManager caches all parsed data
- Mesh data stored for reuse
- No redundant parsing or mesh generation

### Optimization
- Lazy loading of heavy components (FEniCS, GMSH)
- Efficient data structures for large meshes
- Streaming visualization for large datasets

## Future Enhancements

### Planned Features
- Multi-physics simulations
- Advanced mesh refinement
- Real-time parameter updates
- Collaborative simulation sharing
- Export to standard formats (VTU, XDMF)

### Scalability
- Distributed mesh generation
- Cloud-based PDE solving
- GPU acceleration for large problems
- Microservices architecture

## Development Workflow

### Adding New Physics Types
1. Create template in `nlp_parser/templates/`
2. Add material properties in `config/`
3. Update boundary condition templates
4. Test with sample prompts

### Adding New Geometry Types
1. Update `mesh/generators/` with new geometry
2. Add dimension specifications in `config/dimensions.json`
3. Update frontend dimension handling
4. Test mesh generation

### Adding New Visualization Features
1. Update VTK.js visualization in `frontend/templates/`
2. Add new representation modes
3. Update field visualization capabilities
4. Test with various mesh types

---

This architecture ensures a clean separation of concerns while maintaining a central data store for all simulation information. The progressive information gathering approach allows users to build complex simulations incrementally, while the central storage prevents data loss and enables efficient PDE solving.
