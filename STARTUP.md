# Startup Guide

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js (installed via environment.yml or system) for VTK.js tooling

### 1. Automated Setup (Recommended)

```bash
# Run the complete setup script
./setup.sh

# Start the application (start.sh will install node modules if missing)
./start.sh
```

### 2. Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the application (start.sh will install node modules if missing)
python -m uvicorn apps.main_app:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Access the Application

- **Main Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs

## What You'll See

1. **Terminal-like Interface**: Clean, modern web interface with terminal styling
2. **Mesh Generation**: Create 3D meshes for various geometries
3. **Interactive Visualization**: VTK.js-powered 3D mesh viewer with:
   - Mouse controls (rotate, pan, zoom)
   - Surface and volume view modes
   - Different representation modes (points, wireframe, solid)
4. **PDE Solving**: Natural language problem description and solving
5. **Field Visualization**: View solution fields on 3D meshes

## Example Workflow

1. **Generate Mesh**: Select geometry type and dimensions
2. **Preview Mesh**: View 3D mesh with interactive controls
3. **Solve PDE**: Describe your problem in natural language
4. **View Results**: Visualize field variables on the mesh

## Troubleshooting

- **Port 8080 in use**: Change port in startup command
- **VTK.js not loading**: Ensure `npm install` completed successfully
- **Mesh generation fails**: Check geometry parameters and dimensions

## Development Mode

The application runs with `--reload` flag for automatic code reloading during development.
