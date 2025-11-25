#!/bin/bash
# Agentic FEM Startup Script

echo "Starting Agentic FEM - Finite Element Method Application"
echo "=========================================================="

# Check if mamba is available (preferred over conda)
if command -v mamba &> /dev/null; then
    echo "Using mamba environment: agentic-fem"
    mamba activate agentic-fem
    PYTHON_CMD="python"
elif command -v conda &> /dev/null; then
    echo "Using conda environment: agentic-fem"
    conda activate agentic-fem
    PYTHON_CMD="python"
elif [ -d "venv" ]; then
    echo "Using existing virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="./venv/bin/python"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    PYTHON_CMD="./venv/bin/python"
fi

# Install/update requirements quietly
echo "Checking requirements..."
if $PYTHON_CMD -m pip install -q --disable-pip-version-check -r requirements.txt > /dev/null 2>&1; then
    echo "all requirements available"
else
    echo "requirements installation failed" >&2
fi

# Ensure node modules are available (quiet)
if command -v npm >/dev/null 2>&1; then
    if [ ! -d "node_modules" ]; then
        # Prefer deterministic install when lockfile exists
        if [ -f package-lock.json ]; then
            npm ci --silent >/dev/null 2>&1 || true
        else
            npm install --silent >/dev/null 2>&1 || true
        fi
    fi
fi

# Install/copy VTK.js if needed (quiet)
if [ ! -f "frontend/static/js/vtk.js" ] && [ -d "node_modules/vtk.js" ]; then
    mkdir -p frontend/static/js
    cp node_modules/vtk.js/vtk.js frontend/static/js/vtk.js 2>/dev/null || true
    cp node_modules/vtk.js/vtk.js.map frontend/static/js/vtk.js.map 2>/dev/null || true
fi

# Start the application
echo "   Starting application..."
echo "   Access at: http://localhost:8080"
echo "   API docs at: http://localhost:8080/docs"
echo "   Press Ctrl+C to stop"
echo ""

$PYTHON_CMD -m uvicorn apps.main_app:app --host 0.0.0.0 --port 8080 --reload
