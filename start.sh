#!/bin/bash
# Agentic FEM Startup Script

echo "🚀 Starting Agentic FEM - Finite Element Method Application"
echo "=========================================================="

# Check if mamba is available (preferred over conda)
if command -v mamba &> /dev/null; then
    echo "🐍 Using mamba environment: agentic-fem"
    mamba activate agentic-fem
    PYTHON_CMD="python"
elif command -v conda &> /dev/null; then
    echo "🐍 Using conda environment: agentic-fem"
    conda activate agentic-fem
    PYTHON_CMD="python"
elif [ -d "venv" ]; then
    echo "🔧 Using existing virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="./venv/bin/python"
else
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    PYTHON_CMD="./venv/bin/python"
fi

# Install/update requirements
echo "📚 Installing/updating requirements..."
pip install -r requirements.txt

# Install VTK.js if needed
if [ ! -f "frontend/static/js/vtk.js" ]; then
    echo "🎨 Installing VTK.js..."
    npm install
    
    # Create js directory if it doesn't exist
    mkdir -p frontend/static/js
    
    # Copy VTK.js to static directory
    echo "📦 Copying VTK.js to static directory..."
    cp node_modules/vtk.js/vtk.js frontend/static/js/vtk.js
    cp node_modules/vtk.js/vtk.js.map frontend/static/js/vtk.js.map
    echo "✅ VTK.js installed successfully"
fi

# Start the application
echo "   Starting application..."
echo "   Access at: http://localhost:8080"
echo "   API docs at: http://localhost:8080/docs"
echo "   Press Ctrl+C to stop"
echo ""

$PYTHON_CMD -m uvicorn apps.main_app:app --host 0.0.0.0 --port 8080 --reload
