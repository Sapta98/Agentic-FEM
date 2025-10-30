# Quick Start Guide

## 🚀 How to Run the App

### Prerequisites
- Python 3.8+
- Node.js (installed via environment.yml or system) for VTK.js tooling

### Option 1: Mamba Setup (Recommended)

```bash
# 1. Create mamba environment
mamba env create -f environment.yml

# 2. Activate environment
mamba activate agentic-fem

# 3. Start the application (start.sh will install node modules if missing)
./start.sh
```

### Option 2: Automated Setup

```bash
# 1. Run the complete setup script
./setup.sh

# 2. Start the application
./start.sh
```

### Option 3: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the application (start.sh will install node modules if missing)
python -m uvicorn apps.main_app:app --host 0.0.0.0 --port 8080 --reload
```

## 🌐 Access the Application

- **Main Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs

## 📖 Documentation

- **README.md**: Complete documentation and features
- **STARTUP.md**: Detailed startup guide
- **COMPONENTS.md**: Architecture and component overview
- **QUICK_START.md**: This quick start guide

## 🔧 What the Setup Scripts Do

### `setup.sh`
- Checks Python 3.8+ installation
- Checks Node.js and npm installation
- Creates Python virtual environment
- Installs Python requirements
- Installs VTK.js via npm
- Verifies VTK.js installation
- Provides colored output and error handling

### `start.sh`
- Activates virtual environment
- Updates Python requirements
- Installs VTK.js if missing
- Starts the FastAPI application with auto-reload

### `setup_environment.py`
- Python-based setup script with detailed checks
- Cross-platform compatibility
- Comprehensive error handling
- Creates startup scripts automatically

## 🎯 What You'll See

1. **Terminal-like Interface**: Clean, modern web interface
2. **Mesh Generation**: Create 3D meshes for various geometries
3. **Interactive Visualization**: VTK.js-powered 3D mesh viewer
4. **PDE Solving**: Natural language problem description and solving
5. **Field Visualization**: View solution fields on 3D meshes

## 🆘 Troubleshooting

- **Port 8080 in use**: Change port in startup command
- **VTK.js not loading**: Ensure `start.sh` ran with Node available; it will install and copy VTK.js locally
- **Mesh generation fails**: Check geometry parameters and dimensions
- **Python version issues**: Ensure Python 3.8+ is installed

## 🎉 Ready to Go!

After running the setup, you'll have a fully functional finite element method application with 3D visualization capabilities!
