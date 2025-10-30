#!/usr/bin/env python3
"""
Environment Setup and Requirements Check for Agentic FEM
Checks and installs all necessary dependencies for the application.
"""

import subprocess
import sys
import os
import platform
import argparse
import logging
from pathlib import Path
from config.logging_config import configure_logging

def run_command(command, description=""):
    """Run a command and return success status"""
    try:
        logging.info(description)
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"{description} - success")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} - failed")
        logging.error(e.stderr.strip() if e.stderr else str(e))
        return False

def check_python_version():
    """Check if Python version is compatible"""
    logging.info("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        logging.info(f"Python {version.major}.{version.minor}.{version.micro} - compatible")
        return True
    else:
        logging.error(f"Python {version.major}.{version.minor}.{version.micro} - requires Python 3.8+")
        return False

def check_node_npm():
    """Check if Node.js and npm are installed"""
    logging.info("Checking Node.js and npm...")

    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Node.js {result.stdout.strip()} - installed")
            node_ok = True
        else:
            node_ok = False
    except FileNotFoundError:
        logging.warning("Node.js not found - please install Node.js")
        node_ok = False

    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"npm {result.stdout.strip()} - installed")
            npm_ok = True
        else:
            npm_ok = False
    except FileNotFoundError:
        logging.warning("npm not found - please install npm")
        npm_ok = False

    return node_ok and npm_ok

def install_python_requirements():
    """Install Python requirements"""
    logging.info("Installing Python requirements...")

    if not os.path.exists("requirements.txt"):
        logging.error("requirements.txt not found")
        return False

    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

    return success

def install_node_requirements():
    """Install Node.js requirements (VTK.js)"""
    logging.info("Installing Node.js requirements...")

    if not os.path.exists("package.json"):
        logging.error("package.json not found")
        return False

    success = run_command(
        "npm install",
        "Installing Node.js packages (VTK.js)"
    )

    return success

def check_vtk_js():
    """Check if VTK.js is properly installed"""
    logging.info("Checking VTK.js installation...")

    vtk_js_path = Path("frontend/static/js/vtk.js")
    if vtk_js_path.exists():
        size_mb = vtk_js_path.stat().st_size / (1024 * 1024)
        logging.info(f"VTK.js found ({size_mb:.1f} MB)")
        return True
    else:
        logging.warning("VTK.js not found in frontend/static/js/")
        return False

def create_startup_script():
    """Create startup script for easy app launching"""
    logging.info("Creating startup script...")

	startup_script = """#!/bin/bash
	# Agentic FEM Startup Script

	echo "🚀 Starting Agentic FEM - Finite Element Method Application"
	echo "=========================================================="

	# Check if virtual environment exists
	if [ ! -d "venv" ]; then
	echo "📦 Creating virtual environment..."
	python3 -m venv venv
	fi

	# Activate virtual environment
	echo "🔧 Activating virtual environment..."
	source venv/bin/activate

	# Install/update requirements
    echo "Installing/updating requirements..."
    python3 -m pip install -r requirements.txt

	# Install VTK.js if needed
	if [ ! -f "frontend/static/js/vtk.js" ]; then
    echo "Installing VTK.js..."
	npm install
	fi

	# Start the application
    echo "Starting application..."
	echo "   Access at: http://localhost:8080"
	echo "   API docs at: http://localhost:8080/docs"
	echo "   Press Ctrl+C to stop"
	echo ""

    python3 -m uvicorn apps.main_app:app --host 0.0.0.0 --port 8080 --reload
	"""

    with open("start.sh", "w") as f:
		f.write(startup_script)

		# Make executable
        os.chmod("start.sh", 0o755)
        logging.info("Startup script created: ./start.sh")

def main(argv=None):
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Agentic FEM environment setup")
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    args = parser.parse_args(argv)

    configure_logging("DEBUG" if args.verbose else "INFO")
    logging.info("Agentic FEM - Environment Setup")

	# Check system requirements
    python_ok = check_python_version()
    node_ok = check_node_npm()

    if not python_ok:
        logging.error("Setup failed: Python 3.8+ required")
        return False

    if not node_ok:
        logging.error("Setup failed: Node.js and npm required (install from https://nodejs.org/)")
        return False

    logging.info("Installing dependencies...")
    python_success = install_python_requirements()
    node_success = install_node_requirements()

    if not python_success:
        logging.error("Setup failed: Python requirements installation failed")
        return False

    if not node_success:
        logging.error("Setup failed: Node.js requirements installation failed")
        return False

    vtk_ok = check_vtk_js()
    if not vtk_ok:
        logging.warning("VTK.js not found, but Node.js packages were installed")

    create_startup_script()

    logging.info("Setup completed successfully")
    logging.info("Next steps: run ./start.sh and open http://localhost:8080")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
