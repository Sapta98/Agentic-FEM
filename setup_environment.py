#!/usr/bin/env python3
"""
Environment Setup and Requirements Check for Agentic FEM
Checks and installs all necessary dependencies for the application.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description=""):
	"""Run a command and return success status"""
	try:
		print(f"🔧 {description}")
		result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
		print(f"✅ {description} - Success")
		return True
	except subprocess.CalledProcessError as e:
		print(f"❌ {description} - Failed")
		print(f"   Error: {e.stderr}")
		return False

def check_python_version():
	"""Check if Python version is compatible"""
	print("🐍 Checking Python version...")
	version = sys.version_info
	if version.major == 3 and version.minor >= 8:
		print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
		return True
	else:
		print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
		return False

def check_node_npm():
	"""Check if Node.js and npm are installed"""
	print("📦 Checking Node.js and npm...")

	# Check Node.js
	try:
		result = subprocess.run(["node", "--version"], capture_output=True, text=True)
		if result.returncode == 0:
			print(f"✅ Node.js {result.stdout.strip()} - Installed")
			node_ok = True
		else:
			node_ok = False
		except FileNotFoundError:
			print("❌ Node.js not found - Please install Node.js")
			node_ok = False

			# Check npm
			try:
				result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
				if result.returncode == 0:
					print(f"✅ npm {result.stdout.strip()} - Installed")
					npm_ok = True
				else:
					npm_ok = False
				except FileNotFoundError:
					print("❌ npm not found - Please install npm")
					npm_ok = False

					return node_ok and npm_ok

def install_python_requirements():
	"""Install Python requirements"""
	print("📚 Installing Python requirements...")

	# Check if requirements.txt exists
	if not os.path.exists("requirements.txt"):
		print("❌ requirements.txt not found")
		return False

		# Install requirements
		success = run_command(
		f"{sys.executable} -m pip install -r requirements.txt",
		"Installing Python packages"
		)

		return success

def install_node_requirements():
	"""Install Node.js requirements (VTK.js)"""
	print("📦 Installing Node.js requirements...")

	# Check if package.json exists
	if not os.path.exists("package.json"):
		print("❌ package.json not found")
		return False

		# Install npm packages
		success = run_command(
		"npm install",
		"Installing Node.js packages (VTK.js)"
		)

		return success

def check_vtk_js():
	"""Check if VTK.js is properly installed"""
	print("🎨 Checking VTK.js installation...")

	vtk_js_path = Path("frontend/static/js/vtk.js")
	if vtk_js_path.exists():
		size_mb = vtk_js_path.stat().st_size / (1024 * 1024)
		print(f"✅ VTK.js found ({size_mb:.1f} MB)")
		return True
	else:
		print("❌ VTK.js not found in frontend/static/js/")
		return False

def create_startup_script():
	"""Create startup script for easy app launching"""
	print("🚀 Creating startup script...")

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
	echo "📚 Installing/updating requirements..."
	pip install -r requirements.txt

	# Install VTK.js if needed
	if [ ! -f "frontend/static/js/vtk.js" ]; then
	echo "🎨 Installing VTK.js..."
	npm install
	fi

	# Start the application
	echo "🌐 Starting application..."
	echo "   Access at: http://localhost:8080"
	echo "   API docs at: http://localhost:8080/docs"
	echo "   Press Ctrl+C to stop"
	echo ""

	python -m uvicorn apps.main_app:app --host 0.0.0.0 --port 8080 --reload
	"""

	with open("start.sh", "w") as f:
		f.write(startup_script)

		# Make executable
		os.chmod("start.sh", 0o755)
		print("✅ Startup script created: ./start.sh")

def main():
	"""Main setup function"""
	print("🔧 Agentic FEM - Environment Setup")
	print("==================================")

	# Check system requirements
	python_ok = check_python_version()
	node_ok = check_node_npm()

	if not python_ok:
		print("\n❌ Setup failed: Python 3.8+ required")
		return False

		if not node_ok:
			print("\n❌ Setup failed: Node.js and npm required")
			print("   Install from: https://nodejs.org/")
			return False

			# Install requirements
			print("\n📦 Installing dependencies...")
			python_success = install_python_requirements()
			node_success = install_node_requirements()

			if not python_success:
				print("\n❌ Setup failed: Python requirements installation failed")
				return False

				if not node_success:
					print("\n❌ Setup failed: Node.js requirements installation failed")
					return False

					# Check VTK.js
					vtk_ok = check_vtk_js()
					if not vtk_ok:
						print("\n⚠️  Warning: VTK.js not found, but Node.js packages were installed")

						# Create startup script
						create_startup_script()

						print("\n🎉 Setup completed successfully!")
						print("\n📖 Next steps:")
						print("   1. Run: ./start.sh")
						print("   2. Open: http://localhost:8080")
						print("   3. Read: README.md for usage instructions")

						return True

						if __name__ == "__main__":
							success = main()
							sys.exit(0 if success else 1)
