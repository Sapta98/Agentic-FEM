#!/bin/bash
# Agentic FEM - Complete Environment Setup Script

echo "🔧 Agentic FEM - Environment Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}🔧${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_info "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_status "Python $PYTHON_VERSION is compatible"
            return 0
        else
            print_error "Python $PYTHON_VERSION found, but Python 3.8+ is required"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        return 1
    fi
}

# Check if Node.js and npm are installed
check_node() {
    print_info "Checking Node.js and npm..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js $NODE_VERSION found"
    else
        print_error "Node.js not found. Please install Node.js from https://nodejs.org/"
        return 1
    fi
    
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        print_status "npm $NPM_VERSION found"
    else
        print_error "npm not found. Please install npm"
        return 1
    fi
    
    return 0
}

# Create or activate mamba/conda environment
setup_environment() {
    if command -v mamba &> /dev/null; then
        print_info "Using mamba environment: agentic-fem"
        if mamba env list | grep -q "agentic-fem"; then
            print_status "mamba environment 'agentic-fem' already exists"
        else
            print_info "Creating mamba environment: agentic-fem"
            mamba create -n agentic-fem python=3.9 -y
            print_status "mamba environment 'agentic-fem' created"
        fi
        mamba activate agentic-fem
        print_status "mamba environment activated"
    elif command -v conda &> /dev/null; then
        print_info "Using conda environment: agentic-fem"
        if conda env list | grep -q "agentic-fem"; then
            print_status "conda environment 'agentic-fem' already exists"
        else
            print_info "Creating conda environment: agentic-fem"
            conda create -n agentic-fem python=3.9 -y
            print_status "conda environment 'agentic-fem' created"
        fi
        conda activate agentic-fem
        print_status "conda environment activated"
    else
        print_info "Creating Python virtual environment..."
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            print_status "Virtual environment created"
        else
            print_status "Virtual environment already exists"
        fi
        source venv/bin/activate
        print_status "Virtual environment activated"
    fi
}

# Install Python requirements
install_python_requirements() {
    print_info "Installing Python requirements..."
    if [ -f "requirements.txt" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
        print_status "Python requirements installed"
    else
        print_error "requirements.txt not found"
        return 1
    fi
}

# Install Node.js requirements
install_node_requirements() {
    print_info "Installing Node.js requirements (VTK.js)..."
    if [ -f "package.json" ]; then
        npm install
        print_status "Node.js requirements installed"
    else
        print_error "package.json not found"
        return 1
    fi
}

# Check VTK.js installation
check_vtk_js() {
    print_info "Checking VTK.js installation..."
    if [ -f "frontend/static/js/vtk.js" ]; then
        VTK_SIZE=$(du -h frontend/static/js/vtk.js | cut -f1)
        print_status "VTK.js found ($VTK_SIZE)"
    else
        print_warning "VTK.js not found in expected location"
        return 1
    fi
}

# Main setup function
main() {
    echo ""
    
    # Check prerequisites
    if ! check_python; then
        exit 1
    fi
    
    if ! check_node; then
        exit 1
    fi
    
    echo ""
    print_info "All prerequisites satisfied. Starting installation..."
    echo ""
    
    # Create and activate environment
    setup_environment
    
    # Install requirements
    if ! install_python_requirements; then
        print_error "Failed to install Python requirements"
        exit 1
    fi
    
    if ! install_node_requirements; then
        print_error "Failed to install Node.js requirements"
        exit 1
    fi
    
    # Check VTK.js
    check_vtk_js
    
    echo ""
    print_status "Environment setup completed successfully!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Run: ./start.sh"
    echo "   2. Open: http://localhost:8080"
    echo "   3. Read: README.md for usage instructions"
    echo ""
    echo "📖 Documentation:"
    echo "   - README.md: Main documentation"
    echo "   - STARTUP.md: Quick start guide"
    echo "   - COMPONENTS.md: Architecture overview"
    echo ""
}

# Run main function
main "$@"
