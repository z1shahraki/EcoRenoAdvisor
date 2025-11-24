#!/bin/bash

# Setup script using uv for EcoRenoAdvisor

set -e

echo "Setting up EcoRenoAdvisor with uv..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or: pip install uv"
    exit 1
fi

echo "uv found: $(uv --version)"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python $python_version found"
echo ""

# Create virtual environment with uv
echo "Creating virtual environment with uv..."
uv venv
echo "Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo ""

# Install dependencies with uv
echo "Installing dependencies with uv..."
uv pip install -r requirements.txt
echo "Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo ".env file created (please edit if needed)"
    echo ""
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/clean models
echo "Directories created"
echo ""

# Check Docker
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    echo "Docker found"
    echo "  To start Qdrant, run: docker-compose up -d"
else
    echo "WARNING: Docker not found. Please install Docker to run Qdrant."
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Start Qdrant: docker-compose up -d"
echo "3. Download LLM model to models/ directory"
echo "4. Start LLM server: python -m llama_cpp.server --model models/your-model.gguf --host 0.0.0.0 --port 8000"
echo "5. Test basic components: python tests/test_basic_components.py"
echo "6. Run data ingestion: python ingestion/clean_materials.py"
echo "7. Launch UI: python ui/app.py"

