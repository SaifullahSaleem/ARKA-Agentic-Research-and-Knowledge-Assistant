#!/bin/bash

# Setup script for the Agentic AI Interface project

echo "=========================================="
echo "Setting up Agentic AI Interface"
echo "=========================================="

# Create directories
echo "Creating directories..."
mkdir -p papers
mkdir -p pdfs
mkdir -p monitoring/grafana/dashboards

# Check for .env file
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Please edit .env file with your API keys"
    else
        echo "Warning: .env.example not found"
    fi
else
    echo ".env file already exists"
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Setup monitoring
echo "Setting up monitoring..."
python monitoring_setup.py

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: python collect_papers.py"
echo "3. Run: python preprocessing_notebook.py"
echo "4. Run: streamlit run app.py"
echo ""

