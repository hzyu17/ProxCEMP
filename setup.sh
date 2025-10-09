#!/bin/bash

set -e

echo "=========================================="
echo "Motion Planning Setup (Python 3.11)"
echo "=========================================="
echo ""

# Check for Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "Error: python3.11 not found"
    echo "Please install Python 3.11 first"
    exit 1
fi

echo "Found: $(python3.11 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv

# Activate
source venv/bin/activate

# Install packages
echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install mujoco numpy matplotlib glfw imageio imageio-ffmpeg tqdm pillow imgui

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Activate with: source venv/bin/activate"
echo "Run planner: python motion_planner.py"
echo ""