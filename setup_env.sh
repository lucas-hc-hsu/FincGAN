#!/bin/bash
# FincGAN Environment Setup Script
# This script creates and configures the complete FincGAN conda environment

set -e  # Exit on error

echo "================================================"
echo "FincGAN Environment Setup"
echo "================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment from YAML file
echo "Step 1: Creating conda environment 'fincgan'..."
conda env create -f environment.yml

echo ""
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fincgan

echo ""
echo "Step 3: Installing PyTorch with CUDA 11.3..."
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

echo ""
echo "Step 4: Installing DGL with CUDA 11.3..."
pip install dgl-cu113==0.9.1 dglgo -f https://data.dgl.ai/wheels/repo.html

echo ""
echo "Step 5: Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import dgl; print(f'DGL version: {dgl.__version__}')"

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate fincgan"
echo ""
echo "To get started with FincGAN, run:"
echo "  ./run_fincgan.sh --help"
echo ""
