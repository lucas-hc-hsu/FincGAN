#!/bin/bash
###############################################################################
# FincGAN GPU Environment Setup Script
#
# This script sets up a conda environment with GPU support for FincGAN.
# Tested on NVIDIA RTX 5090 with CUDA 13.0 (driver compatible with CUDA 12.4)
#
# Requirements:
#   - NVIDIA GPU with CUDA support
#   - NVIDIA driver 550+ (for CUDA 12.4 support)
#   - Miniconda or Anaconda installed
#
# Usage:
#   chmod +x setup_env_gpu.sh
#   ./setup_env_gpu.sh
###############################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

echo "=========================================="
echo "FincGAN GPU Environment Setup"
echo "=========================================="
echo ""

# Check if NVIDIA GPU is available
print_info "Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. NVIDIA drivers may not be installed."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
print_success "NVIDIA GPU detected"
echo ""

# Check if conda is installed
print_info "Checking for conda installation..."
if ! command -v conda &> /dev/null; then
    print_error "conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi
print_success "Conda found: $(conda --version)"
echo ""

# Create conda environment
print_info "Creating conda environment from environment-gpu.yml..."
if conda env list | grep -q "fincgan-gpu"; then
    print_info "Environment 'fincgan-gpu' already exists. Removing..."
    conda env remove -n fincgan-gpu -y
fi

conda create -n fincgan-gpu python=3.9 --override-channels -c conda-forge -y
print_success "Conda environment created"
echo ""

# Activate environment
print_info "Activating fincgan-gpu environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fincgan-gpu
print_success "Environment activated"
echo ""

# Install basic dependencies
print_info "Installing basic dependencies..."
pip install numpy matplotlib pandas scikit-learn tqdm pytz
print_success "Basic dependencies installed"
echo ""

# Install PyTorch with CUDA 12.4
print_info "Installing PyTorch 2.6.0 with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
print_success "PyTorch installed"
echo ""

# Install DGL
print_info "Installing DGL 2.5.0 with CUDA 12.4 support..."
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html
print_success "DGL installed"
echo ""

# Verify installation
print_info "Verifying installation..."
python3 << 'EOF'
import torch
import dgl
import sys

print("=" * 50)
print("Installation Verification")
print("=" * 50)
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"DGL version: {dgl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\nGPU Test:")
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print(f"  ✓ GPU computation successful")
    print(f"  Device: {z.device}")
else:
    print("WARNING: CUDA not available. GPU acceleration will not work.")
    sys.exit(1)

print("=" * 50)
EOF

if [ $? -eq 0 ]; then
    print_success "Installation verified successfully"
    echo ""
    echo "=========================================="
    echo "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "To activate the environment:"
    echo "  conda activate fincgan-gpu"
    echo ""
    echo "To test FincGAN with GPU:"
    echo "  python3 train.py --gpu_id 0 --n_epoch 5 --seed 10 11"
    echo ""
else
    print_error "Installation verification failed"
    exit 1
fi
