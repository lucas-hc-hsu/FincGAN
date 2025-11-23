# Installation Guide

## Requirements

### Dependencies

- **Python**: >= 3.6
- **PyTorch**: 1.11.0
  - RTX 2080 Ti or higher: CUDA 11.3
  - GTX 1080 Ti: CUDA 10.2
- **DGL**: >= 0.6.0
- **Other Dependencies**:
  - numpy
  - matplotlib
  - tqdm
  - pandas
  - scikit-learn

## Installation Steps

### 1. Install Basic Packages

```bash
pip install matplotlib tqdm numpy pandas scikit-learn
```

### 2. Install PyTorch (CUDA 11.3)

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

### 3. Install DGL

```bash
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
```

## Manual Setup

### Step-by-Step Installation

```bash
# Step 1: Create environment from YAML file
conda env create -f environment.yml
conda activate fincgan

# Step 2: Install PyTorch with CUDA 11.3
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Step 3: Install DGL with CUDA 11.3
pip install dgl-cu113==0.9.1 dglgo -f https://data.dgl.ai/wheels/repo.html

# Step 4: Verify installation
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import dgl; print(f'DGL: {dgl.__version__}')"
```

### Alternative: Without environment.yml

```bash
# Create environment
conda create -n fincgan python=3.9
conda activate fincgan

# Install dependencies
pip install matplotlib tqdm numpy pandas scikit-learn
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install dgl-cu113==0.9.1 dglgo -f https://data.dgl.ai/wheels/repo.html
```

## GPU Requirements

- **Recommended**: GPU with at least 8GB VRAM
- **Supported**: NVIDIA GPUs with CUDA support
- If encountering OOM errors, reduce `--batch_size` parameter

## Verification

After installation, verify your setup:

```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check DGL installation
python3 -c "import dgl; print(f'DGL version: {dgl.__version__}')"

# Check PyTorch version
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Common Installation Issues

### Issue 1: CUDA Version Mismatch

If you have a different CUDA version, adjust the PyTorch installation:

```bash
# For CUDA 10.2
pip3 install torch torchvision torchaudio

# For CUDA 11.6
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Issue 2: DGL Installation Fails

Try installing from the official wheel repository:

```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### Issue 3: Missing Dependencies

Install all dependencies at once:

```bash
pip install -r requirements.txt  # If requirements.txt exists
```

---

**Next Steps**: See [Dataset Information](dataset.md) or return to [Main README](../README.md)
