# Troubleshooting Guide

This guide covers common issues and solutions when working with FincGAN.

## Table of Contents

- [Environment Issues](#environment-issues)
- [Training Issues](#training-issues)
- [Memory Issues](#memory-issues)
- [File and Path Issues](#file-and-path-issues)
- [Performance Issues](#performance-issues)
- [Visualization Issues](#visualization-issues)
- [General Tips](#general-tips)

---

## Environment Issues

### Issue: "conda: command not found"

**Symptom**: Shell cannot find conda command

**Solution**:
```bash
# Manually source conda
source ~/miniconda3/etc/profile.d/conda.sh

# Or for Anaconda
source ~/anaconda3/etc/profile.d/conda.sh

# Then activate environment
conda activate fincgan
```

**Permanent fix**: Add to `~/.bashrc` or `~/.zshrc`:
```bash
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

---

### Issue: "ModuleNotFoundError"

**Symptom**: Python cannot import required modules

**Solution**:
```bash
# Ensure environment is activated
conda activate fincgan

# Install missing modules
pip install torch dgl numpy pandas scikit-learn matplotlib tqdm

# For specific CUDA version
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
```

---

### Issue: CUDA Version Mismatch

**Symptom**: PyTorch cannot find CUDA or version mismatch

**Check CUDA version**:
```bash
nvidia-smi
nvcc --version
```

**Solution**: Install matching PyTorch version:
```bash
# For CUDA 11.3
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# For CUDA 11.6
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# For CUDA 10.2
pip3 install torch==1.11.0+cu102 torchvision==0.12.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

---

### Issue: DGL Installation Fails

**Symptom**: Cannot install or import DGL

**Solution**:
```bash
# Try different installation method
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# For CUDA 11.3
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html

# Verify installation
python3 -c "import dgl; print(dgl.__version__)"
```

---

## Training Issues

### Issue: Training Not Converging

**Symptom**: Loss not decreasing or unstable training

**Solutions**:

1. **Check learning rate**:
```bash
# Reduce learning rate
python3 node_generator.py --lr 0.0001  # default is 0.0002
```

2. **Increase epochs**:
```bash
python3 train.py --n_epoch 200  # instead of 100
```

3. **Verify data**:
```bash
# Check if embeddings exist and are correct
python3 -c "
import torch
user_emb = torch.load('embed/music_hgt_user_emb.pt')
print(f'User embeddings: {user_emb.shape}')
print(f'Contains NaN: {torch.isnan(user_emb).any()}')
print(f'Contains Inf: {torch.isinf(user_emb).any()}')
"
```

---

### Issue: GAN Mode Collapse

**Symptom**: Generator produces similar outputs for all inputs

**Check t-SNE plots**:
```bash
ls tsne/
# Look for clustering of fake nodes in one area
```

**Solutions**:

1. **Increase training epochs**:
```bash
python3 node_generator.py --n_epochs 200
```

2. **Adjust architecture** (requires code modification):
- Increase discriminator capacity
- Add dropout layers
- Use spectral normalization

3. **Try different random seeds**:
```bash
python3 node_generator.py --seed 42
```

---

### Issue: "Graph Already Exists" Warning

**Symptom**: Script loads existing graph instead of generating new one

**Expected behavior**: This is intentional for efficiency

**To regenerate**:
```bash
# Remove existing graph
rm graph_output/music_instrument_gan_0.1007.bin

# Then run again
python3 train.py --setting "gan" --ratio 0.1007
```

---

### Issue: Poor Edge Generator Performance

**Symptom**: Generated graph has too few or too many edges

**Solutions**:

1. **Increase UU training epochs** (most common fix):
```bash
python3 edge_generator_uu.py --n_epoch 150  # instead of 100
```

2. **Adjust thresholds**:
```bash
# More edges (lower thresholds)
python3 graph_generator.py --uu 0.85 --up 0.95

# Fewer edges (higher thresholds)
python3 graph_generator.py --uu 0.95 --up 0.99
```

3. **Check edge generator loss**:
```bash
python3 edge_generator_uu.py --edge_generator_verbose 1
# Loss should decrease steadily
```

---

## Memory Issues

### Issue: CUDA Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Use CPU mode**:
```bash
python3 train.py --gpu_id -1
```

2. **Use different GPU**:
```bash
# Check GPU memory
nvidia-smi

# Use less occupied GPU
python3 train.py --gpu_id 1
```

3. **Reduce batch size** (requires code modification in the Python files)

4. **Clear GPU cache**:
```python
import torch
torch.cuda.empty_cache()
```

5. **Reduce model size** (requires architecture modification)

---

### Issue: System RAM Out of Memory

**Symptom**: System freezes or "Killed" message

**Solutions**:

1. **Monitor memory usage**:
```bash
watch -n 1 free -h
```

2. **Close other applications**

3. **Use swap space**:
```bash
# Check swap
free -h

# Add swap if needed (requires sudo)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

4. **Reduce data size** (process in batches)

---

### Issue: Disk Space Full

**Symptom**: "No space left on device"

**Check disk usage**:
```bash
df -h
du -sh fincgan/*
```

**Solutions**:

1. **Clean temporary files**:
```bash
rm -rf tmp/*
rm -rf __pycache__/
```

2. **Remove old results**:
```bash
# Archive old results
tar -czf old_results_backup.tar.gz results/ figures/
rm -rf results/* figures/*
```

3. **Clean conda cache**:
```bash
conda clean --all
```

---

## File and Path Issues

### Issue: "FileNotFoundError"

**Symptom**: Cannot find required files

**Common causes and solutions**:

1. **Wrong working directory**:
```bash
# Ensure you're in the fincgan directory
pwd
# Should show: /path/to/fincgan

# If not, navigate there
cd /home/hhc_wsl/fincgan
```

2. **Missing data file**:
```bash
# Check if original data exists
ls graph/music_instrument_25.bin

# If missing, download or restore from backup
```

3. **Stage not completed**:
```bash
# If embed files missing, run Stage I
python3 train.py --setting "embedding"

# If generator files missing, run Stages II and III
```

---

### Issue: Permission Denied

**Symptom**: "Permission denied" when running scripts

**Solution**:
```bash
# Make script executable
chmod +x run_fincgan.sh
chmod +x verify_cleanup.sh

# Or run with bash
bash run_fincgan.sh
```

---

### Issue: Import Error After Cleanup

**Symptom**: Cannot import custom modules

**Solution**:
```bash
# Check if __init__.py exists (if using package structure)
# Or run from correct directory
cd /home/hhc_wsl/fincgan
python3 train.py --setting "embedding"

# Verify Python can find modules
python3 -c "import utils; import hgt_model"
```

---

## Performance Issues

### Issue: Training Too Slow

**Symptom**: Each epoch takes very long

**Solutions**:

1. **Verify GPU is being used**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
```

2. **Check GPU utilization**:
```bash
watch -n 1 nvidia-smi
# GPU utilization should be high (>80%)
```

3. **Use faster GPU**:
```bash
python3 train.py --gpu_id 0  # Try different GPU
```

4. **Reduce epochs for testing**:
```bash
# Use fewer epochs for quick testing
./run_fincgan.sh --n-epoch-train 20
```

---

### Issue: Poor Fraud Detection Performance

**Symptom**: Low AUC-ROC, F1-score, or other metrics

**Solutions**:

1. **Increase training epochs**:
```bash
./run_fincgan.sh \
    --n-epoch-emb 30 \
    --n-epoch-gan 200 \
    --n-epoch-uu 150 \
    --n-epoch-up 30 \
    --n-epoch-train 200
```

2. **Try different synthetic node ratios**:
```bash
python3 graph_generator.py --ratio 0.15 0.20 0.25
```

3. **Adjust edge thresholds**:
```bash
python3 graph_generator.py --uu 0.85 --up 0.95
```

4. **Use more random seeds** for stable results:
```bash
python3 train.py --seed 10 11 12 13 14 15
```

5. **Check if GAN trained properly**:
```bash
# Review t-SNE plots
ls tsne/
# Fake nodes should overlap with real nodes
```

---

### Issue: Results Inconsistent Across Runs

**Symptom**: Different results with same parameters

**Solutions**:

1. **Use multiple seeds and average**:
```bash
python3 train.py --seed 10 11 12 13 14
```

2. **Set random seeds in code** (for exact reproducibility)

3. **Use more training epochs** (reduces variance)

---

## Visualization Issues

### Issue: t-SNE Visualization Fails

**Symptom**: Error during t-SNE generation

**Solutions**:

1. **Update scikit-learn**:
```bash
pip install --upgrade scikit-learn
```

2. **Check matplotlib**:
```bash
pip install --upgrade matplotlib
```

3. **Reduce t-SNE sample size** (if OOM, requires code modification)

4. **Use CPU for visualization**:
```bash
python3 node_generator.py --gpu_id -1
```

---

### Issue: Figure Not Displaying

**Symptom**: Plot generated but cannot view

**Solutions**:

1. **Check if figure exists**:
```bash
ls -lh figures/figure_3.png
```

2. **View with different method**:
```bash
# Linux
xdg-open figures/figure_3.png

# macOS
open figures/figure_3.png

# WSL (Windows Subsystem for Linux)
explorer.exe figures/figure_3.png

# Or copy to local machine
scp user@server:/path/to/fincgan/figures/figure_3.png .
```

3. **Use Jupyter notebook**:
```python
from IPython.display import Image
Image('figures/figure_3.png')
```

---

### Issue: Visualization Import Error

**Symptom**: `ImportError: cannot import name 'auto_plot_figure_3'`

**Solutions**:

1. **Check file exists**:
```bash
ls -l visualize.py
```

2. **Verify function exists**:
```python
import visualize
print(dir(visualize))
# Should include 'auto_plot_figure_3'
```

3. **Check for syntax errors**:
```bash
python3 -m py_compile visualize.py
```

---

## General Tips

### Enable Verbose Output

For debugging, always use verbose mode:

```bash
python3 train.py --verbose 1
python3 node_generator.py --gan_verbose 1 --tsne_verbose 2
python3 edge_generator_uu.py --edge_generator_verbose 1
```

---

### Check Logs

Review output logs for errors:

```bash
# Save output to log file
./run_fincgan.sh 2>&1 | tee fincgan_run.log

# Search for errors
grep -i error fincgan_run.log
grep -i warning fincgan_run.log
```

---

### Dry Run Before Execution

Test configuration without training:

```bash
# Preview what will be executed
./run_fincgan.sh --dry-run
```

---

### Verify Environment

Create a verification script:

```bash
cat > verify_env.sh << 'EOF'
#!/bin/bash
echo "=== Environment Verification ==="
conda --version
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import dgl; print(f'DGL: {dgl.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
nvidia-smi
echo "=== Verification Complete ==="
EOF

chmod +x verify_env.sh
./verify_env.sh
```

---

### Backup Important Data

Always backup before major operations:

```bash
# Backup results
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/ figures/

# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz embed/ generator/
```

---

### Monitor System Resources

Use monitoring tools:

```bash
# GPU monitoring
watch -n 1 nvidia-smi

# CPU and memory monitoring
htop

# Disk I/O monitoring
iostat -x 1
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check main README**: See [Main README](../README.md)
2. **Review stage documentation**: See individual stage guides
3. **Check automation guide**: See [Automation Guide](../AUTOMATION_GUIDE.md)
4. **Verify code cleanup**: See [Cleanup Summary](../CLEANUP_SUMMARY.md)
5. **Contact maintainers**: Provide detailed error messages and logs

---

## Debugging Workflow

Follow this systematic approach:

1. **Identify the error**: Read error message carefully
2. **Check prerequisites**: Ensure previous stages completed
3. **Verify files exist**: Check all required files are present
4. **Check environment**: Conda env activated, packages installed
5. **Enable verbose mode**: Get detailed output
6. **Search this guide**: Look for similar issues
7. **Try solutions systematically**: Test one fix at a time
8. **Document what works**: Note successful solutions

---

**Related Documentation:**
- [Installation Guide](installation.md)
- [Complete Workflow](workflow.md)
- [Automation Guide](../AUTOMATION_GUIDE.md)
- [Main README](../README.md)
