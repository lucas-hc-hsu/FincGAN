# Complete Implementation Workflow

This guide provides step-by-step instructions for implementing FincGAN from scratch.

## Overview

FincGAN consists of **5 sequential stages** that must be executed in order:

```
Stage I: Feature Extraction
    ↓
Stage II: Node Generator (GAN)
    ↓
Stage III: Edge Generators (UU + UP)
    ↓
Stage IV: Graph Generation & Training
    ↓
Stage V: Visualization & Analysis
```

---

## Quick Start (Automated)

**Recommended for most users**: Use the automated script

```bash
# Make script executable (first time only)
chmod +x run_fincgan.sh

# Run complete pipeline
./run_fincgan.sh

# Quick test (reduced epochs)
./run_fincgan.sh --n-epoch-emb 5 --n-epoch-gan 2 --n-epoch-uu 10 --n-epoch-up 5 --n-epoch-train 10
```

See [Automation Guide](../AUTOMATION_GUIDE.md) for detailed options.

---

## Manual Implementation (Step-by-Step)

For full control and understanding, follow the manual workflow:

### Prerequisites

1. **Activate conda environment**:
```bash
conda activate fincgan
```

2. **Verify installation**:
```bash
python3 -c "import torch, dgl; print(f'PyTorch: {torch.__version__}, DGL: {dgl.__version__}')"
```

3. **Check data exists**:
```bash
ls graph/music_instrument_25.bin
```

---

### Stage I: Feature Extraction

**Objective**: Extract 256-dimensional embeddings using HGT

```bash
python3 train.py \
    --gpu_id 0 \
    --n_epoch 20 \
    --seed 10 11 \
    --ratio 0.1007 \
    --setting "embedding" \
    --verbose 1
```

**Expected output**:
- `embed/music_hgt_user_emb.pt`
- `embed/music_hgt_product_emb.pt`

**Time**: ~5-10 minutes

**Verify**:
```bash
ls -lh embed/
python3 -c "import torch; print(torch.load('embed/music_hgt_user_emb.pt').shape)"
```

**Details**: [Stage I Documentation](stage1-embedding.md)

---

### Stage II: Node Generator

**Objective**: Train GAN to generate synthetic user nodes

```bash
python3 node_generator.py \
    --gpu_id 0 \
    --n_epochs 5 \
    --gan_verbose 1 \
    --tsne_verbose 1
```

**Expected output**:
- `generator/music_D.pt` (Discriminator)
- `generator/music_G.pt` (Generator)
- `tsne/*.jpg` (6 t-SNE visualization figures)

**Time**: ~20 minutes (5 epochs demo), ~2 hours (200 epochs production)

**Verify**:
```bash
ls -lh generator/
ls -lh tsne/
```

**Details**: [Stage II Documentation](stage2-node-generator.md)

---

### Stage III: Edge Generators

**Objective**: Train edge generators for user-user and user-product connections

#### 3.1 User-User Edge Generator

```bash
python3 edge_generator_uu.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --edge_dir "./generator/" \
    --edge_generator_verbose 1
```

**Time**: ~15-25 minutes

**⚠️ Important**: Use at least 100 epochs for good results

#### 3.2 User-Product Edge Generator

```bash
python3 edge_generator_up.py \
    --gpu_id 0 \
    --n_epoch 20 \
    --edge_dir "./generator/" \
    --edge_generator_verbose 1
```

**Time**: ~5-10 minutes

**Expected output**:
- `generator/uu_generator.pt`
- `generator/up_generator.pt`

**Verify**:
```bash
ls -lh generator/*.pt
```

**Details**: [Stage III Documentation](stage3-edge-generator.md)

---

### Stage IV: Graph Generation and Training

**Objective**: Generate augmented graphs and train fraud detection models

#### Option A: FincGAN Method Only

```bash
# Generate augmented graph
python3 graph_generator.py \
    --ratio 0.1007 \
    --up 0.99 \
    --uu 0.91 \
    --graph_dir "graph_output/" \
    --verbose 1

# Train on augmented graph
python3 train.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --seed 10 11 \
    --ratio 0.1007 \
    --setting "gan" \
    --verbose 1 \
    --graph_dir "graph_output/" \
    --result_dir "results/"
```

**Time**: ~30-50 minutes

#### Option B: All Baseline Methods + FincGAN

```bash
# Run all baseline methods
for method in origin oversampling reweight smote noise graphsmote; do
    python3 train.py \
        --gpu_id 0 \
        --n_epoch 100 \
        --seed 10 11 \
        --ratio 0.1007 \
        --setting "$method" \
        --verbose 0 \
        --graph_dir "graph_output/" \
        --result_dir "results/"
done

# Run FincGAN
python3 graph_generator.py --ratio 0.1007 --up 0.99 --uu 0.91 --graph_dir "graph_output/"
python3 train.py --gpu_id 0 --n_epoch 100 --seed 10 11 --ratio 0.1007 --setting "gan" --graph_dir "graph_output/" --result_dir "results/"
```

**Time**: ~3-4 hours (all methods)

**Expected output**:
- `graph_output/music_instrument_gan_*.bin`
- `results/music_hgt_model_gan.txt`
- `results/music_hgt_model_<method>.txt` (for each baseline)

**Verify**:
```bash
ls -lh graph_output/
ls -lh results/
cat results/music_hgt_model_gan.txt
```

**Details**: [Stage IV Documentation](stage4-training.md)

---

### Stage V: Visualization

**Objective**: Generate comparison plots and analyze results

```bash
# Automatic visualization
python3 -c "from visualize import auto_plot_figure_3; auto_plot_figure_3(result_dir='results/', save_fig=True)"

# View the plot
open figures/figure_3.png  # macOS
xdg-open figures/figure_3.png  # Linux
```

**Time**: ~1 minute

**Expected output**:
- `figures/figure_3.png`

**Verify**:
```bash
ls -lh figures/
```

**Details**: [Stage V Documentation](stage5-visualization.md)

---

## Complete Command Sequence

**Full FincGAN pipeline (copy-paste ready)**:

```bash
# Activate environment
conda activate fincgan

# Stage I: Feature Extraction
python3 train.py --gpu_id 0 --n_epoch 20 --seed 10 11 --ratio 0.1007 --setting "embedding" --verbose 1

# Stage II: Node Generator
python3 node_generator.py --gpu_id 0 --n_epochs 5 --gan_verbose 1 --tsne_verbose 1

# Stage III: Edge Generators
python3 edge_generator_uu.py --gpu_id 0 --n_epoch 100 --edge_dir "./generator/" --edge_generator_verbose 1
python3 edge_generator_up.py --gpu_id 0 --n_epoch 20 --edge_dir "./generator/" --edge_generator_verbose 1

# Stage IV: Graph Generation and Training
python3 graph_generator.py --ratio 0.1007 --up 0.99 --uu 0.91 --graph_dir "graph_output/" --verbose 1
python3 train.py --gpu_id 0 --n_epoch 100 --seed 10 11 --ratio 0.1007 --setting "gan" --verbose 1 --graph_dir "graph_output/" --result_dir "results/"

# Stage V: Visualization
python3 -c "from visualize import auto_plot_figure_3; auto_plot_figure_3(result_dir='results/', save_fig=True)"
```

---

## Workflow Variations

### Quick Test Run (5-10 minutes)

For testing purposes only:

```bash
# Reduced epochs
./run_fincgan.sh \
    --n-epoch-emb 5 \
    --n-epoch-gan 2 \
    --n-epoch-uu 10 \
    --n-epoch-up 5 \
    --n-epoch-train 10
```

### Production Run (2-3 hours)

For reproducing paper results:

```bash
./run_fincgan.sh \
    --n-epoch-emb 30 \
    --n-epoch-gan 200 \
    --n-epoch-uu 150 \
    --n-epoch-up 30 \
    --n-epoch-train 200
```

### Resuming Failed Runs

If a stage fails, skip completed stages:

```bash
# If Stage III failed, skip Stages I and II
./run_fincgan.sh --skip-stages "1,2"

# Or manually continue from Stage III
python3 edge_generator_uu.py --gpu_id 0 --n_epoch 100 --edge_dir "./generator/"
# ... continue with remaining stages
```

### Multiple Ratio Experiments

```bash
# Generate graphs for multiple ratios
python3 graph_generator.py \
    --ratio 0.10 0.15 0.20 \
    --up 0.99 \
    --uu 0.91 \
    --graph_dir "graph_output/"

# Train on each ratio
for ratio in 0.10 0.15 0.20; do
    python3 train.py \
        --gpu_id 0 \
        --n_epoch 100 \
        --seed 10 11 \
        --ratio $ratio \
        --setting "gan" \
        --graph_dir "graph_output/" \
        --result_dir "results/"
done
```

---

## Workflow Verification Checklist

Use this checklist to verify each stage:

- [ ] **Stage I Complete**
  - [ ] `embed/music_hgt_user_emb.pt` exists
  - [ ] `embed/music_hgt_product_emb.pt` exists
  - [ ] Embeddings have correct shape: (7017, 256) and (4684, 256)

- [ ] **Stage II Complete**
  - [ ] `generator/music_D.pt` exists
  - [ ] `generator/music_G.pt` exists
  - [ ] `tsne/*.jpg` files generated (6 files)
  - [ ] t-SNE plots show good distribution overlap

- [ ] **Stage III Complete**
  - [ ] `generator/uu_generator.pt` exists
  - [ ] `generator/up_generator.pt` exists
  - [ ] Both generators can be loaded without errors

- [ ] **Stage IV Complete**
  - [ ] `graph_output/music_instrument_gan_*.bin` exists
  - [ ] `results/music_hgt_model_gan.txt` exists
  - [ ] Results show reasonable metrics (AUC-ROC > 0.80)

- [ ] **Stage V Complete**
  - [ ] `figures/figure_3.png` exists
  - [ ] Plot shows comparison across methods
  - [ ] FincGAN performance is competitive with baselines

---

## Directory Structure After Completion

```
fincgan/
├── graph/                          # Original data
│   └── music_instrument_25.bin
├── embed/                          # Stage I output
│   ├── music_hgt_user_emb.pt
│   └── music_hgt_product_emb.pt
├── generator/                      # Stage II & III output
│   ├── music_D.pt
│   ├── music_G.pt
│   ├── uu_generator.pt
│   └── up_generator.pt
├── tsne/                          # Stage II visualizations
│   ├── tsne.jpg
│   ├── tsne_real_benign.jpg
│   ├── tsne_real_spam.jpg
│   ├── tsne_fake_benign.jpg
│   ├── tsne_fake_spam.jpg
│   └── tsne_real_fake.jpg
├── graph_output/                  # Stage IV generated graphs
│   └── music_instrument_gan_*.bin
├── results/                       # Stage IV training results
│   ├── music_hgt_model_gan.txt
│   ├── music_hgt_model_smote.txt
│   └── ...
├── figures/                       # Stage V visualizations
│   └── figure_3.png
└── tmp/                          # Temporary model checkpoints
    └── music_hgt_model_*.pt
```

---

## Troubleshooting

For detailed troubleshooting, see [Troubleshooting Guide](troubleshooting.md).

### Common Issues

1. **CUDA Out of Memory**: Use `--gpu_id -1` for CPU mode
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Permission denied**: Use `chmod +x run_fincgan.sh`
4. **Poor performance**: Increase epochs, try different ratios

---

## Next Steps

After completing the workflow:

1. **Analyze results**: Compare FincGAN with baselines
2. **Experiment**: Try different parameters (ratios, thresholds, epochs)
3. **Optimize**: Fine-tune for your specific use case
4. **Deploy**: Use trained models for fraud detection

---

**Related Documentation:**
- [Stage I: Embedding](stage1-embedding.md)
- [Stage II: Node Generator](stage2-node-generator.md)
- [Stage III: Edge Generator](stage3-edge-generator.md)
- [Stage IV: Training](stage4-training.md)
- [Stage V: Visualization](stage5-visualization.md)
- [Automation Guide](../AUTOMATION_GUIDE.md)
- [Troubleshooting](troubleshooting.md)
- [Main README](../README.md)
