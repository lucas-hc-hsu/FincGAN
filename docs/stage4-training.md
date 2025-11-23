# Stage IV: Graph Generation and Training

## Objective

Generate augmented graphs using trained models and train fraud detection models on the augmented data.

This stage has two main components:
1. **Generate augmented graphs** using various methods (baselines or FincGAN)
2. **Train HGT models** on the augmented graphs for fraud detection

## Prerequisites

For **FincGAN method** specifically:
- Stage I: Embeddings must exist
- Stage II: GAN models must exist
- Stage III: Edge generators must exist

For **baseline methods**: Only Stage I embeddings are required (some methods don't even need that)

---

## 4.1 Baseline Methods

FincGAN supports multiple baseline methods for comparison:

### Available Methods

| Method | Description | Requires Stage I |
|--------|-------------|------------------|
| `origin` | Original graph (no augmentation) | No |
| `embedding` | Feature extraction only | Yes (this IS Stage I) |
| `oversampling` | Simple oversampling | No |
| `reweight` | Class reweighting | No |
| `smote` | SMOTE algorithm | Yes |
| `noise` | Adding noise to embeddings | Yes |
| `graphsmote` | GraphSMOTE | Yes |
| `gan` | FincGAN (our method) | Yes + Stage II + Stage III |

### Running Baseline Methods

#### SMOTE Method

```bash
python3 train.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --seed 10 11 \
    --ratio 0.1007 \
    --setting "smote" \
    --verbose 1 \
    --graph_dir "graph_output/" \
    --result_dir "results/"
```

#### Noise Method

```bash
python3 train.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --seed 10 11 \
    --ratio 0.1007 \
    --setting "noise" \
    --verbose 1 \
    --graph_dir "graph_output/" \
    --result_dir "results/"
```

#### Oversampling Method

```bash
python3 train.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --seed 10 11 \
    --ratio 0.1007 \
    --setting "oversampling" \
    --verbose 0 \
    --graph_dir "graph_output/" \
    --result_dir "results/"
```

#### Reweight Method

```bash
python3 train.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --seed 10 11 \
    --ratio 0.1007 \
    --setting "reweight" \
    --verbose 0 \
    --graph_dir "graph_output/" \
    --result_dir "results/"
```

#### GraphSMOTE Method

```bash
python3 train.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --seed 10 11 \
    --ratio 0.1007 \
    --setting "graphsmote" \
    --verbose 1 \
    --graph_dir "graph_output/" \
    --result_dir "results/"
```

### Baseline Results

Each baseline method will generate:
- **Graph file** (if applicable): `graph_output/music_instrument_<method>_*.bin`
- **Results file**: `results/music_hgt_model_<method>.txt`

---

## 4.2 FincGAN Method

The FincGAN method uses all trained models to generate an augmented graph.

### Option 1: Multi-Graph Generation (Recommended)

Generate multiple graphs with different ratios in one command:

```bash
python3 graph_generator.py \
    --ratio 0.1007 0.11 0.12 \
    --up 0.99 \
    --uu 0.91 \
    --graph_dir "graph_output/" \
    --verbose 1
```

**Tip**: Generate the graph with the **largest ratio first**, then split the synthetic nodes for different ratio settings to save time.

#### Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--ratio` | Ratio(s) of synthetic nodes (can specify multiple) | 0.1 | 0.1007 0.11 0.12 |
| `--up` | User-Product edge generator threshold | 0.99 | 0.99 (higher = fewer edges) |
| `--uu` | User-User edge generator threshold | 0.91 | 0.91 (higher = fewer edges) |
| `--graph_dir` | Graph output directory | `./graph/` | `./graph_output/` |
| `--verbose` | Show generation details (0/1) | 0 | 1 |

#### Output

Generates augmented graph files:
- `graph_output/music_instrument_gan_0.1007.bin`
- `graph_output/music_instrument_gan_0.11.bin`
- `graph_output/music_instrument_gan_0.12.bin`

### Option 2: Single Graph Generation + Training

Generate one graph and immediately train on it:

```bash
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

This will:
1. Check if `graph_output/music_instrument_gan_0.1007.bin` exists
2. If not, generate it using the trained GAN and edge generators
3. Train HGT model on the augmented graph
4. Save results to `results/music_hgt_model_gan.txt`

---

## Training Parameters

### Common Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--gpu_id` | GPU device ID | 0 | Use available GPU |
| `--n_epoch` | Number of training epochs | 100 | 100+ for final results |
| `--seed` | Random seed(s) | 0 | 10 11 (for reproducibility) |
| `--ratio` | Ratio of synthetic nodes | 0.1 | 0.1007 (based on dataset) |
| `--setting` | Training method | - | "gan", "smote", etc. |
| `--verbose` | Show training details (0/1) | 0 | 1 (to monitor progress) |
| `--graph_dir` | Directory for graphs | `./graph/` | `./graph_output/` |
| `--result_dir` | Directory for results | `./result/` | `./results/` |

### Edge Generation Thresholds

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--uu` | User-User edge threshold | 0.91 | 0.0 - 1.0 |
| `--up` | User-Product edge threshold | 0.99 | 0.0 - 1.0 |

**Note**:
- `--up 0` and `--uu 0` are only for quick demonstration
- Use default values (0.91 and 0.99) for actual implementation
- Higher threshold = fewer edges (stricter edge generation)
- Lower threshold = more edges (looser edge generation)

---

## Training Result Format

Results are saved as CSV files in the `results/` directory:

```
ratio,seed,AUC-PRC,AUC-ROC,F-score,precision,recall,ACC
0.1007,10,0.4515,0.8851,0.4545,0.4630,0.4464,0.9147
0.1007,11,0.4521,0.8847,0.4551,0.4635,0.4470,0.9145
```

### Example Output

```
AUC-PRC: 0.4515
AUC-ROC: 0.8851
F1: 0.4545
Precision: 0.4630
Recall: 0.4464
ACC: 0.9147
```

### Metrics Explanation

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| **AUC-PRC** | Area Under Precision-Recall Curve | 0.40 - 0.50 |
| **AUC-ROC** | Area Under ROC Curve | 0.85 - 0.90 |
| **F1-Score** | Harmonic mean of precision and recall | 0.40 - 0.50 |
| **Precision** | True positives / (True + False positives) | 0.50 - 0.60 |
| **Recall** | True positives / (True + False negatives) | 0.35 - 0.45 |
| **Accuracy** | Overall correctness | 0.92 - 0.93 |

*Note: Values may vary based on configuration and random seeds*

---

## Execution Time

| Method | Graph Generation | Training (100 epochs) | Total |
|--------|------------------|----------------------|-------|
| Baselines | 0-5 minutes | 20-30 minutes | ~30 min |
| FincGAN | 10-20 minutes | 20-30 minutes | ~50 min |

*Times vary based on GPU, ratio, and number of seeds*

---

## Complete Workflow Example

### Run All Baseline Methods + FincGAN

```bash
# Baseline methods
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

# FincGAN method
python3 graph_generator.py \
    --ratio 0.1007 \
    --up 0.99 \
    --uu 0.91 \
    --graph_dir "graph_output/" \
    --verbose 1

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

---

## Verification

### Check Generated Graphs

```bash
ls -lh graph_output/
```

Expected output:
```
music_instrument_gan_0.1007.bin
music_instrument_smote_0.1007.bin
music_instrument_noise_0.1007.bin
...
```

### Check Results

```bash
ls -lh results/
```

Expected output:
```
music_hgt_model_gan.txt
music_hgt_model_smote.txt
music_hgt_model_noise.txt
...
```

### View Results

```bash
cat results/music_hgt_model_gan.txt
```

Or in Python:
```python
import pandas as pd
df = pd.read_csv('results/music_hgt_model_gan.txt')
print(df)
```

---

## Common Issues

### Issue 1: Graph Already Exists

**Symptom**: "Graph already exists, loading directly..."

**Solution**: This is expected behavior. To regenerate:
```bash
rm graph_output/music_instrument_gan_0.1007.bin
```

### Issue 2: Missing Models

**Symptom**: "FileNotFoundError: generator/music_G.pt not found"

**Solution**: Complete Stages I-III first:
```bash
# Stage I
python3 train.py --setting "embedding"
# Stage II
python3 node_generator.py
# Stage III
python3 edge_generator_uu.py
python3 edge_generator_up.py
```

### Issue 3: Poor Performance

**Solution**:
- Increase training epochs: `--n_epoch 200`
- Try different ratios: `--ratio 0.15`
- Adjust edge thresholds: `--uu 0.85 --up 0.95`
- Use more seeds: `--seed 10 11 12 13 14`

### Issue 4: Out of Memory

**Solution**:
- Use CPU: `--gpu_id -1`
- Reduce batch size (requires code modification)
- Close other GPU programs

---

## Next Steps

After completing graph generation and training:

1. **Review results**: Check `results/` directory
2. **Proceed to Stage V**: [Visualization](stage5-visualization.md)
3. **Compare methods**: Analyze which method performs best
4. **Or use automated script**: See [Automation Guide](../AUTOMATION_GUIDE.md)

---

**Related Documentation:**
- [Stage III: Edge Generator](stage3-edge-generator.md)
- [Stage V: Visualization](stage5-visualization.md)
- [Evaluation Metrics](metrics.md)
- [Complete Workflow](workflow.md)
- [Main README](../README.md)
