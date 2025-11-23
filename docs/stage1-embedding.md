# Stage I: Feature Extractor

## Objective

Extract high-dimensional features (default: 256 dimensions) from original node embeddings using Heterogeneous Graph Transformer (HGT).

This stage trains an HGT model on the original graph to learn meaningful representations of users and products. These embeddings will be used in subsequent stages for GAN training and edge generation.

## Execution Command

### Basic Usage

```bash
python3 train.py \
    --gpu_id 0 \
    --n_epoch 20 \
    --seed 10 11 \
    --ratio 0.1007 \
    --setting "embedding" \
    --verbose 1
```

### Advanced Usage

```bash
python3 train.py \
    --gpu_id 2 \
    --n_epoch 20 \
    --seed 10 11 \
    --ratio 0.17 \
    --setting "embedding" \
    --verbose 1 \
    --graph_dir "example_dir/" \
    --result_dir "example_dir/"
```

## Parameter Description

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--gpu_id` | GPU device ID | 0 | Use available GPU |
| `--n_epoch` | Number of training epochs | 100 | 20-30 for demo, 100+ for production |
| `--seed` | Random seed(s) (can specify multiple) | 0 | 10 11 (for reproducibility) |
| `--ratio` | Ratio of synthetic nodes | 0.1 | 0.1007 (based on dataset) |
| `--setting` | Training method | - | **"embedding"** (required for this stage) |
| `--verbose` | Show training details (0/1) | 0 | 1 (to monitor progress) |
| `--graph_dir` | Directory for graph data | `./graph/` | Custom if needed |
| `--result_dir` | Directory for results | `./result/` | Custom if needed |

## Output Files

After successful execution, the following files will be generated:

### Embedding Files

- **User embeddings**: `embed/music_hgt_user_emb.pt`
  - Shape: (num_users, 256)
  - Contains learned 256-dimensional embeddings for all users

- **Product embeddings**: `embed/music_hgt_product_emb.pt`
  - Shape: (num_products, 256)
  - Contains learned 256-dimensional embeddings for all products

### Model Checkpoints

Temporary model files may be saved in:
- `tmp/music_hgt_model_*.pt`

## Expected Output

During training, you should see:

```
Epoch 1/20:
  Train Loss: 0.xxxx
  Val Loss: 0.xxxx
  ...
Epoch 20/20:
  Train Loss: 0.xxxx
  Val Loss: 0.xxxx

Embeddings saved to:
  - embed/music_hgt_user_emb.pt
  - embed/music_hgt_product_emb.pt
```

## Execution Time

- **Quick test** (5-10 epochs): ~2-3 minutes
- **Demo** (20 epochs): ~5-10 minutes
- **Production** (100+ epochs): ~20-30 minutes

*Time varies based on GPU and hardware*

## Verification

To verify the embeddings were generated correctly:

```bash
# Check if embedding files exist
ls -lh embed/

# Verify embedding dimensions in Python
python3 -c "
import torch
user_emb = torch.load('embed/music_hgt_user_emb.pt')
product_emb = torch.load('embed/music_hgt_product_emb.pt')
print(f'User embeddings: {user_emb.shape}')
print(f'Product embeddings: {product_emb.shape}')
"
```

Expected output:
```
User embeddings: torch.Size([7017, 256])
Product embeddings: torch.Size([4684, 256])
```

## Common Issues

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size or use CPU mode:
```bash
python3 train.py --setting "embedding" --gpu_id -1  # Use CPU
```

### Issue 2: Embeddings Not Saved

**Solution**: Ensure the `embed/` directory exists:
```bash
mkdir -p embed
```

### Issue 3: Training Not Converging

**Solution**: Try different learning rate or increase epochs:
```bash
python3 train.py --setting "embedding" --n_epoch 50
```

## Next Steps

After successfully generating embeddings:

1. **Verify embeddings**: Check the generated `.pt` files
2. **Proceed to Stage II**: [Node Generator](stage2-node-generator.md)
3. **Or use automated script**: See [Automation Guide](../AUTOMATION_GUIDE.md)

---

**Related Documentation:**
- [Stage II: Node Generator](stage2-node-generator.md)
- [Complete Workflow](workflow.md)
- [Main README](../README.md)
