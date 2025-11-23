# Stage III: Edge Generator

## Objective

Train two edge generators to create edges between synthetic nodes and existing nodes in the graph:

1. **User-User (u-u) Edge Generator**: Generates sparse user-user relationships
2. **User-Product (u-p) Edge Generator**: Generates dense user-product review edges

These edge generators learn the edge formation patterns from the original graph and apply them to connect synthetic nodes realistically.

## Prerequisites

- **Stage I must be completed**: Embeddings must exist
  - `embed/music_hgt_user_emb.pt`
  - `embed/music_hgt_product_emb.pt`
- **Stage II must be completed**: GAN models must exist (optional but recommended for testing)

## Two Sub-Stages

### 3.1 Train User-User Edge Generator

This generator learns to create **sparse** user-user edges (social connections).

#### Command

```bash
python3 edge_generator_uu.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --edge_dir "./generator/" \
    --edge_generator_verbose 1
```

#### Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--n_epoch` | Number of training epochs | 100 | **100+** (critical) |
| `--edge_dir` | Directory for edge generators | `./embed/` | `./generator/` |
| `--edge_generator_verbose` | Show training details (0/1) | 0 | 1 |
| `--emb_dim` | Embedding dimension | 256 | 256 |
| `--batch_size` | Batch size | - | Auto (based on data) |
| `--gpu_id` | GPU device ID | 0 | Use available GPU |

#### Output

- **Model file**: `generator/uu_generator.pt`
- Edge generator for user-user relationships

#### Execution Time

- **100 epochs**: ~15-25 minutes
- **Recommended**: Do not use less than 100 epochs for good results

---

### 3.2 Train User-Product Edge Generator

This generator learns to create **dense** user-product edges (reviews/interactions).

#### Command

```bash
python3 edge_generator_up.py \
    --gpu_id 0 \
    --n_epoch 20 \
    --edge_dir "./generator/" \
    --edge_generator_verbose 1
```

#### Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--n_epoch` | Number of training epochs | 100 | 20+ |
| `--edge_dir` | Directory for edge generators | `./embed/` | `./generator/` |
| `--edge_generator_verbose` | Show training details (0/1) | 0 | 1 |
| `--emb_dim` | Embedding dimension | 256 | 256 |
| `--batch_size` | Batch size | 9192 | Auto |
| `--gpu_id` | GPU device ID | 0 | Use available GPU |

#### Output

- **Model file**: `generator/up_generator.pt`
- Edge generator for user-product relationships

#### Execution Time

- **20 epochs**: ~5-10 minutes
- Can use fewer epochs than UU generator due to denser edge structure

---

## Complete Execution

Run both edge generators in sequence:

```bash
# Step 1: Train User-User edge generator
python3 edge_generator_uu.py \
    --gpu_id 0 \
    --n_epoch 100 \
    --edge_dir "./generator/" \
    --edge_generator_verbose 1

# Step 2: Train User-Product edge generator
python3 edge_generator_up.py \
    --gpu_id 0 \
    --n_epoch 20 \
    --edge_dir "./generator/" \
    --edge_generator_verbose 1
```

## Expected Output

### User-User Edge Generator Training

```
Training User-User Edge Generator...
Epoch 1/100:
  Loss: 0.xxxx
  ...
Epoch 100/100:
  Loss: 0.xxxx

Model saved to: generator/uu_generator.pt
```

### User-Product Edge Generator Training

```
Training User-Product Edge Generator...
Epoch 1/20:
  Loss: 0.xxxx
  ...
Epoch 20/20:
  Loss: 0.xxxx

Model saved to: generator/up_generator.pt
```

## Verification

### Check Model Files

```bash
ls -lh generator/
```

Expected output:
```
-rw-r--r-- 1 user user 1.8M music_G.pt
-rw-r--r-- 1 user user 2.1M music_D.pt
-rw-r--r-- 1 user user 512K uu_generator.pt
-rw-r--r-- 1 user user 512K up_generator.pt
```

### Test Edge Generators

```python
import torch

# Load user embeddings
user_emb = torch.load('embed/music_hgt_user_emb.pt')
product_emb = torch.load('embed/music_hgt_product_emb.pt')

# Load edge generators
uu_gen = torch.load('generator/uu_generator.pt')
up_gen = torch.load('generator/up_generator.pt')

print("✓ All edge generators loaded successfully")
print(f"User embedding shape: {user_emb.shape}")
print(f"Product embedding shape: {product_emb.shape}")
```

## Understanding Edge Generation

### User-User (UU) Edge Generator

**Purpose**: Connects synthetic users to existing users based on similarity

**Characteristics**:
- **Sparse edges**: Not every user is connected
- **Threshold-based**: Uses similarity threshold (default: 0.91)
- **Longer training**: Requires more epochs due to sparsity

**Edge Formation**:
```python
# Pseudo-code
similarity = compute_similarity(user1_emb, user2_emb)
if similarity > UU_THRESHOLD:
    create_edge(user1, user2)
```

### User-Product (UP) Edge Generator

**Purpose**: Connects synthetic users to products (creates reviews/interactions)

**Characteristics**:
- **Dense edges**: Users typically review multiple products
- **Threshold-based**: Uses similarity threshold (default: 0.99)
- **Faster training**: Requires fewer epochs due to density

**Edge Formation**:
```python
# Pseudo-code
interaction_prob = compute_interaction(user_emb, product_emb)
if interaction_prob > UP_THRESHOLD:
    create_edge(user, product)
```

## Important Notes

### Critical Requirements

⚠️ **Minimum Epochs for UU Generator**: Use **at least 100 epochs** for the user-user edge generator. Lower values will result in poor edge quality.

⚠️ **Embedding Dimension**: Must match the dimension used in Stage I (default: 256)

⚠️ **Edge Directory**: Ensure `edge_dir` is consistent across both generators

### Threshold Values

The edge generation thresholds are set during **Stage IV** when generating the augmented graph:
- `--uu 0.91`: User-User threshold (default)
- `--up 0.99`: User-Product threshold (default)

These values determine how strict the edge generation is. Higher thresholds = fewer edges.
