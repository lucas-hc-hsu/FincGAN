#!/bin/bash
###############################################################################
# FincGAN Automated Training Script
#
# This script automates the complete FincGAN training pipeline:
# - Stage I: Feature Extraction
# - Stage II: Node Generation (GAN)
# - Stage III: Edge Generation (User-User and User-Product)
# - Stage IV: Graph Generation and Training
# - Stage V: Result Visualization
#
# Usage: ./run_fincgan.sh [OPTIONS]
###############################################################################

set -e  # Exit on error

# Default configuration
GPU_ID=0
N_EPOCH_EMBEDDING=20
N_EPOCH_GAN=5
N_EPOCH_UU=100
N_EPOCH_UP=20
N_EPOCH_TRAIN=100
SEED_START=10
SEED_END=11
RATIO=0.1007
UU_THRESHOLD=0.91
UP_THRESHOLD=0.99
GRAPH_DIR="./graph_output/"
RESULT_DIR="./results/"
VERBOSE=1
SKIP_STAGES=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} $1"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_stage() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  Stage $1: $2${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

FincGAN Automated Training Pipeline

OPTIONS:
    -h, --help              Show this help message
    -g, --gpu-id GPU_ID     GPU ID to use (default: 0)
    -r, --ratio RATIO       Ratio of synthetic nodes (default: 0.1007)
    -s, --seed-start N      Starting seed (default: 10)
    -e, --seed-end N        Ending seed (default: 11)
    --n-epoch-emb N         Epochs for embedding (default: 20)
    --n-epoch-gan N         Epochs for GAN training (default: 5)
    --n-epoch-uu N          Epochs for UU edge generator (default: 100)
    --n-epoch-up N          Epochs for UP edge generator (default: 20)
    --n-epoch-train N       Epochs for final training (default: 100)
    --uu-threshold T        UU edge generator threshold (default: 0.91)
    --up-threshold T        UP edge generator threshold (default: 0.99)
    --graph-dir DIR         Directory for graph output (default: ./graph_output/)
    --result-dir DIR        Directory for results (default: ./results/)
    --skip-stages STAGES    Comma-separated stages to skip (e.g., "1,2")
    --quiet                 Suppress verbose output
    --dry-run               Show what would be executed without running

STAGES:
    Stage I   : Feature Extraction (embedding)
    Stage II  : Node Generator (GAN) training
    Stage III : Edge Generator training (UU and UP)
    Stage IV  : Graph generation and training
    Stage V   : Result visualization

EXAMPLES:
    # Run complete pipeline with default settings
    $0

    # Run with custom epochs and GPU
    $0 --gpu-id 2 --n-epoch-gan 10 --n-epoch-train 50

    # Skip Stage I and II (use existing embeddings and GAN models)
    $0 --skip-stages "1,2"

    # Quick test run
    $0 --n-epoch-emb 5 --n-epoch-gan 2 --n-epoch-uu 10 --n-epoch-up 5 --n-epoch-train 10

EOF
    exit 0
}

# Parse command line arguments
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -g|--gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        -r|--ratio)
            RATIO="$2"
            shift 2
            ;;
        -s|--seed-start)
            SEED_START="$2"
            shift 2
            ;;
        -e|--seed-end)
            SEED_END="$2"
            shift 2
            ;;
        --n-epoch-emb)
            N_EPOCH_EMBEDDING="$2"
            shift 2
            ;;
        --n-epoch-gan)
            N_EPOCH_GAN="$2"
            shift 2
            ;;
        --n-epoch-uu)
            N_EPOCH_UU="$2"
            shift 2
            ;;
        --n-epoch-up)
            N_EPOCH_UP="$2"
            shift 2
            ;;
        --n-epoch-train)
            N_EPOCH_TRAIN="$2"
            shift 2
            ;;
        --uu-threshold)
            UU_THRESHOLD="$2"
            shift 2
            ;;
        --up-threshold)
            UP_THRESHOLD="$2"
            shift 2
            ;;
        --graph-dir)
            GRAPH_DIR="$2"
            shift 2
            ;;
        --result-dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        --skip-stages)
            SKIP_STAGES="$2"
            shift 2
            ;;
        --quiet)
            VERBOSE=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to check if stage should be skipped
should_skip_stage() {
    local stage=$1
    if [[ $SKIP_STAGES == *"$stage"* ]]; then
        return 0  # True - should skip
    else
        return 1  # False - should not skip
    fi
}

# Function to run command or show dry-run
run_cmd() {
    if [ $DRY_RUN -eq 1 ]; then
        print_info "[DRY RUN] Would execute: $*"
    else
        if [ $VERBOSE -eq 1 ]; then
            print_info "Executing: $*"
        fi
        "$@"
        if [ $? -eq 0 ]; then
            if [ $VERBOSE -eq 1 ]; then
                print_success "Command completed successfully"
            fi
        else
            print_error "Command failed with exit code $?"
            exit 1
        fi
    fi
}

# Check if conda environment is activated
check_environment() {
    print_info "Checking Python environment..."

    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3."
        exit 1
    fi

    # Check if conda environment is already activated
    if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
        print_info "Using active conda environment: $CONDA_DEFAULT_ENV"
    else
        # Try to activate conda environment
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
            # Try fincgan-gpu first, fall back to fincgan
            if conda env list | grep -q "fincgan-gpu"; then
                print_info "Activating fincgan-gpu conda environment..."
                conda activate fincgan-gpu
                print_success "GPU environment activated"
            elif conda env list | grep -q "fincgan"; then
                print_info "Activating fincgan conda environment..."
                conda activate fincgan
                print_success "Environment activated"
            else
                print_warning "fincgan environment not found, using current environment"
            fi
        fi
    fi

    # Verify required modules
    python3 -c "import torch, dgl, numpy, sklearn" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Required Python modules not found. Please check your environment."
        exit 1
    fi
    print_success "All required modules found"
}

# Create output directories
create_directories() {
    print_info "Creating output directories..."
    run_cmd mkdir -p "$GRAPH_DIR"
    run_cmd mkdir -p "$RESULT_DIR"
    run_cmd mkdir -p "embed/"
    run_cmd mkdir -p "generator/"
    run_cmd mkdir -p "tsne/"
    run_cmd mkdir -p "tmp/"
    run_cmd mkdir -p "figures/"
    print_success "Directories created"
}

# Display configuration
display_config() {
    print_header "FincGAN Training Configuration"
    cat << EOF
  GPU ID              : $GPU_ID
  Ratio               : $RATIO
  Seeds               : $SEED_START to $SEED_END

  Epochs:
    - Embedding       : $N_EPOCH_EMBEDDING
    - GAN Training    : $N_EPOCH_GAN
    - UU Generator    : $N_EPOCH_UU
    - UP Generator    : $N_EPOCH_UP
    - Final Training  : $N_EPOCH_TRAIN

  Thresholds:
    - UU Generator    : $UU_THRESHOLD
    - UP Generator    : $UP_THRESHOLD

  Output Directories:
    - Graphs          : $GRAPH_DIR
    - Results         : $RESULT_DIR

  Skip Stages         : ${SKIP_STAGES:-None}
  Verbose             : $([ $VERBOSE -eq 1 ] && echo "Yes" || echo "No")
  Dry Run             : $([ $DRY_RUN -eq 1 ] && echo "Yes" || echo "No")

EOF

    if [ $DRY_RUN -eq 1 ]; then
        print_warning "DRY RUN MODE - No commands will be executed"
    fi
}

# Stage I: Feature Extraction
stage_embedding() {
    if should_skip_stage "1"; then
        print_warning "Skipping Stage I (Feature Extraction)"
        return
    fi

    print_stage "I" "Feature Extraction"
    print_info "Training HGT model to extract node embeddings..."

    run_cmd python3 -m fincgan.train \
        --gpu_id "$GPU_ID" \
        --n_epoch "$N_EPOCH_EMBEDDING" \
        --seed "$SEED_START" "$SEED_END" \
        --ratio "$RATIO" \
        --setting "embedding" \
        --verbose "$VERBOSE" \
        --graph_dir "$GRAPH_DIR" \
        --result_dir "$RESULT_DIR"

    # Check if embeddings were created
    if [ ! -f "embed/music_hgt_user_emb.pt" ] || [ ! -f "embed/music_hgt_product_emb.pt" ]; then
        print_error "Embeddings not found after Stage I"
        exit 1
    fi

    print_success "Stage I completed - Embeddings saved to embed/"
}

# Stage II: Node Generator
stage_node_generator() {
    if should_skip_stage "2"; then
        print_warning "Skipping Stage II (Node Generator)"
        return
    fi

    print_stage "II" "Node Generator (GAN) Training"
    print_info "Training GAN to generate synthetic user nodes..."

    run_cmd python3 -m fincgan.node_generator \
        --gpu_id "$GPU_ID" \
        --n_epochs "$N_EPOCH_GAN" \
        --gan_dir "generator/" \
        --gan_verbose "$VERBOSE" \
        --tsne_dir "tsne/" \
        --tsne_verbose "$VERBOSE"

    # Check if GAN models were created
    if [ ! -f "generator/music_D.pt" ] || [ ! -f "generator/music_G.pt" ]; then
        print_error "GAN models not found after Stage II"
        exit 1
    fi

    print_success "Stage II completed - GAN models saved to generator/"

    if [ -f "tsne/tsne.jpg" ]; then
        print_info "t-SNE visualizations saved to tsne/"
    fi
}

# Stage III: Edge Generators
stage_edge_generators() {
    if should_skip_stage "3"; then
        print_warning "Skipping Stage III (Edge Generators)"
        return
    fi

    print_stage "III" "Edge Generator Training"

    # User-User Edge Generator
    print_info "Training User-User edge generator..."
    run_cmd python3 -m fincgan.edge_generator_uu \
        --gpu_id "$GPU_ID" \
        --n_epoch "$N_EPOCH_UU" \
        --edge_dir "generator/" \
        --edge_generator_verbose "$VERBOSE"

    if [ ! -f "generator/uu_generator.pt" ]; then
        print_error "UU generator not found after training"
        exit 1
    fi
    print_success "User-User edge generator trained"

    # User-Product Edge Generator
    print_info "Training User-Product edge generator..."
    run_cmd python3 -m fincgan.edge_generator_up \
        --gpu_id "$GPU_ID" \
        --n_epoch "$N_EPOCH_UP" \
        --edge_dir "generator/" \
        --edge_generator_verbose "$VERBOSE"

    if [ ! -f "generator/up_generator.pt" ]; then
        print_error "UP generator not found after training"
        exit 1
    fi
    print_success "User-Product edge generator trained"

    print_success "Stage III completed - Edge generators saved to generator/"
}

# Stage IV: Graph Generation and Training
stage_graph_generation() {
    if should_skip_stage "4"; then
        print_warning "Skipping Stage IV (Graph Generation and Training)"
        return
    fi

    print_stage "IV" "Graph Generation and Training"
    print_info "Generating augmented graph using FincGAN..."

    run_cmd python3 -m fincgan.graph_generator \
        --ratio "$RATIO" \
        --up "$UP_THRESHOLD" \
        --uu "$UU_THRESHOLD" \
        --graph_dir "$GRAPH_DIR" \
        --verbose "$VERBOSE"

    print_success "Graph generated successfully"

    print_info "Training HGT model on augmented graph..."
    run_cmd python3 -m fincgan.train \
        --gpu_id "$GPU_ID" \
        --n_epoch "$N_EPOCH_TRAIN" \
        --seed "$SEED_START" "$SEED_END" \
        --ratio "$RATIO" \
        --setting "gan" \
        --verbose "$VERBOSE" \
        --graph_dir "$GRAPH_DIR" \
        --result_dir "$RESULT_DIR"

    print_success "Stage IV completed - Training results saved to $RESULT_DIR"
}

# Stage V: Visualization
stage_visualization() {
    if should_skip_stage "5"; then
        print_warning "Skipping Stage V (Visualization)"
        return
    fi

    print_stage "V" "Result Visualization"
    print_info "Generating result visualizations..."

    run_cmd python3 -c "from fincgan.visualize import auto_plot_figure_3; auto_plot_figure_3(result_dir='$RESULT_DIR', save_fig=True)"

    if [ -f "figures/figure_3.png" ]; then
        print_success "Stage V completed - Visualization saved to figures/figure_3.png"
    else
        print_warning "Visualization file not found, but stage completed"
    fi
}

# Display final results
display_results() {
    print_header "Training Results Summary"

    if [ -f "$RESULT_DIR/music_hgt_model_gan.txt" ]; then
        print_info "FincGAN Results:"
        echo ""
        python3 << EOF
import pandas as pd
try:
    df = pd.read_csv("$RESULT_DIR/music_hgt_model_gan.txt")
    if len(df) > 0:
        print("  Method: FincGAN")
        print("  Ratio: {:.4f}".format(df['ratio'].iloc[0]))
        print("  AUC-PRC: {:.4f}".format(df['prc'].iloc[0]))
        print("  AUC-ROC: {:.4f}".format(df['roc'].iloc[0]))
        print("  F1-Score: {:.4f}".format(df['f1'].iloc[0]))
        print("  Precision: {:.4f}".format(df['precision'].iloc[0]))
        print("  Recall: {:.4f}".format(df['recall'].iloc[0]))
        print("  Accuracy: {:.4f}".format(df['acc'].iloc[0]))
    else:
        print("  No results found in file")
except Exception as e:
    print(f"  Error reading results: {e}")
EOF
        echo ""
    else
        print_warning "Results file not found: $RESULT_DIR/music_hgt_model_gan.txt"
    fi

    print_info "Generated Files:"
    echo ""
    echo "  Embeddings:"
    [ -f "embed/music_hgt_user_emb.pt" ] && echo "    ✓ embed/music_hgt_user_emb.pt"
    [ -f "embed/music_hgt_product_emb.pt" ] && echo "    ✓ embed/music_hgt_product_emb.pt"
    echo ""
    echo "  GAN Models:"
    [ -f "generator/music_D.pt" ] && echo "    ✓ generator/music_D.pt"
    [ -f "generator/music_G.pt" ] && echo "    ✓ generator/music_G.pt"
    echo ""
    echo "  Edge Generators:"
    [ -f "generator/uu_generator.pt" ] && echo "    ✓ generator/uu_generator.pt"
    [ -f "generator/up_generator.pt" ] && echo "    ✓ generator/up_generator.pt"
    echo ""
    echo "  Visualizations:"
    [ -f "figures/figure_3.png" ] && echo "    ✓ figures/figure_3.png"
    [ -d "tsne/" ] && [ "$(ls -A tsne/)" ] && echo "    ✓ tsne/*.jpg ($(ls tsne/*.jpg 2>/dev/null | wc -l) files)"
    echo ""
}

# Main execution
main() {
    local start_time=$(date +%s)

    print_header "FincGAN Automated Training Pipeline"

    # Setup
    check_environment
    create_directories
    display_config

    if [ $DRY_RUN -eq 1 ]; then
        print_info "Dry run completed - no commands were executed"
        exit 0
    fi

    # Execute stages
    stage_embedding
    stage_node_generator
    stage_edge_generators
    stage_graph_generation
    stage_visualization

    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))

    display_results

    print_header "Pipeline Completed Successfully! 🎉"
    printf "  Total Time: %02d:%02d:%02d\n\n" $hours $minutes $seconds

    print_info "Next steps:"
    echo "  - View results in: $RESULT_DIR"
    echo "  - View visualizations in: figures/"
    echo "  - View t-SNE plots in: tsne/"
    echo ""
}

# Run main function
main "$@"
