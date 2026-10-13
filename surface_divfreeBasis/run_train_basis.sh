#!/usr/bin/env bash
# Divergence-free vector Laplacian eigenbasis on one mesh: train K basis
# functions (a single model for this shape) and export them.
#
# Usage:
#   bash run_train_basis.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_train_basis.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Mesh and output.  Other meshes in data/:
#   bimba.obj, eight_v2.obj, ARMADILLO21622.obj
MESH="$ROOT/data/Bunny.obj"
OUT_DIR="$ROOT/results/Bunny"

# Number of basis functions
K=60

# Sparse SDF grid
BSP_RES=500
PADDING=0.15
FD_HALO=2

# Network
INPUT_SIZE=3
HIDDEN=128
LR=1e-3

# Training
NUM_EPOCHS=10000
NUM_SURFACE=20000
NUM_NARROWBAND=20000
BAND_WIDTH=0.01
CODE_VALUE=0.5
RIDGE=1e-8
EPS=1e-12
CLIP_GRAD=1.0
LOG_EVERY=100
SAVE_EVERY=1000

# Loss weights
LAMBDA_ORTHO=1e3
LAMBDA_NORM=1000.0
LAMBDA_MEAN=0.0
LAMBDA_CPM=10.0

EXPORT_BATCH_SIZE=4096

# ===========================================================

mkdir -p "$OUT_DIR"

# Train the K modes, then export basis_000.npz ... and model.pt to OUT_DIR.
"$PYTHON" "$ROOT/train_basis.py" "$MESH" \
    -k "$K" \
    --bsp-res "$BSP_RES" \
    --padding "$PADDING" \
    --fd-halo "$FD_HALO" \
    --input-size "$INPUT_SIZE" \
    --hidden "$HIDDEN" \
    --lr "$LR" \
    --num-epochs "$NUM_EPOCHS" \
    --num-surface "$NUM_SURFACE" \
    --num-narrowband "$NUM_NARROWBAND" \
    --band-width "$BAND_WIDTH" \
    --code-value "$CODE_VALUE" \
    --ridge "$RIDGE" \
    --eps "$EPS" \
    --clip-grad "$CLIP_GRAD" \
    --log-every "$LOG_EVERY" \
    --save-every "$SAVE_EVERY" \
    --lambda-ortho "$LAMBDA_ORTHO" \
    --lambda-norm "$LAMBDA_NORM" \
    --lambda-mean "$LAMBDA_MEAN" \
    --lambda-cpm "$LAMBDA_CPM" \
    --export-batch-size "$EXPORT_BATCH_SIZE" \
    --save-dir "$OUT_DIR/checkpoints" \
    --out-dir "$OUT_DIR"

echo "MODEL: $OUT_DIR/model.pt"
echo "BASIS: $OUT_DIR/basis_000.npz ... ($K modes)"
