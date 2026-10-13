#!/usr/bin/env bash
# Divergence-free vector Laplacian eigenbasis on a family of shapes
# (generalization over the shape space): one model is trained for a sequence
# of meshes, the position of a mesh in the sequence is the shape code in
# [0, 1] given to the networks.  After training the basis is exported for
# every mesh of the sequence.
#
# Usage:
#   bash run_train_basis_time.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_train_basis_time.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Sequence of meshes (sorted by file name; same vertices and faces in every
# file) and output
OBJ_DIR="$ROOT/data/hand_2_top60"
OBJ_PATTERN="*.obj"
OUT_DIR="$ROOT/results/hand_2_top60"

# shared: one centre and scale for all meshes of the sequence; none: as given
NORMALIZE="shared"

# Number of basis functions
K=50

# Network
INPUT_SIZE=3
HIDDEN=128
LR=1e-3

# Training (one random mesh of the sequence per iteration)
NUM_EPOCHS=15000
NUM_SURFACE=20000
NUM_CPM=10000
RIDGE=1e-8
EPS=1e-12
CLIP_GRAD=1.0
LOG_EVERY=50
SAVE_EVERY=1000

# Loss weights
LAMBDA_ORTHO=1e3
LAMBDA_NORM=1000
LAMBDA_MEAN=1.0
LAMBDA_CPM=1000

EXPORT_BATCH_SIZE=4096

# ===========================================================

mkdir -p "$OUT_DIR"

# Train the K modes over the sequence, then export frame_0000/basis_000.npz ...
# for every mesh and model.pt to OUT_DIR.
"$PYTHON" "$ROOT/train_basis_time.py" \
    --source obj-seq \
    --obj-dir "$OBJ_DIR" \
    --obj-pattern "$OBJ_PATTERN" \
    --normalize "$NORMALIZE" \
    -k "$K" \
    --input-size "$INPUT_SIZE" \
    --hidden "$HIDDEN" \
    --lr "$LR" \
    --num-epochs "$NUM_EPOCHS" \
    --num-surface "$NUM_SURFACE" \
    --num-cpm "$NUM_CPM" \
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
echo "BASIS: $OUT_DIR/frame_0000/basis_000.npz ... ($K modes for every mesh)"
