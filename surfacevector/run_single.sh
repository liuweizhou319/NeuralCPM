#!/usr/bin/env bash
# Vector field design on one mesh for given anchors (face ids + target
# vectors) at one fixed rotation angle: train a single model and export the
# field.
#
# Usage:
#   bash run_single.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_single.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Mesh and anchors.  Other pairs in data/:
#   botijo.obj + botijo_constraints_v0.npz
#   ARMADILLO.obj + ARMADILLO_constraints_v0.npz
#   99276_thin_star_2.obj + 99276_thin_star_2_constraints.npz
#   37384.obj + 37384_constraints_v0.npz
MESH="$ROOT/data/78481root_5w.obj"
CONSTRAINTS="$ROOT/data/78481root_5w_constraints_v3.npz"
OUT_DIR="$ROOT/results/single/78481root_5w"

# Rotation angle of the anchors in their tangent planes (radians)
THETA=0

# Sparse SDF grid
BSP_RES=500
PADDING=0.15
FD_HALO=2

# Network
HIDDEN=128
LR=1e-3

# Training
NUM_EPOCHS=3000
NUM_SURFACE=30000
NUM_NARROWBAND=10000
BAND_WIDTH=0.01
CLIP_GRAD=1.0
LOG_EVERY=100
SAVE_EVERY=1000

# Loss weights
LAMBDA_DATA=100.0
LAMBDA_SMOOTH_PHI=1.0
LAMBDA_SMOOTH_PSI=1.0
LAMBDA_CPM=1.0
LAMBDA_MEAN=5.0
LAMBDA_ORTH=1.0

EXPORT_BATCH_SIZE=4096

# ===========================================================

mkdir -p "$OUT_DIR"

# Train with the fixed angle THETA, then export the field to
# OUT_DIR/vecfield_000.npz and the checkpoint to OUT_DIR/model.pt.
"$PYTHON" "$ROOT/train_vecfield.py" "$MESH" \
    --constraints "$CONSTRAINTS" \
    --theta "$THETA" \
    --bsp-res "$BSP_RES" \
    --padding "$PADDING" \
    --fd-halo "$FD_HALO" \
    --hidden "$HIDDEN" \
    --lr "$LR" \
    --num-epochs "$NUM_EPOCHS" \
    --num-surface "$NUM_SURFACE" \
    --num-narrowband "$NUM_NARROWBAND" \
    --band-width "$BAND_WIDTH" \
    --clip-grad "$CLIP_GRAD" \
    --log-every "$LOG_EVERY" \
    --save-every "$SAVE_EVERY" \
    --lambda-data "$LAMBDA_DATA" \
    --lambda-smooth-phi "$LAMBDA_SMOOTH_PHI" \
    --lambda-smooth-psi "$LAMBDA_SMOOTH_PSI" \
    --lambda-cpm "$LAMBDA_CPM" \
    --lambda-mean "$LAMBDA_MEAN" \
    --lambda-orth "$LAMBDA_ORTH" \
    --export-batch-size "$EXPORT_BATCH_SIZE" \
    --no-profile \
    --save-dir "$OUT_DIR/checkpoints" \
    --out-dir "$OUT_DIR"

echo "MODEL: $OUT_DIR/model.pt"
echo "FIELD: $OUT_DIR/vecfield_000.npz"
