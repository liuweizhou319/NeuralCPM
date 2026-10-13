#!/usr/bin/env bash
# Angle generalization for given anchors (face ids + target vectors): the
# anchors are rotated by an angle theta in their tangent planes, theta is
# sampled in [0, 2*pi) at every epoch and given to the network.  One model
# then gives the vector field for any angle; after training the field is
# exported for EXPORT_K equally spaced angles.
#
# Usage:
#   bash run_generalization.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_generalization.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Mesh and anchors.  Other pairs in data/:
#   ARMADILLO.obj + ARMADILLO_constraints_v0.npz
#   78481root_5w.obj + 78481root_5w_constraints_v3.npz
#   99276_thin_star_2.obj + 99276_thin_star_2_constraints.npz
#   37384.obj + 37384_constraints_v0.npz
MESH="$ROOT/data/botijo.obj"
CONSTRAINTS="$ROOT/data/botijo_constraints_v0.npz"
OUT_DIR="$ROOT/results/generalization/botijo"

# Number of equally spaced angles in [0, 2*pi) exported after training
EXPORT_K=128

# Sparse SDF grid
BSP_RES=500
PADDING=0.15
FD_HALO=2

# Network
HIDDEN=128
LR=1e-3

# Training
NUM_EPOCHS=20000
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

# Train with a random angle per epoch (no --theta), then export the fields to
# OUT_DIR/vecfield_000.npz ... and the checkpoint to OUT_DIR/model.pt.
"$PYTHON" "$ROOT/train_vecfield.py" "$MESH" \
    --constraints "$CONSTRAINTS" \
    -k "$EXPORT_K" \
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

echo "MODEL:  $OUT_DIR/model.pt"
echo "FIELDS: $OUT_DIR/vecfield_000.npz ... ($EXPORT_K angles)"
