#!/usr/bin/env bash
# Frame field on one mesh: build the sparse SDF, extract the
# feature curves, train the field and export it at the face centers.
#
# Usage:
#   bash run_train_frame_field.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_train_frame_field.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Mesh and output.  Other meshes in data/:
#   rockerarm, fandisk, vase_line, gargoyle
MODEL_NAME="Bunny"
MESH="$ROOT/data/${MODEL_NAME}.obj"
OUT_DIR="$ROOT/results/${MODEL_NAME}"
OUTPUT_NPZ="${MODEL_NAME}.npz"     # face-center field, written to OUT_DIR

# Sparse SDF grid
SDF_RES=500
SDF_PADDING=0.15
FD_HALO=4

# Network
HIDDEN=128
N_LAYERS=4

# Optimizer
LR_SCHEDULE="fixed"                # fixed: constant LR_FIXED | cosine: from LR to 1e-6
LR_FIXED=5e-5
LR=1e-3

# Training
NUM_EPOCHS=10000
CLIP_GRAD=1.0
LOG_EVERY=50
SAVE_EVERY=2000

# Surface samples of an epoch.  per_face_adj: one random point per face,
# smoothness between adjacent faces.  random: NUM_SURFACE random points,
# smoothness between the KNN_K nearest neighbors.
SURFACE_PLACEMENT="per_face_adj"   # random | per_face_adj
NUM_SURFACE=30000
KNN_K=6

# Samples around the surface for the closest-point term: NUM_NARROWBAND
# surface points moved by Gaussian noise of variance BAND_WIDTH
NUM_NARROWBAND=20000
BAND_WIDTH=0.015

# Loss weights
LAMBDA_SMOOTH=100.0                # smoothness
LAMBDA_FEAT=100.0                  # alignment with the curvature directions
LAMBDA_FEAT_EDGE=100.0             # alignment on the feature curves (per_face_adj only)
LAMBDA_CPM=1.0                   # closest-point term

# Smoothness energy and reference frame
SMOOTH_P=2.0                       # exponent of the smoothness energy
REF_MODE="axis_swap"               # normal | principal | axis_swap

# Feature curves (sharp edges found with libigl, dihedral angle in degrees)
EXTRACT_FEATURE_CURVES=1           # 0 | 1
FEAT_ANGLE=30.0

# Feature alignment
FEAT_ALIGN_WEIGHT="abs_abs_diff"   # k1_minus_k2 | abs_abs_diff
CURV_WEIGHT_MODE="scale01"         # none | scale01 | bottom_floor (needs abs_abs_diff)
FEAT_EDGE_TARGET="mincurv"         # mincurv: direction of minimal curvature
                                   # tangent: tangent of the feature curve

# Schedule of the two feature weights.  fixed: constant.  anneal: 0.2x -> 1x
# over the first PLATEAU_START of the training, 1x until PLATEAU_END, then a
# cosine decay to 0.2x (0 < start < end < 1).
FEAT_WEIGHT_SCHEDULE="anneal"      # fixed | anneal
FEAT_SCHEDULE_PLATEAU_START=0.2
FEAT_SCHEDULE_PLATEAU_END=0.8

# Schedule of the closest-point weight.  fixed: constant.  ramp: 0 for the
# first CPM_RAMP_HOLD_FRAC of the training (in [0, 1)), then a linear rise to
# LAMBDA_CPM.
CPM_SCHEDULE="ramp"                # fixed | ramp
CPM_RAMP_HOLD_FRAC=0.99

# ===========================================================

mkdir -p "$OUT_DIR"

ARGS=(
  "$MESH"
  --sdf-res "$SDF_RES"
  --sdf-padding "$SDF_PADDING"
  --fd-halo "$FD_HALO"
  --hidden "$HIDDEN"
  --n-layers "$N_LAYERS"
  --lr-schedule "$LR_SCHEDULE"
  --lr-fixed "$LR_FIXED"
  --lr "$LR"
  --num-epochs "$NUM_EPOCHS"
  --clip-grad "$CLIP_GRAD"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --surface-placement "$SURFACE_PLACEMENT"
  --num-surface "$NUM_SURFACE"
  --knn-k "$KNN_K"
  --num-narrowband "$NUM_NARROWBAND"
  --band-width "$BAND_WIDTH"
  --lambda-smooth "$LAMBDA_SMOOTH"
  --lambda-feat "$LAMBDA_FEAT"
  --lambda-feat-edge "$LAMBDA_FEAT_EDGE"
  --lambda-cpm "$LAMBDA_CPM"
  --smooth-p "$SMOOTH_P"
  --ref-mode "$REF_MODE"
  --feat-angle "$FEAT_ANGLE"
  --feat-align-weight "$FEAT_ALIGN_WEIGHT"
  --curv-weight-mode "$CURV_WEIGHT_MODE"
  --feat-edge-target "$FEAT_EDGE_TARGET"
  --feat-weight-schedule "$FEAT_WEIGHT_SCHEDULE"
  --feat-schedule-plateau-start "$FEAT_SCHEDULE_PLATEAU_START"
  --feat-schedule-plateau-end "$FEAT_SCHEDULE_PLATEAU_END"
  --cpm-schedule "$CPM_SCHEDULE"
  --cpm-ramp-hold-frac "$CPM_RAMP_HOLD_FRAC"
  --output-npz "$OUTPUT_NPZ"
  --out-dir "$OUT_DIR"
)
if [[ "$EXTRACT_FEATURE_CURVES" == "1" ]]; then
  ARGS+=(--extract-feature-curves)
fi

"$PYTHON" "$ROOT/train_frame_field.py" "${ARGS[@]}"

echo "MODEL: $OUT_DIR/model.pt"
echo "FIELD: $OUT_DIR/$OUTPUT_NPZ"
