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
#   fandisk, vase_line, gargoyle, Bunny
MODEL_NAME="rockerarm"
MESH="$ROOT/data/${MODEL_NAME}.obj"
OUT_DIR="$ROOT/results/${MODEL_NAME}"
OUTPUT_NPZ="${MODEL_NAME}.npz"     # face-center field, written to OUT_DIR

# Sparse SDF grid
SDF_RES=500

# Smoothness
SMOOTH_P=2.0
SURFACE_PLACEMENT="per_face_adj"   # random | per_face_adj
REF_MODE="axis_swap"               # normal | principal | axis_swap

# Optimizer
LR_SCHEDULE="fixed"                # fixed | cosine

# Feature curves (sharp edges found with libigl, dihedral angle in degrees)
EXTRACT_FEATURE_CURVES=1           # 0 | 1
FEAT_ANGLE=30.0

# Feature alignment
FEAT_ALIGN_WEIGHT="abs_abs_diff"   # k1_minus_k2 | abs_abs_diff
CURV_WEIGHT_MODE="scale01"         # none | scale01 | bottom_floor (needs abs_abs_diff)
FEAT_EDGE_TARGET="mincurv"         # mincurv: direction of minimal curvature
                                   # tangent: tangent of the feature curve

# Schedule of the feature weight.  fixed: constant.  anneal: 0.2x -> 1x over
# the first PLATEAU_START of the training, 1x until PLATEAU_END, then a cosine
# decay to 0.2x (0 < start < end < 1).
FEAT_WEIGHT_SCHEDULE="anneal"      # fixed | anneal
FEAT_SCHEDULE_PLATEAU_START=0.2
FEAT_SCHEDULE_PLATEAU_END=0.8

# Closest-point term: its weight is 0 for the first CPM_RAMP_HOLD_FRAC of the
# training and then rises linearly to its final value (in [0, 1))
CPM_RAMP_HOLD_FRAC=0.99

# ===========================================================

mkdir -p "$OUT_DIR"

ARGS=(
  "$MESH"
  --sdf-res "$SDF_RES"
  --smooth-p "$SMOOTH_P"
  --surface-placement "$SURFACE_PLACEMENT"
  --ref-mode "$REF_MODE"
  --lr-schedule "$LR_SCHEDULE"
  --feat-angle "$FEAT_ANGLE"
  --feat-align-weight "$FEAT_ALIGN_WEIGHT"
  --curv-weight-mode "$CURV_WEIGHT_MODE"
  --feat-edge-target "$FEAT_EDGE_TARGET"
  --feat-weight-schedule "$FEAT_WEIGHT_SCHEDULE"
  --feat-schedule-plateau-start "$FEAT_SCHEDULE_PLATEAU_START"
  --feat-schedule-plateau-end "$FEAT_SCHEDULE_PLATEAU_END"
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
