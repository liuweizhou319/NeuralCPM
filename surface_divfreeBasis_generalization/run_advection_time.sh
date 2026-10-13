#!/usr/bin/env bash
# Density advection on a sequence of shapes with the basis trained by
# run_train_basis_time.sh (surface_basis_advection_time.py).
#
# Usage:
#   bash run_advection_time.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_advection_time.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Model trained by run_train_basis_time.sh
MODEL="$ROOT/results/hand_2_top60/model.pt"

# Sequence of meshes and normalization: the same as for training
OBJ_DIR="$ROOT/data/hand_2_top60"
OBJ_PATTERN="*.obj"
NORMALIZE="shared"

OUT="$ROOT/results/hand_2_top60/advection_time.npz"

# Number of basis functions used (empty: all trained modes)
NUM_BASIS=""
RIDGE=1e-6

# Initial density blob: its centre is the vertex CENTER_VERTEX, or the point
# CENTER="x,y,z" when CENTER is not empty
CENTER=""
CENTER_VERTEX=1000
RADIUS=0.05

# Initial tangent velocity: direction "x,y,z" (empty: random) and speed
DIRECTION="0,-1,0"
SPEED=1.0

# 1 = re-inject the density blob and the initial velocity at every step
CONTINUOUS_SOURCES=1

# Time step of a semi-Lagrangian step, number of steps on each mesh of the
# sequence (density and coefficients are kept when the mesh changes), and the
# index of the first mesh
DT=0.0005
STEPS_PER_FRAME=100
START_FRAME=0

SAVE_EVERY=100
LOG_EVERY=100
SEED=0

# ===========================================================

mkdir -p "$(dirname "$OUT")"

ARGS=(
  "$MODEL"
  --obj-dir "$OBJ_DIR"
  --obj-pattern "$OBJ_PATTERN"
  --normalize "$NORMALIZE"
  --ridge "$RIDGE"
  --radius "$RADIUS"
  --speed "$SPEED"
  --dt "$DT"
  --steps-per-frame "$STEPS_PER_FRAME"
  --start-frame "$START_FRAME"
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --seed "$SEED"
  --out "$OUT"
)
if [[ -n "$NUM_BASIS" ]]; then
  ARGS+=(--num-basis "$NUM_BASIS")
fi
if [[ -n "$CENTER" ]]; then
  ARGS+=(--center "$CENTER")
else
  ARGS+=(--center-vertex "$CENTER_VERTEX")
fi
if [[ -n "$DIRECTION" ]]; then
  ARGS+=(--direction "$DIRECTION")
fi
if [[ "$CONTINUOUS_SOURCES" -eq 1 ]]; then
  ARGS+=(--continuous-sources)
fi

"$PYTHON" "$ROOT/surface_basis_advection_time.py" "${ARGS[@]}"

echo "OUT: $OUT"
echo "View: python vis_advection_time.py $OUT"
