#!/usr/bin/env bash
# Screened diffusion on one mesh with given sources and Dirichlet values:
# train the network, solve the FEM reference, compare the two.
#
# Usage:
#   bash run_single.sh
#   SDF_RES=300 FD_HALO=8 bash run_single.sh
#   SKIP_TRAIN=1 bash run_single.sh          # only FEM + error (needs diffusion.npz)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Mesh and sources (source positions and Dirichlet values).
# Other pairs in data/:
#   Bunny.obj + Bunny_2.json                    SDF_RES=300 FD_HALO=6
#   fandisk_5w.obj + fandisk_5w_4v.json         SDF_RES=200 FD_HALO=6
#   MechanicalPart_2.5w.obj + MechanicalPart_2.5w_constraints_v0.json
#                                               SDF_RES=70  FD_HALO=2  EPOCHS=6000
NAME="fandisk"
MESH="$ROOT/data/fandisk.obj"
SOURCES="$ROOT/data/fandisk_4v.json"

# Sparse SDF grid.  SDF_RES sets the voxel spacing:
#   voxel_size = (1 + 2*SDF_PADDING) * L_max / (SDF_RES - 1)
# (L_max = longest bounding-box edge of the normalised mesh).  FD_HALO is the
# number of voxel layers kept around the surface; raise it together with
# SDF_RES to keep a band of similar width.
SDF_RES="${SDF_RES:-200}"
SDF_PADDING=0.15
FD_HALO="${FD_HALO:-6}"

OUT_DIR="$ROOT/results/single/${NAME}_res${SDF_RES}_halo${FD_HALO}"

# Training
EPOCHS=10000
LR=1e-3
HIDDEN=128
NUM_SURFACE=20000
NUM_NARROWBAND=10000
NUM_PER_SOURCE=256
BAND_WIDTH=0.01
SEED=0

# Loss weights
LAMBDA_DIR=1.0
LAMBDA_MASS=1.0
LAMBDA_CPM=1.0
# Weight of the sources (overrides "weight" in the JSON for all sources)
LAMBDA_DATA=1.0

SKIP_TRAIN="${SKIP_TRAIN:-0}"

# ===========================================================

NN_NPZ="$OUT_DIR/diffusion.npz"
FEM_NPZ="$OUT_DIR/fem_diffusion.npz"
MSE_TXT="$OUT_DIR/nn_vs_fem_mse.txt"
SOURCES_RUN="$OUT_DIR/constraints_lambda_data.json"

mkdir -p "$OUT_DIR"

echo "LAMBDA_DIR=$LAMBDA_DIR  LAMBDA_MASS=$LAMBDA_MASS  LAMBDA_DATA=$LAMBDA_DATA  LAMBDA_CPM=$LAMBDA_CPM"

# Apply LAMBDA_DATA to every source weight (NN and FEM must use the same file).
"$PYTHON" - <<PY
import json
from pathlib import Path
src = Path(r"$SOURCES")
dst = Path(r"$SOURCES_RUN")
w = float("$LAMBDA_DATA")
data = json.loads(src.read_text(encoding="utf-8"))
if not isinstance(data, list):
    raise SystemExit("constraints JSON must be a list")
for item in data:
    item["weight"] = w
dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"wrote {dst} with weight={w} for {len(data)} sources")
PY

# 1. Train the network (fixed Dirichlet values from the JSON).
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  "$PYTHON" "$ROOT/train_screen.py" "$MESH" \
      --source-spec "$SOURCES_RUN" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --hidden "$HIDDEN" \
      --num-surface "$NUM_SURFACE" \
      --num-narrowband "$NUM_NARROWBAND" \
      --num-per-source "$NUM_PER_SOURCE" \
      --band-width "$BAND_WIDTH" \
      --lambda-dir "$LAMBDA_DIR" \
      --lambda-mass "$LAMBDA_MASS" \
      --lambda-cpm "$LAMBDA_CPM" \
      --sdf-res "$SDF_RES" \
      --sdf-padding "$SDF_PADDING" \
      --fd-halo "$FD_HALO" \
      --seed "$SEED" \
      --out-dir "$OUT_DIR"
fi

# 2. FEM reference (same mesh and sources; independent of the SDF grid).
"$PYTHON" "$ROOT/fem_screen.py" "$MESH" \
    --source-spec "$SOURCES_RUN" \
    --lambda-dir "$LAMBDA_DIR" \
    --lambda-mass "$LAMBDA_MASS" \
    --out "$FEM_NPZ"

# 3. Error of the network at the mesh vertices.
"$PYTHON" "$ROOT/compare_nn_fem_mse.py" "$NN_NPZ" "$FEM_NPZ" -o "$MSE_TXT"

echo "NN:  $NN_NPZ"
echo "FEM: $FEM_NPZ"
echo "MSE: $MSE_TXT"
