#!/usr/bin/env bash
# Thin star: a single training (no generalization) with one curve source,
# followed by the FEM reference and the comparison.
#
# Usage:
#   bash run_thin_star.sh
#   SKIP_TRAIN=1 bash run_thin_star.sh       # only FEM + error (needs diffusion.npz)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Mesh and source (one geodesic circle with value 1)
MESH="$ROOT/data/99276_thin_star.obj"
SOURCES="$ROOT/data/99276_thin_star_1.json"

# Sparse SDF grid
SDF_RES=150
SDF_PADDING=0.15
FD_HALO=2

OUT_DIR="$ROOT/results/single/thin_star_res${SDF_RES}_halo${FD_HALO}"

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

# 1. Train the network.
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

# 2. FEM reference.
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
