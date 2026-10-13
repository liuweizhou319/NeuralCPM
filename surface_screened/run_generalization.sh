#!/usr/bin/env bash
# Dirichlet-value generalization: the sources keep their positions, their
# Dirichlet values become inputs of the network and are sampled uniformly in
# [G_MIN, G_MAX] at every epoch.  This script trains the network;
# run_evaluation.sh evaluates it.
#
# Usage:
#   bash run_generalization.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_generalization.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Input and output.  For the bunny use
#   Bunny.obj + Bunny_2.json, SOURCE_INDICES=(0 1 2 3), G_MIN=-2, G_MAX=2,
#   SDF_RES=400, NUM_SURFACE=20000, NUM_NARROWBAND=10000
MESH="$ROOT/data/dc.obj"
SOURCES="$ROOT/data/dc_constraints_v0.json"
MODEL_DIR="$ROOT/results/generalization/dc"

# Sources whose Dirichlet value is generalized (indices into the JSON list,
# ascending) and the range the values are sampled from
SOURCE_INDICES=(0 1 2 3 4)
G_MIN=-3
G_MAX=2

# Sparse SDF grid
SDF_RES=300
SDF_PADDING=0.15
FD_HALO=2

# Training
EPOCHS=30000
LR=1e-3
HIDDEN=128
NUM_SURFACE=30000
NUM_NARROWBAND=5000
NUM_PER_SOURCE=256
BAND_WIDTH=0.01
SEED=0

# Loss weights
LAMBDA_DIR=1.0
LAMBDA_MASS=1.0
LAMBDA_CPM=1.0
# Weight of the sources (overrides "weight" in the JSON for all sources)
LAMBDA_DATA=1.0

# ===========================================================

TRAIN_CODES="$MODEL_DIR/train_codes.txt"
SOURCES_RUN="$MODEL_DIR/constraints_lambda_data.json"

mkdir -p "$MODEL_DIR"

echo "LAMBDA_DIR=$LAMBDA_DIR  LAMBDA_MASS=$LAMBDA_MASS  LAMBDA_DATA=$LAMBDA_DATA  LAMBDA_CPM=$LAMBDA_CPM"

# Apply LAMBDA_DATA to every source weight (NN and FEM must use the same file;
# run_evaluation.sh reads it from MODEL_DIR).
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

VALUE_SRC_ARGS=()
for idx in "${SOURCE_INDICES[@]}"; do
  VALUE_SRC_ARGS+=(--value-generalize-source "$idx")
done

# Train; the Dirichlet values used at every epoch are logged to train_codes.txt.
"$PYTHON" "$ROOT/train_screen.py" "$MESH" \
    --source-spec "$SOURCES_RUN" \
    "${VALUE_SRC_ARGS[@]}" \
    --value-min "$G_MIN" \
    --value-max "$G_MAX" \
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
    --out-dir "$MODEL_DIR" \
    --train-codes-out "$TRAIN_CODES"

echo "MODEL:       $MODEL_DIR/model.pt"
echo "TRAIN_CODES: $TRAIN_CODES"
