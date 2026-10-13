#!/usr/bin/env bash
# SDF-resolution study on the fandisk at two mesh resolutions.
# For every (SDF_RES, FD_HALO) pair: train with given sources and Dirichlet
# values, compare with the FEM reference, and record the error together with
# the voxel spacing of the sparse SDF grid.
#
# Usage:
#   bash run_sdf_sweep.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_sdf_sweep.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# "SDF_RES,FD_HALO" pairs.  The voxel spacing of the sparse SDF grid is
#   voxel_size = (1 + 2*SDF_PADDING) * L_max / (SDF_RES - 1)
# with L_max = 1.377 for both fandisk meshes, e.g.
#   SDF_RES   20     50     100    200     300     400
#   spacing   0.094  0.037  0.018  0.0090  0.0060  0.0045
# FD_HALO voxel layers are kept around the surface.

# fandisk.obj: 7,229 vertices
FANDISK_PAIRS=(20,3 30,3 40,3 50,3 60,3 70,4 90,4 100,4 150,5 200,6 250,7 300,8)

# fandisk_5w.obj: 28,617 vertices
FANDISK_5W_PAIRS=(20,3 35,3 50,3 200,6 250,7 300,8 400,2)

SDF_PADDING=0.15
OUT_ROOT="$ROOT/results/sdf_sweep"

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

# ===========================================================

echo "LAMBDA_DIR=$LAMBDA_DIR  LAMBDA_MASS=$LAMBDA_MASS  LAMBDA_DATA=$LAMBDA_DATA  LAMBDA_CPM=$LAMBDA_CPM"

# apply_lambda_data SRC DST
# Apply LAMBDA_DATA to every source weight (NN and FEM must use the same file).
apply_lambda_data() {
"$PYTHON" - <<PY
import json
from pathlib import Path
src = Path(r"$1")
dst = Path(r"$2")
w = float("$LAMBDA_DATA")
data = json.loads(src.read_text(encoding="utf-8"))
if not isinstance(data, list):
    raise SystemExit("constraints JSON must be a list")
for item in data:
    item["weight"] = w
dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"wrote {dst} with weight={w} for {len(data)} sources")
PY
}

# run_sweep NAME MESH SOURCES PAIR...
run_sweep() {
  local name="$1" mesh="$2"
  local dir="$OUT_ROOT/$name"
  local sources="$dir/constraints_lambda_data.json"
  local fem_npz="$dir/fem_diffusion.npz"
  local summary="$dir/summary.txt"
  mkdir -p "$dir"
  apply_lambda_data "$3" "$sources"
  shift 3
  : > "$summary"

  # FEM reference: one solve per mesh, it does not depend on the SDF grid.
  "$PYTHON" "$ROOT/fem_screen.py" "$mesh" \
      --source-spec "$sources" \
      --lambda-dir "$LAMBDA_DIR" \
      --lambda-mass "$LAMBDA_MASS" \
      --out "$fem_npz"

  local pair res halo out voxel
  for pair in "$@"; do
    res="${pair%,*}"
    halo="${pair#*,}"
    out="$dir/res${res}_halo${halo}"
    mkdir -p "$out"

    # 1. Voxel spacing of this grid (third column of the sdf_spacing.py table).
    voxel="$("$PYTHON" "$ROOT/sdf_spacing.py" "$mesh" --sdf-res "$res" \
                 --fd-halo "$halo" --sdf-padding "$SDF_PADDING" --tsv \
             | tail -n 1 | cut -f 3)"
    echo "==> $name  SDF_RES=$res  FD_HALO=$halo  voxel_size=$voxel"

    # 2. Train on the sparse SDF grid of this resolution.  A pair that is
    #    already trained is not repeated (delete its folder to redo it).
    if [[ ! -f "$out/diffusion.npz" ]]; then
      "$PYTHON" "$ROOT/train_screen.py" "$mesh" \
          --source-spec "$sources" \
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
          --sdf-res "$res" \
          --sdf-padding "$SDF_PADDING" \
          --fd-halo "$halo" \
          --seed "$SEED" \
          --out-dir "$out"
    fi

    # 3. Error of the network at the mesh vertices.
    "$PYTHON" "$ROOT/compare_nn_fem_mse.py" "$out/diffusion.npz" "$fem_npz" \
        -o "$out/nn_vs_fem_mse.txt"

    # 4. Append "SDF_RES,FD_HALO", the spacing and the errors to the summary.
    {
      echo "$pair"
      echo "voxel_size: $voxel"
      grep -E '^(mse|rmse|rel_l2|mean_abs_error|max_abs_error):' "$out/nn_vs_fem_mse.txt"
      echo
    } >> "$summary"
  done

  # rel_l2 versus SDF_RES
  "$PYTHON" "$ROOT/plot_sdf_sweep.py" "$summary"
  echo "SUMMARY: $summary"
}

run_sweep fandisk    "$ROOT/data/fandisk.obj"    "$ROOT/data/fandisk_4v.json"    "${FANDISK_PAIRS[@]}"
run_sweep fandisk_5w "$ROOT/data/fandisk_5w.obj" "$ROOT/data/fandisk_5w_4v.json" "${FANDISK_5W_PAIRS[@]}"
