#!/usr/bin/env bash
# Quantitative evaluation of the Dirichlet-value generalization trained by
# run_generalization.sh.  The network is compared with one FEM solve per
# vector of Dirichlet values, on three sets of vectors:
#   train   vectors seen during training
#   interp  unseen vectors in [G_MIN, G_MAX]
#   extrap  vectors in [EXTRAP_MIN, EXTRAP_MAX] with at least one value
#           outside [G_MIN, G_MAX]
#
# Usage:
#   bash run_evaluation.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_evaluation.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

# ======================== parameters ========================

# Same as in run_generalization.sh
MESH="$ROOT/data/dc.obj"
MODEL_DIR="$ROOT/results/generalization/dc"
SOURCE_INDICES=(0 1 2 3 4)
G_MIN=-3
G_MAX=2
SDF_RES=300
SDF_PADDING=0.15
FD_HALO=2
LAMBDA_DIR=1.0
LAMBDA_MASS=1.0

# Evaluation
OUT_DIR="$MODEL_DIR/eval"
EXTRAP_MIN=-3.5
EXTRAP_MAX=2.5
NUM_TRAIN=2000
NUM_INTERP=2000
NUM_EXTRAP=2000
SEED=0
DEVICE="cuda"
NORMAL_SOURCE="sdf"

# ===========================================================

MODEL="$MODEL_DIR/model.pt"
TRAIN_CODES="$MODEL_DIR/train_codes.txt"
# Sources with the weights used for training (written by run_generalization.sh)
SOURCES="$MODEL_DIR/constraints_lambda_data.json"

mkdir -p "$OUT_DIR"

VALUE_SRC_ARGS=()
for idx in "${SOURCE_INDICES[@]}"; do
  VALUE_SRC_ARGS+=(--value-generalize-source "$idx")
done

# 1. Build the train / interp / extrap vectors of Dirichlet values.
"$PYTHON" "$ROOT/prepare_dirichlet_splits.py" \
    --train-codes "$TRAIN_CODES" \
    --out-dir "$OUT_DIR" \
    --g-min "$G_MIN" \
    --g-max "$G_MAX" \
    --extrap-min "$EXTRAP_MIN" \
    --extrap-max "$EXTRAP_MAX" \
    --inner-min "$G_MIN" \
    --inner-max "$G_MAX" \
    --num-train "$NUM_TRAIN" \
    --num-interp "$NUM_INTERP" \
    --num-extrap "$NUM_EXTRAP" \
    --seed "$SEED"

# 2. Network vs FEM for every vector of a set, then plot the errors.
#    run_eval NAME G_MIN G_MAX [extra arguments of the evaluation script]
run_eval() {
  local name="$1" g_min="$2" g_max="$3"
  shift 3
  local txt="$OUT_DIR/metrics_${name}.txt"

  "$PYTHON" "$ROOT/evaluate_dirichlet_generalization.py" \
      --mesh "$MESH" \
      --source-spec "$SOURCES" \
      --checkpoint "$MODEL" \
      "${VALUE_SRC_ARGS[@]}" \
      --codes-file "$OUT_DIR/codes_${name}.txt" \
      --g-min "$g_min" \
      --g-max "$g_max" \
      --seed "$SEED" \
      --lambda-dir "$LAMBDA_DIR" \
      --lambda-mass "$LAMBDA_MASS" \
      --device "$DEVICE" \
      --normal-source "$NORMAL_SOURCE" \
      --sdf-res "$SDF_RES" \
      --sdf-padding "$SDF_PADDING" \
      --fd-halo "$FD_HALO" \
      "$@" \
      --output "$txt"

  "$PYTHON" "$ROOT/plot_generalization_mse.py" "$txt" \
      --output "$OUT_DIR/mse_${name}.png" \
      --out-rel "$OUT_DIR/rel_l2_${name}.png" \
      --title "NN vs FEM: MSE ($name)" \
      --title-rel "NN vs FEM: rel_l2 ($name)"
}

run_eval train  "$G_MIN" "$G_MAX"
run_eval interp "$G_MIN" "$G_MAX"
run_eval extrap "$EXTRAP_MIN" "$EXTRAP_MAX" --allow-code-extrapolation

echo "OUT_DIR: $OUT_DIR"
echo "  metrics_train/interp/extrap.txt   (mse and rel_l2 per vector; mean in the header)"
echo "  mse_*.png, rel_l2_*.png"
