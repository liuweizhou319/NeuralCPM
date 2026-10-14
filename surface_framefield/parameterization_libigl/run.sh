#!/usr/bin/env bash
# Run surfaceframefield_igl on a mesh and its per-face frame field.
#
# Usage:
#   ./run.sh                                         # use the defaults below
#   ./run.sh <mesh.obj> <framefield.txt> [output_uv.obj]
#
# A bare file name that does not exist relative to the current directory
# is looked up in input/.
set -euo pipefail

MESH="79279_baseline.obj"
FIELD="79279_baseline_framefield.txt"
OUTPUT=""  # empty: the program writes <mesh dir>/<mesh name>_uv.obj

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$ROOT/build/windows/x64/release/surfaceframefield_igl.exe"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

MESH="${1:-$MESH}"
FIELD="${2:-$FIELD}"
OUTPUT="${3:-$OUTPUT}"

resolve() {
  if [ -f "$1" ]; then
    printf '%s\n' "$1"
  elif [ -f "$ROOT/input/$1" ]; then
    printf '%s\n' "$ROOT/input/$1"
  else
    echo "File not found: $1" >&2
    exit 1
  fi
}

if [ ! -f "$EXE" ]; then
  echo "Executable not found: $EXE" >&2
  echo "Build it first: xmake f -m release && xmake" >&2
  exit 1
fi

MESH="$(resolve "$MESH")"
FIELD="$(resolve "$FIELD")"

args=("$MESH" "$FIELD")
if [ -n "$OUTPUT" ]; then
  args+=("$OUTPUT")
fi

exec "$EXE" "${args[@]}"
