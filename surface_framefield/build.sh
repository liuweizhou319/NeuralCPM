#!/usr/bin/env bash
# Initialize dependencies and build the parameterization and quad extraction tools.
# Usage: ./build.sh [additional CMake configure arguments]
# Example: JOBS=4 ./build.sh -DPOLYSCOPE_SOURCE_DIR=/path/to/polyscope
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-4}"

for tool in git cmake; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: $tool is required. Install it before running build.sh." >&2
        exit 1
    fi
done

cd "$ROOT"
git submodule update --init --recursive -- libQEx
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "$@"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --parallel "$JOBS"

echo "Build complete: $BUILD_DIR"
