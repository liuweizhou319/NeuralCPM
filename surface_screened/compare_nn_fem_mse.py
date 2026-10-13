#!/usr/bin/env python3
"""Error metrics between the network field and the FEM reference of one run.

Step 3 of the single-run pipeline (train_screen.py -> fem_screen.py -> this
script).  Both inputs are npz exports on the same mesh.  The FIRST positional
argument is the network export (diffusion.npz written by train_screen.py),
the SECOND is the FEM reference (written by fem_screen.py).  The script checks
that the two files share the mesh (same array shapes, vertices equal within
rtol=1e-5 / atol=1e-6, faces identical) and then compares the per-vertex
field (--field, default u_vertex).  The vertex tolerance is needed because the
network side normalizes the mesh in float32 and the FEM side in float64, so
the stored vertex positions differ by about 1e-7.

Metrics
-------
With err_i = u_nn(v_i) - u_fem(v_i) at the N mesh vertices v_i:

    mse            = (1/N) * sum_i err_i^2
    rmse           = sqrt(mse)
    rel_l2         = sqrt(sum_i err_i^2 / sum_i u_fem(v_i)^2)
    mean_abs_error = (1/N) * sum_i |err_i|
    max_abs_error  = max_i |err_i|

All of them are plain (unweighted) means / sums over the mesh vertices, not
area-weighted surface integrals.  rel_l2 is normalized by the field of the
SECOND file, so the FEM reference must be the second argument; rel_l2 is nan
if that field is identically zero.

Output
------
Nine "key: value" lines, printed to stdout and, with -o, also written to a
text file:

    nn_npz, fem_npz        resolved paths of the two inputs
    field, n_vertices      compared npz key and number of vertices
    mse, rmse, rel_l2, mean_abs_error, max_abs_error
                           the metrics above, with 17 significant digits

Other scripts read the metric lines by their "key:" prefix, so the line format
must not change.

Usage (from the repository root)
--------------------------------
The paths below continue the train_screen.py usage example (--out-dir
results/single/fandisk).  run_single.sh writes by default to
results/single/fandisk_res200_halo6 instead.

    python compare_nn_fem_mse.py results/single/fandisk/diffusion.npz results/single/fandisk/fem_diffusion.npz -o results/single/fandisk/nn_vs_fem_mse.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    """Command-line arguments: the two npz files, the field and the report path."""
    p = argparse.ArgumentParser()
    p.add_argument("nn_npz", type=Path, help="NN export npz (diffusion.npz)")
    p.add_argument("fem_npz", type=Path, help="FEM export npz")
    p.add_argument("--field", default="u_vertex")
    p.add_argument("-o", "--output", type=Path, default=None, help="optional txt report")
    return p.parse_args()


def load_u(path: Path, field: str):
    """Return (vertices float64, faces int64, flattened float64 field) of an npz."""
    with np.load(path) as data:
        v = np.asarray(data["mesh_vertices"], dtype=np.float64)
        f = np.asarray(data["mesh_faces"], dtype=np.int64)
        u = np.asarray(data[field], dtype=np.float64).reshape(-1)
    if len(u) != len(v):
        raise ValueError(f"{path}: {field} length {len(u)} != n_verts {len(v)}")
    return v, f, u


def main():
    """Check that both files share the mesh, then print / write the metrics."""
    args = parse_args()
    v1, f1, u_nn = load_u(args.nn_npz, args.field)
    v2, f2, u_fem = load_u(args.fem_npz, args.field)
    if v1.shape != v2.shape or f1.shape != f2.shape:
        raise ValueError("mesh shape mismatch between NN and FEM npz")
    # Loose tolerance on purpose: float32 (network) vs float64 (FEM) mesh
    # normalization differ by about 1e-7.
    if not np.allclose(v1, v2, rtol=1e-5, atol=1e-6):
        raise ValueError("mesh vertices differ between NN and FEM npz")
    if not np.array_equal(f1, f2):
        raise ValueError("mesh faces differ between NN and FEM npz")

    # Plain vertex means / sums (not area-weighted); the FEM field is the
    # rel_l2 denominator.
    err = u_nn - u_fem
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    mx = float(np.max(np.abs(err)))
    den = float(np.sum(u_fem ** 2))
    rel_l2 = (
        float(np.sqrt(np.sum(err ** 2) / den)) if den > 0.0 else float("nan")
    )

    # The 'key: value' layout below is parsed by other scripts; keep it.
    lines = [
        f"nn_npz: {args.nn_npz.resolve()}",
        f"fem_npz: {args.fem_npz.resolve()}",
        f"field: {args.field}",
        f"n_vertices: {len(v1)}",
        f"mse: {mse:.17g}",
        f"rmse: {rmse:.17g}",
        f"rel_l2: {rel_l2:.17g}",
        f"mean_abs_error: {mae:.17g}",
        f"max_abs_error: {mx:.17g}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
