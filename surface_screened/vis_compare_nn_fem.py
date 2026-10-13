"""Polyscope viewer for the per-vertex difference between two exported fields.

Qualitative companion of compare_nn_fem_mse.py.  It loads two npz exports on
the same mesh, typically the network result (diffusion.npz written by
train_screen.py) and the FEM reference (written by fem_screen.py), prints
mean |error|, rmse and max |error| over the mesh vertices (plain vertex
statistics, as in compare_nn_fem_mse.py) and opens an interactive polyscope
window with four scalar quantities on the mesh:

    <stem of the first file>    field of the first file (shown at start)
    <stem of the second file>   field of the second file, same color range
    error                       first minus second, symmetric color range
    |error|                     absolute difference

The two files need different base names, because the first two quantities are
named after the file stems.

The two files must share the mesh: same array shapes, identical faces, and
vertices equal within rtol=1e-5 / atol=1e-6.  This is the tolerance of
compare_nn_fem_mse.py; it is needed because the network side normalizes the
mesh in float32 and the FEM side in float64, so the stored vertex positions
differ by about 1e-7.

The compared npz key is --field (default u_vertex).  No file is written.
Needs polyscope and a display.

Usage (from the repository root)
--------------------------------
The paths below continue the train_screen.py usage example (--out-dir
results/single/fandisk).  run_single.sh writes by default to
results/single/fandisk_res200_halo6 instead.

    python vis_compare_nn_fem.py results/single/fandisk/diffusion.npz results/single/fandisk/fem_diffusion.npz
"""

import argparse
from pathlib import Path
import numpy as np

def parse_args():
    """Command-line arguments: the two npz files and the field to compare."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_a")
    parser.add_argument("input_b")
    parser.add_argument("--field", default="u_vertex")
    return parser.parse_args()

def load_npz(path, field):
    """Return (vertices float64, faces int32, flattened float64 field) of an npz."""
    with np.load(path) as data:
        v = np.asarray(data["mesh_vertices"], dtype=np.float64)
        f = np.asarray(data["mesh_faces"], dtype=np.int32)
        u = np.asarray(data[field], dtype=np.float64).reshape(-1)
    if len(u) != len(v):
        raise ValueError(f"{path}: {field} length mismatch")
    return v, f, u

def main():
    """Print the error statistics and show both fields and their difference."""
    args = parse_args()
    path_a, path_b = Path(args.input_a), Path(args.input_b)
    v, f, ua = load_npz(path_a, args.field)
    v2, f2, ub = load_npz(path_b, args.field)
    # Same vertex tolerance as compare_nn_fem_mse.py: float32 (network) vs
    # float64 (FEM) mesh normalization differ by about 1e-7.
    if v.shape != v2.shape or f.shape != f2.shape or not np.allclose(v, v2, rtol=1e-5, atol=1e-6) or not np.array_equal(f, f2):
        raise ValueError("The two npz files do not share the same mesh.")

    err = ua - ub
    rng = (float(min(ua.min(), ub.min())), float(max(ua.max(), ub.max())))
    emax = max(float(np.abs(err).max()), 1e-8)
    print(f"mean(|error|)={np.mean(np.abs(err)):.6g}, rmse={np.sqrt(np.mean(err ** 2)):.6g}, max(|error|)={np.max(np.abs(err)):.6g}")

    import polyscope as ps
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    sm = ps.register_surface_mesh("mesh", v, f, smooth_shade=True)
    sm.add_scalar_quantity(path_a.stem, ua, cmap="viridis", vminmax=rng, enabled=True)
    sm.add_scalar_quantity(path_b.stem, ub, cmap="viridis", vminmax=rng, enabled=False)
    sm.add_scalar_quantity("error", err, cmap="coolwarm", vminmax=(-emax, emax), enabled=False)
    sm.add_scalar_quantity("|error|", np.abs(err), cmap="magma", enabled=False)
    ps.show()

if __name__ == "__main__":
    main()
