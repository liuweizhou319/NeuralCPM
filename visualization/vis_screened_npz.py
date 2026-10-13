#!/usr/bin/env python3
"""Visualize a screened-diffusion result npz with polyscope.

Shows the field u on the mesh together with the constraints stored in the
same file: point sources as spheres and curve sources as tubes, colored by
their Dirichlet value.  Works for diffusion.npz (network) and
fem_diffusion.npz (FEM reference).

Example:
  python vis_screened_npz.py
  python vis_screened_npz.py ../surface_screened/results/single/thin_star_res150_halo2/diffusion.npz
  python vis_screened_npz.py ../surface_screened/results/single/thin_star_res150_halo2/fem_diffusion.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Default: the result of surface_screened/run_thin_star.sh
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = ROOT / "surface_screened/results/single/thin_star_res150_halo2/diffusion.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path, nargs="?", default=DEFAULT_NPZ,
                   help="diffusion.npz or fem_diffusion.npz")
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.npz)

    V = data["mesh_vertices"]
    F = data["mesh_faces"]
    u = data["u_vertex"]
    pts, pts_val = data["src_pts"], data["src_pts_value"]
    p1, p2, seg_val = data["src_seg_p1"], data["src_seg_p2"], data["src_seg_value"]

    # One color range for the field and the constraint values, so the two can
    # be compared directly.
    values = np.concatenate([u, pts_val, seg_val])
    vminmax = (float(values.min()), float(values.max()))
    cmap = "viridis"
    if vminmax[0] < 0 < vminmax[1]:
        vabs = max(abs(vminmax[0]), abs(vminmax[1]))
        vminmax, cmap = (-vabs, vabs), "coolwarm"

    print(f"{args.npz}")
    print(f"  mesh: {len(V)} vertices, {len(F)} faces")
    print(f"  u in [{u.min():.4f}, {u.max():.4f}]")
    print(f"  constraints: {len(pts)} points {np.unique(pts_val)}, "
          f"{len(p1)} curve segments {np.unique(seg_val)}")

    import polyscope as ps

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Result
    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)
    sm.add_scalar_quantity("u", u, cmap=cmap, vminmax=vminmax, enabled=True)
    sm.add_scalar_quantity("u (autoscale)", u, cmap="viridis", enabled=False)
    if "grad_mag" in data.files:
        sm.add_scalar_quantity("|grad u|", data["grad_mag"], cmap="blues", enabled=False)

    # Constraints: point sources
    if len(pts) > 0:
        pc = ps.register_point_cloud("constraint points", pts, radius=0.006)
        pc.add_scalar_quantity("value", pts_val, cmap=cmap, vminmax=vminmax, enabled=True)

    # Constraints: curve sources, stored as segments p1 -> p2
    if len(p1) > 0:
        n = len(p1)
        nodes = np.concatenate([p1, p2], axis=0)
        edges = np.stack([np.arange(n), np.arange(n, 2 * n)], axis=1)
        cn = ps.register_curve_network("constraint curves", nodes, edges, radius=0.003)
        cn.add_scalar_quantity("value", seg_val, defined_on="edges",
                               cmap=cmap, vminmax=vminmax, enabled=True)

    ps.show()


if __name__ == "__main__":
    main()
