#!/usr/bin/env python3
"""Visualize a frame field result npz with polyscope.

Reads the face-center file written by run_train_frame_field.sh
(<mesh name>.npz) and shows the frame field as a cross at every face, together
with the feature curves of the mesh.  Also in the polyscope window (switched
off at the start): the two axes e1 and e2 of the frame, the direction of
minimal curvature t_feat that the field is aligned with, and the per-face
scalars of the file (rotation angle, alignment and smoothness terms).

Example:
  python vis_framefield_npz.py
  python vis_framefield_npz.py ../surface_framefield/results/Bunny/Bunny.npz
  python vis_framefield_npz.py ../surface_framefield/results/Bunny/Bunny.npz --length 0.008
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Default: the result of surface_framefield/run_train_frame_field.sh
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = ROOT / "surface_framefield/results/Bunny/Bunny.npz"

# Per-face scalars of the file
SCALARS = ("theta", "w_curv", "feat_align", "feat_align_weighted",
           "smooth_transport_edge_adj", "smooth_transport_vertex_ring",
           "smooth_knn_edge_adj", "smooth_knn_vertex_ring")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path, nargs="?", default=DEFAULT_NPZ,
                   help="<mesh name>.npz written by run_train_frame_field.sh")
    p.add_argument("--length", type=float, default=0.004,
                   help="length of the vectors, relative to the size of the scene")
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.npz)

    V = data["mesh_vertices"]
    F = data["mesh_faces"]
    e1 = data["e1"]
    e2 = data["e2"]
    radius = args.length / 8

    print(f"{args.npz}")
    print(f"  mesh: {len(V)} vertices, {len(F)} faces")
    print(f"  reference frame: {data['ref_mode']}")

    import polyscope as ps

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)

    # Frame field: (1, 0) in the basis (e1, e2) with n_sym=4 is drawn as the
    # four directions e1, e2, -e1, -e2 of the cross.
    sm.add_tangent_vector_quantity("cross", np.tile([1.0, 0.0], (len(F), 1)), e1, e2, n_sym=4,
                                   defined_on="faces", enabled=True, length=args.length,
                                   radius=radius, color=(0.2, 0.5, 0.9))
    for name, color in (("e1", (0.9, 0.5, 0.1)), ("e2", (0.1, 0.7, 0.3)), ("t_feat", (0.1, 0.1, 0.1))):
        sm.add_vector_quantity(name, data[name], defined_on="faces", enabled=False,
                               length=args.length, radius=radius, color=color)
    for name in SCALARS:
        sm.add_scalar_quantity(name, data[name], defined_on="faces", cmap="viridis", enabled=False)

    # Feature curves (in the file when the training extracted them): the
    # midpoints of the sharp edges and the directions of these edges
    if "feat_curve_pts" in data.files and len(data["feat_curve_pts"]) > 0:
        print(f"  feature curves: {len(data['feat_curve_pts'])} sharp edges")
        curves = ps.register_point_cloud("feature curves", data["feat_curve_pts"],
                                         radius=0.002, color=(1.0, 0.2, 0.2))
        curves.add_vector_quantity("tangent", data["feat_curve_tangent"], enabled=False,
                                   length=args.length, radius=radius, color=(0.9, 0.1, 0.1))

    ps.show()


if __name__ == "__main__":
    main()
