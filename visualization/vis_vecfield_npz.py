#!/usr/bin/env python3
"""Visualize one vector-field result npz with polyscope (single model).

Shows the vector field u at the face centers and the anchors stored in the
same file: at every anchor face the target vector (red) and the vector of the
network (blue).  The vectors are drawn with their true length.

Example:
  python vis_vecfield_npz.py
  python vis_vecfield_npz.py ../surfacevector/results/single/78481root_5w/vecfield_000.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Default: the result of surfacevector/run_single.sh
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = ROOT / "surfacevector/results/single/78481root_5w/vecfield_000.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path, nargs="?", default=DEFAULT_NPZ,
                   help="vecfield_000.npz written by run_single.sh")
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.npz)

    V = data["mesh_vertices"]
    F = data["mesh_faces"]
    fc = data["face_centers"]
    u = data["u"]
    c_ids = data["constraint_face_ids"]
    c_target = data["constraint_vectors_rotated"]
    theta = float(data["theta"])

    print(f"{args.npz}")
    print(f"  mesh: {len(V)} vertices, {len(F)} faces")
    print(f"  theta = {theta:.4f} rad ({np.degrees(theta):.1f} deg)")
    print(f"  anchors: {len(c_ids)} faces {c_ids.tolist()}")
    if len(c_ids) > 0:
        err = np.linalg.norm(u[c_ids] - c_target, axis=1)
        print(f"  |u - target| at the anchors: mean {err.mean():.3e}, max {err.max():.3e}")

    import polyscope as ps

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Result: the vector field and its two parts
    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)
    sm.add_vector_quantity("u", u, defined_on="faces", vectortype="ambient",
                           enabled=True, color=(0.2, 0.5, 0.9))
    sm.add_vector_quantity("u_pot", data["u_pot"], defined_on="faces", vectortype="ambient",
                           enabled=False, color=(0.9, 0.5, 0.1))
    sm.add_vector_quantity("u_stream", data["u_stream"], defined_on="faces", vectortype="ambient",
                           enabled=False, color=(0.1, 0.7, 0.3))
    sm.add_scalar_quantity("|u|", np.linalg.norm(u, axis=1), defined_on="faces",
                           cmap="reds", enabled=False)
    sm.add_scalar_quantity("phi", data["phi"], defined_on="faces", cmap="viridis", enabled=False)
    sm.add_scalar_quantity("psi", data["psi"], defined_on="faces", cmap="coolwarm", enabled=False)

    # Constraints: anchors with their target vector and the network vector
    if len(c_ids) > 0:
        cloud = ps.register_point_cloud("anchors", fc[c_ids], radius=0.008, color=(1.0, 0.2, 0.2))
        cloud.add_vector_quantity("target", c_target, vectortype="ambient",
                                  color=(0.9, 0.1, 0.1), enabled=True)
        cloud.add_vector_quantity("network", u[c_ids], vectortype="ambient",
                                  color=(0.1, 0.1, 0.9), enabled=True)

    ps.show()


if __name__ == "__main__":
    main()
