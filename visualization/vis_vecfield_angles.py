#!/usr/bin/env python3
"""Visualize the angle generalization of a vector field with polyscope.

Reads the folder written by run_generalization.sh (vecfield_000.npz,
vecfield_001.npz, ...: one file per rotation angle of the anchors) and shows
one angle at a time.  A slider in the polyscope window selects the angle; the
vector field and the rotated anchor vectors are updated.  The vectors are
drawn with their true length.

Example:
  python vis_vecfield_angles.py
  python vis_vecfield_angles.py ../surfacevector/results/generalization/botijo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Default: the result of surfacevector/run_generalization.sh
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "surfacevector/results/generalization/botijo"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path, nargs="?", default=DEFAULT_DIR,
                   help="folder with vecfield_*.npz written by run_generalization.sh")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(args.folder.glob("vecfield_*.npz"))
    if not files:
        raise SystemExit(f"no vecfield_*.npz in {args.folder}")

    # The mesh and the anchor faces are the same in every file.
    first = np.load(files[0])
    V = first["mesh_vertices"]
    F = first["mesh_faces"]
    fc = first["face_centers"]
    c_ids = first["constraint_face_ids"]

    print(f"{args.folder}: {len(files)} angles")
    print(f"  mesh: {len(V)} vertices, {len(F)} faces")
    print(f"  anchors: {len(c_ids)} faces {c_ids.tolist()}")

    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)
    cloud = None
    if len(c_ids) > 0:
        cloud = ps.register_point_cloud("anchors", fc[c_ids], radius=0.008, color=(1.0, 0.2, 0.2))

    state = {"index": 0, "theta": 0.0}

    def show(index):
        """Load one angle and replace the displayed vectors."""
        data = np.load(files[index])
        u = data["u"]
        state["index"] = index
        state["theta"] = float(data["theta"])

        # Result: the vector field for this angle
        sm.add_vector_quantity("u", u, defined_on="faces", vectortype="ambient",
                               enabled=True, color=(0.2, 0.5, 0.9))
        # Constraints: anchor vectors rotated by this angle, and the network vectors there
        if cloud is not None:
            cloud.add_vector_quantity("target", data["constraint_vectors_rotated"],
                                      vectortype="ambient", color=(0.9, 0.1, 0.1), enabled=True)
            cloud.add_vector_quantity("network", u[c_ids], vectortype="ambient",
                                      color=(0.1, 0.1, 0.9), enabled=True)

    def callback():
        changed, index = psim.SliderInt("angle index", state["index"], 0, len(files) - 1)
        psim.TextUnformatted(f"theta = {state['theta']:.4f} rad ({np.degrees(state['theta']):.1f} deg)")
        psim.TextUnformatted(files[state["index"]].name)
        if changed:
            show(index)

    show(0)
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
