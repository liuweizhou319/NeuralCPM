#!/usr/bin/env python3
"""Visualize a divergence-free basis with polyscope (single model).

Reads the folder written by run_train_basis.sh (basis_000.npz, basis_001.npz,
...: one file per basis field) and shows one field at a time.  A slider in the
polyscope window selects the field.  Shown on the mesh: the vector field u,
its stream function psi, the vorticity omega and |u|.

Example:
  python vis_dfbasis.py
  python vis_dfbasis.py ../surface_divfreeBasis/results/Bunny
  python vis_dfbasis.py ../surface_divfreeBasis/results/Bunny --length 0.03
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Default: the result of surface_divfreeBasis/run_train_basis.sh
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "surface_divfreeBasis/results/Bunny"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path, nargs="?", default=DEFAULT_DIR,
                   help="folder with basis_*.npz written by run_train_basis.sh")
    p.add_argument("--length", type=float, default=0.02,
                   help="length of the longest vector, relative to the size of the scene")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(args.folder.glob("basis_[0-9][0-9][0-9].npz"))
    if not files:
        raise SystemExit(f"no basis_*.npz in {args.folder}")

    # The mesh is the same in every file.
    first = np.load(files[0])
    V = first["mesh_vertices"]
    F = first["mesh_faces"]

    print(f"{args.folder}: {len(files)} basis fields")
    print(f"  mesh: {len(V)} vertices, {len(F)} faces")

    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)

    state = {"index": 0}

    def show(index, enabled=None):
        """Load one basis field and replace the quantities on the mesh.

        enabled=None keeps what is switched on in the polyscope window.
        """
        data = np.load(files[index])
        u = data["u"]
        state["index"] = index

        sm.add_vector_quantity("u", u, defined_on="faces", enabled=enabled,
                               length=args.length, color=(0.2, 0.5, 0.9))
        sm.add_scalar_quantity("psi", data["psi"], defined_on="faces", cmap="coolwarm", enabled=enabled)
        sm.add_scalar_quantity("omega", data["omega"], defined_on="faces", cmap="coolwarm")
        sm.add_scalar_quantity("|u|", np.linalg.norm(u, axis=1), defined_on="faces", cmap="reds")

    def callback():
        changed, index = psim.SliderInt("basis index", state["index"], 0, len(files) - 1)
        psim.TextUnformatted(files[state["index"]].name)
        if changed:
            show(index)

    show(0, enabled=True)
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
