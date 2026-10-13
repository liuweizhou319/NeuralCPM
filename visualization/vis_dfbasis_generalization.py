#!/usr/bin/env python3
"""Visualize a divergence-free basis on a family of shapes with polyscope.

Reads the folder written by run_train_basis_time.sh (frame_0000/, frame_0001/,
...: one folder per mesh of the sequence, each with basis_000.npz,
basis_001.npz, ... and meta.npz).  Two sliders in the polyscope window select
the shape and the basis field, so one basis field can be followed over the
shapes.  Shown on the mesh: the vector field u, its stream function psi, the
vorticity omega and |u|.

Example:
  python vis_dfbasis_generalization.py
  python vis_dfbasis_generalization.py ../surface_divfreeBasis_generalization/results/hand_2_top60
  python vis_dfbasis_generalization.py ../surface_divfreeBasis_generalization/results/hand_2_top60 --length 0.03
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Default: the result of surface_divfreeBasis_generalization/run_train_basis_time.sh
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "surface_divfreeBasis_generalization/results/hand_2_top60"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path, nargs="?", default=DEFAULT_DIR,
                   help="folder with frame_*/basis_*.npz written by run_train_basis_time.sh")
    p.add_argument("--length", type=float, default=0.02,
                   help="length of the longest vector, relative to the size of the scene")
    return p.parse_args()


def main():
    args = parse_args()
    shapes = sorted(d for d in args.folder.glob("frame_*") if d.is_dir())
    if not shapes:
        raise SystemExit(f"no frame_*/ folders in {args.folder}")
    num_basis = len(list(shapes[0].glob("basis_[0-9][0-9][0-9].npz")))
    if num_basis == 0:
        raise SystemExit(f"no basis_*.npz in {shapes[0]}")

    print(f"{args.folder}: {len(shapes)} shapes, {num_basis} basis fields each")

    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    state = {"shape": 0, "basis": 0, "code": 0.0}

    def show(shape, basis, enabled=None):
        """Load one basis field on one shape and replace what is displayed.

        enabled=None keeps what is switched on in the polyscope window.
        """
        data = np.load(shapes[shape] / f"basis_{basis:03d}.npz")
        u = data["u"]
        state["shape"] = shape
        state["basis"] = basis
        state["code"] = float(np.load(shapes[shape] / "meta.npz")["code"])

        # Registering a mesh under the same name replaces the previous shape.
        sm = ps.register_surface_mesh("mesh", data["mesh_vertices"], data["mesh_faces"], smooth_shade=True)
        sm.add_vector_quantity("u", u, defined_on="faces", enabled=enabled,
                               length=args.length, color=(0.2, 0.5, 0.9))
        sm.add_scalar_quantity("psi", data["psi"], defined_on="faces", cmap="coolwarm", enabled=enabled)
        sm.add_scalar_quantity("omega", data["omega"], defined_on="faces", cmap="coolwarm")
        sm.add_scalar_quantity("|u|", np.linalg.norm(u, axis=1), defined_on="faces", cmap="reds")

    def callback():
        changed_shape, shape = psim.SliderInt("shape index", state["shape"], 0, len(shapes) - 1)
        changed_basis, basis = psim.SliderInt("basis index", state["basis"], 0, num_basis - 1)
        psim.TextUnformatted(f"shape code = {state['code']:.4f}")
        psim.TextUnformatted(f"{shapes[state['shape']].name}/basis_{state['basis']:03d}.npz")
        if changed_shape or changed_basis:
            show(shape, basis)

    show(0, 0, enabled=True)
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
