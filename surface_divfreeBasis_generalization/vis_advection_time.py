"""
Visualize saved advection trajectories (NPZ from surface_basis_advection_time.py) with Polyscope.

  python vis_advection_time.py results/hand_2_top60/advection_time.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def _default_frame_indices(n_frames: int, k: int = 10) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    if n_frames <= k:
        return np.arange(n_frames, dtype=int)
    return np.unique(np.round(np.linspace(0, n_frames - 1, k)).astype(int))


def _parse_frames(s: str) -> np.ndarray:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("empty --frames")
    return np.asarray(out, dtype=int)


def main():
    ap = argparse.ArgumentParser(description="Polyscope: rho on mesh from advection NPZ")
    ap.add_argument(
        "npz",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "advection", "bimba_adv.npz"),
        help="Path to .npz (default: advection/bimba_adv.npz next to this script)",
    )
    ap.add_argument(
        "--frames",
        type=_parse_frames,
        default=None,
        help="Comma-separated global frame indices (default: 10 evenly spaced)",
    )
    ap.add_argument("--fps", type=float, default=30.0, help="Playback speed when play is on")
    args = ap.parse_args()

    path = args.npz
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    d = np.load(path, allow_pickle=False)
    V_seq = np.asarray(d["mesh_vertices_seq"], dtype=np.float64)   # (T, Nv, 3)
    F = np.asarray(d["mesh_faces"], dtype=np.int64)
    rho_stack = np.asarray(d["rho"], dtype=np.float64)
    frame_idx = np.asarray(d["frame"], dtype=np.int64)             # snapshot -> mesh frame
    t_arr = frame_idx.astype(np.float64)
    V = V_seq[int(frame_idx[0])]

    if rho_stack.ndim == 1:
        rho_stack = rho_stack[np.newaxis, :]
    n_frames, n_v = rho_stack.shape
    if n_v != len(V):
        print(f"rho shape {rho_stack.shape} incompatible with {len(V)} vertices", file=sys.stderr)
        sys.exit(1)

    if args.frames is not None:
        key_idx = args.frames
    else:
        key_idx = _default_frame_indices(n_frames, 10)

    bad = key_idx[(key_idx < 0) | (key_idx >= n_frames)]
    if len(bad):
        print(f"frame index out of range [0, {n_frames - 1}]: {bad}", file=sys.stderr)
        sys.exit(1)

    rho_keys = rho_stack[key_idx]
    rmax = float(rho_keys.max())
    rmin = float(rho_keys.min())
    if rmax <= rmin:
        rmax = rmin + 1e-12

    import polyscope as ps
    import polyscope.imgui as psim

    K = len(key_idx)
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)
    sm.add_scalar_quantity(
        "density",
        rho_stack[int(key_idx[0])],
        cmap="viridis",
        enabled=True,
        vminmax=(rmin, rmax),
    )

    state = {"k": 0, "play": False, "last": time.time()}

    def cb():
        psim.PushItemWidth(220)
        _, state["k"] = psim.SliderInt("keyframe", state["k"], 0, K - 1)
        psim.PopItemWidth()
        _, state["play"] = psim.Checkbox("play", state["play"])
        if state["play"] and K > 1:
            now = time.time()
            dt = 1.0 / max(args.fps, 1e-6)
            if now - state["last"] >= dt:
                state["k"] = (state["k"] + 1) % K
                state["last"] = now
            ps.request_redraw()

        fi = int(key_idx[state["k"]])
        sm.update_vertex_positions(V_seq[int(frame_idx[fi])])
        sm.add_scalar_quantity(
            "density",
            rho_stack[fi],
            cmap="viridis",
            enabled=True,
            vminmax=(rmin, rmax),
        )
        psim.Text(
            f"snapshot {fi}/{n_frames - 1}   key {state['k']}/{K - 1}   "
            f"mesh frame = {int(frame_idx[fi])}"
        )
        psim.Text(f"rho in [{rmin:.5g}, {rmax:.5g}]  (over selected frames)")

    ps.set_user_callback(cb)
    ps.show()


if __name__ == "__main__":
    main()
